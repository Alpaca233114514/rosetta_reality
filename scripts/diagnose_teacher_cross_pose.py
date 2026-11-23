"""Independent bounded cross-pose diagnostic; never a formal teacher Gate result.

Runtime dependencies load only inside main. The injected runner supports fake
environments without importing Torch, MuJoCo, models, or datasets.
"""
from __future__ import annotations

import argparse
import dataclasses
import enum
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (1900, 1901, 1902, 1903, 1904)
CANDIDATE_MODULE = "rosetta_reality.sim.coupled_translation_teacher"
OBSERVER_MODULE = "rosetta_reality.sim.scripted_teacher_observer"
CONTRACT_PATH = "configs/sim/aloha_insertion_smolvla.yaml"


def finite_json(value: Any) -> Any:
    """Copy numeric evidence to JSON, rejecting every nonfinite value."""
    if isinstance(value, enum.Enum):
        return finite_json(value.value)
    if dataclasses.is_dataclass(value):
        return {field.name: finite_json(getattr(value, field.name))
                for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(key): finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if hasattr(value, "detach"):
        return finite_json(value.detach().cpu().tolist())
    if hasattr(value, "tolist"):
        return finite_json(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite diagnostic value")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported diagnostic value: {type(value).__name__}")


def write_json(path: Path, payload: dict) -> None:
    encoded = json.dumps(finite_json(payload), indent=2, sort_keys=True,
                         allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)


def validate_state_finite(value: Any) -> None:
    """Check camera tensors in place without materializing pixels as JSON."""
    if isinstance(value, dict):
        for item in value.values():
            validate_state_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            validate_state_finite(item)
    elif hasattr(value, "isfinite"):
        if not bool(value.isfinite().all()):
            raise ValueError("nonfinite environment tensor")
    elif hasattr(value, "dtype") and hasattr(value, "shape"):
        # NumPy integer/bool image arrays cannot contain NaN/Inf.
        if value.dtype.kind in "fc":
            import numpy as np
            if not bool(np.isfinite(value).all()):
                raise ValueError("nonfinite environment array")
        elif value.dtype.kind not in "biu":
            raise TypeError("unsupported environment array dtype")
    else:
        finite_json(value)


def run_diagnostic(output: Path, *, contract: Any, environment_factory: Any,
                   teacher_factory: Any, observer: Any, max_steps: int = 500,
                   max_seconds: float = 600, clock: Any = time.monotonic,
                   provenance: dict | None = None) -> dict:
    """Run all registered seeds despite task failure/refusal/safety stops.

    Internal exceptions and nonfinite values stop the whole diagnostic. A
    cooperative deadline complements the caller's external hard supervisor.
    Existing output roots are rejected before constructing any environment.
    """
    if not 1 <= max_steps <= 500 or not 0 < max_seconds <= 600:
        raise ValueError("diagnostic bounds: steps 1..500, seconds (0,600]")
    output.mkdir(parents=True, exist_ok=False)
    started = clock()
    episodes = []
    stop_all = None
    for seed in SEEDS:
        result = {"seed": seed, "status": "not_started", "success": None,
                  "attempted": False, "reset_completed": False, "completed": False,
                  "rollout_length": 0, "maximum_reward": None,
                  "joint_limit_violations": 0, "unexpected_collisions": 0,
                  "action_clip_entry_count": 0, "action_clipped_fields": [],
                  "refusal": None, "trace": f"seed-{seed}.jsonl"}
        environment = None
        fields = set()
        if stop_all is None and clock() - started >= max_seconds:
            stop_all = "time_budget_exhausted"
        if stop_all:
            result["status"] = f"not_started_{stop_all}"
        else:
            try:
                result["attempted"] = True
                environment = environment_factory(contract, maximum_episode_steps=max_steps)
                environment.reset(seed=seed)
                result["reset_completed"] = True
                result["success"] = False
                teacher = teacher_factory()
                teacher.reset()
                result["status"] = "budget_exhausted"
                with (output / result["trace"]).open("x", encoding="utf-8") as trace:
                    def emit(payload: dict) -> None:
                        trace.write(json.dumps(finite_json(payload), sort_keys=True,
                                               allow_nan=False) + "\n")
                        trace.flush()

                    for step in range(max_steps):
                        if clock() - started >= max_seconds:
                            result["status"] = stop_all = "time_budget_exhausted"
                            break
                        geometry = observer(environment, contract)
                        snapshot = finite_json(geometry)
                        limits = int(environment.state_limit_violation_count())
                        collisions = int(environment.unexpected_collision_count())
                        result["joint_limit_violations"] += limits
                        result["unexpected_collisions"] += collisions
                        if limits or collisions:
                            result["status"] = "safety_stop"
                            emit({"step": step, "event": "pre_action_safety_stop",
                                  "observation": snapshot, "joint_limits": limits,
                                  "unexpected_collisions": collisions})
                            break
                        decision = teacher.decide(geometry)
                        phase = finite_json(getattr(teacher, "_phase", None))
                        if decision.refusal is not None:
                            result["status"] = "refused"
                            result["refusal"] = finite_json(decision.refusal)
                            emit({"step": step, "event": "refusal", "phase": phase,
                                  "observation": snapshot, "refusal": result["refusal"]})
                            break
                        # Finiteness is checked before clipping and before step.
                        requested = finite_json(decision.action)
                        contract.validate_tensor(decision.action, allow_chunk=False)
                        action, mask = contract.clip(decision.action)
                        applied = finite_json(action)
                        clipped = finite_json(mask)
                        result["action_clip_entry_count"] += sum(bool(item) for item in clipped)
                        fields.update(name for name, item in zip(contract.dimension_names, clipped) if item)
                        if clock() - started >= max_seconds:
                            result["status"] = stop_all = "time_budget_exhausted"
                            break
                        # Flush the attempted command first, preserving failure evidence.
                        emit({"step": step, "event": "action", "phase": phase,
                              "observation": snapshot, "requested_action": requested,
                              "applied_action": applied, "clip_mask": clipped})
                        state, reward, done, info = environment.step(action)
                        result["rollout_length"] += 1
                        reward = finite_json(float(reward))
                        after = finite_json(observer(environment, contract))
                        validate_state_finite(state)
                        finite_json(info)
                        limits = int(environment.state_limit_violation_count())
                        collisions = int(environment.unexpected_collision_count())
                        result["joint_limit_violations"] += limits
                        result["unexpected_collisions"] += collisions
                        result["maximum_reward"] = max(reward, result["maximum_reward"]
                                                        if result["maximum_reward"] is not None else reward)
                        result["success"] |= bool(info.get("is_success", False))
                        emit({"step": step, "event": "step_result", "reward": reward,
                              "done": bool(done), "is_success": bool(info.get("is_success", False)),
                              "terminated": bool(info.get("terminated", False)),
                              "observation": after, "contact_pairs": environment.contact_pairs(),
                              "joint_limits": limits, "unexpected_collisions": collisions})
                        if limits or collisions:
                            result["status"] = "safety_stop"
                            break
                        if clock() - started >= max_seconds:
                            result["status"] = stop_all = "time_budget_exhausted"
                            break
                        if done:
                            result["status"] = "success" if result["success"] else "task_failed"
                            break
            except Exception as error:
                result.update(status="infra_failed", error_type=type(error).__name__)
                stop_all = "infra_failed"
            finally:
                if environment is not None:
                    try:
                        environment.close()
                    except Exception as error:
                        result.update(status="infra_failed", close_error_type=type(error).__name__)
                        stop_all = "infra_failed"
        result["action_clipped_fields"] = sorted(fields)
        result["completed"] = result["status"] in {
            "success", "task_failed", "budget_exhausted", "refused", "safety_stop"}
        write_json(output / f"seed-{seed}.json", result)
        episodes.append(result)
    summary = {"schema": "rosetta.teacher_cross_pose_diagnostic.v1",
               "candidate_module": CANDIDATE_MODULE, "observer_module": OBSERVER_MODULE,
               "action_contract": CONTRACT_PATH, "registered_seeds": list(SEEDS),
               "seed_denominator": len(SEEDS), "max_steps_per_seed": max_steps,
               "max_seconds": max_seconds, "wall_seconds": clock() - started,
               "status": stop_all or "completed", "episodes": episodes,
               "success_count": sum(item["success"] is True for item in episodes),
               "successes": sum(item["success"] is True for item in episodes),
               "measured_attempts": sum(item["reset_completed"] for item in episodes),
               "registered_attempts": len(SEEDS),
               "provider_qualified": False, "formal_gate_result": None,
               "g1_required": False, "original_g2_protocol_equivalent": False,
               "model_forward_calls": 0, "optimizer_steps": 0, "dataset_loads": 0,
               "provenance": provenance or {},
               "limitations": ["episode stops immediately on physical limit or unexpected collision",
                               "task failure does not block later independent seeds",
                               "external supervisor must enforce hard wall timeout"]}
    write_json(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("diagnostic output root must be create-only")
    if not 0 < args.max_seconds <= 600 or not 1 <= args.max_steps <= 500:
        parser.error("max-seconds must be (0,600]; max-steps must be 1..500")
    sys.path.insert(0, str(ROOT / "src"))
    paths = [CONTRACT_PATH, "scripts/diagnose_teacher_cross_pose.py",
             "src/rosetta_reality/sim/coupled_translation_teacher.py",
             "src/rosetta_reality/sim/geometric_teacher.py",
             "src/rosetta_reality/sim/scripted_teacher.py",
             "src/rosetta_reality/sim/scripted_teacher_observer.py",
             "src/rosetta_reality/sim/gym_aloha.py"]
    try:
        from rosetta_reality.sim import load_action_contract
        from rosetta_reality.sim.coupled_translation_teacher import build_teacher
        from rosetta_reality.sim.gym_aloha import GymAlohaEnvironment
        from rosetta_reality.sim.scripted_teacher_observer import build_observation
        identity = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths}
        contract = load_action_contract(ROOT / CONTRACT_PATH)
    except Exception as bootstrap_error:
        # Preserve complete denominator even if backend imports/config fail.
        error_type = type(bootstrap_error).__name__

        def failed_environment(*_args, **_kwargs):
            raise RuntimeError("diagnostic bootstrap failed")

        summary = run_diagnostic(args.output, contract=None,
                                 environment_factory=failed_environment,
                                 teacher_factory=None, observer=None,
                                 max_steps=args.max_steps, max_seconds=args.max_seconds,
                                 provenance={"bootstrap_error_type": error_type})
    else:
        summary = run_diagnostic(args.output, contract=contract,
                                 environment_factory=GymAlohaEnvironment,
                                 teacher_factory=build_teacher, observer=build_observation,
                                 max_steps=args.max_steps, max_seconds=args.max_seconds,
                                 provenance={"source_sha256": identity})
    print(json.dumps({"status": summary["status"], "success_count": summary["success_count"],
                      "seed_denominator": len(SEEDS), "provider_qualified": False}))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
