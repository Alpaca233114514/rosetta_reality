"""Zero-optimizer CUDA horizon and noise-mode closed-loop diagnostic (P0-B).

Replaces the XPU-locked ``smolvla_sim_gate.py execution-diagnostic`` path for
AutoDL CUDA evaluations without altering any historical Gate script or plan.
Sweeps registered ``actions_per_inference`` values (e.g. 1/5/10/25/50) and
inference noise modes (``seeded_standard_normal`` vs ``zeros``) while recording
fine-grained MuJoCo task milestones, safety diagnostics, and bounded-sine
gripper fold counts. Never updates weights and never claims a formal Gate pass.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def _load_generalization_controls():
    try:
        from rosetta_reality.vla import generalization_controls as module

        return module
    except ModuleNotFoundError:
        path = ROOT / "src/rosetta_reality/vla/generalization_controls.py"
        spec = importlib.util.spec_from_file_location("_rosetta_generalization_controls", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["_rosetta_generalization_controls"] = module
        spec.loader.exec_module(module)
        return module


_GC = _load_generalization_controls()
TemporalEnsembler = _GC.TemporalEnsembler
apply_clamped_bounded_sine_override = _GC.apply_clamped_bounded_sine_override
compute_trajectory_dynamics = _GC.compute_trajectory_dynamics

SUPPORTED_NOISE_MODES = ("seeded_standard_normal", "zeros")
HALF_PI = math.pi / 2.0
GRIPPER_INDICES = (6, 13)


def sha(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path | str) -> dict:
    def reject(constant: str):
        raise ValueError("Nonfinite JSON constant: " + constant)

    value = json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object: " + Path(path).name)
    return value


def write_json_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def euclidean_distance(a, b) -> float:
    pa = a.detach().cpu().tolist() if hasattr(a, "detach") else list(a)
    pb = b.detach().cpu().tolist() if hasattr(b, "detach") else list(b)
    if len(pa) != 3 or len(pb) != 3:
        raise ValueError("Expected 3D coordinates")
    sq = sum((float(x) - float(y)) ** 2 for x, y in zip(pa, pb, strict=True))
    dist = math.sqrt(sq)
    if not math.isfinite(dist):
        raise ValueError("Nonfinite Euclidean distance")
    return dist


def count_bounded_sine_folds(internal_chunk, executed_slots: int) -> dict[str, int]:
    """Count pre-sin gripper predictions exceeding ±pi/2 within executed chunk slots.
    统计实际执行槽位内，bounded-sine 解码前超过 ±pi/2 的夹爪回折次数。
    """
    rows = internal_chunk.detach().cpu().tolist() if hasattr(internal_chunk, "detach") else internal_chunk
    if isinstance(rows, list) and rows and isinstance(rows[0], list) and rows[0] and isinstance(rows[0][0], list):
        if len(rows) != 1:
            raise ValueError("Expected batch-1 action chunk")
        rows = rows[0]
    slots = rows[:executed_slots]
    left_folds = 0
    right_folds = 0
    for slot in slots:
        lv, rv = float(slot[GRIPPER_INDICES[0]]), float(slot[GRIPPER_INDICES[1]])
        if not (math.isfinite(lv) and math.isfinite(rv)):
            raise ValueError("Nonfinite internal gripper action")
        if abs(lv) > HALF_PI:
            left_folds += 1
        if abs(rv) > HALF_PI:
            right_folds += 1
    return {
        "left_gripper_folds": left_folds,
        "right_gripper_folds": right_folds,
        "total_folds": left_folds + right_folds,
    }


class MilestoneTracker:
    """Accumulate fine-grained insertion milestones across one rollout episode.
    按物理步累积插入任务四阶段里程碑（接近、接触、连续双抓、对准与插入），避免仅靠 0/5 二值成功率丢失诊断信号。
    """

    def __init__(self) -> None:
        self.steps = 0
        self.min_left_eef_to_socket_m = math.inf
        self.min_right_eef_to_peg_m = math.inf
        self.min_peg_to_socket_m = math.inf
        self.socket_grasp_steps = 0
        self.peg_grasp_steps = 0
        self.dual_grasp_steps = 0
        self.current_dual_grasp_streak = 0
        self.max_dual_grasp_streak = 0
        self.peg_socket_contact_steps = 0
        self.pin_contact_steps = 0
        self.max_reward = 0.0

    def observe(self, geometry, reward: float) -> None:
        if not math.isfinite(reward):
            raise ValueError("Nonfinite reward")
        self.steps += 1
        self.max_reward = max(self.max_reward, float(reward))
        self.min_left_eef_to_socket_m = min(
            self.min_left_eef_to_socket_m,
            euclidean_distance(geometry.left_eef.position, geometry.socket.position),
        )
        self.min_right_eef_to_peg_m = min(
            self.min_right_eef_to_peg_m,
            euclidean_distance(geometry.right_eef.position, geometry.peg.position),
        )
        self.min_peg_to_socket_m = min(
            self.min_peg_to_socket_m,
            euclidean_distance(geometry.peg.position, geometry.socket.position),
        )
        left_ok = bool(geometry.socket_grasp_contact)
        right_ok = bool(geometry.peg_grasp_contact)
        self.socket_grasp_steps += int(left_ok)
        self.peg_grasp_steps += int(right_ok)
        if left_ok and right_ok:
            self.dual_grasp_steps += 1
            self.current_dual_grasp_streak += 1
            self.max_dual_grasp_streak = max(
                self.max_dual_grasp_streak, self.current_dual_grasp_streak
            )
        else:
            self.current_dual_grasp_streak = 0
        self.peg_socket_contact_steps += int(bool(geometry.peg_socket_contact))
        self.pin_contact_steps += int(bool(geometry.pin_contact))

    def summary(self) -> dict:
        return {
            "observed_steps": self.steps,
            "maximum_reward": self.max_reward,
            "min_left_eef_to_socket_m": (
                None if not math.isfinite(self.min_left_eef_to_socket_m) else self.min_left_eef_to_socket_m
            ),
            "min_right_eef_to_peg_m": (
                None if not math.isfinite(self.min_right_eef_to_peg_m) else self.min_right_eef_to_peg_m
            ),
            "min_peg_to_socket_m": (
                None if not math.isfinite(self.min_peg_to_socket_m) else self.min_peg_to_socket_m
            ),
            "socket_grasp_steps": self.socket_grasp_steps,
            "peg_grasp_steps": self.peg_grasp_steps,
            "dual_grasp_steps": self.dual_grasp_steps,
            "max_dual_grasp_streak": self.max_dual_grasp_streak,
            "peg_socket_contact_steps": self.peg_socket_contact_steps,
            "pin_contact_steps": self.pin_contact_steps,
        }


def validate_sweep_config(
    *,
    actions_per_inference_list: list[int],
    noise_modes: list[str],
    seeds: list[int],
    maximum_steps: int,
    chunk_length: int = 50,
    stop_on_safety: bool = False,
    temporal_ensemble_k: float | None = None,
    clamp_bounded_sine: bool = False,
) -> None:
    if not actions_per_inference_list or any(
        type(k) is not int or not 1 <= k <= chunk_length for k in actions_per_inference_list
    ):
        raise ValueError(f"actions_per_inference must be integers in [1, {chunk_length}]")
    if len(set(actions_per_inference_list)) != len(actions_per_inference_list):
        raise ValueError("actions_per_inference contains duplicate entries")
    if not noise_modes or any(m not in SUPPORTED_NOISE_MODES for m in noise_modes):
        raise ValueError(f"noise_modes must be subset of {SUPPORTED_NOISE_MODES}")
    if len(set(noise_modes)) != len(noise_modes):
        raise ValueError("noise_modes contains duplicate entries")
    if not seeds or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("seeds must be non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds contains duplicate entries")
    if type(maximum_steps) is not int or not 1 <= maximum_steps <= 500:
        raise ValueError("maximum_steps must be an integer in [1, 500]")
    if type(stop_on_safety) is not bool:
        raise ValueError("stop_on_safety must be a boolean")
    if temporal_ensemble_k is not None:
        if not isinstance(temporal_ensemble_k, (int, float)) or isinstance(temporal_ensemble_k, bool):
            raise ValueError("temporal_ensemble_k must be a non-negative finite number")
        if not math.isfinite(float(temporal_ensemble_k)) or float(temporal_ensemble_k) < 0.0:
            raise ValueError("temporal_ensemble_k must be a non-negative finite number")
    if type(clamp_bounded_sine) is not bool:
        raise ValueError("clamp_bounded_sine must be a boolean")


def _extract_internal_chunk(policy):
    post = getattr(policy, "postprocessor", None)
    if post is None or not hasattr(post, "steps"):
        return None
    decoders = [
        step
        for step in post.steps
        if getattr(step.__class__, "_registry_name", None) == "rosetta_pi_aloha_postprocessor"
    ]
    if len(decoders) != 1:
        return None
    internal = getattr(decoders[0], "last_model_action", None)
    if internal is None:
        return None
    rows = internal.detach().cpu().tolist() if hasattr(internal, "detach") else internal
    if isinstance(rows, list) and rows and isinstance(rows[0], list) and rows[0] and isinstance(rows[0][0], list):
        if len(rows) != 1:
            raise ValueError("Expected batch-1 internal action chunk")
        rows = rows[0]
    return rows


def run_episode(
    policy,
    contract,
    instruction: str,
    *,
    seed: int,
    maximum_steps: int,
    actions_per_inference: int,
    noise_mode: str,
    stop_on_safety: bool = False,
    temporal_ensemble_k: float | None = None,
    clamp_bounded_sine: bool = False,
    env_factory=None,
    geometry_reader=None,
) -> dict:
    """Execute one rollout for a given (actions_per_inference, noise_mode, seed).

    When ``temporal_ensemble_k`` is set, predicts every step and fuses overlapping
    chunks via exponential decay ``exp(-k * age)`` (ACT/SmolVLA temporal ensemble)
    instead of open-loop chunk execution. Injecting ``env_factory`` and
    ``geometry_reader`` allows pure unit testing without loading MuJoCo or weights.
    """
    if temporal_ensemble_k is not None and actions_per_inference != 1:
        raise ValueError("temporal_ensemble_k requires actions_per_inference == 1")
    policy_noise_seed = None if noise_mode == "zeros" else seed
    policy.configure_noise(noise_mode, policy_noise_seed)
    if env_factory is None:
        from rosetta_reality.sim import GymAlohaEnvironment

        env = GymAlohaEnvironment(contract, maximum_episode_steps=maximum_steps)
    else:
        env = env_factory(contract, maximum_steps)
    if geometry_reader is None:
        from rosetta_reality.sim.scripted_teacher_observer import build_observation

        geometry_reader = build_observation

    milestones = MilestoneTracker()
    executed_actions = []
    observed_states = []
    ensembler = None
    inference_calls = 0
    inference_seconds = 0.0
    simulation_seconds = 0.0
    raw_clip_count = 0
    joint_limit_violations = 0
    violated_joints: Counter[str] = Counter()
    unexpected_collisions = 0
    unexpected_pairs: Counter[str] = Counter()
    left_folds = 0
    right_folds = 0
    clamped_override_steps = 0
    success = False
    outcome = "task_failed"

    try:
        observation = env.reset(seed=seed)
        while len(executed_actions) < maximum_steps:
            t0 = time.perf_counter()
            raw_chunk, projected_chunk = policy.predict(observation, instruction)
            inference_seconds += time.perf_counter() - t0
            inference_calls += 1

            internal_rows = _extract_internal_chunk(policy)
            remaining = min(actions_per_inference, maximum_steps - len(executed_actions))
            if internal_rows is not None:
                folds = count_bounded_sine_folds(internal_rows, remaining)
                left_folds += folds["left_gripper_folds"]
                right_folds += folds["right_gripper_folds"]

            if temporal_ensemble_k is not None:
                chunk_rows = (
                    raw_chunk.detach().cpu().tolist()
                    if hasattr(raw_chunk, "detach")
                    else [list(r) for r in raw_chunk]
                )
                if clamp_bounded_sine and internal_rows is not None:
                    chunk_rows = [
                        apply_clamped_bounded_sine_override(r_slot, i_slot)
                        for r_slot, i_slot in zip(chunk_rows, internal_rows, strict=True)
                    ]
                    clamped_override_steps += 1
                if ensembler is None:
                    ensembler = TemporalEnsembler(
                        chunk_size=len(chunk_rows),
                        decay_k=float(temporal_ensemble_k),
                        action_dim=len(chunk_rows[0]),
                    )
                slots_to_execute = [ensembler.update(chunk_rows)]
            else:
                steps_to_run = remaining
                slots_to_execute = []
                for slot_index in range(steps_to_run):
                    raw_slot = raw_chunk[slot_index]
                    if clamp_bounded_sine and internal_rows is not None:
                        raw_slot = apply_clamped_bounded_sine_override(
                            raw_slot, internal_rows[slot_index]
                        )
                        clamped_override_steps += 1
                    slots_to_execute.append(raw_slot)

            done = False
            safety_stopped = False
            for raw_slot in slots_to_execute:
                command, clip_mask = contract.clip(raw_slot)
                raw_clip_count += int(clip_mask.sum().item() if hasattr(clip_mask, "sum") else sum(clip_mask))
                ts = time.perf_counter()
                observation, reward, done, info = env.step(command)
                simulation_seconds += time.perf_counter() - ts
                executed_actions.append(command)
                if isinstance(observation, dict) and "robot_state" in observation:
                    observed_states.append(observation["robot_state"])
                milestones.observe(geometry_reader(env, contract), float(reward))

                diag = env.diagnostic_snapshot()
                step_jlv = len(diag.get("joint_limit_violations", []))
                joint_limit_violations += step_jlv
                for row in diag.get("joint_limit_violations", []):
                    violated_joints[str(row.get("name", "unknown"))] += 1
                step_collisions = 0
                for first, second in env.contact_pairs():
                    if env.is_unexpected_collision_pair(first, second):
                        step_collisions += 1
                        unexpected_collisions += 1
                        unexpected_pairs[" <-> ".join(sorted((first, second)))] += 1
                success = success or bool(info.get("is_success", False))
                if stop_on_safety and (step_jlv or step_collisions):
                    outcome = "physical_safety_stop"
                    safety_stopped = True
                    break
                if done:
                    break
            if safety_stopped:
                break
            if done:
                outcome = "task_success" if success else "task_failed"
                break
        else:
            outcome = "task_success" if success else "task_failed"
    finally:
        env.close()

    smoothness = 0.0
    if len(executed_actions) > 1:
        diffs = []
        for prev, curr in zip(executed_actions[:-1], executed_actions[1:], strict=True):
            p_list = prev.detach().cpu().tolist() if hasattr(prev, "detach") else list(prev)
            c_list = curr.detach().cpu().tolist() if hasattr(curr, "detach") else list(curr)
            diffs.append(math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(c_list, p_list, strict=True))))
        smoothness = sum(diffs) / len(diffs)

    dynamics = compute_trajectory_dynamics(
        executed_actions,
        observed_states if len(observed_states) == len(executed_actions) else None,
    )

    return {
        "seed": seed,
        "noise_mode": noise_mode,
        "policy_noise_seed": policy_noise_seed,
        "actions_per_inference": actions_per_inference,
        "temporal_ensemble_k": temporal_ensemble_k,
        "clamp_bounded_sine": clamp_bounded_sine,
        "clamped_override_steps": clamped_override_steps,
        "rollout_length": len(executed_actions),
        "policy_inference_calls": inference_calls,
        "success": success,
        "outcome": outcome,
        "milestones": milestones.summary(),
        "dynamics": dynamics,
        "joint_limit_violations": joint_limit_violations,
        "violated_joints": dict(sorted(violated_joints.items())),
        "unexpected_collisions": unexpected_collisions,
        "unexpected_collision_pairs": dict(sorted(unexpected_pairs.items())),
        "raw_clipped_fields": raw_clip_count,
        "bounded_sine_folds": {
            "left_gripper": left_folds,
            "right_gripper": right_folds,
            "total": left_folds + right_folds,
        },
        "action_smoothness_l2": smoothness,
        "mean_policy_inference_seconds": inference_seconds / max(1, inference_calls),
        "mean_simulation_step_seconds": simulation_seconds / max(1, len(executed_actions)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--sim-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--actions-per-inference", type=int, nargs="+", default=[1, 5, 10, 25, 50])
    parser.add_argument("--noise-modes", nargs="+", default=["seeded_standard_normal", "zeros"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002, 1003, 1004])
    parser.add_argument("--maximum-steps", type=int, default=500)
    parser.add_argument("--instruction", default="Insert the peg into the socket.")
    parser.add_argument("--stop-on-safety", action="store_true")
    parser.add_argument(
        "--temporal-ensemble-k",
        type=float,
        default=None,
        help="Optional exponential decay k (e.g. 0.01) to add a temporal-ensemble sweep arm per noise mode.",
    )
    parser.add_argument(
        "--clamp-bounded-sine",
        action="store_true",
        help="Clamp pre-sin gripper latents to [-pi/2, +pi/2] before decoding (zero-training A/B test).",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)

    validate_sweep_config(
        actions_per_inference_list=args.actions_per_inference,
        noise_modes=args.noise_modes,
        seeds=args.seeds,
        maximum_steps=args.maximum_steps,
        stop_on_safety=args.stop_on_safety,
        temporal_ensemble_k=args.temporal_ensemble_k,
        clamp_bounded_sine=args.clamp_bounded_sine,
    )
    extra_te = 1 if args.temporal_ensemble_k is not None else 0
    total_conditions = (len(args.actions_per_inference) + extra_te) * len(args.noise_modes)
    if args.check_only:
        print(
            json.dumps(
                {
                    "schema": "rosetta.horizon_sweep_check.v1",
                    "valid": True,
                    "conditions": total_conditions,
                    "episodes_total": total_conditions * len(args.seeds),
                    "temporal_ensemble_k": args.temporal_ensemble_k,
                    "clamp_bounded_sine": args.clamp_bounded_sine,
                    "model_or_physics_executed": False,
                },
                sort_keys=True,
            )
        )
        return 0

    if os.environ.get("ROSETTA_TORCH_DEVICE") != "cuda" or any(
        os.environ.get(k) != "1" for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")
    ):
        raise RuntimeError("Offline registered AutoDL CUDA runtime required")
    if args.output.exists():
        raise FileExistsError("Horizon sweep output directory must be create-only")

    import torch
    from diagnose_smolvla_base_five_scene import parameter_digest
    from rosetta_reality.sim import load_action_contract
    import smolvla_autodl_vfunfreeze_sim_gate as reference

    artifact = args.artifact.resolve(strict=True)
    sim_plan = args.sim_plan.resolve(strict=True)
    config = read_json(artifact / "config.json")
    normalization = read_json(artifact / "normalization.json")
    contract_path = ROOT / "configs/sim/aloha_insertion_smolvla.yaml"
    contract = load_action_contract(contract_path)

    args.output.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_per_process_memory_fraction(0.4)
    reference._ACTIVE_SIM_PLAN = sim_plan
    policy = reference._build_online_cuda_class()(artifact, config, normalization, contract)
    before_digest = parameter_digest(policy.policy)

    started = time.monotonic()
    conditions = []
    sweep_specs: list[tuple[int, float | None, str]] = [
        (k_steps, None, f"k{k_steps:02d}") for k_steps in args.actions_per_inference
    ]
    if args.temporal_ensemble_k is not None:
        sweep_specs.append((1, float(args.temporal_ensemble_k), f"te{args.temporal_ensemble_k:g}"))

    for noise_mode in args.noise_modes:
        for k_steps, te_k, label in sweep_specs:
            episodes = []
            for seed in args.seeds:
                ep = run_episode(
                    policy,
                    contract,
                    args.instruction,
                    seed=seed,
                    maximum_steps=args.maximum_steps,
                    actions_per_inference=k_steps,
                    noise_mode=noise_mode,
                    stop_on_safety=args.stop_on_safety,
                    temporal_ensemble_k=te_k,
                    clamp_bounded_sine=args.clamp_bounded_sine,
                )
                episodes.append(ep)
            condition_summary = {
                "noise_mode": noise_mode,
                "actions_per_inference": k_steps,
                "temporal_ensemble_k": te_k,
                "clamp_bounded_sine": args.clamp_bounded_sine,
                "success_count": sum(1 for e in episodes if e["success"]),
                "episode_count": len(episodes),
                "max_reward": max(e["milestones"]["maximum_reward"] for e in episodes),
                "any_socket_grasp": any(e["milestones"]["socket_grasp_steps"] > 0 for e in episodes),
                "any_peg_grasp": any(e["milestones"]["peg_grasp_steps"] > 0 for e in episodes),
                "max_dual_grasp_streak": max(e["milestones"]["max_dual_grasp_streak"] for e in episodes),
                "mean_arm_signflip_rate": sum(e["dynamics"]["arm_signflip_rate"] for e in episodes)
                / max(1, len(episodes)),
                "mean_grip_signflip_rate": sum(e["dynamics"]["grip_signflip_rate"] for e in episodes)
                / max(1, len(episodes)),
                "total_joint_limit_violations": sum(e["joint_limit_violations"] for e in episodes),
                "total_unexpected_collisions": sum(e["unexpected_collisions"] for e in episodes),
                "total_bounded_sine_folds": sum(e["bounded_sine_folds"]["total"] for e in episodes),
                "episodes": episodes,
            }
            conditions.append(condition_summary)
            write_json_new(
                args.output / f"condition-{noise_mode}-{label}.json",
                condition_summary,
            )

    after_digest = parameter_digest(policy.policy)
    if before_digest != after_digest:
        raise RuntimeError("Policy parameters/buffers changed during zero-optimizer horizon sweep")

    report = {
        "schema": "rosetta.smolvla_horizon_sweep.v1",
        "status": "completed",
        "diagnostic_only": True,
        "formal_gate_result": None,
        "m2_complete": False,
        "optimizer_updates": 0,
        "hidden_test_loaded": False,
        "shutdown_requested": False,
        "clamp_bounded_sine": args.clamp_bounded_sine,
        "temporal_ensemble_k": args.temporal_ensemble_k,
        "artifact_manifest_sha256": sha(artifact / "manifest.json"),
        "sim_plan_sha256": sha(sim_plan),
        "action_contract_sha256": sha(contract_path),
        "parameters_before_sha256": before_digest,
        "parameters_after_sha256": after_digest,
        "parameters_unchanged": before_digest == after_digest,
        "elapsed_seconds": time.monotonic() - started,
        "conditions": conditions,
    }
    write_json_new(args.output / "summary.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
