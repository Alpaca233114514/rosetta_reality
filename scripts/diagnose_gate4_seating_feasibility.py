"""Non-gating, create-only Gate 4 seating diagnostic; no models or datasets."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from diagnose_g2_seating_feasibility import _privileged_seater_step  # noqa: E402
from rosetta_reality.sim import load_action_contract  # noqa: E402
from rosetta_reality.sim.geometric_teacher import (  # noqa: E402
    GeometricCommandSettings,
    GeometricEscapeSettings,
    build_teacher,
)
from rosetta_reality.sim.geometry_teacher import compose_pose, relative_pose  # noqa: E402
from rosetta_reality.sim.gym_aloha import GymAlohaEnvironment  # noqa: E402
from rosetta_reality.sim.mink_aloha_ik import MinkAlohaIkSolver  # noqa: E402
from rosetta_reality.sim.scripted_teacher import (  # noqa: E402
    ScriptedInsertionStage,
    default_ik_settings,
)
from rosetta_reality.sim.scripted_teacher_observer import build_observation  # noqa: E402

SEEDS = (1000, 1001, 1002, 1003, 1004)
BUDGET = 500
REGISTRATION = ROOT / (
    "reports/training/m2-smolvla-t2-seating-feasibility-gate4-"
    "preregistration-2026-09-06.json"
)
OUTPUT = ROOT / "runs/m2-t2-seating-feasibility-gate4-2026-09-06"


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def task_success(reward, done, info):
    return reward >= 4.0 and done and bool(info.get("is_success", False))


def classify(rows):
    if tuple(row["seed"] for row in rows) != SEEDS:
        raise ValueError("The complete ordered five-seed inventory is required")
    if any(row["error"] for row in rows):
        return "inconclusive_instrument_failure"
    count = sum(row["seatable"] for row in rows)
    if count == 5:
        return "all_seatable_learning_interaction_gap"
    if count:
        return "mixed_seed_specific_learning_interaction_gap"
    return "no_solution_found_protocol_envelope_candidate_not_impossibility_proof"


class AuditedEnvironment(GymAlohaEnvironment):
    """Same reset/step implementation as Gate 4, with observation-only audit."""

    def __init__(self, contract):
        super().__init__(contract, maximum_episode_steps=BUDGET)
        self.rows = []
        self.stage = "teacher"
        self.last_info = {}

    def step(self, action):
        import torch

        if len(self.rows) >= BUDGET:
            raise RuntimeError("500-step budget exhausted")
        if self.last_info.get("terminated") or self.last_info.get("truncated"):
            raise RuntimeError("Cannot step after termination/truncation")
        before = build_observation(self, self.contract).robot_state
        self.contract.validate_tensor(action, allow_chunk=False)
        _, mask = self.contract.clip(action)
        if not bool(torch.isfinite(action).all()) or bool(mask.any()):
            raise RuntimeError("Nonfinite or out-of-contract executed action")
        arm_indices = [*range(6), *range(7, 13)]
        delta = float((action - before)[arm_indices].abs().max())
        if delta > 0.060002:
            raise RuntimeError("Executed arm delta exceeds registered 0.06 rad bound")
        observation, reward, done, info = super().step(action)
        self.last_info = info
        if not math.isfinite(reward) or not bool(
            torch.isfinite(observation["robot_state"]).all()
        ):
            raise RuntimeError("Nonfinite simulator output")
        self.rows.append({
            "step": len(self.rows) + 1, "stage": self.stage,
            "reward": reward, "done": done, "success": task_success(reward, done, info),
            "terminated": info["terminated"], "truncated": info["truncated"],
            "arm_delta_rad": delta, "action": action.tolist(),
            "joint_limit_violations": self.state_limit_violation_count(),
            "unexpected_collisions": self.unexpected_collision_count(),
        })
        return observation, reward, done, info


def run_seed(seed, contract, solver, seated_rel=None, *, calibration=False, control=False):
    import torch

    teacher = build_teacher(escape_settings=GeometricEscapeSettings(hold_enabled=False))
    environment = AuditedEnvironment(contract)
    result = {"seed": seed, "seatable": False, "error": None, "handoff_step": None}
    captured = None
    try:
        environment.reset(seed=seed)
        result["reset_snapshot"] = environment.diagnostic_snapshot()
        teacher.reset()
        privileged = False
        for _ in range(BUDGET):
            geometry = build_observation(environment, contract)
            gap = None
            if seated_rel is not None:
                ideal = compose_pose(geometry.peg, seated_rel)
                gap = 1000 * float(torch.linalg.vector_norm(
                    geometry.socket.position - ideal.position
                ))
            if privileged:
                environment.stage = "privileged_insert"
                _privileged_seater_step(
                    environment, contract, teacher, solver, seated_rel,
                    GeometricCommandSettings().ik_seed_shifts, 0.06,
                )
            else:
                decision = teacher.decide(geometry)
                if decision.refusal is not None:
                    result["failure_phase"] = teacher.phase.value
                    result["refusal"] = str(decision.refusal)
                    break
                environment.stage = teacher.phase.value
                clipped, _mask = contract.clip(decision.action)
                environment.step(clipped)
            row = environment.rows[-1]
            row["pre_step_socket_gap_mm"] = gap
            if row["success"]:
                result["seatable"] = True
                # Preserve the historical probe's PRE-terminal transform convention.
                captured = relative_pose(geometry.peg, geometry.socket)
                break
            if row["done"]:
                break
            if not calibration and not privileged and (
                teacher.phase is ScriptedInsertionStage.INSERT
                and (control or teacher._movement_stalled())
            ):
                privileged = True
                result["handoff_step"] = len(environment.rows)
        gaps = [r["pre_step_socket_gap_mm"] for r in environment.rows
                if r.get("pre_step_socket_gap_mm") is not None]
        privileged_gaps = [r["pre_step_socket_gap_mm"] for r in environment.rows
                           if r["stage"] == "privileged_insert"]
        result.update(
            steps=len(environment.rows),
            privileged_steps=len(privileged_gaps),
            minimum_socket_gap_mm=min(gaps) if gaps else None,
            privileged_minimum_socket_gap_mm=min(privileged_gaps) if privileged_gaps else None,
            final_pre_step_socket_gap_mm=gaps[-1] if gaps else None,
            maximum_reward=max((r["reward"] for r in environment.rows), default=0),
            failure_phase=(None if result["seatable"] else
                           result.get("failure_phase", environment.stage)),
            final_snapshot=environment.diagnostic_snapshot(),
            joint_limit_violations=sum(r["joint_limit_violations"] for r in environment.rows),
            unexpected_collisions=sum(r["unexpected_collisions"] for r in environment.rows),
        )
    except (RuntimeError, ValueError, ZeroDivisionError) as error:
        result.update(error=f"{type(error).__name__}: {error}", steps=len(environment.rows),
                      failure_phase=environment.stage)
    finally:
        result["trace"] = environment.rows
        environment.close()
    return result, captured


def main():
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8-sig"))
    for name, expected in registration["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Registered source identity drift: {name}")
    image = os.environ.get("ROSETTA_CONTAINER_IMAGE_ID")
    if image != registration["container_image_id"]:
        raise RuntimeError("Registered container identity mismatch")
    OUTPUT.mkdir(exist_ok=False)
    write_new(OUTPUT / "registration.json", registration)
    contract = load_action_contract(ROOT / "configs/sim/aloha_insertion_smolvla.yaml")
    solver = MinkAlohaIkSolver.from_gym_aloha(
        left_site="cali_left_site1", right_site="cali_right_site1",
        settings=default_ik_settings(),
    )
    write_new(OUTPUT / "runtime.json", {
        "container_image_id": image, "ik_settings": asdict(default_ik_settings()),
        "command_settings": asdict(GeometricCommandSettings()),
        "escape_settings": asdict(GeometricEscapeSettings(hold_enabled=False)),
        "registration_sha256": hashlib.sha256(REGISTRATION.read_bytes()).hexdigest(),
        "gating": False, "hidden_test_loaded": False, "model_loaded": False,
    })
    calibration, seated_rel = run_seed(1903, contract, solver, calibration=True)
    write_new(OUTPUT / "calibration-1903.json", calibration)
    print(
        f"calibration: success={calibration['seatable']} steps={calibration['steps']}",
        flush=True,
    )
    if not calibration["seatable"] or seated_rel is None or calibration["error"]:
        raise RuntimeError("Calibration failed; Gate 4 seeds not opened")
    write_new(OUTPUT / "seated-transform.json", {
        "position": seated_rel.position.tolist(), "quaternion": seated_rel.quaternion.tolist(),
        "convention": "pre-terminal geometry, historical 1901 probe convention",
    })
    control, _ = run_seed(1903, contract, solver, seated_rel, control=True)
    write_new(OUTPUT / "control-1903.json", control)
    print(f"control: success={control['seatable']} steps={control['steps']}", flush=True)
    if control["error"] or not control["seatable"] or not control.get("privileged_steps"):
        raise RuntimeError("Privileged sanity control failed; Gate 4 seeds not opened")
    rows = []
    for seed in SEEDS:
        result, _ = run_seed(seed, contract, solver, seated_rel)
        write_new(OUTPUT / f"seed-{seed}.json", result)
        rows.append({key: value for key, value in result.items() if key != "trace"})
        print(json.dumps(rows[-1]), flush=True)
        if result["error"]:
            raise RuntimeError(f"Instrument failed on seed {seed}; evidence preserved")
    write_new(OUTPUT / "summary.json", {"gating": False, "rows": rows,
                                       "verdict": classify(rows)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
