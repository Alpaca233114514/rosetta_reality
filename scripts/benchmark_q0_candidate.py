"""Bounded Q0 cost screening: at most three hold actions, no policy calls."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))


def benchmark() -> dict:
    import numpy as np

    from rosetta_reality.sim import load_action_contract
    from rosetta_reality.sim.coupled_translation_teacher import (
        CANDIDATE_ID,
        TRANSLATIONS_M,
        build_teacher,
    )
    from rosetta_reality.sim.gym_aloha import GymAlohaEnvironment
    from rosetta_reality.sim.scripted_teacher import (
        ScriptedInsertionStage,
        _expanded_robot_qpos,
    )
    from rosetta_reality.sim.scripted_teacher_observer import build_observation

    versions = {}
    for package in ("torch", "mujoco", "dm-control", "gym-aloha", "mink", "qpsolvers", "daqp"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_available"
    result = {"schema": "rosetta.q0_candidate_benchmark.v1",
              "candidate_id": CANDIDATE_ID, "seed": 10,
              "environment_steps": 0, "model_forward_calls": 0,
              "benchmark_steps_attempted": 0, "benchmark_step_cap": 3,
              "benchmark_counts_as_gate1": False,
              "optimizer_steps": 0, "package_versions": versions,
              "work_cap_seconds": 1800, "screening_action_cap": 3000,
              "estimate_margin": 1.5, "passed": False,
              "estimate_is_guaranteed_upper_bound": False,
              "limitations": ["single_reset_geometry_not_loaded_contact_configuration",
                              "three_hold_steps_do_not_measure_loaded_contact_step_latency",
                              "solver_tail_latency_and_real_stage_mix_unmeasured"]}
    environment = None
    started = time.perf_counter()
    try:
        contract = load_action_contract(REPOSITORY_ROOT / "configs/sim/aloha_insertion_smolvla.yaml")
        environment = GymAlohaEnvironment(contract, maximum_episode_steps=500)
        reset_started = time.perf_counter()
        environment.reset(seed=10)
        observation = build_observation(environment, contract)
        reset_seconds = time.perf_counter() - reset_started
        result["reset_and_observation_seconds"] = reset_seconds
        initial_joint_limits = environment.state_limit_violation_count()
        initial_collisions = environment.unexpected_collision_count()
        result["reset_safety"] = {"joint_limit_violations": initial_joint_limits,
                                  "unexpected_collisions": initial_collisions}
        if initial_joint_limits or initial_collisions:
            raise RuntimeError("unsafe reset state")
        # Preserve the reset geometry for the 54-solve cost sample. The hold
        # commands below benchmark real step/render/observer overhead only.
        step_observation = observation
        step_measurements = []
        result["step_measurements"] = step_measurements
        for step_index in range(3):
            step_started = time.perf_counter()
            action = step_observation.robot_state.clone()
            contract.validate_tensor(action, allow_chunk=False)
            _, clip_mask = contract.clip(action)
            if bool(clip_mask.any()):
                raise RuntimeError("current-state hold command lies outside Action Contract")
            result["benchmark_steps_attempted"] += 1
            state, reward, done, info = environment.step(action)
            result["environment_steps"] += 1
            step_observation = build_observation(environment, contract)
            contract.validate_tensor(state["robot_state"], allow_chunk=False)
            contract.validate_tensor(step_observation.robot_state, allow_chunk=False)
            joint_limits = environment.state_limit_violation_count()
            collisions = environment.unexpected_collision_count()
            reward_finite = math.isfinite(reward)
            clipped = bool(environment.last_clip_mask.any()) or bool(info.get("clipped_fields"))
            safe = reward_finite and not (joint_limits or collisions or clipped or done)
            step_measurements.append({"step": step_index,
                                      "seconds": time.perf_counter() - step_started,
                                      "joint_limit_violations": joint_limits,
                                      "unexpected_collisions": collisions,
                                      "reward_finite": reward_finite,
                                      "action_clipped": clipped, "done": done,
                                      "safe": safe})
            if not safe:
                raise RuntimeError("unsafe or prematurely terminated benchmark hold step")
        maximum_step_seconds = max(item["seconds"] for item in step_measurements)
        result["maximum_step_render_observer_seconds"] = maximum_step_seconds
        teacher = build_teacher()
        # Only enumerate the registered branch family. No candidate decision is
        # executed and no candidate setting/acceptance function is overridden.
        teacher._phase = ScriptedInsertionStage.INSERT
        seeds = teacher._ik_seeds(_expanded_robot_qpos(observation.robot_state))
        if len(seeds) != 9 or len(TRANSLATIONS_M) != 6:
            raise RuntimeError("registered nine-branch six-target benchmark identity changed")
        measurements = []
        result["measurements"] = measurements
        loop_started = time.perf_counter()
        for seed_index, seed in enumerate(seeds):
            for translation_index, translation in enumerate(TRANSLATIONS_M):
                delta = np.asarray(translation, dtype=np.float64)
                solve_started = time.perf_counter()
                solved = teacher._solver.solve(
                    seed,
                    left_position=observation.left_eef.position.numpy() + delta,
                    left_quaternion_wxyz=observation.left_eef.quaternion.numpy(),
                    right_position=observation.right_eef.position.numpy() + delta,
                    right_quaternion_wxyz=observation.right_eef.quaternion.numpy(),
                )
                elapsed = time.perf_counter() - solve_started
                if not np.isfinite(solved.qpos).all() or not all(
                    math.isfinite(float(value))
                    for value in (*solved.position_errors_m, *solved.orientation_errors_rad)
                ):
                    raise ValueError("non-finite benchmark IK result")
                measurements.append({"seed_index": seed_index,
                                     "translation_index": translation_index,
                                     "seconds": elapsed, "iterations": solved.iterations})
        loop_seconds = time.perf_counter() - loop_started
        estimate = 6 * reset_seconds + (loop_seconds + maximum_step_seconds) * 3000 * 1.5
        if not math.isfinite(estimate):
            raise ValueError("non-finite benchmark estimate")
        result.update({"reset_and_observation_seconds": reset_seconds,
                       "ik_loop_54_seconds": loop_seconds, "solve_count": len(measurements),
                       "total_iterations": sum(item["iterations"] for item in measurements),
                       "maximum_single_solve_seconds": max(item["seconds"] for item in measurements),
                       "solver_maximum_iterations": teacher._solver.settings.maximum_iterations,
                       "estimated_screening_reset_count": 6,
                       "estimate_formula": "6*cold_reset+(54_solve_loop+max_hold_step)*3000*1.5",
                       "estimated_screening_seconds": estimate,
                       "measurements": measurements, "passed": estimate <= 1800,
                       "status": "budget_admitted" if estimate <= 1800 else "budget_estimate_exceeded"})
    except Exception as error:
        result.update({"status": "benchmark_failed", "error_type": type(error).__name__,
                       "passed": False})
    finally:
        if environment is not None:
            try:
                environment.close()
            except Exception as error:
                result.update({"status": "benchmark_close_failed",
                               "close_error_type": type(error).__name__, "passed": False})
        result["wall_seconds"] = time.perf_counter() - started
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("benchmark output must be create-only")
    try:
        payload = benchmark()
    except Exception as error:
        payload = {"schema": "rosetta.q0_candidate_benchmark.v1",
                   "status": "benchmark_bootstrap_failed", "passed": False,
                   "error_type": type(error).__name__}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": payload["status"], "passed": payload["passed"]}))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
