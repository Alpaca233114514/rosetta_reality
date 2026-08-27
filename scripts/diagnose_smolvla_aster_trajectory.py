"""Trace Aster against the registered expert replay from one train initial state."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import sim_gate as dataset_gate  # noqa: E402
import smolvla_action_repair_sim_gate as repair_gate  # noqa: E402
import smolvla_sim_gate as simulator  # noqa: E402

from rosetta_reality.data import resolve_prepared_cache  # noqa: E402
from rosetta_reality.data.config import load_dataset_config  # noqa: E402
from rosetta_reality.experiment import (  # noqa: E402
    file_sha256,
    stable_hash,
    workspace_code_identity,
)
from rosetta_reality.features import create_json  # noqa: E402
from rosetta_reality.sim import GymAlohaEnvironment, load_action_contract  # noqa: E402

DEFAULT_PLAN = REPOSITORY_ROOT / "configs/vla/smolvla_450m_aloha_insertion_aster_trace_sim_007.yaml"
DATASET_CONFIG = REPOSITORY_ROOT / "configs/data/aloha_sim_insertion_m2.yaml"
ASTER_PLAN = REPOSITORY_ROOT / "configs/vla/smolvla_450m_aloha_insertion_aster_batch8_003.yaml"
STATE_MAE_THRESHOLDS = (0.005, 0.01, 0.025, 0.05, 0.1)


def _state(observation: dict[str, Any], dimension: int) -> torch.Tensor:
    value = observation.get("robot_state")
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != (dimension,):
        raise ValueError("Trajectory observation violates the registered state contract.")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError("Trajectory state contains NaN or Inf.")
    return value.to(torch.float32).cpu()


def _new_crossings(
    value: float,
    step: int,
    crossings: dict[str, int | None],
) -> bool:
    """Record each first state-MAE threshold crossing and report any change."""

    changed = False
    for threshold in STATE_MAE_THRESHOLDS:
        key = f"{threshold:g}"
        if crossings[key] is None and value >= threshold:
            crossings[key] = step
            changed = True
    return changed


def _pose_delta(
    expert: dict[str, Any], policy: dict[str, Any], body: str
) -> dict[str, Any] | None:
    expert_pose = expert.get("bodies", {}).get(body)
    policy_pose = policy.get("bodies", {}).get(body)
    if not isinstance(expert_pose, dict) or not isinstance(policy_pose, dict):
        return None
    expert_position = torch.tensor(expert_pose["position"], dtype=torch.float64)
    policy_position = torch.tensor(policy_pose["position"], dtype=torch.float64)
    difference = policy_position - expert_position
    return {
        "expert_position": expert_position.tolist(),
        "policy_position": policy_position.tolist(),
        "policy_minus_expert": difference.tolist(),
        "position_l2": float(difference.square().sum().sqrt()),
    }


def _main(args: argparse.Namespace) -> int:
    plan_path = args.plan.resolve()
    run_root = simulator._absolute_root("ROSETTA_RUN_ROOT")
    compiler_cache = (
        run_root / "compiler_cache" / f"aster-trace-{file_sha256(plan_path)[:12]}"
    )
    triton_cache = compiler_cache / "triton"
    inductor_cache = compiler_cache / "inductor"
    triton_cache.mkdir(parents=True, exist_ok=True)
    inductor_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(triton_cache)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_cache)
    plan, artifact, manifest, config, normalization = simulator._load_artifact(plan_path)
    contract_path = simulator._repository_path(plan["action_contract"]["path"])
    contract = load_action_contract(contract_path)
    aster_plan = simulator._load_yaml(ASTER_PLAN)
    if (
        args.episode not in aster_plan["training"]["episodes"]
        or args.episode in aster_plan["validation"]["episodes"]
        or args.episode in {31, 6, 1, 24, 5}
    ):
        raise ValueError("Trajectory trace must use a registered Aster train episode.")
    dataset_config = load_dataset_config(DATASET_CONFIG)
    dataset_root, dataset_manifest = resolve_prepared_cache(
        dataset_config,
        REPOSITORY_ROOT,
        validate_checksums=True,
    )
    rows = dataset_gate._dataset_rows(dataset_root, args.episode, dataset_config.fields)
    maximum_steps = min(args.maximum_steps, len(rows))
    if maximum_steps < 3:
        raise ValueError("Trajectory trace requires at least three expert rows.")

    policy = repair_gate._ActionRepairOnlineSmolVLA(
        artifact,
        config,
        normalization,
        contract,
    )
    policy.configure_noise("seeded_standard_normal", args.policy_noise_seed)
    expert_environment = GymAlohaEnvironment(
        contract, maximum_episode_steps=maximum_steps
    )
    policy_environment = GymAlohaEnvironment(
        contract, maximum_episode_steps=maximum_steps
    )
    crossings = {f"{value:g}": None for value in STATE_MAE_THRESHOLDS}
    first_events: dict[str, int | None] = {
        "expert_nonzero_reward": None,
        "policy_nonzero_reward": None,
        "expert_done": None,
        "policy_done": None,
        "policy_joint_limit_violation": None,
        "policy_unexpected_collision": None,
    }
    trace_steps: list[dict[str, Any]] = []
    state_mae_values: list[float] = []
    action_mae_values: list[float] = []
    expert_maximum_reward = 0.0
    policy_maximum_reward = 0.0

    def first(name: str, step: int, condition: bool) -> bool:
        if condition and first_events[name] is None:
            first_events[name] = step
            return True
        return False

    try:
        expert_observation = dict(expert_environment.reset(seed=args.seed))
        policy_observation = dict(policy_environment.reset(seed=args.seed))
        reset_state_mae = float(
            (
                _state(policy_observation, policy.state_dimension)
                - _state(expert_observation, policy.state_dimension)
            )
            .abs()
            .mean()
        )
        reset_snapshot = {
            "expert": expert_environment.diagnostic_snapshot(),
            "policy": policy_environment.diagnostic_snapshot(),
        }
        if reset_state_mae > 1e-7:
            raise RuntimeError("Independent simulator resets are not deterministic.")

        for step, row in enumerate(rows[:maximum_steps]):
            expert_pre_state = _state(expert_observation, policy.state_dimension)
            policy_pre_state = _state(policy_observation, policy.state_dimension)
            expert_raw = torch.as_tensor(
                row[dataset_config.fields.action], dtype=torch.float32
            )
            contract.validate_tensor(expert_raw, allow_chunk=False)
            expert_action, expert_clip = contract.clip(expert_raw)
            policy_raw_chunk, policy_processed_chunk = policy.predict(
                policy_observation,
                plan["inference"]["instruction"],
            )
            policy_raw = policy_raw_chunk[0]
            policy_processed = policy_processed_chunk[0]
            policy_action, policy_clip = contract.clip(policy_processed)

            expert_observation, expert_reward, expert_done, expert_info = (
                expert_environment.step(expert_action)
            )
            policy_observation, policy_reward, policy_done, policy_info = (
                policy_environment.step(policy_action)
            )
            expert_observation = dict(expert_observation)
            policy_observation = dict(policy_observation)
            expert_state = _state(expert_observation, policy.state_dimension)
            policy_state = _state(policy_observation, policy.state_dimension)
            state_difference = policy_state - expert_state
            action_difference = policy_action - expert_action
            state_mae = float(state_difference.abs().mean())
            action_mae = float(action_difference.abs().mean())
            state_mae_values.append(state_mae)
            action_mae_values.append(action_mae)
            expert_maximum_reward = max(expert_maximum_reward, float(expert_reward))
            policy_maximum_reward = max(policy_maximum_reward, float(policy_reward))

            expert_snapshot = expert_environment.diagnostic_snapshot()
            policy_snapshot = policy_environment.diagnostic_snapshot()
            policy_unexpected = sum(
                policy_environment.is_unexpected_collision_pair(first_name, second_name)
                for first_name, second_name in policy_environment.contact_pairs()
            )
            event = _new_crossings(state_mae, step, crossings)
            event |= first("expert_nonzero_reward", step, expert_reward != 0.0)
            event |= first("policy_nonzero_reward", step, policy_reward != 0.0)
            event |= first("expert_done", step, expert_done)
            event |= first("policy_done", step, policy_done)
            event |= first(
                "policy_joint_limit_violation",
                step,
                bool(policy_snapshot["joint_limit_violations"]),
            )
            event |= first(
                "policy_unexpected_collision", step, policy_unexpected > 0
            )
            if step < 12 or step % 25 == 0 or event or step + 1 == maximum_steps:
                trace_steps.append(
                    {
                        "step": step,
                        "dataset_frame": int(row[dataset_config.fields.frame_index]),
                        "expert_reference": {
                            "state_conditioned": False,
                            "recovery_oracle": False,
                        },
                        "state": {
                            "expert_pre": expert_pre_state.tolist(),
                            "policy_pre": policy_pre_state.tolist(),
                            "expert_post": expert_state.tolist(),
                            "policy_post": policy_state.tolist(),
                            "post_mae": state_mae,
                            "post_l2": float(state_difference.square().sum().sqrt()),
                        },
                        "action": {
                            "expert_raw": expert_raw.tolist(),
                            "expert_executed": expert_action.tolist(),
                            "expert_clipped_elements": int(expert_clip.sum()),
                            "policy_raw": policy_raw.tolist(),
                            "policy_processed": policy_processed.tolist(),
                            "policy_executed": policy_action.tolist(),
                            "policy_clipped_elements": int(policy_clip.sum()),
                            "executed_mae_vs_time_indexed_expert": action_mae,
                        },
                        "outcome": {
                            "expert_reward": float(expert_reward),
                            "policy_reward": float(policy_reward),
                            "expert_done": bool(expert_done),
                            "policy_done": bool(policy_done),
                            "expert_success": bool(expert_info.get("is_success", False)),
                            "policy_success": bool(policy_info.get("is_success", False)),
                            "policy_unexpected_collisions": policy_unexpected,
                        },
                        "pose_delta": {
                            body: _pose_delta(expert_snapshot, policy_snapshot, body)
                            for body in (
                                "peg",
                                "socket",
                                "vx300s_left/gripper_link",
                                "vx300s_right/gripper_link",
                            )
                        },
                        "expert_snapshot": expert_snapshot,
                        "policy_snapshot": policy_snapshot,
                    }
                )
            if expert_done or policy_done:
                break

        report = {
            "schema_version": 1,
            "status": "complete",
            "diagnostic": "aster_train_pose_expert_policy_trajectory_divergence",
            "experiment_id": plan["experiment_id"],
            "artifact_id": manifest["artifact_id"],
            "artifact_manifest_sha256": file_sha256(artifact / "manifest.json"),
            "simulation_plan_sha256": file_sha256(plan_path),
            "dataset_revision": dataset_manifest.resolved_revision,
            "dataset_manifest_sha256": file_sha256(dataset_root / "manifest.json"),
            "episode": args.episode,
            "seed": args.seed,
            "policy_noise_seed": args.policy_noise_seed,
            "maximum_steps": maximum_steps,
            "protocol": {
                "expert": "time_indexed_dataset_action_in_independent_environment",
                "policy": "receding_horizon_first_action",
                "warning": (
                    "After divergence the time-indexed expert action is a reference, "
                    "not a state-conditioned recovery oracle."
                ),
                "test_split_opened": False,
            },
            "reset": {
                "cross_environment_state_mae": reset_state_mae,
                "snapshot": reset_snapshot,
            },
            "summary": {
                "steps_executed": len(state_mae_values),
                "first_state_mae_crossings": crossings,
                "first_events": first_events,
                "step_zero_action_mae": action_mae_values[0],
                "step_zero_post_state_mae": state_mae_values[0],
                "maximum_state_mae": max(state_mae_values),
                "final_state_mae": state_mae_values[-1],
                "mean_time_indexed_action_mae": sum(action_mae_values)
                / len(action_mae_values),
                "expert_maximum_reward": expert_maximum_reward,
                "policy_maximum_reward": policy_maximum_reward,
            },
            "trace_steps": trace_steps,
            "code_identity": workspace_code_identity(REPOSITORY_ROOT),
            "hidden_test_loaded": False,
        }
        json.dumps(report, allow_nan=False)
        destination = (
            simulator._absolute_root("ROSETTA_RUN_ROOT")
            / plan["experiment_id"]
            / "diagnostics"
            / f"aster-trajectory-{stable_hash(report)[:16]}.json"
        )
        create_json(destination, report)
        print(json.dumps(report["summary"], indent=2, sort_keys=True))
        print(f"Report: {destination.relative_to(simulator._absolute_root('ROSETTA_RUN_ROOT'))}")
        return 0
    finally:
        expert_environment.close()
        policy_environment.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--episode", type=int, default=2)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--policy-noise-seed", type=int, default=10)
    parser.add_argument("--maximum-steps", type=int, default=320)
    args = parser.parse_args()
    if args.maximum_steps <= 0 or args.seed < 0 or args.policy_noise_seed < 0:
        raise ValueError("Trajectory limits and seeds must be non-negative.")
    if not math.isfinite(float(args.maximum_steps)):
        raise ValueError("Trajectory maximum steps must be finite.")
    return _main(args)


if __name__ == "__main__":
    raise SystemExit(main())
