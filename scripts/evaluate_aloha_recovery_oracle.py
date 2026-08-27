"""Evaluate a train-only state-conditioned recovery oracle without writing labels."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rosetta_reality.data import resolve_prepared_cache  # noqa: E402
from rosetta_reality.data.config import DatasetConfig, load_dataset_config  # noqa: E402
from rosetta_reality.experiment import (  # noqa: E402
    file_sha256,
    stable_hash,
    workspace_code_identity,
)
from rosetta_reality.features import create_json  # noqa: E402
from rosetta_reality.sim import (  # noqa: E402
    GymAlohaEnvironment,
    OracleOutOfDistributionError,
    OracleReferenceTrajectory,
    StateConditionedTrajectoryOracle,
    load_action_contract,
)
from rosetta_reality.sim.action_contract import ActionContract  # noqa: E402

DEFAULT_PLAN = REPOSITORY_ROOT / "configs/sim/aloha_insertion_recovery_oracle_001.yaml"
LEFT_SITE = "cali_left_site1"
RIGHT_SITE = "cali_right_site1"
LEFT_JOINTS = (
    "vx300s_left/waist",
    "vx300s_left/shoulder",
    "vx300s_left/elbow",
    "vx300s_left/forearm_roll",
    "vx300s_left/wrist_angle",
    "vx300s_left/wrist_rotate",
)
RIGHT_JOINTS = (
    "vx300s_right/waist",
    "vx300s_right/shoulder",
    "vx300s_right/elbow",
    "vx300s_right/forearm_roll",
    "vx300s_right/wrist_angle",
    "vx300s_right/wrist_rotate",
)


@dataclass(frozen=True, slots=True)
class SourceTrajectory:
    """Successful source replay plus simulator object-state provenance."""

    reference: OracleReferenceTrajectory
    object_positions: dict[str, Tensor]
    gripper_close_index: int
    maximum_reward: float
    steps_executed: int


@dataclass(frozen=True, slots=True)
class RetargetResult:
    """One locally retargeted logical robot state or action."""

    value: Tensor
    maximum_ik_error: float
    ik_success: bool


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _load_plan(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    plan = _mapping(raw, "Recovery oracle plan")
    if plan.get("schema_version") != 1:
        raise ValueError("Recovery oracle plan must use schema version one.")
    if plan.get("status") != "diagnostic_preregistered_no_label_collection":
        raise ValueError("Recovery oracle plan is not the registered diagnostic plan.")
    for relative, expected in _mapping(
        plan.get("implementation_files"), "implementation_files"
    ).items():
        path_value = Path(str(relative))
        if path_value.is_absolute() or ".." in path_value.parts:
            raise ValueError(f"Unsafe implementation path: {relative!r}.")
        if expected == "PLACEHOLDER" or file_sha256(REPOSITORY_ROOT / path_value) != expected:
            raise ValueError(f"Recovery oracle implementation identity differs: {relative}.")
    return plan


def _repository_path(raw: str) -> Path:
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Expected a repository-relative path, received {raw!r}.")
    return REPOSITORY_ROOT / relative


def _run_root() -> Path:
    raw = os.environ.get("ROSETTA_RUN_ROOT")
    if not raw:
        raise OSError("ROSETTA_RUN_ROOT must be defined by the container launcher.")
    root = Path(raw).resolve()
    if not root.is_absolute():
        raise ValueError("ROSETTA_RUN_ROOT must resolve to an absolute path.")
    return root


def _validate_plan_boundaries(plan: dict[str, Any]) -> None:
    scope = _mapping(plan.get("scope"), "scope")
    train = {int(value) for value in scope.get("train_episodes", [])}
    validation = {int(value) for value in scope.get("validation_episodes", [])}
    hidden = {int(value) for value in scope.get("hidden_test_episodes", [])}
    if not train or train & validation or train & hidden or validation & hidden:
        raise ValueError("Recovery oracle dataset splits are empty or overlap.")
    if scope.get("hidden_test_loaded") is not False:
        raise ValueError("Recovery oracle plan must keep the hidden test sealed.")
    candidates = plan.get("source_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Recovery oracle plan requires source candidates.")
    source_episodes = [int(_mapping(item, "source candidate")["episode"]) for item in candidates]
    if len(source_episodes) != len(set(source_episodes)) or not set(source_episodes) <= train:
        raise ValueError("Recovery oracle sources must be unique train-only episodes.")

    evaluation = _mapping(plan.get("evaluation"), "evaluation")
    if evaluation.get("stages") != "exact_control_then_tuning_then_development":
        raise ValueError("Recovery oracle plan must preserve the staged evaluation boundary.")
    tuning = {int(value) for value in evaluation.get("tuning_simulator_seeds", [])}
    development = {int(value) for value in evaluation.get("development_simulator_seeds", [])}
    collection = {
        int(value) for value in evaluation.get("reserved_collection_simulator_seeds", [])
    }
    policy_gate = {int(value) for value in evaluation.get("reserved_policy_gate4_seeds", [])}
    seed_groups = (tuning, development, collection, policy_gate)
    if any(not group for group in seed_groups):
        raise ValueError(
            "Tuning, development, collection, and policy Gate seeds must be registered."
        )
    overlaps = any(
        first & second
        for index, first in enumerate(seed_groups)
        for second in seed_groups[index + 1 :]
    )
    if overlaps:
        raise ValueError(
            "Tuning, development, collection, and policy Gate seeds must be disjoint."
        )
    exact = _mapping(evaluation.get("exact_control"), "exact_control")
    if int(exact["episode"]) not in train:
        raise ValueError("Exact oracle control must use a train episode.")


def _robot_state(observation: dict[str, Any], dimension: int) -> Tensor:
    state = observation.get("robot_state")
    if not isinstance(state, Tensor) or state.shape != (dimension,):
        raise ValueError("Simulator observation violates the registered robot-state shape.")
    state = state.detach().to(torch.float32).cpu()
    if not bool(torch.isfinite(state).all()):
        raise ValueError("Simulator robot state contains NaN or Inf.")
    return state


def _physics(environment: GymAlohaEnvironment) -> Any:
    unwrapped = getattr(environment.raw_environment, "unwrapped", environment.raw_environment)
    control_environment = getattr(unwrapped, "_env", None)
    physics = getattr(control_environment, "physics", None)
    if physics is None:
        raise RuntimeError("Recovery oracle requires the registered MuJoCo backend.")
    return physics


def _object_positions(environment: GymAlohaEnvironment) -> dict[str, Tensor]:
    bodies = environment.diagnostic_snapshot().get("bodies", {})
    result: dict[str, Tensor] = {}
    for name in ("socket", "peg"):
        pose = bodies.get(name)
        if not isinstance(pose, dict):
            raise ValueError(f"Simulator diagnostic snapshot lacks {name!r}.")
        position = torch.as_tensor(pose.get("position"), dtype=torch.float32)
        if position.shape != (3,) or not bool(torch.isfinite(position).all()):
            raise ValueError(f"Simulator {name} position is invalid.")
        result[name] = position
    return result


def _trajectory_rows(
    root: Path,
    episode: int,
    dataset_config: DatasetConfig,
    contract: ActionContract,
) -> list[dict[str, Any]]:
    import pyarrow.dataset as arrow_dataset

    fields = dataset_config.fields
    dataset = arrow_dataset.dataset(root / "data", format="parquet")
    table = dataset.to_table(
        columns=[
            fields.episode_index,
            fields.frame_index,
            fields.timestamp,
            fields.state,
            fields.action,
        ],
        filter=arrow_dataset.field(fields.episode_index) == episode,
    )
    rows = sorted(table.to_pylist(), key=lambda row: int(row[fields.frame_index]))
    if not rows:
        raise ValueError(f"Train source episode {episode} contains no rows.")
    for expected_frame, row in enumerate(rows):
        if (
            int(row[fields.episode_index]) != episode
            or int(row[fields.frame_index]) != expected_frame
        ):
            raise ValueError("Source trajectory is not an exact contiguous episode.")
        timestamp = float(row[fields.timestamp])
        if not math.isclose(
            timestamp,
            expected_frame / contract.frequency_hz,
            rel_tol=0.0,
            abs_tol=1e-4,
        ):
            raise ValueError("Source trajectory violates the Action Contract frequency.")
        action = torch.as_tensor(row[fields.action], dtype=torch.float32)
        contract.validate_tensor(action, allow_chunk=False)
    return rows


def _gripper_close_index(actions: Tensor) -> int:
    gates: list[int] = []
    for column in (6, 13):
        values = actions[:, column]
        peak = int(torch.argmax(values))
        candidates = torch.nonzero(values[peak:].lt(0.35), as_tuple=False).flatten()
        if candidates.numel() == 0:
            raise ValueError("Successful source has no post-open gripper-close transition.")
        gates.append(peak + int(candidates[0]))
    return max(gates)


def _replay_source(
    rows: list[dict[str, Any]],
    *,
    episode: int,
    seed: int,
    dataset_config: DatasetConfig,
    contract: ActionContract,
) -> SourceTrajectory:
    environment = GymAlohaEnvironment(contract, maximum_episode_steps=len(rows))
    states: list[Tensor] = []
    actions: list[Tensor] = []
    objects: dict[str, list[Tensor]] = {"socket": [], "peg": []}
    first_progress: int | None = None
    maximum_reward = 0.0
    success = False
    try:
        observation = dict(environment.reset(seed=seed))
        for index, row in enumerate(rows):
            states.append(_robot_state(observation, contract.dimension))
            positions = _object_positions(environment)
            for name in objects:
                objects[name].append(positions[name])
            raw_action = torch.as_tensor(
                row[dataset_config.fields.action], dtype=torch.float32
            )
            action, _clip_mask = contract.clip(raw_action)
            actions.append(action.detach().cpu())
            observation_value, reward, done, info = environment.step(action)
            observation = dict(observation_value)
            maximum_reward = max(maximum_reward, float(reward))
            if reward >= 1.0 and first_progress is None:
                first_progress = index
            if done:
                success = bool(info.get("is_success", False))
                break
    finally:
        environment.close()
    if not success or maximum_reward != 4.0 or first_progress is None:
        raise RuntimeError(
            f"Source episode {episode} seed {seed} is not a successful reward-four replay."
        )
    state_tensor = torch.stack(states)
    action_tensor = torch.stack(actions)
    reference = OracleReferenceTrajectory(
        states=state_tensor,
        actions=action_tensor,
        source_episode=episode,
        source_seed=seed,
        first_progress_index=first_progress,
        terminal_reward=maximum_reward,
        terminal_success=success,
    )
    return SourceTrajectory(
        reference=reference,
        object_positions={name: torch.stack(values) for name, values in objects.items()},
        gripper_close_index=_gripper_close_index(action_tensor),
        maximum_reward=maximum_reward,
        steps_executed=reference.length,
    )


def _expanded_robot_qpos(logical: Tensor) -> np.ndarray:
    from gym_aloha.tasks.sim import unnormalize_puppet_gripper_position

    value = logical.detach().to(torch.float64).cpu().numpy()
    left_gripper = float(unnormalize_puppet_gripper_position(float(value[6])))
    right_gripper = float(unnormalize_puppet_gripper_position(float(value[13])))
    return np.concatenate(
        (
            value[:6],
            np.asarray([left_gripper, -left_gripper]),
            value[7:13],
            np.asarray([right_gripper, -right_gripper]),
        )
    )


def _site_pose(physics: Any, site_name: str) -> tuple[np.ndarray, np.ndarray]:
    from dm_control.mujoco.wrapper.mjbindings import mjlib

    position = np.asarray(physics.named.data.site_xpos[site_name]).copy()
    quaternion = np.empty(4, dtype=physics.data.qpos.dtype)
    mjlib.mju_mat2Quat(quaternion, physics.named.data.site_xmat[site_name])
    return position, quaternion


def _retarget(
    scratch_physics: Any,
    logical: Tensor,
    *,
    left_shift: Tensor,
    right_shift: Tensor,
    contract: ActionContract,
    settings: dict[str, Any],
) -> RetargetResult:
    from dm_control.utils.inverse_kinematics import qpos_from_site_pose

    logical = logical.detach().to(torch.float32).cpu()
    contract.validate_tensor(logical, allow_chunk=False)
    if float(torch.cat((left_shift, right_shift)).abs().max()) <= 1e-9:
        return RetargetResult(
            value=logical.clone(),
            maximum_ik_error=0.0,
            ik_success=True,
        )
    scratch_physics.data.qpos[:16] = _expanded_robot_qpos(logical)
    scratch_physics.forward()
    results: list[Any] = []
    for site_name, joint_names, shift in (
        (LEFT_SITE, LEFT_JOINTS, left_shift),
        (RIGHT_SITE, RIGHT_JOINTS, right_shift),
    ):
        source_position, source_quaternion = _site_pose(scratch_physics, site_name)
        result = qpos_from_site_pose(
            scratch_physics,
            site_name,
            target_pos=source_position + shift.detach().cpu().numpy(),
            target_quat=source_quaternion,
            joint_names=joint_names,
            tol=float(settings["ik_tolerance"]),
            rot_weight=float(settings["ik_rotation_weight"]),
            regularization_threshold=float(settings["ik_regularization_threshold"]),
            regularization_strength=float(settings["ik_regularization_strength"]),
            max_update_norm=float(settings["ik_maximum_update_norm"]),
            max_steps=int(settings["ik_maximum_steps"]),
            inplace=False,
        )
        results.append(result)
    value = logical.clone()
    value[:6] = torch.as_tensor(results[0].qpos[:6], dtype=torch.float32)
    value[7:13] = torch.as_tensor(results[1].qpos[8:14], dtype=torch.float32)
    clipped, clip_mask = contract.clip(value)
    maximum_error = max(float(result.err_norm) for result in results)
    success = all(bool(result.success) for result in results)
    success = success and not bool(clip_mask[:6].any()) and not bool(clip_mask[7:13].any())
    success = success and maximum_error <= float(settings["maximum_accepted_ik_error"])
    return RetargetResult(
        value=clipped.detach().cpu(),
        maximum_ik_error=maximum_error,
        ik_success=success,
    )


def _offset_weights(source: SourceTrajectory, settings: dict[str, Any]) -> Tensor:
    approach_end = max(
        1,
        source.gripper_close_index
        - int(settings["approach_full_offset_steps_before_gripper_close"]),
    )
    fade_steps = int(settings["post_progress_offset_fade_steps"])
    if fade_steps <= 0:
        raise ValueError("Post-progress offset fade must be positive.")
    weights = torch.empty(source.reference.length, dtype=torch.float32)
    for index in range(source.reference.length):
        if index <= approach_end:
            weight = index / approach_end
        elif index <= source.reference.first_progress_index:
            weight = 1.0
        else:
            weight = max(
                0.0,
                1.0
                - (index - source.reference.first_progress_index) / fade_steps,
            )
        weights[index] = weight
    return weights


def _target_reference(
    source: SourceTrajectory,
    *,
    target_initial_objects: dict[str, Tensor],
    scratch_physics: Any,
    contract: ActionContract,
    settings: dict[str, Any],
) -> tuple[OracleReferenceTrajectory, Tensor, dict[str, Tensor], dict[str, Any]]:
    weights = _offset_weights(source, settings)
    offsets = {
        name: target_initial_objects[name] - positions[0]
        for name, positions in source.object_positions.items()
    }
    stride = int(settings["reference_anchor_stride"])
    if stride <= 0:
        raise ValueError("Reference IK anchor stride must be positive.")
    anchor_indices = sorted(
        {
            0,
            source.reference.length - 1,
            source.gripper_close_index,
            source.reference.first_progress_index,
            *range(0, source.reference.length, stride),
        }
    )
    anchor_deltas: list[Tensor] = []
    maximum_ik_error = 0.0
    ik_failures = 0
    for index in anchor_indices:
        weight = weights[index]
        result = _retarget(
            scratch_physics,
            source.reference.actions[index],
            left_shift=offsets["socket"] * weight,
            right_shift=offsets["peg"] * weight,
            contract=contract,
            settings=settings,
        )
        anchor_deltas.append(result.value - source.reference.actions[index])
        maximum_ik_error = max(maximum_ik_error, result.maximum_ik_error)
        ik_failures += int(not result.ik_success)
    correction = torch.empty_like(source.reference.actions)
    for anchor_position in range(len(anchor_indices) - 1):
        start = anchor_indices[anchor_position]
        stop = anchor_indices[anchor_position + 1]
        start_delta = anchor_deltas[anchor_position]
        stop_delta = anchor_deltas[anchor_position + 1]
        width = stop - start
        for index in range(start, stop + 1):
            fraction = (index - start) / width
            correction[index] = start_delta.lerp(stop_delta, fraction)
    transformed_states = source.reference.states + correction
    transformed_actions = source.reference.actions + correction
    transformed_actions, action_clip_mask = contract.clip(transformed_actions)
    transformed_states, state_clip_mask = contract.clip(transformed_states)
    ik_failures += int(bool(action_clip_mask.any()) or bool(state_clip_mask.any()))
    reference = OracleReferenceTrajectory(
        states=transformed_states,
        actions=transformed_actions,
        source_episode=source.reference.source_episode,
        source_seed=source.reference.source_seed,
        first_progress_index=source.reference.first_progress_index,
        terminal_reward=source.reference.terminal_reward,
        terminal_success=True,
    )
    return reference, weights, offsets, {
        "reference_ik_failures": ik_failures,
        "reference_maximum_ik_error": maximum_ik_error,
        "reference_ik_anchor_count": len(anchor_indices),
        "reference_ik_anchor_stride": stride,
    }


def _source_distance(source: SourceTrajectory, target: dict[str, Tensor]) -> float:
    difference = torch.cat(
        (
            source.object_positions["socket"][0][:2] - target["socket"][:2],
            source.object_positions["peg"][0][:2] - target["peg"][:2],
        )
    )
    return float(difference.square().sum().sqrt())


def _evaluate_target(
    *,
    seed: int,
    sources: list[SourceTrajectory],
    contract: ActionContract,
    plan: dict[str, Any],
    forced_source_episode: int | None = None,
) -> dict[str, Any]:
    evaluation = _mapping(plan["evaluation"], "evaluation")
    oracle_settings = _mapping(plan["oracle"], "oracle")
    retarget_settings = _mapping(plan["retargeting"], "retargeting")
    maximum_steps = int(evaluation["maximum_steps"])
    environment = GymAlohaEnvironment(contract, maximum_episode_steps=maximum_steps)
    scratch = GymAlohaEnvironment(contract, maximum_episode_steps=maximum_steps)
    trace: list[dict[str, Any]] = []
    maximum_reward = 0.0
    last_reward = 0.0
    success = False
    out_of_distribution = False
    action_ik_failures = 0
    maximum_action_ik_error = 0.0
    maximum_state_distance = 0.0
    maximum_object_correction = 0.0
    correction_saturations = 0
    try:
        observation = dict(environment.reset(seed=seed))
        scratch.reset(seed=0)
        target_initial = _object_positions(environment)
        eligible = sources
        if forced_source_episode is not None:
            eligible = [
                source
                for source in sources
                if source.reference.source_episode == forced_source_episode
            ]
            if len(eligible) != 1:
                raise ValueError("Exact control source episode is not unique in the source bank.")
        source = min(eligible, key=lambda value: _source_distance(value, target_initial))
        source_pose_distance = _source_distance(source, target_initial)
        exact_pose_control = source_pose_distance <= 1e-9
        if (
            exact_pose_control
            and retarget_settings.get("exact_control_disables_online_correction") is not True
        ):
            raise ValueError("Exact-pose control must disable online object correction.")
        reference, weights, offsets, reference_diagnostics = _target_reference(
            source,
            target_initial_objects=target_initial,
            scratch_physics=_physics(scratch),
            contract=contract,
            settings=retarget_settings,
        )
        oracle = StateConditionedTrajectoryOracle(
            reference,
            maximum_lookahead=int(oracle_settings["maximum_lookahead"]),
            maximum_state_distance=float(oracle_settings["maximum_state_distance"]),
            maximum_progress_state_distance=float(
                oracle_settings["maximum_progress_state_distance"]
            ),
            progress_reward_threshold=float(oracle_settings["progress_reward_threshold"]),
            post_progress_skip=int(oracle_settings["post_progress_skip"]),
        )
        for step in range(maximum_steps):
            state = _robot_state(observation, contract.dimension)
            try:
                decision = oracle.decide(state, observed_reward=last_reward)
            except OracleOutOfDistributionError as error:
                out_of_distribution = True
                trace.append(
                    {
                        "step": step,
                        "event": "oracle_out_of_distribution",
                        "error": str(error),
                    }
                )
                break
            index = decision.reference_index
            weight = float(weights[index])
            current_objects = _object_positions(environment)
            corrections: dict[str, Tensor] = {}
            correction_active = (
                not exact_pose_control
                and (
                    weight
                    >= float(retarget_settings["correction_activation_offset_weight"])
                    or decision.progress_unlocked
                )
            )
            limit = float(retarget_settings["maximum_object_correction_m"])
            for name in ("socket", "peg"):
                expected = source.object_positions[name][index] + offsets[name] * weight
                residual = current_objects[name] - expected
                correction = (
                    residual.clamp(min=-limit, max=limit)
                    if correction_active
                    else residual * 0
                )
                correction_saturations += int(
                    bool(residual.abs().gt(limit).any()) and correction_active
                )
                corrections[name] = correction
                maximum_object_correction = max(
                    maximum_object_correction,
                    float(correction.abs().max()),
                )
            correction_norm = float(
                torch.cat((corrections["socket"], corrections["peg"])).abs().max()
            )
            retargeted = _retarget(
                _physics(scratch),
                decision.action,
                left_shift=corrections["socket"],
                right_shift=corrections["peg"],
                contract=contract,
                settings=retarget_settings,
            )
            maximum_action_ik_error = max(
                maximum_action_ik_error, retargeted.maximum_ik_error
            )
            action_ik_failures += int(not retargeted.ik_success)
            if not retargeted.ik_success:
                trace.append(
                    {
                        "step": step,
                        "event": "action_ik_failure",
                        "reference_index": index,
                        "ik_error": retargeted.maximum_ik_error,
                    }
                )
                break
            observation_value, reward, done, info = environment.step(retargeted.value)
            observation = dict(observation_value)
            last_reward = float(reward)
            maximum_reward = max(maximum_reward, last_reward)
            maximum_state_distance = max(maximum_state_distance, decision.state_distance)
            if (
                step < 5
                or step % 50 == 0
                or reward != 0.0
                or done
                or correction_saturations > 0 and step % 10 == 0
            ):
                trace.append(
                    {
                        "step": step,
                        "reference_index": index,
                        "state_distance": decision.state_distance,
                        "offset_weight": weight,
                        "progress_unlocked": decision.progress_unlocked,
                        "reward": last_reward,
                        "online_correction_maximum_absolute_m": correction_norm,
                        "corrections": {
                            name: corrections[name].tolist() for name in corrections
                        },
                    }
                )
            if done:
                success = bool(info.get("is_success", False))
                break
        return {
            "seed": seed,
            "status": "passed" if success else "failed",
            "success": success,
            "maximum_reward": maximum_reward,
            "steps_executed": step + 1,
            "source_episode": source.reference.source_episode,
            "source_seed": source.reference.source_seed,
            "source_initial_pose_distance_xy_l2": source_pose_distance,
            "oracle_out_of_distribution": out_of_distribution,
            "maximum_state_distance": maximum_state_distance,
            "action_ik_failures": action_ik_failures,
            "maximum_action_ik_error": maximum_action_ik_error,
            "maximum_object_correction_m": maximum_object_correction,
            "object_correction_saturations": correction_saturations,
            **reference_diagnostics,
            "trace": trace,
        }
    finally:
        environment.close()
        scratch.close()


def _selected_source_candidates(
    plan: dict[str, Any], *, stage: str
) -> list[dict[str, Any]]:
    candidates = [
        _mapping(value, "source candidate") for value in plan["source_candidates"]
    ]
    if stage != "exact":
        return candidates
    exact = _mapping(_mapping(plan["evaluation"], "evaluation")["exact_control"], "exact")
    selected = [item for item in candidates if int(item["episode"]) == int(exact["episode"])]
    if len(selected) != 1:
        raise ValueError("Exact stage requires exactly one matching source candidate.")
    return selected


def _main(plan_path: Path, *, stage: str) -> int:
    plan = _load_plan(plan_path)
    _validate_plan_boundaries(plan)
    scope = _mapping(plan["scope"], "scope")
    dataset_config_path = _repository_path(str(scope["dataset_config"]))
    action_contract_path = _repository_path(str(scope["action_contract"]))
    dataset_config = load_dataset_config(dataset_config_path)
    contract = load_action_contract(action_contract_path)
    if dataset_config.revision != scope["dataset_revision"]:
        raise ValueError("Recovery plan dataset revision differs from the dataset config.")
    dataset_root, dataset_manifest = resolve_prepared_cache(
        dataset_config,
        REPOSITORY_ROOT,
        validate_checksums=True,
    )
    if dataset_manifest.resolved_revision != scope["dataset_revision"]:
        raise ValueError("Prepared cache revision differs from the recovery plan.")

    sources: list[SourceTrajectory] = []
    source_reports: list[dict[str, Any]] = []
    for item in _selected_source_candidates(plan, stage=stage):
        episode = int(item["episode"])
        seed = int(item["simulator_seed"])
        rows = _trajectory_rows(dataset_root, episode, dataset_config, contract)
        source = _replay_source(
            rows,
            episode=episode,
            seed=seed,
            dataset_config=dataset_config,
            contract=contract,
        )
        sources.append(source)
        source_reports.append(
            {
                "episode": episode,
                "simulator_seed": seed,
                "steps_executed": source.steps_executed,
                "maximum_reward": source.maximum_reward,
                "first_progress_index": source.reference.first_progress_index,
                "gripper_close_index": source.gripper_close_index,
            }
        )
        print(
            f"source episode={episode} seed={seed} reward=4 steps={source.steps_executed}",
            flush=True,
        )

    evaluation = _mapping(plan["evaluation"], "evaluation")
    exact = _mapping(evaluation["exact_control"], "exact_control")
    exact_report: dict[str, Any] | None = None
    tuning_reports: list[dict[str, Any]] = []
    development_reports: list[dict[str, Any]] = []
    if stage in {"exact", "full"}:
        print("evaluating exact control", flush=True)
        exact_report = _evaluate_target(
            seed=int(exact["simulator_seed"]),
            sources=sources,
            contract=contract,
            plan=plan,
            forced_source_episode=int(exact["episode"]),
        )
    if stage == "tuning":
        for seed in evaluation["tuning_simulator_seeds"]:
            print(f"evaluating tuning seed={seed}", flush=True)
            tuning_reports.append(
                _evaluate_target(
                    seed=int(seed), sources=sources, contract=contract, plan=plan
                )
            )
    if stage == "full" and exact_report is not None and exact_report["success"]:
        for seed in evaluation["development_simulator_seeds"]:
            print(f"evaluating development seed={seed}", flush=True)
            development_reports.append(
                _evaluate_target(
                    seed=int(seed), sources=sources, contract=contract, plan=plan
                )
            )
    acceptance = _mapping(plan["acceptance"], "acceptance")
    all_reports = [
        *([] if exact_report is None else [exact_report]),
        *tuning_reports,
        *development_reports,
    ]
    exact_successes = int(exact_report is not None and exact_report["success"])
    development_successes = sum(int(report["success"]) for report in development_reports)
    ood_failures = sum(int(report["oracle_out_of_distribution"]) for report in all_reports)
    ik_failures = sum(
        int(report["reference_ik_failures"]) + int(report["action_ik_failures"])
        for report in all_reports
    )
    criteria = {
        "exact_control_successes": exact_successes
        >= int(acceptance["exact_control_successes_required"]),
        "development_successes": development_successes
        >= int(acceptance["development_successes_required"]),
        "oracle_out_of_distribution_failures": ood_failures
        <= int(acceptance["maximum_oracle_out_of_distribution_failures"]),
        "inverse_kinematics_failures": ik_failures
        <= int(acceptance["maximum_ik_failures"]),
        "hidden_test_loaded": acceptance.get("hidden_test_loaded") is False,
        "recovery_label_write_disabled": acceptance.get(
            "recovery_labels_authorized_on_pass"
        )
        is False,
    }
    passed = stage == "full" and all(criteria.values())
    if stage == "exact":
        stage_passed = bool(
            exact_report is not None
            and exact_report["success"]
            and exact_report["reference_ik_failures"] == 0
            and exact_report["action_ik_failures"] == 0
            and not exact_report["oracle_out_of_distribution"]
        )
    elif stage == "tuning":
        stage_passed = bool(tuning_reports) and all(
            report["success"]
            and report["reference_ik_failures"] == 0
            and report["action_ik_failures"] == 0
            and not report["oracle_out_of_distribution"]
            for report in tuning_reports
        )
    else:
        stage_passed = passed
    if passed:
        status = "passed"
    elif stage_passed:
        status = "diagnostic_passed"
    else:
        status = "failed"
    report = {
        "schema_version": 1,
        "report_id": f"{plan['plan_id']}-{stable_hash(all_reports)[:16]}",
        "status": status,
        "diagnostic": "state_conditioned_recovery_oracle_evaluation",
        "stage": stage,
        "plan_id": plan["plan_id"],
        "plan_sha256": file_sha256(plan_path),
        "dataset_revision": dataset_manifest.resolved_revision,
        "dataset_manifest_sha256": file_sha256(dataset_root / "manifest.json"),
        "action_contract_sha256": file_sha256(action_contract_path),
        "source_replays": source_reports,
        "exact_control": exact_report,
        "tuning": tuning_reports,
        "development": development_reports,
        "acceptance": {
            "criteria": criteria,
            "exact_successes": exact_successes,
            "development_successes": development_successes,
            "oracle_out_of_distribution_failures": ood_failures,
            "inverse_kinematics_failures": ik_failures,
        },
        "provenance": {
            "label_type": "state_and_reward_conditioned_oracle_action",
            "state_conditioned": True,
            "time_indexed_reference": False,
            "recovery_labels_written": False,
            "hidden_test_loaded": False,
            "validation_episodes_loaded": False,
            "policy_gate4_seeds_executed": False,
            "collection_seeds_executed": False,
        },
        "code_identity": workspace_code_identity(REPOSITORY_ROOT),
    }
    json.dumps(report, allow_nan=False)
    output = _mapping(plan["output"], "output")
    if output.get("reports_are_scoped_by_plan_sha256_and_stage") is not True:
        raise ValueError("Recovery oracle reports must be scoped by plan identity and stage.")
    destination = (
        _run_root()
        / str(output["run_directory"])
        / file_sha256(plan_path)[:16]
        / f"{stage}.json"
    )
    create_json(destination, report)
    print(json.dumps({"status": report["status"], **report["acceptance"]}, indent=2))
    print(f"Report: {destination}")
    return 0 if stage_passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--stage", choices=("exact", "tuning", "full"), default="full")
    args = parser.parse_args()
    return _main(args.plan.resolve(), stage=args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
