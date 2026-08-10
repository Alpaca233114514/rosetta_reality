"""Run M2 Gym-ALOHA Action Contract gates before policy training or rollout."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from statistics import median
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
DEFAULT_CONTRACT = REPOSITORY_ROOT / "configs" / "sim" / "aloha_insertion.yaml"
DEFAULT_DATASET = REPOSITORY_ROOT / "configs" / "data" / "aloha_sim_insertion_m2.yaml"
DEFAULT_EXPERIMENT_ID = "m2-qwen08b-frozen-001"
TRACE_STATE_THRESHOLDS = (0.01, 0.025, 0.05, 0.1)
TRACE_MAXIMUM_ALIGNMENT_MAE = 0.005
TRACE_MAXIMUM_INITIAL_STATE_MAE = 0.05
TEACHER_DECOMPOSITION_STEPS = 3
TEACHER_STEP_ZERO_DETERMINISM_TOLERANCE = 1e-6
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rosetta_reality.data import ordered_feature_names, resolve_prepared_cache  # noqa: E402
from rosetta_reality.data.adapters.lerobot_v3 import LeRobotV3Adapter  # noqa: E402
from rosetta_reality.data.config import load_dataset_config  # noqa: E402
from rosetta_reality.data.normalization import (  # noqa: E402
    DatasetStatistics,
    denormalize,
    normalize,
)
from rosetta_reality.experiment import (  # noqa: E402
    file_sha256,
    load_experiment_config,
    stable_hash,
    workspace_code_identity,
)
from rosetta_reality.features import create_json  # noqa: E402
from rosetta_reality.models.backbones.qwen35 import Qwen35Backbone  # noqa: E402
from rosetta_reality.sim import GymAlohaEnvironment, load_action_contract  # noqa: E402
from rosetta_reality.train.m2 import build_policy_with_backbone  # noqa: E402


def _run_root() -> Path:
    value = os.environ.get("ROSETTA_RUN_ROOT")
    return Path(value) if value else REPOSITORY_ROOT / "runs"


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read a finite JSON object, rejecting permissive NaN/Infinity constants."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {_display_path(path)}.")
    json.dumps(value, allow_nan=False)
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _report_experiment_id(payload: dict[str, Any]) -> str:
    experiment_id = str(payload.get("experiment_id", DEFAULT_EXPERIMENT_ID))
    if (
        not experiment_id
        or experiment_id in {".", ".."}
        or any(character.isspace() or character in "/\\" for character in experiment_id)
    ):
        raise ValueError("Report experiment_id must be a path-safe token.")
    return experiment_id


def _write_report(gate: str, payload: dict[str, Any]) -> Path:
    digest = stable_hash(payload)[:12]
    experiment_id = _report_experiment_id(payload)
    path = _run_root() / experiment_id / "gates" / f"{gate}-{digest}.json"
    return create_json(path, payload)


def _write_diagnostic(name: str, payload: dict[str, Any]) -> Path:
    digest = stable_hash(payload)[:12]
    experiment_id = _report_experiment_id(payload)
    path = _run_root() / experiment_id / "diagnostics" / f"{name}-{digest}.json"
    return create_json(path, payload)


def _write_trajectory_report(payload: dict[str, Any]) -> Path:
    """Create one report per immutable trace identity and protocol."""

    json.dumps(payload, allow_nan=False)
    logical_identity = {
        "identity": payload["identity"],
        "protocol": payload["protocol"],
    }
    digest = stable_hash(logical_identity)[:16]
    experiment_id = _report_experiment_id(payload)
    phase = str(payload["protocol"]["phase"])
    if phase not in {"smoke", "full"}:
        raise ValueError("Trajectory report phase must be either 'smoke' or 'full'.")
    path = (
        _run_root()
        / experiment_id
        / "diagnostics"
        / f"trajectory-divergence-{phase}-{digest}.json"
    )
    return create_json(path, payload)


def _write_teacher_forced_report(payload: dict[str, Any]) -> Path:
    """Create one immutable report for the fixed five-episode decomposition."""

    json.dumps(payload, allow_nan=False)
    logical_identity = {
        "identity": payload["identity"],
        "protocol": payload["protocol"],
    }
    digest = stable_hash(logical_identity)[:16]
    experiment_id = _report_experiment_id(payload)
    path = (
        _run_root()
        / experiment_id
        / "diagnostics"
        / f"teacher-forced-decomposition-{digest}.json"
    )
    return create_json(path, payload)


def _write_recorded_domain_report(payload: dict[str, Any]) -> Path:
    """Create one immutable validation-only recorded-domain report."""

    json.dumps(payload, allow_nan=False)
    logical_identity = {
        "identity": payload["identity"],
        "protocol": payload["protocol"],
    }
    digest = stable_hash(logical_identity)[:16]
    experiment_id = _report_experiment_id(payload)
    path = (
        _run_root()
        / experiment_id
        / "diagnostics"
        / f"recorded-domain-probe-{digest}.json"
    )
    return create_json(path, payload)


def _cache_stride_matched_frames(
    frame_indices: tuple[int, ...],
    frame_stride: int,
) -> list[int]:
    """Return probe frames that are actual anchors for the configured cache stride."""

    if isinstance(frame_stride, bool) or not isinstance(frame_stride, int) or frame_stride <= 0:
        raise ValueError("Configured training frame stride must be a positive integer.")
    return [frame_index for frame_index in frame_indices if frame_index % frame_stride == 0]


def _write_domain_factorial_report(payload: dict[str, Any]) -> Path:
    """Create one immutable validation-only image/state factorial report."""

    json.dumps(payload, allow_nan=False)
    logical_identity = {
        "identity": payload["identity"],
        "protocol": payload["protocol"],
    }
    digest = stable_hash(logical_identity)[:16]
    experiment_id = _report_experiment_id(payload)
    path = (
        _run_root()
        / experiment_id
        / "diagnostics"
        / f"domain-factorial-probe-{digest}.json"
    )
    return create_json(path, payload)


def _display_path(path: Path) -> Path:
    if path.is_relative_to(REPOSITORY_ROOT):
        return path.relative_to(REPOSITORY_ROOT)
    return Path(path.name)


def _state(observation: Any, dimension: int) -> torch.Tensor:
    value = observation.get("robot_state")
    if not isinstance(value, torch.Tensor) or value.shape != (dimension,):
        raise ValueError("Simulator observation violates the 14-dimensional state contract.")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Simulator observation contains NaN or Inf.")
    return value


def _requested_torch_device() -> torch.device:
    value = os.environ.get("ROSETTA_TORCH_DEVICE", "cpu")
    if value not in {"cpu", "xpu"}:
        raise ValueError("ROSETTA_TORCH_DEVICE must be either cpu or xpu.")
    if value == "xpu" and not torch.xpu.is_available():
        raise RuntimeError("ROSETTA_TORCH_DEVICE=xpu but PyTorch XPU is unavailable.")
    return torch.device(value)


def scripted(
    contract_path: Path,
    *,
    seed: int,
    steps_per_dimension: int,
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
) -> int:
    """Gate 1: perturb each logical actuator without using a neural network."""

    contract = load_action_contract(contract_path)
    if steps_per_dimension <= 0:
        raise ValueError("steps_per_dimension must be positive.")
    environment_maximum_steps = steps_per_dimension + 1
    environment = GymAlohaEnvironment(
        contract,
        maximum_episode_steps=environment_maximum_steps,
    )
    results: list[dict[str, Any]] = []
    try:
        for index, dimension in enumerate(contract.dimensions):
            initial = _state(environment.reset(seed=seed), contract.dimension)
            target = initial.clone()
            room_positive = dimension.maximum - float(initial[index])
            room_negative = float(initial[index]) - dimension.minimum
            sign = 1.0 if room_positive >= room_negative else -1.0
            magnitude = min(0.02, max(room_positive, room_negative) * 0.05)
            target[index] += sign * magnitude
            target, _ = contract.clip(target)
            observation = None
            done = False
            for _ in range(steps_per_dimension):
                observation, _, done, _ = environment.step(target)
                if done:
                    break
            assert observation is not None
            final = _state(observation, contract.dimension)
            movement = final - initial
            directed = float(movement[index]) * sign
            mirror_index = index + 7 if index < 7 else index - 7
            mirror_movement = abs(float(movement[mirror_index]))
            targeted_movement = abs(float(movement[index]))
            passed = (
                not done
                and directed > 1e-6
                and targeted_movement > mirror_movement
                and bool((final >= contract.lower_bounds - 1e-5).all())
                and bool((final <= contract.upper_bounds + 1e-5).all())
            )
            results.append(
                {
                    "index": index,
                    "name": dimension.name,
                    "requested_delta": sign * magnitude,
                    "observed_delta": float(movement[index]),
                    "mirror_delta_abs": mirror_movement,
                    "passed": passed,
                }
            )

        invalid_rejected = False
        invalid = torch.zeros(contract.dimension)
        invalid[0] = torch.nan
        try:
            environment.step(invalid)
        except ValueError:
            invalid_rejected = True
        out_of_range = contract.upper_bounds + 1.0
        _, clip_mask = contract.clip(out_of_range)
        all_passed = all(result["passed"] for result in results)
        passed = all_passed and invalid_rejected and bool(clip_mask.all())
        report = {
            "schema_version": 2,
            "gate": "m2_gate_1_scripted_action",
            "experiment_id": experiment_id,
            "status": "passed" if passed else "failed",
            "seed": seed,
            "steps_per_dimension": steps_per_dimension,
            "environment_maximum_steps": environment_maximum_steps,
            "action_contract_sha256": file_sha256(contract_path),
            "invalid_action_rejected": invalid_rejected,
            "out_of_range_fields_clipped": int(clip_mask.sum()),
            "dimensions": results,
        }
        path = _write_report("gate1", report)
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"Report: {_display_path(path)}")
        return 0 if passed else 1
    finally:
        environment.close()


def _dataset_rows(root: Path, episode: int, fields: Any) -> list[dict[str, Any]]:
    import pyarrow.dataset as arrow_dataset

    dataset = arrow_dataset.dataset(root / "data", format="parquet")
    table = dataset.to_table(
        columns=[fields.action, fields.state, fields.timestamp, fields.frame_index],
        filter=arrow_dataset.field(fields.episode_index) == episode,
    )
    rows = table.to_pylist()
    return sorted(rows, key=lambda row: int(row[fields.frame_index]))


def _validated_initial_alignment(
    path: Path,
    *,
    episode: int,
    seed: int,
    dataset_revision: str,
    dataset_manifest_sha256: str,
    maximum_mae: float,
) -> dict[str, Any]:
    if maximum_mae < 0:
        raise ValueError("Maximum initial-image alignment MAE must be non-negative.")
    report = _read_json_object(path)
    if report.get("gate") == "m2_gate_2_dataset_action_replay":
        if report.get("schema_version") != 2 or report.get("status") != "passed":
            raise ValueError("Gate 2 alignment source must be a passed schema-v2 report.")
        if report.get("dataset_revision") != dataset_revision:
            raise ValueError("Gate 2 alignment source dataset revision differs.")
        if report.get("dataset_manifest_sha256") != dataset_manifest_sha256:
            raise ValueError("Gate 2 alignment source dataset manifest differs.")
        if report.get("episode") != episode or report.get("seed") != seed:
            raise ValueError("Gate 2 alignment source episode or seed differs.")
        criteria = report.get("acceptance_criteria", {})
        alignment = report.get("initial_object_pose_alignment", {})
        if (
            criteria.get("initial_object_pose_image_alignment") is not True
            or alignment.get("within_tolerance") is not True
        ):
            raise ValueError("Gate 2 alignment source did not pass image alignment.")
        pooled_mae = float(alignment.get("pooled_4x4_mae", math.inf))
        if not math.isfinite(pooled_mae):
            raise ValueError("Gate 2 alignment source MAE is not finite.")
        return {
            "source_type": "passed_gate2_replay",
            "report_sha256": file_sha256(path),
            "selected_seed": seed,
            "pooled_4x4_mae": pooled_mae,
            "maximum_pooled_4x4_mae": maximum_mae,
            "within_tolerance": pooled_mae <= maximum_mae,
        }
    if report.get("status") != "complete":
        raise ValueError("Initial-image alignment report is incomplete.")
    dataset = report.get("dataset", {})
    if dataset.get("revision") != dataset_revision:
        raise ValueError("Initial-image alignment dataset revision differs.")
    if dataset.get("manifest_sha256") != dataset_manifest_sha256:
        raise ValueError("Initial-image alignment dataset manifest differs.")
    matches = [
        item
        for item in report.get("episodes", [])
        if int(item.get("episode", -1)) == episode
    ]
    if len(matches) != 1:
        raise ValueError("Initial-image alignment report must contain the selected episode once.")
    selected = matches[0]
    if int(selected.get("selected_seed", -1)) != seed:
        raise ValueError("Initial-image alignment seed differs from Gate 2.")
    alignment = selected.get("selected_alignment", {})
    pooled_mae = float(alignment.get("pooled_4x4_mae", math.inf))
    if not math.isfinite(pooled_mae):
        raise ValueError("Initial-image alignment MAE is not finite.")
    return {
        "report_sha256": file_sha256(path),
        "selected_seed": seed,
        "pixel_mae": float(alignment.get("pixel_mae", math.inf)),
        "pixel_rmse": float(alignment.get("pixel_rmse", math.inf)),
        "pooled_4x4_mae": pooled_mae,
        "maximum_pooled_4x4_mae": maximum_mae,
        "within_tolerance": pooled_mae <= maximum_mae,
    }


def _require_validation_episode(experiment: dict[str, Any], episode: int) -> None:
    """Reject train, hidden-test, and undeclared episodes before dataset access."""

    raw_split = experiment.get("dataset", {}).get("split", {})
    validation = {int(value) for value in raw_split.get("validation", [])}
    hidden_test = {int(value) for value in raw_split.get("test", [])}
    if episode in hidden_test:
        raise ValueError("Trajectory divergence must not open a hidden-test episode.")
    if episode not in validation:
        raise ValueError("Trajectory divergence is restricted to validation episodes.")


def _trace_alignment(
    path: Path,
    *,
    episode: int,
    validation_episodes: set[int],
    dataset_revision: str,
    dataset_manifest_sha256: str,
    action_contract_sha256: str,
    frequency_hz: float,
    experiment_id: str,
    experiment_config_sha256: str,
    maximum_mae: float,
) -> dict[str, Any]:
    """Bind one validation episode to a finite image-selected simulator seed."""

    report = _read_json_object(path)
    if report.get("schema_version") != 1 or report.get("status") != "complete":
        raise ValueError("Trajectory alignment report schema or status is invalid.")
    if report.get("diagnostic") != "m2_expert_replay_with_image_aligned_initial_seed":
        raise ValueError("Trajectory divergence requires an expert-replay alignment report.")
    if report.get("action_mode") != "contract_clipped":
        raise ValueError("Trajectory divergence alignment must use contract-clipped replay.")
    if report.get("action_contract_sha256") != action_contract_sha256:
        raise ValueError("Trajectory alignment Action Contract differs from the trace.")
    if float(report.get("frequency_hz", math.nan)) != frequency_hz:
        raise ValueError("Trajectory alignment control frequency differs from the trace.")
    validation_scope = report.get("validation_scope", {})
    scoped_episode_values = validation_scope.get("episodes", [])
    if (
        validation_scope.get("experiment_id") != experiment_id
        or validation_scope.get("experiment_config_sha256")
        != experiment_config_sha256
        or validation_scope.get("split") != "validation"
        or validation_scope.get("test_split_opened") is not False
        or not isinstance(scoped_episode_values, list)
        or any(type(value) is not int for value in scoped_episode_values)
        or len(scoped_episode_values) != len(validation_episodes)
        or set(scoped_episode_values) != validation_episodes
    ):
        raise ValueError("Trajectory alignment is not bound to this validation split.")
    image_identity = report.get("initial_image_artifact", {})
    if (
        image_identity.get("dataset_revision") != dataset_revision
        or image_identity.get("dataset_manifest_sha256") != dataset_manifest_sha256
        or image_identity.get("validation_scope") != validation_scope
        or not _is_sha256(image_identity.get("identity_hash"))
        or not _is_sha256(image_identity.get("manifest_sha256"))
    ):
        raise ValueError("Trajectory initial-image dataset identity differs.")
    image_episode_values = image_identity.get("episodes", [])
    if not isinstance(image_episode_values, list) or any(
        type(value) is not int for value in image_episode_values
    ):
        raise ValueError("Trajectory initial-image episodes must be explicit integers.")
    report_episode_items = report.get("episodes", [])
    if not isinstance(report_episode_items, list) or any(
        not isinstance(item, dict)
        or type(item.get("episode")) is not int
        or type(item.get("selected_seed")) is not int
        or item["selected_seed"] < 0
        for item in report_episode_items
    ):
        raise ValueError("Trajectory alignment episodes or seeds are malformed.")
    image_episodes = {
        int(value)
        for value in image_episode_values
    }
    report_episode_values = [item["episode"] for item in report_episode_items]
    report_episodes = set(report_episode_values)
    if (
        len(image_episode_values) != len(image_episodes)
        or image_episodes != validation_episodes
        or len(report_episode_values) != len(report_episodes)
        or report_episodes != validation_episodes
        or episode not in report_episodes
    ):
        raise ValueError(
            "Trajectory alignment must cover the exact validation split without duplicates."
        )
    matches = [
        item
        for item in report_episode_items
        if item["episode"] == episode
    ]
    if len(matches) != 1:
        raise ValueError("Alignment report must contain the validation episode exactly once.")
    seed = int(matches[0].get("selected_seed", -1))
    alignment = _validated_initial_alignment(
        path,
        episode=episode,
        seed=seed,
        dataset_revision=dataset_revision,
        dataset_manifest_sha256=dataset_manifest_sha256,
        maximum_mae=maximum_mae,
    )
    finite_metrics = (
        alignment["pixel_mae"],
        alignment["pixel_rmse"],
        alignment["pooled_4x4_mae"],
    )
    if not all(math.isfinite(float(value)) for value in finite_metrics):
        raise ValueError("Trajectory alignment metrics must be finite.")
    if not alignment["within_tolerance"]:
        raise ValueError("Trajectory alignment exceeds the fixed image-MAE tolerance.")
    return alignment


def _first_crossings(
    values: list[float],
    thresholds: tuple[float, ...] = TRACE_STATE_THRESHOLDS,
) -> dict[str, int | None]:
    """Return the first zero-based step whose metric reaches each fixed threshold."""

    if (
        not thresholds
        or any(value <= 0 for value in thresholds)
        or tuple(sorted(set(thresholds))) != thresholds
    ):
        raise ValueError("Trace thresholds must be positive, unique, and increasing.")
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("Trace crossing metrics must be finite and non-negative.")
    return {
        str(threshold): next(
            (index for index, value in enumerate(values) if value >= threshold),
            None,
        )
        for threshold in thresholds
    }


def _contact_metrics(environment: GymAlohaEnvironment) -> dict[str, Any]:
    """Return canonical current contacts without hiding repeated contact points."""

    pairs = [tuple(sorted(pair)) for pair in environment.contact_pairs()]
    pairs.sort()
    unexpected = [
        pair
        for pair in pairs
        if environment.is_unexpected_collision_pair(pair[0], pair[1])
    ]
    return {
        "pairs": [" <-> ".join(pair) for pair in pairs],
        "count": len(pairs),
        "unexpected_pairs": [" <-> ".join(pair) for pair in unexpected],
        "unexpected_count": len(unexpected),
    }


def _trace_prefix_digest(steps: list[dict[str, Any]]) -> str:
    """Hash the first three deterministic state/action/reward comparisons."""

    def stream(value: dict[str, Any]) -> dict[str, Any]:
        return {
            name: value[name]
            for name in (
                "pre_state",
                "raw_action",
                "clipped_action",
                "raw_clip_mask",
                "post_state",
                "reward",
                "joint_limits",
                "contacts",
            )
        }

    canonical = []
    for item in steps[:3]:
        divergence = item["divergence"]
        canonical.append(
            {
                "step_index": item["step_index"],
                "dataset_frame_index": item["dataset_frame_index"],
                "dataset_timestamp": item["dataset_timestamp"],
                "expert": stream(item["expert"]),
                "policy": stream(item["policy"]),
                "divergence": {
                    name: divergence[name]
                    for name in (
                        "post_state_mae",
                        "post_state_l2",
                        "post_state_maximum_absolute_difference",
                        "clipped_action_mae",
                        "clipped_action_l2",
                        "reward_delta_policy_minus_expert",
                        "reward_diverged",
                    )
                },
            }
        )
    return stable_hash(canonical)


def _validated_trace_smoke_report(
    path: Path,
    *,
    identity: dict[str, Any],
) -> dict[str, Any]:
    """Require a matching, completed three-step trace before a full trace."""

    report = _read_json_object(path)
    protocol = report.get("protocol", {})
    summary = report.get("summary", {})
    reset = report.get("reset", {})
    steps = report.get("steps", [])
    if (
        report.get("status") != "passed"
        or report.get("diagnostic") != "m2_validation_trajectory_divergence"
        or report.get("identity") != identity
        or protocol.get("phase") != "smoke"
        or protocol.get("steps_requested") != 3
        or summary.get("steps_executed") != 3
        or summary.get("end_reason") != "requested_steps_completed"
        or reset.get("cross_environment_aligned") is not True
        or report.get("test_split_opened") is not False
        or not isinstance(steps, list)
        or len(steps) != 3
    ):
        raise ValueError("Full trajectory requires a matching passed three-step smoke report.")
    prefix = summary.get("canonical_first_three_steps_sha256")
    if (
        not _is_sha256(prefix)
        or _trace_prefix_digest(steps) != prefix
    ):
        raise ValueError("Trace smoke report has no canonical three-step digest.")
    return {
        "path": path.name,
        "sha256": file_sha256(path),
        "canonical_first_three_steps_sha256": prefix,
    }


def replay(
    contract_path: Path,
    dataset_path: Path,
    *,
    episode: int,
    maximum_steps: int,
    seed: int,
    initial_alignment_report: Path,
    maximum_alignment_mae: float,
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
) -> int:
    """Gate 2: replay pinned expert actions through the contract and simulator."""

    contract = load_action_contract(contract_path)
    dataset_config = load_dataset_config(dataset_path)
    root, manifest = resolve_prepared_cache(dataset_config, REPOSITORY_ROOT)
    cleaning = json.loads((root / "cleaning_report.json").read_text(encoding="utf-8"))
    if cleaning.get("status") != "validated_clean":
        raise ValueError("Dataset replay requires a validated-clean cache.")
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    dataset_manifest_sha256 = file_sha256(root / "manifest.json")
    initial_alignment = _validated_initial_alignment(
        initial_alignment_report,
        episode=episode,
        seed=seed,
        dataset_revision=manifest.resolved_revision,
        dataset_manifest_sha256=dataset_manifest_sha256,
        maximum_mae=maximum_alignment_mae,
    )
    contract.validate_order(ordered_feature_names(root, dataset_config.fields.action))
    if float(info["fps"]) != contract.frequency_hz:
        raise ValueError("Dataset and simulator control frequencies differ.")
    rows = _dataset_rows(root, episode, dataset_config.fields)[:maximum_steps]
    if len(rows) < 2:
        raise ValueError("Dataset replay selection contains fewer than two frames.")
    timestamps = [float(row[dataset_config.fields.timestamp]) for row in rows]
    maximum_timing_error = max(
        abs((current - previous) - 1.0 / contract.frequency_hz)
        for previous, current in zip(timestamps, timestamps[1:])
    )

    environment = GymAlohaEnvironment(contract, maximum_episode_steps=maximum_steps)
    clipped_elements = 0
    clipped_by_dimension = torch.zeros(contract.dimension, dtype=torch.long)
    maximum_source_overshoot = torch.zeros(contract.dimension)
    direction_matches = 0
    direction_trials = 0
    tracking_errors: list[float] = []
    done_early = False
    try:
        previous = _state(environment.reset(seed=seed), contract.dimension)
        dataset_initial = torch.as_tensor(
            rows[0][dataset_config.fields.state], dtype=torch.float32
        )
        initial_state_mae = float((previous - dataset_initial).abs().mean())
        rewards: list[float] = []
        task_success = False
        terminated = False
        truncated = False
        for row_index, row in enumerate(rows):
            action = torch.as_tensor(row[dataset_config.fields.action], dtype=torch.float32)
            contract.validate_tensor(action, allow_chunk=False)
            target, clip_mask = contract.clip(action)
            clipped_elements += int(clip_mask.sum())
            clipped_by_dimension += clip_mask.to(dtype=torch.long)
            maximum_source_overshoot = torch.maximum(
                maximum_source_overshoot,
                (action - target).abs(),
            )
            observation, reward, done, step_info = environment.step(action)
            rewards.append(reward)
            task_success = task_success or bool(step_info.get("is_success", False))
            terminated = terminated or bool(step_info.get("terminated", False))
            truncated = truncated or bool(step_info.get("truncated", False))
            current = _state(observation, contract.dimension)
            requested_delta = target - previous
            observed_delta = current - previous
            active = requested_delta.abs() > 1e-3
            direction_matches += int(
                ((requested_delta[active] * observed_delta[active]) > 0).sum()
            )
            direction_trials += int(active.sum())
            tracking_errors.append(float((current - target).abs().mean()))
            previous = current
            if done:
                done_early = row_index + 1 < len(rows)
                break
        clipping_rate = clipped_elements / (len(tracking_errors) * contract.dimension)
        direction_agreement = direction_matches / max(1, direction_trials)
        source_tolerances = contract.source_overshoot_tolerances
        criteria = {
            "completed_requested_steps_or_task_success": not done_early or task_success,
            "task_success": task_success,
            "not_truncated_before_task_success": not truncated or task_success,
            "initial_object_pose_image_alignment": initial_alignment["within_tolerance"],
            "timestamp_alignment": maximum_timing_error <= 1e-4,
            "source_clipping_within_contract_tolerance": bool(
                (maximum_source_overshoot <= source_tolerances + 1e-6).all()
            ),
            "direction_agreement": direction_agreement >= 0.70,
            "initial_state_alignment": initial_state_mae <= 0.05,
            "mean_target_tracking": (
                sum(tracking_errors) / len(tracking_errors) <= 0.05
            ),
            "maximum_target_tracking": max(tracking_errors) <= 0.10,
            "finite_tracking": all(math.isfinite(value) for value in tracking_errors),
        }
        passed = all(criteria.values())
        dimension_clipping = {
            name: {
                "clipped": int(clipped_by_dimension[index]),
                "maximum_source_overshoot": float(maximum_source_overshoot[index]),
                "allowed_source_overshoot": float(source_tolerances[index]),
            }
            for index, name in enumerate(contract.dimension_names)
        }
        if task_success:
            end_reason = "task_success"
        elif truncated:
            end_reason = "time_limit_truncation"
        elif terminated:
            end_reason = "environment_termination"
        else:
            end_reason = "requested_steps_completed"
        report = {
            "schema_version": 2,
            "gate": "m2_gate_2_dataset_action_replay",
            "experiment_id": experiment_id,
            "status": "passed" if passed else "failed",
            "episode": episode,
            "steps": len(tracking_errors),
            "seed": seed,
            "dataset_revision": manifest.resolved_revision,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "cleaning_report_sha256": file_sha256(root / "cleaning_report.json"),
            "action_contract_sha256": file_sha256(contract_path),
            "frequency_hz": contract.frequency_hz,
            "maximum_timestamp_step_error": maximum_timing_error,
            "initial_state_mae": initial_state_mae,
            "mean_target_tracking_mae": sum(tracking_errors) / len(tracking_errors),
            "maximum_target_tracking_mae": max(tracking_errors),
            "direction_agreement": direction_agreement,
            "clipped_elements": clipped_elements,
            "clipping_rate": clipping_rate,
            "dimension_clipping": dimension_clipping,
            "acceptance_criteria": criteria,
            "ended_before_requested_steps": done_early,
            "end_reason": end_reason,
            "task_success": task_success,
            "maximum_reward": max(rewards, default=0.0),
            "terminated": terminated,
            "truncated": truncated,
            "initial_object_pose_alignment": initial_alignment,
        }
        path = _write_report("gate2", report)
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"Report: {_display_path(path)}")
        return 0 if passed else 1
    finally:
        environment.close()


def _load_online_artifact(
    config_path: Path,
    artifact_root: Path,
) -> tuple[Any, DatasetStatistics, Any, dict[str, Any], str]:
    experiment = load_experiment_config(config_path, REPOSITORY_ROOT)
    artifact_manifest_path = artifact_root / "manifest.json"
    artifact_manifest = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
    if artifact_manifest.get("status") != "verified":
        raise ValueError("Online rollout requires a verified exported artifact.")
    for name, expected in artifact_manifest["files"].items():
        if file_sha256(artifact_root / name) != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}.")
    artifact_config = json.loads((artifact_root / "config.json").read_text(encoding="utf-8"))
    if artifact_manifest.get("experiment_id") != experiment["experiment_id"]:
        raise ValueError("Artifact manifest experiment differs from the rollout config.")
    if artifact_config.get("experiment_id") != experiment["experiment_id"]:
        raise ValueError("Artifact config experiment differs from the rollout config.")
    artifact_contract = _read_json_object(artifact_root / "action_contract.json")
    contract = load_action_contract(REPOSITORY_ROOT / experiment["action_contract"])
    canonical_contract = json.loads(json.dumps(asdict(contract), allow_nan=False))
    if artifact_contract != canonical_contract:
        raise ValueError("Artifact Action Contract differs from the rollout config.")
    normalization = json.loads((artifact_root / "normalization.json").read_text(encoding="utf-8"))
    if normalization.get("source_split") != "train":
        raise ValueError("Online artifact normalization did not originate from train only.")
    statistics = DatasetStatistics.from_dict(normalization["statistics"])
    model_environment = experiment["backbone"]["local_root_environment"]
    model_value = os.environ.get(model_environment)
    if not model_value:
        raise ValueError(f"{model_environment} is required for online rollout.")
    model_root = Path(model_value)
    for name, expected in artifact_config["base_model_file_hashes"].items():
        path = model_root / name
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"Local base-model identity mismatch: {name}.")
    configured = experiment["backbone"]
    dtype = getattr(torch, str(configured["dtype"]), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported online backbone dtype: {configured['dtype']!r}.")
    payload = torch.load(artifact_root / "model.pt", map_location="cpu", weights_only=True)
    model_contract = payload["model_contract"]
    if int(model_contract["chunk_size"]) != contract.chunk_length:
        raise ValueError("Artifact chunk size differs from the Action Contract.")
    configured_parameterization = experiment["action_expert"].get(
        "prediction_parameterization", "absolute"
    )
    if model_contract.get("prediction_parameterization", "absolute") != (
        configured_parameterization
    ):
        raise ValueError("Artifact action parameterization differs from the experiment config.")
    local_model_config = json.loads((model_root / "config.json").read_text(encoding="utf-8"))
    text_config = local_model_config.get("text_config", local_model_config)
    model_hidden_size = int(text_config["hidden_size"])
    device = _requested_torch_device()
    backbone = Qwen35Backbone(
        str(model_root),
        hidden_size=model_hidden_size,
        device=device,
        dtype=dtype,
        local_files_only=True,
        freeze=True,
        pooling=str(configured["pooling"]),
        prompt_template=str(configured["processor"]["prompt"]),
        prompt_mode=str(configured["processor"].get("prompt_mode", "auto")),
        model_kwargs={"low_cpu_mem_usage": True},
        processor_kwargs={
            "min_pixels": int(configured["processor"]["min_pixels"]),
            "max_pixels": int(configured["processor"]["max_pixels"]),
        },
    )
    if backbone.hidden_size != int(model_contract["feature_dim"]):
        raise ValueError("Online backbone pooling dimension differs from the artifact contract.")
    policy = build_policy_with_backbone(
        experiment,
        backbone,
        state_dim=int(model_contract["state_dim"]),
        action_dim=int(model_contract["action_dim"]),
        chunk_size=int(model_contract["chunk_size"]),
        statistics=statistics,
    )
    policy.load_state_dict(payload["model_state"], strict=True)
    policy.to(device)
    policy.eval()
    dataset_config = load_dataset_config(REPOSITORY_ROOT / experiment["dataset"]["config"])
    instruction = dataset_config.expected_instruction
    if not instruction:
        raise ValueError("Online rollout requires an explicit configured instruction.")
    return policy, statistics, contract, artifact_manifest, instruction


def _percentile_95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def _rollout_episode(
    policy: Any,
    statistics: DatasetStatistics,
    contract: Any,
    instruction: str,
    *,
    seed: int,
    maximum_steps: int,
) -> dict[str, Any]:
    environment = GymAlohaEnvironment(contract, maximum_episode_steps=maximum_steps)
    inference_latencies: list[float] = []
    simulation_latencies: list[float] = []
    executed_actions: list[torch.Tensor] = []
    raw_limit_violations = 0
    executed_limit_violations = 0
    invalid_actions = 0
    joint_limit_violations = 0
    unexpected_collisions = 0
    unexpected_collision_pairs: Counter[str] = Counter()
    contact_events = 0
    rewards: list[float] = []
    reward_histogram: Counter[str] = Counter()
    reward_first_steps: dict[str, int] = {}
    success = False
    terminated = False
    truncated = False
    try:
        device = _requested_torch_device()
        observation = environment.reset(seed=seed)
        _state(observation, contract.dimension)
        reset_pairs = environment.contact_pairs()
        reset_unexpected_pairs = [
            pair
            for pair in reset_pairs
            if environment.is_unexpected_collision_pair(*pair)
        ]
        while len(executed_actions) < maximum_steps:
            state = _state(observation, contract.dimension)
            started = time.perf_counter()
            with torch.inference_mode():
                normalized_chunk = policy(
                    {"images": observation["images"], "instruction": instruction},
                    normalize(state.unsqueeze(0), statistics.state).to(device),
                )
                raw_chunk = denormalize(normalized_chunk, statistics.action)[
                    0, : contract.chunk_execution_steps
                ].cpu()
            inference_latencies.append(time.perf_counter() - started)
            if not bool(torch.isfinite(raw_chunk).all()):
                invalid_actions += max(
                    1,
                    int((~torch.isfinite(raw_chunk)).any(dim=-1).sum()),
                )
                break
            done = False
            for raw_action in raw_chunk:
                clipped, mask = contract.clip(raw_action)
                raw_limit_violations += int(mask.sum())
                started = time.perf_counter()
                observation, reward, done, info = environment.step(clipped)
                simulation_latencies.append(time.perf_counter() - started)
                step_index = len(executed_actions)
                executed_actions.append(clipped)
                rewards.append(reward)
                reward_key = f"{reward:g}"
                reward_histogram[reward_key] += 1
                reward_first_steps.setdefault(reward_key, step_index)
                next_state = _state(observation, contract.dimension)
                executed_limit_violations += int(
                    (
                        (clipped < contract.lower_bounds)
                        | (clipped > contract.upper_bounds)
                    ).sum()
                )
                joint_limit_violations += int(
                    (
                        (next_state < contract.lower_bounds - 1e-5)
                        | (next_state > contract.upper_bounds + 1e-5)
                    ).sum()
                )
                pairs = environment.contact_pairs()
                contact_events += len(pairs)
                for pair in pairs:
                    if environment.is_unexpected_collision_pair(*pair):
                        unexpected_collisions += 1
                        key = " <-> ".join(sorted(pair))
                        unexpected_collision_pairs[key] += 1
                success = success or bool(info.get("is_success", False))
                terminated = terminated or bool(info.get("terminated", False))
                truncated = truncated or bool(info.get("truncated", False))
                if done or len(executed_actions) >= maximum_steps:
                    break
            if done:
                break
        smoothness = 0.0
        if len(executed_actions) > 1:
            stacked = torch.stack(executed_actions)
            smoothness = float((stacked[1:] - stacked[:-1]).square().sum(dim=-1).sqrt().mean())
        action_elements = max(1, len(executed_actions) * contract.dimension)
        if success:
            end_reason = "task_success"
        elif truncated:
            end_reason = "time_limit_truncation"
        elif terminated:
            end_reason = "environment_termination"
        elif invalid_actions:
            end_reason = "invalid_action"
        else:
            end_reason = "requested_steps_completed"
        return {
            "seed": seed,
            "policy_device": str(device),
            "success": success,
            "terminated": terminated,
            "truncated": truncated,
            "rollout_length": len(executed_actions),
            "maximum_reward": max(rewards, default=0.0),
            "reward_histogram": dict(sorted(reward_histogram.items())),
            "reward_first_steps": dict(sorted(reward_first_steps.items())),
            "end_reason": end_reason,
            "chunk_execution": contract.chunk_execution,
            "chunk_execution_steps": contract.chunk_execution_steps,
            "policy_inference_calls": len(inference_latencies),
            "invalid_action_rate": invalid_actions
            / max(1, len(executed_actions) + invalid_actions),
            "raw_limit_violation_rate": raw_limit_violations / action_elements,
            "executed_limit_violation_rate": executed_limit_violations / action_elements,
            "joint_limit_violations": joint_limit_violations,
            "unexpected_collisions": unexpected_collisions,
            "unexpected_collision_pairs": dict(sorted(unexpected_collision_pairs.items())),
            "contact_events": contact_events,
            "reset_contact_events": len(reset_pairs),
            "reset_unexpected_collisions": len(reset_unexpected_pairs),
            "action_smoothness_l2": smoothness,
            "mean_policy_inference_seconds": (
                sum(inference_latencies) / len(inference_latencies)
                if inference_latencies
                else 0.0
            ),
            "p95_policy_inference_seconds": _percentile_95(inference_latencies),
            "amortized_policy_inference_seconds_per_step": (
                sum(inference_latencies) / max(1, len(executed_actions))
            ),
            "mean_simulation_step_seconds": (
                sum(simulation_latencies) / len(simulation_latencies)
                if simulation_latencies
                else 0.0
            ),
            "p95_simulation_step_seconds": _percentile_95(simulation_latencies),
        }
    finally:
        environment.close()


def _joint_limit_metrics(state: torch.Tensor, contract: Any) -> dict[str, Any]:
    """Describe physical-state fields outside the declared joint limits."""

    mask = (state < contract.lower_bounds - 1e-5) | (
        state > contract.upper_bounds + 1e-5
    )
    return {
        "count": int(mask.sum()),
        "fields": [
            name
            for name, violated in zip(contract.dimension_names, mask.tolist())
            if violated
        ],
    }


def _reset_comparison(
    expert_observation: dict[str, Any],
    policy_observation: dict[str, Any],
    contract: Any,
) -> dict[str, Any]:
    """Require two independently reset environments to start identically."""

    expert_state = _state(expert_observation, contract.dimension)
    policy_state = _state(policy_observation, contract.dimension)
    state_difference = expert_state - policy_state
    expert_images = expert_observation.get("images")
    policy_images = policy_observation.get("images")
    if not isinstance(expert_images, dict) or not isinstance(policy_images, dict):
        raise ValueError("Trajectory reset requires image mappings from both environments.")
    if set(expert_images) != set(policy_images) or not expert_images:
        raise ValueError("Trajectory reset camera mappings differ or are empty.")
    cameras: dict[str, Any] = {}
    for name in sorted(expert_images):
        expert_image = expert_images[name]
        policy_image = policy_images[name]
        if (
            not isinstance(expert_image, torch.Tensor)
            or not isinstance(policy_image, torch.Tensor)
            or expert_image.shape != policy_image.shape
        ):
            raise ValueError(f"Trajectory reset camera {name!r} has incompatible tensors.")
        if not bool(torch.isfinite(expert_image).all() and torch.isfinite(policy_image).all()):
            raise ValueError(f"Trajectory reset camera {name!r} contains NaN or Inf.")
        difference = (expert_image - policy_image).abs()
        cameras[name] = {
            "shape": list(expert_image.shape),
            "mae": float(difference.mean()),
            "maximum_absolute_difference": float(difference.max()),
        }
    state_mae = float(state_difference.abs().mean())
    state_maximum = float(state_difference.abs().max())
    aligned = state_maximum <= 1e-7 and all(
        value["maximum_absolute_difference"] <= 1e-7 for value in cameras.values()
    )
    if not aligned:
        raise ValueError("Independent expert/policy simulator resets are not aligned.")
    return {
        "cross_environment_aligned": True,
        "state_mae": state_mae,
        "state_maximum_absolute_difference": state_maximum,
        "cameras": cameras,
    }


def _environment_backend_objects(environment: Any) -> tuple[Any, ...]:
    """Expose backend identities so paired traces cannot share simulator state."""

    raw = getattr(environment, "raw_environment", None)
    if raw is None:
        raise RuntimeError("Trajectory environment does not expose its raw backend.")
    unwrapped = getattr(raw, "unwrapped", raw)
    control_environment = getattr(unwrapped, "_env", None)
    physics = getattr(control_environment, "physics", None)
    return tuple(
        value
        for value in (raw, unwrapped, control_environment, physics)
        if value is not None
    )


def _validated_trace_artifact(
    config_path: Path,
    artifact_root: Path,
    experiment: dict[str, Any],
    contract: Any,
    *,
    dataset_revision: str,
    dataset_manifest_sha256: str,
) -> dict[str, Any]:
    """Cross-bind the exported artifact, feature cache, data, config, and contract."""

    manifest_path = artifact_root / "manifest.json"
    manifest = _read_json_object(manifest_path)
    artifact_id = manifest.get("artifact_id")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "verified"
        or manifest.get("reload", {}).get("verified") is not True
        or not isinstance(artifact_id, str)
        or not artifact_id
        or artifact_id != artifact_root.name
        or manifest.get("experiment_id") != experiment["experiment_id"]
    ):
        raise ValueError("Trajectory artifact manifest identity or verification is invalid.")
    files = manifest.get("files", {})
    if not isinstance(files, dict) or not {
        "config.json",
        "action_contract.json",
        "model.pt",
        "normalization.json",
    }.issubset(files):
        raise ValueError("Trajectory artifact manifest is missing required files.")
    for name, expected in files.items():
        if not isinstance(name, str) or file_sha256(artifact_root / name) != expected:
            raise ValueError(f"Trajectory artifact checksum mismatch: {name}.")

    artifact_config = _read_json_object(artifact_root / "config.json")
    backbone = experiment["backbone"]
    expected_config = {
        "experiment_id": experiment["experiment_id"],
        "adaptation": backbone["adaptation"],
        "base_model": backbone["identifier"],
        "feature_layer": backbone["feature_layer"],
        "pooling": backbone["pooling"],
        "processor": backbone["processor"],
        "action_expert": experiment["action_expert"],
    }
    for name, expected in expected_config.items():
        if artifact_config.get(name) != expected:
            raise ValueError(f"Trajectory artifact config differs at {name}.")
    model_contract = artifact_config.get("model_contract", {})
    if (
        model_contract.get("state_dim") != contract.dimension
        or model_contract.get("action_dim") != contract.dimension
        or model_contract.get("chunk_size") != contract.chunk_length
        or model_contract.get("prediction_parameterization", "absolute")
        != experiment["action_expert"].get("prediction_parameterization", "absolute")
    ):
        raise ValueError("Trajectory artifact model contract differs from the experiment.")
    canonical_contract = json.loads(json.dumps(asdict(contract), allow_nan=False))
    if _read_json_object(artifact_root / "action_contract.json") != canonical_contract:
        raise ValueError("Trajectory artifact Action Contract differs from the experiment.")

    feature_identity_hash = artifact_config.get("feature_cache_identity")
    if not _is_sha256(feature_identity_hash):
        raise ValueError("Trajectory artifact has no full feature-cache identity.")
    feature_root = Path(
        os.environ.get("ROSETTA_FEATURE_ROOT", REPOSITORY_ROOT / "feature_cache")
    )
    feature_manifest_path = (
        feature_root
        / experiment["experiment_id"]
        / feature_identity_hash[:16]
        / "manifest.json"
    )
    feature_manifest = _read_json_object(feature_manifest_path)
    feature_identity = feature_manifest.get("identity", {})
    if (
        file_sha256(feature_manifest_path) != manifest.get("feature_manifest_sha256")
        or feature_manifest.get("status") != "complete"
        or feature_manifest.get("identity_hash") != feature_identity_hash
        or feature_identity.get("experiment_id") != experiment["experiment_id"]
        or feature_identity.get("experiment_config_sha256") != file_sha256(config_path)
        or feature_identity.get("action_contract_sha256")
        != file_sha256(REPOSITORY_ROOT / experiment["action_contract"])
        or feature_identity.get("dataset", {}).get("revision") != dataset_revision
        or feature_identity.get("dataset", {}).get("manifest_sha256")
        != dataset_manifest_sha256
        or feature_identity.get("split") != experiment["dataset"]["split"]
        or feature_identity.get("processor") != backbone["processor"]
        or feature_identity.get("feature", {}).get("pooling") != backbone["pooling"]
        or feature_identity.get("feature", {}).get("layer") != backbone["feature_layer"]
        or feature_identity.get("model", {}).get("identifier") != backbone["identifier"]
        or feature_identity.get("model", {}).get("adaptation") != backbone["adaptation"]
        or feature_identity.get("model", {}).get("files")
        != artifact_config.get("base_model_file_hashes")
    ):
        raise ValueError("Trajectory feature cache is not bound to this artifact/config/data.")
    return manifest


def _policy_first_action(
    policy: Any,
    statistics: DatasetStatistics,
    contract: Any,
    instruction: str,
    observation: dict[str, Any],
    state: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Run one closed-loop inference and return the denormalized first action."""

    with torch.inference_mode():
        normalized_chunk = policy(
            {
                "images": observation["images"],
                "instruction": instruction,
            },
            normalize(state.unsqueeze(0), statistics.state).to(device),
        )
        expected_shape = (1, contract.chunk_length, contract.dimension)
        if not isinstance(normalized_chunk, torch.Tensor) or tuple(
            normalized_chunk.shape
        ) != expected_shape:
            raise ValueError("Trajectory policy output violates the action-chunk contract.")
        raw_action = denormalize(normalized_chunk, statistics.action)[0, 0]
    raw_action = raw_action.to(torch.float32).cpu()
    contract.validate_tensor(raw_action, allow_chunk=False)
    return raw_action


def _trajectory_divergence_episode(
    policy: Any,
    statistics: DatasetStatistics,
    contract: Any,
    instruction: str,
    rows: list[dict[str, Any]],
    fields: Any,
    *,
    seed: int,
    maximum_steps: int,
) -> dict[str, Any]:
    """Compare time-indexed expert replay with a policy in two independent environments."""

    if maximum_steps <= 0 or len(rows) < maximum_steps:
        raise ValueError("Trajectory rows must cover every requested comparison step.")
    if contract.chunk_execution != "receding_horizon_first_action":
        raise ValueError("Trajectory divergence requires receding-horizon first-action execution.")
    if contract.chunk_execution_steps != 1:
        raise ValueError("Trajectory divergence executes exactly one policy action per inference.")

    expert_environment = GymAlohaEnvironment(
        contract,
        maximum_episode_steps=len(rows),
    )
    policy_environment: GymAlohaEnvironment | None = None
    steps: list[dict[str, Any]] = []
    state_mae_values: list[float] = []
    state_l2_values: list[float] = []
    first_events: dict[str, int | None] = {
        "expert_source_limit_violation": None,
        "policy_raw_limit_violation": None,
        "expert_joint_limit_violation": None,
        "policy_joint_limit_violation": None,
        "expert_unexpected_collision": None,
        "policy_unexpected_collision": None,
        "expert_nonzero_reward": None,
        "policy_nonzero_reward": None,
        "reward_divergence": None,
        "termination_divergence": None,
    }
    last_expert_done = False
    last_policy_done = False
    end_reason = "requested_steps_completed"

    def record_first(name: str, step_index: int, condition: bool) -> None:
        if condition and first_events[name] is None:
            first_events[name] = step_index

    try:
        policy_environment = GymAlohaEnvironment(
            contract,
            maximum_episode_steps=len(rows),
        )
        if policy_environment is expert_environment:
            raise RuntimeError("Trajectory divergence requires two distinct environments.")
        expert_backends = _environment_backend_objects(expert_environment)
        policy_backends = _environment_backend_objects(policy_environment)
        if any(
            expert_backend is policy_backend
            for expert_backend in expert_backends
            for policy_backend in policy_backends
        ):
            raise RuntimeError("Trajectory environments share a simulator backend.")
        expert_observation = dict(expert_environment.reset(seed=seed))
        policy_observation = dict(policy_environment.reset(seed=seed))
        reset = _reset_comparison(expert_observation, policy_observation, contract)
        reset["expert_contacts"] = _contact_metrics(expert_environment)
        reset["policy_contacts"] = _contact_metrics(policy_environment)
        dataset_initial_state = torch.as_tensor(
            rows[0][fields.state], dtype=torch.float32
        )
        contract.validate_tensor(dataset_initial_state, allow_chunk=False)
        reset["dataset_initial_state_mae"] = float(
            (_state(expert_observation, contract.dimension) - dataset_initial_state)
            .abs()
            .mean()
        )
        reset["maximum_dataset_initial_state_mae"] = (
            TRACE_MAXIMUM_INITIAL_STATE_MAE
        )
        if (
            not math.isfinite(reset["dataset_initial_state_mae"])
            or reset["dataset_initial_state_mae"] > TRACE_MAXIMUM_INITIAL_STATE_MAE
        ):
            raise ValueError("Trajectory simulator reset differs from the dataset state.")

        device = _requested_torch_device()
        for step_index, row in enumerate(rows[:maximum_steps]):
            expert_pre_state = _state(expert_observation, contract.dimension)
            policy_pre_state = _state(policy_observation, contract.dimension)
            expert_raw_action = torch.as_tensor(
                row[fields.action], dtype=torch.float32
            )
            contract.validate_tensor(expert_raw_action, allow_chunk=False)
            expert_action, expert_clip_mask = contract.clip(expert_raw_action)

            policy_raw_action = _policy_first_action(
                policy,
                statistics,
                contract,
                instruction,
                policy_observation,
                policy_pre_state,
                device,
            )
            policy_action, policy_clip_mask = contract.clip(policy_raw_action)

            expert_observation, expert_reward, expert_done, expert_info = (
                expert_environment.step(expert_action)
            )
            policy_observation, policy_reward, policy_done, policy_info = (
                policy_environment.step(policy_action)
            )
            if not math.isfinite(expert_reward) or not math.isfinite(policy_reward):
                raise ValueError("Trajectory reward contains NaN or Inf.")
            expert_observation = dict(expert_observation)
            policy_observation = dict(policy_observation)
            expert_state = _state(expert_observation, contract.dimension)
            policy_state = _state(policy_observation, contract.dimension)
            state_difference = policy_state - expert_state
            state_mae = float(state_difference.abs().mean())
            state_l2 = float(state_difference.square().sum().sqrt())
            state_mae_values.append(state_mae)
            state_l2_values.append(state_l2)
            action_difference = policy_action - expert_action
            expert_limits = _joint_limit_metrics(expert_state, contract)
            policy_limits = _joint_limit_metrics(policy_state, contract)
            expert_contacts = _contact_metrics(expert_environment)
            policy_contacts = _contact_metrics(policy_environment)
            reward_diverged = expert_reward != policy_reward
            termination_diverged = expert_done != policy_done

            record_first(
                "expert_source_limit_violation",
                step_index,
                bool(expert_clip_mask.any()),
            )
            record_first(
                "policy_raw_limit_violation",
                step_index,
                bool(policy_clip_mask.any()),
            )
            record_first(
                "expert_joint_limit_violation",
                step_index,
                expert_limits["count"] > 0,
            )
            record_first(
                "policy_joint_limit_violation",
                step_index,
                policy_limits["count"] > 0,
            )
            record_first(
                "expert_unexpected_collision",
                step_index,
                expert_contacts["unexpected_count"] > 0,
            )
            record_first(
                "policy_unexpected_collision",
                step_index,
                policy_contacts["unexpected_count"] > 0,
            )
            record_first("expert_nonzero_reward", step_index, expert_reward != 0.0)
            record_first("policy_nonzero_reward", step_index, policy_reward != 0.0)
            record_first("reward_divergence", step_index, reward_diverged)
            record_first("termination_divergence", step_index, termination_diverged)

            reference = {
                "type": "time_indexed_expert_reference",
                "state_conditioned": False,
                "recovery_oracle": False,
                "warning": (
                    "After policy divergence this action is not a recovery oracle for the "
                    "policy-visited state."
                ),
            }
            expert_success = bool(expert_info.get("is_success", False))
            policy_success = bool(policy_info.get("is_success", False))
            step_report = {
                "step_index": step_index,
                "dataset_frame_index": int(row[fields.frame_index]),
                "dataset_timestamp": float(row[fields.timestamp]),
                "expert_reference": reference,
                "expert": {
                    "pre_state": expert_pre_state.tolist(),
                    "raw_action": expert_raw_action.tolist(),
                    "clipped_action": expert_action.tolist(),
                    "raw_clip_mask": expert_clip_mask.tolist(),
                    "raw_clipped_elements": int(expert_clip_mask.sum()),
                    "post_state": expert_state.tolist(),
                    "reward": expert_reward,
                    "done": expert_done,
                    "terminated": bool(expert_info.get("terminated", False)),
                    "truncated": bool(expert_info.get("truncated", False)),
                    "success": expert_success,
                    "joint_limits": expert_limits,
                    "contacts": expert_contacts,
                },
                "policy": {
                    "pre_state": policy_pre_state.tolist(),
                    "raw_action": policy_raw_action.tolist(),
                    "clipped_action": policy_action.tolist(),
                    "raw_clip_mask": policy_clip_mask.tolist(),
                    "raw_clipped_elements": int(policy_clip_mask.sum()),
                    "post_state": policy_state.tolist(),
                    "reward": policy_reward,
                    "done": policy_done,
                    "terminated": bool(policy_info.get("terminated", False)),
                    "truncated": bool(policy_info.get("truncated", False)),
                    "success": policy_success,
                    "joint_limits": policy_limits,
                    "contacts": policy_contacts,
                },
                "divergence": {
                    "post_state_mae": state_mae,
                    "post_state_l2": state_l2,
                    "post_state_maximum_absolute_difference": float(
                        state_difference.abs().max()
                    ),
                    "clipped_action_mae": float(action_difference.abs().mean()),
                    "clipped_action_l2": float(
                        action_difference.square().sum().sqrt()
                    ),
                    "reward_delta_policy_minus_expert": policy_reward - expert_reward,
                    "reward_diverged": reward_diverged,
                    "success_diverged": expert_success != policy_success,
                    "termination_diverged": termination_diverged,
                },
            }
            steps.append(step_report)
            last_expert_done = expert_done
            last_policy_done = policy_done
            if expert_done or policy_done:
                if expert_done and policy_done:
                    end_reason = "both_environments_done"
                elif expert_done:
                    end_reason = "expert_environment_done"
                else:
                    end_reason = "policy_environment_done"
                break

        prefix = _trace_prefix_digest(steps) if len(steps) >= 3 else None
        return {
            "reset": reset,
            "steps": steps,
            "summary": {
                "steps_executed": len(steps),
                "end_reason": end_reason,
                "expert_done": last_expert_done,
                "policy_done": last_policy_done,
                "maximum_post_state_mae": max(state_mae_values, default=0.0),
                "maximum_post_state_l2": max(state_l2_values, default=0.0),
                "final_post_state_mae": state_mae_values[-1] if state_mae_values else 0.0,
                "final_post_state_l2": state_l2_values[-1] if state_l2_values else 0.0,
                "first_crossings": {
                    "post_state_mae": _first_crossings(state_mae_values),
                    "post_state_l2": _first_crossings(state_l2_values),
                },
                "first_events": first_events,
                "canonical_first_three_steps_sha256": prefix,
            },
        }
    finally:
        expert_environment.close()
        if policy_environment is not None and policy_environment is not expert_environment:
            policy_environment.close()


def _difference_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    difference = left - right
    if not bool(torch.isfinite(difference).all()):
        raise ValueError("Teacher-forced decomposition contains NaN or Inf.")
    return {
        "mae": float(difference.abs().mean()),
        "l2": float(difference.square().sum().sqrt()),
        "maximum_absolute_difference": float(difference.abs().max()),
    }


def _factorial_action_metrics(
    recorded_image_recorded_state: torch.Tensor,
    recorded_image_sim_state: torch.Tensor,
    sim_image_recorded_state: torch.Tensor,
    sim_image_sim_state: torch.Tensor,
) -> dict[str, dict[str, float]]:
    interaction = (
        sim_image_sim_state
        - recorded_image_sim_state
        - sim_image_recorded_state
        + recorded_image_recorded_state
    )
    return {
        "state_swap_at_recorded_image": _difference_metrics(
            recorded_image_sim_state,
            recorded_image_recorded_state,
        ),
        "image_swap_at_recorded_state": _difference_metrics(
            sim_image_recorded_state,
            recorded_image_recorded_state,
        ),
        "joint_recorded_to_sim_swap": _difference_metrics(
            sim_image_sim_state,
            recorded_image_recorded_state,
        ),
        "image_state_interaction": _difference_metrics(
            interaction,
            torch.zeros_like(interaction),
        ),
    }


def _teacher_forced_decomposition_episode(
    policy: Any,
    statistics: DatasetStatistics,
    contract: Any,
    instruction: str,
    rows: list[dict[str, Any]],
    fields: Any,
    *,
    seed: int,
) -> dict[str, Any]:
    """Separate policy-on-expert-stream error from closed-loop stream shift."""

    if len(rows) < TEACHER_DECOMPOSITION_STEPS:
        raise ValueError("Teacher-forced decomposition requires three dataset rows.")
    if (
        contract.chunk_execution != "receding_horizon_first_action"
        or contract.chunk_execution_steps != 1
    ):
        raise ValueError("Teacher-forced decomposition requires first-action execution.")

    expert_environment = GymAlohaEnvironment(
        contract,
        maximum_episode_steps=len(rows),
    )
    closed_environment: GymAlohaEnvironment | None = None
    steps: list[dict[str, Any]] = []
    try:
        closed_environment = GymAlohaEnvironment(
            contract,
            maximum_episode_steps=len(rows),
        )
        if closed_environment is expert_environment:
            raise RuntimeError("Teacher-forced decomposition requires distinct environments.")
        expert_backends = _environment_backend_objects(expert_environment)
        closed_backends = _environment_backend_objects(closed_environment)
        if any(
            expert_backend is closed_backend
            for expert_backend in expert_backends
            for closed_backend in closed_backends
        ):
            raise RuntimeError("Teacher-forced environments share a simulator backend.")

        expert_observation = dict(expert_environment.reset(seed=seed))
        closed_observation = dict(closed_environment.reset(seed=seed))
        reset = _reset_comparison(expert_observation, closed_observation, contract)
        dataset_initial_state = torch.as_tensor(
            rows[0][fields.state], dtype=torch.float32
        )
        contract.validate_tensor(dataset_initial_state, allow_chunk=False)
        dataset_initial_state_mae = float(
            (_state(expert_observation, contract.dimension) - dataset_initial_state)
            .abs()
            .mean()
        )
        if (
            not math.isfinite(dataset_initial_state_mae)
            or dataset_initial_state_mae > TRACE_MAXIMUM_INITIAL_STATE_MAE
        ):
            raise ValueError("Teacher-forced reset differs from the dataset state.")
        reset.update(
            {
                "dataset_initial_state_mae": dataset_initial_state_mae,
                "maximum_dataset_initial_state_mae": (
                    TRACE_MAXIMUM_INITIAL_STATE_MAE
                ),
                "expert_contacts": _contact_metrics(expert_environment),
                "closed_contacts": _contact_metrics(closed_environment),
            }
        )

        device = _requested_torch_device()
        for step_index, row in enumerate(rows[:TEACHER_DECOMPOSITION_STEPS]):
            expert_state = _state(expert_observation, contract.dimension)
            closed_state = _state(closed_observation, contract.dimension)
            dataset_raw_action = torch.as_tensor(
                row[fields.action], dtype=torch.float32
            )
            contract.validate_tensor(dataset_raw_action, allow_chunk=False)
            dataset_action, dataset_clip_mask = contract.clip(dataset_raw_action)
            teacher_raw_action = _policy_first_action(
                policy,
                statistics,
                contract,
                instruction,
                expert_observation,
                expert_state,
                device,
            )
            closed_raw_action = _policy_first_action(
                policy,
                statistics,
                contract,
                instruction,
                closed_observation,
                closed_state,
                device,
            )
            teacher_action, teacher_clip_mask = contract.clip(teacher_raw_action)
            closed_action, closed_clip_mask = contract.clip(closed_raw_action)
            same_input_difference = _difference_metrics(
                closed_raw_action,
                teacher_raw_action,
            )
            if (
                step_index == 0
                and same_input_difference["maximum_absolute_difference"]
                > TEACHER_STEP_ZERO_DETERMINISM_TOLERANCE
            ):
                raise RuntimeError(
                    "Teacher and closed streams disagree on identical reset inputs."
                )

            next_expert, expert_reward, expert_done, expert_info = (
                expert_environment.step(dataset_action)
            )
            next_closed, closed_reward, closed_done, closed_info = (
                closed_environment.step(closed_action)
            )
            if not math.isfinite(expert_reward) or not math.isfinite(closed_reward):
                raise ValueError("Teacher-forced decomposition reward contains NaN or Inf.")
            expert_observation = dict(next_expert)
            closed_observation = dict(next_closed)
            expert_post_state = _state(expert_observation, contract.dimension)
            closed_post_state = _state(closed_observation, contract.dimension)
            steps.append(
                {
                    "step_index": step_index,
                    "dataset_frame_index": int(row[fields.frame_index]),
                    "dataset_timestamp": float(row[fields.timestamp]),
                    "dataset_reference": {
                        "type": "time_indexed_expert_reference",
                        "state_conditioned": False,
                        "recovery_oracle": False,
                        "raw_action": dataset_raw_action.tolist(),
                        "clipped_action": dataset_action.tolist(),
                        "raw_clip_mask": dataset_clip_mask.tolist(),
                        "raw_clipped_elements": int(dataset_clip_mask.sum()),
                    },
                    "policy_on_expert_stream": {
                        "executed": False,
                        "observation_stream": "expert_simulator",
                        "pre_state": expert_state.tolist(),
                        "raw_action": teacher_raw_action.tolist(),
                        "clipped_action": teacher_action.tolist(),
                        "raw_clip_mask": teacher_clip_mask.tolist(),
                        "raw_clipped_elements": int(teacher_clip_mask.sum()),
                    },
                    "policy_closed_loop": {
                        "executed": True,
                        "observation_stream": "closed_loop_policy_simulator",
                        "pre_state": closed_state.tolist(),
                        "raw_action": closed_raw_action.tolist(),
                        "clipped_action": closed_action.tolist(),
                        "raw_clip_mask": closed_clip_mask.tolist(),
                        "raw_clipped_elements": int(closed_clip_mask.sum()),
                    },
                    "outcomes": {
                        "expert_reward": expert_reward,
                        "closed_reward": closed_reward,
                        "expert_done": expert_done,
                        "closed_done": closed_done,
                        "expert_terminated": bool(expert_info.get("terminated", False)),
                        "expert_truncated": bool(expert_info.get("truncated", False)),
                        "closed_terminated": bool(closed_info.get("terminated", False)),
                        "closed_truncated": bool(closed_info.get("truncated", False)),
                        "expert_post_state": expert_post_state.tolist(),
                        "closed_post_state": closed_post_state.tolist(),
                        "expert_contacts": _contact_metrics(expert_environment),
                        "closed_contacts": _contact_metrics(closed_environment),
                    },
                    "decomposition": {
                        "comparison_space": "contract_clipped_physical_action",
                        "policy_on_expert_vs_dataset": _difference_metrics(
                            teacher_action,
                            dataset_action,
                        ),
                        "closed_vs_policy_on_expert": _difference_metrics(
                            closed_action,
                            teacher_action,
                        ),
                        "closed_vs_dataset": _difference_metrics(
                            closed_action,
                            dataset_action,
                        ),
                        "closed_vs_expert_post_state": _difference_metrics(
                            closed_post_state,
                            expert_post_state,
                        ),
                        "same_input_raw_action_check": same_input_difference,
                    },
                }
            )
            if expert_done or closed_done:
                raise RuntimeError(
                    "Teacher-forced three-step diagnostic ended before its fixed horizon."
                )

        return {
            "reset": reset,
            "steps": steps,
            "summary": {
                "steps_executed": len(steps),
                "step_zero_same_input_maximum_absolute_difference": steps[0][
                    "decomposition"
                ]["same_input_raw_action_check"]["maximum_absolute_difference"],
                "policy_on_expert_vs_dataset_action_l2": [
                    step["decomposition"]["policy_on_expert_vs_dataset"]["l2"]
                    for step in steps
                ],
                "closed_vs_policy_on_expert_action_l2": [
                    step["decomposition"]["closed_vs_policy_on_expert"]["l2"]
                    for step in steps
                ],
                "closed_vs_expert_post_state_l2": [
                    step["decomposition"]["closed_vs_expert_post_state"]["l2"]
                    for step in steps
                ],
            },
        }
    finally:
        expert_environment.close()
        if closed_environment is not None and closed_environment is not expert_environment:
            closed_environment.close()


def trajectory_divergence(
    config_path: Path,
    artifact_root: Path,
    alignment_report_path: Path,
    *,
    phase: str,
    episode: int,
    maximum_steps: int | None,
    maximum_alignment_mae: float,
    smoke_report_path: Path | None,
) -> int:
    """Trace validation-only expert/policy divergence without opening hidden test."""

    experiment = load_experiment_config(config_path, REPOSITORY_ROOT)
    _require_validation_episode(experiment, episode)
    validation_values = [
        int(value) for value in experiment["dataset"]["split"]["validation"]
    ]
    validation_episodes = set(validation_values)
    hidden_test_episodes = {
        int(value) for value in experiment["dataset"]["split"]["test"]
    }
    if (
        len(validation_values) != len(validation_episodes)
        or validation_episodes & hidden_test_episodes
    ):
        raise ValueError("Trajectory validation split must be unique and disjoint from test.")
    if maximum_alignment_mae != TRACE_MAXIMUM_ALIGNMENT_MAE:
        raise ValueError("Trajectory alignment tolerance is fixed at 0.005 MAE.")
    if phase == "smoke":
        if maximum_steps not in (None, 3) or smoke_report_path is not None:
            raise ValueError("Smoke trace is fixed at three steps and takes no smoke report.")
        requested_steps = 3
    elif phase == "full":
        if (
            (maximum_steps is not None and maximum_steps <= 3)
            or smoke_report_path is None
        ):
            raise ValueError("Full trace requires the whole episode and a smoke report.")
        requested_steps = 0
    else:
        raise ValueError("Trajectory phase must be either 'smoke' or 'full'.")

    dataset_path = REPOSITORY_ROOT / experiment["dataset"]["config"]
    dataset_config = load_dataset_config(dataset_path)
    root, manifest = resolve_prepared_cache(
        dataset_config,
        REPOSITORY_ROOT,
        validate_checksums=True,
    )
    cleaning_path = root / "cleaning_report.json"
    cleaning = json.loads(cleaning_path.read_text(encoding="utf-8"))
    if cleaning.get("status") != "validated_clean":
        raise ValueError("Trajectory divergence requires a validated-clean dataset cache.")
    dataset_manifest_path = root / "manifest.json"
    dataset_manifest_sha256 = file_sha256(dataset_manifest_path)
    contract_path = REPOSITORY_ROOT / experiment["action_contract"]
    contract = load_action_contract(contract_path)
    contract.validate_order(ordered_feature_names(root, dataset_config.fields.action))
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if float(info["fps"]) != contract.frequency_hz:
        raise ValueError("Trajectory dataset and Action Contract frequencies differ.")
    rows = _dataset_rows(root, episode, dataset_config.fields)
    if phase == "full":
        if maximum_steps is not None and maximum_steps != len(rows):
            raise ValueError("Full trajectory cannot shorten the validation episode.")
        requested_steps = len(rows)
    if len(rows) < requested_steps:
        raise ValueError("Validation episode is shorter than the requested trace.")
    frame_indices = [int(row[dataset_config.fields.frame_index]) for row in rows]
    if frame_indices != list(range(len(rows))):
        raise ValueError("Trajectory frame indices must be contiguous from zero.")
    timestamps = [float(row[dataset_config.fields.timestamp]) for row in rows[:requested_steps]]
    if not all(math.isfinite(value) for value in timestamps):
        raise ValueError("Trajectory timestamps contain NaN or Inf.")
    maximum_timing_error = max(
        (
            abs((current - previous) - 1.0 / contract.frequency_hz)
            for previous, current in zip(timestamps, timestamps[1:])
        ),
        default=0.0,
    )
    if maximum_timing_error > 1e-4:
        raise ValueError("Trajectory timestamps do not match the Action Contract frequency.")
    alignment = _trace_alignment(
        alignment_report_path,
        episode=episode,
        validation_episodes=validation_episodes,
        dataset_revision=manifest.resolved_revision,
        dataset_manifest_sha256=dataset_manifest_sha256,
        action_contract_sha256=file_sha256(contract_path),
        frequency_hz=contract.frequency_hz,
        experiment_id=experiment["experiment_id"],
        experiment_config_sha256=file_sha256(config_path),
        maximum_mae=maximum_alignment_mae,
    )
    artifact_manifest_path = artifact_root / "manifest.json"
    artifact_manifest = _validated_trace_artifact(
        config_path,
        artifact_root,
        experiment,
        contract,
        dataset_revision=manifest.resolved_revision,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )
    identity = {
        "experiment_config_sha256": file_sha256(config_path),
        "artifact_id": artifact_manifest.get("artifact_id"),
        "artifact_manifest_sha256": file_sha256(artifact_manifest_path),
        "dataset_revision": manifest.resolved_revision,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "cleaning_report_sha256": file_sha256(cleaning_path),
        "action_contract_sha256": file_sha256(contract_path),
        "alignment_report_sha256": file_sha256(alignment_report_path),
        "episode": episode,
        "seed": alignment["selected_seed"],
        "runtime_device": str(_requested_torch_device()),
        "evaluation_code": workspace_code_identity(REPOSITORY_ROOT),
    }
    smoke_prerequisite = None
    if phase == "full":
        assert smoke_report_path is not None
        smoke_prerequisite = _validated_trace_smoke_report(
            smoke_report_path,
            identity=identity,
        )

    policy, statistics, loaded_contract, loaded_manifest, instruction = (
        _load_online_artifact(config_path, artifact_root)
    )
    if loaded_contract != contract:
        raise ValueError("Loaded artifact Action Contract differs from the trace contract.")
    if loaded_manifest.get("artifact_id") != identity["artifact_id"]:
        raise ValueError("Loaded artifact identity differs from the trace identity.")
    trace = _trajectory_divergence_episode(
        policy,
        statistics,
        contract,
        instruction,
        rows,
        dataset_config.fields,
        seed=int(alignment["selected_seed"]),
        maximum_steps=requested_steps,
    )
    prefix = trace["summary"]["canonical_first_three_steps_sha256"]
    if phase == "full" and prefix != smoke_prerequisite[
        "canonical_first_three_steps_sha256"
    ]:
        raise RuntimeError("Full trajectory does not reproduce the smoke trace prefix.")
    smoke_passed = (
        phase != "smoke"
        or (
            trace["summary"]["steps_executed"] == 3
            and trace["summary"]["end_reason"] == "requested_steps_completed"
            and trace["reset"]["cross_environment_aligned"] is True
            and isinstance(prefix, str)
        )
    )
    report = {
        "schema_version": 1,
        "status": "passed" if phase == "smoke" and smoke_passed else "complete",
        "diagnostic": "m2_validation_trajectory_divergence",
        "experiment_id": experiment["experiment_id"],
        "identity": identity,
        "protocol": {
            "phase": phase,
            "split": "validation",
            "steps_requested": requested_steps,
            "simulator_horizon": len(rows),
            "policy_execution": "receding_horizon_first_action",
            "state_crossing_thresholds": list(TRACE_STATE_THRESHOLDS),
            "crossing_definition": "first zero-based step where metric >= threshold",
            "maximum_timestamp_step_error": maximum_timing_error,
        },
        "expert_reference": {
            "type": "time_indexed_expert_reference",
            "state_conditioned": False,
            "recovery_oracle": False,
            "warning": (
                "After policy divergence the dataset action is not a recovery oracle for the "
                "policy-visited state."
            ),
        },
        "alignment": alignment,
        "smoke_prerequisite": smoke_prerequisite,
        "reset": trace["reset"],
        "steps": trace["steps"],
        "summary": trace["summary"],
        "optimizer_steps": 0,
        "test_split_opened": False,
    }
    if phase == "smoke" and not smoke_passed:
        report["status"] = "failed"
    json.dumps(report, allow_nan=False)
    path = _write_trajectory_report(report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    print(f"Report: {_display_path(path)}")
    return 0 if report["status"] in {"passed", "complete"} else 1


def teacher_forced_decomposition(
    config_path: Path,
    artifact_root: Path,
    alignment_report_path: Path,
) -> int:
    """Decompose validation error on expert-simulator versus closed-loop streams."""

    experiment = load_experiment_config(config_path, REPOSITORY_ROOT)
    validation_episodes = [
        int(value) for value in experiment["dataset"]["split"]["validation"]
    ]
    hidden_test_episodes = {
        int(value) for value in experiment["dataset"]["split"]["test"]
    }
    if (
        not validation_episodes
        or len(validation_episodes) != len(set(validation_episodes))
        or set(validation_episodes) & hidden_test_episodes
    ):
        raise ValueError("Teacher-forced validation split must be unique and disjoint from test.")

    dataset_path = REPOSITORY_ROOT / experiment["dataset"]["config"]
    dataset_config = load_dataset_config(dataset_path)
    root, manifest = resolve_prepared_cache(
        dataset_config,
        REPOSITORY_ROOT,
        validate_checksums=True,
    )
    cleaning_path = root / "cleaning_report.json"
    if _read_json_object(cleaning_path).get("status") != "validated_clean":
        raise ValueError("Teacher-forced decomposition requires validated-clean data.")
    dataset_manifest_path = root / "manifest.json"
    dataset_manifest_sha256 = file_sha256(dataset_manifest_path)
    contract_path = REPOSITORY_ROOT / experiment["action_contract"]
    contract = load_action_contract(contract_path)
    contract.validate_order(ordered_feature_names(root, dataset_config.fields.action))
    info = _read_json_object(root / "meta" / "info.json")
    if float(info["fps"]) != contract.frequency_hz:
        raise ValueError("Teacher-forced dataset and contract frequencies differ.")

    artifact_manifest_path = artifact_root / "manifest.json"
    artifact_manifest = _validated_trace_artifact(
        config_path,
        artifact_root,
        experiment,
        contract,
        dataset_revision=manifest.resolved_revision,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )
    alignments: dict[int, dict[str, Any]] = {}
    for episode in validation_episodes:
        alignments[episode] = _trace_alignment(
            alignment_report_path,
            episode=episode,
            validation_episodes=set(validation_episodes),
            dataset_revision=manifest.resolved_revision,
            dataset_manifest_sha256=dataset_manifest_sha256,
            action_contract_sha256=file_sha256(contract_path),
            frequency_hz=contract.frequency_hz,
            experiment_id=experiment["experiment_id"],
            experiment_config_sha256=file_sha256(config_path),
            maximum_mae=TRACE_MAXIMUM_ALIGNMENT_MAE,
        )

    identity = {
        "experiment_config_sha256": file_sha256(config_path),
        "artifact_id": artifact_manifest["artifact_id"],
        "artifact_manifest_sha256": file_sha256(artifact_manifest_path),
        "dataset_revision": manifest.resolved_revision,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "cleaning_report_sha256": file_sha256(cleaning_path),
        "action_contract_sha256": file_sha256(contract_path),
        "alignment_report_sha256": file_sha256(alignment_report_path),
        "episode_seeds": {
            str(episode): int(alignments[episode]["selected_seed"])
            for episode in validation_episodes
        },
        "runtime_device": str(_requested_torch_device()),
        "evaluation_code": workspace_code_identity(REPOSITORY_ROOT),
    }
    policy, statistics, loaded_contract, loaded_manifest, instruction = (
        _load_online_artifact(config_path, artifact_root)
    )
    if loaded_contract != contract or loaded_manifest.get("artifact_id") != identity[
        "artifact_id"
    ]:
        raise ValueError("Teacher-forced loaded artifact identity differs.")

    episode_reports: list[dict[str, Any]] = []
    for episode in validation_episodes:
        rows = _dataset_rows(root, episode, dataset_config.fields)
        if len(rows) < TEACHER_DECOMPOSITION_STEPS:
            raise ValueError("Teacher-forced validation episode is shorter than three rows.")
        frame_indices = [int(row[dataset_config.fields.frame_index]) for row in rows]
        if frame_indices != list(range(len(rows))):
            raise ValueError("Teacher-forced frame indices must be contiguous from zero.")
        timestamps = [
            float(row[dataset_config.fields.timestamp])
            for row in rows[:TEACHER_DECOMPOSITION_STEPS]
        ]
        maximum_timing_error = max(
            (
                abs((current - previous) - 1.0 / contract.frequency_hz)
                for previous, current in zip(timestamps, timestamps[1:])
            ),
            default=0.0,
        )
        if (
            not all(math.isfinite(value) for value in timestamps)
            or maximum_timing_error > 1e-4
        ):
            raise ValueError("Teacher-forced timestamps differ from the contract.")
        result = _teacher_forced_decomposition_episode(
            policy,
            statistics,
            contract,
            instruction,
            rows,
            dataset_config.fields,
            seed=int(alignments[episode]["selected_seed"]),
        )
        episode_reports.append(
            {
                "episode": episode,
                "seed": int(alignments[episode]["selected_seed"]),
                "alignment": alignments[episode],
                "maximum_timestamp_step_error": maximum_timing_error,
                **result,
            }
        )

    median_teacher_dataset = [
        float(
            median(
                report["summary"]["policy_on_expert_vs_dataset_action_l2"][index]
                for report in episode_reports
            )
        )
        for index in range(TEACHER_DECOMPOSITION_STEPS)
    ]
    median_closed_teacher = [
        float(
            median(
                report["summary"]["closed_vs_policy_on_expert_action_l2"][index]
                for report in episode_reports
            )
        )
        for index in range(TEACHER_DECOMPOSITION_STEPS)
    ]
    report = {
        "schema_version": 1,
        "status": "complete",
        "diagnostic": "m2_validation_teacher_forced_closed_loop_decomposition",
        "experiment_id": experiment["experiment_id"],
        "identity": identity,
        "protocol": {
            "split": "validation",
            "episodes": validation_episodes,
            "steps_per_episode": TEACHER_DECOMPOSITION_STEPS,
            "total_steps": len(validation_episodes) * TEACHER_DECOMPOSITION_STEPS,
            "policy_execution": "receding_horizon_first_action",
            "comparison_space": "contract_clipped_physical_action",
            "teacher_stream_executed": False,
            "step_zero_determinism_tolerance": (
                TEACHER_STEP_ZERO_DETERMINISM_TOLERANCE
            ),
        },
        "interpretation_contract": {
            "dataset_reference": "time_indexed_expert_reference",
            "dataset_reference_state_conditioned": False,
            "dataset_reference_recovery_oracle": False,
            "policy_on_expert_stream": (
                "The same learned policy evaluated on the expert simulator stream; it is "
                "not an expert policy and is never executed."
            ),
            "limitation": (
                "This separates within-simulator closed-loop stream shift but does not isolate "
                "the original recorded-image domain from simulator alignment error."
            ),
        },
        "episodes": episode_reports,
        "summary": {
            "episodes_completed": len(episode_reports),
            "total_steps": sum(
                item["summary"]["steps_executed"] for item in episode_reports
            ),
            "maximum_step_zero_same_input_difference": max(
                item["summary"][
                    "step_zero_same_input_maximum_absolute_difference"
                ]
                for item in episode_reports
            ),
            "median_policy_on_expert_vs_dataset_action_l2_by_step": (
                median_teacher_dataset
            ),
            "median_closed_vs_policy_on_expert_action_l2_by_step": (
                median_closed_teacher
            ),
        },
        "optimizer_steps": 0,
        "test_split_opened": False,
    }
    json.dumps(report, allow_nan=False)
    path = _write_teacher_forced_report(report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    print(f"Report: {_display_path(path)}")
    return 0


def recorded_domain_probe(
    config_path: Path,
    artifact_root: Path,
) -> int:
    """Evaluate the online artifact on original validation images and states."""

    experiment = load_experiment_config(config_path, REPOSITORY_ROOT)
    validation_episodes = tuple(
        int(value) for value in experiment["dataset"]["split"]["validation"]
    )
    hidden_test_episodes = {
        int(value) for value in experiment["dataset"]["split"]["test"]
    }
    if (
        not validation_episodes
        or len(validation_episodes) != len(set(validation_episodes))
        or set(validation_episodes) & hidden_test_episodes
    ):
        raise ValueError("Recorded-domain validation must be unique and disjoint from test.")
    frame_indices = (0, 1, 2, 5, 10)
    configured_training_frame_stride = int(experiment["dataset"]["frame_stride"])
    cache_stride_matched_frames = _cache_stride_matched_frames(
        frame_indices,
        configured_training_frame_stride,
    )

    dataset_path = REPOSITORY_ROOT / experiment["dataset"]["config"]
    dataset_config = load_dataset_config(dataset_path)
    root, manifest = resolve_prepared_cache(
        dataset_config,
        REPOSITORY_ROOT,
        validate_checksums=True,
    )
    cleaning_path = root / "cleaning_report.json"
    if _read_json_object(cleaning_path).get("status") != "validated_clean":
        raise ValueError("Recorded-domain probe requires validated-clean data.")
    dataset_manifest_path = root / "manifest.json"
    dataset_manifest_sha256 = file_sha256(dataset_manifest_path)
    contract_path = REPOSITORY_ROOT / experiment["action_contract"]
    contract = load_action_contract(contract_path)
    contract.validate_order(ordered_feature_names(root, dataset_config.fields.action))
    if float(_read_json_object(root / "meta" / "info.json")["fps"]) != (
        contract.frequency_hz
    ):
        raise ValueError("Recorded-domain dataset and contract frequencies differ.")

    artifact_manifest_path = artifact_root / "manifest.json"
    artifact_manifest = _validated_trace_artifact(
        config_path,
        artifact_root,
        experiment,
        contract,
        dataset_revision=manifest.resolved_revision,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )
    identity = {
        "experiment_config_sha256": file_sha256(config_path),
        "artifact_id": artifact_manifest["artifact_id"],
        "artifact_manifest_sha256": file_sha256(artifact_manifest_path),
        "dataset_revision": manifest.resolved_revision,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "cleaning_report_sha256": file_sha256(cleaning_path),
        "action_contract_sha256": file_sha256(contract_path),
        "runtime_device": str(_requested_torch_device()),
        "evaluation_code": workspace_code_identity(REPOSITORY_ROOT),
    }
    policy, statistics, loaded_contract, loaded_manifest, instruction = (
        _load_online_artifact(config_path, artifact_root)
    )
    if loaded_contract != contract or loaded_manifest.get("artifact_id") != identity[
        "artifact_id"
    ]:
        raise ValueError("Recorded-domain loaded artifact identity differs.")

    adapter = LeRobotV3Adapter(
        repo_id=dataset_config.repo_id,
        revision=manifest.resolved_revision,
        root=root,
        episodes=validation_episodes,
        cameras=dataset_config.cameras,
        fields=dataset_config.fields,
        embodiment=dataset_config.embodiment,
        license_name=dataset_config.license,
    )
    required = {
        (episode, frame_index)
        for episode in validation_episodes
        for frame_index in frame_indices
    }
    source_indices: dict[tuple[int, int], int] = {}
    for source_index in range(len(adapter)):
        reference = adapter.frame_reference(source_index)
        key = (int(reference.episode_id), reference.frame_index)
        if key in required:
            if key in source_indices:
                raise ValueError("Recorded-domain dataset frame is duplicated.")
            source_indices[key] = source_index
        if len(source_indices) == len(required):
            break
    if set(source_indices) != required:
        raise ValueError("Recorded-domain dataset is missing a required validation frame.")

    device = _requested_torch_device()
    episode_reports: list[dict[str, Any]] = []
    for episode in validation_episodes:
        frame_reports: list[dict[str, Any]] = []
        for frame_index in frame_indices:
            frame = adapter[source_indices[(episode, frame_index)]]
            if (
                int(frame.episode_id) != episode
                or frame.frame_index != frame_index
                or frame.instruction != instruction
                or not math.isfinite(frame.timestamp)
                or abs(frame.timestamp - frame_index / contract.frequency_hz) > 1e-4
            ):
                raise ValueError("Recorded-domain frame identity or timing differs.")
            state = frame.robot_state.to(torch.float32).cpu()
            dataset_raw_action = frame.action.to(torch.float32).cpu()
            contract.validate_tensor(state, allow_chunk=False)
            contract.validate_tensor(dataset_raw_action, allow_chunk=False)
            predicted_raw_action = _policy_first_action(
                policy,
                statistics,
                contract,
                instruction,
                {"images": frame.images},
                state,
                device,
            )
            dataset_action, dataset_clip_mask = contract.clip(dataset_raw_action)
            predicted_action, predicted_clip_mask = contract.clip(predicted_raw_action)
            frame_reports.append(
                {
                    "frame_index": frame_index,
                    "timestamp": frame.timestamp,
                    "image_shapes": {
                        name: list(image.shape) for name, image in frame.images.items()
                    },
                    "dataset_reference": {
                        "raw_action": dataset_raw_action.tolist(),
                        "clipped_action": dataset_action.tolist(),
                        "raw_clip_mask": dataset_clip_mask.tolist(),
                        "raw_clipped_elements": int(dataset_clip_mask.sum()),
                    },
                    "policy_on_recorded_domain": {
                        "raw_action": predicted_raw_action.tolist(),
                        "clipped_action": predicted_action.tolist(),
                        "raw_clip_mask": predicted_clip_mask.tolist(),
                        "raw_clipped_elements": int(predicted_clip_mask.sum()),
                    },
                    "error": {
                        "comparison_space": "contract_clipped_physical_action",
                        **_difference_metrics(predicted_action, dataset_action),
                    },
                }
            )
        episode_reports.append(
            {
                "episode": episode,
                "frames": frame_reports,
            }
        )

    median_l2_by_frame = {
        str(frame_index): float(
            median(
                next(
                    frame["error"]["l2"]
                    for frame in episode_report["frames"]
                    if frame["frame_index"] == frame_index
                )
                for episode_report in episode_reports
            )
        )
        for frame_index in frame_indices
    }
    median_mae_by_frame = {
        str(frame_index): float(
            median(
                next(
                    frame["error"]["mae"]
                    for frame in episode_report["frames"]
                    if frame["frame_index"] == frame_index
                )
                for episode_report in episode_reports
            )
        )
        for frame_index in frame_indices
    }
    report = {
        "schema_version": 1,
        "status": "complete",
        "diagnostic": "m2_validation_recorded_domain_online_probe",
        "experiment_id": experiment["experiment_id"],
        "identity": identity,
        "protocol": {
            "split": "validation",
            "episodes": list(validation_episodes),
            "frame_indices": list(frame_indices),
            "early_contiguous_frames": [0, 1, 2],
            "cache_stride_matched_frames": cache_stride_matched_frames,
            "configured_training_frame_stride": configured_training_frame_stride,
            "policy_execution": "inference_only_first_action",
            "comparison_space": "contract_clipped_physical_action",
        },
        "episodes": episode_reports,
        "summary": {
            "episodes_completed": len(episode_reports),
            "frames_evaluated": sum(
                len(episode_report["frames"])
                for episode_report in episode_reports
            ),
            "median_action_l2_by_frame": median_l2_by_frame,
            "median_action_mae_by_frame": median_mae_by_frame,
        },
        "optimizer_steps": 0,
        "test_split_opened": False,
    }
    json.dumps(report, allow_nan=False)
    path = _write_recorded_domain_report(report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    print(f"Report: {_display_path(path)}")
    return 0


def domain_factorial_probe(
    config_path: Path,
    artifact_root: Path,
    alignment_report_path: Path,
    recorded_images_path: Path,
) -> int:
    """Cross recorded/simulator images and states at aligned validation resets."""

    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    from scripts.diagnose_m2 import _load_initial_images

    experiment = load_experiment_config(config_path, REPOSITORY_ROOT)
    validation_episodes = tuple(
        int(value) for value in experiment["dataset"]["split"]["validation"]
    )
    hidden_test_episodes = {
        int(value) for value in experiment["dataset"]["split"]["test"]
    }
    if (
        not validation_episodes
        or len(validation_episodes) != len(set(validation_episodes))
        or set(validation_episodes) & hidden_test_episodes
    ):
        raise ValueError("Domain factorial validation must be unique and disjoint from test.")

    dataset_path = REPOSITORY_ROOT / experiment["dataset"]["config"]
    dataset_config = load_dataset_config(dataset_path)
    root, manifest = resolve_prepared_cache(
        dataset_config,
        REPOSITORY_ROOT,
        validate_checksums=True,
    )
    cleaning_path = root / "cleaning_report.json"
    if _read_json_object(cleaning_path).get("status") != "validated_clean":
        raise ValueError("Domain factorial probe requires validated-clean data.")
    dataset_manifest_path = root / "manifest.json"
    dataset_manifest_sha256 = file_sha256(dataset_manifest_path)
    contract_path = REPOSITORY_ROOT / experiment["action_contract"]
    contract = load_action_contract(contract_path)
    contract.validate_order(ordered_feature_names(root, dataset_config.fields.action))
    if float(_read_json_object(root / "meta" / "info.json")["fps"]) != (
        contract.frequency_hz
    ):
        raise ValueError("Domain factorial dataset and contract frequencies differ.")

    expected_scope = {
        "experiment_id": experiment["experiment_id"],
        "experiment_config_sha256": file_sha256(config_path),
        "split": "validation",
        "episodes": list(validation_episodes),
        "test_split_opened": False,
    }
    recorded_images, recorded_image_identity = _load_initial_images(
        recorded_images_path,
        expected_manifest_sha256=dataset_manifest_sha256,
        expected_revision=manifest.resolved_revision,
        episodes=validation_episodes,
    )
    if recorded_image_identity.get("validation_scope") != expected_scope:
        raise ValueError("Domain factorial recorded images differ from validation config.")

    artifact_manifest_path = artifact_root / "manifest.json"
    artifact_manifest = _validated_trace_artifact(
        config_path,
        artifact_root,
        experiment,
        contract,
        dataset_revision=manifest.resolved_revision,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )
    alignments: dict[int, dict[str, Any]] = {}
    for episode in validation_episodes:
        alignments[episode] = _trace_alignment(
            alignment_report_path,
            episode=episode,
            validation_episodes=set(validation_episodes),
            dataset_revision=manifest.resolved_revision,
            dataset_manifest_sha256=dataset_manifest_sha256,
            action_contract_sha256=file_sha256(contract_path),
            frequency_hz=contract.frequency_hz,
            experiment_id=experiment["experiment_id"],
            experiment_config_sha256=file_sha256(config_path),
            maximum_mae=TRACE_MAXIMUM_ALIGNMENT_MAE,
        )
    identity = {
        "experiment_config_sha256": file_sha256(config_path),
        "artifact_id": artifact_manifest["artifact_id"],
        "artifact_manifest_sha256": file_sha256(artifact_manifest_path),
        "dataset_revision": manifest.resolved_revision,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "cleaning_report_sha256": file_sha256(cleaning_path),
        "action_contract_sha256": file_sha256(contract_path),
        "alignment_report_sha256": file_sha256(alignment_report_path),
        "recorded_image_identity": recorded_image_identity,
        "episode_seeds": {
            str(episode): int(alignments[episode]["selected_seed"])
            for episode in validation_episodes
        },
        "runtime_device": str(_requested_torch_device()),
        "evaluation_code": workspace_code_identity(REPOSITORY_ROOT),
    }
    policy, statistics, loaded_contract, loaded_manifest, instruction = (
        _load_online_artifact(config_path, artifact_root)
    )
    if loaded_contract != contract or loaded_manifest.get("artifact_id") != identity[
        "artifact_id"
    ]:
        raise ValueError("Domain factorial loaded artifact identity differs.")

    device = _requested_torch_device()
    camera_name = next(iter(dataset_config.cameras))
    episode_reports: list[dict[str, Any]] = []
    for episode in validation_episodes:
        rows = _dataset_rows(root, episode, dataset_config.fields)
        if not rows or int(rows[0][dataset_config.fields.frame_index]) != 0:
            raise ValueError("Domain factorial dataset has no frame-zero row.")
        recorded_state = torch.as_tensor(
            rows[0][dataset_config.fields.state], dtype=torch.float32
        )
        dataset_raw_action = torch.as_tensor(
            rows[0][dataset_config.fields.action], dtype=torch.float32
        )
        contract.validate_tensor(recorded_state, allow_chunk=False)
        contract.validate_tensor(dataset_raw_action, allow_chunk=False)
        recorded_image = recorded_images[episode]
        if not bool(torch.isfinite(recorded_image).all()):
            raise ValueError("Domain factorial recorded image contains NaN or Inf.")

        environment = GymAlohaEnvironment(
            contract,
            maximum_episode_steps=len(rows),
        )
        try:
            simulator_observation = dict(
                environment.reset(seed=int(alignments[episode]["selected_seed"]))
            )
            simulator_state = _state(simulator_observation, contract.dimension)
            simulator_image = simulator_observation["images"].get(camera_name)
            if (
                not isinstance(simulator_image, torch.Tensor)
                or simulator_image.shape != recorded_image.shape
                or not bool(torch.isfinite(simulator_image).all())
            ):
                raise ValueError("Domain factorial simulator image is invalid.")
            state_mae = float((simulator_state - recorded_state).abs().mean())
            if state_mae > TRACE_MAXIMUM_INITIAL_STATE_MAE:
                raise ValueError("Domain factorial simulator state exceeds alignment tolerance.")
            recorded_observation = {"images": {camera_name: recorded_image}}
            simulator_images = {"images": {camera_name: simulator_image}}
            raw_actions = {
                "recorded_image_recorded_state": _policy_first_action(
                    policy,
                    statistics,
                    contract,
                    instruction,
                    recorded_observation,
                    recorded_state,
                    device,
                ),
                "recorded_image_sim_state": _policy_first_action(
                    policy,
                    statistics,
                    contract,
                    instruction,
                    recorded_observation,
                    simulator_state,
                    device,
                ),
                "sim_image_recorded_state": _policy_first_action(
                    policy,
                    statistics,
                    contract,
                    instruction,
                    simulator_images,
                    recorded_state,
                    device,
                ),
                "sim_image_sim_state": _policy_first_action(
                    policy,
                    statistics,
                    contract,
                    instruction,
                    simulator_images,
                    simulator_state,
                    device,
                ),
            }
            clipped_actions: dict[str, torch.Tensor] = {}
            action_reports: dict[str, Any] = {}
            for name, raw_action in raw_actions.items():
                clipped, clip_mask = contract.clip(raw_action)
                clipped_actions[name] = clipped
                action_reports[name] = {
                    "raw_action": raw_action.tolist(),
                    "clipped_action": clipped.tolist(),
                    "raw_clip_mask": clip_mask.tolist(),
                    "raw_clipped_elements": int(clip_mask.sum()),
                }
            dataset_action, dataset_clip_mask = contract.clip(dataset_raw_action)
            factorial = _factorial_action_metrics(
                clipped_actions["recorded_image_recorded_state"],
                clipped_actions["recorded_image_sim_state"],
                clipped_actions["sim_image_recorded_state"],
                clipped_actions["sim_image_sim_state"],
            )
            episode_reports.append(
                {
                    "episode": episode,
                    "seed": int(alignments[episode]["selected_seed"]),
                    "alignment": alignments[episode],
                    "recorded_vs_sim_state_mae": state_mae,
                    "recorded_vs_sim_image_mae": float(
                        (recorded_image - simulator_image).abs().mean()
                    ),
                    "dataset_reference": {
                        "raw_action": dataset_raw_action.tolist(),
                        "clipped_action": dataset_action.tolist(),
                        "raw_clip_mask": dataset_clip_mask.tolist(),
                        "raw_clipped_elements": int(dataset_clip_mask.sum()),
                        "recovery_oracle": False,
                    },
                    "policy_actions": action_reports,
                    "factorial_response": factorial,
                    "recorded_domain_error": _difference_metrics(
                        clipped_actions["recorded_image_recorded_state"],
                        dataset_action,
                    ),
                    "simulator_domain_error": _difference_metrics(
                        clipped_actions["sim_image_sim_state"],
                        dataset_action,
                    ),
                    "reset_contacts": _contact_metrics(environment),
                }
            )
        finally:
            environment.close()

    response_names = (
        "state_swap_at_recorded_image",
        "image_swap_at_recorded_state",
        "joint_recorded_to_sim_swap",
        "image_state_interaction",
    )
    median_response_l2 = {
        name: float(
            median(
                item["factorial_response"][name]["l2"]
                for item in episode_reports
            )
        )
        for name in response_names
    }
    report = {
        "schema_version": 1,
        "status": "complete",
        "diagnostic": "m2_validation_recorded_sim_image_state_factorial_probe",
        "experiment_id": experiment["experiment_id"],
        "identity": identity,
        "protocol": {
            "split": "validation",
            "episodes": list(validation_episodes),
            "frame_index": 0,
            "factors": {
                "image": ["recorded", "simulator_aligned_reset"],
                "state": ["recorded", "simulator_aligned_reset"],
            },
            "policy_execution": "inference_only_first_action",
            "comparison_space": "contract_clipped_physical_action",
        },
        "episodes": episode_reports,
        "summary": {
            "episodes_completed": len(episode_reports),
            "median_factorial_response_l2": median_response_l2,
            "median_recorded_domain_error_l2": float(
                median(item["recorded_domain_error"]["l2"] for item in episode_reports)
            ),
            "median_simulator_domain_error_l2": float(
                median(item["simulator_domain_error"]["l2"] for item in episode_reports)
            ),
        },
        "interpretation_contract": {
            "factorial_responses_are_model_sensitivity_not_causal_training_effects": True,
            "simulator_alignment_is_image_nearest_not_exact_object_pose": True,
            "dataset_reference_recovery_oracle": False,
        },
        "optimizer_steps": 0,
        "test_split_opened": False,
    }
    json.dumps(report, allow_nan=False)
    path = _write_domain_factorial_report(report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    print(f"Report: {_display_path(path)}")
    return 0


def _small_rollout_acceptance(
    metrics: dict[str, Any],
    *,
    artifact_reload_verified: bool,
) -> dict[str, bool]:
    return {
        "completed_nonempty_rollout": metrics["rollout_length"] > 0,
        "finite_actions": metrics["invalid_action_rate"] == 0,
        "raw_actions_within_contract": metrics["raw_limit_violation_rate"] == 0,
        "executed_actions_within_contract": (
            metrics["executed_limit_violation_rate"] == 0
        ),
        "joint_limits_respected": metrics["joint_limit_violations"] == 0,
        "no_unexpected_collisions": metrics["unexpected_collisions"] == 0,
        "artifact_reload_verified": artifact_reload_verified,
    }


def small_policy_rollout(
    config_path: Path,
    artifact_root: Path,
    *,
    seed: int,
    maximum_steps: int,
) -> int:
    """Gate 3: run a short observation-action-observation policy loop."""

    policy, statistics, contract, artifact_manifest, instruction = _load_online_artifact(
        config_path,
        artifact_root,
    )
    metrics = _rollout_episode(
        policy,
        statistics,
        contract,
        instruction,
        seed=seed,
        maximum_steps=maximum_steps,
    )
    criteria = _small_rollout_acceptance(
        metrics,
        artifact_reload_verified=artifact_manifest["reload"]["verified"] is True,
    )
    passed = all(criteria.values())
    report = {
        "schema_version": 2,
        "gate": "m2_gate_3_small_policy_rollout",
        "experiment_id": artifact_manifest["experiment_id"],
        "status": "passed" if passed else "failed",
        "artifact_id": artifact_manifest["artifact_id"],
        "artifact_manifest_sha256": file_sha256(artifact_root / "manifest.json"),
        "artifact_reload_verified": artifact_manifest["reload"]["verified"],
        "evaluation_code": workspace_code_identity(REPOSITORY_ROOT),
        "chunk_execution": contract.chunk_execution,
        "chunk_execution_steps": contract.chunk_execution_steps,
        "acceptance_criteria": criteria,
        "metrics": metrics,
    }
    path = _write_report("gate3", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report: {_display_path(path)}")
    return 0 if passed else 1


def _task_evaluation_acceptance(
    aggregate: dict[str, Any],
    *,
    total_steps: int,
    minimum_task_success_rate: float,
    maximum_unexpected_collisions: int,
) -> dict[str, bool]:
    if not 0 <= minimum_task_success_rate <= 1:
        raise ValueError("minimum_task_success_rate must be between zero and one.")
    if maximum_unexpected_collisions < 0:
        raise ValueError("maximum_unexpected_collisions must be non-negative.")
    return {
        "completed_nonempty_rollouts": total_steps > 0,
        "finite_actions": aggregate["invalid_action_rate"] == 0,
        "raw_actions_within_contract": aggregate["raw_limit_violation_rate"] == 0,
        "executed_actions_within_contract": aggregate["executed_limit_violation_rate"] == 0,
        "joint_limits_respected": aggregate["joint_limit_violations"] == 0,
        "minimum_task_success_rate": (
            aggregate["task_success_rate"] >= minimum_task_success_rate
        ),
        "maximum_unexpected_collisions": (
            aggregate["unexpected_collisions"] <= maximum_unexpected_collisions
        ),
    }


def task_evaluation(
    config_path: Path,
    artifact_root: Path,
    *,
    seeds: list[int],
    maximum_steps: int,
    minimum_task_success_rate: float,
    maximum_unexpected_collisions: int,
) -> int:
    """Gate 4: evaluate multiple deterministic development rollouts."""

    policy, statistics, contract, artifact_manifest, instruction = _load_online_artifact(
        config_path,
        artifact_root,
    )
    episodes: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds, start=1):
        print(f"Gate 4 episode {index}/{len(seeds)} seed={seed}", flush=True)
        episode = _rollout_episode(
            policy,
            statistics,
            contract,
            instruction,
            seed=seed,
            maximum_steps=maximum_steps,
        )
        episodes.append(episode)
        print(
            "Gate 4 episode result "
            f"seed={seed} success={episode['success']} "
            f"steps={episode['rollout_length']} "
            f"unexpected_collisions={episode['unexpected_collisions']}",
            flush=True,
        )
    total_steps = sum(episode["rollout_length"] for episode in episodes)
    total_inference_calls = sum(episode["policy_inference_calls"] for episode in episodes)
    aggregate = {
        "task_success_rate": sum(episode["success"] for episode in episodes) / len(episodes),
        "mean_rollout_length": total_steps / len(episodes),
        "invalid_action_rate": sum(
            episode["invalid_action_rate"] * max(1, episode["rollout_length"])
            for episode in episodes
        )
        / max(1, total_steps),
        "raw_limit_violation_rate": sum(
            episode["raw_limit_violation_rate"] * max(1, episode["rollout_length"])
            for episode in episodes
        )
        / max(1, total_steps),
        "executed_limit_violation_rate": sum(
            episode["executed_limit_violation_rate"] * max(1, episode["rollout_length"])
            for episode in episodes
        )
        / max(1, total_steps),
        "joint_limit_violations": sum(
            episode["joint_limit_violations"] for episode in episodes
        ),
        "unexpected_collisions": sum(
            episode["unexpected_collisions"] for episode in episodes
        ),
        "successful_episodes": sum(episode["success"] for episode in episodes),
        "maximum_reward": max(episode["maximum_reward"] for episode in episodes),
        "mean_action_smoothness_l2": sum(
            episode["action_smoothness_l2"] for episode in episodes
        )
        / len(episodes),
        "mean_policy_inference_seconds": sum(
            episode["mean_policy_inference_seconds"] * episode["policy_inference_calls"]
            for episode in episodes
        )
        / max(1, total_inference_calls),
        "policy_inference_calls": total_inference_calls,
        "amortized_policy_inference_seconds_per_step": sum(
            episode["mean_policy_inference_seconds"] * episode["policy_inference_calls"]
            for episode in episodes
        )
        / max(1, total_steps),
        "mean_simulation_step_seconds": sum(
            episode["mean_simulation_step_seconds"] for episode in episodes
        )
        / len(episodes),
    }
    criteria = _task_evaluation_acceptance(
        aggregate,
        total_steps=total_steps,
        minimum_task_success_rate=minimum_task_success_rate,
        maximum_unexpected_collisions=maximum_unexpected_collisions,
    )
    passed = all(criteria.values())
    report = {
        "schema_version": 2,
        "gate": "m2_gate_4_development_task_evaluation",
        "experiment_id": artifact_manifest["experiment_id"],
        "status": "passed" if passed else "failed",
        "artifact_id": artifact_manifest["artifact_id"],
        "artifact_manifest_sha256": file_sha256(artifact_root / "manifest.json"),
        "evaluation_code": workspace_code_identity(REPOSITORY_ROOT),
        "seeds": seeds,
        "maximum_steps": maximum_steps,
        "chunk_execution": contract.chunk_execution,
        "chunk_execution_steps": contract.chunk_execution_steps,
        "acceptance_thresholds": {
            "minimum_task_success_rate": minimum_task_success_rate,
            "maximum_unexpected_collisions": maximum_unexpected_collisions,
        },
        "acceptance_criteria": criteria,
        "safety_execution_status": (
            "passed"
            if all(
                criteria[name]
                for name in (
                    "completed_nonempty_rollouts",
                    "finite_actions",
                    "raw_actions_within_contract",
                    "executed_actions_within_contract",
                    "joint_limits_respected",
                )
            )
            else "failed"
        ),
        "task_capability_status": (
            "passed"
            if criteria["minimum_task_success_rate"]
            and criteria["maximum_unexpected_collisions"]
            else "failed"
        ),
        "aggregate": aggregate,
        "episodes": episodes,
        "collision_metric": (
            "heuristic unexpected robot self, table, or non-gripper contact; "
            "same-arm internal gripper-finger contacts excluded"
        ),
    }
    path = _write_report("gate4", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report: {_display_path(path)}")
    return 0 if report["status"] == "passed" else 1


def execution_strategy_diagnostic(
    config_path: Path,
    artifact_root: Path,
    *,
    seeds: list[int],
    maximum_steps: int,
    actions_per_inference: int,
) -> int:
    """Compare a non-gate chunk execution strategy without changing the artifact."""

    policy, statistics, contract, artifact_manifest, instruction = _load_online_artifact(
        config_path,
        artifact_root,
    )
    if not 1 <= actions_per_inference <= contract.chunk_length:
        raise ValueError("actions_per_inference must be within the artifact chunk length.")
    diagnostic_contract = replace(
        contract,
        chunk_execution=f"diagnostic_open_loop_first_{actions_per_inference}_then_reobserve",
        chunk_execution_steps=actions_per_inference,
    )
    episodes: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds, start=1):
        print(
            f"Execution diagnostic episode {index}/{len(seeds)} seed={seed}",
            flush=True,
        )
        episode = _rollout_episode(
            policy,
            statistics,
            diagnostic_contract,
            instruction,
            seed=seed,
            maximum_steps=maximum_steps,
        )
        episodes.append(episode)
        print(
            "Execution diagnostic result "
            f"seed={seed} success={episode['success']} "
            f"reward={episode['maximum_reward']} steps={episode['rollout_length']}",
            flush=True,
        )
    total_steps = sum(episode["rollout_length"] for episode in episodes)
    report = {
        "schema_version": 1,
        "status": "complete",
        "diagnostic": "non_gate_chunk_execution_strategy",
        "experiment_id": artifact_manifest["experiment_id"],
        "artifact_id": artifact_manifest["artifact_id"],
        "artifact_manifest_sha256": file_sha256(artifact_root / "manifest.json"),
        "seeds": seeds,
        "maximum_steps": maximum_steps,
        "declared_contract": {
            "chunk_execution": contract.chunk_execution,
            "chunk_execution_steps": contract.chunk_execution_steps,
            "action_contract_sha256": file_sha256(
                REPOSITORY_ROOT
                / load_experiment_config(config_path, REPOSITORY_ROOT)["action_contract"]
            ),
        },
        "diagnostic_deviation": {
            "chunk_execution": diagnostic_contract.chunk_execution,
            "chunk_execution_steps": diagnostic_contract.chunk_execution_steps,
            "warning": (
                "This deliberately deviates from the exported Action Contract and cannot be "
                "reported as Gate 3 or Gate 4 evidence."
            ),
        },
        "aggregate": {
            "task_success_rate": sum(episode["success"] for episode in episodes)
            / len(episodes),
            "successful_episodes": sum(episode["success"] for episode in episodes),
            "maximum_reward": max(episode["maximum_reward"] for episode in episodes),
            "mean_rollout_length": total_steps / len(episodes),
            "unexpected_collisions": sum(
                episode["unexpected_collisions"] for episode in episodes
            ),
            "raw_limit_violation_rate": sum(
                episode["raw_limit_violation_rate"] * episode["rollout_length"]
                for episode in episodes
            )
            / max(1, total_steps),
        },
        "episodes": episodes,
    }
    path = _write_diagnostic("execution-strategy", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report: {_display_path(path)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    gate1 = subparsers.add_parser("scripted")
    gate1.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    gate1.add_argument("--seed", type=int, default=20260809)
    gate1.add_argument("--steps-per-dimension", type=int, default=5)
    gate1.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    gate2 = subparsers.add_parser("replay")
    gate2.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    gate2.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    gate2.add_argument("--episode", type=int, default=2)
    gate2.add_argument("--maximum-steps", type=int, default=500)
    gate2.add_argument("--seed", type=int, default=10)
    gate2.add_argument("--initial-alignment-report", type=Path, required=True)
    gate2.add_argument("--maximum-alignment-mae", type=float, default=0.005)
    gate2.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    gate3 = subparsers.add_parser("small-rollout")
    gate3.add_argument("--config", type=Path, default=None)
    gate3.add_argument("--artifact", type=Path, required=True)
    gate3.add_argument("--seed", type=int, default=20260809)
    gate3.add_argument("--maximum-steps", type=int, default=20)
    gate4 = subparsers.add_parser("task-eval")
    gate4.add_argument("--config", type=Path, default=None)
    gate4.add_argument("--artifact", type=Path, required=True)
    gate4.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002, 1003, 1004])
    gate4.add_argument("--maximum-steps", type=int, default=500)
    gate4.add_argument("--minimum-task-success-rate", type=float, default=0.2)
    gate4.add_argument("--maximum-unexpected-collisions", type=int, default=0)
    execution = subparsers.add_parser("execution-diagnostic")
    execution.add_argument("--config", type=Path, default=None)
    execution.add_argument("--artifact", type=Path, required=True)
    execution.add_argument("--seeds", type=int, nargs="+", default=[1000])
    execution.add_argument("--maximum-steps", type=int, default=500)
    execution.add_argument("--actions-per-inference", type=int, default=5)
    trajectory = subparsers.add_parser("trajectory-divergence")
    trajectory.add_argument("--config", type=Path, default=None)
    trajectory.add_argument("--artifact", type=Path, required=True)
    trajectory.add_argument("--alignment-report", type=Path, required=True)
    trajectory.add_argument("--phase", choices=("smoke", "full"), required=True)
    trajectory.add_argument("--episode", type=int, required=True)
    trajectory.add_argument("--maximum-steps", type=int)
    trajectory.add_argument("--maximum-alignment-mae", type=float, default=0.005)
    trajectory.add_argument("--smoke-report", type=Path)
    decomposition = subparsers.add_parser("teacher-forced-decomposition")
    decomposition.add_argument("--config", type=Path, default=None)
    decomposition.add_argument("--artifact", type=Path, required=True)
    decomposition.add_argument("--alignment-report", type=Path, required=True)
    recorded = subparsers.add_parser("recorded-domain-probe")
    recorded.add_argument("--config", type=Path, default=None)
    recorded.add_argument("--artifact", type=Path, required=True)
    factorial = subparsers.add_parser("domain-factorial-probe")
    factorial.add_argument("--config", type=Path, default=None)
    factorial.add_argument("--artifact", type=Path, required=True)
    factorial.add_argument("--alignment-report", type=Path, required=True)
    factorial.add_argument("--recorded-images", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "scripted":
        return scripted(
            args.contract.resolve(),
            seed=args.seed,
            steps_per_dimension=args.steps_per_dimension,
            experiment_id=args.experiment_id,
        )
    if args.command == "replay":
        return replay(
            args.contract.resolve(),
            args.dataset.resolve(),
            episode=args.episode,
            maximum_steps=args.maximum_steps,
            seed=args.seed,
            initial_alignment_report=args.initial_alignment_report.resolve(),
            maximum_alignment_mae=args.maximum_alignment_mae,
            experiment_id=args.experiment_id,
        )
    config_path = (
        args.config.resolve()
        if args.config is not None
        else REPOSITORY_ROOT
        / "configs"
        / "experiments"
        / "m2_qwen08b_frozen_001.yaml"
    )
    if args.command == "small-rollout":
        return small_policy_rollout(
            config_path,
            args.artifact.resolve(),
            seed=args.seed,
            maximum_steps=args.maximum_steps,
        )
    if args.command == "trajectory-divergence":
        return trajectory_divergence(
            config_path,
            args.artifact.resolve(),
            args.alignment_report.resolve(),
            phase=args.phase,
            episode=args.episode,
            maximum_steps=args.maximum_steps,
            maximum_alignment_mae=args.maximum_alignment_mae,
            smoke_report_path=(
                None if args.smoke_report is None else args.smoke_report.resolve()
            ),
        )
    if args.command == "teacher-forced-decomposition":
        return teacher_forced_decomposition(
            config_path,
            args.artifact.resolve(),
            args.alignment_report.resolve(),
        )
    if args.command == "recorded-domain-probe":
        return recorded_domain_probe(
            config_path,
            args.artifact.resolve(),
        )
    if args.command == "domain-factorial-probe":
        return domain_factorial_probe(
            config_path,
            args.artifact.resolve(),
            args.alignment_report.resolve(),
            args.recorded_images.resolve(),
        )
    if args.command == "execution-diagnostic":
        return execution_strategy_diagnostic(
            config_path,
            args.artifact.resolve(),
            seeds=args.seeds,
            maximum_steps=args.maximum_steps,
            actions_per_inference=args.actions_per_inference,
        )
    return task_evaluation(
        config_path,
        args.artifact.resolve(),
        seeds=args.seeds,
        maximum_steps=args.maximum_steps,
        minimum_task_success_rate=args.minimum_task_success_rate,
        maximum_unexpected_collisions=args.maximum_unexpected_collisions,
    )


if __name__ == "__main__":
    raise SystemExit(main())
