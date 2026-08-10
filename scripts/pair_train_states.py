"""Create train-only simulator-state pairs for the pre-registered v010 experiment.

The command is deliberately split in two. ``export-images`` is the only path that
decodes dataset images. ``generate`` consumes that immutable artifact, reads only
train feature shards and train-filtered trajectory rows, and never opens hidden
test tensors.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rosetta_reality.data.adapters.lerobot_v3 import LeRobotV3Adapter  # noqa: E402
from rosetta_reality.data.cache_resolver import (  # noqa: E402
    ordered_feature_names,
    resolve_prepared_cache,
)
from rosetta_reality.data.config import DatasetConfig, load_dataset_config  # noqa: E402
from rosetta_reality.experiment import (  # noqa: E402
    file_sha256,
    load_experiment_config,
    stable_hash,
)
from rosetta_reality.features.cache import (  # noqa: E402
    CachedFeatureDataset,
    create_json,
    save_tensor_shard,
)
from rosetta_reality.sim.action_contract import (  # noqa: E402
    ActionContract,
    load_action_contract,
)
from rosetta_reality.sim.gym_aloha import GymAlohaEnvironment  # noqa: E402

DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "configs"
    / "experiments"
    / "m2_qwen08b_frozen_010_train_state_pairing_xpu.yaml"
)
SCHEMA_VERSION = 1
TRAIN_EPISODES = 40
CANDIDATE_SEED_START = 0
CANDIDATE_SEED_COUNT = 256
TOP_K = 5
MAXIMUM_POOLED_4X4_MAE = 0.005
MAXIMUM_RECORDED_STATE_MAE = 0.025
MINIMUM_PAIRED_SAMPLES = 1980
EXPECTED_FRAME_STRIDE = 5
EXPECTED_CHUNK_LENGTH = 8


def _run_root() -> Path:
    raw = os.environ.get("ROSETTA_RUN_ROOT")
    return Path(raw) if raw else REPOSITORY_ROOT / "runs"


def _strict_json_object(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"JSON contains a non-finite constant: {value}.")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON mapping at {path}.")
    json.dumps(value, allow_nan=False)
    return value


def _exact_train_scope(
    experiment: Mapping[str, Any],
    requested_episodes: Sequence[int] | None = None,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    """Validate the exact ordered train scope before any dataset decode or shard load."""

    split = experiment.get("dataset", {}).get("split", {})
    if not isinstance(split, Mapping):
        raise ValueError("Experiment dataset split must be a mapping.")
    raw_train = split.get("train")
    raw_validation = split.get("validation")
    raw_test = split.get("test")
    if not all(isinstance(values, list) for values in (raw_train, raw_validation, raw_test)):
        raise ValueError("Experiment train, validation, and test splits must be lists.")
    if any(
        type(value) is not int
        for values in (raw_train, raw_validation, raw_test)
        for value in values
    ):
        raise ValueError("Experiment episode identifiers must be exact integers.")
    train = tuple(raw_train)
    validation = tuple(raw_validation)
    test = tuple(raw_test)
    if len(train) != TRAIN_EPISODES or len(train) != len(set(train)):
        raise ValueError("State pairing requires exactly 40 unique train episodes.")
    if len(validation) != len(set(validation)) or len(test) != len(set(test)):
        raise ValueError("Validation and test episode lists must not contain duplicates.")
    if set(train) & (set(validation) | set(test)) or set(validation) & set(test):
        raise ValueError("Train, validation, and hidden-test episode scopes overlap.")
    requested = train if requested_episodes is None else tuple(requested_episodes)
    if any(type(value) is not int for value in requested):
        raise ValueError("Requested train episodes must be exact integers.")
    if requested != train:
        raise ValueError("Only the exact ordered train split may be used for state pairing.")
    return train, {
        "split": "train",
        "episodes": list(train),
        "validation_split_opened": False,
        "test_split_opened": False,
    }


def _dataset_context(
    experiment: Mapping[str, Any],
) -> tuple[Path, DatasetConfig, Path, Any, Path, ActionContract]:
    dataset_relative = Path(str(experiment["dataset"]["config"]))
    contract_relative = Path(str(experiment["action_contract"]))
    if (
        dataset_relative.is_absolute()
        or ".." in dataset_relative.parts
        or contract_relative.is_absolute()
        or ".." in contract_relative.parts
    ):
        raise ValueError("Dataset and Action Contract paths must be repository-relative.")
    dataset_path = REPOSITORY_ROOT / dataset_relative
    contract_path = REPOSITORY_ROOT / contract_relative
    dataset_config = load_dataset_config(dataset_path)
    # Full-cache checksum validation can read rows outside the selected split. Identity is
    # instead pinned by the prepared manifest and the visible feature manifest below.
    dataset_root, dataset_manifest = resolve_prepared_cache(
        dataset_config,
        REPOSITORY_ROOT,
        validate_checksums=False,
    )
    contract = load_action_contract(contract_path)
    contract.validate_order(ordered_feature_names(dataset_root, dataset_config.fields.action))
    contract.validate_order(ordered_feature_names(dataset_root, dataset_config.fields.state))
    if (
        contract.dimension != dataset_config.expected_action_dim
        or contract.dimension != dataset_config.expected_state_dim
    ):
        raise ValueError("Dataset state/action widths and Action Contract dimension differ.")
    if (
        contract.chunk_length != EXPECTED_CHUNK_LENGTH
        or dataset_config.chunk_size != EXPECTED_CHUNK_LENGTH
        or contract.chunk_execution_steps != 1
    ):
        raise ValueError("v010 requires an eight-action chunk and first-action execution.")
    return (
        dataset_path,
        dataset_config,
        dataset_root,
        dataset_manifest,
        contract_path,
        contract,
    )


def _decode_initial_images(
    *,
    root: Path,
    revision: str,
    dataset_config: DatasetConfig,
    episodes: tuple[int, ...],
) -> dict[int, torch.Tensor]:
    adapter = LeRobotV3Adapter(
        repo_id=dataset_config.repo_id,
        revision=revision,
        root=root,
        episodes=episodes,
        cameras=dataset_config.cameras,
        fields=dataset_config.fields,
        embodiment=dataset_config.embodiment,
        license_name=dataset_config.license,
    )
    frame_zero_indices: dict[int, int] = {}
    for index in range(len(adapter)):
        reference = adapter.frame_reference(index)
        episode = int(reference.episode_id)
        if episode not in set(episodes):
            raise ValueError("Train-scoped image adapter exposed a non-train episode.")
        if int(reference.frame_index) == 0 and episode not in frame_zero_indices:
            frame_zero_indices[episode] = index
        if len(frame_zero_indices) == len(episodes):
            break
    missing = [episode for episode in episodes if episode not in frame_zero_indices]
    if missing:
        raise ValueError(f"Dataset has no frame-zero image for train episodes: {missing}.")
    camera = next(iter(dataset_config.cameras))
    return {episode: adapter[frame_zero_indices[episode]].images[camera] for episode in episodes}


def _same_tensor_payload(actual: Any, expected: dict[str, Any]) -> bool:
    if not isinstance(actual, dict) or set(actual) != set(expected):
        return False
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, torch.Tensor):
            if not isinstance(actual_value, torch.Tensor) or not torch.equal(
                actual_value, expected_value
            ):
                return False
        elif actual_value != expected_value:
            return False
    return True


def _create_tensor_once(path: Path, payload: dict[str, Any]) -> None:
    if path.is_symlink():
        raise FileExistsError(f"Refusing to use a symlink as a tensor artifact: {path}.")
    if path.exists():
        existing = torch.load(path, map_location="cpu", weights_only=True)
        if not _same_tensor_payload(existing, payload):
            raise FileExistsError(f"Refusing to overwrite different tensor content at {path}.")
        return
    save_tensor_shard(path, payload)


def export_images(
    config_path: Path = DEFAULT_CONFIG,
    *,
    requested_episodes: Sequence[int] | None = None,
) -> Path:
    """Create immutable per-episode frame-zero images for the exact train split."""

    config_path = config_path.resolve()
    experiment = load_experiment_config(config_path, REPOSITORY_ROOT)
    # This guard intentionally precedes dataset config loading, cache discovery, and decode.
    episodes, scope = _exact_train_scope(experiment, requested_episodes)
    (
        _dataset_path,
        dataset_config,
        dataset_root,
        dataset_manifest,
        _contract_path,
        _contract,
    ) = _dataset_context(experiment)
    if not set(episodes).issubset(dataset_config.episodes):
        raise ValueError("Train episodes are outside the pinned dataset selection.")
    identity = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "train_state_pairing_initial_images_v1",
        "experiment_id": experiment["experiment_id"],
        "experiment_config_sha256": file_sha256(config_path),
        "dataset_repo_id": dataset_config.repo_id,
        "dataset_revision": dataset_manifest.resolved_revision,
        "dataset_manifest_sha256": file_sha256(dataset_root / "manifest.json"),
        "camera": next(iter(dataset_config.cameras)),
        "decoder": "LeRobotDataset video_backend=pyav",
        "scope": scope,
    }
    identity_hash = stable_hash(identity)
    destination = (
        _run_root()
        / str(experiment["experiment_id"])
        / "state-pairing"
        / "initial-images"
        / identity_hash[:16]
    )
    if destination.exists() and not destination.is_dir():
        raise FileExistsError("Initial-image artifact destination is not a directory.")

    images = _decode_initial_images(
        root=dataset_root,
        revision=dataset_manifest.resolved_revision,
        dataset_config=dataset_config,
        episodes=episodes,
    )
    files: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        image = images[episode].detach().to(torch.float32).cpu().contiguous()
        if image.ndim != 3 or not bool(torch.isfinite(image).all()):
            raise ValueError(f"Initial image for train episode {episode} is invalid.")
        filename = f"episode-{episode:06d}.pt"
        path = destination / filename
        _create_tensor_once(
            path,
            {
                "schema_version": SCHEMA_VERSION,
                "identity_hash": identity_hash,
                "episode": episode,
                "image": image,
            },
        )
        files[str(episode)] = {
            "path": filename,
            "sha256": file_sha256(path),
            "shape": list(image.shape),
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "identity": identity,
        "identity_hash": identity_hash,
        "scope": scope,
        "files": files,
    }
    manifest_path = destination / "manifest.json"
    if manifest_path.is_symlink():
        raise FileExistsError("Refusing to use a symlink as an initial-image manifest.")
    create_json(manifest_path, manifest)
    expected_names = {"manifest.json"} | {f"episode-{episode:06d}.pt" for episode in episodes}
    if {entry.name for entry in destination.iterdir()} != expected_names:
        raise ValueError("Initial-image artifact contains undeclared files.")
    print(
        json.dumps(
            {
                "status": "complete",
                "artifact": (
                    Path("runs")
                    / str(experiment["experiment_id"])
                    / "state-pairing"
                    / "initial-images"
                    / identity_hash[:16]
                ).as_posix(),
                "identity_hash": identity_hash,
                "episodes": list(episodes),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return destination


def _validate_visible_feature_manifest(
    path: Path,
    *,
    experiment: Mapping[str, Any],
    config_path: Path,
    episodes: tuple[int, ...],
    contract_path: Path,
) -> dict[str, Any]:
    manifest = _strict_json_object(path)
    identity = manifest.get("identity")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or not isinstance(identity, dict)
        or manifest.get("identity_hash") != stable_hash(identity)
    ):
        raise ValueError("Feature manifest is not a complete content-addressed cache.")
    split = identity.get("split")
    selection = identity.get("selection")
    derivation = manifest.get("derivation")
    feature_derivation = identity.get("feature", {}).get("derivation")
    if (
        identity.get("experiment_id") != experiment["experiment_id"]
        or identity.get("experiment_config_sha256") != file_sha256(config_path)
        or split != experiment["dataset"]["split"]
        or not isinstance(selection, dict)
        or int(selection.get("frame_stride", -1)) != EXPECTED_FRAME_STRIDE
        or int(selection.get("action_chunk_length", -1)) != EXPECTED_CHUNK_LENGTH
        or identity.get("action_contract_sha256") != file_sha256(contract_path)
        or not isinstance(derivation, dict)
        or derivation.get("type") != "verified_visible_identity_rebind_v1"
        or derivation != feature_derivation
        or derivation.get("materialized_splits") != ["train", "validation"]
        or derivation.get("withheld_splits") != ["test"]
        or derivation.get("hidden_test_loaded") is not False
        or manifest.get("hidden_test_loaded") is not False
        or manifest.get("hidden_test_materialized") is not False
        or manifest.get("materialized_splits") != ["train", "validation"]
        or manifest.get("withheld_splits") != ["test"]
        or manifest.get("samples", {}).get("test") != 0
    ):
        raise ValueError("Feature manifest is not the exact v010 visible-only cache.")
    raw_shards = manifest.get("shards")
    if not isinstance(raw_shards, dict) or raw_shards.get("test") != []:
        raise ValueError("Visible feature manifest must withhold all hidden-test shards.")
    train_records = raw_shards.get("train")
    if not isinstance(train_records, list) or not train_records:
        raise ValueError("Visible feature manifest contains no train shards.")
    record_episodes: list[int] = []
    for record in train_records:
        if not isinstance(record, dict) or type(record.get("episode")) is not int:
            raise ValueError("Train feature shard record is invalid.")
        episode = int(record["episode"])
        relative = Path(str(record.get("path", "")))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or len(relative.parts) != 3
            or relative.parts[:2] != ("shards", "train")
        ):
            raise ValueError("Train feature shard path escapes the train-only namespace.")
        record_episodes.append(episode)
    if len(record_episodes) != len(set(record_episodes)) or set(record_episodes) != set(episodes):
        raise ValueError("Train feature shards do not exactly cover the train split.")
    return manifest


def _load_initial_images(
    path: Path,
    *,
    experiment: Mapping[str, Any],
    config_path: Path,
    scope: dict[str, Any],
    dataset_config: DatasetConfig,
    dataset_revision: str,
    dataset_manifest_sha256: str,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    expected_parent = (
        _run_root() / str(experiment["experiment_id"]) / "state-pairing" / "initial-images"
    ).resolve()
    resolved = path.resolve()
    if resolved.parent != expected_parent or not resolved.is_dir() or resolved.is_symlink():
        raise ValueError("Initial-image artifact is outside the fixed train-only run path.")
    manifest_path = resolved / "manifest.json"
    manifest = _strict_json_object(manifest_path)
    identity = manifest.get("identity")
    expected_identity = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "train_state_pairing_initial_images_v1",
        "experiment_id": experiment["experiment_id"],
        "experiment_config_sha256": file_sha256(config_path),
        "dataset_repo_id": dataset_config.repo_id,
        "dataset_revision": dataset_revision,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "camera": next(iter(dataset_config.cameras)),
        "decoder": "LeRobotDataset video_backend=pyav",
        "scope": scope,
    }
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or identity != expected_identity
        or manifest.get("identity_hash") != stable_hash(expected_identity)
        or manifest.get("scope") != scope
        or resolved.name != stable_hash(expected_identity)[:16]
    ):
        raise ValueError("Initial-image artifact identity or train-only scope differs.")
    episodes = tuple(scope["episodes"])
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {str(value) for value in episodes}:
        raise ValueError("Initial-image records do not exactly cover the train split.")
    expected_names = {"manifest.json"} | {f"episode-{episode:06d}.pt" for episode in episodes}
    if {entry.name for entry in resolved.iterdir()} != expected_names:
        raise ValueError("Initial-image artifact contains undeclared files.")
    images: dict[int, torch.Tensor] = {}
    for episode in episodes:
        filename = f"episode-{episode:06d}.pt"
        record = files[str(episode)]
        if not isinstance(record, dict) or record.get("path") != filename:
            raise ValueError(f"Initial-image path differs for train episode {episode}.")
        shard_path = resolved / filename
        if shard_path.is_symlink() or file_sha256(shard_path) != record.get("sha256"):
            raise ValueError(f"Initial-image checksum differs for train episode {episode}.")
        shard = torch.load(shard_path, map_location="cpu", weights_only=True)
        if (
            not isinstance(shard, dict)
            or shard.get("schema_version") != SCHEMA_VERSION
            or shard.get("identity_hash") != manifest["identity_hash"]
            or shard.get("episode") != episode
        ):
            raise ValueError(f"Initial-image identity differs for train episode {episode}.")
        image = shard.get("image")
        if (
            not isinstance(image, torch.Tensor)
            or image.ndim != 3
            or list(image.shape) != record.get("shape")
            or not bool(torch.isfinite(image).all())
        ):
            raise ValueError(f"Initial image is invalid for train episode {episode}.")
        images[episode] = image.to(torch.float32).cpu()
    return images, {
        "identity_hash": manifest["identity_hash"],
        "manifest_sha256": file_sha256(manifest_path),
    }


def _alignment_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    if reference.ndim != 3 or reference.shape != candidate.shape:
        raise ValueError("Image alignment requires matching [channel, height, width] tensors.")
    if min(reference.shape[-2:]) < 4:
        raise ValueError("Image alignment requires spatial dimensions of at least four pixels.")
    reference = reference.to(torch.float32)
    candidate = candidate.to(torch.float32)
    if not bool(torch.isfinite(reference).all()) or not bool(torch.isfinite(candidate).all()):
        raise ValueError("Image alignment received NaN or Inf.")
    difference = reference - candidate
    pooled_reference = torch.nn.functional.avg_pool2d(reference.unsqueeze(0), kernel_size=4)
    pooled_candidate = torch.nn.functional.avg_pool2d(candidate.unsqueeze(0), kernel_size=4)
    result = {
        "pixel_mae": float(difference.abs().mean()),
        "pixel_rmse": float(difference.square().mean().sqrt()),
        "pooled_4x4_mae": float((pooled_reference - pooled_candidate).abs().mean()),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Image alignment produced a non-finite metric.")
    return result


def _search_initial_seed(
    environment: Any,
    reference_image: torch.Tensor,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for seed in range(
        CANDIDATE_SEED_START,
        CANDIDATE_SEED_START + CANDIDATE_SEED_COUNT,
    ):
        observation = environment.reset(seed=seed)
        images = observation.get("images")
        if not isinstance(images, Mapping) or len(images) != 1:
            raise ValueError("Seed alignment requires exactly one simulator camera.")
        candidate = next(iter(images.values()))
        if not isinstance(candidate, torch.Tensor):
            raise ValueError("Simulator camera did not return a tensor.")
        candidates.append({"seed": seed, **_alignment_metrics(reference_image, candidate)})
    candidates.sort(key=lambda value: (value["pooled_4x4_mae"], value["seed"]))
    top = candidates[:TOP_K]
    if len(top) != TOP_K or top[0]["pooled_4x4_mae"] > MAXIMUM_POOLED_4X4_MAE:
        raise ValueError("No reset seed satisfies the pre-registered pooled 4x4 MAE limit.")
    return top


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
        raise ValueError(f"Train episode {episode} contains no trajectory rows.")
    for expected_frame, row in enumerate(rows):
        if (
            type(row[fields.episode_index]) is not int
            or int(row[fields.episode_index]) != episode
            or type(row[fields.frame_index]) is not int
            or int(row[fields.frame_index]) != expected_frame
        ):
            raise ValueError("Train trajectory rows are not exact contiguous episode frames.")
        timestamp = float(row[fields.timestamp])
        expected_timestamp = expected_frame / contract.frequency_hz
        if not math.isfinite(timestamp) or not math.isclose(
            timestamp, expected_timestamp, rel_tol=0.0, abs_tol=1e-4
        ):
            raise ValueError("Train trajectory timestamps violate the Action Contract frequency.")
        state = torch.as_tensor(row[fields.state], dtype=torch.float32)
        action = torch.as_tensor(row[fields.action], dtype=torch.float32)
        contract.validate_tensor(action, allow_chunk=False)
        if state.shape != (contract.dimension,) or not bool(torch.isfinite(state).all()):
            raise ValueError("Train trajectory state is invalid.")
    return rows


def _replay_episode(
    environment: Any,
    contract: ActionContract,
    rows: Sequence[Mapping[str, Any]],
    *,
    action_field: str,
    state_field: str,
    frame_field: str,
    seed: int,
) -> dict[str, Any]:
    """Capture state_t before action_t and return a half-open valid prefix."""

    observation = environment.reset(seed=seed)
    states: dict[int, torch.Tensor] = {}
    state_maes: dict[int, float] = {}
    cutoff = len(rows)
    cutoff_reason = "trajectory_end"
    executed_steps = 0
    for position, row in enumerate(rows):
        frame = int(row[frame_field])
        if frame != position:
            raise ValueError("Replay requires contiguous zero-based frame indices.")
        simulator_state = (
            torch.as_tensor(observation.get("robot_state"), dtype=torch.float32).detach().cpu()
        )
        recorded_state = torch.as_tensor(row[state_field], dtype=torch.float32).cpu()
        if (
            simulator_state.shape != recorded_state.shape
            or simulator_state.shape != (contract.dimension,)
            or not bool(torch.isfinite(simulator_state).all())
            or not bool(torch.isfinite(recorded_state).all())
        ):
            raise ValueError("Recorded and simulator states violate the state contract.")
        state_mae = float((recorded_state - simulator_state).abs().mean())
        if not math.isfinite(state_mae):
            raise ValueError("Recorded-vs-simulator state MAE is non-finite.")
        if state_mae > MAXIMUM_RECORDED_STATE_MAE:
            cutoff = frame
            cutoff_reason = "recorded_state_mae"
            break
        # This clone is intentionally captured before contract-clipped action_t is stepped.
        states[frame] = simulator_state.clone()
        state_maes[frame] = state_mae
        raw_action = torch.as_tensor(row[action_field], dtype=torch.float32)
        contract.validate_tensor(raw_action, allow_chunk=False)
        clipped_action, _clip_mask = contract.clip(raw_action)
        observation, _reward, done, _info = environment.step(clipped_action)
        executed_steps += 1
        if bool(done):
            # action_t completed, so state_t is inside the half-open [0, t + 1) prefix.
            cutoff = frame + 1
            cutoff_reason = "done"
            break
    return {
        "states": states,
        "state_maes": state_maes,
        "exclusive_cutoff": cutoff,
        "cutoff_reason": cutoff_reason,
        "executed_steps": executed_steps,
    }


def _assemble_pairing(
    *,
    episode_ids: torch.Tensor,
    frame_indices: torch.Tensor,
    recorded_states: torch.Tensor,
    replays: Mapping[int, Mapping[str, Any]],
    train_episodes: Sequence[int],
    frame_stride: int = EXPECTED_FRAME_STRIDE,
    chunk_length: int = EXPECTED_CHUNK_LENGTH,
) -> tuple[torch.Tensor, torch.Tensor, str, dict[int, int]]:
    if (
        episode_ids.ndim != 1
        or frame_indices.shape != episode_ids.shape
        or recorded_states.ndim != 2
        or recorded_states.shape[0] != episode_ids.shape[0]
        or not bool(torch.isfinite(recorded_states).all())
    ):
        raise ValueError("Feature-cache train tensors violate the pairing contract.")
    paired = recorded_states.detach().to(torch.float32).cpu().clone()
    mask = torch.zeros(episode_ids.shape[0], dtype=torch.bool)
    eligible_keys: list[list[int]] = []
    counts = {int(episode): 0 for episode in train_episodes}
    seen_keys: set[tuple[int, int]] = set()
    for index, (episode_value, frame_value) in enumerate(zip(episode_ids, frame_indices)):
        episode = int(episode_value)
        frame = int(frame_value)
        key = (episode, frame)
        if key in seen_keys or episode not in counts:
            raise ValueError("Feature-cache train sample keys are duplicate or out of scope.")
        seen_keys.add(key)
        replay = replays.get(episode)
        if not isinstance(replay, Mapping):
            raise ValueError(f"Train episode {episode} has no replay result.")
        cutoff = int(replay.get("exclusive_cutoff", -1))
        states = replay.get("states")
        if not isinstance(states, Mapping) or cutoff < 0:
            raise ValueError("Replay result lacks a valid half-open state prefix.")
        eligible = (
            frame % frame_stride == 0
            and frame + chunk_length <= cutoff
            and all(step in states for step in range(frame, frame + chunk_length))
        )
        if not eligible:
            continue
        simulator_state = states[frame]
        if (
            not isinstance(simulator_state, torch.Tensor)
            or simulator_state.shape != recorded_states[index].shape
            or not bool(torch.isfinite(simulator_state).all())
        ):
            raise ValueError("Eligible simulator state is invalid.")
        paired[index] = simulator_state.to(torch.float32).cpu()
        mask[index] = True
        counts[episode] += 1
        eligible_keys.append([episode, frame])
    return paired, mask, stable_hash(eligible_keys), counts


def _validate_recorded_cache_states(
    train: CachedFeatureDataset,
    rows_by_episode: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    state_field: str,
) -> None:
    for index, (episode_value, frame_value) in enumerate(
        zip(train.episode_ids, train.frame_indices)
    ):
        episode = int(episode_value)
        frame = int(frame_value)
        rows = rows_by_episode.get(episode)
        if rows is None or not 0 <= frame < len(rows):
            raise ValueError("Feature-cache train key has no matching train trajectory row.")
        recorded = torch.as_tensor(rows[frame][state_field], dtype=torch.float32)
        if not torch.equal(recorded, train.robot_state[index].to(torch.float32).cpu()):
            raise ValueError("Feature-cache recorded state differs from the pinned train row.")


def _output_protocol(
    *,
    eligible_key_sha256: str,
    image_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "type": "aligned_expert_replay_simulator_state_pairing_v1",
        "label_type": "time_indexed_expert_reference",
        "state_conditioned": False,
        "recovery_oracle": False,
        "pre_action_state": True,
        "candidate_seed_start": CANDIDATE_SEED_START,
        "candidate_seed_count": CANDIDATE_SEED_COUNT,
        "top_k": TOP_K,
        "maximum_pooled_4x4_mae": MAXIMUM_POOLED_4X4_MAE,
        "maximum_recorded_state_mae": MAXIMUM_RECORDED_STATE_MAE,
        "minimum_paired_samples": MINIMUM_PAIRED_SAMPLES,
        "eligible_key_sha256": eligible_key_sha256,
        "frame_stride": EXPECTED_FRAME_STRIDE,
        "action_chunk_length": EXPECTED_CHUNK_LENGTH,
        "cutoff_interval": "half_open",
        "initial_image_identity_hash": image_identity["identity_hash"],
        "initial_image_manifest_sha256": image_identity["manifest_sha256"],
    }


def _ordered_episode_reports(
    *,
    train_episodes: Sequence[int],
    episode_ids: torch.Tensor,
    frame_indices: torch.Tensor,
    pairing_mask: torch.Tensor,
    replays: Mapping[int, Mapping[str, Any]],
    reset_reports: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for episode in train_episodes:
        selection = episode_ids.eq(int(episode))
        source_frames = frame_indices[selection].to(torch.long).tolist()
        eligible_frames = frame_indices[selection & pairing_mask].to(torch.long).tolist()
        replay = replays.get(int(episode))
        reset = reset_reports.get(int(episode))
        if not isinstance(replay, Mapping) or not isinstance(reset, Mapping):
            raise ValueError(f"Train episode {episode} lacks replay provenance.")
        if not source_frames or not eligible_frames:
            raise ValueError(f"Train episode {episode} lacks source or paired anchors.")
        if eligible_frames != source_frames[: len(eligible_frames)] or any(
            current - previous != EXPECTED_FRAME_STRIDE
            for previous, current in zip(eligible_frames[:-1], eligible_frames[1:], strict=True)
        ):
            raise ValueError("Eligible frame indices do not form the declared train prefix.")
        exclusive_stop = int(replay.get("exclusive_cutoff", -1))
        if eligible_frames[-1] + EXPECTED_CHUNK_LENGTH > exclusive_stop:
            raise ValueError("Eligible train prefix extends beyond its exclusive replay cutoff.")
        reports.append(
            {
                "episode": int(episode),
                "source_anchor_count": len(source_frames),
                "paired_anchor_count": len(eligible_frames),
                "eligible_frame_indices": eligible_frames,
                "exclusive_valid_step_stop": exclusive_stop,
                "selected_seed": int(reset["selected_seed"]),
                "top_candidates": reset["top_candidates"],
                "cutoff_reason": replay["cutoff_reason"],
                "executed_steps": int(replay["executed_steps"]),
            }
        )
    return reports


def generate(
    *,
    config_path: Path,
    feature_manifest_path: Path,
    initial_images_path: Path,
    environment_factory: Any = GymAlohaEnvironment,
) -> Path:
    """Generate and publish the complete immutable train-only pairing artifact."""

    config_path = config_path.resolve()
    feature_manifest_path = feature_manifest_path.resolve()
    experiment = load_experiment_config(config_path, REPOSITORY_ROOT)
    # No feature shard, dataset row, or image shard is touched before this scope gate.
    episodes, scope = _exact_train_scope(experiment)
    (
        _dataset_path,
        dataset_config,
        dataset_root,
        dataset_manifest,
        contract_path,
        contract,
    ) = _dataset_context(experiment)
    if int(experiment["dataset"]["frame_stride"]) != EXPECTED_FRAME_STRIDE:
        raise ValueError("v010 state pairing requires frame_stride=5.")
    feature_manifest = _validate_visible_feature_manifest(
        feature_manifest_path,
        experiment=experiment,
        config_path=config_path,
        episodes=episodes,
        contract_path=contract_path,
    )
    feature_dataset = feature_manifest["identity"].get("dataset", {})
    dataset_manifest_sha256 = file_sha256(dataset_root / "manifest.json")
    if (
        feature_dataset.get("revision") != dataset_manifest.resolved_revision
        or feature_dataset.get("manifest_sha256") != dataset_manifest_sha256
        or feature_dataset.get("repo_id") != dataset_config.repo_id
    ):
        raise ValueError("Feature cache and prepared dataset identities differ.")
    images, image_identity = _load_initial_images(
        initial_images_path,
        experiment=experiment,
        config_path=config_path,
        scope=scope,
        dataset_config=dataset_config,
        dataset_revision=dataset_manifest.resolved_revision,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )

    # The only feature payloads loaded by this command are the validated train shards.
    train = CachedFeatureDataset(feature_manifest_path, "train")
    rows_by_episode = {
        episode: _trajectory_rows(dataset_root, episode, dataset_config, contract)
        for episode in episodes
    }
    _validate_recorded_cache_states(
        train,
        rows_by_episode,
        state_field=dataset_config.fields.state,
    )
    environment = environment_factory(
        contract,
        maximum_episode_steps=max(len(rows) for rows in rows_by_episode.values()),
    )
    replays: dict[int, dict[str, Any]] = {}
    reset_reports: dict[int, dict[str, Any]] = {}
    try:
        for episode in episodes:
            top = _search_initial_seed(environment, images[episode])
            best = top[0]
            replay = _replay_episode(
                environment,
                contract,
                rows_by_episode[episode],
                action_field=dataset_config.fields.action,
                state_field=dataset_config.fields.state,
                frame_field=dataset_config.fields.frame_index,
                seed=int(best["seed"]),
            )
            replays[episode] = replay
            reset_reports[episode] = {
                "selected_seed": int(best["seed"]),
                "top_candidates": top,
            }
    finally:
        environment.close()

    paired, mask, eligible_digest, counts = _assemble_pairing(
        episode_ids=train.episode_ids,
        frame_indices=train.frame_indices,
        recorded_states=train.robot_state,
        replays=replays,
        train_episodes=episodes,
    )
    paired_count = int(mask.sum())
    if paired_count < MINIMUM_PAIRED_SAMPLES:
        raise ValueError(
            f"Paired train coverage {paired_count} is below {MINIMUM_PAIRED_SAMPLES}; "
            "no complete artifact was published."
        )
    missing_coverage = [episode for episode in episodes if counts[episode] < 1]
    if missing_coverage:
        raise ValueError(
            f"Train episodes lack an eligible paired prefix: {missing_coverage}; "
            "no complete artifact was published."
        )
    if not torch.equal(paired[~mask], train.robot_state[~mask].to(torch.float32).cpu()):
        raise RuntimeError("Noneligible samples did not retain the recorded-state fallback.")
    episode_reports = _ordered_episode_reports(
        train_episodes=episodes,
        episode_ids=train.episode_ids,
        frame_indices=train.frame_indices,
        pairing_mask=mask,
        replays=replays,
        reset_reports=reset_reports,
    )
    protocol = _output_protocol(
        eligible_key_sha256=eligible_digest,
        image_identity=image_identity,
    )
    identity = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment["experiment_id"],
        "experiment_config_sha256": file_sha256(config_path),
        "feature_cache_identity": feature_manifest["identity_hash"],
        "feature_manifest_sha256": file_sha256(feature_manifest_path),
        "dataset_revision": dataset_manifest.resolved_revision,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "action_contract_sha256": file_sha256(contract_path),
        "scope": scope,
        "protocol": protocol,
    }
    identity_hash = stable_hash(identity)
    payload = {
        "identity_hash": identity_hash,
        "feature_cache_identity": feature_manifest["identity_hash"],
        "paired_robot_state": paired,
        "pairing_mask": mask,
        "episode_ids": train.episode_ids.to(torch.long).cpu().clone(),
        "frame_indices": train.frame_indices.to(torch.long).cpu().clone(),
    }
    if any(
        not bool(torch.isfinite(value).all())
        for value in (
            payload["paired_robot_state"],
            payload["episode_ids"],
            payload["frame_indices"],
        )
    ):
        raise ValueError("State-pairing tensor payload contains NaN or Inf.")
    destination = _run_root() / str(experiment["experiment_id"]) / "state-pairing"
    tensor_path = destination / "paired-states.pt"
    _create_tensor_once(tensor_path, payload)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "identity": identity,
        "identity_hash": identity_hash,
        "scope": scope,
        "protocol": protocol,
        "samples": {
            "total": len(train),
            "paired": paired_count,
            "unpaired": len(train) - paired_count,
            "paired_per_episode": {str(key): counts[key] for key in episodes},
        },
        "tensor": {
            "path": "paired-states.pt",
            "sha256": file_sha256(tensor_path),
            "dtype": str(paired.dtype).removeprefix("torch."),
            "shape": list(paired.shape),
        },
        "episodes": episode_reports,
    }
    json.dumps(manifest, allow_nan=False)
    manifest_path = destination / "manifest.json"
    if manifest_path.is_symlink():
        raise FileExistsError("Refusing to use a symlink as a state-pairing manifest.")
    create_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": "complete",
                "manifest": (
                    Path("runs")
                    / str(experiment["experiment_id"])
                    / "state-pairing"
                    / "manifest.json"
                ).as_posix(),
                "identity_hash": identity_hash,
                "samples": manifest["samples"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return manifest_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser(
        "export-images", description="Export exact train-split frame-zero images."
    )
    export_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    generate_parser = subparsers.add_parser(
        "generate", description="Generate the complete v010 train state-pairing artifact."
    )
    generate_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    generate_parser.add_argument("--feature-manifest", type=Path, required=True)
    generate_parser.add_argument("--initial-images", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "export-images":
        export_images(args.config)
        return 0
    generate(
        config_path=args.config,
        feature_manifest_path=args.feature_manifest,
        initial_images_path=args.initial_images,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
