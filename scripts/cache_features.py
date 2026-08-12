"""Build or inspect an immutable, train-normalized frozen Qwen feature cache."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
DEFAULT_CONFIG = (
    REPOSITORY_ROOT / "configs" / "experiments" / "m2_qwen08b_frozen_001.yaml"
)
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rosetta_reality.data import (  # noqa: E402
    ActionChunkDataset,
    ordered_feature_names,
    resolve_prepared_cache,
)
from rosetta_reality.data.adapters import LeRobotV3Adapter  # noqa: E402
from rosetta_reality.data.config import load_dataset_config  # noqa: E402
from rosetta_reality.data.normalization import (  # noqa: E402
    DatasetStatistics,
    RunningMoments,
)
from rosetta_reality.experiment import (  # noqa: E402
    file_sha256,
    load_experiment_config,
    stable_hash,
    workspace_code_identity,
)
from rosetta_reality.features import (  # noqa: E402
    create_json,
    load_feature_manifest,
    save_tensor_shard,
)
from rosetta_reality.models.backbones.qwen35 import Qwen35Backbone  # noqa: E402
from rosetta_reality.sim import ActionContract, load_action_contract  # noqa: E402

COMPOSITION_POOLINGS = ("attention_masked_mean", "image_spatial_2x2")
COMBINED_POOLING = "attention_masked_mean_plus_image_spatial_2x2"
SHARD_TENSOR_KEYS = (
    "features",
    "robot_state",
    "actions",
    "episode_ids",
    "frame_indices",
)
VISIBLE_MATERIALIZED_SPLITS = ("train", "validation")
VISIBLE_WITHHELD_SPLITS = ("test",)


def _feature_root() -> Path:
    configured = os.environ.get("ROSETTA_FEATURE_ROOT")
    return Path(configured) if configured else REPOSITORY_ROOT / "feature_cache"


def _torch_device() -> torch.device:
    configured = os.environ.get("ROSETTA_TORCH_DEVICE", "cpu")
    device = torch.device(configured)
    if device.type == "xpu" and (not hasattr(torch, "xpu") or not torch.xpu.is_available()):
        raise RuntimeError("Frozen feature extraction requested XPU, but XPU is unavailable.")
    if device.type not in {"cpu", "xpu"}:
        raise ValueError(f"Unsupported frozen feature device: {device.type}.")
    return device


def _selected_model_files(root: Path, manifest_files: dict[str, Any]) -> list[Path]:
    """Resolve every manifest-declared file that can contribute to model identity."""

    paths: list[Path] = []
    for relative_text in sorted(manifest_files):
        relative = Path(relative_text)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != relative_text
        ):
            raise ValueError(f"Local Base-model manifest path is unsafe: {relative_text!r}.")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Manifest-declared model file is missing: {relative_text}.")
        paths.append(path)
    names = set(manifest_files)
    if "config.json" not in names or not any(name.endswith(".safetensors") for name in names):
        raise FileNotFoundError("Local model manifest is missing config or weight files.")
    return paths


def _model_identity(root: Path, configured: dict[str, Any]) -> dict[str, Any]:
    manifest_path = root / str(configured["manifest"])
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Local model is missing its revision-scoped model_manifest.json; "
            "refusing to infer Base identity from tensor shape or directory name."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "validated"
        or manifest.get("source") != "huggingface"
        or manifest.get("repo_id") != configured["identifier"]
    ):
        raise ValueError("Local model manifest does not identify the configured Base model.")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    text_config = config.get("text_config", config)
    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, dict) or not manifest_files:
        raise ValueError("Local model manifest does not contain file checksums.")
    files = _selected_model_files(root, manifest_files)
    file_hashes: dict[str, str] = {}
    for path in files:
        relative = path.relative_to(root).as_posix()
        record = manifest_files.get(relative)
        digest = file_sha256(path)
        if (
            not isinstance(record, dict)
            or record.get("sha256") != digest
            or record.get("bytes") != path.stat().st_size
        ):
            raise ValueError(f"Local Base-model manifest checksum mismatch: {relative}.")
        file_hashes[relative] = digest
    if int(manifest.get("model_contract", {}).get("hidden_size", -1)) != int(
        text_config["hidden_size"]
    ):
        raise ValueError("Local Base-model manifest hidden size differs from config.json.")
    return {
        "source": manifest["source"],
        "family": configured["family"],
        "identifier": configured["identifier"],
        "scale": configured["scale"],
        "adaptation": configured["adaptation"],
        "revision": manifest["revision"],
        "hidden_size": int(text_config["hidden_size"]),
        "manifest_sha256": file_sha256(manifest_path),
        "files": file_hashes,
    }


def _prepared_identity(root: Path, manifest: Any) -> dict[str, Any]:
    cleaning_path = root / "cleaning_report.json"
    if not cleaning_path.is_file():
        raise FileNotFoundError("Dataset cleaning_report.json is required before feature caching.")
    cleaning = json.loads(cleaning_path.read_text(encoding="utf-8"))
    if cleaning.get("status") != "validated_clean":
        raise ValueError("Dataset cleaning status is not validated_clean.")
    return {
        "repo_id": manifest.repo_id,
        "revision": manifest.resolved_revision,
        "episodes": list(manifest.episodes),
        "cameras": manifest.cameras,
        "fields": manifest.fields,
        "manifest_sha256": file_sha256(root / "manifest.json"),
        "checksums_sha256": file_sha256(root / "cache_checksums.json"),
        "cleaning_report_sha256": file_sha256(cleaning_path),
    }


def _train_statistics(
    adapter: LeRobotV3Adapter,
    train_episodes: set[int],
    contract: ActionContract,
) -> tuple[DatasetStatistics, dict[str, Any]]:
    state = RunningMoments()
    action = RunningMoments()
    clipped_by_dimension = torch.zeros(contract.dimension, dtype=torch.long)
    maximum_source_overshoot = torch.zeros(contract.dimension)
    for source_index in range(len(adapter)):
        reference = adapter.frame_reference(source_index)
        if int(reference.episode_id) not in train_episodes:
            continue
        state.update(adapter.state_at(source_index))
        source_action = adapter.action_at(source_index)
        clipped_action, clip_mask = contract.clip(source_action)
        clipped_by_dimension += clip_mask.to(torch.long)
        maximum_source_overshoot = torch.maximum(
            maximum_source_overshoot,
            (source_action - clipped_action).abs(),
        )
        _validate_source_overshoot(
            contract,
            maximum_source_overshoot,
            context=f"train episode {reference.episode_id} frame {reference.frame_index}",
        )
        action.update(clipped_action)
    if state.count == 0 or action.count == 0:
        raise ValueError("Training split produced no normalization observations.")
    statistics = DatasetStatistics(
        state=state.finalize(),
        action=action.finalize(),
        state_count=state.count,
        action_count=action.count,
    )
    return statistics, _action_transform_report(
        contract,
        clipped_by_dimension,
        maximum_source_overshoot,
        source_vectors=action.count,
    )


def _validate_source_overshoot(
    contract: ActionContract,
    maximum_source_overshoot: torch.Tensor,
    *,
    context: str,
) -> None:
    """Reject source actions whose clipping exceeds the physical import tolerance."""

    received = maximum_source_overshoot.detach().to(dtype=torch.float32, device="cpu")
    if received.shape != (contract.dimension,) or not bool(torch.isfinite(received).all()):
        raise ValueError("Source-action overshoot must be one finite value per dimension.")
    allowed = contract.source_overshoot_tolerances.to(dtype=torch.float32, device="cpu")
    violation = received > allowed + 1e-6
    if not bool(violation.any()):
        return
    details = ", ".join(
        f"{contract.dimension_names[index]}={float(received[index]):.8g}>"
        f"{float(allowed[index]):.8g}"
        for index in violation.nonzero(as_tuple=False).flatten().tolist()
    )
    raise ValueError(
        f"Source action exceeds the declared overshoot tolerance at {context}: {details}."
    )


def _action_transform_report(
    contract: ActionContract,
    clipped_by_dimension: torch.Tensor,
    maximum_source_overshoot: torch.Tensor,
    *,
    source_vectors: int,
) -> dict[str, Any]:
    clipped_elements = int(clipped_by_dimension.sum())
    total_elements = source_vectors * contract.dimension
    return {
        "schema_version": 1,
        "type": "clip_to_rosetta_contract_v1",
        "source_vectors": source_vectors,
        "source_elements": total_elements,
        "clipped_elements": clipped_elements,
        "clipping_rate": clipped_elements / total_elements if total_elements else 0.0,
        "dimensions": {
            name: {
                "clipped": int(clipped_by_dimension[index]),
                "maximum_source_overshoot": float(maximum_source_overshoot[index]),
                "allowed_source_overshoot": float(
                    contract.source_overshoot_tolerances[index]
                ),
            }
            for index, name in enumerate(contract.dimension_names)
        },
    }


def _visible_materialization_scope(experiment: dict[str, Any]) -> dict[str, Any]:
    """Freeze the train/validation-only scope before dataset or model payload I/O."""

    raw_split = experiment.get("dataset", {}).get("split")
    if not isinstance(raw_split, dict):
        raise ValueError("Visible cache build requires a declared dataset split mapping.")
    split: dict[str, list[int]] = {}
    for name in (*VISIBLE_MATERIALIZED_SPLITS, *VISIBLE_WITHHELD_SPLITS):
        values = raw_split.get(name)
        if (
            not isinstance(values, list)
            or not values
            or any(type(value) is not int for value in values)
            or len(values) != len(set(values))
        ):
            raise ValueError(f"Visible cache split {name!r} must be unique non-empty integers.")
        split[name] = [int(value) for value in values]
    train = set(split["train"])
    validation = set(split["validation"])
    hidden_test = set(split["test"])
    if train & validation or train & hidden_test or validation & hidden_test:
        raise ValueError("Visible cache train, validation, and hidden-test scopes overlap.")
    materialized_episodes = [
        episode
        for name in VISIBLE_MATERIALIZED_SPLITS
        for episode in split[name]
    ]
    if hidden_test & set(materialized_episodes):
        raise ValueError("Hidden-test episodes cannot enter visible cache materialization.")
    return {
        "schema_version": 1,
        "type": "direct_visible_feature_cache_build_v1",
        "materialized_splits": list(VISIBLE_MATERIALIZED_SPLITS),
        "withheld_splits": list(VISIBLE_WITHHELD_SPLITS),
        "materialized_episodes": {
            name: split[name] for name in VISIBLE_MATERIALIZED_SPLITS
        },
        "withheld_episodes": {"test": split["test"]},
        "adapter_episodes": materialized_episodes,
        "hidden_test_loaded": False,
        "hidden_test_materialized": False,
    }


def _validated_checksum_inventory(root: Path) -> dict[str, str]:
    path = root / "cache_checksums.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    files = payload.get("files")
    if (
        payload.get("version") != 1
        or payload.get("algorithm") != "sha256"
        or not isinstance(files, dict)
        or not files
    ):
        raise ValueError("Prepared-cache checksum inventory is invalid.")
    inventory: dict[str, str] = {}
    for relative_text, expected in files.items():
        if not isinstance(relative_text, str):
            raise ValueError("Prepared-cache checksum inventory contains an unsafe entry.")
        relative = Path(relative_text)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != relative_text
            or not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise ValueError("Prepared-cache checksum inventory contains an unsafe entry.")
        inventory[relative_text] = expected
    return inventory


def _validate_checksum_paths(
    root: Path,
    inventory: dict[str, str],
    relative_paths: set[str],
) -> None:
    for relative_text in sorted(relative_paths):
        expected = inventory.get(relative_text)
        if expected is None:
            raise ValueError(f"Prepared-cache checksum is missing: {relative_text}.")
        path = root / relative_text
        if not path.is_file():
            raise FileNotFoundError(f"Prepared-cache file is missing: {relative_text}.")
        if file_sha256(path) != expected:
            raise ValueError(f"Prepared-cache checksum mismatch: {relative_text}.")


def _validate_visible_cache_checksums(
    root: Path,
    materialization: dict[str, Any],
    cameras: dict[str, str],
) -> int:
    """Verify only metadata and payload files needed by visible episodes."""

    inventory = _validated_checksum_inventory(root)
    metadata_paths = {
        relative for relative in inventory if relative.startswith("meta/")
    }
    episode_metadata = sorted(
        relative
        for relative in metadata_paths
        if relative.startswith("meta/episodes/") and relative.endswith(".parquet")
    )
    if "meta/info.json" not in metadata_paths or not episode_metadata:
        raise ValueError("Visible cache requires checksummed dataset metadata.")
    _validate_checksum_paths(root, inventory, metadata_paths)

    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    data_template = info.get("data_path")
    video_template = info.get("video_path")
    if not isinstance(data_template, str) or not isinstance(video_template, str):
        raise ValueError("Visible cache dataset paths are not declared.")

    import pyarrow.dataset as arrow_dataset

    selected = {
        int(episode)
        for episodes in materialization["materialized_episodes"].values()
        for episode in episodes
    }
    columns = ["episode_index", "data/chunk_index", "data/file_index"]
    for video_key in cameras.values():
        columns.extend(
            [
                f"videos/{video_key}/chunk_index",
                f"videos/{video_key}/file_index",
            ]
        )
    dataset = arrow_dataset.dataset(
        [str(root / relative) for relative in episode_metadata],
        format="parquet",
    )
    rows = dataset.to_table(
        columns=columns,
        filter=arrow_dataset.field("episode_index").isin(selected),
    ).to_pylist()
    found = [int(row["episode_index"]) for row in rows]
    if len(found) != len(set(found)) or set(found) != selected:
        raise ValueError("Visible cache episode metadata differs from the registered split.")

    required_paths = set(metadata_paths)
    for row in rows:
        required_paths.add(
            data_template.format(
                chunk_index=int(row["data/chunk_index"]),
                file_index=int(row["data/file_index"]),
            )
        )
        for video_key in cameras.values():
            required_paths.add(
                video_template.format(
                    video_key=video_key,
                    chunk_index=int(row[f"videos/{video_key}/chunk_index"]),
                    file_index=int(row[f"videos/{video_key}/file_index"]),
                )
            )
    _validate_checksum_paths(root, inventory, required_paths)
    return len(required_paths)


def _split_lookup(
    experiment: dict[str, Any],
    split_names: tuple[str, ...] = ("train", "validation", "test"),
) -> dict[int, str]:
    split = experiment["dataset"]["split"]
    return {
        int(episode): name
        for name in split_names
        for episode in split[name]
    }


def _anchors_by_episode(
    dataset: ActionChunkDataset,
    split_lookup: dict[int, str],
    frame_stride: int,
) -> dict[int, list[int]]:
    if frame_stride <= 0:
        raise ValueError("dataset.frame_stride must be positive.")
    result: defaultdict[int, list[int]] = defaultdict(list)
    for dataset_index in range(len(dataset)):
        reference = dataset.anchor_reference(dataset_index)
        episode = int(reference.episode_id)
        if episode in split_lookup and reference.frame_index % frame_stride == 0:
            result[episode].append(dataset_index)
    missing = sorted(set(split_lookup) - set(result))
    if missing:
        raise ValueError(f"No action-chunk anchors selected for episodes: {missing}.")
    return dict(result)


def _context(config_path: Path, *, visible_only: bool = False) -> dict[str, Any]:
    experiment = load_experiment_config(config_path, REPOSITORY_ROOT)
    materialization = (
        _visible_materialization_scope(experiment) if visible_only else None
    )
    dataset_path = REPOSITORY_ROOT / experiment["dataset"]["config"]
    dataset_config = load_dataset_config(dataset_path)
    dataset_root, dataset_manifest = resolve_prepared_cache(
        dataset_config,
        REPOSITORY_ROOT,
        validate_checksums=not visible_only,
    )
    if materialization is not None:
        _validate_visible_cache_checksums(
            dataset_root,
            materialization,
            dataset_config.cameras,
        )
    contract_path = REPOSITORY_ROOT / experiment["action_contract"]
    contract = load_action_contract(contract_path)
    contract.validate_order(ordered_feature_names(dataset_root, dataset_config.fields.action))
    if contract.dimension != dataset_config.expected_action_dim:
        raise ValueError("Dataset action width and Action Contract differ.")

    model_environment = experiment["backbone"]["local_root_environment"]
    raw_model_root = os.environ.get(model_environment)
    if not raw_model_root:
        raise ValueError(f"{model_environment} must identify an existing local model directory.")
    model_root = Path(raw_model_root)
    if not model_root.is_dir():
        raise FileNotFoundError("Configured local model directory does not exist.")
    model_identity = _model_identity(model_root, experiment["backbone"])

    adapter_episodes = (
        tuple(materialization["adapter_episodes"])
        if materialization is not None
        else dataset_config.episodes
    )
    unknown_adapter_episodes = sorted(set(adapter_episodes) - set(dataset_config.episodes))
    if unknown_adapter_episodes:
        raise ValueError(
            f"Visible adapter episodes are outside the pinned dataset: {unknown_adapter_episodes}."
        )
    if materialization is not None and set(adapter_episodes) & set(
        materialization["withheld_episodes"]["test"]
    ):
        raise ValueError("Hidden-test episodes reached visible adapter construction.")
    adapter = LeRobotV3Adapter(
        repo_id=dataset_config.repo_id,
        revision=dataset_manifest.resolved_revision,
        root=dataset_root,
        episodes=adapter_episodes,
        cameras=dataset_config.cameras,
        fields=dataset_config.fields,
        embodiment=dataset_config.embodiment,
        license_name=dataset_config.license,
    )
    chunked = ActionChunkDataset(adapter, dataset_config.chunk_size)
    split_lookup = _split_lookup(
        experiment,
        VISIBLE_MATERIALIZED_SPLITS
        if materialization is not None
        else ("train", "validation", "test"),
    )
    train_statistics, train_action_transform = _train_statistics(
        adapter,
        {int(episode) for episode in experiment["dataset"]["split"]["train"]},
        contract,
    )
    normalization = {
        "source_split": "train",
        "episodes": sorted(
            episode for episode, name in split_lookup.items() if name == "train"
        ),
        "statistics": train_statistics.to_dict(),
        "action_transform": train_action_transform,
    }
    identity = {
        "schema_version": 1,
        "experiment_id": experiment["experiment_id"],
        "experiment_config_sha256": file_sha256(config_path),
        "code": workspace_code_identity(REPOSITORY_ROOT),
        "dataset": _prepared_identity(dataset_root, dataset_manifest),
        "split": experiment["dataset"]["split"],
        "selection": {
            "frame_stride": int(experiment["dataset"]["frame_stride"]),
            "action_chunk_length": dataset_config.chunk_size,
            "action_transform": "clip_to_rosetta_contract_v1",
        },
        "model": model_identity,
        "processor": experiment["backbone"]["processor"],
        "feature": {
            "layer": experiment["backbone"]["feature_layer"],
            "pooling": experiment["backbone"]["pooling"],
            "storage_dtype": "float16",
            "execution_device": _torch_device().type,
        },
        "normalization_sha256": stable_hash(normalization),
        "action_contract_sha256": file_sha256(contract_path),
    }
    if materialization is not None:
        identity["materialization"] = materialization
    identity_hash = stable_hash(identity)
    root = _feature_root() / experiment["experiment_id"] / identity_hash[:16]
    return {
        "experiment": experiment,
        "dataset_config": dataset_config,
        "dataset_root": dataset_root,
        "adapter": adapter,
        "chunked": chunked,
        "split_lookup": split_lookup,
        "anchors": _anchors_by_episode(
            chunked,
            split_lookup,
            int(experiment["dataset"]["frame_stride"]),
        ),
        "normalization": normalization,
        "identity": identity,
        "identity_hash": identity_hash,
        "cache_root": root,
        "model_root": model_root,
        "contract": contract,
        "materialization": materialization,
    }


def _backbone(context: dict[str, Any]) -> Qwen35Backbone:
    configured = context["experiment"]["backbone"]
    dtype = getattr(torch, str(configured["dtype"]), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported backbone dtype: {configured['dtype']!r}.")
    return Qwen35Backbone(
        str(context["model_root"]),
        hidden_size=int(context["identity"]["model"]["hidden_size"]),
        device=_torch_device(),
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


def _smoke_context(context: dict[str, Any]) -> int:
    """Run one real image/instruction forward without writing a cache."""

    first_episode = context["experiment"]["dataset"]["split"]["train"][0]
    sample = context["chunked"][context["anchors"][first_episode][0]]
    backbone = _backbone(context)
    started = time.perf_counter()
    with torch.inference_mode():
        feature = backbone({"images": sample.images, "instruction": sample.instruction})
    elapsed = time.perf_counter() - started
    if feature.shape != (1, backbone.hidden_size) or not bool(torch.isfinite(feature).all()):
        raise RuntimeError(f"Invalid frozen feature output: {tuple(feature.shape)}.")
    if feature.requires_grad or any(parameter.requires_grad for parameter in backbone.parameters()):
        raise RuntimeError("Frozen-backbone smoke unexpectedly retained gradients.")
    print("Frozen Qwen real-sample smoke passed")
    print(f"Feature shape: {tuple(feature.shape)}")
    print(f"Feature dtype: {feature.dtype}")
    print(f"Elapsed seconds: {elapsed:.3f}")
    print(f"Identity: {context['identity_hash']}")
    return 0


def smoke(config_path: Path) -> int:
    """Run the legacy all-split-context frozen Qwen smoke."""

    return _smoke_context(_context(config_path))


def smoke_visible(config_path: Path) -> int:
    """Run one train anchor through the exact visible-only context without writes."""

    return _smoke_context(_context(config_path, visible_only=True))


def _existing_shard(
    path: Path,
    identity_hash: str,
    split: str,
    episode: int,
    contract: ActionContract,
) -> tuple[int, dict[str, Any]]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if (
        value.get("identity_hash") != identity_hash
        or value.get("split") != split
        or int(value.get("episode", -1)) != episode
    ):
        raise ValueError(f"Existing feature shard identity mismatch: {path}.")
    tensors = {
        key: value.get(key)
        for key in ("features", "robot_state", "actions", "episode_ids", "frame_indices")
    }
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors.values()):
        raise ValueError(f"Existing feature shard is missing tensors: {path}.")
    features = tensors["features"]
    actions = tensors["actions"]
    if features.ndim != 2 or actions.ndim != 3:
        raise ValueError(f"Existing feature shard is invalid: {path}.")
    count = features.shape[0]
    if any(tensor.shape[0] != count for tensor in tensors.values()):
        raise ValueError(f"Existing feature shard sample counts differ: {path}.")
    if any(not bool(torch.isfinite(tensor).all()) for tensor in tensors.values()):
        raise ValueError(f"Existing feature shard contains non-finite values: {path}.")
    _, violations = contract.clip(actions)
    if bool(violations.any()):
        raise ValueError(f"Existing feature shard has out-of-contract targets: {path}.")
    transform = value.get("action_transform")
    if not isinstance(transform, dict) or transform.get("type") != (
        "clip_to_rosetta_contract_v1"
    ):
        raise ValueError(f"Existing feature shard lacks action-transform provenance: {path}.")
    dimensions = transform.get("dimensions")
    if not isinstance(dimensions, dict) or set(dimensions) != set(contract.dimension_names):
        raise ValueError(f"Existing feature shard has incomplete action-transform data: {path}.")
    maximum_source_overshoot = torch.tensor(
        [
            float(dimensions[name].get("maximum_source_overshoot", math.inf))
            if isinstance(dimensions[name], dict)
            else math.inf
            for name in contract.dimension_names
        ],
        dtype=torch.float32,
    )
    _validate_source_overshoot(
        contract,
        maximum_source_overshoot,
        context=f"existing shard {path.name}",
    )
    return count, transform


def _feature_manifest_payload(
    context: dict[str, Any],
    shard_records: dict[str, list[dict[str, Any]]],
    *,
    normalization_sha256: str,
) -> dict[str, Any]:
    if set(shard_records) != {"train", "validation", "test"}:
        raise ValueError("Feature shard records must declare all three split names.")
    samples = {
        split: sum(int(record["samples"]) for record in records)
        for split, records in shard_records.items()
    }
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "identity_hash": context["identity_hash"],
        "identity": context["identity"],
        "normalization_path": "normalization.json",
        "normalization_sha256": normalization_sha256,
        "shards": shard_records,
        "samples": samples,
    }
    materialization = context.get("materialization")
    if materialization is None:
        return manifest
    if context["identity"].get("materialization") != materialization:
        raise ValueError("Visible cache identity lacks its materialization scope.")
    if shard_records["test"] or samples["test"] != 0:
        raise ValueError("Visible cache manifest cannot contain hidden-test shards.")
    for split in VISIBLE_MATERIALIZED_SPLITS:
        records = shard_records[split]
        actual = [int(record.get("episode", -1)) for record in records]
        expected = [int(value) for value in materialization["materialized_episodes"][split]]
        if not records or len(actual) != len(set(actual)) or set(actual) != set(expected):
            raise ValueError(f"Visible cache {split} shards differ from the fixed scope.")
    manifest.update(
        hidden_test_materialized=False,
        hidden_test_loaded=False,
        materialized_splits=list(VISIBLE_MATERIALIZED_SPLITS),
        withheld_splits=list(VISIBLE_WITHHELD_SPLITS),
    )
    return manifest


def _build_cache(context: dict[str, Any]) -> int:
    """Create missing per-episode shards and a complete immutable manifest."""

    materialization = context.get("materialization")
    if materialization is not None:
        expected_lookup = {
            int(episode): split
            for split in VISIBLE_MATERIALIZED_SPLITS
            for episode in materialization["materialized_episodes"][split]
        }
        if (
            set(context["anchors"]) != set(expected_lookup)
            or context["split_lookup"] != expected_lookup
        ):
            raise ValueError(
                "Visible cache anchors must exactly cover train and validation before writes."
            )
    root: Path = context["cache_root"]
    root.mkdir(parents=True, exist_ok=True)
    create_json(root / "identity.json", context["identity"])
    create_json(root / "normalization.json", context["normalization"])
    backbone: Qwen35Backbone | None = None
    shard_records: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    total_started = time.perf_counter()

    for episode in sorted(context["anchors"]):
        split = context["split_lookup"][episode]
        relative = Path("shards") / split / f"episode-{episode:03d}.pt"
        path = root / relative
        expected_count = len(context["anchors"][episode])
        if path.exists():
            actual_count, shard_action_transform = _existing_shard(
                path,
                context["identity_hash"],
                split,
                episode,
                context["contract"],
            )
            if actual_count != expected_count:
                raise ValueError(
                    f"Existing shard count differs for episode {episode}: "
                    f"expected {expected_count}, received {actual_count}."
                )
            print(f"Validated existing shard {relative} ({actual_count} samples)", flush=True)
        else:
            if backbone is None:
                backbone = _backbone(context)
            features: list[torch.Tensor] = []
            states: list[torch.Tensor] = []
            actions: list[torch.Tensor] = []
            frames: list[int] = []
            clipped_by_dimension = torch.zeros(
                context["contract"].dimension,
                dtype=torch.long,
            )
            maximum_source_overshoot = torch.zeros(context["contract"].dimension)
            source_action_vectors = 0
            episode_started = time.perf_counter()
            for offset, dataset_index in enumerate(context["anchors"][episode], start=1):
                sample = context["chunked"][dataset_index]
                with torch.inference_mode():
                    feature = backbone(
                        {"images": sample.images, "instruction": sample.instruction}
                    )
                if feature.shape != (1, backbone.hidden_size):
                    raise RuntimeError(
                        f"Unexpected feature shape for episode {episode}: {tuple(feature.shape)}."
                    )
                if not bool(torch.isfinite(feature).all()):
                    raise FloatingPointError(f"Non-finite feature in episode {episode}.")
                features.append(feature[0].to(torch.float16).cpu())
                states.append(sample.robot_state.to(torch.float32).cpu())
                clipped_actions, clip_mask = context["contract"].clip(sample.actions)
                clipped_by_dimension += clip_mask.sum(dim=0).to(torch.long)
                maximum_source_overshoot = torch.maximum(
                    maximum_source_overshoot,
                    (sample.actions - clipped_actions).abs().max(dim=0).values,
                )
                _validate_source_overshoot(
                    context["contract"],
                    maximum_source_overshoot,
                    context=(
                        f"{split} episode {episode} frame {sample.frame_index}"
                    ),
                )
                source_action_vectors += sample.actions.shape[0]
                actions.append(clipped_actions.to(torch.float32).cpu())
                frames.append(sample.frame_index)
                if offset == 1 or offset % 10 == 0 or offset == expected_count:
                    print(
                        f"episode={episode:03d} split={split} "
                        f"sample={offset}/{expected_count}",
                        flush=True,
                    )
            shard_action_transform = _action_transform_report(
                context["contract"],
                clipped_by_dimension,
                maximum_source_overshoot,
                source_vectors=source_action_vectors,
            )
            payload = {
                "schema_version": 1,
                "identity_hash": context["identity_hash"],
                "split": split,
                "episode": episode,
                "features": torch.stack(features),
                "robot_state": torch.stack(states),
                "actions": torch.stack(actions),
                "episode_ids": torch.full((expected_count,), episode, dtype=torch.long),
                "frame_indices": torch.tensor(frames, dtype=torch.long),
                "action_transform": shard_action_transform,
            }
            save_tensor_shard(path, payload)
            print(
                f"Created {relative} in {time.perf_counter() - episode_started:.1f}s",
                flush=True,
            )
        shard_records[split].append(
            {
                "episode": episode,
                "path": relative.as_posix(),
                "samples": expected_count,
                "sha256": file_sha256(path),
                "action_transform": shard_action_transform,
            }
        )

    manifest = _feature_manifest_payload(
        context,
        shard_records,
        normalization_sha256=file_sha256(root / "normalization.json"),
    )
    create_json(root / "manifest.json", manifest)
    print(
        "Visible-only frozen feature cache complete"
        if materialization is not None
        else "Frozen feature cache complete"
    )
    print(f"Cache identity: {context['identity_hash']}")
    print(f"Samples: {json.dumps(manifest['samples'], sort_keys=True)}")
    print(f"Elapsed seconds: {time.perf_counter() - total_started:.1f}")
    return 0


def build(config_path: Path) -> int:
    """Build the legacy all-split frozen feature cache."""

    return _build_cache(_context(config_path))


def build_visible(config_path: Path) -> int:
    """Build train/validation Qwen shards while withholding hidden test."""

    return _build_cache(_context(config_path, visible_only=True))


def _source_semantics(identity: dict[str, Any]) -> dict[str, Any]:
    feature = identity.get("feature")
    if not isinstance(feature, dict):
        raise ValueError("Feature-cache source identity lacks feature metadata.")
    required = {
        "dataset": identity.get("dataset"),
        "split": identity.get("split"),
        "selection": identity.get("selection"),
        "model": identity.get("model"),
        "processor": identity.get("processor"),
        "feature_layer": feature.get("layer"),
        "storage_dtype": feature.get("storage_dtype"),
        "normalization_sha256": identity.get("normalization_sha256"),
        "action_contract_sha256": identity.get("action_contract_sha256"),
    }
    if any(value is None for value in required.values()):
        raise ValueError("Feature-cache source identity is incomplete.")
    return required


def _validated_source_manifest(raw_path: Path) -> dict[str, Any]:
    path = raw_path.resolve()
    manifest = load_feature_manifest(path)
    identity = manifest.get("identity")
    identity_hash = manifest.get("identity_hash")
    if not isinstance(identity, dict) or not isinstance(identity_hash, str):
        raise ValueError("Feature-cache source manifest lacks an identity.")
    if stable_hash(identity) != identity_hash:
        raise ValueError("Feature-cache source manifest identity hash is invalid.")
    pooling = identity.get("feature", {}).get("pooling")
    if not isinstance(pooling, str):
        raise ValueError("Feature-cache source pooling is missing.")
    normalization_relative = Path(str(manifest.get("normalization_path", "")))
    if normalization_relative.is_absolute() or ".." in normalization_relative.parts:
        raise ValueError("Unsafe source normalization path.")
    normalization_path = path.parent / normalization_relative
    if file_sha256(normalization_path) != manifest.get("normalization_sha256"):
        raise ValueError("Feature-cache source normalization checksum mismatch.")
    return {
        "path": path,
        "root": path.parent,
        "manifest": manifest,
        "identity": identity,
        "pooling": pooling,
        "normalization": json.loads(normalization_path.read_text(encoding="utf-8")),
    }


def _ordered_composition_sources(paths: list[Path]) -> list[dict[str, Any]]:
    if len(paths) != len(COMPOSITION_POOLINGS):
        raise ValueError("Combined cache composition requires exactly two source manifests.")
    by_pooling: dict[str, dict[str, Any]] = {}
    for raw_path in paths:
        source = _validated_source_manifest(raw_path)
        pooling = source["pooling"]
        if pooling not in COMPOSITION_POOLINGS:
            raise ValueError(f"Unsupported composition source pooling: {pooling!r}.")
        if pooling in by_pooling:
            raise ValueError(f"Duplicate composition source pooling: {pooling}.")
        by_pooling[pooling] = source
    if set(by_pooling) != set(COMPOSITION_POOLINGS):
        raise ValueError("Composition sources do not contain the required pooling pair.")
    return [by_pooling[pooling] for pooling in COMPOSITION_POOLINGS]


def _source_shard_records(source: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    raw_shards = source["manifest"].get("shards")
    if not isinstance(raw_shards, dict):
        raise ValueError("Feature-cache source manifest lacks shard records.")
    for split in ("train", "validation", "test"):
        records = raw_shards.get(split)
        if not isinstance(records, list) or not records:
            raise ValueError(f"Feature-cache source has no {split} shards.")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("Feature-cache source shard record must be a mapping.")
            episode = int(record.get("episode", -1))
            key = (split, episode)
            if episode < 0 or key in result:
                raise ValueError("Feature-cache source has invalid or duplicate episodes.")
            result[key] = record
    return result


def _visible_source_shard_records(
    source: dict[str, Any],
    experiment: dict[str, Any],
) -> dict[tuple[str, int], dict[str, Any]]:
    """Validate one exact train/validation-only source without opening test tensors."""

    manifest = source["manifest"]
    identity = source["identity"]
    materialization = identity.get("materialization")
    raw_shards = manifest.get("shards")
    samples = manifest.get("samples")
    split = experiment["dataset"]["split"]
    expected = {
        name: [int(episode) for episode in split[name]]
        for name in (*VISIBLE_MATERIALIZED_SPLITS, *VISIBLE_WITHHELD_SPLITS)
    }
    if (
        manifest.get("hidden_test_loaded") is not False
        or manifest.get("hidden_test_materialized") is not False
        or manifest.get("materialized_splits") != list(VISIBLE_MATERIALIZED_SPLITS)
        or manifest.get("withheld_splits") != list(VISIBLE_WITHHELD_SPLITS)
        or not isinstance(raw_shards, dict)
        or set(raw_shards) != {"train", "validation", "test"}
        or raw_shards.get("test") != []
        or not isinstance(samples, dict)
        or set(samples) != {"train", "validation", "test"}
        or samples.get("test") != 0
    ):
        raise ValueError("Visible derivation source did not withhold hidden-test tensors.")
    if (
        not isinstance(materialization, dict)
        or materialization.get("schema_version") != 1
        or materialization.get("type") != "direct_visible_feature_cache_build_v1"
        or materialization.get("hidden_test_loaded") is not False
        or materialization.get("hidden_test_materialized") is not False
        or materialization.get("materialized_splits")
        != list(VISIBLE_MATERIALIZED_SPLITS)
        or materialization.get("withheld_splits") != list(VISIBLE_WITHHELD_SPLITS)
        or materialization.get("materialized_episodes")
        != {name: expected[name] for name in VISIBLE_MATERIALIZED_SPLITS}
        or materialization.get("withheld_episodes") != {"test": expected["test"]}
        or materialization.get("adapter_episodes")
        != expected["train"] + expected["validation"]
    ):
        raise ValueError("Visible derivation source identity has a false hidden-test scope.")
    test_root = source["root"] / "shards" / "test"
    if test_root.exists() and any(path.is_file() for path in test_root.rglob("*")):
        raise ValueError("Visible derivation source contains unmanifested hidden-test files.")

    result: dict[tuple[str, int], dict[str, Any]] = {}
    for name in VISIBLE_MATERIALIZED_SPLITS:
        records = raw_shards.get(name)
        if not isinstance(records, list) or len(records) != len(expected[name]):
            raise ValueError(f"Visible derivation source has the wrong {name} shard count.")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("Visible source shard record must be a mapping.")
            episode = int(record.get("episode", -1))
            key = (name, episode)
            relative = Path(str(record.get("path", "")))
            if (
                episode < 0
                or key in result
                or relative.is_absolute()
                or ".." in relative.parts
                or relative.parts[:2] != ("shards", name)
            ):
                raise ValueError("Visible source has an invalid shard identity or path.")
            result[key] = record
        if {episode for split_name, episode in result if split_name == name} != set(
            expected[name]
        ):
            raise ValueError(f"Visible derivation source {name} episodes differ.")
        if sum(int(record.get("samples", -1)) for record in records) != int(
            samples[name]
        ):
            raise ValueError(f"Visible derivation source {name} sample count differs.")
    expected_keys = {
        (name, episode)
        for name in VISIBLE_MATERIALIZED_SPLITS
        for episode in expected[name]
    }
    if set(result) != expected_keys:
        raise ValueError("Visible derivation source records do not exactly match scope.")
    return result


def _load_composition_shard(
    source: dict[str, Any],
    record: dict[str, Any],
    *,
    split: str,
    episode: int,
    expected_feature_dim: int,
    contract: ActionContract,
) -> dict[str, Any]:
    relative = Path(str(record.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe source feature-shard path.")
    path = source["root"] / relative
    if file_sha256(path) != record.get("sha256"):
        raise ValueError(f"Source feature-shard checksum mismatch: {relative}.")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if (
        value.get("identity_hash") != source["manifest"]["identity_hash"]
        or value.get("split") != split
        or int(value.get("episode", -1)) != episode
    ):
        raise ValueError(f"Source feature-shard identity mismatch: {relative}.")
    tensors = {key: value.get(key) for key in SHARD_TENSOR_KEYS}
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors.values()):
        raise ValueError(f"Source feature-shard is missing tensors: {relative}.")
    features = tensors["features"]
    actions = tensors["actions"]
    count = int(record.get("samples", -1))
    if features.shape != (count, expected_feature_dim):
        raise ValueError(f"Source feature width or sample count differs: {relative}.")
    if actions.ndim != 3 or any(tensor.shape[0] != count for tensor in tensors.values()):
        raise ValueError(f"Source feature-shard tensor ranks differ: {relative}.")
    if any(not bool(torch.isfinite(tensor).all()) for tensor in tensors.values()):
        raise ValueError(f"Source feature-shard contains non-finite values: {relative}.")
    if not bool(tensors["episode_ids"].eq(episode).all()):
        raise ValueError(f"Source feature-shard episode tensor differs: {relative}.")
    _, violations = contract.clip(actions)
    if bool(violations.any()):
        raise ValueError(f"Source feature-shard targets violate the Action Contract: {relative}.")
    transform = value.get("action_transform")
    if not isinstance(transform, dict) or transform != record.get("action_transform"):
        raise ValueError(f"Source feature-shard action transform differs: {relative}.")
    return value


def _validate_existing_composition_shard(path: Path, expected: dict[str, Any]) -> None:
    actual = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("identity_hash", "split", "episode", "action_transform"):
        if actual.get(key) != expected.get(key):
            raise ValueError(f"Existing composed shard metadata differs: {path}.")
    for key in SHARD_TENSOR_KEYS:
        if not isinstance(actual.get(key), torch.Tensor) or not torch.equal(
            actual[key], expected[key]
        ):
            raise ValueError(f"Existing composed shard tensor differs for {key}: {path}.")


def compose(config_path: Path, source_paths: list[Path]) -> int:
    """Create a strict global-plus-spatial cache from two immutable source caches."""

    context = _context(config_path)
    if context["experiment"]["backbone"]["pooling"] != COMBINED_POOLING:
        raise ValueError("Cache composition requires the declared combined pooling mode.")
    sources = _ordered_composition_sources(source_paths)
    target_semantics = _source_semantics(context["identity"])
    source_semantics = [_source_semantics(source["identity"]) for source in sources]
    if any(semantics != target_semantics for semantics in source_semantics):
        raise ValueError("Composition source and target semantic identities differ.")
    if sources[0]["normalization"] != context["normalization"] or any(
        source["normalization"] != sources[0]["normalization"] for source in sources[1:]
    ):
        raise ValueError("Composition source normalization payloads differ.")

    hidden_size = int(context["identity"]["model"]["hidden_size"])
    source_feature_dims = (hidden_size, hidden_size * 4)
    derivation = {
        "schema_version": 1,
        "type": "ordered_feature_concatenation_v1",
        "output_pooling": COMBINED_POOLING,
        "output_feature_dim": sum(source_feature_dims),
        "sources": [
            {
                "experiment_id": source["identity"]["experiment_id"],
                "pooling": source["pooling"],
                "feature_dim": feature_dim,
                "identity_hash": source["manifest"]["identity_hash"],
                "manifest_sha256": file_sha256(source["path"]),
            }
            for source, feature_dim in zip(sources, source_feature_dims)
        ],
    }
    identity = copy.deepcopy(context["identity"])
    identity["feature"]["execution_device"] = "derived_from_source_caches"
    identity["feature"]["derivation"] = derivation
    identity_hash = stable_hash(identity)
    root = _feature_root() / context["experiment"]["experiment_id"] / identity_hash[:16]
    root.mkdir(parents=True, exist_ok=True)
    create_json(root / "identity.json", identity)
    create_json(root / "normalization.json", context["normalization"])

    records_by_source = [_source_shard_records(source) for source in sources]
    expected_keys = {
        (context["split_lookup"][episode], episode) for episode in context["anchors"]
    }
    if any(set(records) != expected_keys for records in records_by_source):
        raise ValueError("Composition source shard episodes differ from the target split.")

    shard_records: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    started = time.perf_counter()
    for split, episode in sorted(expected_keys, key=lambda value: value[1]):
        values = [
            _load_composition_shard(
                source,
                records[(split, episode)],
                split=split,
                episode=episode,
                expected_feature_dim=feature_dim,
                contract=context["contract"],
            )
            for source, records, feature_dim in zip(
                sources,
                records_by_source,
                source_feature_dims,
            )
        ]
        for key in SHARD_TENSOR_KEYS[1:]:
            if not torch.equal(values[0][key], values[1][key]):
                raise ValueError(
                    f"Composition source tensor {key!r} differs for episode {episode}."
                )
        if values[0]["action_transform"] != values[1]["action_transform"]:
            raise ValueError(f"Composition action transforms differ for episode {episode}.")
        expected_count = len(context["anchors"][episode])
        if values[0]["features"].shape[0] != expected_count:
            raise ValueError(f"Composition sample count differs for episode {episode}.")
        payload = {
            "schema_version": 1,
            "identity_hash": identity_hash,
            "split": split,
            "episode": episode,
            "features": torch.cat(
                (values[0]["features"], values[1]["features"]), dim=-1
            ),
            "robot_state": values[0]["robot_state"],
            "actions": values[0]["actions"],
            "episode_ids": values[0]["episode_ids"],
            "frame_indices": values[0]["frame_indices"],
            "action_transform": values[0]["action_transform"],
        }
        relative = Path("shards") / split / f"episode-{episode:03d}.pt"
        path = root / relative
        if path.exists():
            _validate_existing_composition_shard(path, payload)
            print(f"Validated existing composed shard {relative}", flush=True)
        else:
            save_tensor_shard(path, payload)
            print(f"Created composed shard {relative}", flush=True)
        shard_records[split].append(
            {
                "episode": episode,
                "path": relative.as_posix(),
                "samples": expected_count,
                "sha256": file_sha256(path),
                "action_transform": payload["action_transform"],
            }
        )

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "identity_hash": identity_hash,
        "identity": identity,
        "normalization_path": "normalization.json",
        "normalization_sha256": file_sha256(root / "normalization.json"),
        "derivation": derivation,
        "shards": shard_records,
        "samples": {
            split: sum(record["samples"] for record in records)
            for split, records in shard_records.items()
        },
    }
    create_json(root / "manifest.json", manifest)
    print("Composed frozen feature cache complete")
    print(f"Cache identity: {identity_hash}")
    print(f"Samples: {json.dumps(manifest['samples'], sort_keys=True)}")
    print(f"Elapsed seconds: {time.perf_counter() - started:.1f}")
    return 0


def _pooling_feature_dim(pooling: str, hidden_size: int) -> int:
    multipliers = {
        "attention_masked_mean": 1,
        "image_token_mean": 1,
        "image_spatial_2x2": 4,
        COMBINED_POOLING: 5,
    }
    try:
        return hidden_size * multipliers[pooling]
    except KeyError as error:
        raise ValueError(f"Unsupported derived-cache pooling: {pooling!r}.") from error


def derive(config_path: Path, source_paths: list[Path]) -> int:
    """Rebind verified, semantically identical features to a downstream experiment."""

    if len(source_paths) != 1:
        raise ValueError("Derived cache creation requires exactly one source manifest.")
    context = _context(config_path)
    source = _validated_source_manifest(source_paths[0])
    target_pooling = str(context["experiment"]["backbone"]["pooling"])
    if source["pooling"] != target_pooling:
        raise ValueError("Derived cache source and target pooling modes differ.")
    if _source_semantics(source["identity"]) != _source_semantics(context["identity"]):
        raise ValueError("Derived cache source and target semantic identities differ.")
    if source["normalization"] != context["normalization"]:
        raise ValueError("Derived cache source and target normalization payloads differ.")

    derivation = {
        "schema_version": 1,
        "type": "verified_identity_rebind_v1",
        "tensor_transform": "identity",
        "source": {
            "experiment_id": source["identity"]["experiment_id"],
            "pooling": source["pooling"],
            "identity_hash": source["manifest"]["identity_hash"],
            "manifest_sha256": file_sha256(source["path"]),
        },
    }
    identity = copy.deepcopy(context["identity"])
    identity["feature"]["execution_device"] = "derived_from_source_cache"
    identity["feature"]["derivation"] = derivation
    identity_hash = stable_hash(identity)
    root = _feature_root() / context["experiment"]["experiment_id"] / identity_hash[:16]
    root.mkdir(parents=True, exist_ok=True)
    create_json(root / "identity.json", identity)
    create_json(root / "normalization.json", context["normalization"])

    records = _source_shard_records(source)
    expected_keys = {
        (context["split_lookup"][episode], episode) for episode in context["anchors"]
    }
    if set(records) != expected_keys:
        raise ValueError("Derived cache source shard episodes differ from the target split.")
    feature_dim = _pooling_feature_dim(
        target_pooling,
        int(context["identity"]["model"]["hidden_size"]),
    )
    shard_records: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    started = time.perf_counter()
    for split, episode in sorted(expected_keys, key=lambda value: value[1]):
        value = _load_composition_shard(
            source,
            records[(split, episode)],
            split=split,
            episode=episode,
            expected_feature_dim=feature_dim,
            contract=context["contract"],
        )
        expected_count = len(context["anchors"][episode])
        if value["features"].shape[0] != expected_count:
            raise ValueError(f"Derived cache sample count differs for episode {episode}.")
        payload = {
            "schema_version": 1,
            "identity_hash": identity_hash,
            "split": split,
            "episode": episode,
            **{key: value[key] for key in SHARD_TENSOR_KEYS},
            "action_transform": value["action_transform"],
        }
        relative = Path("shards") / split / f"episode-{episode:03d}.pt"
        path = root / relative
        if path.exists():
            _validate_existing_composition_shard(path, payload)
            print(f"Validated existing derived shard {relative}", flush=True)
        else:
            save_tensor_shard(path, payload)
            print(f"Created derived shard {relative}", flush=True)
        shard_records[split].append(
            {
                "episode": episode,
                "path": relative.as_posix(),
                "samples": expected_count,
                "sha256": file_sha256(path),
                "action_transform": payload["action_transform"],
            }
        )

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "identity_hash": identity_hash,
        "identity": identity,
        "normalization_path": "normalization.json",
        "normalization_sha256": file_sha256(root / "normalization.json"),
        "derivation": derivation,
        "shards": shard_records,
        "samples": {
            split: sum(record["samples"] for record in split_records)
            for split, split_records in shard_records.items()
        },
    }
    create_json(root / "manifest.json", manifest)
    print("Derived frozen feature cache complete")
    print(f"Cache identity: {identity_hash}")
    print(f"Samples: {json.dumps(manifest['samples'], sort_keys=True)}")
    print(f"Elapsed seconds: {time.perf_counter() - started:.1f}")
    return 0


def _validated_visible_controlled_change(
    experiment: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    """Validate that a visible identity rebind cannot hide a second experiment axis."""

    controlled = experiment.get("controlled_change")
    changed_axis = (
        str(controlled.get("changed_axis", ""))
        if isinstance(controlled, dict)
        else ""
    )
    if (
        not isinstance(controlled, dict)
        or controlled.get("reference_experiment")
        != source["identity"].get("experiment_id")
        or controlled.get("feature_derivation") != "verified_identity_rebind_v1"
    ):
        raise ValueError("Visible derivation controlled-change identity is invalid.")
    if changed_axis.startswith("training."):
        return controlled
    if changed_axis != "action_expert.fusion_dim":
        raise ValueError(
            "Visible derivation only supports training axes or the exact "
            "action_expert.fusion_dim axis."
        )

    reference_value = controlled.get("reference_value")
    candidate_value = controlled.get("candidate_value")
    if (
        isinstance(reference_value, bool)
        or not isinstance(reference_value, int)
        or reference_value <= 0
        or isinstance(candidate_value, bool)
        or not isinstance(candidate_value, int)
        or candidate_value <= 0
    ):
        raise ValueError("Fusion dimensions must be positive integers.")
    raw_reference_path = Path(str(controlled.get("reference_config", "")))
    if (
        not raw_reference_path.parts
        or raw_reference_path.is_absolute()
        or ".." in raw_reference_path.parts
    ):
        raise ValueError("Fusion derivation reference_config must be repository-relative.")
    reference_path = (REPOSITORY_ROOT / raw_reference_path).resolve()
    try:
        reference_path.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(
            "Fusion derivation reference_config escapes the repository."
        ) from exc
    if file_sha256(reference_path) != source["identity"].get(
        "experiment_config_sha256"
    ):
        raise ValueError("Fusion derivation reference config SHA does not match source.")
    reference = load_experiment_config(reference_path, REPOSITORY_ROOT)
    if reference.get("experiment_id") != controlled["reference_experiment"]:
        raise ValueError("Fusion derivation reference experiment identity differs.")
    for key in (
        "backbone",
        "dataset",
        "action_contract",
        "training",
        "benchmark",
        "evaluation",
        "simulation",
        "resources",
        "acceptance",
        "stop_conditions",
        "m2_completion_eligible",
    ):
        if experiment.get(key) != reference.get(key):
            raise ValueError(f"Fusion derivation changes the additional {key} axis.")
    reference_expert = copy.deepcopy(reference["action_expert"])
    candidate_expert = copy.deepcopy(experiment["action_expert"])
    if reference_expert.pop("fusion_dim", None) != reference_value:
        raise ValueError("Fusion derivation reference_value differs from its config.")
    if candidate_expert.pop("fusion_dim", None) != candidate_value:
        raise ValueError("Fusion derivation candidate_value differs from its config.")
    if candidate_expert != reference_expert:
        raise ValueError("Fusion derivation changes another action_expert axis.")
    return controlled


def derive_visible(config_path: Path, source_paths: list[Path]) -> int:
    """Rebind only train and validation shards while withholding hidden test tensors."""

    if len(source_paths) != 1:
        raise ValueError("Visible derived cache creation requires exactly one source manifest.")
    experiment = load_experiment_config(config_path, REPOSITORY_ROOT)
    source = _validated_source_manifest(source_paths[0])
    controlled = _validated_visible_controlled_change(experiment, source)
    declared_source = REPOSITORY_ROOT / str(controlled.get("feature_source_manifest", ""))
    if declared_source.resolve() != source["path"]:
        raise ValueError("Visible derivation source differs from the controlled config.")
    records = _visible_source_shard_records(source, experiment)
    source_identity = source["identity"]
    dataset_config = load_dataset_config(
        REPOSITORY_ROOT / str(experiment["dataset"]["config"])
    )
    contract_path = REPOSITORY_ROOT / str(experiment["action_contract"])
    source_selection = source_identity.get("selection", {})
    source_model = source_identity.get("model", {})
    source_feature = source_identity.get("feature", {})
    source_dataset = source_identity.get("dataset", {})
    if (
        source_identity.get("split") != experiment["dataset"]["split"]
        or source_dataset.get("repo_id") != dataset_config.repo_id
        or source_dataset.get("revision") != dataset_config.revision
        or source_dataset.get("episodes") != list(dataset_config.episodes)
        or source_model.get("identifier") != experiment["backbone"]["identifier"]
        or source_model.get("adaptation") != experiment["backbone"]["adaptation"]
        or source_feature.get("pooling") != experiment["backbone"]["pooling"]
        or source_feature.get("layer") != experiment["backbone"]["feature_layer"]
        or source_identity.get("processor") != experiment["backbone"]["processor"]
        or int(source_selection.get("frame_stride", -1))
        != int(experiment["dataset"]["frame_stride"])
        or int(source_selection.get("action_chunk_length", -1))
        != dataset_config.chunk_size
        or source_identity.get("action_contract_sha256") != file_sha256(contract_path)
    ):
        raise ValueError("Visible derivation source and target feature semantics differ.")
    if source["normalization"].get("source_split") != "train":
        raise ValueError("Visible derivation requires train-only normalization.")

    visible_keys = {
        (split, int(episode))
        for split in VISIBLE_MATERIALIZED_SPLITS
        for episode in experiment["dataset"]["split"][split]
    }
    if set(records) != visible_keys:
        raise ValueError("Visible source cache differs from the exact controlled scope.")
    feature_dim = _pooling_feature_dim(
        str(experiment["backbone"]["pooling"]), int(source_model["hidden_size"])
    )
    contract = load_action_contract(contract_path)
    ordered_keys = sorted(visible_keys, key=lambda value: value[1])
    validated_shards = {
        key: _load_composition_shard(
            source,
            records[key],
            split=key[0],
            episode=key[1],
            expected_feature_dim=feature_dim,
            contract=contract,
        )
        for key in ordered_keys
    }

    derivation = {
        "schema_version": 1,
        "type": "verified_visible_identity_rebind_v1",
        "tensor_transform": "identity",
        "materialized_splits": ["train", "validation"],
        "withheld_splits": ["test"],
        "hidden_test_loaded": False,
        "hidden_test_materialized": False,
        "changed_axis": controlled["changed_axis"],
        "reference_value": controlled.get("reference_value"),
        "candidate_value": controlled.get("candidate_value"),
        "source": {
            "experiment_id": source_identity["experiment_id"],
            "pooling": source["pooling"],
            "identity_hash": source["manifest"]["identity_hash"],
            "manifest_sha256": file_sha256(source["path"]),
        },
    }
    identity = copy.deepcopy(source_identity)
    identity["experiment_id"] = experiment["experiment_id"]
    identity["experiment_config_sha256"] = file_sha256(config_path)
    identity["code"] = workspace_code_identity(REPOSITORY_ROOT)
    identity["feature"]["execution_device"] = "derived_from_visible_source_cache"
    identity["feature"]["derivation"] = derivation
    identity_hash = stable_hash(identity)
    root = _feature_root() / experiment["experiment_id"] / identity_hash[:16]
    root.mkdir(parents=True, exist_ok=True)
    create_json(root / "identity.json", identity)
    create_json(root / "normalization.json", source["normalization"])
    shard_records: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    started = time.perf_counter()
    for split, episode in ordered_keys:
        value = validated_shards[(split, episode)]
        payload = {
            "schema_version": 1,
            "identity_hash": identity_hash,
            "split": split,
            "episode": episode,
            **{key: value[key] for key in SHARD_TENSOR_KEYS},
            "action_transform": value["action_transform"],
        }
        relative = Path("shards") / split / f"episode-{episode:03d}.pt"
        path = root / relative
        if path.exists():
            _validate_existing_composition_shard(path, payload)
        else:
            save_tensor_shard(path, payload)
        shard_records[split].append(
            {
                "episode": episode,
                "path": relative.as_posix(),
                "samples": int(value["features"].shape[0]),
                "sha256": file_sha256(path),
                "action_transform": value["action_transform"],
            }
        )
        print(f"Materialized visible shard {relative}", flush=True)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "identity_hash": identity_hash,
        "identity": identity,
        "normalization_path": "normalization.json",
        "normalization_sha256": file_sha256(root / "normalization.json"),
        "derivation": derivation,
        "shards": shard_records,
        "samples": {
            split: sum(record["samples"] for record in split_records)
            for split, split_records in shard_records.items()
        },
        "hidden_test_materialized": False,
        "hidden_test_loaded": False,
        "materialized_splits": ["train", "validation"],
        "withheld_splits": ["test"],
    }
    create_json(root / "manifest.json", manifest)
    print("Visible-only derived frozen feature cache complete")
    print(f"Cache identity: {identity_hash}")
    print(f"Samples: {json.dumps(manifest['samples'], sort_keys=True)}")
    print(f"Elapsed seconds: {time.perf_counter() - started:.1f}")
    return 0


def inspect() -> int:
    """Read complete cache manifests without loading features or models."""

    root = _feature_root()
    manifests = sorted(root.glob("*/[0-9a-f]*/manifest.json")) if root.exists() else []
    if not manifests:
        print("No complete frozen feature cache found.")
        return 1
    for path in manifests:
        value = json.loads(path.read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    "experiment_id": value.get("identity", {}).get("experiment_id"),
                    "identity_hash": value.get("identity_hash"),
                    "samples": value.get("samples"),
                    "status": value.get("status"),
                },
                sort_keys=True,
            )
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "smoke",
            "smoke-visible",
            "build",
            "build-visible",
            "compose",
            "derive",
            "derive-visible",
            "inspect",
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-manifest", action="append", type=Path, default=[])
    args = parser.parse_args()
    if args.command == "inspect":
        return inspect()
    config_path = args.config.resolve()
    if args.command == "smoke":
        return smoke(config_path)
    if args.command == "smoke-visible":
        return smoke_visible(config_path)
    if args.command == "build-visible":
        return build_visible(config_path)
    if args.command == "compose":
        return compose(config_path, args.source_manifest)
    if args.command == "derive":
        return derive(config_path, args.source_manifest)
    if args.command == "derive-visible":
        return derive_visible(config_path, args.source_manifest)
    return build(config_path)


if __name__ == "__main__":
    raise SystemExit(main())
