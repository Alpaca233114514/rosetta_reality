"""Create immutable train-only SmolVLA normalization statistics from selected parquet rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs/vla/smolvla_450m_aloha_insertion.yaml"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rosetta_reality.experiment import file_sha256, stable_hash  # noqa: E402
from rosetta_reality.features import create_json  # noqa: E402
from rosetta_reality.sim import load_action_contract  # noqa: E402
from rosetta_reality.vla import (  # noqa: E402
    load_smolvla_action_space,
    load_smolvla_experiment,
)
from rosetta_reality.vla.processor import (  # noqa: E402
    standard_aloha_action_to_model,
    standard_aloha_state_to_pi,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping: {path.name}.")
    return value


def _dataset_root(experiment: dict[str, Any]) -> Path:
    raw = os.environ.get("ROSETTA_DATA_ROOT")
    root = Path(raw) if raw else REPOSITORY_ROOT / "data/lerobot_m2"
    if raw and not root.is_absolute():
        raise ValueError("ROSETTA_DATA_ROOT must be absolute.")
    repo_id = str(experiment["dataset"]["identifier"]).replace("/", "--")
    return root / repo_id / str(experiment["dataset"]["revision"])


def _statistics(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    if values.ndim != 2 or len(values) <= 0 or not np.isfinite(values).all():
        raise ValueError("Train-only normalization input must be a finite non-empty matrix.")
    standard_deviation = values.std(axis=0, ddof=0)
    if not np.isfinite(standard_deviation).all() or np.any(standard_deviation <= 0):
        raise ValueError("Train-only normalization has a non-positive standard deviation.")
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": standard_deviation.tolist(),
        "count": [int(len(values))],
    }


def _safe_relative(path: str) -> Path:
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Dataset checksum manifest contains an unsafe path.")
    return relative


def _json_file_sha256(payload: dict[str, Any]) -> str:
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _validate_view(
    view_root: Path,
    normalization_report: Path,
    expected_files: dict[str, str],
) -> None:
    manifest_path = view_root / "view_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("status") != "complete"
        or manifest.get("stage") != "smolvla_train_only_dataset_view"
        or manifest.get("normalization_report_sha256") != file_sha256(normalization_report)
        or manifest.get("files") != expected_files
    ):
        raise ValueError("Existing train-only dataset view identity is invalid.")
    for relative_name, expected_sha256 in expected_files.items():
        path = view_root / _safe_relative(relative_name)
        if not path.is_file() or file_sha256(path) != expected_sha256:
            raise ValueError("Existing train-only dataset view checksum is invalid.")


def _create_view(
    source_root: Path,
    view_root: Path,
    normalization_report: Path,
    view_stats: dict[str, Any],
    source_checksums: dict[str, str],
) -> None:
    copied_files = {
        name: sha256 for name, sha256 in source_checksums.items() if name != "meta/stats.json"
    }
    view_stats_payload = view_stats
    view_stats_sha256 = _json_file_sha256(view_stats_payload)
    expected_files = {**copied_files, "meta/stats.json": view_stats_sha256}
    if view_root.exists():
        _validate_view(view_root, normalization_report, expected_files)
        return
    partial = view_root.with_name(f"{view_root.name}.{uuid.uuid4().hex}.partial")
    partial.mkdir(parents=True, exist_ok=False)
    for relative_name, expected_sha256 in copied_files.items():
        relative = _safe_relative(relative_name)
        source = source_root / relative
        destination = partial / relative
        if not source.is_file() or file_sha256(source) != expected_sha256:
            raise ValueError("Source dataset checksum changed while creating the train-only view.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        # DrvFS and other bind-mounted filesystems may reject metadata updates
        # (`copystat`/`utime`) even though byte-for-byte copies are supported.
        # Dataset-view identity is checksum based, so metadata preservation is
        # neither required nor desirable here.
        shutil.copyfile(source, destination)
        if file_sha256(destination) != expected_sha256:
            raise ValueError("Copied train-only dataset view file failed checksum verification.")
    stats_path = partial / "meta/stats.json"
    create_json(stats_path, view_stats_payload)
    if file_sha256(stats_path) != view_stats_sha256:
        raise ValueError("Train-only dataset view stats checksum is invalid.")
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "stage": "smolvla_train_only_dataset_view",
        "normalization_report_sha256": file_sha256(normalization_report),
        "source_cache_checksums_sha256": file_sha256(source_root / "cache_checksums.json"),
        "files": expected_files,
        "validation_episodes_loaded": False,
        "hidden_test_loaded": False,
    }
    create_json(partial / "view_manifest.json", manifest)
    partial.rename(view_root)
    _validate_view(view_root, normalization_report, expected_files)


def prepare(config_path: Path) -> Path:
    import pyarrow.dataset as arrow_dataset
    from lerobot.datasets.factory import IMAGENET_STATS

    experiment = load_smolvla_experiment(config_path, REPOSITORY_ROOT)
    dataset_config_path = REPOSITORY_ROOT / str(experiment["dataset"]["config"])
    dataset_config = _load_yaml(dataset_config_path)
    contract_path = REPOSITORY_ROOT / str(experiment["action_contract"]["derived"])
    contract = load_action_contract(contract_path)
    action_space = load_smolvla_action_space(experiment)
    train_episodes = [int(value) for value in experiment["dataset"]["train_episodes"]]
    validation_episodes = {int(value) for value in experiment["dataset"]["validation_episodes"]}
    test_episodes = {int(value) for value in experiment["dataset"]["test_episodes"]}
    if (
        len(train_episodes) != len(set(train_episodes))
        or set(train_episodes) & validation_episodes
        or set(train_episodes) & test_episodes
        or validation_episodes & test_episodes
    ):
        raise ValueError("Train, validation and hidden-test episode identities must be disjoint.")
    root = _dataset_root(experiment)
    manifest_path = root / "manifest.json"
    metadata_stats_path = root / "meta/stats.json"
    cache_checksums_path = root / "cache_checksums.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("resolved_revision") != experiment["dataset"]["revision"]
        or manifest.get("repo_id") != experiment["dataset"]["identifier"]
    ):
        raise ValueError("Prepared dataset identity differs from the VLA experiment.")
    fields = dataset_config["fields"]
    dataset = arrow_dataset.dataset(root / "data", format="parquet")
    table = dataset.to_table(
        columns=[fields["state"], fields["action"], fields["episode_index"]],
        filter=arrow_dataset.field(fields["episode_index"]).isin(train_episodes),
    )
    episode_ids = np.asarray(table.column(fields["episode_index"]).to_pylist(), dtype=np.int64)
    loaded_episodes = sorted({int(value) for value in episode_ids.tolist()})
    if loaded_episodes != sorted(train_episodes):
        raise ValueError("Train-only normalization materialized an unexpected episode set.")
    if any(value in validation_episodes or value in test_episodes for value in episode_ids):
        raise ValueError("Train-only normalization crossed the validation or hidden-test boundary.")
    states = np.asarray(table.column(fields["state"]).to_pylist(), dtype=np.float64)
    actions = np.asarray(table.column(fields["action"]).to_pylist(), dtype=np.float64)
    if states.shape != actions.shape or states.ndim != 2 or states.shape[1] != contract.dimension:
        raise ValueError("Train-only state/action shape differs from the Action Contract.")
    lower = contract.lower_bounds.numpy().astype(np.float64, copy=False)
    upper = contract.upper_bounds.numpy().astype(np.float64, copy=False)
    tolerances = contract.source_overshoot_tolerances.numpy().astype(np.float64, copy=False)
    below = np.maximum(lower - actions, 0.0)
    above = np.maximum(actions - upper, 0.0)
    overshoot = np.maximum(below, above)
    projection_mask = overshoot > 0.0
    if action_space.target_projection == "action_contract_clip":
        if np.any(overshoot > tolerances + 1e-12):
            raise ValueError(
                "Train-only source action exceeds the Action Contract overshoot tolerance."
            )
        actions = np.maximum(np.minimum(actions, upper), lower)
    standard_action_ranges = {
        "minimum": actions.min(axis=0).tolist(),
        "maximum": actions.max(axis=0).tolist(),
    }
    states = standard_aloha_state_to_pi(torch.from_numpy(states)).numpy()
    actions = standard_aloha_action_to_model(
        torch.from_numpy(actions), action_space.representation_adapter
    ).numpy()
    projection_diagnostics = {
        "mode": action_space.target_projection,
        "stage": action_space.target_projection_stage,
        "source_element_count": int(projection_mask.size),
        "projected_element_count": int(projection_mask.sum()),
        "projected_element_rate": float(projection_mask.mean()),
        "projected_row_count": int(projection_mask.any(axis=1).sum()),
        "per_dimension_projected_count": projection_mask.sum(axis=0).astype(int).tolist(),
        "per_dimension_maximum_overshoot": overshoot.max(axis=0).tolist(),
    }
    representation_diagnostics = {
        "name": action_space.representation_adapter,
        "stage": action_space.representation_adapter_stage,
        "upstream_revision": experiment["upstream"]["revision"],
        "standard_action_range": standard_action_ranges,
        "internal_action_range": {
            "minimum": actions.min(axis=0).tolist(),
            "maximum": actions.max(axis=0).tolist(),
        },
    }
    effective_stats = {
        fields["state"]: _statistics(states),
        fields["action"]: _statistics(actions),
    }
    camera_fields = sorted(str(value) for value in dataset_config["cameras"].values())
    visual_statistics = json.loads(json.dumps(IMAGENET_STATS))
    view_stats = {**effective_stats, **{key: visual_statistics for key in camera_fields}}
    if not all(
        math.isfinite(float(value))
        for stats in effective_stats.values()
        for key in ("min", "max", "mean", "std")
        for value in stats[key]
    ):
        raise FloatingPointError("Train-only normalization produced a non-finite statistic.")
    identity = {
        "schema_version": 1,
        "status": "complete",
        "stage": "smolvla_train_only_normalization",
        "experiment_id": experiment["experiment_id"],
        "experiment_config_sha256": file_sha256(config_path),
        "dataset_config_sha256": file_sha256(dataset_config_path),
        "dataset_id": experiment["dataset"]["identifier"],
        "dataset_revision": experiment["dataset"]["revision"],
        "dataset_manifest_sha256": file_sha256(manifest_path),
        "source_metadata_stats_sha256": file_sha256(metadata_stats_path),
        "source_cache_checksums_sha256": file_sha256(cache_checksums_path),
        "action_contract_sha256": file_sha256(contract_path),
        "action_space": action_space.as_dict(),
        "target_projection": projection_diagnostics,
        "representation_adapter": representation_diagnostics,
        "source_split": "train",
        "train_episodes": train_episodes,
        "train_rows": int(len(episode_ids)),
        "validation_episodes_loaded": False,
        "hidden_test_loaded": False,
        "population_standard_deviation": True,
        "visual_statistics_policy": "imagenet_constants",
        "visual_statistics_source": "fixed_constants_not_dataset_rows",
        "visual_features": camera_fields,
        "visual_statistics": visual_statistics,
        "effective_stats": effective_stats,
    }
    json.dumps(identity, allow_nan=False)
    digest = stable_hash(identity)[:16]
    run_root = Path(os.environ.get("ROSETTA_RUN_ROOT", REPOSITORY_ROOT / "runs"))
    if not run_root.is_absolute():
        run_root = (REPOSITORY_ROOT / run_root).resolve()
    destination = (
        run_root / str(experiment["experiment_id"]) / "normalization" / f"train-only-{digest}.json"
    )
    view_root = (
        run_root / str(experiment["experiment_id"]) / "dataset_views" / f"train-only-{digest}"
    )
    report = {
        **identity,
        "dataset_view": view_root.relative_to(run_root).as_posix(),
        "dataset_view_manifest": (
            view_root.relative_to(run_root) / "view_manifest.json"
        ).as_posix(),
    }
    create_json(destination, report)
    checksum_manifest = json.loads(cache_checksums_path.read_text(encoding="utf-8"))
    source_checksums = checksum_manifest.get("files")
    if not isinstance(source_checksums, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in source_checksums.items()
    ):
        raise ValueError("Source dataset checksum manifest is invalid.")
    _create_view(root, view_root, destination, view_stats, source_checksums)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report: {destination.relative_to(REPOSITORY_ROOT).as_posix()}")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    prepare(args.config.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
