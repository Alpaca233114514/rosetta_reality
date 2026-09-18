"""Run immutable pre-training baselines on validation episodes only."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
DEFAULT_CONFIG = (
    REPOSITORY_ROOT / "configs" / "experiments" / "m2_qwen08b_frozen_001.yaml"
)
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rosetta_reality.data.normalization import DatasetStatistics  # noqa: E402
from rosetta_reality.eval import action_metrics  # noqa: E402
from rosetta_reality.experiment import file_sha256, load_experiment_config  # noqa: E402
from rosetta_reality.features import (  # noqa: E402
    CachedFeatureDataset,
    create_json,
    load_feature_manifest,
)
from rosetta_reality.sim import load_action_contract  # noqa: E402
from rosetta_reality.train.m2 import build_cached_policy, predict_denormalized  # noqa: E402


def _feature_root() -> Path:
    value = os.environ.get("ROSETTA_FEATURE_ROOT")
    return Path(value) if value else REPOSITORY_ROOT / "feature_cache"


def _run_root() -> Path:
    value = os.environ.get("ROSETTA_RUN_ROOT")
    return Path(value) if value else REPOSITORY_ROOT / "runs"


def _manifest_path(experiment_id: str, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    candidates = sorted((_feature_root() / experiment_id).glob("*/manifest.json"))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one complete feature cache for {experiment_id!r}, "
            f"received {len(candidates)}."
        )
    return candidates[0]


def _statistics(manifest_path: Path, manifest: dict[str, Any]) -> DatasetStatistics:
    normalization_path = manifest_path.parent / str(manifest["normalization_path"])
    if file_sha256(normalization_path) != manifest["normalization_sha256"]:
        raise ValueError("Feature-cache normalization checksum mismatch.")
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    if normalization.get("source_split") != "train":
        raise ValueError("Normalization statistics were not computed from train only.")
    return DatasetStatistics.from_dict(normalization["statistics"])


def benchmark(config_path: Path, feature_manifest: Path | None) -> Path:
    """Evaluate three pre-training baselines without touching hidden test shards."""

    experiment = load_experiment_config(config_path, REPOSITORY_ROOT)
    manifest_path = _manifest_path(experiment["experiment_id"], feature_manifest)
    manifest = load_feature_manifest(manifest_path)
    identity = manifest["identity"]
    if identity["experiment_id"] != experiment["experiment_id"]:
        raise ValueError("Feature cache belongs to a different experiment.")
    if identity["experiment_config_sha256"] != file_sha256(config_path):
        raise ValueError("Feature cache was created from a different experiment config.")
    statistics = _statistics(manifest_path, manifest)
    validation = CachedFeatureDataset(manifest_path, "validation")
    contract = load_action_contract(REPOSITORY_ROOT / experiment["action_contract"])
    targets = validation.actions
    _, target_violations = contract.clip(targets)
    if bool(target_violations.any()):
        raise ValueError("Validation targets were not transformed into the Action Contract.")

    raw_train_mean = statistics.action.mean.view(1, 1, -1).expand_as(targets)
    train_mean, _ = contract.clip(raw_train_mean)
    raw_persistence = validation.robot_state[:, None, :].expand_as(targets)
    persistence, _ = contract.clip(raw_persistence)

    seed = int(experiment["training"]["seed"])
    torch.manual_seed(seed)
    model = build_cached_policy(
        experiment,
        feature_dim=validation.features.shape[-1],
        state_dim=validation.robot_state.shape[-1],
        action_dim=validation.actions.shape[-1],
        chunk_size=validation.actions.shape[-2],
        statistics=statistics,
    )
    loader = DataLoader(
        validation,
        batch_size=int(experiment["training"]["batch_size"]),
        shuffle=False,
    )
    untrained_prediction, untrained_target, raw_untrained_prediction = predict_denormalized(
        model,
        loader,
        statistics,
        contract,
    )
    if not torch.allclose(untrained_target, targets, atol=1e-6, rtol=1e-6):
        raise RuntimeError("Validation target order changed during benchmark loading.")

    metrics = {
        "train_action_mean": action_metrics(
            train_mean,
            targets,
            contract,
            statistics.action,
            raw_predicted=raw_train_mean,
        ),
        "current_state_persistence": action_metrics(
            persistence,
            targets,
            contract,
            statistics.action,
            raw_predicted=raw_persistence,
        ),
        "deterministic_untrained_policy": action_metrics(
            untrained_prediction,
            targets,
            contract,
            statistics.action,
            raw_predicted=raw_untrained_prediction,
        ),
    }
    declared = set(experiment["benchmark"]["baselines"])
    if set(metrics) != declared:
        raise ValueError("Implemented baseline set differs from the experiment declaration.")
    report = {
        "schema_version": 1,
        "status": "complete",
        "stage": "pre_training",
        "experiment_id": experiment["experiment_id"],
        "experiment_config_sha256": file_sha256(config_path),
        "feature_cache_identity": manifest["identity_hash"],
        "feature_manifest_sha256": file_sha256(manifest_path),
        "normalization_source_split": "train",
        "evaluated_split": "validation",
        "hidden_test_loaded": False,
        "seed": seed,
        "samples": len(validation),
        "metrics": metrics,
    }
    destination = (
        _run_root()
        / experiment["experiment_id"]
        / "benchmark"
        / f"pre-training-{manifest['identity_hash'][:16]}.json"
    )
    create_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--feature-manifest", type=Path)
    args = parser.parse_args()
    destination = benchmark(args.config.resolve(), args.feature_manifest)
    print(f"Benchmark report: {destination.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
