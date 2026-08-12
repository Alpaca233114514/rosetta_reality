"""Evaluate a verified M2 artifact on validation or the once-hidden test split."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

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
from rosetta_reality.experiment import (  # noqa: E402
    file_sha256,
    load_experiment_config,
    validate_frozen_artifact_recipe,
)
from rosetta_reality.features import (  # noqa: E402
    CachedFeatureDataset,
    create_json,
    load_feature_manifest,
)
from rosetta_reality.sim import load_action_contract  # noqa: E402
from rosetta_reality.train.m2 import build_cached_policy, predict_denormalized  # noqa: E402


def _run_root() -> Path:
    value = os.environ.get("ROSETTA_RUN_ROOT")
    return Path(value) if value else REPOSITORY_ROOT / "runs"


def _matching_action_contract(experiment: dict, artifact_root: Path):
    """Load the configured contract only when it exactly matches the artifact."""

    artifact_contract = json.loads(
        (artifact_root / "action_contract.json").read_text(encoding="utf-8")
    )
    contract = load_action_contract(REPOSITORY_ROOT / experiment["action_contract"])
    canonical_contract = json.loads(json.dumps(asdict(contract), allow_nan=False))
    if artifact_contract != canonical_contract:
        raise ValueError("Exported Action Contract differs from the evaluation config.")
    return contract


def evaluate(
    *,
    config_path: Path,
    feature_manifest_path: Path,
    artifact_root: Path,
    split: str,
) -> Path:
    """Load a public-style artifact path and evaluate physical action metrics."""

    experiment = load_experiment_config(config_path, REPOSITORY_ROOT)
    feature_manifest = load_feature_manifest(feature_manifest_path)
    artifact_manifest_path = artifact_root / "manifest.json"
    artifact_manifest = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
    if artifact_manifest.get("status") != "verified":
        raise ValueError("Evaluation requires a verified exported artifact.")
    for name, expected in artifact_manifest["files"].items():
        if file_sha256(artifact_root / name) != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}.")
    artifact_config = json.loads((artifact_root / "config.json").read_text(encoding="utf-8"))
    if artifact_manifest.get("experiment_id") != experiment["experiment_id"]:
        raise ValueError("Artifact manifest and experiment identifiers differ.")
    if artifact_config["feature_cache_identity"] != feature_manifest["identity_hash"]:
        raise ValueError("Artifact and feature-cache identities differ.")
    validate_frozen_artifact_recipe(
        experiment,
        artifact_config,
        context="Evaluation artifact",
    )
    contract = _matching_action_contract(experiment, artifact_root)
    normalization = json.loads((artifact_root / "normalization.json").read_text(encoding="utf-8"))
    if normalization.get("source_split") != "train":
        raise ValueError("Artifact normalization did not originate from train only.")
    statistics = DatasetStatistics.from_dict(normalization["statistics"])
    model_payload = torch.load(artifact_root / "model.pt", map_location="cpu", weights_only=True)
    model_contract = model_payload["model_contract"]
    model = build_cached_policy(
        experiment,
        feature_dim=int(model_contract["feature_dim"]),
        state_dim=int(model_contract["state_dim"]),
        action_dim=int(model_contract["action_dim"]),
        chunk_size=int(model_contract["chunk_size"]),
        statistics=statistics,
    )
    model.load_state_dict(model_payload["model_state"], strict=True)
    dataset = CachedFeatureDataset(feature_manifest_path, split)
    loader = DataLoader(
        dataset,
        batch_size=int(experiment["training"]["batch_size"]),
        shuffle=False,
    )
    predicted, target, raw_predicted = predict_denormalized(
        model,
        loader,
        statistics,
        contract,
    )
    metrics = action_metrics(
        predicted,
        target,
        contract,
        statistics.action,
        raw_predicted=raw_predicted,
    )
    report = {
        "schema_version": 1,
        "status": "complete",
        "experiment_id": experiment["experiment_id"],
        "artifact_id": artifact_manifest["artifact_id"],
        "artifact_manifest_sha256": file_sha256(artifact_manifest_path),
        "feature_cache_identity": feature_manifest["identity_hash"],
        "split": split,
        "hidden_test_opened": split == "test",
        "metrics": metrics,
    }
    destination = (
        _run_root()
        / experiment["experiment_id"]
        / "evaluation"
        / f"{artifact_manifest['artifact_id']}-{split}.json"
    )
    create_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return destination


def main() -> int:
    """Parse artifact and split selection."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    args = parser.parse_args()
    destination = evaluate(
        config_path=args.config.resolve(),
        feature_manifest_path=args.feature_manifest.resolve(),
        artifact_root=args.artifact.resolve(),
        split=args.split,
    )
    print(f"Evaluation report: {destination.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
