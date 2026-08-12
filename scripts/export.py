"""Export and independently reload a frozen-backbone M2 action artifact."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
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

from rosetta_reality.data.normalization import DatasetStatistics  # noqa: E402
from rosetta_reality.experiment import (  # noqa: E402
    file_sha256,
    frozen_artifact_recipe,
    load_experiment_config,
    workspace_code_identity,
)
from rosetta_reality.features import (  # noqa: E402
    CachedFeatureDataset,
    create_json,
    load_feature_manifest,
    save_tensor_shard,
)
from rosetta_reality.sim import load_action_contract  # noqa: E402
from rosetta_reality.train.m2 import build_cached_policy, normalized_batch  # noqa: E402


def _artifact_root() -> Path:
    value = os.environ.get("ROSETTA_ARTIFACT_ROOT")
    return Path(value) if value else REPOSITORY_ROOT / "artifacts"


def _model(
    experiment: dict[str, Any],
    model_contract: dict[str, Any],
    statistics: DatasetStatistics,
) -> torch.nn.Module:
    return build_cached_policy(
        experiment,
        feature_dim=int(model_contract["feature_dim"]),
        state_dim=int(model_contract["state_dim"]),
        action_dim=int(model_contract["action_dim"]),
        chunk_size=int(model_contract["chunk_size"]),
        statistics=statistics,
    )


def _write_text(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite artifact text: {path}.")
    with path.open("x", encoding="utf-8", newline="\n") as file:
        file.write(content)


def export(
    *,
    config_path: Path,
    feature_manifest_path: Path,
    training_manifest_path: Path,
    checkpoint_path: Path,
    artifact_id: str,
) -> Path:
    """Export only Rosetta components, then reload them through a separate path."""

    experiment = load_experiment_config(config_path, REPOSITORY_ROOT)
    feature_manifest = load_feature_manifest(feature_manifest_path)
    training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    experiment_id = experiment["experiment_id"]
    config_sha256 = file_sha256(config_path)
    feature_identity = feature_manifest.get("identity", {})
    if (
        feature_identity.get("experiment_id") != experiment_id
        or feature_identity.get("experiment_config_sha256") != config_sha256
    ):
        raise ValueError("Feature cache does not belong to the requested experiment.")
    if training_manifest.get("status") != "complete":
        raise ValueError("Only an accepted complete training run may be exported.")
    if (
        training_manifest.get("experiment_id") != experiment_id
        or training_manifest.get("experiment_config_sha256") != config_sha256
    ):
        raise ValueError("Training manifest does not belong to the requested experiment.")
    if training_manifest.get("feature_cache_identity") != feature_manifest["identity_hash"]:
        raise ValueError("Training and feature-cache identities differ.")
    if checkpoint_path.name != training_manifest.get("best_checkpoint"):
        raise ValueError("Export checkpoint is not the recorded best checkpoint.")
    expected_checkpoint_hash = training_manifest["checkpoints"].get(checkpoint_path.name)
    if file_sha256(checkpoint_path) != expected_checkpoint_hash:
        raise ValueError("Best checkpoint checksum differs from the training manifest.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("experiment_id") != experiment_id
        or checkpoint.get("experiment_config_sha256") != config_sha256
    ):
        raise ValueError("Checkpoint does not belong to the requested experiment.")
    if checkpoint.get("feature_cache_identity") != feature_manifest["identity_hash"]:
        raise ValueError("Checkpoint and feature-cache identities differ.")
    if checkpoint.get("state_pairing") != training_manifest.get("state_pairing"):
        raise ValueError("Checkpoint and training state-pairing identities differ.")
    if checkpoint.get("action_loss_protocol") != training_manifest.get(
        "action_loss_protocol"
    ):
        raise ValueError("Checkpoint and training action-loss protocols differ.")

    normalization_path = feature_manifest_path.parent / feature_manifest["normalization_path"]
    if file_sha256(normalization_path) != feature_manifest["normalization_sha256"]:
        raise ValueError("Normalization checksum differs from the feature manifest.")
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    statistics = DatasetStatistics.from_dict(normalization["statistics"])
    model_contract = checkpoint["model_contract"]
    model = _model(experiment, model_contract, statistics)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    validation = CachedFeatureDataset(feature_manifest_path, "validation")
    raw_batch = {
        "features": validation.features[:2],
        "robot_state": validation.robot_state[:2],
        "actions": validation.actions[:2],
    }
    observations, state, _ = normalized_batch(raw_batch, statistics)
    with torch.inference_mode():
        reference_prediction = model(observations, state)

    destination = _artifact_root() / experiment["experiment_id"] / artifact_id
    destination.mkdir(parents=True, exist_ok=False)
    model_path = destination / "model.pt"
    save_tensor_shard(
        model_path,
        {
            "schema_version": 1,
            "experiment_id": experiment["experiment_id"],
            "base_model": experiment["backbone"]["identifier"],
            "base_model_included": False,
            "adaptation": "frozen",
            "feature_cache_identity": feature_manifest["identity_hash"],
            "training_run_id": training_manifest["run_id"],
            "training_epoch": checkpoint["epoch"],
            "state_pairing": training_manifest.get("state_pairing"),
            "action_loss_protocol": training_manifest.get("action_loss_protocol"),
            "model_contract": model_contract,
            "model_state": checkpoint["model_state"],
        },
    )
    artifact_config = {
        "schema_version": 1,
        **frozen_artifact_recipe(experiment),
        "base_model_revision": feature_manifest["identity"]["model"]["revision"],
        "base_model_file_hashes": feature_manifest["identity"]["model"]["files"],
        "base_model_included": False,
        "state_pairing": training_manifest.get("state_pairing"),
        "action_loss_protocol": training_manifest.get("action_loss_protocol"),
        "model_contract": model_contract,
        "feature_cache_identity": feature_manifest["identity_hash"],
    }
    create_json(destination / "config.json", artifact_config)
    create_json(destination / "normalization.json", normalization)
    contract = load_action_contract(REPOSITORY_ROOT / experiment["action_contract"])
    create_json(destination / "action_contract.json", asdict(contract))

    loaded = torch.load(model_path, map_location="cpu", weights_only=True)
    reloaded_model = _model(experiment, loaded["model_contract"], statistics)
    reloaded_model.load_state_dict(loaded["model_state"], strict=True)
    reloaded_model.eval()
    with torch.inference_mode():
        reloaded_prediction = reloaded_model(observations, state)
    maximum_difference = float((reference_prediction - reloaded_prediction).abs().max())
    reload_verified = torch.equal(reference_prediction, reloaded_prediction)
    if not reload_verified:
        raise RuntimeError("Reloaded artifact predictions differ from the checkpoint model.")

    model_card = f"""# Rosetta Reality {artifact_id}

Experimental research-only development artifact for simulated ALOHA insertion.
It has not been validated on a physical robot and must not be represented as an
autonomous real-robot controller.

- Base model: `{experiment['backbone']['identifier']}` (required separately; not included)
- Adaptation: frozen backbone with pooled offline features
- Action prediction parameterization:
  `{experiment['action_expert'].get('prediction_parameterization', 'absolute')}`
- Output projection: clip every decoded action to the recorded Action Contract
- Dataset: `{feature_manifest['identity']['dataset']['repo_id']}` at immutable revision
  `{feature_manifest['identity']['dataset']['revision']}`
- Inputs: top-camera image, language instruction, and 14-dimensional robot state
- Outputs: 8 absolute 14-dimensional joint-position target actions at 50 Hz
- Train-only simulator-state pairing: {training_manifest.get('state_pairing') is not None}
- Action-loss protocol: `{training_manifest.get('action_loss_protocol')}`
- Best validation {training_manifest['primary_metric']}:
  {training_manifest['best_validation_value']:.8f}
- Declared baseline {training_manifest['baseline']}:
  {training_manifest['baseline_value']:.8f}
- Artifact reload: exact tensor equality verified

Limitations: development-scale data and simulation only; no physical-robot safety
validation, no guarantee of task success, and no claim of cross-embodiment transfer.
"""
    _write_text(destination / "MODEL_CARD.md", model_card)
    files = {
        path.name: file_sha256(path)
        for path in sorted(destination.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "schema_version": 1,
        "status": "verified",
        "artifact_id": artifact_id,
        "experiment_id": experiment["experiment_id"],
        "training_manifest_sha256": file_sha256(training_manifest_path),
        "source_checkpoint_sha256": file_sha256(checkpoint_path),
        "feature_manifest_sha256": file_sha256(feature_manifest_path),
        "code": workspace_code_identity(REPOSITORY_ROOT),
        "reload": {
            "verified": reload_verified,
            "samples": reference_prediction.shape[0],
            "maximum_absolute_difference": maximum_difference,
        },
        "files": files,
    }
    create_json(destination / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return destination


def main() -> int:
    """Parse explicit source identities and export a new artifact directory."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact-id", required=True)
    args = parser.parse_args()
    destination = export(
        config_path=args.config.resolve(),
        feature_manifest_path=args.feature_manifest.resolve(),
        training_manifest_path=args.training_manifest.resolve(),
        checkpoint_path=args.checkpoint.resolve(),
        artifact_id=args.artifact_id,
    )
    print(f"Artifact directory: {destination.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
