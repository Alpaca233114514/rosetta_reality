"""Compute immutable train-only action baselines before any SmolVLA optimizer step."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
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


def _metric(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = predicted - target
    return {
        "action_mae": float(np.abs(error).mean()),
        "action_rmse": float(np.sqrt(np.square(error).mean())),
        "first_action_mae": float(np.abs(error[:, 0]).mean()),
    }


def _chunks(values: np.ndarray, episode_ids: np.ndarray, chunk_size: int) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for episode in dict.fromkeys(int(value) for value in episode_ids):
        episode_values = values[episode_ids == episode]
        if not len(episode_values):
            continue
        for index in range(len(episode_values)):
            chunk = episode_values[index : index + chunk_size]
            if len(chunk) < chunk_size:
                padding = np.repeat(chunk[-1:], chunk_size - len(chunk), axis=0)
                chunk = np.concatenate((chunk, padding), axis=0)
            pieces.append(chunk)
    if not pieces:
        raise ValueError("Validation split did not produce action chunks.")
    return np.stack(pieces)


def _project_actions(
    actions: np.ndarray,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    tolerances: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the registered source-tolerance check and physical target projection."""

    overshoot = np.maximum(np.maximum(lower - actions, actions - upper), 0.0)
    if np.any(overshoot > tolerances + 1e-6):
        index = np.unravel_index(int(np.argmax(overshoot - tolerances)), actions.shape)
        raise ValueError(
            "Benchmark action exceeds the registered source overshoot tolerance "
            f"at row {index[0]}, dimension {index[1]}."
        )
    projected = np.clip(actions, lower, upper)
    changed = projected != actions
    return projected, {
        "mode": "action_contract_clip",
        "stage": "before_baseline_statistics_and_metrics",
        "projected_elements": int(changed.sum()),
        "total_elements": int(changed.size),
        "projection_rate": float(changed.mean()),
        "maximum_source_overshoot": float(overshoot.max(initial=0.0)),
    }


def benchmark(config_path: Path) -> Path:
    import pyarrow.dataset as arrow_dataset

    experiment = load_smolvla_experiment(config_path, REPOSITORY_ROOT)
    action_space = load_smolvla_action_space(experiment)
    dataset_config = _load_yaml(REPOSITORY_ROOT / experiment["dataset"]["config"])
    action_contract_path = REPOSITORY_ROOT / experiment["action_contract"]["derived"]
    contract = load_action_contract(action_contract_path)
    train_episodes = [int(value) for value in experiment["dataset"]["train_episodes"]]
    validation_episodes = [int(value) for value in experiment["dataset"]["validation_episodes"]]
    test_episodes = {int(value) for value in experiment["dataset"]["test_episodes"]}
    if set(train_episodes) & set(validation_episodes) or (
        set(train_episodes) | set(validation_episodes)
    ) & test_episodes:
        raise ValueError("Train, validation and hidden-test episodes must be disjoint.")
    scope = train_episodes + validation_episodes
    root = _dataset_root(experiment)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("resolved_revision") != experiment["dataset"]["revision"]
        or manifest.get("repo_id") != experiment["dataset"]["identifier"]
    ):
        raise ValueError("Prepared dataset identity differs from the VLA experiment.")
    fields = dataset_config["fields"]
    dataset = arrow_dataset.dataset(root / "data", format="parquet")
    table = dataset.to_table(
        columns=[
            fields["action"],
            fields["state"],
            fields["episode_index"],
            fields["frame_index"],
        ],
        filter=arrow_dataset.field(fields["episode_index"]).isin(scope),
    )
    rows = sorted(
        table.to_pylist(),
        key=lambda row: (int(row[fields["episode_index"]]), int(row[fields["frame_index"]])),
    )
    episode_ids = np.asarray([int(row[fields["episode_index"]]) for row in rows])
    if set(episode_ids.tolist()) != set(scope) or any(
        value in test_episodes for value in episode_ids
    ):
        raise ValueError("Benchmark materialized an unexpected or hidden-test episode.")
    actions = np.asarray([row[fields["action"]] for row in rows], dtype=np.float64)
    states = np.asarray([row[fields["state"]] for row in rows], dtype=np.float64)
    if actions.ndim != 2 or states.shape != actions.shape or actions.shape[1] != contract.dimension:
        raise ValueError("Dataset state/action shape differs from the Action Contract.")
    if not np.isfinite(actions).all() or not np.isfinite(states).all():
        raise ValueError("Dataset benchmark inputs contain NaN or Inf.")

    train_mask = np.isin(episode_ids, train_episodes)
    validation_mask = np.isin(episode_ids, validation_episodes)
    target_projection: dict[str, Any] = {
        "mode": "none",
        "stage": "none",
        "projected_elements": 0,
        "total_elements": int(actions.size),
        "projection_rate": 0.0,
        "maximum_source_overshoot": 0.0,
    }
    if action_space.target_projection == "action_contract_clip":
        actions, target_projection = _project_actions(
            actions,
            lower=contract.lower_bounds.numpy(),
            upper=contract.upper_bounds.numpy(),
            tolerances=contract.source_overshoot_tolerances.numpy(),
        )
    train_mean = actions[train_mask].mean(axis=0)
    target = _chunks(actions[validation_mask], episode_ids[validation_mask], contract.chunk_length)
    validation_states = states[validation_mask]
    if len(validation_states) != len(target):
        raise RuntimeError("Validation state and chunk counts differ.")
    train_mean_prediction = np.broadcast_to(train_mean, target.shape)
    if action_space.explicit:
        validation_states = np.clip(
            validation_states,
            contract.lower_bounds.numpy(),
            contract.upper_bounds.numpy(),
        )
    state_prediction = np.broadcast_to(validation_states[:, None, :], target.shape)
    metrics = {
        "train_action_mean": _metric(train_mean_prediction, target),
        "current_state_persistence": _metric(state_prediction, target),
    }
    declared = set(experiment["benchmark"]["baselines"])
    if set(metrics) != declared:
        raise ValueError("Implemented baseline set differs from the preregistered config.")
    if not all(math.isfinite(metric) for values in metrics.values() for metric in values.values()):
        raise ValueError("Pre-training benchmark produced a non-finite metric.")
    report = {
        "schema_version": 1,
        "status": "complete",
        "stage": "pre_training",
        "experiment_id": experiment["experiment_id"],
        "experiment_config_sha256": file_sha256(config_path),
        "dataset_id": experiment["dataset"]["identifier"],
        "dataset_revision": experiment["dataset"]["revision"],
        "dataset_manifest_sha256": file_sha256(manifest_path),
        "action_contract_sha256": file_sha256(action_contract_path),
        "action_space": action_space.as_dict(),
        "target_projection": target_projection,
        "normalization_source_split": "train",
        "evaluated_split": "validation",
        "train_episodes": train_episodes,
        "validation_episodes": validation_episodes,
        "hidden_test_loaded": False,
        "validation_samples": int(validation_mask.sum()),
        "chunk_size": contract.chunk_length,
        "metrics": metrics,
    }
    digest = stable_hash(report)[:16]
    run_root = Path(os.environ.get("ROSETTA_RUN_ROOT", REPOSITORY_ROOT / "runs"))
    destination = (
        run_root
        / str(experiment["experiment_id"])
        / "benchmark"
        / f"pre-training-{digest}.json"
    )
    create_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Benchmark report: {destination.name}")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    benchmark(args.config.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
