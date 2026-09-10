"""Split-safe, non-gating vision diagnostics; imports never load data or models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def verify_deploy_artifact(root: Path, artifact_id: str, experiment_id: str) -> str:
    """Verify a local deploy inventory before deserializing any model weights."""
    from rosetta_reality.experiment import file_sha256

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("artifact_id") != artifact_id
        or manifest.get("experiment_id") != experiment_id
        or manifest.get("status") != "verified"
        or manifest.get("hidden_test_loaded") is not False
        or manifest.get("reload", {}).get("exact_tensor_equality") is not True
    ):
        raise ValueError("Deploy manifest identity/reload boundary is invalid.")
    files = manifest.get("files")
    required = {
        "config.json",
        "normalization.json",
        "action_contract.json",
        "pretrained_model/config.json",
    }
    if not isinstance(files, dict) or not required <= files.keys():
        raise ValueError("Deploy manifest must cover configs, normalization and Action Contract.")
    if not any(
        name.startswith("pretrained_model/") and name.endswith(".safetensors") for name in files
    ):
        raise ValueError("Deploy manifest does not cover model weights.")
    for name, checksum in files.items():
        relative = Path(name)
        target = (root / relative).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not target.is_relative_to(root.resolve())
        ):
            raise ValueError("Unsafe deploy inventory path.")
        if not target.is_file() or file_sha256(target) != checksum:
            raise ValueError("Deploy inventory checksum mismatch.")
    normalization = json.loads((root / "normalization.json").read_text(encoding="utf-8"))
    if (
        normalization.get("source_split") != "train"
        or normalization.get("hidden_test_loaded") is not False
    ):
        raise ValueError("Deploy normalization must come from train only.")
    return file_sha256(manifest_path)


def validate_splits(train: list[int], validation: list[int], hidden: list[int]) -> None:
    groups = [train, validation, hidden]
    if any(not group or len(group) != len(set(group)) for group in groups):
        raise ValueError("Episode splits must be nonempty and unique.")
    if any(set(a) & set(b) for i, a in enumerate(groups) for b in groups[i + 1 :]):
        raise ValueError("Episode splits must be disjoint, including hidden-test episodes.")


def read_frame_zero(
    root: Path, config: Any, episodes: list[int], hidden: list[int]
) -> list[dict[str, Any]]:
    """Filter at the Arrow scan boundary, before materializing any sample rows.

    Shared Parquet containers may physically contain multiple splits. Reading
    file metadata/checksum bytes is not permission to materialize hidden rows.
    """
    import pyarrow.dataset as arrow

    if not episodes or len(episodes) != len(set(episodes)) or set(episodes) & set(hidden):
        raise ValueError("Requested episodes are empty, duplicated or include hidden-test.")
    if not set(episodes) <= set(config.episodes):
        raise ValueError("Requested episodes are outside the pinned dataset config.")
    fields = config.fields
    table = arrow.dataset(root / "data", format="parquet").to_table(
        columns=[fields.episode_index, fields.frame_index, fields.action, fields.state],
        filter=(arrow.field(fields.frame_index) == 0)
        & arrow.field(fields.episode_index).isin(episodes),
    )
    rows = table.to_pylist()
    ids = [int(row[fields.episode_index]) for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(episodes):
        raise ValueError("Exactly one frame-zero row per requested episode is required.")
    by_episode = dict(zip(ids, rows, strict=True))
    ordered = [by_episode[episode] for episode in episodes]
    for field in (fields.action, fields.state):
        values = np.asarray([row[field] for row in ordered], dtype=np.float64)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("Frame-zero actions/states must be finite vectors.")
    return ordered


def load_frame_zero_context(repository_root: Path, split: str = "train") -> dict[str, Any]:
    """Resolve one checksum-verified cache for labels AND images, offline."""
    from rosetta_reality.data import resolve_prepared_cache
    from rosetta_reality.data.config import load_dataset_config
    from rosetta_reality.vla.action_space import load_smolvla_experiment

    config = load_dataset_config(repository_root / "configs/data/aloha_sim_insertion_m2.yaml")
    experiment = load_smolvla_experiment(
        repository_root
        / "configs/vla/smolvla_450m_aloha_insertion_action_repair_bounded_gripper_003.yaml",
        repository_root,
    )
    data = experiment["dataset"]
    if config.repo_id != data["identifier"] or config.revision != data["revision"]:
        raise ValueError("Experiment and dataset revision identities disagree.")
    train, validation, hidden = (
        list(data[key]) for key in ("train_episodes", "validation_episodes", "test_episodes")
    )
    validate_splits(train, validation, hidden)
    if split not in {"train", "validation", "non_hidden"}:
        raise ValueError("Unsupported diagnostic split.")
    episodes = {"train": train, "validation": validation, "non_hidden": train + validation}[split]
    root, manifest = resolve_prepared_cache(config, repository_root, validate_checksums=True)
    rows = read_frame_zero(root, config, episodes, hidden)
    return {
        "config": config,
        "root": root,
        "manifest": manifest,
        "episodes": episodes,
        "train": train,
        "validation": validation,
        "actions": np.asarray([row[config.fields.action] for row in rows], dtype=np.float64),
        "states": np.asarray([row[config.fields.state] for row in rows], dtype=np.float64),
    }


def spatial_features(tokens: np.ndarray, grid: tuple[int, int], bins: int = 2) -> np.ndarray:
    """Ordered grid pooling, retaining coarse spatial layout without huge caches."""
    values = np.asarray(tokens, dtype=np.float64)
    height, width = grid
    if (
        values.ndim != 2
        or height * width != len(values)
        or bins < 1
        or min(height, width) < bins
        or not np.isfinite(values).all()
    ):
        raise ValueError("Spatial features require finite tokens and an explicit matching grid.")
    pixels = values.reshape(height, width, -1)
    return np.concatenate(
        [
            pixels[np.ix_(rows, columns)].mean(axis=(0, 1))
            for rows in np.array_split(np.arange(height), bins)
            for columns in np.array_split(np.arange(width), bins)
        ]
    )


def ridge_predict(x: np.ndarray, y: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    """Dual ridge: memory scales with episode count, not squared feature width."""
    if alpha <= 0 or not np.isfinite(alpha):
        raise ValueError("Ridge alpha must be finite and positive.")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    z = (x - mean) / scale
    coefficients = np.linalg.solve(z @ z.T + alpha * np.eye(len(x)), y - y.mean(axis=0))
    return ((target - mean) / scale) @ z.T @ coefficients + y.mean(axis=0)


def fit_probe(
    features: np.ndarray,
    actions: np.ndarray,
    train_count: int,
    *,
    alphas: tuple[float, ...] = (0.001, 0.1, 1.0, 10.0, 100.0),
    seed: int = 20260905,
) -> dict[str, Any]:
    """Select alpha using train folds only, then evaluate once on validation.

    Train CV is a tuning score, not an unbiased final performance estimate.
    Every row is one distinct episode; caller must preserve train-first order.
    """
    x, y = np.asarray(features, dtype=np.float64), np.asarray(actions, dtype=np.float64)
    if (
        x.ndim != 2
        or y.ndim != 2
        or len(x) != len(y)
        or not 5 <= train_count < len(x)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or not alphas
    ):
        raise ValueError("Probe needs finite aligned matrices and separate train/validation rows.")
    train = np.arange(train_count)
    folds = np.array_split(np.random.default_rng(seed).permutation(train), 5)
    scores = []
    for alpha in alphas:
        predictions = np.empty_like(y[:train_count])
        for held in folds:
            fit = np.setdiff1d(train, held)
            predictions[held] = ridge_predict(x[fit], y[fit], x[held], alpha)
        scores.append(float(np.abs(predictions - y[:train_count]).mean()))
    alpha = alphas[int(np.argmin(scores))]
    prediction = ridge_predict(x[:train_count], y[:train_count], x[train_count:], alpha)
    truth = y[train_count:]
    mean_mae = float(np.abs(truth - y[:train_count].mean(axis=0)).mean())
    median_mae = float(np.abs(truth - np.median(y[:train_count], axis=0)).mean())
    mae = float(np.abs(prediction - truth).mean())
    return {
        "alpha": alpha,
        "alpha_candidates": list(alphas),
        "train_cv_mae_by_alpha": scores,
        "train_fold_seed": seed,
        "train_cv_tuning_mae": min(scores),
        "validation_mae": mae,
        "train_mean_baseline_validation_mae": mean_mae,
        "train_median_baseline_validation_mae": median_mae,
        "validation_per_dimension_mae": np.abs(prediction - truth).mean(axis=0).tolist(),
        "interpretation": (
            "This readout improves both constant baselines on this validation split."
            if mae < min(mean_mae, median_mae)
            else "This readout does not beat both baselines; information absence is unproven."
        ),
        "gating": False,
    }


def paired_alignment(
    correct: np.ndarray, mismatched: np.ndarray, truth: np.ndarray
) -> dict[str, Any]:
    """Score paired interventions; input changes alone never establish correctness."""
    arrays = [np.asarray(value, dtype=np.float64) for value in (correct, mismatched, truth)]
    if any(
        value.ndim != 2 or value.shape[1] == 0 or not np.isfinite(value).all()
        for value in arrays
    ):
        raise ValueError("Alignment inputs must be finite sample-by-action matrices.")
    correct, mismatched, truth = arrays
    if correct.shape != mismatched.shape or correct.shape != truth.shape or len(truth) < 2:
        raise ValueError("Alignment inputs must have matching shapes and at least two episodes.")
    good, bad = np.abs(correct - truth), np.abs(mismatched - truth)
    correlations: list[float | None] = []
    for dim in range(truth.shape[1]):
        correlations.append(
            float(np.corrcoef(correct[:, dim], truth[:, dim])[0, 1])
            if min(correct[:, dim].std(), truth[:, dim].std()) > 1e-9
            else None
        )
    return {
        "correct_image_mae": float(good.mean()),
        "mismatched_image_mae": float(bad.mean()),
        "paired_mae_gain": float((bad - good).mean()),
        "per_episode_mae_gain": (bad - good).mean(axis=1).tolist(),
        "correct_per_dimension_mae": good.mean(axis=0).tolist(),
        "output_target_correlation_per_dimension": correlations,
        "image_output_shift": float(np.abs(correct - mismatched).mean()),
        "gating": False,
        "interpretation": "Positive paired MAE gain supports visual alignment on this slice only.",
    }
