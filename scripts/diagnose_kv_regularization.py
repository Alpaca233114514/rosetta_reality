"""Bounded CPU reanalysis of saved visual features; no policy or dataset loader."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from rosetta_reality.vla.contextual_kv_probe import _metrics
from rosetta_reality.vla.vision_diagnostics import ridge_predict, validate_splits


def select_alpha(x, y, alphas, seed):
    """Only these supplied training rows participate in selection and scaling."""
    indices = np.arange(len(y))
    folds = np.array_split(np.random.default_rng(seed).permutation(indices), 5)
    scores = []
    for alpha in alphas:
        oof = np.empty_like(y)
        for held in folds:
            fitted = np.setdiff1d(indices, held)
            oof[held] = ridge_predict(x[fitted], y[fitted], x[held], alpha)
        scores.append(float(np.abs(oof - y).mean()))
    return float(alphas[int(np.argmin(scores))]), scores


def nested_predictions(x, y, alphas, seed):
    """Five outer folds; each alpha uses five inner folds of outer-train only."""
    indices = np.arange(len(y))
    folds = np.array_split(np.random.default_rng(seed).permutation(indices), 5)
    predictions = np.empty_like(y)
    means, medians = np.empty_like(y), np.empty_like(y)
    records = []
    for number, held in enumerate(folds):
        fitted = np.setdiff1d(indices, held)
        alpha, scores = select_alpha(x[fitted], y[fitted], alphas, seed + number + 1)
        predictions[held] = ridge_predict(x[fitted], y[fitted], x[held], alpha)
        means[held], medians[held] = y[fitted].mean(0), np.median(y[fitted], axis=0)
        records.append({"held_rows": held.tolist(), "alpha": alpha, "inner_scores": scores})
    return predictions, means, medians, records


def compare(x, y, *, train_count, groups, grids, seed):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if (
        x.ndim != 2
        or y.ndim != 2
        or len(x) != len(y)
        or x.shape[1] == 0
        or not 10 <= train_count <= len(y) - 2
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("Expected finite aligned features with disjoint train/development rows.")
    dims = [d for group in groups.values() for d in group]
    if not groups or any(not g for g in groups.values()) or sorted(dims) != list(range(y.shape[1])):
        raise ValueError("Unit groups must partition all dimensions exactly once.")
    for grid in grids.values():
        if not grid or any(not np.isfinite(a) or a <= 0 for a in grid):
            raise ValueError("Alpha grids must contain finite positive values.")
    tx, ty = x[:train_count], y[:train_count]
    results, arrays = {}, {}
    for name, grid in grids.items():
        alpha, scores = select_alpha(tx, ty, grid, seed)
        pred = ridge_predict(tx, ty, x, alpha)
        nested, means, medians, folds = nested_predictions(tx, ty, grid, seed)
        nested_metrics = {}
        for group, indices in groups.items():
            nested_metrics[group] = {
                "mae": float(np.abs(nested[:, indices] - ty[:, indices]).mean()),
                "fold_train_mean_mae": float(np.abs(means[:, indices] - ty[:, indices]).mean()),
                "fold_train_median_mae": float(np.abs(medians[:, indices] - ty[:, indices]).mean()),
                "per_episode_mae": np.abs(nested[:, indices] - ty[:, indices]).mean(-1).tolist(),
            }
        scale = tx.std(0)
        z = (tx - tx.mean(0)) / np.where(scale < 1e-12, 1.0, scale)
        eigenvalues = np.maximum(np.linalg.eigvalsh(z @ z.T), 0)
        results[name] = {
            "alpha": alpha,
            "grid": grid,
            "train_cv_tuning_scores": scores,
            "selected_at_upper_boundary": alpha == max(grid),
            "effective_ridge_degrees_of_freedom_excluding_intercept": float(
                np.sum(eigenvalues / (eigenvalues + alpha))
            ),
            "train_fit": _metrics(pred[:train_count], ty, ty, groups),
            "nested_train_oof": nested_metrics,
            "outer_folds": folds,
            "development": _metrics(pred[train_count:], y[train_count:], ty, groups),
        }
        arrays[name + "_predictions"] = pred
        arrays[name + "_nested_oof"] = nested
        arrays[name + "_fold_mean"] = means
        arrays[name + "_fold_median"] = medians
    return results, arrays


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8-sig"))
    if os.environ.get("ROSETTA_CONTAINER_IMAGE_ID") != plan["runtime"]["image"]:
        raise ValueError("Registered container image required.")
    for path, expected in plan["input_and_code_sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"Registered identity drift: {path}")
    old = json.loads(Path(plan["source_result"]).read_text(encoding="utf-8"))
    train, dev, hidden = (
        plan["train_episodes"],
        plan["development_episodes"],
        plan["hidden_episodes"],
    )
    validate_splits(train, dev, hidden)
    if old["train_episodes"] != train or old["development_episodes"] != dev:
        raise ValueError("Source split drift.")
    if old["hidden_rows_materialized"] is not False or old["status"] != "completed":
        raise ValueError("Source evidence is not accepted.")
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    with np.load(plan["features"], allow_pickle=False) as saved:
        if saved["episodes"].tolist() != train + dev:
            raise ValueError("Saved episode order differs from registration.")
        x, y = saved["early"].copy(), saved["actions"].copy()
    if x.shape != (45, 2560) or y.shape != (45, 14):
        raise ValueError("Saved feature/action shape drift.")
    result, arrays = compare(
        x,
        y,
        train_count=len(train),
        groups=plan["groups"],
        grids=plan["grids"],
        seed=plan["seed"],
    )
    historical = old["results"]["arms"]["early"]["predictions"]
    if not np.allclose(arrays["historical_grid_predictions"], historical, atol=1e-10, rtol=0):
        raise ValueError("Historical early-layer readout did not reproduce.")
    elapsed = time.monotonic() - started
    if elapsed > plan["runtime"]["maximum_seconds"]:
        raise TimeoutError("Registered analysis budget exceeded.")
    with (output / "predictions.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(output / "predictions.npz", allow_pickle=False) as saved:
        if any(not np.array_equal(array, saved[name]) for name, array in arrays.items()):
            raise ValueError("Output array reload mismatch.")
    comparison = {}
    for group in plan["groups"]:
        a, b = (result[name]["development"][group] for name in plan["grids"])
        comparison[group] = {
            "expanded_minus_historical_mae": b["mae"] - a["mae"],
            "expanded_beats_both_constants": b["mae"]
            < min(b["train_mean_baseline_mae"], b["train_median_baseline_mae"]) - plan["epsilon"],
            "expanded_positive_visual_gain": b["paired_mae_gain"] > plan["epsilon"],
        }
    report = {
        "id": plan["id"],
        "status": "completed",
        "plan_sha256": digest(args.plan),
        "source_result_sha256": digest(plan["source_result"]),
        "source_features_sha256": digest(plan["features"]),
        "historical_prediction_max_abs_difference": float(
            np.max(np.abs(arrays["historical_grid_predictions"] - np.asarray(historical)))
        ),
        "arms": result,
        "comparison": comparison,
        "elapsed_seconds": elapsed,
        "numpy_version": np.__version__,
        "array_reload_exact": True,
        "predictions_sha256": digest(output / "predictions.npz"),
        "new_model_forwards": 0,
        "policy_optimizer_created": False,
        "raw_data_loaded": False,
        "hidden_rows_materialized": False,
        "development_used_for_selection": False,
        "gating": False,
        "policy_improvement": "not measured",
        "task_success": "not measured",
        "m2_complete": False,
    }
    with (output / "result.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "completed", "comparison": comparison, "seconds": elapsed}))


if __name__ == "__main__":
    main()
