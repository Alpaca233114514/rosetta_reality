"""Train-only common-trajectory bias diagnostic on saved native outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from scripts.diagnose_kv_full_chunk import chunk_metrics
from scripts.diagnose_kv_regularization import digest


def correct_bias(predictions, targets, train_count):
    predictions, targets = (np.asarray(v, dtype=np.float64) for v in (predictions, targets))
    if (
        predictions.shape != targets.shape
        or predictions.ndim != 3
        or not 2 <= train_count < len(targets)
        or not np.isfinite(predictions).all()
        or not np.isfinite(targets).all()
    ):
        raise ValueError("Expected finite aligned chunks with separate train/development rows.")
    residuals = targets[:train_count] - predictions[:train_count]
    bias = residuals.mean(0)
    corrected = predictions + bias
    loo = predictions[:train_count] + (residuals.sum(0) - residuals) / (train_count - 1)
    return corrected, loo, bias


def main():
    from rosetta_reality.sim.action_contract import load_action_contract

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8-sig"))
    if os.environ.get("ROSETTA_CONTAINER_IMAGE_ID") != plan["runtime"]["image"]:
        raise ValueError("Registered offline container required.")
    for path, sha in plan["input_and_code_sha256"].items():
        if digest(path) != sha:
            raise ValueError(f"Identity drift: {path}")
    contract = load_action_contract(Path(plan["action_contract"]))
    lower, upper = contract.lower_bounds.numpy(), contract.upper_bounds.numpy()
    groups = {}
    for index, dimension in enumerate(contract.dimensions):
        groups.setdefault(dimension.unit, []).append(index)
    with np.load(plan["arrays"], allow_pickle=False) as saved:
        targets, native = saved["projected_targets"].copy(), saved["native_standard"].copy()
    if targets.shape != (45, 50, 14) or native.shape != targets.shape:
        raise ValueError("Registered saved chunk shape differs.")
    corrected, loo, bias = correct_bias(native, targets, 40)
    control = np.clip(native, lower, upper)
    candidate = np.clip(corrected, lower, upper)
    loo = np.clip(loo, lower, upper)
    windows = {
        "full": (0, 50),
        "first": (0, 1),
        "early": (0, 10),
        "middle": (10, 25),
        "late": (25, 50),
        "last": (49, 50),
    }
    metrics = {
        window: {
            name: chunk_metrics(value[40:], targets[40:], targets[:40], groups, start, stop)
            for name, value in (("control", control), ("bias_only", candidate))
        }
        for window, (start, stop) in windows.items()
    }
    loo_baseline_mean = (targets[:40].sum(0) - targets[:40]) / 39
    loo_baseline_median = np.stack(
        [np.median(np.delete(targets[:40], i, axis=0), axis=0) for i in range(40)]
    )
    loo_scores = {
        group: {
            "mae": float(np.abs(loo[..., dims] - targets[:40, :, dims]).mean()),
            "fold_mean_mae": float(
                np.abs(loo_baseline_mean[..., dims] - targets[:40, :, dims]).mean()
            ),
            "fold_median_mae": float(
                np.abs(loo_baseline_median[..., dims] - targets[:40, :, dims]).mean()
            ),
        }
        for group, dims in groups.items()
    }
    criteria = {}
    for group in groups:
        a, b = metrics["full"]["control"][group], metrics["full"]["bias_only"][group]
        criteria[group] = {
            "full_development_reduction_at_least_20_percent": b["mae"] < 0.8 * a["mae"],
            "full_development_beats_constants": b["mae"]
            < min(b["train_mean_baseline_mae"], b["train_median_baseline_mae"]) - 1e-8,
            "full_development_positive_image_gain": b["paired_mae_gain"] > 1e-8,
            "calibration_loo_beats_constants": loo_scores[group]["mae"]
            < min(loo_scores[group]["fold_mean_mae"], loo_scores[group]["fold_median_mae"]) - 1e-8,
            "first_action_not_regressed": metrics["first"]["bias_only"][group]["mae"]
            <= metrics["first"]["control"][group]["mae"] + 1e-8,
        }
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    arrays = dict(
        control=control,
        unprojected_corrected=corrected,
        candidate=candidate,
        calibration_loo=loo,
        bias=bias,
    )
    with (output / "arrays.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(output / "arrays.npz", allow_pickle=False) as saved:
        if not all(np.array_equal(v, saved[k]) for k, v in arrays.items()):
            raise ValueError("Saved array reload differs.")
    report = {
        "id": plan["id"],
        "status": "completed",
        "plan_sha256": digest(args.plan),
        "windows": metrics,
        "calibration_loo": loo_scores,
        "criteria": criteria,
        "registered_bias_diagnostic_passed": all(all(v.values()) for v in criteria.values()),
        "control_clipped_elements": int(np.count_nonzero(native != control)),
        "candidate_clipped_elements": int(np.count_nonzero(corrected != candidate)),
        "candidate_clipped_by_dimension": (corrected != candidate).sum(axis=(0, 1)).tolist(),
        "bias_mean_by_dimension": bias.mean(0).tolist(),
        "bias_max_abs_by_dimension": np.abs(bias).max(0).tolist(),
        "arrays_sha256": digest(output / "arrays.npz"),
        "array_reload_exact": True,
        "development_used_to_fit_bias": False,
        "new_model_forwards": 0,
        "raw_data_loaded": False,
        "policy_modified": False,
        "task_success": "not measured",
        "m2_complete": False,
    }
    with (output / "result.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "completed", "criteria": criteria}))


if __name__ == "__main__":
    main()
