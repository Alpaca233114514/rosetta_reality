"""Train-selected global prediction time shift with fixed endpoint holding."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from scripts.diagnose_kv_full_chunk import chunk_metrics
from scripts.diagnose_kv_regularization import digest


def shift_chunk(prediction, lag):
    if isinstance(lag, bool) or not isinstance(lag, (int, np.integer)):
        raise ValueError("Lag must be an integer number of prediction slots.")
    indices = np.clip(np.arange(prediction.shape[1]) + lag, 0, prediction.shape[1] - 1)
    return prediction[:, indices, :]


def choose_lag(prediction, targets, lags):
    scores = [float(np.abs(shift_chunk(prediction, lag) - targets).mean()) for lag in lags]
    # A fixed tie break prefers the smallest magnitude, then negative lag.
    chosen = min(range(len(lags)), key=lambda i: (scores[i], abs(lags[i]), lags[i]))
    return lags[chosen], scores


def analyze(prediction, targets, lags, train_count):
    if (
        prediction.shape != targets.shape
        or prediction.ndim != 3
        or not np.isfinite(prediction).all()
        or not np.isfinite(targets).all()
        or not 2 <= train_count < len(targets)
    ):
        raise ValueError("Expected finite aligned chunks and separate train/development rows.")
    lag, scores = choose_lag(prediction[:train_count], targets[:train_count], lags)
    candidate = shift_chunk(prediction, lag)
    loo = np.empty_like(prediction[:train_count])
    loo_lags = []
    for i in range(train_count):
        remaining = np.arange(train_count) != i
        value, _ = choose_lag(
            prediction[:train_count][remaining], targets[:train_count][remaining], lags
        )
        loo[i] = shift_chunk(prediction[i : i + 1], value)[0]
        loo_lags.append(value)
    return candidate, loo, {"lag": lag, "scores": scores, "loo_lags": loo_lags}


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
    with np.load(plan["arrays"], allow_pickle=False) as saved:
        prediction, targets = saved["native_standard"].copy(), saved["projected_targets"].copy()
    if prediction.shape != (45, 50, 14) or targets.shape != prediction.shape:
        raise ValueError("Registered chunk shape differs.")
    contract = load_action_contract(Path(plan["action_contract"]))
    prediction = np.clip(prediction, contract.lower_bounds.numpy(), contract.upper_bounds.numpy())
    candidate, loo, selection = analyze(prediction, targets, plan["lags"], 40)
    groups = {}
    for d, dimension in enumerate(contract.dimensions):
        groups.setdefault(dimension.unit, []).append(d)
    windows = {
        "full": (0, 50),
        "first": (0, 1),
        "early": (0, 10),
        "middle": (10, 25),
        "late": (25, 50),
        "last": (49, 50),
    }
    metrics = {
        name: {
            window: chunk_metrics(values[40:], targets[40:], targets[:40], groups, a, b)
            for window, (a, b) in windows.items()
        }
        for name, values in (("control", prediction), ("shifted", candidate))
    }
    loo_mean = (targets[:40].sum(0) - targets[:40]) / 39
    loo_median = np.stack(
        [np.median(np.delete(targets[:40], i, axis=0), axis=0) for i in range(40)]
    )
    group_selection = {
        name: dict(
            zip(
                ("lag", "scores"),
                choose_lag(prediction[:40, :, dims], targets[:40, :, dims], plan["lags"]),
                strict=True,
            )
        )
        for name, dims in groups.items()
    }
    loo_scores = {
        group: {
            "mae": float(np.abs(loo[..., dims] - targets[:40, :, dims]).mean()),
            "control_mae": float(np.abs(prediction[:40, :, dims] - targets[:40, :, dims]).mean()),
            "mean_mae": float(np.abs(loo_mean[..., dims] - targets[:40, :, dims]).mean()),
            "median_mae": float(np.abs(loo_median[..., dims] - targets[:40, :, dims]).mean()),
        }
        for group, dims in groups.items()
    }
    criteria = {}
    for group in groups:
        a, b = metrics["control"]["full"][group], metrics["shifted"]["full"][group]
        criteria[group] = {
            "dev_reduction_at_least_20_percent": b["mae"] < 0.8 * a["mae"],
            "dev_beats_both_constants": b["mae"]
            < min(b["train_mean_baseline_mae"], b["train_median_baseline_mae"]) - 1e-8,
            "dev_positive_image_gain": b["paired_mae_gain"] > 1e-8,
            "loo_beats_both_constants": loo_scores[group]["mae"]
            < min(loo_scores[group]["mean_mae"], loo_scores[group]["median_mae"]) - 1e-8,
            "first_action_not_regressed": metrics["shifted"]["first"][group]["mae"]
            <= metrics["control"]["first"][group]["mae"] + 1e-8,
        }
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    arrays = dict(shifted=candidate, loo=loo)
    with (output / "arrays.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(output / "arrays.npz", allow_pickle=False) as saved:
        assert all(np.array_equal(v, saved[k]) for k, v in arrays.items())
    report = {
        "id": plan["id"],
        "status": "completed",
        "plan_sha256": digest(args.plan),
        "selection": selection,
        "train_group_only_diagnostics_not_applied": group_selection,
        "metrics": metrics,
        "loo": loo_scores,
        "criteria": criteria,
        "global_shift_supported": all(all(v.values()) for v in criteria.values()),
        "first_action_array_exactly_unchanged": bool(
            np.array_equal(candidate[:, 0], prediction[:, 0])
        ),
        "endpoint_held_slots": abs(selection["lag"]),
        "array_sha256": digest(output / "arrays.npz"),
        "array_reload_exact": True,
        "new_model_forwards": 0,
        "raw_data_loaded": False,
        "policy_modified": False,
        "task_success": "not measured",
        "m2_complete": False,
    }
    with (output / "result.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "completed", "selection": selection, "criteria": criteria}))


if __name__ == "__main__":
    main()
