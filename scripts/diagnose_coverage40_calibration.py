"""Apply one preregistered train-only calibration to historical B predictions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from rosetta_reality.vla.visual_coverage import file_hash, read_bundle
from scripts.diagnose_chunk_bias import correct_bias
from scripts.diagnose_chunk_time_shift import analyze as analyze_shift
from scripts.diagnose_kv_full_chunk import chunk_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8-sig"))
    if os.environ.get("ROSETTA_CONTAINER_IMAGE_ID") != plan["image"]:
        raise ValueError("Registered offline container required")
    for name, sha in plan["sha256"].items():
        if file_hash(Path(name)) != sha:
            raise ValueError(f"Input/code identity drift: {name}")
    verification = json.loads(Path(plan["verification"]).read_text())
    if (
        verification["status"] != "passed"
        or verification["recomputed_sha256"] != plan["historical_comparison_sha256"]
    ):
        raise ValueError("Historical reproduction prerequisite failed")
    saved, metadata = read_bundle(Path(plan["bundle"]))
    if metadata["arm"] != "B" or metadata["episodes"] != plan["episodes"]:
        raise ValueError("Wrong historical model or sample order")
    if metadata["noise_conditions"] != plan["noise_conditions"]:
        raise ValueError("Noise identities changed")
    target = saved["standard_targets"].astype(np.float64)
    lower = np.array([d["minimum"] for d in metadata["dimensions"]])
    upper = np.array([d["maximum"] for d in metadata["dimensions"]])
    if np.max(np.abs(target - np.clip(target, lower, upper))) > 1e-8:
        raise ValueError("Historical targets exceed physical bounds")
    groups = {}
    for index, dimension in enumerate(metadata["dimensions"]):
        groups.setdefault(dimension["unit"], []).append(index)
    train_count = len(metadata["views"]["train40"])
    if train_count != 40 or len(target) != 45:
        raise ValueError("Registered split sizes changed")
    ref = target[:train_count]
    loo_mean = (ref.sum(0) - ref) / (train_count - 1)
    loo_median = np.stack([np.median(np.delete(ref, i, axis=0), axis=0) for i in range(40)])
    windows = dict(
        full=(0, 50), first=(0, 1), early=(0, 10), middle=(10, 25), late=(25, 50), last=(49, 50)
    )
    records, arrays = [], {}
    for noise_index, noise_seed in enumerate(metadata["noise_conditions"]):
        native = saved["standard_predictions"][noise_index].astype(np.float64)
        control = np.clip(native, lower, upper)
        if not np.array_equal(control, native):
            raise ValueError("Historical native control unexpectedly requires clipping")
        if plan["method"] == "bias":
            unprojected, loo_raw, bias = correct_bias(native, target, train_count)
            detail = {"bias_mean_by_dimension": bias.mean(0).tolist()}
            arrays[f"noise_{noise_index}_bias"] = bias
        elif plan["method"] == "time_shift":
            unprojected, loo_raw, detail = analyze_shift(control, target, plan["lags"], train_count)
        else:
            raise ValueError("Unknown single calibration method")
        candidate = np.clip(unprojected, lower, upper)
        loo = np.clip(loo_raw, lower, upper)
        metrics = {
            window: {
                name: chunk_metrics(value[40:], target[40:], ref, groups, start, stop)
                for name, value in (("control", control), ("candidate", candidate))
            }
            for window, (start, stop) in windows.items()
        }
        loo_scores = {
            group: {
                "mae": float(np.abs(loo[..., dims] - ref[..., dims]).mean()),
                "mean_mae": float(np.abs(loo_mean[..., dims] - ref[..., dims]).mean()),
                "median_mae": float(np.abs(loo_median[..., dims] - ref[..., dims]).mean()),
            }
            for group, dims in groups.items()
        }
        criteria = {}
        for group in groups:
            c = metrics["full"]["control"][group]
            v = metrics["full"]["candidate"][group]
            scores = loo_scores[group]
            criteria[group] = {
                "full_dev_reduction_at_least_20_percent": v["mae"] < 0.8 * c["mae"],
                "dev_beats_both_constants": v["mae"]
                < min(v["train_mean_baseline_mae"], v["train_median_baseline_mae"]) - 1e-8,
                "dev_positive_image_gain": v["paired_mae_gain"] > 1e-8,
                "loo_beats_both_constants": scores["mae"]
                < min(scores["mean_mae"], scores["median_mae"]) - 1e-8,
                "first_action_not_regressed": metrics["first"]["candidate"][group]["mae"]
                <= metrics["first"]["control"][group]["mae"] + 1e-8,
            }
        records.append(
            {
                "noise_seed": noise_seed,
                "selection": detail,
                "metrics": metrics,
                "loo": loo_scores,
                "criteria": criteria,
                "passed": all(all(v.values()) for v in criteria.values()),
                "first_actions_exactly_unchanged": bool(
                    np.array_equal(candidate[:, 0], native[:, 0])
                ),
                "candidate_clipped_by_dimension": (candidate != unprojected)
                .sum(axis=(0, 1))
                .tolist(),
            }
        )
        arrays.update(
            {
                f"noise_{noise_index}_candidate": candidate,
                f"noise_{noise_index}_unprojected": unprojected,
                f"noise_{noise_index}_loo": loo,
            }
        )
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    with (output / "arrays.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(output / "arrays.npz", allow_pickle=False) as loaded:
        if not all(np.array_equal(value, loaded[key]) for key, value in arrays.items()):
            raise ValueError("Saved-array exact reload failed")
    report = {
        "id": plan["id"],
        "status": "completed",
        "method": plan["method"],
        "plan_sha256": file_hash(args.plan),
        "conditions": records,
        "passed_conditions": sum(r["passed"] for r in records),
        "registered_diagnostic_supported": sum(r["passed"] for r in records) >= 3,
        "arrays_sha256": file_hash(output / "arrays.npz"),
        "array_reload_exact": True,
        "model_forwards": 0,
        "optimizer_steps": 0,
        "policy_modified": False,
        "hidden_test_loaded": False,
        "task_success": "not measured",
        "m2_complete": False,
    }
    with (output / "result.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"method": plan["method"], "passed_conditions": report["passed_conditions"]}))


if __name__ == "__main__":
    main()
