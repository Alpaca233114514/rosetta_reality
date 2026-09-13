"""Decompose early Hestia saved-array generalization gaps against train constants."""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import numpy as np

from scripts.diagnose_hestia_checkpoint_localization import (
    file_hash,
    load_inputs,
    noise_summary,
    read_json,
    require,
)

PLAN = Path("reports/training/m2-smolvla-hestia-early-gap-plan-2026-09-13.json")


def decompose(pred, target, train_target, train_pred):
    p, y, tr, tp = (
        np.asarray(x, dtype=np.float64) for x in (pred, target, train_target, train_pred)
    )
    require(
        p.ndim == 4
        and p.shape[1:] == y.shape
        and tp.ndim == 4
        and tp.shape[0] == p.shape[0]
        and tp.shape[1:] == tr.shape
        and tr.shape[1:] == y.shape[1:]
        and all(x.size and np.isfinite(x).all() for x in (p, y, tr, tp)),
        "Finite aligned train/evaluation arrays required",
    )
    axis = (1, 2, 3)
    constant = tr.mean(0)
    median = np.median(tr, axis=0)
    template = np.broadcast_to(tp.mean(1, keepdims=True), p.shape)
    pc, yc = p - p.mean(1, keepdims=True), y - y.mean(0, keepdims=True)
    bias = np.square(p.mean(1) - y.mean(0)).mean((1, 2))
    cbias = np.square(constant - y.mean(0)).mean()
    variance = np.square(pc).mean(axis)
    covariance = (pc * yc).mean(axis)
    target_var = np.square(yc).mean()
    mse = np.square(p - y).mean(axis)
    baseline_mse = np.square(constant - y).mean()
    gap = mse - baseline_mse
    terms = bias - cbias + variance - 2 * covariance
    require(np.allclose(gap, terms, atol=1e-12, rtol=1e-12), "Gap identity failed")
    # Independent expansion without centering the arrays.
    cross = (p * y).mean(axis) - (p.mean(1) * y.mean(0)).mean((1, 2))
    require(
        np.allclose(cross, covariance, atol=1e-12, rtol=1e-12),
        "Uncentered covariance identity failed",
    )
    errors = {}
    for name, value in (
        ("model", p),
        ("train_prediction_template", template),
        ("train_label_mean", np.broadcast_to(constant, p.shape)),
        ("train_label_median", np.broadcast_to(median, p.shape)),
    ):
        err = value - y
        errors[name] = {
            "mae": noise_summary(np.abs(err).mean(axis)),
            "mse": noise_summary(np.square(err).mean(axis)),
            "mae_by_noise_episode": np.abs(err).mean((2, 3)).tolist(),
        }
    return {
        "errors": errors,
        "model_minus_train_mean_mse": noise_summary(gap),
        "squared_scene_mean_bias_difference": noise_summary(bias - cbias),
        "prediction_scene_variance": noise_summary(variance),
        "minus_twice_scene_covariance": noise_summary(-2 * covariance),
        "scene_correlation_by_noise": [
            float(c / np.sqrt(v * target_var)) if v * target_var > 0 else None
            for c, v in zip(covariance, variance, strict=True)
        ],
        "template_minus_model_mae": noise_summary(
            np.abs(template - y).mean(axis) - np.abs(p - y).mean(axis)
        ),
        "template_minus_model_mse": noise_summary(np.square(template - y).mean(axis) - mse),
        "decomposition_max_absolute_residual": float(np.max(np.abs(gap - terms))),
    }


def main():
    start = time.monotonic()
    plan = read_json(PLAN)
    arrays, meta, rows, prior, files = load_inputs(plan)
    groups = {}
    for unit, kind in (("radian", "joint"), ("normalized", "gripper")):
        groups[kind] = [i for i, d in enumerate(meta["dimensions"]) if d["unit"] == unit]
        for side in ("left", "right"):
            groups[side + "_" + kind] = [
                i for i in groups[kind] if meta["dimensions"][i]["name"].startswith(side + "_")
            ]
    result = {
        "id": plan["id"],
        "plan_sha256": file_hash(PLAN),
        "groups": groups,
        "noise_conditions": meta["noise_conditions"],
        "episodes": meta["views"],
        "verified_curve_files": files,
        "metrics": {},
        "historical_checks": 0,
        "model_forwards": 0,
        "optimizer_steps": 0,
        "hidden_loaded": False,
        "new_gate3": False,
        "new_gate4": False,
        "checkpoint_selection": False,
    }
    for step, saved in arrays.items():
        sr = result["metrics"][str(step)] = {}
        for space in ("standard", "normalized"):
            pr = sr[space] = {}
            p, y = saved[space + "_predictions"], saved[space + "_targets"]
            for view, indices in rows.items():
                vr = pr[view] = {}
                for window, (a, b) in plan["windows"].items():
                    wr = vr[window] = {}
                    for group, dims in groups.items():
                        value = decompose(
                            p[:, indices, a:b][..., dims],
                            y[indices, a:b][..., dims],
                            y[rows["train40"], a:b][..., dims],
                            p[:, rows["train40"], a:b][..., dims],
                        )
                        wr[group] = value
                        old = prior["steps"][str(step)][space][view][window][group]
                        for metric in ("mae", "mse"):
                            require(
                                np.allclose(
                                    value["errors"]["model"][metric]["by_noise"],
                                    old[metric + "_by_noise"],
                                    atol=1e-12,
                                    rtol=1e-12,
                                ),
                                "Historical metric reproduction failed",
                            )
                            result["historical_checks"] += 1
    result["seconds"] = time.monotonic() - start
    result["maximum_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    require(result["seconds"] < plan["maximum_seconds"], "Analysis deadline exceeded")
    require(result["maximum_rss_bytes"] < plan["maximum_rss_bytes"], "Analysis memory exceeded")
    out = Path(plan["output"]) / "result.json"
    with out.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(
        json.dumps(
            {k: result[k] for k in ("id", "historical_checks", "seconds", "maximum_rss_bytes")}
        )
    )
    for step in plan["steps"]:
        for group in ("left_joint", "right_joint", "left_gripper", "right_gripper"):
            q = result["metrics"][str(step)]["standard"]["dev5"]["full"][group]
            print(
                json.dumps(
                    {
                        "step": step,
                        "group": group,
                        "mae": {k: v["mae"]["mean"] for k, v in q["errors"].items()},
                        "gap_terms": {
                            k: q[k]["mean"]
                            for k in (
                                "model_minus_train_mean_mse",
                                "squared_scene_mean_bias_difference",
                                "prediction_scene_variance",
                                "minus_twice_scene_covariance",
                            )
                        },
                        "correlation": q["scene_correlation_by_noise"],
                    }
                )
            )


if __name__ == "__main__":
    main()
