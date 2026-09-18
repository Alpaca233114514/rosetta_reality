"""Describe A/B action component errors using immutable prediction bundles."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from rosetta_reality.vla.visual_coverage import file_hash, read_bundle


def decompose(prediction, target, reference):
    p, y, ref = (np.asarray(v, dtype=np.float64) for v in (prediction, target, reference))
    if p.shape != y.shape or p.ndim != 3 or not all(np.isfinite(v).all() for v in (p, y, ref)):
        raise ValueError("Finite aligned [scene,time,dimension] arrays required")
    pm, ym = p.mean(0), y.mean(0)
    pc, yc = p - pm, y - ym
    pv, tv = float(np.square(pc).mean()), float(np.square(yc).mean())
    covariance = float((pc * yc).mean())
    bias = float(np.square(pm - ym).mean())
    mse = float(np.square(p - y).mean())
    reconstructed = tv + pv - 2 * covariance + bias
    if not np.isclose(mse, reconstructed, rtol=1e-12, atol=1e-12):
        raise ValueError("MSE decomposition identity failed")
    return {
        "mse": mse,
        "mae": float(np.abs(p - y).mean()),
        "target_scene_variance": tv,
        "prediction_scene_variance": pv,
        "scene_covariance": covariance,
        "mean_bias_squared": bias,
        "prediction_to_target_variance_ratio": pv / tv if tv > 0 else None,
        "scene_correlation": covariance / np.sqrt(pv * tv) if pv * tv > 0 else None,
        "train40_mean_mse": float(np.square(y - ref.mean(0)).mean()),
        "train40_mean_mae": float(np.abs(y - ref.mean(0)).mean()),
        "train40_median_mae": float(np.abs(y - np.median(ref, axis=0)).mean()),
        "identity_absolute_error": abs(mse - reconstructed),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8-sig"))
    if os.environ.get("ROSETTA_CONTAINER_IMAGE_ID") != plan["image"]:
        raise ValueError("Registered offline container required")
    for name, sha in plan["sha256"].items():
        if file_hash(Path(name)) != sha:
            raise ValueError(f"Input/code identity changed: {name}")
    check = json.loads(Path(plan["verification"]).read_text())
    if check["status"] != "passed":
        raise ValueError("Historical reproduction prerequisite failed")
    records = []
    windows = dict(
        full=(0, 50), first=(0, 1), early=(0, 10), middle=(10, 25), late=(25, 50), last=(49, 50)
    )
    for arm in "AB":
        arrays, meta = read_bundle(Path(plan["bundle_root"]) / f"{arm}-first")
        if meta["arm"] != arm or meta["episodes"] != plan["episodes"]:
            raise ValueError("Arm or sample identity differs")
        groups = {
            "joint": [],
            "gripper": [],
            "left_joint": [],
            "right_joint": [],
            "left_gripper": [],
            "right_gripper": [],
        }
        for index, dimension in enumerate(meta["dimensions"]):
            component = "joint" if dimension["unit"] == "radian" else "gripper"
            side = dimension["name"].split("_")[0]
            groups[component].append(index)
            groups[side + "_" + component].append(index)
        if any(not dims for dims in groups.values()):
            raise ValueError("Incomplete action component groups")
        for space in ("normalized", "standard"):
            target = arrays[space + "_targets"]
            prediction = arrays[space + "_predictions"]
            for noise, seed in enumerate(meta["noise_conditions"]):
                for view, episodes in meta["views"].items():
                    rows = [meta["episodes"].index(ep) for ep in episodes]
                    p, y = prediction[noise, rows], target[rows]
                    full_per_dimension = np.square(p - y).mean(axis=(0, 1))
                    for window, (start, stop) in windows.items():
                        values = {}
                        for group, dims in groups.items():
                            value = decompose(
                                p[:, start:stop, dims],
                                y[:, start:stop, dims],
                                target[:40, start:stop, dims],
                            )
                            if window == "full" and not np.isclose(
                                value["mse"],
                                full_per_dimension[dims].mean(),
                                rtol=1e-12,
                                atol=1e-12,
                            ):
                                raise ValueError("Per-dimension aggregation differs")
                            values[group] = value
                        records.append(
                            {
                                "arm": arm,
                                "space": space,
                                "noise_seed": seed,
                                "view": view,
                                "window": window,
                                "groups": values,
                            }
                        )
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "id": plan["id"],
        "status": "completed",
        "plan_sha256": file_hash(args.plan),
        "records": records,
        "all_decomposition_identities_passed": True,
        "model_forwards": 0,
        "optimizer_steps": 0,
        "hidden_test_loaded": False,
        "policy_modified": False,
        "task_success": "not measured",
        "m2_complete": False,
    }
    with (output / "result.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"status": "completed", "records": len(records)}))


if __name__ == "__main__":
    main()
