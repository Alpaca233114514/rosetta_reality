"""Decompose B-to-C error changes after the recovered Hestia fit test."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from rosetta_reality.vla import visual_fit as fit
from scripts.diagnose_coverage40_components import decompose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8-sig"))
    if os.environ.get("ROSETTA_CONTAINER_IMAGE_ID") != plan["image"]:
        raise ValueError("Registered offline container required")
    for name, sha in plan["sha256"].items():
        if fit.file_hash(Path(name)) != sha:
            raise ValueError(f"Input/code drift: {name}")
    proof = json.loads(Path(plan["verification"]).read_text())
    if proof["status"] != "passed" or proof["comparison_file_sha_exact"] is not True:
        raise ValueError("Hestia recovery verification required")
    root = Path(plan["root"])
    b, bm = fit.read_bundle(root / "B-first")
    c, cm = fit.read_bundle(root / "C-first")
    fit.compare_arms(b, bm, c, cm)
    groups = {
        "joint": [],
        "gripper": [],
        "left_joint": [],
        "right_joint": [],
        "left_gripper": [],
        "right_gripper": [],
    }
    for index, dimension in enumerate(bm["dimensions"]):
        group = "joint" if dimension["unit"] == "radian" else "gripper"
        side = dimension["name"].split("_")[0]
        groups[group].append(index)
        groups[side + "_" + group].append(index)
    windows = dict(
        full=(0, 50), first=(0, 1), early=(0, 10), middle=(10, 25), late=(25, 50), last=(49, 50)
    )
    records = []
    for space in ("normalized", "standard"):
        target = b[space + "_targets"]
        for noise, seed in enumerate(bm["noise_conditions"]):
            for view in ("train40", "dev5"):
                rows = [bm["episodes"].index(ep) for ep in bm["views"][view]]
                for window, (start, stop) in windows.items():
                    values = {}
                    for group, dims in groups.items():
                        target_slice = target[rows, start:stop][:, :, dims]
                        reference = target[:40, start:stop, dims]
                        a = decompose(
                            b[space + "_predictions"][noise, rows, start:stop][:, :, dims],
                            target_slice,
                            reference,
                        )
                        z = decompose(
                            c[space + "_predictions"][noise, rows, start:stop][:, :, dims],
                            target_slice,
                            reference,
                        )
                        delta = {
                            "mse": z["mse"] - a["mse"],
                            "prediction_variance": z["prediction_scene_variance"]
                            - a["prediction_scene_variance"],
                            "minus_twice_covariance": -2
                            * (z["scene_covariance"] - a["scene_covariance"]),
                            "mean_bias_squared": z["mean_bias_squared"] - a["mean_bias_squared"],
                        }
                        if a["target_scene_variance"] != z[
                            "target_scene_variance"
                        ] or not np.isclose(
                            delta["mse"],
                            sum(v for k, v in delta.items() if k != "mse"),
                            rtol=1e-12,
                            atol=1e-12,
                        ):
                            raise ValueError("Between-arm decomposition identity failed")
                        values[group] = {"B": a, "C": z, "delta": delta}
                    records.append(
                        {
                            "space": space,
                            "noise_seed": seed,
                            "view": view,
                            "window": window,
                            "groups": values,
                        }
                    )
    summary = []
    for space in ("normalized", "standard"):
        for view in ("train40", "dev5"):
            for window in windows:
                matched = [
                    r
                    for r in records
                    if r["space"] == space and r["view"] == view and r["window"] == window
                ]
                fields = {}
                for group in groups:
                    fields[group] = {}
                    for arm in ("B", "C", "delta"):
                        fields[group][arm] = {}
                        for metric in matched[0]["groups"][group][arm]:
                            items = [r["groups"][group][arm][metric] for r in matched]
                            fields[group][arm][metric] = (
                                None
                                if any(v is None for v in items)
                                else {
                                    "mean": float(np.mean(items)),
                                    "min": min(items),
                                    "max": max(items),
                                }
                            )
                summary.append({"space": space, "view": view, "window": window, "groups": fields})
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("records.json", records), ("summary.json", summary)):
        with (output / name).open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
    result = {
        "status": "completed",
        "plan_sha256": fit.file_hash(args.plan),
        "records_sha256": fit.file_hash(output / "records.json"),
        "summary_sha256": fit.file_hash(output / "summary.json"),
        "all_decomposition_identities_passed": True,
        "record_count": len(records),
        "new_model_forwards": 0,
        "new_optimizer_steps": 0,
        "hidden_test_loaded": False,
        "policy_modified": False,
        "m2_complete": False,
    }
    with (output / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
