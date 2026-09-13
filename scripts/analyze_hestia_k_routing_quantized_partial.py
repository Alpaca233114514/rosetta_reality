"""Analyze sealed same-checkpoint image K/V interventions, without a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require(value, message):
    if not value:
        raise ValueError(message)


def paired_effect(base, treatment, target):
    base, treatment, target = (np.asarray(x, dtype=np.float64) for x in (base, treatment, target))
    require(
        base.shape == treatment.shape and base.shape[1:] == target.shape, "Paired shapes differ"
    )
    require(all(np.isfinite(x).all() for x in (base, treatment, target)), "Nonfinite pair")
    change = treatment - base
    axis = (1, 2, 3)
    movement = np.square(change).mean(axis)
    alignment = -2 * (change * (target - base)).mean(axis)
    mse_delta = (np.square(treatment - target) - np.square(base - target)).mean(axis)
    require(
        np.allclose(mse_delta, movement + alignment, atol=1e-12, rtol=1e-12), "Pair identity failed"
    )
    return {
        "mse_delta_by_noise": mse_delta.tolist(),
        "mae_delta_by_noise": (np.abs(treatment - target) - np.abs(base - target))
        .mean(axis)
        .tolist(),
        "movement_squared_by_noise": movement.tolist(),
        "minus_twice_residual_alignment_by_noise": alignment.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    plan_path = args.plan.resolve()
    plan = read(plan_path)
    workspace = Path.cwd().resolve()
    output = (workspace / plan["output"]).resolve()
    require(output.is_relative_to(workspace) and not output.exists(), "Fresh local output required")
    require(
        os.environ.get("ROSETTA_CONTAINER_IMAGE_ID") == plan["image"], "Container identity differs"
    )
    source = Path("/source")
    for name, expected in plan["source_sha256"].items():
        require(sha(source / name) == expected, f"Source drift: {name}")
    for name, expected in plan["local_sha256"].items():
        require(sha(workspace / name) == expected, f"Local drift: {name}")
    # Source is read-only and SHA checked before importing its frozen numeric helpers.
    sys.path.insert(0, str(source))
    from scripts.diagnose_hestia_checkpoint_localization import load_inputs
    from scripts.diagnose_hestia_early_gap import decompose

    os.chdir(source)
    originals, meta, rows, _, _ = load_inputs(read(source / plan["original_plan"]))
    os.chdir(workspace)
    recovered = workspace / plan["recovered"]
    handoff = read(recovered / "handoff-manifest.json")
    receipt = read(workspace / plan["receipt"])
    require(
        receipt["all_file_sha256_matched"] is True
        and receipt["manifest_sha256"] == sha(recovered / "handoff-manifest.json"),
        "Transport receipt differs",
    )
    for name, item in handoff["files"].items():
        path = recovered / name
        require(
            path.resolve().is_relative_to(recovered) and not path.is_symlink(), "Unsafe result path"
        )
        require(
            path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], "Result drift"
        )
    worker = read(recovered / "worker-exited.json")
    require(
        worker["error"]
        == {"type": "RuntimeError", "message": "Condition base640_rmass collector exit 1"}
        and worker["optimizer_steps"] == 0,
        "Unexpected partial-worker failure",
    )
    require(worker["completed_conditions"] == plan["conditions"], "Condition coverage differs")
    registration = read(recovered / "registration.json")
    template_path = workspace / plan["template"]
    require(
        registration["template_sha256"] == sha(template_path), "Worker template identity differs"
    )
    groups = {}
    for unit, kind in (("radian", "joint"), ("normalized", "gripper")):
        groups[kind] = [i for i, d in enumerate(meta["dimensions"]) if d["unit"] == unit]
        for side in ("left", "right"):
            groups[f"{side}_{kind}"] = [
                i for i in groups[kind] if meta["dimensions"][i]["name"].startswith(side + "_")
            ]
    arrays, reports = {}, {}
    for condition in plan["conditions"]:
        base = int(condition.split("_")[0].removeprefix("base"))
        folder = recovered / condition
        report = read(folder / "result.json")
        require(
            report["condition"] == condition and report["base_step"] == base,
            "Condition identity differs",
        )
        require(
            report["all_parameters_unchanged"]
            and report["model_forwards"] == 180
            and report["optimizer_steps"] == 0
            and report["hidden_loaded"] is False,
            "Collector contract failed",
        )
        require(sha(folder / "arrays.npz") == report["array_sha256"], "Collector arrays changed")
        with np.load(folder / "arrays.npz", allow_pickle=False) as saved:
            a = {k: saved[k] for k in saved.files}
        require(set(a) == set(originals[base]), "Array field set differs")
        for key, value in a.items():
            require(
                value.shape == originals[base][key].shape and np.isfinite(value).all(),
                "Array schema differs",
            )
            if (
                key in {"noise", "standard_targets", "normalized_targets"}
                or condition in plan["controls"]
            ):
                require(
                    np.array_equal(value, originals[base][key]),
                    "Native control/target/noise not exact",
                )
        if condition in plan["controls"]:
            require(report["same_device_control"]["passed"], "Control not accepted")
        if condition == "base640_rfull":
            with np.load(workspace / plan["reference_direction"], allow_pickle=False) as previous:
                require(
                    all(np.array_equal(a[k], previous[k]) for k in a),
                    "Prior real head1+2 direction replay differs",
                )
            require(report["same_device_control"]["passed"], "Direction replay not accepted")
        arrays[condition], reports[condition] = a, report
    metrics, effects, compact = {}, {}, []
    for condition, a in arrays.items():
        base = int(condition.split("_")[0].removeprefix("base"))
        result = metrics[condition] = {}
        effect = effects[condition] = {}
        for space in ("standard", "normalized"):
            result[space], effect[space] = {}, {}
            p, y = a[space + "_predictions"], a[space + "_targets"]
            native = arrays[f"base{base}_native"][space + "_predictions"]
            for view, indices in rows.items():
                result[space][view], effect[space][view] = {}, {}
                for window, (start, stop) in plan["windows"].items():
                    result[space][view][window], effect[space][view][window] = {}, {}
                    for group, dims in groups.items():
                        yp = y[indices, start:stop][..., dims]
                        pp = p[:, indices, start:stop][..., dims]
                        value = decompose(
                            pp,
                            yp,
                            y[rows["train40"], start:stop][..., dims],
                            native[:, rows["train40"], start:stop][..., dims],
                        )
                        result[space][view][window][group] = value
                        pair = paired_effect(native[:, indices, start:stop][..., dims], pp, yp)
                        effect[space][view][window][group] = pair
                        if space == "standard" and view == "dev5" and window == "full":
                            compact.append(
                                {
                                    "condition": condition,
                                    "group": group,
                                    "mae": value["errors"]["model"]["mae"],
                                    "mse": value["errors"]["model"]["mse"],
                                    "pair": pair,
                                    "mean_bias_delta": value["squared_scene_mean_bias_difference"],
                                    "scene_variance": value["prediction_scene_variance"],
                                    "minus_twice_covariance": value["minus_twice_scene_covariance"],
                                    "train_label_mean_mae": value["errors"]["train_label_mean"][
                                        "mae"
                                    ]["mean"],
                                    "train_label_median_mae": value["errors"]["train_label_median"][
                                        "mae"
                                    ]["mean"],
                                }
                            )
    result = {
        "id": plan["id"],
        "plan_sha256": sha(plan_path),
        "handoff_manifest_sha256": sha(recovered / "handoff-manifest.json"),
        "conditions": plan["conditions"],
        "metrics": metrics,
        "effects": effects,
        "compact": compact,
        "native_and_restore_controls_exact": True,
        "parameters_unchanged": True,
        "remote_completed_model_forwards": 720,
        "full_factor_comparison": "not measured: rmass failed isolation",
        "local_model_forwards": 0,
        "optimizer_steps": 0,
        "hidden_loaded": False,
        "m2_complete": False,
        "new_gate34": "not measured",
        "seconds": time.monotonic() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    require(
        result["seconds"] < 300 and result["peak_rss_bytes"] < 2 * 1024**3,
        "Analysis budget exceeded",
    )
    output.mkdir(parents=True, exist_ok=False)
    with (output / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(
        json.dumps({k: v for k, v in result.items() if k not in {"metrics", "effects", "compact"}})
    )
    for entry in compact:
        if entry["group"] in {"left_joint", "right_joint", "left_gripper", "right_gripper"}:
            print(json.dumps(entry))


if __name__ == "__main__":
    main()
