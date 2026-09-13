"""Analyze the four registered causal conditions from verified saved arrays only."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from scripts.diagnose_hestia_checkpoint_localization import (
    file_hash,
    read_json,
    require,
    split_rows,
    validate_layout,
)
from scripts.diagnose_hestia_command_pose import (
    CommandKinematics,
    rotation_error,
    runtime_identity,
)
from scripts.hestia_parameter_crossover import kv_keys

CONDITIONS = ("base1280", "base640", "base1280_kv640", "base640_kv1280")
IMAGE = "sha256:fb3c1bbda42881fac9b7725d3acb436119934f0c60af29747339d5951c3039da"
ROOT = Path("runs/hestia-parameter-crossover-v2-recovered-20260912-002/verified")
OUTPUT = Path("runs/hestia-parameter-crossover-analysis-20260912-001/verified")


def summarize(values):
    return {
        "mean": float(values.mean()),
        "by_noise": values.mean(1).tolist(),
        "by_episode": values.mean(0).tolist(),
        "by_noise_episode": values.tolist(),
    }


def causal(values):
    a, b, c, d = (values[name] for name in CONDITIONS)
    delta, recovery, damage = a - b, a - c, d - b
    interaction = a - c - d + b
    require(np.allclose(interaction, recovery - damage), "Interaction identity failed")
    return {
        "endpoint_delta": summarize(delta),
        "forward_recovery": summarize(recovery),
        "reverse_damage": summarize(damage),
        "interaction": summarize(interaction),
        "bidirectional_regression_by_noise": (
            (delta.mean(1) > 0) & (recovery.mean(1) > 0) & (damage.mean(1) > 0)
        ).tolist(),
        "recovery_fraction_of_positive_mean_delta": (
            float(recovery.mean() / delta.mean()) if delta.mean() > 0 else None
        ),
        "damage_fraction_of_positive_mean_delta": (
            float(damage.mean() / delta.mean()) if delta.mean() > 0 else None
        ),
    }


def main():
    started = time.monotonic()
    require(os.environ.get("ROSETTA_CONTAINER_IMAGE_ID") == IMAGE, "Pinned image required")
    require(os.environ.get("HF_HUB_OFFLINE") == "1", "Offline execution required")
    require(not OUTPUT.exists(), "Preserve previous analysis")
    handoff = read_json(ROOT / "handoff-manifest.json")
    receipt = read_json("runs/hestia-parameter-crossover-v2-receiver-20260912-002/receipt.json")
    require(receipt["all_file_sha256_matched"] is True, "Verified transfer required")
    require(
        receipt["manifest_sha256"] == file_hash(ROOT / "handoff-manifest.json"), "Receipt mismatch"
    )
    for name, entry in handoff["files"].items():
        p = ROOT / name
        require(not p.is_symlink() and p.resolve().is_relative_to(ROOT.resolve()), "Unsafe input")
        require(
            p.stat().st_size == entry["bytes"] and file_hash(p) == entry["sha256"], "Input drift"
        )
    worker = read_json(ROOT / "worker-exited.json")
    require(
        worker["completed_conditions"] == list(CONDITIONS) and not worker["error"], "Incomplete run"
    )
    template = read_json(
        "reports/training/m2-smolvla-hestia-parameter-crossover-v2-template-2026-09-12.json"
    )
    historical = Path("runs/hestia-recovered-20260911-001/C-first")
    require(
        file_hash(historical / "manifest.json")
        == template["sha256"][str(historical / "manifest.json")],
        "Metadata drift",
    )
    manifest = read_json(historical / "manifest.json")
    meta = manifest["metadata"]
    rows = split_rows(meta)
    require(rows["train40"] == list(range(40)) and rows["dev5"] == list(range(40, 45)), "Row drift")
    mask_path = historical / manifest["arrays"]["valid_mask"]["path"]
    require(file_hash(mask_path) == manifest["arrays"]["valid_mask"]["sha256"], "Mask drift")
    mask = np.load(mask_path, allow_pickle=False)
    arrays, reports = {}, {}
    for name in CONDITIONS:
        reports[name] = r = read_json(ROOT / name / "result.json")
        require(
            r["condition"] == name and r["all_parameters_unchanged"] and r["raw_images_exact"],
            "Collection failure",
        )
        require(
            r["optimizer_steps"] == 0 and r["model_forwards"] == 180 and not r["hidden_loaded"],
            "Scope drift",
        )
        with np.load(ROOT / name / "arrays.npz", allow_pickle=False) as archive:
            arrays[name] = {k: archive[k] for k in archive.files}
        validate_layout(arrays[name], meta, mask)
        if name in CONDITIONS[:2]:
            require(r["same_device_control"]["passed"], "Endpoint failed")
            for key in ("normalized_predictions", "standard_predictions"):
                require(r["same_device_control"][key]["max_abs"] == 0, "Inexact endpoint")
            step = "001280" if name == "base1280" else "000640"
            reference_path = (
                Path("runs/hestia-cuda-curve-recovered-20260912-001") / step / "arrays.npz"
            )
            require(
                file_hash(reference_path) == template["sha256"][str(reference_path)],
                "Reference drift",
            )
            with np.load(reference_path, allow_pickle=False) as reference:
                for key in ("normalized_predictions", "standard_predictions"):
                    require(
                        np.array_equal(arrays[name][key], reference[key]), "Endpoint array differs"
                    )
        else:
            intervention = read_json(ROOT / name / "intervention.json")
            allowed = set(kv_keys())
            require(set(intervention["changed_keys"]) == allowed, "Incomplete intervention")
            require(
                intervention["other_parameters_exact"] and intervention["donor_copy_exact"],
                "Inexact intervention",
            )
            before, after = intervention["before"], intervention["after"]
            require(set(before) == set(after), "Parameter set drift")
            require(
                {k for k in before if before[k] != after[k]} == allowed, "Parameter digest mismatch"
            )
        for key in ("standard_targets", "normalized_targets", "noise"):
            require(
                np.array_equal(arrays[name][key], arrays[CONDITIONS[0]][key]), "Unpaired inputs"
            )
    require(
        runtime_identity()
        == read_json("runs/hestia-command-pose-preflight-20260912-001/runtime.json"),
        "FK identity drift",
    )
    fk = CommandKinematics(meta["dimensions"])
    y = arrays[CONDITIONS[0]]["standard_targets"].astype(np.float64)
    predictions = {name: a["standard_predictions"].astype(np.float64) for name, a in arrays.items()}
    # Constants retain the historical definition: per-slot mean/median of train40.
    # Their train scores are descriptive in-sample references, not cross-validation.
    predictions.update(
        {
            "train_mean": np.broadcast_to(y[:40].mean(0), (4, *y.shape)),
            "train_median": np.broadcast_to(np.median(y[:40], axis=0), (4, *y.shape)),
        }
    )
    target_position, target_rotation = fk.poses(y)
    errors = {}
    for name, pred in predictions.items():
        position, rotation = fk.poses(pred)
        e = {}
        for side, arm in enumerate(("left", "right")):
            for joint, suffix in ((True, "joint_rad"), (False, "gripper")):
                dims = [
                    i
                    for i, d in enumerate(meta["dimensions"])
                    if d["name"].startswith(arm + "_") and (d["unit"] == "radian") == joint
                ]
                e[arm + "_" + suffix] = np.abs(pred[..., dims] - y[None, ..., dims]).mean(-1)
            e[arm + "_position_m"] = np.linalg.norm(
                position[..., side, :] - target_position[None, ..., side, :], axis=-1
            )
            target = np.broadcast_to(
                target_rotation[None, ..., side, :, :], rotation[..., side, :, :].shape
            )
            e[arm + "_orientation_rad"] = rotation_error(rotation[..., side, :, :], target)
        errors[name] = e
    results = {}
    for split, indices in rows.items():
        results[split] = {}
        for window, slots in (("full", slice(0, 50)), ("first", slice(0, 1))):
            results[split][window] = out = {}
            for metric in errors[CONDITIONS[0]]:
                values = {name: e[metric][:, indices, slots].mean(-1) for name, e in errors.items()}
                out[metric] = {
                    "errors": {name: summarize(v) for name, v in values.items()},
                    "causal": causal(values),
                    "beats_both_constants_by_noise": {
                        name: (
                            (values[name].mean(1) < values["train_mean"].mean(1))
                            & (values[name].mean(1) < values["train_median"].mean(1))
                        ).tolist()
                        for name in CONDITIONS
                    },
                }
    OUTPUT.mkdir()
    result = {
        "schema_version": 1,
        "source_manifest_sha256": receipt["manifest_sha256"],
        "analysis_source_sha256": file_hash(Path(__file__)),
        "image": IMAGE,
        "episodes": {name: [meta["episodes"][i] for i in rr] for name, rr in rows.items()},
        "noise_conditions": meta["noise_conditions"],
        "results": results,
        "seconds": time.monotonic() - started,
        "model_forwards_local": 0,
        "optimizer_steps": 0,
        "fk_is_commanded_pose_only": True,
        "hidden_loaded": False,
        "new_gate3": "not measured",
        "new_gate4": "not measured",
    }
    with (OUTPUT / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "status": "passed",
                "seconds": result["seconds"],
                "dev_full": {
                    k: {
                        "errors": {n: v["mean"] for n, v in m["errors"].items()},
                        "causal": m["causal"],
                    }
                    for k, m in results["dev5"]["full"].items()
                },
            }
        )
    )


if __name__ == "__main__":
    main()
