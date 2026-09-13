"""Analyze all eight single-layer V causal conditions from verified saved arrays only."""

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
from scripts.hestia_image_value_component import CONDITIONS, component_mode

IMAGE = "sha256:fb3c1bbda42881fac9b7725d3acb436119934f0c60af29747339d5951c3039da"
ROOT = Path("runs/hestia-image-value-component-recovered-20260912-001/verified")
OUTPUT = Path("runs/hestia-image-value-component-analysis-20260912-001/verified")


def summarize(values):
    return {
        "mean": float(values.mean()),
        "by_noise": values.mean(1).tolist(),
        "by_episode": values.mean(0).tolist(),
        "by_noise_episode": values.tolist(),
    }


def scene_error_terms(prediction, target):
    """Exact MSE decomposition over paired scenes, independently per saved noise."""
    p, y = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    require(p.ndim == 4 and p.shape[1:] == y.shape, "Paired scene/time/dimension layout required")
    require(np.isfinite(p).all() and np.isfinite(y).all(), "Nonfinite decomposition input")
    pc, yc = p - p.mean(1, keepdims=True), y - y.mean(0, keepdims=True)
    mse = np.square(p - y).mean((1, 2, 3))
    bias = np.square(p.mean(1) - y.mean(0)).mean((1, 2))
    variance = np.square(pc).mean((1, 2, 3))
    target_variance = np.full(len(p), np.square(yc).mean())
    covariance_term = -2 * (pc * yc).mean((1, 2, 3))
    require(
        np.allclose(
            mse, bias + variance + target_variance + covariance_term, atol=1e-12, rtol=1e-12
        ),
        "Scene error identity failed",
    )
    return {
        "mse": mse,
        "squared_scene_mean_bias": bias,
        "prediction_scene_variance": variance,
        "target_scene_variance": target_variance,
        "minus_twice_scene_covariance": covariance_term,
    }


def causal(values, kind):
    forward, reverse = f"base1280_v{kind}640", f"base640_v{kind}1280"
    a, b, c, d = (values[name] for name in ("base1280", "base640", forward, reverse))
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
    receipt = read_json("runs/hestia-image-value-component-receiver-20260912-001/receipt.json")
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
        "reports/training/m2-smolvla-hestia-image-value-component-template-2026-09-12.json"
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
        intervention = read_json(ROOT / name / "intervention.json")
        require(intervention == r["image_value_component"], "Intervention report differs")
        require(intervention["mode"] == component_mode(name), "Wrong component")
        require(intervention["parameter_intervention"] is False, "Unexpected weight substitution")
        require(
            intervention["scope"] == "real_image_tokens_0_64"
            and intervention["untouched_positions"] == [64, 241],
            "Intervention scope drift",
        )
        require(intervention["hook_calls"] == 14400, "Missing denoise calls")
        require(intervention["projected_donor_evaluations"] == 360, "Missing row/layer projections")
        require(intervention["native_inputs_and_unmodified_outputs_repeat_exactly"], "Prefix drift")
        require(
            name == "base1280" or intervention["reciprocal_delta_matches_calibration"],
            "Delta drift",
        )
        require(intervention["training_rows_for_component_means"] == list(range(40)), "Split leak")
        require(not intervention["labels_used_for_components"], "Label leakage")
        require(
            intervention["calibration_sha256"] == file_hash(ROOT / "base1280/calibration.json"),
            "Calibration drift",
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
    previous = Path("runs/hestia-value-route-recovered-20260912-001/verified")
    previous_receipt = read_json("runs/hestia-value-route-receiver-20260912-001/receipt.json")
    previous_manifest = read_json(previous / "handoff-manifest.json")
    require(
        previous_receipt["manifest_sha256"] == file_hash(previous / "handoff-manifest.json"),
        "Previous receipt drift",
    )
    for name in ("base1280", "base640", "base1280_vimage640", "base640_vimage1280"):
        p = previous / name / "arrays.npz"
        require(
            file_hash(p) == previous_manifest["files"][f"{name}/arrays.npz"]["sha256"],
            "Previous arrays drift",
        )
        with np.load(p, allow_pickle=False) as saved:
            for key in ("standard_targets", "normalized_targets", "noise"):
                require(
                    np.array_equal(saved[key], arrays[CONDITIONS[0]][key]),
                    "Cross-run pairing drift",
                )
            if name in CONDITIONS[:2]:
                for key in ("normalized_predictions", "standard_predictions"):
                    require(
                        np.array_equal(saved[key], arrays[name][key]),
                        "Cross-run native endpoint differs",
                    )
            else:
                current = (
                    "base1280_vfull640" if name == "base1280_vimage640" else "base640_vfull1280"
                )
                for key in ("normalized_predictions", "standard_predictions"):
                    require(
                        np.array_equal(saved[key], arrays[current][key]),
                        "Full activation control differs from prior parameter substitution",
                    )
                require(
                    reports[current]["same_device_control"]["passed"],
                    "Full positive control failed",
                )
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
                    "causal": {
                        str(kind): causal(values, kind)
                        for kind in ("full", "global", "token", "scene")
                    },
                    "group_minus_sum_individual_effects": {
                        "forward_recovery": summarize(
                            values["base1280"]
                            - values["base1280_vfull640"]
                            - sum(
                                values["base1280"] - values[f"base1280_v{i}640"]
                                for i in ("global", "token", "scene")
                            )
                        ),
                        "reverse_damage": summarize(
                            values["base640_vfull1280"]
                            - values["base640"]
                            - sum(
                                values[f"base640_v{i}1280"] - values["base640"]
                                for i in ("global", "token", "scene")
                            )
                        ),
                    },
                    "beats_both_constants_by_noise": {
                        name: (
                            (values[name].mean(1) < values["train_mean"].mean(1))
                            & (values[name].mean(1) < values["train_median"].mean(1))
                        ).tolist()
                        for name in CONDITIONS
                    },
                }
    decomposition = {}
    for split, indices in rows.items():
        decomposition[split] = {}
        windows = {
            "first": (0, 1),
            "full": (0, 50),
            **{f"slots_{i}_{i + 9}": (i, i + 10) for i in range(0, 50, 10)},
        }
        for window, (start, stop) in windows.items():
            decomposition[split][window] = {}
            for arm in ("left", "right"):
                for joint, suffix in ((True, "joint"), (False, "gripper")):
                    dims = [
                        i
                        for i, d in enumerate(meta["dimensions"])
                        if d["name"].startswith(arm + "_") and (d["unit"] == "radian") == joint
                    ]
                    target = y[indices, start:stop][..., dims]
                    terms = {
                        name: scene_error_terms(p[:, indices, start:stop][..., dims], target)
                        for name, p in predictions.items()
                    }
                    changes = {}
                    for name in CONDITIONS[2:]:
                        base = "base1280" if name.startswith("base1280") else "base640"
                        delta = {k: terms[name][k] - terms[base][k] for k in terms[name]}
                        changes[name] = {
                            k: {"mean": float(v.mean()), "by_noise": v.tolist()}
                            for k, v in delta.items()
                        }
                    decomposition[split][window][arm + "_" + suffix] = changes
    OUTPUT.mkdir()
    result = {
        "schema_version": 1,
        "source_manifest_sha256": receipt["manifest_sha256"],
        "analysis_source_sha256": file_hash(Path(__file__)),
        "image": IMAGE,
        "episodes": {name: [meta["episodes"][i] for i in rr] for name, rr in rows.items()},
        "noise_conditions": meta["noise_conditions"],
        "results": results,
        "scene_error_decomposition_changes": decomposition,
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
