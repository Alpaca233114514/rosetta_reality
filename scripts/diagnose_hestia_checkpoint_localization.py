"""Localize saved Hestia 640-to-1280 errors without loading models or raw data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import yaml


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def noise_summary(values):
    a = np.asarray(values, dtype=np.float64)
    require(a.ndim == 1 and np.isfinite(a).all(), "Finite per-noise values required")
    return {"mean": float(a.mean()), "by_noise": a.tolist()}


def compare(p, q, y, reference):
    """Paired [noise,scene,time,dimension] comparison; average errors, not predictions."""
    p, q, y, reference = (np.asarray(a, dtype=np.float64) for a in (p, q, y, reference))
    require(
        p.ndim == 4
        and p.shape == q.shape
        and p.shape[1:] == y.shape
        and reference.ndim == 3
        and reference.shape[1:] == y.shape[1:]
        and all(a.size and np.isfinite(a).all() for a in (p, q, y, reference)),
        "Finite aligned predictions, targets and train references required",
    )
    axis = (1, 2, 3)
    metrics = {}
    decompositions = []
    for step, pred in ((640, p), (1280, q)):
        error = pred - y
        pc = pred - pred.mean(1, keepdims=True)
        yc = y - y.mean(0, keepdims=True)
        pv = np.square(pc).mean(axis)
        tv = float(np.square(yc).mean())
        cov = (pc * yc).mean(axis)
        bias = np.square(pred.mean(1) - y.mean(0)).mean((1, 2))
        mse = np.square(error).mean(axis)
        require(
            np.allclose(mse, tv + pv - 2 * cov + bias, rtol=1e-12, atol=1e-12),
            "Scene MSE decomposition failed",
        )
        corr = [
            float(c / np.sqrt(v * tv)) if v * tv > 0 else None for c, v in zip(cov, pv, strict=True)
        ]
        metrics[str(step)] = {
            "mae": noise_summary(np.abs(error).mean(axis)),
            "mse": noise_summary(mse),
            "scene_correlation_by_noise": corr,
            "prediction_variance": noise_summary(pv),
            "target_variance": tv,
            "covariance": noise_summary(cov),
            "squared_mean_bias": noise_summary(bias),
        }
        decompositions.append((mse, pv, cov, bias))
    a, b = decompositions
    delta = q - p
    movement = np.square(delta).mean(axis)
    residual_alignment = -2 * (delta * (y - p)).mean(axis)
    require(
        np.allclose(b[0] - a[0], movement + residual_alignment, rtol=1e-12, atol=1e-12),
        "Paired residual identity failed",
    )
    metrics["change"] = {
        "mae": noise_summary(np.abs(q - y).mean(axis) - np.abs(p - y).mean(axis)),
        "mse": noise_summary(b[0] - a[0]),
        "prediction_variance": noise_summary(b[1] - a[1]),
        "minus_twice_covariance": noise_summary(-2 * (b[2] - a[2])),
        "squared_mean_bias": noise_summary(b[3] - a[3]),
        "squared_prediction_movement": noise_summary(movement),
        "minus_twice_residual_alignment": noise_summary(residual_alignment),
    }
    metrics["constants"] = {
        "train_mean_mae": float(np.abs(y - reference.mean(0)).mean()),
        "train_median_mae": float(np.abs(y - np.median(reference, axis=0)).mean()),
    }
    return metrics


def split_rows(meta):
    episodes = meta["episodes"]
    train, dev = meta["views"]["train40"], meta["views"]["dev5"]
    require(len(episodes) == len(set(episodes)), "Duplicate episode identity")
    require(len(train) == 40 and len(dev) == 5, "Registered split sizes differ")
    require(len(set(train)) == 40 and len(set(dev)) == 5, "Duplicate view identity")
    require(not set(train) & set(dev), "Train/development overlap")
    require(set(episodes) == set(train) | set(dev), "Unregistered rows")
    require(not set(episodes) & set(meta["hidden_episodes"]), "Hidden row present")
    require(meta["hidden_test_loaded"] is False, "Hidden-access metadata differs")
    return {
        view: [episodes.index(ep) for ep in values]
        for view, values in (("train40", train), ("dev5", dev))
    }


def transitions(values, threshold):
    a = np.asarray(values, dtype=np.float64)
    require(a.ndim == 1 and len(a) > 1 and np.isfinite(a).all(), "Invalid event series")
    opened = a >= threshold
    return {
        "start": float(a[0]),
        "end": float(a[-1]),
        "starts_below_threshold": bool(not opened[0]),
        "downcross_slots": (np.flatnonzero(opened[:-1] & ~opened[1:]) + 1).tolist(),
        "upcross_slots": (np.flatnonzero(~opened[:-1] & opened[1:]) + 1).tolist(),
    }


def validate_layout(arrays, meta, mask):
    n, t, d = len(meta["episodes"]), meta["chunk_length"], len(meta["dimensions"])
    k = len(meta["noise_conditions"])
    require(
        mask.shape == (n, t, d) and mask.dtype == np.bool_ and mask.all(),
        "Complete physical-action boolean mask required; no padded observations",
    )
    require(
        arrays["noise"].shape == (k, 1, t, meta["max_action_dim"]),
        "Noise must retain padded native model width",
    )
    for space in ("standard", "normalized"):
        require(
            arrays[space + "_predictions"].shape == (k, n, t, d)
            and arrays[space + "_targets"].shape == (n, t, d),
            "Saved predictions/targets must use physical-action width",
        )
    require(all(np.isfinite(a).all() for a in arrays.values()), "Nonfinite input")


def load_inputs(plan):
    require(
        os.environ.get("ROSETTA_CONTAINER_IMAGE_ID") == plan["image"],
        "Registered offline container required",
    )
    for name, expected in plan["sha256"].items():
        require(not Path(name).is_symlink() and file_hash(name) == expected, f"Input drift: {name}")
    closure = read_json(plan["curve_closure"])
    root = Path(plan["curve_root"])
    require(
        file_hash(root / "handoff-manifest.json") == closure["transfer"]["manifest_sha256"],
        "Recovered manifest differs from closure",
    )
    handoff = read_json(root / "handoff-manifest.json")
    for name, entry in handoff["files"].items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe input path")
        path = root / relative
        require(
            not path.is_symlink()
            and path.stat().st_size == entry["bytes"]
            and file_hash(path) == entry["sha256"],
            f"Recovered input drift: {name}",
        )
    prior = read_json(plan["prior_summary"])
    require(file_hash(plan["prior_summary"]) == closure["analysis_sha256"], "Prior analysis drift")
    old_root = Path(plan["historical_bundle"])
    manifest = read_json(old_root / "manifest.json")
    meta = manifest["metadata"]
    rows = split_rows(meta)
    contract = yaml.safe_load(Path(plan["action_contract"]).read_text())["action"]
    require(
        file_hash(plan["action_contract"]) == meta["common_identity"]["physical_contract_sha256"],
        "Physical Action Contract drift",
    )
    require(
        meta["chunk_length"] == contract["chunk_length"] == plan["chunk_length"]
        and contract["frequency_hz"] == plan["frequency_hz"],
        "Time contract drift",
    )
    for actual, expected in zip(meta["dimensions"], contract["dimensions"], strict=True):
        require(
            all(actual[k] == expected[k] for k in ("name", "unit", "minimum", "maximum")),
            "Action ordering, units or bounds drift",
        )
    old = {}
    for name, entry in manifest["arrays"].items():
        require(entry["path"] == name + ".npy", "Historical array path differs")
        path = old_root / entry["path"]
        require(
            not path.is_symlink() and file_hash(path) == entry["sha256"], "Historical file drift"
        )
        old[name] = np.load(path, allow_pickle=False)
    validate_layout(old, meta, old["valid_mask"])
    n = len(meta["episodes"])
    arrays = {}
    for step in plan["steps"]:
        report = read_json(root / f"{step:06d}/result.json")
        path = root / f"{step:06d}/arrays.npz"
        require(
            report["step"] == step and report["array_sha256"] == file_hash(path),
            "Checkpoint array identity differs",
        )
        require(
            report["raw_images_exact"]
            and report["all_parameters_unchanged"]
            and report["optimizer_steps"] == 0
            and report["hidden_loaded"] is False
            and report["native_trace_count"] == len(meta["noise_conditions"]) * n,
            "Historical collection prerequisites failed",
        )
        with np.load(path, allow_pickle=False) as saved:
            arrays[step] = {k: saved[k] for k in saved.files}
        validate_layout(arrays[step], meta, old["valid_mask"])
        for name in ("noise", "standard_targets", "normalized_targets"):
            require(np.array_equal(arrays[step][name], old[name]), "Target/noise identity differs")
        if step == 1280:
            for name in ("standard_predictions", "normalized_predictions"):
                require(
                    np.array_equal(arrays[step][name], old[name]), "Final CUDA reproduction failed"
                )
    return arrays, meta, rows, prior, len(handoff["files"])


def analyze(arrays, meta, rows, plan, prior):
    groups = {}
    for unit, kind in (("radian", "joint"), ("normalized", "gripper")):
        groups[kind] = [i for i, d in enumerate(meta["dimensions"]) if d["unit"] == unit]
        for side in ("left", "right"):
            groups[side + "_" + kind] = [
                i for i in groups[kind] if meta["dimensions"][i]["name"].startswith(side + "_")
            ]
    require(all(groups.values()), "Missing action group")
    result = {
        "groups": groups,
        "windows": plan["windows"],
        "noise_conditions": meta["noise_conditions"],
        "episodes": {v: meta["views"][v] for v in rows},
        "metrics": {},
        "localization": {},
        "gripper_events": [],
        "historical_metric_group_checks": 0,
    }
    for space in ("standard", "normalized"):
        p, q = (arrays[step][space + "_predictions"] for step in (640, 1280))
        target = arrays[640][space + "_targets"]
        result["metrics"][space] = {}
        for view, indices in rows.items():
            vr = result["metrics"][space][view] = {}
            for window, (start, stop) in plan["windows"].items():
                wr = vr[window] = {}
                for group, dims in groups.items():
                    y = target[indices, start:stop][..., dims]
                    ref = target[rows["train40"], start:stop][..., dims]
                    a, b = (x[:, indices, start:stop][..., dims] for x in (p, q))
                    wr[group] = compare(a, b, y, ref)
                    if window in ("full", "first", "last"):
                        for step in (640, 1280):
                            old = prior["steps"][str(step)][space][view][window][group]
                            for metric in ("mae", "mse"):
                                require(
                                    np.allclose(
                                        wr[group][str(step)][metric]["by_noise"],
                                        old[metric + "_by_noise"],
                                        rtol=1e-12,
                                        atol=1e-12,
                                    ),
                                    "Independent historical metric reproduction failed",
                                )
                            result["historical_metric_group_checks"] += 1
        if space == "standard":
            for group, dims in groups.items():
                dev = rows["dev5"]
                y = target[dev][..., dims]
                a, b = (x[:, dev][..., dims] for x in (p, q))
                ae, be = np.abs(a - y), np.abs(b - y)
                delta = be - ae
                # Every dimension/time/scene contributes with the full fixed denominator.
                full_delta = delta.mean((1, 2, 3))
                per_scene_contribution = delta.mean((2, 3)) / len(dev)
                require(
                    np.allclose(per_scene_contribution.sum(1), full_delta, atol=1e-12),
                    "Scene contribution sum failed",
                )
                blocks = {}
                for block in plan["partition"]:
                    start, stop = plan["windows"][block]
                    contribution = delta[:, :, start:stop].sum((1, 2, 3)) / np.prod(delta.shape[1:])
                    blocks[block] = noise_summary(contribution)
                require(
                    np.allclose(
                        sum(np.array(v["by_noise"]) for v in blocks.values()),
                        full_delta,
                        atol=1e-12,
                    ),
                    "Time contribution sum failed",
                )
                result["localization"][group] = {
                    "full_mae_delta": noise_summary(full_delta),
                    "time_contributions": blocks,
                    "scene_contributions_by_noise": per_scene_contribution.tolist(),
                    "per_scene_mae_by_noise": {
                        "640": ae.mean((2, 3)).tolist(),
                        "1280": be.mean((2, 3)).tolist(),
                    },
                    "per_scene_slot_mae_by_noise": {
                        "640": ae.mean(3).tolist(),
                        "1280": be.mean(3).tolist(),
                    },
                    "per_dimension_mae_by_noise": {
                        "640": ae.mean((1, 2)).tolist(),
                        "1280": be.mean((1, 2)).tolist(),
                    },
                    "leave_one_scene_out_delta": {
                        str(meta["episodes"][idx]): noise_summary(
                            np.delete(delta, i, axis=1).mean((1, 2, 3))
                        )
                        for i, idx in enumerate(dev)
                    },
                }
            for idx in rows["dev5"]:
                for dim in groups["gripper"]:
                    result["gripper_events"].append(
                        {
                            "episode": meta["episodes"][idx],
                            "dimension": meta["dimensions"][dim]["name"],
                            "threshold": plan["gripper_threshold"],
                            "target": transitions(target[idx, :, dim], plan["gripper_threshold"]),
                            "predictions": {
                                str(step): [
                                    transitions(a[i, idx, :, dim], plan["gripper_threshold"])
                                    for i in range(len(meta["noise_conditions"]))
                                ]
                                for step, a in ((640, p), (1280, q))
                            },
                        }
                    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    plan = read_json(args.plan)
    require(plan["steps"] == [640, 1280], "Only the registered pair may be compared")
    partition = [slot for name in plan["partition"] for slot in range(*plan["windows"][name])]
    require(partition == list(range(plan["chunk_length"])), "Partition must cover every slot once")
    output = Path(plan["output"])
    require(output.is_dir() and not any(output.iterdir()), "New empty output directory required")
    arrays, meta, rows, prior, count = load_inputs(plan)
    result = analyze(arrays, meta, rows, plan, prior)
    elapsed = time.monotonic() - started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    require(
        elapsed < plan["maximum_seconds"] and rss < plan["maximum_rss_bytes"], "Budget exceeded"
    )
    with (output / "summary.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    receipt = {
        "status": "completed",
        "plan_sha256": file_hash(args.plan),
        "summary_sha256": file_hash(output / "summary.json"),
        "recovered_files_verified": count,
        "historical_metric_group_checks": result["historical_metric_group_checks"],
        "final_cuda_arrays_exact": True,
        "all_additive_identities_passed": True,
        "seconds": elapsed,
        "peak_rss_bytes": rss,
        "new_model_forwards": 0,
        "new_optimizer_steps": 0,
        "hidden_loaded": False,
        "checkpoint_selection": False,
        "gate34": "not measured",
        "m2_complete": False,
    }
    with (output / "result.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
