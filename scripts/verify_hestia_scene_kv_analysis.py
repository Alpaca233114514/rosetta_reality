"""Independently recompute saved-array audit fields, without policy or calibration fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_summary(record, values):
    a = np.asarray(values, dtype=np.float64)
    expected = {
        "mean": sum(float(x) for x in a.flat) / a.size,
        "by_noise": [sum(row) / len(row) for row in a],
        "by_episode": [sum(a[:, i]) / len(a) for i in range(a.shape[1])],
        "by_noise_episode": a,
        "leave_one_episode_out_by_noise": [
            [np.delete(row, i).mean() for i in range(len(row))] for row in a
        ],
    }
    count = 0
    for key, value in expected.items():
        np.testing.assert_allclose(record[key], value, atol=1e-12, rtol=1e-12)
        count += np.asarray(value).size
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not Path("/.dockerenv").exists():
        raise ValueError("Container execution required")
    if args.output.exists():
        raise FileExistsError("Preserve verification")
    result, meta = read(args.analysis), read(args.metadata)["metadata"]
    arrays = {}
    scalar_checks = 0
    for base in (640, 1280):
        for mode in ("native", "self", "kmean", "vmean", "kvmean"):
            condition = f"base{base}_{mode}"
            with np.load(args.root / condition / "arrays.npz", allow_pickle=False) as archive:
                arrays[condition] = {k: archive[k] for k in archive.files}
    for name, record in result["metrics"].items():
        base, space, view, window, group, metric = name.split("/")
        indices = [meta["episodes"].index(e) for e in meta["views"][view]]
        train_indices = [meta["episodes"].index(e) for e in meta["views"]["train40"]]
        side, kind = group.split("_")
        unit = "radian" if kind == "joint" else "normalized"
        dims = [
            i
            for i, d in enumerate(meta["dimensions"])
            if d["name"].startswith(side + "_") and d["unit"] == unit
        ]
        stop = 50 if window == "full" else 1
        target = arrays[f"base{base}_native"][space + "_targets"].astype(np.float64)
        train = target[train_indices, :stop][..., dims]
        errors, predictions = {}, {}
        for mode in ("native", "self", "kmean", "vmean", "kvmean"):
            saved = arrays[f"base{base}_{mode}"][space + "_predictions"]
            errors[mode] = np.empty((4, len(indices)), dtype=np.float64)
            predictions[mode] = saved[:, indices, :stop][..., dims].astype(np.float64)
            # Explicit scene/noise indexing independently checks the primary vectorized slicing.
            for noise in range(4):
                for position, row in enumerate(indices):
                    diff = (
                        saved[noise, row, :stop][:, dims].astype(np.float64)
                        - target[row, :stop][:, dims]
                    )
                    errors[mode][noise, position] = np.mean(
                        np.abs(diff) if metric == "mae" else diff * diff
                    )
            scalar_checks += check_summary(record["errors"][mode], errors[mode])
        y = target[indices, :stop][..., dims]
        for kind, constant in (("mean", train.mean(0)), ("median", np.median(train, axis=0))):
            error = constant - y
            value = np.mean(np.abs(error) if metric == "mae" else error**2)
            np.testing.assert_allclose(record["constants"][kind], value, atol=1e-12, rtol=1e-12)
            scalar_checks += 1
        a, b, c, d = [errors[mode] for mode in ("native", "kmean", "vmean", "kvmean")]
        joint, k, v = d - a, b - a, c - a
        effects = {
            "k_given_old_v": k,
            "k_given_new_v": joint - v,
            "v_given_old_k": v,
            "v_given_new_k": joint - k,
            "interaction": joint - k - v,
            "k_order_average": 0.5 * (joint + k - v),
            "v_order_average": 0.5 * (joint - k + v),
            "joint_effect": joint,
            "v_minus_k_order_average": v - k,
        }
        for key, value in effects.items():
            scalar_checks += check_summary(record["factorial"][key], value)
        for mode in ("kmean", "vmean", "kvmean"):
            scalar_checks += check_summary(record["gain_from_native"][mode], a - errors[mode])
        for mode, error in errors.items():
            expected = error.mean(1) < min(record["constants"].values())
            np.testing.assert_array_equal(record["beats_both_constants"][mode], expected)
            scalar_checks += 4
        if metric == "mse":
            for mode, p in predictions.items():
                mean_p, mean_y = p.mean(1), y.mean(0)
                variance = (p * p).mean((1, 2, 3)) - (mean_p * mean_p).mean((1, 2))
                cross = (p * y).mean((1, 2, 3)) - (mean_p * mean_y).mean((1, 2))
                bias = ((mean_p - mean_y) ** 2).mean((1, 2)) - (
                    (train.mean(0) - mean_y) ** 2
                ).mean()
                gap = errors[mode].mean(1) - ((train.mean(0) - y) ** 2).mean()
                for key, value in (
                    ("bias", bias),
                    ("variance", variance),
                    ("minus_twice_covariance", -2 * cross),
                    ("gap_to_train_mean", gap),
                ):
                    np.testing.assert_allclose(
                        record["mse_scene_decomposition"][mode][key], value, atol=1e-12, rtol=1e-12
                    )
                    scalar_checks += 4
    verified = {
        "status": "passed",
        "independent_scalar_checks": int(scalar_checks),
        "analysis_sha256": digest(args.analysis),
        "metadata_sha256": digest(args.metadata),
        "new_model_forwards": 0,
        "optimizer_steps": 0,
        "hidden_loaded": False,
    }
    with args.output.open("x") as stream:
        json.dump(verified, stream, indent=2, allow_nan=False)
    print(json.dumps(verified))


if __name__ == "__main__":
    main()
