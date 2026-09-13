"""Read actual CUDA C prefixes with train-only nested linear probes."""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import numpy as np

from scripts.diagnose_hestia_checkpoint_localization import (
    file_hash,
    read_json,
    require,
    split_rows,
)
from scripts.diagnose_hestia_command_pose import CommandKinematics, runtime_identity
from scripts.diagnose_hestia_pose_readout import describe, pose_metrics
from scripts.diagnose_hestia_scene_phase import first_upcross


def linear_weights(train, query, alphas):
    x, q = np.asarray(train, float), np.asarray(query, float)
    require(x.ndim == q.ndim == 2 and x.shape[1] == q.shape[1], "Matching feature widths required")
    require(np.isfinite(x).all() and np.isfinite(q).all(), "Nonfinite features")
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    z, v = (x - mean) / scale, (q - mean) / scale
    kernel, cross = z @ z.T, v @ z.T
    result = []
    for alpha in alphas:
        require(np.isfinite(alpha) and alpha > 0, "Positive alpha required")
        weight = np.linalg.solve(kernel + alpha * np.eye(len(x)), cross.T).T
        weight += (1 - weight.sum(1, keepdims=True)) / len(x)
        result.append(weight)
    return np.stack(result)


def multi_crossfit(features, target, groups, alphas, seed, train_count=40):
    x, y = np.asarray(features, float), np.asarray(target, float)
    require(
        x.ndim == y.ndim == 2 and len(x) == len(y) and 10 <= train_count < len(y),
        "Aligned split required",
    )
    require(np.isfinite(x).all() and np.isfinite(y).all(), "Finite targets required")
    require(
        sorted(i for cols in groups.values() for i in cols) == list(range(y.shape[1])),
        "Partition required",
    )
    prediction, means, medians = np.empty_like(y), np.empty_like(y), np.empty_like(y)
    records = []
    for query in range(train_count + 1):
        held = np.array([query]) if query < train_count else np.arange(train_count, len(y))
        donors = np.array([j for j in range(train_count) if j != query])
        inner_seed = seed + query if query < train_count else seed
        folds = np.array_split(np.random.default_rng(inner_seed).permutation(donors), 5)
        oof = np.empty((len(alphas), len(donors), y.shape[1]))
        lookup = {j: i for i, j in enumerate(donors)}
        for validation in folds:
            fit = np.setdiff1d(donors, validation)
            weights = linear_weights(x[fit], x[validation], alphas)
            oof[:, [lookup[j] for j in validation]] = weights @ y[fit]
        weights = linear_weights(x[donors], x[held], alphas)
        candidates = weights @ y[donors]
        record = {"query_rows": held.tolist(), "donor_rows": donors.tolist(), "groups": {}}
        for name, cols in groups.items():
            scores = np.abs(oof[..., cols] - y[donors][:, cols][None]).mean((1, 2))
            choice = int(np.argmin(scores))
            for j, h in enumerate(held):
                prediction[h, cols] = candidates[choice, j, cols]
            record["groups"][name] = {"alpha": float(alphas[choice]), "inner_mae": scores.tolist()}
        means[held], medians[held] = y[donors].mean(0), np.median(y[donors], axis=0)
        records.append(record)
    return prediction, means, medians, records


def pool(k, v, bins):
    require(k.shape == v.shape == (1, 64, 320), "Native prefix shape differs")
    require(bins in (1, 2), "Registered pooling required")
    a = np.concatenate((k[0], v[0]), axis=-1).astype(float).reshape(8, 8, 640)
    return np.concatenate(
        [
            a[np.ix_(r, c)].mean((0, 1))
            for r in np.array_split(np.arange(8), bins)
            for c in np.array_split(np.arange(8), bins)
        ]
    )


def run(plan, plan_path):
    start = time.monotonic()
    output = Path(plan["output"])
    require(output.is_dir() and not any(output.iterdir()), "Fresh output required")
    require(
        os.environ.get("ROSETTA_CONTAINER_IMAGE_ID") == plan["image"]
        and os.environ.get("HF_HUB_OFFLINE") == "1",
        "Pinned offline container required",
    )
    for name, sha in plan["sha256"].items():
        require(file_hash(name) == sha, f"Input/source drift: {name}")
    require(runtime_identity() == read_json(plan["runtime_identity"]), "Native FK identity differs")
    meta = read_json(plan["metadata"])["metadata"]
    eps = meta["episodes"]
    rows = split_rows(meta)
    require(
        rows["train40"] == list(range(40)) and rows["dev5"] == list(range(40, 45)),
        "Train-first order required",
    )
    root = Path(plan["prefix_root"])
    receipt = read_json(root / "001280/prefix-observer.json")
    require(
        receipt["status"] == "passed"
        and receipt["native_full_arrays_exact"]
        and len(receipt["records"]) == 45,
        "Complete exact collector required",
    )
    feature = {
        f"layer{layer}_{stage}_{b}": []
        for layer in (1, 15)
        for stage in ("input", "projected")
        for b in (1, 2)
    }
    for i, record in enumerate(receipt["records"]):
        require(
            record["row"] == i and record["file"] == f"prefix-row-{i:03d}.npz",
            "Prefix row identity differs",
        )
        path = root / "001280" / record["file"]
        require(file_hash(path) == record["sha256"], "Prefix hash differs")
        with np.load(path, allow_pickle=False) as a:
            for layer in (1, 15):
                for stage in ("input", "projected"):
                    for b in (1, 2):
                        feature[f"layer{layer}_{stage}_{b}"].append(
                            pool(a[f"layer{layer}_k_{stage}"], a[f"layer{layer}_v_{stage}"], b)
                        )
    with np.load(root / "001280/arrays.npz", allow_pickle=False) as a:
        full = a["standard_targets"]
        native = a["standard_predictions"]
    require(
        np.array_equal(full, np.load(plan["historical_targets"], allow_pickle=False))
        and np.array_equal(native, np.load(plan["historical_predictions"], allow_pickle=False)),
        "Native output identity differs",
    )
    with np.load(plan["scene_arrays"], allow_pickle=False) as a:
        coords, y = a["coordinates"], a["projected_targets"]
        require(a["episodes"].tolist() == eps, "Scene order differs")
    with np.load(plan["aligned_arrays"], allow_pickle=False) as a:
        oracle = a["event_oracle_targets"]
    ag = {key: [] for key in ("left_joint", "right_joint", "left_gripper", "right_gripper")}
    for i, d in enumerate(meta["dimensions"]):
        ag[d["name"].split("_")[0] + ("_joint" if d["unit"] == "radian" else "_gripper")].append(i)
    onset = np.stack(
        [first_upcross(y[:, :, ag[arm + "_gripper"][0]], 0.5) for arm in ("left", "right")], axis=1
    ).astype(float)
    combined = np.concatenate((coords, onset, full.reshape(45, -1), oracle.reshape(45, -1)), axis=1)
    groups = {"socket_px": [0, 1], "peg_px": [2, 3], "left_onset": [4], "right_onset": [5]}
    slices = {
        "full": (6, 6 + full.shape[1] * full.shape[2]),
        "oracle": (6 + full.shape[1] * full.shape[2], combined.shape[1]),
    }
    for label, arr in (("full", full), ("oracle", oracle)):
        lo, _ = slices[label]
        for g, cols in ag.items():
            groups[label + "_" + g] = [
                lo + t * len(meta["dimensions"]) + c for t in range(arr.shape[1]) for c in cols
            ]
    lower = np.array([d["minimum"] for d in meta["dimensions"]])
    upper = np.array([d["maximum"] for d in meta["dimensions"]])
    fk = CommandKinematics(meta["dimensions"])
    arrays, results = {}, {}
    for label, x in feature.items():
        require(time.monotonic() - start < plan["maximum_seconds"], "Deadline exceeded")
        pred, mean, median, folds = multi_crossfit(
            x, combined, groups, plan["alphas"], plan["seed"]
        )
        record = {"folds": folds, "groups": {}, "pose": {}, "first_action": {}, "clip_elements": {}}
        arrays[label + "_raw"] = pred.copy()
        for coord, target in (("full", full), ("oracle", oracle)):
            lo, hi = slices[coord]
            raw = pred[:, lo:hi].reshape(target.shape).copy()
            clipped = np.clip(raw, lower, upper)
            pred[:, lo:hi] = clipped.reshape(45, -1)
            record["clip_elements"][coord] = int(np.count_nonzero(raw != clipped))
            record["pose"][coord] = {}
            for method, p in (("probe", pred), ("mean", mean), ("median", median)):
                pose, _ = pose_metrics(p[:, lo:hi].reshape(target.shape), fk.poses(target), fk, eps)
                record["pose"][coord][method] = pose
                if coord == "full":
                    for g, cols in ag.items():
                        record["first_action"].setdefault(g, {})[method] = describe(
                            np.abs(
                                p[:, lo:hi].reshape(target.shape)[:, :1, cols] - target[:, :1, cols]
                            ).mean(-1),
                            eps,
                        )
        for g, cols in groups.items():
            record["groups"][g] = {
                method: describe(np.abs(p[:, cols] - combined[:, cols]), eps)
                for method, p in (("probe", pred), ("mean", mean), ("median", median))
            }
        for i, name in enumerate(("socket_x", "socket_y", "peg_x", "peg_y")):
            record["groups"][name] = {
                method: describe(np.abs(p[:, i : i + 1] - combined[:, i : i + 1]), eps)
                for method, p in (("probe", pred), ("mean", mean), ("median", median))
            }
        arrays[label + "_predictions"] = pred
        results[label] = record
        print(json.dumps({"completed_representation": label}), flush=True)
    output = Path(plan["output"])
    require(output.is_dir() and not any(output.iterdir()), "Fresh output required")
    with (output / "arrays.npz").open("xb") as f:
        np.savez_compressed(f, **arrays)
    with np.load(output / "arrays.npz", allow_pickle=False) as a:
        require(all(np.array_equal(a[k], v) for k, v in arrays.items()), "Array reload differs")
    seconds = time.monotonic() - start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    require(seconds < plan["maximum_seconds"] and rss < 2 * 1024**3, "Resource budget exceeded")
    for name, sha in plan["sha256"].items():
        require(file_hash(name) == sha, f"Post-run drift: {name}")
    result = {
        "id": plan["id"],
        "status": "completed",
        "plan_sha256": file_hash(plan_path),
        "array_sha256": file_hash(output / "arrays.npz"),
        "metrics": results,
        "seconds": seconds,
        "peak_rss_bytes": rss,
        "array_reload_exact": True,
        "policy_forwards": 0,
        "policy_optimizer_steps": 0,
        "raw_sample_rows": 0,
        "hidden_loaded": False,
        "unique_root_cause_established": False,
        "m2_complete": False,
    }
    with (output / "result.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k != "metrics"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    run(read_json(args.plan), args.plan)
