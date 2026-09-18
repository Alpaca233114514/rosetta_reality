"""Separate train-selected shifts from label-oracle timing error bounds."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from rosetta_reality.vla.visual_fit import file_hash, read_bundle
from scripts.diagnose_chunk_time_shift import analyze, shift_chunk


def oracle_align(prediction, target, lags):
    """Uses each row's target deliberately; this is NOT a deployable predictor."""
    if prediction.shape != target.shape or prediction.ndim != 3 or 0 not in lags:
        raise ValueError("Aligned chunks and a zero-shift control required")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Finite arrays required")
    candidates = np.stack([shift_chunk(prediction, lag) for lag in lags])
    scores = np.abs(candidates - target[None]).mean(axis=(2, 3))
    chosen = np.asarray(
        [
            min(range(len(lags)), key=lambda k: (scores[k, i], abs(lags[k]), lags[k]))
            for i in range(len(target))
        ]
    )
    aligned = candidates[chosen, np.arange(len(target))]
    if (
        np.abs(aligned - target).mean(axis=(1, 2))
        > np.abs(prediction - target).mean(axis=(1, 2)) + 1e-12
    ).any():
        raise ValueError("Oracle cannot be worse than its zero-shift candidate")
    return aligned, np.asarray(lags)[chosen], scores


def metrics(prediction, target, start, stop):
    residual = prediction[:, start:stop] - target[:, start:stop]
    return {
        "mae": float(np.abs(residual).mean()),
        "mse": float(np.square(residual).mean()),
        "per_episode_mae": np.abs(residual).mean(axis=(1, 2)).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8-sig"))
    if (
        os.environ.get("ROSETTA_CONTAINER_IMAGE_ID") != plan["image"]
        or os.environ.get("HF_HUB_OFFLINE") != "1"
    ):
        raise ValueError("Registered offline CPU container required")
    for name, sha in plan["sha256"].items():
        if file_hash(Path(name)) != sha:
            raise ValueError(f"Input/code drift: {name}")
    saved, metadata = read_bundle(Path(plan["candidate_bundle"]))
    if metadata["episodes"] != plan["episodes"] or metadata["arm"] != "C":
        raise ValueError("Candidate/split drift")
    target = saved["standard_targets"]
    groups = {
        name: []
        for name in (
            "joint",
            "gripper",
            "left_joint",
            "right_joint",
            "left_gripper",
            "right_gripper",
        )
    }
    for d, dimension in enumerate(metadata["dimensions"]):
        kind = "joint" if dimension["unit"] == "radian" else "gripper"
        groups[kind].append(d)
        groups[dimension["name"].split("_")[0] + "_" + kind].append(d)
    windows = dict(
        full=(0, 50), first=(0, 1), early=(0, 10), middle=(10, 25), late=(25, 50), last=(49, 50)
    )
    mean = np.broadcast_to(target[:40].mean(0), target.shape).copy()
    median = np.broadcast_to(np.median(target[:40], axis=0), target.shape).copy()
    arrays, records = {}, {}
    for i, seed in enumerate(metadata["noise_conditions"]):
        native = saved["standard_predictions"][i]
        global_shift, loo, selected = analyze(native, target, plan["lags"], 40)
        arrays[f"{seed}_global"] = global_shift
        arrays[f"{seed}_loo"] = loo
        record = {"global_shift": selected, "groups": {}}
        for group, dims in groups.items():
            y = target[..., dims]
            values = {
                "native": native[..., dims],
                "global_shift": global_shift[..., dims],
                "train_mean": mean[..., dims],
                "train_median": median[..., dims],
            }
            oracle_lags = {}
            for name in ("native", "train_mean", "train_median"):
                aligned, lags, scores = oracle_align(values[name], y, plan["lags"])
                values[name + "_oracle"] = aligned
                oracle_lags[name] = lags.tolist()
                arrays[f"{seed}_{group}_{name}_oracle"] = aligned
                arrays[f"{seed}_{group}_{name}_scores"] = scores
            # Group oracle is allowed its own lag and is thus an intentionally
            # generous bound for the shared global-shift hypothesis.
            if (
                np.abs(values["native_oracle"] - y).mean()
                > np.abs(values["global_shift"] - y).mean() + 1e-12
            ):
                raise ValueError("Oracle bound failed to contain global candidate")
            views = {
                view: {
                    window: {
                        name: metrics(p[rows], y[rows], start, stop) for name, p in values.items()
                    }
                    for window, (start, stop) in windows.items()
                }
                for view, rows in (
                    ("train40_descriptive", slice(0, 40)),
                    ("dev5_oracle_not_validation", slice(40, 45)),
                )
            }
            full = views["dev5_oracle_not_validation"]["full"]
            record["groups"][group] = {
                "oracle_lags": oracle_lags,
                "views": views,
                "native_oracle_beats_equally_aligned_constants": full["native_oracle"]["mae"]
                < min(full["train_mean_oracle"]["mae"], full["train_median_oracle"]["mae"]) - 1e-8,
            }
        records[str(seed)] = record
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    with (output / "arrays.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(output / "arrays.npz", allow_pickle=False) as restored:
        if not all(np.array_equal(a, restored[key]) for key, a in arrays.items()):
            raise ValueError("Reload mismatch")
    result = {
        "id": plan["id"],
        "oracle_uses_target_labels": True,
        "deployable_repair": False,
        "new_model_forwards": 0,
        "optimizer_steps": 0,
        "hidden_loaded": False,
        "records": records,
        "array_sha256": file_hash(output / "arrays.npz"),
        "array_reload_exact": True,
    }
    with (output / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                seed: {
                    "global_lag": r["global_shift"]["lag"],
                    "oracle_beats_aligned_constants": {
                        g: value["native_oracle_beats_equally_aligned_constants"]
                        for g, value in r["groups"].items()
                    },
                }
                for seed, r in records.items()
            }
        )
    )


if __name__ == "__main__":
    main()
