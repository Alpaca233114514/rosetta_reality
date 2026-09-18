"""Exploratory CPU-only replay of sealed predictions; never imports a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    loo = (x.sum() - x) / (len(x) - 1)
    return dict(
        mean=float(x.mean()),
        minimum=float(x.min()),
        maximum=float(x.max()),
        leave_one_episode_out=[float(loo.min()), float(loo.max())],
    )


def crossing(trace, threshold):
    """First downward crossing, retaining both forms of censoring."""
    if trace[0] <= threshold:
        return dict(status="already_below_at_start", slot=None)
    hits = np.flatnonzero((trace[:-1] > threshold) & (trace[1:] <= threshold))
    if not len(hits):
        return dict(status="no_crossing_in_window", slot=None)
    i = int(hits[0])
    slot = i + float((trace[i] - threshold) / (trace[i] - trace[i + 1]))
    return dict(status="observed", slot=slot)


def shape(trace):
    closing = np.maximum(-np.diff(trace), 0)
    total = float(closing.sum())
    return dict(
        start=float(trace[0]),
        end=float(trace[-1]),
        minimum=float(trace.min()),
        maximum=float(trace.max()),
        total_closing_variation=total,
        closing_center_slot=(
            float(closing @ (np.arange(len(closing)) + 0.5) / total) if total > 1e-8 else None
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    assert platform.system() == "Linux" and Path("/.dockerenv").exists()
    assert plan["status"] == "exploratory_frozen_before_replay"
    for name, expected in plan["sha256"].items():
        assert digest(ROOT / name) == expected, name
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(ROOT / plan["collection"], allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    with np.load(ROOT / plan["data"], allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    p, w, y = [a[k].astype(np.float64) for k in ("correct", "wrong", "targets")]
    ids = a["identities"]
    train = np.isin(ids[:, 0], plan["train_episodes"])
    assert p.shape == w.shape == (4, 180, 50, 14) and a["valid_mask"].all()
    assert len(set(map(tuple, ids))) == 180 and train.sum() == 160
    for x in (p, w, y):
        assert np.isfinite(x).all()
    data_rows = {tuple(pair): i for i, pair in enumerate(d["sample_identities"])}
    ix = [data_rows[tuple(pair)] for pair in ids]
    np.testing.assert_array_equal(y, d["sample_targets"][ix])
    raw_state = d["sample_states"][ix]
    baselines = np.empty((2, 180, 50, 14), dtype=np.float64)
    debiased = np.empty_like(p)
    for row, (episode, frame) in enumerate(ids):
        donors = np.flatnonzero(train & (ids[:, 1] == frame) & (ids[:, 0] != episode))
        baselines[0, row] = y[donors].mean(axis=0)
        baselines[1, row] = np.median(y[donors], axis=0)
        debiased[:, row] = p[:, row] - (p[:, donors] - y[None, donors]).mean(axis=1)
    train_actions = d["projected_actions"][
        np.isin(d["identities"][:, 0], plan["train_episodes"]), 6
    ]
    low, high = np.quantile(train_actions, [0.1, 0.9])
    thresholds = low + np.asarray(plan["threshold_fractions"]) * (high - low)
    records, first, summaries, joint_summary = [], [], [], []
    for noise in range(4):
        for row, (episode, frame) in enumerate(ids):
            split = "train_loo" if train[row] else "development"
            common = dict(
                episode=int(episode),
                frame=int(frame),
                sample_row=row,
                noise_index=noise,
                split=split,
            )
            if frame in plan["gripper_frames"]:
                q, wrong, target = p[noise, row, :, 6], w[noise, row, :, 6], y[row, :, 6]
                error = q - target
                rec = dict(
                    **common,
                    correct_mae=float(abs(error).mean()),
                    correct_mse=float(np.mean(error**2)),
                    signed_bias=float(error.mean()),
                    temporal_bias_squared=float(error.mean() ** 2),
                    temporal_residual_variance=float(error.var()),
                    image_gain_mse=float(np.mean((wrong - target) ** 2 - error**2)),
                    mean_baseline_mae=float(abs(baselines[0, row, :, 6] - target).mean()),
                    median_baseline_mae=float(abs(baselines[1, row, :, 6] - target).mean()),
                    correct_shape=shape(q),
                    target_shape=shape(target),
                    wrong_shape=shape(wrong),
                )
                rec["segments"] = []
                for lo, hi in plan["segments"]:
                    e = error[lo:hi]
                    rec["segments"].append(
                        dict(
                            start=lo,
                            stop=hi,
                            mae=float(abs(e).mean()),
                            contribution_to_full_mse=float(np.sum(e**2) / 50),
                            image_gain_mse=float(
                                np.mean((wrong[lo:hi] - target[lo:hi]) ** 2 - e**2)
                            ),
                        )
                    )
                rec["crossings"] = [
                    dict(
                        threshold=float(t),
                        correct=crossing(q, t),
                        target=crossing(target, t),
                        wrong=crossing(wrong, t),
                    )
                    for t in thresholds
                ]
                records.append(rec)
            if frame == 0:
                for dim in plan["joint_dimensions"]:
                    target, pred = float(y[row, 0, dim]), float(p[noise, row, 0, dim])
                    first.append(
                        dict(
                            **common,
                            dimension=dim,
                            name=plan["dimension_names"][dim],
                            target=target,
                            prediction=pred,
                            state=float(raw_state[row, dim]),
                            signed_error=pred - target,
                            absolute_error=abs(pred - target),
                            mean_baseline=float(baselines[0, row, 0, dim]),
                            median_baseline=float(baselines[1, row, 0, dim]),
                            train_bias_adjusted_prediction=float(debiased[noise, row, 0, dim]),
                        )
                    )
        for frame in plan["gripper_frames"]:
            for split in ("train_loo", "development"):
                rows = [
                    r
                    for r in records
                    if r["noise_index"] == noise and r["frame"] == frame and r["split"] == split
                ]
                summaries.append(
                    dict(
                        noise_index=noise,
                        frame=frame,
                        split=split,
                        episodes=len(rows),
                        metrics={
                            k: stats([r[k] for r in rows])
                            for k in (
                                "correct_mae",
                                "correct_mse",
                                "signed_bias",
                                "image_gain_mse",
                                "temporal_bias_squared",
                                "temporal_residual_variance",
                                "mean_baseline_mae",
                                "median_baseline_mae",
                            )
                        },
                    )
                )
        for split, mask in (("train_loo", train), ("development", ~train)):
            rows = np.flatnonzero(mask & (ids[:, 1] == 0))
            for label, dims in (
                ("left_joints", list(range(6))),
                ("right_joints", list(range(7, 13))),
            ):
                pred, target = p[noise, rows, 0][:, dims], y[rows, 0][:, dims]
                error = pred - target
                joint_summary.append(
                    dict(
                        noise_index=noise,
                        split=split,
                        group=label,
                        correct_mae=stats(abs(error).mean(axis=1)),
                        mean_baseline_mae=stats(
                            abs(baselines[0, rows, 0][:, dims] - target).mean(axis=1)
                        ),
                        median_baseline_mae=stats(
                            abs(baselines[1, rows, 0][:, dims] - target).mean(axis=1)
                        ),
                        state_hold_mae=stats(abs(raw_state[rows][:, dims] - target).mean(axis=1)),
                        train_bias_adjusted_mae=stats(
                            abs(debiased[noise, rows, 0][:, dims] - target).mean(axis=1)
                        ),
                        residual_mean_bias_squared=float(np.mean(error.mean(axis=0) ** 2)),
                        residual_scene_variance=float(np.mean(error.var(axis=0))),
                        mse=float(np.mean(error**2)),
                    )
                )
    for name, rows in (("gripper-samples.jsonl", records), ("first-action-joints.jsonl", first)):
        with (args.output / name).open("x") as f:
            for row in rows:
                f.write(json.dumps(row, allow_nan=False) + "\n")
    np.savez_compressed(
        args.output / "arrays.npz",
        identities=ids,
        correct=p,
        wrong=w,
        targets=y,
        raw_targets=a["raw_targets"],
        valid_mask=a["valid_mask"],
        noise=a["noise"],
        raw_state=raw_state,
        train_mask=train,
        baselines=baselines,
        train_bias_adjusted=debiased,
        thresholds=thresholds,
    )
    write(
        args.output / "result.json",
        dict(
            status="completed_exploratory",
            model_calls=0,
            optimizer_steps=0,
            plan_sha256=digest(args.plan),
            training_threshold_anchors=[float(low), float(high)],
            thresholds=thresholds.tolist(),
            gripper_summary=summaries,
            first_action_summary=joint_summary,
            gripper_rows=len(records),
            first_action_joint_rows=len(first),
            runtime=dict(
                python=platform.python_version(),
                numpy=np.__version__,
                image=os.environ.get("ROSETTA_CONTAINER_IMAGE_ID"),
                gpu_used=False,
                network="none",
            ),
            limits=plan["limits"],
        ),
    )
    write(
        args.output / "manifest.json",
        dict(
            plan_sha256=digest(args.plan),
            files={
                f.name: dict(bytes=f.stat().st_size, sha256=digest(f))
                for f in sorted(args.output.iterdir())
                if f.is_file()
            },
        ),
    )
    print(json.dumps(dict(status="completed", gripper_rows=len(records), joint_rows=len(first))))


if __name__ == "__main__":
    main()
