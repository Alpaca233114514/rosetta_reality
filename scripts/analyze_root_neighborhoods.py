"""Offline, episode-disjoint neighborhood diagnosis from sealed saved arrays."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def seal(output):
    write(
        output / "manifest.json",
        {
            "files": {
                p.relative_to(output).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size}
                for p in sorted(output.rglob("*"))
                if p.is_file()
            }
        },
    )


def load_plan(path):
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if plan["model_calls"] != 0 or plan["optimizer_steps"] != 0:
        raise ValueError("Saved-array analysis only")
    train, dev, hidden = map(set, (plan["train"], plan["development"], plan["hidden"]))
    if len(train) != 40 or len(dev) != 5 or train & dev or (train | dev) & hidden:
        raise ValueError("Invalid split")
    for name, expected in plan["files"].items():
        member = (ROOT / name).resolve()
        if not member.is_relative_to(ROOT) or sha(member) != expected:
            raise ValueError("Source identity drift: " + name)
    return plan


def neighbors(features, episodes, train, *, standardize=False):
    x = np.asarray(features, dtype=np.float64)
    episodes = np.asarray(episodes)
    if len(set(episodes.tolist())) != len(episodes) or not np.isfinite(x).all():
        raise ValueError("Unique episodes and finite inputs required")
    donor_rows = np.flatnonzero(np.isin(episodes, train))
    if len(donor_rows) != len(train) or len(train) < 4:
        raise ValueError("Missing training donors")
    out, distances, scales = [], [], []
    for i, ep in enumerate(episodes):
        allowed = donor_rows[episodes[donor_rows] != ep]
        # Exclude the query episode from scaling as well as neighbor selection.
        scale = x[allowed].std(axis=0) if standardize else np.ones(x.shape[1])
        scale = np.where(scale > 1e-12, scale, 1.0)
        d = np.linalg.norm((x - x[i]) / scale, axis=1)
        selected = sorted(allowed.tolist(), key=lambda j: (d[j], int(episodes[j])))[:3]
        out.append(selected)
        distances.append(d[selected].tolist())
        scales.append(scale.tolist())
    return np.asarray(out), distances, scales


def crossing(trace, threshold):
    x = np.asarray(trace)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("Finite nonempty trace required")
    if x[0] <= threshold:
        return {"status": "already_below_at_start", "slot": None}
    hits = np.flatnonzero((x[:-1] > threshold) & (x[1:] <= threshold))
    if not len(hits):
        return {"status": "no_crossing_in_window", "slot": None}
    i = int(hits[0])
    return {"status": "observed", "slot": i + float((x[i] - threshold) / (x[i] - x[i + 1]))}


def run(plan, output, *, resource_evidence=None):
    def arrays(name):
        with np.load(ROOT / plan[name], allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    data, geometry, paired = arrays("data"), arrays("geometry"), arrays("paired")
    episodes = plan["train"] + plan["development"]
    ids = paired["identities"]
    if set(map(tuple, ids.tolist())) != {(e, f) for e in episodes for f in (0, 125, 250, 375)}:
        raise ValueError("Prediction cohort changed")
    if len(ids) != 180 or not paired["valid_mask"].all():
        raise ValueError("Expected complete unpadded historical grid")
    if geometry["episodes"].tolist() != episodes:
        raise ValueError("Geometry episode order differs")
    if set(map(tuple, data["identities"].tolist())) != {
        (e, f) for e in episodes for f in range(500)
    }:
        raise ValueError("Data contains missing or unexpected episodes/frames")
    sample_lookup = {tuple(v): i for i, v in enumerate(data["sample_identities"])}
    raw_lookup = {tuple(v): i for i, v in enumerate(data["identities"])}
    for i, pair in enumerate(ids):
        np.testing.assert_array_equal(
            paired["targets"][i], data["sample_targets"][sample_lookup[tuple(pair)]]
        )
    for key in ("early_correct", "late_correct"):
        if paired[key].shape != (4, 180, 50, 14) or not np.isfinite(paired[key]).all():
            raise ValueError("Prediction shape or finiteness drift")
    train_actions = data["projected_actions"][np.isin(data["identities"][:, 0], plan["train"])]
    for value in (train_actions, paired["targets"], data["states"], geometry["coordinates"]):
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite target or neighbor feature")
    thresholds = {}
    for side, dim in (("left_gripper", 6), ("right_gripper", 13)):
        low, high = np.quantile(train_actions[:, dim], [0.1, 0.9])
        thresholds[side] = (low + np.array([0.25, 0.5, 0.75]) * (high - low)).tolist()
    rows, events, neighborhood = [], [], []
    groups = plan["groups"]
    for frame in (0, 125, 250):
        ix = np.asarray(
            [np.flatnonzero((ids[:, 0] == ep) & (ids[:, 1] == frame))[0] for ep in episodes]
        )
        y = paired["targets"][ix]
        states = np.stack([data["states"][raw_lookup[ep, frame]] for ep in episodes])
        for mode, features, standardize in (
            ("initial_geometry", geometry["coordinates"] / [640, 480, 640, 480], False),
            ("current_state", states, True),
        ):
            near, distances, scales = neighbors(
                features, episodes, plan["train"], standardize=standardize
            )
            for i, ep in enumerate(episodes):
                donors = near[i]
                allowed = [j for j, e in enumerate(episodes) if e in plan["train"] and e != ep]
                neighbor_prediction = y[donors].mean(axis=0)
                mean, median = y[allowed].mean(axis=0), np.median(y[allowed], axis=0)
                common = {
                    "episode": ep,
                    "frame": frame,
                    "mode": mode,
                    "split": "train_loo" if ep in plan["train"] else "development",
                }
                neighborhood.append(
                    {
                        **common,
                        "donors": [episodes[j] for j in donors],
                        "distances": distances[i],
                        "scale": scales[i],
                    }
                )
                for group, dims in groups.items():
                    for window, slots in (("first", slice(0, 1)), ("full", slice(None))):
                        target = y[i, slots][:, dims]
                        selected = y[donors, slots][:, :, dims]
                        base = {
                            **common,
                            "group": group,
                            "window": window,
                            "neighbor_mae": float(
                                np.abs(neighbor_prediction[slots][:, dims] - target).mean()
                            ),
                            "neighbor_label_std": float(selected.std(axis=0).mean()),
                            "mean_mae": float(np.abs(mean[slots][:, dims] - target).mean()),
                            "median_mae": float(np.abs(median[slots][:, dims] - target).mean()),
                        }
                        for endpoint, key in ((2500, "early_correct"), (5000, "late_correct")):
                            for noise in range(4):
                                pred = paired[key][noise, ix[i], slots][:, dims]
                                error = pred - target
                                rows.append(
                                    {
                                        **base,
                                        "checkpoint": endpoint,
                                        "noise": noise,
                                        "model_mae": float(np.abs(error).mean()),
                                        "model_mse": float(np.square(error).mean()),
                                        "mean_bias": float(error.mean()),
                                        "within_chunk_bias_mse": float(
                                            np.square(error.mean(axis=0)).mean()
                                        ),
                                    }
                                )
                if frame in (125, 250):
                    for side, dim in (("left_gripper", 6), ("right_gripper", 13)):
                        for threshold in thresholds[side]:
                            target_event = crossing(y[i, :, dim], threshold)
                            donor_events = [crossing(y[j, :, dim], threshold) for j in donors]
                            observed_slots = [
                                v["slot"] for v in donor_events if v["slot"] is not None
                            ]
                            for endpoint, key in ((2500, "early_correct"), (5000, "late_correct")):
                                for noise in range(4):
                                    prediction = crossing(
                                        paired[key][noise, ix[i], :, dim], threshold
                                    )
                                    events.append(
                                        {
                                            **common,
                                            "side": side,
                                            "threshold": threshold,
                                            "checkpoint": endpoint,
                                            "noise": noise,
                                            "target": target_event,
                                            "prediction": prediction,
                                            "neighbors": donor_events,
                                            "neighbor_observed_count": len(observed_slots),
                                            "neighbor_observed_timing_range": max(observed_slots)
                                            - min(observed_slots)
                                            if observed_slots
                                            else None,
                                            "neighbor_median_timing_error": float(
                                                np.median(observed_slots)
                                            )
                                            - target_event["slot"]
                                            if observed_slots and target_event["slot"] is not None
                                            else None,
                                            "neighbor_status_disagreement": sum(
                                                v["status"] != target_event["status"]
                                                for v in donor_events
                                            )
                                            / 3,
                                            "model_timing_error": prediction["slot"]
                                            - target_event["slot"]
                                            if prediction["slot"] is not None
                                            and target_event["slot"] is not None
                                            else None,
                                        }
                                    )
    output.mkdir(parents=True, exist_ok=False)
    with (output / "metrics.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name, values in (("events.jsonl", events), ("neighbors.jsonl", neighborhood)):
        with (output / name).open("x", encoding="utf-8") as stream:
            for value in values:
                stream.write(json.dumps(value, allow_nan=False) + "\n")
    summary = []
    for frame in (0, 125, 250):
        for mode in ("initial_geometry", "current_state"):
            for group in groups:
                selected = [
                    r
                    for r in rows
                    if r["frame"] == frame
                    and r["mode"] == mode
                    and r["group"] == group
                    and r["window"] == "first"
                    and r["checkpoint"] == 5000
                    and r["noise"] == 0
                    and r["split"] == "development"
                ]
                summary.append(
                    {
                        "frame": frame,
                        "mode": mode,
                        "group": group,
                        **{
                            k: float(np.mean([r[k] for r in selected]))
                            for k in (
                                "model_mae",
                                "neighbor_mae",
                                "mean_mae",
                                "median_mae",
                                "neighbor_label_std",
                            )
                        },
                    }
                )
    write(
        output / "result.json",
        {
            "status": "completed_exploratory",
            "rows": len(rows),
            "episodes": episodes,
            "thresholds": thresholds,
            "summary_zero_noise_first": summary,
            "model_calls": 0,
            "optimizer_steps": 0,
            "hidden_rows_loaded": 0,
            "limitations": [
                "Initial geometry is only a 2D proxy, not current image similarity.",
                "Neighbor dispersion cannot establish label errors or irreducible ambiguity.",
                "Four noise conditions and chunk slots are not independent scenes.",
                "Command crossings are not physical contact times; no Gate change.",
            ],
            "inputs": plan["files"],
            "resource_evidence": resource_evidence,
        },
    )
    seal(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or not Path("/.dockerenv").exists():
        raise RuntimeError("Run through WSL into Linux Docker")
    from root_analysis_resources import verify_resource_envelope

    plan = load_plan(args.plan)
    run(plan, args.output, resource_evidence=verify_resource_envelope(plan))


if __name__ == "__main__":
    main()
