"""Independent NumPy arithmetic replay; does not import production analysis helpers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def event(series, threshold):
    if series[0] <= threshold:
        return "already_below_at_start", None
    for slot in range(1, len(series)):
        if series[slot] <= threshold < series[slot - 1]:
            return "observed", slot - 1 + float(
                (series[slot - 1] - threshold) / (series[slot - 1] - series[slot])
            )
    return "no_crossing_in_window", None


def verify(plan_path, source, output):
    plan = json.loads(plan_path.read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    if {p.name for p in source.iterdir() if p.is_file()} != set(manifest["files"]) | {
        "manifest.json"
    }:
        raise ValueError("Analysis inventory differs")
    for name, item in manifest["files"].items():
        path = source / name
        if (
            path.stat().st_size != item["bytes"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]
        ):
            raise ValueError("Analysis seal changed")
    for name, expected in plan["files"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Plan source changed")
    with np.load(ROOT / plan["paired"], allow_pickle=False) as z:
        pair = dict(z)
    with np.load(ROOT / plan["data"], allow_pickle=False) as z:
        data = dict(z)
    with np.load(ROOT / plan["geometry"], allow_pickle=False) as z:
        geo = dict(z)
    episodes = plan["train"] + plan["development"]
    index = {tuple(key): i for i, key in enumerate(pair["identities"])}
    numeric_index = {tuple(key): i for i, key in enumerate(data["identities"])}
    donors_by_key = {}
    checks = 0
    for line in (source / "neighbors.jsonl").read_text().splitlines():
        row = json.loads(line)
        ep, frame = row["episode"], row["frame"]
        allowed = [e for e in plan["train"] if e != ep]
        x = (
            geo["coordinates"] / np.array([640, 480, 640, 480])
            if row["mode"] == "initial_geometry"
            else np.array([data["states"][numeric_index[e, frame]] for e in episodes])
        )
        allowed_ix = [episodes.index(e) for e in allowed]
        scale = (
            np.std(x[allowed_ix], axis=0) if row["mode"] == "current_state" else np.ones(x.shape[1])
        )
        scale[scale <= 1e-12] = 1
        distances = [
            (
                float(
                    np.sqrt(np.sum(((x[episodes.index(e)] - x[episodes.index(ep)]) / scale) ** 2))
                ),
                e,
            )
            for e in allowed
        ]
        nearest = sorted(distances)[:3]
        if row["donors"] != [e for _, e in nearest]:
            raise ValueError("Donor identity differs")
        np.testing.assert_allclose(
            row["distances"], [d for d, _ in nearest], rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(row["scale"], scale, rtol=1e-12, atol=1e-12)
        donors_by_key[ep, frame, row["mode"]] = row["donors"]
        checks += 1
    rows = list(csv.DictReader((source / "metrics.csv").open()))
    for row in rows:
        ep, frame, noise, step = (int(row[k]) for k in ("episode", "frame", "noise", "checkpoint"))
        slots = [0] if row["window"] == "first" else list(range(50))
        dims = plan["groups"][row["group"]]
        ix = index[ep, frame]
        y = pair["targets"][ix][np.ix_(slots, dims)]
        p = pair["early_correct" if step == 2500 else "late_correct"][noise, ix][
            np.ix_(slots, dims)
        ]
        donor_ids = [index[e, frame] for e in donors_by_key[ep, frame, row["mode"]]]
        allowed = [index[e, frame] for e in plan["train"] if e != ep]
        near = pair["targets"][donor_ids][:, slots][:, :, dims]
        all_y = pair["targets"][allowed][:, slots][:, :, dims]
        expected = {
            "model_mae": np.abs(p - y).mean(),
            "model_mse": ((p - y) ** 2).mean(),
            "mean_bias": (p - y).mean(),
            "within_chunk_bias_mse": ((p - y).mean(axis=0) ** 2).mean(),
            "neighbor_mae": np.abs(near.mean(axis=0) - y).mean(),
            "neighbor_label_std": near.std(axis=0).mean(),
            "mean_mae": np.abs(all_y.mean(axis=0) - y).mean(),
            "median_mae": np.abs(np.median(all_y, axis=0) - y).mean(),
        }
        for name, value in expected.items():
            np.testing.assert_allclose(float(row[name]), value, atol=1e-12, rtol=1e-10)
            checks += 1
    event_checks = 0
    for line in (source / "events.jsonl").read_text().splitlines():
        row = json.loads(line)
        ep, frame, noise = (row[k] for k in ("episode", "frame", "noise"))
        ix, dim = index[ep, frame], 6 if row["side"] == "left_gripper" else 13
        series = [
            pair["targets"][ix, :, dim],
            pair["early_correct" if row["checkpoint"] == 2500 else "late_correct"][
                noise, ix, :, dim
            ],
        ]
        for recorded, values in zip((row["target"], row["prediction"]), series, strict=True):
            status, slot = event(values, row["threshold"])
            if status != recorded["status"] or (slot is None) != (recorded["slot"] is None):
                raise ValueError("Event censoring differs")
            if slot is not None:
                np.testing.assert_allclose(slot, recorded["slot"], rtol=1e-12, atol=1e-12)
            event_checks += 1
    # Reconcile historical headline using untouched predictions, not CSV summaries.
    hist = []
    for noise in range(4):
        dev_ix = [index[e, 125] for e in plan["development"]]
        mse = (
            (pair["late_correct"][noise, dev_ix, :, 6] - pair["targets"][dev_ix, :, 6]) ** 2
        ).mean(axis=1)
        share = float(sum(mse[plan["development"].index(e)] for e in (13, 33)) / sum(mse))
        if not 0.885 <= share <= 0.931:
            raise ValueError("Historical 13/33 error share does not reproduce")
        hist.append(share)
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "passed",
        "metric_and_neighbor_checks": checks,
        "event_checks": event_checks,
        "historical_13_33_mse_share": hist,
        "production_helpers_imported": False,
        "input_manifest_sha256": hashlib.sha256(
            (source / "manifest.json").read_bytes()
        ).hexdigest(),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.plan, args.input, args.output)
