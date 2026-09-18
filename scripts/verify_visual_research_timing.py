"""Independent scalar replay and evidence tables for the saved-array exploration."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    root = args.input
    for name, entry in json.loads((root / "manifest.json").read_text())["files"].items():
        path = root / name
        assert path.stat().st_size == entry["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    with np.load(root / "arrays.npz", allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    p, w, y = [a[k] for k in ("correct", "wrong", "targets")]
    ids, train = a["identities"], a["train_mask"]
    records = [
        json.loads(line) for line in (root / "gripper-samples.jsonl").read_text().splitlines()
    ]
    first = [
        json.loads(line) for line in (root / "first-action-joints.jsonl").read_text().splitlines()
    ]
    summary = json.loads((root / "result.json").read_text())
    prior = json.loads(args.prior.read_text())["records"]
    checks = 0

    def eq(x, target):
        nonlocal checks
        np.testing.assert_allclose(x, target, atol=2e-12, rtol=2e-10)
        checks += int(np.size(x))

    # Recompute donor pools without producer helpers; assert no development fitting.
    for j, (ep, frame) in enumerate(ids):
        pool = [k for k in range(len(ids)) if train[k] and ids[k, 1] == frame and ids[k, 0] != ep]
        eq(a["baselines"][0, j], sum(y[k] for k in pool) / len(pool))
        eq(a["baselines"][1, j], np.median([y[k] for k in pool], axis=0))
        for n in range(4):
            adjustment = sum((p[n, k] - y[k]) for k in pool) / len(pool)
            eq(a["train_bias_adjusted"][n, j], p[n, j] - adjustment)
    event_checks = 0
    for r in records:
        n, j = r["noise_index"], r["sample_row"]
        assert list(ids[j]) == [r["episode"], r["frame"]]
        errors = [float(p[n, j, slot, 6] - y[j, slot, 6]) for slot in range(50)]
        eq(r["correct_mae"], sum(map(abs, errors)) / 50)
        eq(r["correct_mse"], sum(e * e for e in errors) / 50)
        eq(r["signed_bias"], sum(errors) / 50)
        eq(r["temporal_bias_squared"] + r["temporal_residual_variance"], r["correct_mse"])
        eq(
            r["image_gain_mse"],
            sum(float((w[n, j, s, 6] - y[j, s, 6]) ** 2) - errors[s] ** 2 for s in range(50)) / 50,
        )
        eq(sum(s["contribution_to_full_mse"] for s in r["segments"]), r["correct_mse"])
        for segment in r["segments"]:
            lo, hi = segment["start"], segment["stop"]
            eq(segment["mae"], sum(abs(e) for e in errors[lo:hi]) / (hi - lo))
        for event in r["crossings"]:
            for label, trace in (
                ("correct", p[n, j, :, 6]),
                ("target", y[j, :, 6]),
                ("wrong", w[n, j, :, 6]),
            ):
                t, observed = event["threshold"], event[label]
                if trace[0] <= t:
                    assert observed == {"status": "already_below_at_start", "slot": None}
                else:
                    hits = [s for s in range(1, 50) if trace[s - 1] > t >= trace[s]]
                    if hits:
                        s = hits[0]
                        expected = s - (t - trace[s]) / (trace[s - 1] - trace[s])
                        assert observed["status"] == "observed"
                        eq(observed["slot"], expected)
                    else:
                        assert observed == {"status": "no_crossing_in_window", "slot": None}
                event_checks += 1
    for r in first:
        n, j, dim = r["noise_index"], r["sample_row"], r["dimension"]
        eq(r["prediction"], p[n, j, 0, dim])
        eq(r["target"], y[j, 0, dim])
        eq(r["signed_error"], p[n, j, 0, dim] - y[j, 0, dim])
        eq(r["train_bias_adjusted_prediction"], a["train_bias_adjusted"][n, j, 0, dim])
    reconciled = 0
    for r in summary["gripper_summary"]:
        old = next(
            x
            for x in prior
            if x["noise_index"] == r["noise_index"]
            and x["frame"] == r["frame"]
            and x["split"] == r["split"]
            and x["window"] == "full_chunk"
            and x["group"] == "left_gripper"
        )
        for key in (
            "correct_mae",
            "correct_mse",
            "image_gain_mse",
            "mean_baseline_mae",
            "median_baseline_mae",
        ):
            old_key = {
                "mean_baseline_mae": "train_mean_mae",
                "median_baseline_mae": "train_median_mae",
            }.get(key, key)
            # Prior result contract is inspected before execution; use its episode values.
            eq(r["metrics"][key]["mean"], np.mean(list(old[old_key]["per_episode"].values())))
            reconciled += 1
    for r in summary["first_action_summary"]:
        n = r["noise_index"]
        mask = train if r["split"] == "train_loo" else ~train
        rows = np.flatnonzero(mask & (ids[:, 1] == 0))
        dims = list(range(6)) if r["group"] == "left_joints" else list(range(7, 13))
        target = y[rows, 0][:, dims]
        for key, pred in (
            ("correct_mae", p[n, rows, 0][:, dims]),
            ("mean_baseline_mae", a["baselines"][0, rows, 0][:, dims]),
            ("median_baseline_mae", a["baselines"][1, rows, 0][:, dims]),
            ("state_hold_mae", a["raw_state"][rows][:, dims]),
            ("train_bias_adjusted_mae", a["train_bias_adjusted"][n, rows, 0][:, dims]),
        ):
            values = abs(pred - target).mean(axis=1)
            eq(r[key]["mean"], values.mean())
            loo = [np.delete(values, j).mean() for j in range(len(values))]
            eq(r[key]["leave_one_episode_out"], [min(loo), max(loo)])
        eq(r["mse"], np.mean((p[n, rows, 0][:, dims] - target) ** 2))
        eq(r["residual_mean_bias_squared"] + r["residual_scene_variance"], r["mse"])
    dev = [r for r in records if r["split"] == "development"]
    tables = []
    for n in range(4):
        for frame in (125, 250):
            rows = [r for r in dev if r["noise_index"] == n and r["frame"] == frame]
            total = sum(r["correct_mse"] for r in rows)
            for r in rows:
                table = {
                    k: r[k]
                    for k in (
                        "noise_index",
                        "frame",
                        "episode",
                        "correct_mae",
                        "correct_mse",
                        "signed_bias",
                        "image_gain_mse",
                    )
                }
                table["share_of_development_squared_error"] = r["correct_mse"] / total
                table["temporal_bias_share"] = r["temporal_bias_squared"] / r["correct_mse"]
                table["target_range"] = r["target_shape"]["maximum"] - r["target_shape"]["minimum"]
                table["first_action_prediction"] = r["correct_shape"]["start"]
                table["first_action_target"] = r["target_shape"]["start"]
                table["crossings"] = []
                for cross in r["crossings"]:
                    measured = cross["target"]["status"] == cross["correct"]["status"] == "observed"
                    table["crossings"].append(
                        dict(
                            threshold=cross["threshold"],
                            target_status=cross["target"]["status"],
                            prediction_status=cross["correct"]["status"],
                            predicted_minus_target_slots=(
                                cross["correct"]["slot"] - cross["target"]["slot"]
                                if measured
                                else None
                            ),
                        )
                    )
                tables.append(table)
    dim_table, gains = [], []
    for n in range(4):
        rows = np.flatnonzero((~train) & (ids[:, 1] == 0))
        for dim in list(range(6)) + list(range(7, 13)):
            e = p[n, rows, 0, dim] - y[rows, 0, dim]
            dim_table.append(
                dict(
                    noise_index=n,
                    dimension=dim,
                    signed_mean_error=float(e.mean()),
                    mse=float(np.mean(e**2)),
                    mae=float(np.mean(abs(e))),
                    target_mean=float(y[rows, 0, dim].mean()),
                    prediction_mean=float(p[n, rows, 0, dim].mean()),
                )
            )
        for group, dims in (("left_joints", list(range(6))), ("right_joints", list(range(7, 13)))):
            e = abs(p[n, rows, 0][:, dims] - y[rows, 0][:, dims]).mean(axis=1)
            corrected = abs(
                a["train_bias_adjusted"][n, rows, 0][:, dims] - y[rows, 0][:, dims]
            ).mean(axis=1)
            benefit = e - corrected
            loo = [(sum(benefit) - b) / (len(benefit) - 1) for b in benefit]
            gains.append(
                dict(
                    noise_index=n,
                    group=group,
                    mae_reduction=float(benefit.mean()),
                    fractional_mae_reduction=float(benefit.mean() / e.mean()),
                    leave_one_episode_out=[float(min(loo)), float(max(loo))],
                    per_episode_benefit=[
                        dict(episode=int(ids[j, 0]), benefit=float(b))
                        for j, b in zip(rows, benefit)
                    ],
                )
            )
    result = dict(
        status="passed",
        scalar_checks=checks,
        crossing_status_checks=event_checks,
        prior_metric_reconciliations=reconciled,
        producer_metric_helpers_imported=False,
        development_gripper=tables,
        development_first_action_by_dimension=dim_table,
        development_train_only_bias_subtraction=gains,
        caveat=(
            "All analyses exploratory; train-only adjustment is an offline diagnostic, "
            "not a deployable repair."
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "scalar_checks",
                    "crossing_status_checks",
                    "prior_metric_reconciliations",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
