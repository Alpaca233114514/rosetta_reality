"""Inspect initial geometry versus bounded action-phase labels, without policy fitting."""

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


def geometry_neighbors(coordinates, episodes, train_rows, k):
    """Select donors using input geometry only, with stable identity-based tie breaking."""
    x = np.asarray(coordinates, dtype=np.float64)
    require(x.shape == (len(episodes), 4) and np.isfinite(x).all(), "Finite two-object coordinates")
    require(len(set(episodes)) == len(episodes), "Duplicate episode identity")
    require(0 < k < len(train_rows), "Invalid fixed neighbor count")
    z = x / [640, 480, 640, 480]
    distance = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=2)
    allowed, nearest = [], []
    for i in range(len(x)):
        candidates = [j for j in train_rows if episodes[j] != episodes[i]]
        donors = sorted(candidates, key=lambda j: (distance[i, j], episodes[j]))
        allowed.append(np.asarray(donors, dtype=int))
        nearest.append(donors[:k])
    return distance, np.asarray(nearest, dtype=int), allowed


def first_upcross(series, threshold):
    """Return observed first crossings; None is right-censoring, never an invented time."""
    a = np.asarray(series, dtype=np.float64)
    require(a.ndim == 2 and a.shape[1] > 0 and np.isfinite(a).all(), "Finite [scene,time] series")
    require(np.all(a[:, 0] < threshold), "An initially open gripper has no observed onset")
    return [
        int(np.flatnonzero(row >= threshold)[0]) if (row >= threshold).any() else None for row in a
    ]


def event_scores(onsets, nearest, allowed, rows):
    observed = np.asarray([t is not None for t in onsets], dtype=bool)
    truth, probabilities, baselines = [], [], []
    rows_detail = []
    near_discord, all_discord, near_gaps, all_gaps = [], [], [], []
    timing_errors, baseline_timing_errors = [], []
    for i in rows:
        donors, reference = nearest[i], allowed[i]
        p, b = float(observed[donors].mean()), float(observed[reference].mean())
        truth.append(observed[i])
        probabilities.append(p)
        baselines.append(b)
        near_discord.extend((observed[donors] != observed[i]).tolist())
        all_discord.extend((observed[reference] != observed[i]).tolist())
        measured = [onsets[j] for j in donors if onsets[j] is not None]
        all_measured = [onsets[j] for j in reference if onsets[j] is not None]
        onset = float(np.median(measured)) if measured else None
        baseline_onset = float(np.median(all_measured)) if all_measured else None
        if onsets[i] is not None:
            near_gaps.extend(abs(onsets[i] - t) for t in measured)
            all_gaps.extend(abs(onsets[i] - t) for t in all_measured)
            if onset is not None and baseline_onset is not None:
                timing_errors.append(abs(onsets[i] - onset))
                baseline_timing_errors.append(abs(onsets[i] - baseline_onset))
        rows_detail.append(
            {
                "row": int(i),
                "target_onset": onsets[i],
                "neighbor_probability": p,
                "constant_probability": b,
                "neighbor_onset": onset,
                "constant_onset": baseline_onset,
            }
        )
    y, p, b = (np.asarray(a) for a in (truth, probabilities, baselines))

    def confusion(values):
        pred = values >= 0.5
        return {
            "tp": int((pred & y).sum()),
            "fp": int((pred & ~y).sum()),
            "fn": int((~pred & y).sum()),
            "tn": int((~pred & ~y).sum()),
        }

    return {
        "count": len(rows),
        "observed_event_count": int(y.sum()),
        "neighbor_brier": float(np.square(p - y).mean()),
        "constant_brier": float(np.square(b - y).mean()),
        "neighbor_confusion": confusion(p),
        "constant_confusion": confusion(b),
        "neighbor_pair_event_disagreement": float(np.mean(near_discord)),
        "all_donor_pair_event_disagreement": float(np.mean(all_discord)),
        "neighbor_observed_pair_time_gap": {
            "count": len(near_gaps),
            "mean_frames": float(np.mean(near_gaps)) if near_gaps else None,
        },
        "all_donor_observed_pair_time_gap": {
            "count": len(all_gaps),
            "mean_frames": float(np.mean(all_gaps)) if all_gaps else None,
        },
        "conditional_onset_mae": {
            "common_count": len(timing_errors),
            "neighbor_frames": float(np.mean(timing_errors)) if timing_errors else None,
            "constant_frames": float(np.mean(baseline_timing_errors)) if timing_errors else None,
        },
        "rows": rows_detail,
    }


def trajectory_scores(target, nearest, allowed, groups, rows, windows):
    predicted = target[nearest].mean(axis=1)
    mean = np.stack([target[donors].mean(0) for donors in allowed])
    median = np.stack([np.median(target[donors], axis=0) for donors in allowed])
    result = {}
    for view, indices in rows.items():
        result[view] = {}
        for name, (start, stop) in windows.items():
            result[view][name] = {}
            for group, dims in groups.items():
                y = target[indices, start:stop][..., dims]
                result[view][name][group] = {
                    method: float(np.abs(value[indices, start:stop][..., dims] - y).mean())
                    for method, value in (
                        ("neighbor_mae", predicted),
                        ("mean_mae", mean),
                        ("median_mae", median),
                    )
                }
    return result, predicted


def read_phase_actions(root, cfg, episodes, hidden, stop):
    """Filter episode and prefix before materialization; do not request images/future state."""
    import pyarrow.dataset as arrow

    require(
        episodes and len(set(episodes)) == len(episodes) and not set(episodes) & set(hidden),
        "Invalid or hidden request",
    )
    require(0 < stop <= 100, "Unregistered time range")
    f = cfg.fields
    table = arrow.dataset(root / "data", format="parquet").to_table(
        columns=[f.episode_index, f.frame_index, f.timestamp, f.action],
        filter=arrow.field(f.episode_index).isin(episodes)
        & (arrow.field(f.frame_index) >= 0)
        & (arrow.field(f.frame_index) < stop),
    )
    # Defense before constructing Python row objects.
    require(set(table[f.episode_index].to_pylist()) == set(episodes), "Unexpected scanned episodes")
    records = table.to_pylist()
    by_id = {(int(r[f.episode_index]), int(r[f.frame_index])): r for r in records}
    wanted = {(ep, t) for ep in episodes for t in range(stop)}
    require(
        len(records) == len(wanted) and set(by_id) == wanted, "Missing or duplicate prefix rows"
    )
    for (ep, t), row in by_id.items():
        require(abs(row[f.timestamp] - t / 50) <= 1e-6, "Timestamp differs from 50 Hz")
        require(np.isfinite(row[f.action]).all(), "Nonfinite label")
    return np.asarray([[by_id[ep, t][f.action] for t in range(stop)] for ep in episodes]), len(
        records
    )


def inspect_inputs(plan):
    from rosetta_reality.data import resolve_prepared_cache
    from rosetta_reality.data.cache_resolver import ordered_feature_names
    from rosetta_reality.data.config import load_dataset_config
    from rosetta_reality.sim.action_contract import load_action_contract

    require(
        os.environ.get("ROSETTA_CONTAINER_IMAGE_ID") == plan["image"]
        and os.environ.get("HF_HUB_OFFLINE") == "1",
        "Registered offline runtime required",
    )
    for name, sha in plan["sha256"].items():
        require(not Path(name).is_symlink() and file_hash(name) == sha, f"Input drift: {name}")
    cfg = load_dataset_config(Path(plan["dataset_config"]))
    root, manifest = resolve_prepared_cache(cfg, Path.cwd(), validate_checksums=True)
    require(
        file_hash(root / "manifest.json") == plan["dataset_manifest_sha256"], "Cache manifest drift"
    )
    contract = load_action_contract(Path(plan["action_contract"]))
    require(
        ordered_feature_names(root, cfg.fields.action) == contract.dimension_names,
        "Dataset/contract action ordering differs",
    )
    require(contract.frequency_hz == 50 and contract.chunk_length == 50, "Time contract drift")
    old = read_json(Path(plan["historical_bundle"]) / "manifest.json")
    meta = old["metadata"]
    split_rows(meta)
    require(
        meta["episodes"] == plan["episodes"] and meta["hidden_episodes"] == plan["hidden_episodes"],
        "Registered episode identity differs",
    )
    require(
        manifest.resolved_revision == meta["common_identity"]["data_revision"], "Revision drift"
    )
    require(
        file_hash(plan["action_contract"]) == meta["common_identity"]["physical_contract_sha256"],
        "Historical physical contract differs",
    )
    require(
        [d["name"] for d in meta["dimensions"]] == list(contract.dimension_names),
        "Saved action ordering",
    )
    require(len(set(meta["nonvisual_hashes"])) == 1, "Historical nonvisual conditioning differs")
    review = read_json(Path(plan["geometry_root"]) / "visual-review.json")
    authority = read_json(plan["geometry_authority"])
    for name in ("coordinates.npz", "extraction.json", "visual-review.json"):
        path = Path(plan["geometry_root"]) / name
        require(
            file_hash(path) == authority["evidence_sha256"][path.as_posix()],
            "Geometry authority differs",
        )
    require(review["accepted_images"] == len(meta["episodes"]), "Incomplete image review")
    for name, sha in review["sha256"].items():
        require(file_hash(Path(plan["geometry_root"]) / name) == sha, "Reviewed evidence drift")
    return root, cfg, contract, old


def main():
    from scripts.diagnose_kv_full_chunk import project_targets

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("inspect", "analyze"))
    args = parser.parse_args()
    plan = read_json(args.plan)
    started = time.monotonic()
    root, cfg, contract, old = inspect_inputs(plan)
    if args.stage == "inspect":
        print(
            json.dumps(
                {
                    "status": "passed",
                    "cache_manifest_sha256": file_hash(root / "manifest.json"),
                    "cache_checksums_verified": True,
                    "coordinate_authority_verified": True,
                    "new_rows_materialized": 0,
                }
            )
        )
        return
    output = Path(plan["output"])
    require(output.is_dir() and not any(output.iterdir()), "New empty output directory required")
    meta, bundle = old["metadata"], Path(plan["historical_bundle"])
    rows = split_rows(meta)
    with np.load(Path(plan["geometry_root"]) / "coordinates.npz", allow_pickle=False) as saved:
        coords = saved["coordinates"].copy()
    extraction = read_json(Path(plan["geometry_root"]) / "extraction.json")
    require(
        [r["episode"] for r in extraction["rows"]] == meta["episodes"], "Coordinate order differs"
    )
    expected = [[*r["blue"]["centroid"], *r["red"]["centroid"]] for r in extraction["rows"]]
    require(np.array_equal(coords, expected), "Coordinate content differs")
    distance, nearest, allowed = geometry_neighbors(
        coords, meta["episodes"], rows["train40"], plan["k"]
    )
    pair_order = sorted(
        [(i, j) for p, i in enumerate(rows["train40"]) for j in rows["train40"][p + 1 :]],
        key=lambda ij: (distance[ij], meta["episodes"][ij[0]], meta["episodes"][ij[1]]),
    )
    # Input-only choices are finalized before new target labels are materialized.
    raw, materialized = read_phase_actions(
        root, cfg, meta["episodes"], meta["hidden_episodes"], plan["frames"]
    )
    target, projection = project_targets(raw, contract, file_hash(plan["action_contract"]))
    historical = {}
    for name in ("standard_targets", "standard_predictions"):
        entry = old["arrays"][name]
        require(
            entry["path"] == name + ".npy" and file_hash(bundle / entry["path"]) == entry["sha256"],
            "Historical saved array differs",
        )
        historical[name] = np.load(bundle / entry["path"], allow_pickle=False)
    prefix_error = float(np.abs(target[:, :50] - historical["standard_targets"]).max())
    require(
        prefix_error <= plan["prefix_atol"], "First 50 projected labels do not match historical C"
    )
    groups = {
        "left_wrist_angle": [
            meta["dimensions"].index(
                next(d for d in meta["dimensions"] if d["name"] == "left_wrist_angle")
            )
        ]
    }
    for side in ("left", "right"):
        groups[side + "_joint"] = [
            i
            for i, d in enumerate(meta["dimensions"])
            if d["name"].startswith(side + "_") and d["unit"] == "radian"
        ]
        groups[side + "_gripper"] = [
            i for i, d in enumerate(meta["dimensions"]) if d["name"] == side + "_gripper"
        ]
    result = {
        "id": plan["id"],
        "status": "completed",
        "plan_sha256": file_hash(args.plan),
        "new_action_rows_materialized": materialized,
        "new_raw_images_decoded": 0,
        "future_states_materialized": False,
        "hidden_rows_materialized": False,
        "prefix_max_abs": prefix_error,
        "projection": projection,
        "events": {},
        "nearest_train_pairs": [],
        "development_neighbors": [],
        "groups": groups,
    }
    onset_records = {}
    for horizon in plan["event_horizons"]:
        result["events"][str(horizon)] = {}
        for side in ("left_gripper", "right_gripper"):
            d = groups[side][0]
            onsets = first_upcross(target[:, :horizon, d], plan["threshold"])
            onset_records[horizon, side] = onsets
            result["events"][str(horizon)][side] = {
                view: event_scores(onsets, nearest, allowed, indices)
                for view, indices in rows.items()
            }
    for i, j in pair_order[: plan["closest_pairs"]]:
        result["nearest_train_pairs"].append(
            {
                "episodes": [meta["episodes"][i], meta["episodes"][j]],
                "distance": float(distance[i, j]),
                "coordinates": [coords[i].tolist(), coords[j].tolist()],
                "onsets100": {
                    side: [onset_records[100, side][i], onset_records[100, side][j]]
                    for side in ("left_gripper", "right_gripper")
                },
                "left_wrist_first50_pair_mae": float(
                    np.abs(
                        target[i, :50, groups["left_wrist_angle"]]
                        - target[j, :50, groups["left_wrist_angle"]]
                    ).mean()
                ),
            }
        )
    for i in rows["dev5"]:
        result["development_neighbors"].append(
            {
                "episode": meta["episodes"][i],
                "coordinates": coords[i].tolist(),
                "donors": [meta["episodes"][j] for j in nearest[i]],
                "distances": distance[i, nearest[i]].tolist(),
                "events": {
                    side: {
                        "target_onset100": onset_records[100, side][i],
                        "donor_onsets100": [onset_records[100, side][j] for j in nearest[i]],
                        "native1280_onsets50_by_noise": first_upcross(
                            historical["standard_predictions"][:, i, :, groups[side][0]],
                            plan["threshold"],
                        ),
                    }
                    for side in ("left_gripper", "right_gripper")
                },
            }
        )
    result["trajectory"], predicted = trajectory_scores(
        target, nearest, allowed, groups, rows, plan["windows"]
    )
    arrays = {
        "raw_targets": raw,
        "projected_targets": target,
        "coordinates": coords,
        "neighbors": nearest,
        "episodes": np.asarray(meta["episodes"]),
        "neighbor_predictions": predicted,
    }
    with (output / "arrays.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(output / "arrays.npz", allow_pickle=False) as restored:
        require(
            all(np.array_equal(a, restored[k]) for k, a in arrays.items()),
            "Derived array reload differs",
        )
    result.update(
        array_sha256=file_hash(output / "arrays.npz"),
        array_reload_exact=True,
        seconds=time.monotonic() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        new_model_forwards=0,
        policy_optimizer_steps=0,
        m2_complete=False,
        gate34="not measured",
        query_future_labels_used_for_donor_selection=False,
    )
    require(
        result["seconds"] < plan["maximum_seconds"]
        and result["peak_rss_bytes"] < plan["maximum_rss_bytes"],
        "Budget exceeded",
    )
    with (output / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("events", "trajectory", "projection", "nearest_train_pairs")
            }
        )
    )


if __name__ == "__main__":
    main()
