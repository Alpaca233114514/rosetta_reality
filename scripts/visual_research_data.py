"""Bounded nonhidden data audit and independent video packet timestamp checks."""

from __future__ import annotations

import json
import os
from pathlib import Path

from rosetta_reality.vla.visual_research import require, seal_bundle, sha256, write_json


def extract(plan, output, root):
    import numpy as np
    import pyarrow.dataset as arrow

    from rosetta_reality.data.cache_resolver import resolve_prepared_cache
    from rosetta_reality.data.config import load_dataset_config
    from rosetta_reality.sim import load_action_contract
    from rosetta_reality.vla.action_space import load_smolvla_experiment

    require(os.environ.get("HF_HUB_OFFLINE") == "1", "Offline data boundary required")
    config = load_dataset_config(root / plan["dataset_config"])
    exp = load_smolvla_experiment(root / plan["experiment"], root)
    train, dev, hidden = (
        exp["dataset"][key] for key in ("train_episodes", "validation_episodes", "test_episodes")
    )
    require(train == plan["train_episodes"] and dev == plan["development_episodes"], "Split drift")
    require(not (set(train + dev) & set(hidden)), "Hidden split overlap")
    cache, _ = resolve_prepared_cache(config, root, validate_checksums=True)
    require(
        sha256(cache / "manifest.json") == plan["dataset_manifest_sha256"], "Cache identity drift"
    )
    require(
        sha256(cache / "cache_checksums.json") == plan["cache_checksums_sha256"],
        "Cache content inventory drift",
    )
    info = json.loads((cache / "meta/info.json").read_text())
    contract = load_action_contract(root / exp["action_contract"]["derived"])
    lower, upper = contract.lower_bounds.numpy(), contract.upper_bounds.numpy()
    groups = {}
    for side in ("left", "right"):
        for kind in ("joints", "gripper"):
            groups[side + "_" + kind] = [
                i
                for i, d in enumerate(contract.dimensions)
                if d.name.startswith(side + "_") and (("gripper" in d.name) == (kind == "gripper"))
            ]
    require(all(groups.values()), "Action dimension names do not define registered groups")
    dataset = arrow.dataset(cache / "data", format="parquet")
    columns = [
        config.fields.episode_index,
        config.fields.frame_index,
        config.fields.timestamp,
        config.fields.state,
        config.fields.action,
    ]
    checksums = json.loads((cache / "cache_checksums.json").read_text())["files"]
    source_root = Path(os.environ["ROSETTA_AUTODL_ROOT"])
    schedule_path = source_root / plan["training_schedule_source"]
    ingress_path = source_root / plan["training_ingress_source"]
    require(sha256(schedule_path) == plan["training_schedule_sha256"], "Historical schedule drift")
    require(sha256(ingress_path) == plan["training_ingress_sha256"], "Historical ingress drift")
    schedule = json.loads(schedule_path.read_text())["sample_identities"]
    ingress = json.loads(ingress_path.read_text())
    require(
        [[r["episode"], r["frame"]] for r in ingress["records"]] == schedule
        and ingress["optimizer_steps"] == 5000,
        "Historical consumed sequence differs",
    )
    exposure = {}
    for checkpoint in (2500, 5000):
        seen = set(map(tuple, schedule[: checkpoint * 4]))
        target_counts = {ep: np.zeros(501, dtype=np.int64) for ep in train}
        for ep, frame in schedule[: checkpoint * 4]:
            target_counts[ep][frame] += 1
            target_counts[ep][min(frame + contract.chunk_length, 500)] -= 1
        exposure[checkpoint] = (seen, {ep: np.cumsum(v)[:500] for ep, v in target_counts.items()})
    episodes, raw, states, timestamps, ids = [], [], [], [], []
    sample_ids, targets, sample_states, masks = [], [], [], []
    event_rows, index_rows = [], []
    for ep in train + dev:
        rows = []
        for fragment in dataset.get_fragments():
            fragment_rows = fragment.to_table(
                columns=columns, filter=arrow.field(config.fields.episode_index) == ep
            ).to_pylist()
            name = str(fragment.path).replace(str(cache) + "/", "")
            require(name in checksums, "Unsealed raw source fragment")
            for row in fragment_rows:
                row["_source_file"] = name
            rows.extend(fragment_rows)
        rows.sort(key=lambda r: r[config.fields.frame_index])
        n = len(rows)
        require(
            n == 500 and [r[config.fields.frame_index] for r in rows] == list(range(n)),
            "Noncontiguous frames",
        )
        a = np.asarray([r[config.fields.action] for r in rows], dtype=np.float64)
        s = np.asarray([r[config.fields.state] for r in rows], dtype=np.float64)
        t = np.asarray([r[config.fields.timestamp] for r in rows], dtype=np.float64)
        require(a.shape == s.shape == (n, contract.dimension), "Raw dimensions differ")
        require(all(np.isfinite(x).all() for x in (a, s, t)), "Nonfinite raw data")
        require(
            np.allclose(t, np.arange(n) / info["fps"], atol=1e-5, rtol=0), "Numeric timestamp drift"
        )
        projected = np.clip(a, lower, upper)
        require(
            np.all(np.abs(a - projected) <= contract.source_overshoot_tolerances.numpy() + 1e-6),
            "Unregistered target overshoot",
        )
        raw.append(a)
        states.append(s)
        timestamps.append(t)
        ids.extend([[ep, f] for f in range(n)])
        episodes.append(
            {
                "episode": ep,
                "frames": n,
                "split": "train" if ep in train else "development",
                "raw_action_sha256": __import__("hashlib").sha256(a.tobytes()).hexdigest(),
            }
        )
        for frame in plan["frame_offsets"] + plan["boundary_offsets"]:
            positions = frame + np.arange(contract.chunk_length)
            sample_ids.append([ep, frame])
            targets.append(projected[np.minimum(positions, n - 1)])
            sample_states.append(s[frame])
            masks.append(positions < n)
        for side in ("left", "right"):
            k = groups[side + "_gripper"][0]
            crossings = np.flatnonzero((projected[1:, k] >= 0.5) != (projected[:-1, k] >= 0.5)) + 1
            event_rows.append(
                {
                    "episode": ep,
                    "side": side,
                    "crossing_frames": crossings.tolist(),
                    "definition": "command aperture crossing 0.5; not actual contact or grasp",
                    "fraction_above_0_5": float((projected[:, k] >= 0.5).mean()),
                }
            )
        for f in range(n):
            index_rows.append(
                {
                    "episode": ep,
                    "frame": f,
                    "row_index": len(index_rows),
                    "split": "train" if ep in train else "development",
                    "timestamp": float(t[f]),
                    "source_parquet": rows[f]["_source_file"],
                    "source_parquet_sha256": checksums[rows[f]["_source_file"]],
                    "dataset_revision": config.revision,
                    "projection_changed_dimensions": np.flatnonzero(a[f] != projected[f]).tolist(),
                    "input_seen_at_checkpoint": {
                        str(k): (ep, f) in value[0] for k, value in exposure.items()
                    },
                    "valid_target_supervision_count_at_checkpoint": {
                        str(k): int(value[1][ep][f]) if ep in train else 0
                        for k, value in exposure.items()
                    },
                }
            )
    raw, states = np.concatenate(raw), np.concatenate(states)
    projected = np.clip(raw, lower, upper)
    arrays = {
        "raw_actions": raw,
        "projected_actions": projected,
        "states": states,
        "identities": np.asarray(ids, dtype=np.int64),
        "timestamps": np.concatenate(timestamps),
        "sample_identities": np.asarray(sample_ids, dtype=np.int64),
        "sample_targets": np.asarray(targets),
        "sample_states": np.asarray(sample_states),
        "sample_valid_mask": np.asarray(masks),
        "lower_bounds": lower,
        "upper_bounds": upper,
        "checkpoint_steps": np.array([2500, 5000], dtype=np.int64),
        "input_seen_at_checkpoint": np.array(
            [[row["input_seen_at_checkpoint"][str(k)] for row in index_rows] for k in (2500, 5000)],
            dtype=bool,
        ),
        "valid_target_supervision_count_at_checkpoint": np.array(
            [
                [row["valid_target_supervision_count_at_checkpoint"][str(k)] for row in index_rows]
                for k in (2500, 5000)
            ],
            dtype=np.int64,
        ),
    }
    with (output / "arrays.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with (output / "samples.jsonl").open("x") as stream:
        for row in index_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    train_rows = np.isin(arrays["identities"][:, 0], train)
    summary = {
        "status": "numeric_data_verified",
        "episode_count": len(episodes),
        "train_rows": int(train_rows.sum()),
        "development_rows": int((~train_rows).sum()),
        "hidden_rows_materialized": 0,
        "model_loaded": False,
        "optimizer_steps": 0,
        "groups": groups,
        "episodes": episodes,
        "command_events": event_rows,
        "source_files_sha256": json.loads((cache / "cache_checksums.json").read_text())["files"],
        "projection_count_by_dimension": (raw != projected).sum(axis=0).tolist(),
        "projection_max_abs_by_dimension": np.abs(raw - projected).max(axis=0).tolist(),
        "sample_count": len(sample_ids),
        "sample_valid_slots": int(arrays["sample_valid_mask"].sum()),
        "physical_image_state_action_alignment": "not established by numeric checks",
        "independent_video_pts": "separate stage",
        "new_unique_scenes": "not inferred from frames",
        "training_schedule_sha256": sha256(schedule_path),
        "training_ingress_sha256": sha256(ingress_path),
        "training_exposure": {
            str(k): {
                "input_exposures": k * 4,
                "unique_input_frames": len(exposure[k][0]),
                "valid_target_slots": int(
                    arrays["valid_target_supervision_count_at_checkpoint"][i].sum()
                ),
                "unique_target_frames": int(
                    (arrays["valid_target_supervision_count_at_checkpoint"][i] > 0).sum()
                ),
                "main_grid_seen_inputs": sum(
                    (ep, frame) in exposure[k][0] for ep in train for frame in plan["frame_offsets"]
                ),
            }
            for i, k in enumerate((2500, 5000))
        },
    }
    write_json(output / "result.json", summary)
    seal_bundle(
        output,
        {"plan_sha256": plan["_sha256"], "stage": "data", "dataset_revision": config.revision},
    )
    return summary


def video_pts(plan, output, root):
    """Read packet PTS independently from LeRobot nearest-timestamp selection.

    Demuxing does not decode images. Packet PTS validates the media time grid,
    not decoded pixel identity or physical correspondence with robot state.
    """
    import av
    import numpy as np
    import pyarrow.dataset as arrow

    from rosetta_reality.data.cache_resolver import resolve_prepared_cache
    from rosetta_reality.data.config import load_dataset_config

    cfg = load_dataset_config(root / plan["dataset_config"])
    cache, _ = resolve_prepared_cache(cfg, root, validate_checksums=True)
    require(
        sha256(cache / "manifest.json") == plan["dataset_manifest_sha256"], "Cache identity drift"
    )
    info = json.loads((cache / "meta/info.json").read_text())
    allowed = plan["train_episodes"] + plan["development_episodes"]
    meta = (
        arrow.dataset(cache / "meta/episodes", format="parquet")
        .to_table(filter=arrow.field("episode_index").isin(allowed))
        .to_pylist()
    )
    require(
        sorted(r["episode_index"] for r in meta) == sorted(allowed), "Episode metadata mismatch"
    )
    by_video, records = {}, []
    camera = cfg.cameras["top"]
    for row in meta:
        prefix = "videos/" + camera + "/"
        path = info["video_path"].format(
            video_key=camera,
            chunk_index=row[prefix + "chunk_index"],
            file_index=row[prefix + "file_index"],
        )
        by_video.setdefault(path, []).append(row)
    for path, rows in by_video.items():
        with av.open(str(cache / path)) as container:
            stream = container.streams.video[0]
            time_base = float(stream.time_base)
            pts = np.array(
                [p.pts * time_base for p in container.demux(stream) if p.pts is not None],
                dtype=np.float64,
            )
        require(len(pts) > 0 and np.isfinite(pts).all(), "No finite packet PTS")
        pts.sort()
        for row in rows:
            prefix = "videos/" + camera + "/"
            start, stop = float(row[prefix + "from_timestamp"]), float(row[prefix + "to_timestamp"])
            local = pts[(pts >= start - 1e-7) & (pts < stop - 1e-7)]
            desired = start + np.arange(int(row["length"])) / info["fps"]
            distances = (
                [float(np.min(np.abs(local - value))) for value in desired] if len(local) else []
            )
            passed = len(local) == len(desired) and bool(distances) and max(distances) <= 1e-5
            records.append(
                {
                    "episode": int(row["episode_index"]),
                    "video": path,
                    "video_sha256": sha256(cache / path),
                    "from_timestamp": start,
                    "to_timestamp": stop,
                    "packet_count": len(local),
                    "expected_frames": len(desired),
                    "max_nearest_pts_error_seconds": max(distances) if distances else None,
                    "duplicate_pts": int(len(local) - len(np.unique(local))),
                    "grid_passed": passed,
                    "pts_seconds": local.tolist(),
                }
            )
    write_json(
        output / "result.json",
        {
            "status": "passed" if all(r["grid_passed"] for r in records) else "failed",
            "records": records,
            "hidden_rows_materialized": 0,
            "pixels_decoded": 0,
            "model_loaded": False,
            "scope": "independent demux packet PTS versus per-episode numeric time grid",
            "physical_alignment": "not measured",
            "decoder_pixel_parity": "not measured",
        },
    )
    seal_bundle(output, {"stage": "pts", "plan_sha256": plan["_sha256"]})
    return records


def supervision(plan, output, inputs, root):
    import numpy as np

    from rosetta_reality.vla.visual_research import episode_summary, verify_bundle
    from scripts.diagnose_hestia_scene_phase import geometry_neighbors

    require(inputs is not None, "Supervision requires --input data bundle")
    require(verify_bundle(inputs)["plan_sha256"] == plan["_sha256"], "Data plan mismatch")
    with np.load(inputs / "arrays.npz", allow_pickle=False) as saved:
        raw, projected, ids = (saved[k] for k in ("raw_actions", "projected_actions", "identities"))
    with np.load(root / plan["scene_phase_arrays"], allow_pickle=False) as saved:
        coordinates, episodes, old_raw = (
            saved[k] for k in ("coordinates", "episodes", "raw_targets")
        )
    require(
        episodes.tolist() == plan["train_episodes"] + plan["development_episodes"],
        "Geometry episode order drift",
    )
    current = np.stack([raw[ids[:, 0] == ep] for ep in episodes])
    require(
        np.array_equal(current[:, : old_raw.shape[1]], old_raw),
        "Retained geometry labels do not reproduce fresh raw reads",
    )
    target = np.stack([projected[ids[:, 0] == ep] for ep in episodes])
    distance, neighbors, allowed = geometry_neighbors(
        coordinates, episodes.tolist(), list(range(40)), plan["analysis"]["geometry_k"]
    )
    predicted = np.stack([target[n].mean(axis=0) for n in neighbors])
    means = np.stack([target[n].mean(axis=0) for n in allowed])
    medians = np.stack([np.median(target[n], axis=0) for n in allowed])
    data_report = json.loads((inputs / "result.json").read_text())
    groups, records = data_report["groups"], []
    for frame in plan["frame_offsets"]:
        for window, slots in (
            ("first", slice(frame, frame + 1)),
            ("full_chunk", slice(frame, frame + 50)),
        ):
            for group, dims in groups.items():
                y = target[:, slots][:, :, dims]
                errors = {
                    "geometry_3nn_mae": np.abs(predicted[:, slots][:, :, dims] - y).mean(
                        axis=(1, 2)
                    ),
                    "train_mean_mae": np.abs(means[:, slots][:, :, dims] - y).mean(axis=(1, 2)),
                    "train_median_mae": np.abs(medians[:, slots][:, :, dims] - y).mean(axis=(1, 2)),
                }
                for split, ix in (("train_loo", slice(0, 40)), ("development", slice(40, 45))):
                    records.append(
                        {
                            "frame": frame,
                            "window": window,
                            "group": group,
                            "split": split,
                            **{k: episode_summary(v[ix], episodes[ix]) for k, v in errors.items()},
                        }
                    )
    train_nearest = distance[np.arange(40), neighbors[:40, 0]]
    threshold = float(np.quantile(train_nearest, 0.95))
    support = [
        {
            "episode": int(ep),
            "nearest_train_episode": int(episodes[neighbors[i, 0]]),
            "distance": float(distance[i, neighbors[i, 0]]),
            "above_train_loo_95pct": bool(distance[i, neighbors[i, 0]] > threshold),
        }
        for i, ep in enumerate(episodes)
    ]
    arrays = {
        "episodes": episodes,
        "coordinates": coordinates,
        "neighbors": neighbors,
        "distance": distance,
        "raw_targets": current,
        "projected_targets": target,
        "geometry_3nn_predictions": predicted,
        "train_mean": means,
        "train_median": medians,
    }
    with (output / "arrays.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    write_json(
        output / "result.json",
        {
            "status": "completed_non_gating",
            "records": records,
            "support": support,
            "train_loo_distance_95pct": threshold,
            "duplicate_geometry_pairs": [
                [int(episodes[i]), int(episodes[j])]
                for i in range(45)
                for j in range(i + 1, 45)
                if distance[i, j] == 0
            ],
            "historical_raw_label_reproduction_exact": True,
            "geometry_role": "initial 2D two-object proxy only",
            "model_loaded": False,
            "optimizer_steps": 0,
            "hidden_rows_materialized": 0,
            "limitations": [
                "LOO excludes query labels; no policy retraining",
                "geometry does not include full 3D pose or later observations",
                "development is not an untouched final test",
                "neighbor failure cannot prove irreducible label noise",
            ],
        },
    )
    seal_bundle(
        output,
        {
            "stage": "supervision",
            "plan_sha256": plan["_sha256"],
            "input_manifest_sha256": sha256(inputs / "manifest.json"),
        },
    )


def pixels(plan, output, root):
    """Independent timestamp seek against SHA-bound prior LeRobot decode evidence."""
    import hashlib

    import av
    import numpy as np
    import pyarrow.dataset as arrow

    from rosetta_reality.data.cache_resolver import resolve_prepared_cache
    from rosetta_reality.data.config import load_dataset_config

    prior_path = root / plan["prior_decoded_samples"]
    require(
        sha256(prior_path) == plan["evidence_sha256"][plan["prior_decoded_samples"]],
        "Prior decode evidence drift",
    )
    prior = {(r["episode"], r["frame"]): r["sha256"] for r in json.loads(prior_path.read_text())}
    config = load_dataset_config(root / plan["dataset_config"])
    cache, _ = resolve_prepared_cache(config, root, validate_checksums=True)
    info = json.loads((cache / "meta/info.json").read_text())
    allowed = plan["train_episodes"] + plan["development_episodes"]
    rows = (
        arrow.dataset(cache / "meta/episodes", format="parquet")
        .to_table(filter=arrow.field("episode_index").isin(allowed))
        .to_pylist()
    )
    records, images = [], {}
    keyframes = {}
    prefix = "videos/" + config.cameras["top"] + "/"
    for row in rows:
        ep = int(row["episode_index"])
        relative = info["video_path"].format(
            video_key=config.cameras["top"],
            chunk_index=row[prefix + "chunk_index"],
            file_index=row[prefix + "file_index"],
        )
        if relative not in keyframes:
            with av.open(str(cache / relative)) as media:
                track = media.streams.video[0]
                keyframes[relative] = sorted(
                    p.pts * float(track.time_base)
                    for p in media.demux(track)
                    if p.pts is not None and p.is_keyframe
                )
        for offset in plan["pixel_offsets"]:
            requested = float(row[prefix + "from_timestamp"]) + offset / info["fps"]
            preceding = [t for t in keyframes[relative] if t <= requested + 1e-7]
            if not preceding or max(preceding) < float(row[prefix + "from_timestamp"]) - 1e-7:
                records.append(
                    {
                        "episode": ep,
                        "frame": offset,
                        "pixel_hash_equal": None,
                        "status": "not_decoded_cross_episode_keyframe_preroll",
                    }
                )
                continue
            with av.open(str(cache / relative)) as container:
                stream = container.streams.video[0]
                stream.thread_type = "NONE"
                stream.thread_count = 1
                container.seek(
                    round(max(preceding) / float(stream.time_base)), stream=stream, backward=True
                )
                previous, candidate = None, None
                for frame in container.decode(stream):
                    stamp = float(frame.pts * frame.time_base)
                    if stamp >= requested - 1e-7:
                        candidate = (stamp, frame)
                        if previous is not None and abs(previous[0] - requested) < abs(
                            stamp - requested
                        ):
                            candidate = previous
                        break
                    previous = (stamp, frame)
                require(candidate is not None, "Independent decoder did not reach requested frame")
                stamp, frame = candidate
                require(
                    abs(stamp - requested) <= 1 / info["fps"] / 2 + 1e-6,
                    "Decoded PTS is not the requested frame",
                )
                rgb = np.ascontiguousarray(frame.to_ndarray(format="rgb24").transpose(2, 0, 1))
            digest = hashlib.sha256(rgb.tobytes()).hexdigest()
            key = f"episode_{ep:02d}_frame_{offset:03d}"
            images[key] = rgb
            records.append(
                {
                    "episode": ep,
                    "frame": offset,
                    "video": relative,
                    "requested_timestamp": requested,
                    "decoded_timestamp": stamp,
                    "sha256": digest,
                    "prior_native_sha256": prior[ep, offset],
                    "pixel_hash_equal": digest == prior[ep, offset],
                    "array_key": key,
                }
            )
    with (output / "images.npz").open("xb") as stream:
        np.savez_compressed(stream, **images)
    write_json(
        output / "result.json",
        {
            "status": "passed"
            if all(r["pixel_hash_equal"] is True for r in records)
            else "differences_or_unmeasured_samples",
            "records": records,
            "sample_count": len(records),
            "hidden_pixels_saved": 0,
            "decode_preroll": (
                "only seek when the preceding keyframe lies in the same nonhidden episode"
            ),
            "physical_alignment": "not established by pixel equality",
            "model_loaded": False,
        },
    )
    seal_bundle(output, {"stage": "pixels", "plan_sha256": plan["_sha256"]})
    return records
