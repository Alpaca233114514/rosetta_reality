"""Independent arithmetic replay of the no-card research data package.

Uses saved raw arrays directly, without calling the producer's metric,
neighbor, projection or timestamp functions. Does not load models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def verify(bundle, amendment=None, schedule_path=None, ingress_path=None, decoded_path=None):
    checked_files = 0
    manifests = list(bundle.rglob("manifest.json"))
    if amendment is not None:
        manifests.append(amendment / "manifest.json")
    for manifest in manifests:
        for name, record in json.loads(manifest.read_text())["files"].items():
            path = (manifest.parent / name).resolve()
            assert path.is_relative_to(manifest.parent.resolve())
            assert path.stat().st_size == record["bytes"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
            checked_files += 1
    with np.load(bundle / "data/arrays.npz", allow_pickle=False) as data:
        raw, projected, ids = (data[k] for k in ("raw_actions", "projected_actions", "identities"))
        assert len(raw) == 22500 and len(np.unique(ids, axis=0)) == 22500
        assert not np.isin(ids[:, 0], [31, 6, 1, 24, 5]).any()
        expected = np.minimum(np.maximum(raw, data["lower_bounds"]), data["upper_bounds"])
        np.testing.assert_array_equal(projected, expected)
        assert int((raw != projected).sum()) == 6973
        for i, (episode, frame) in enumerate(data["sample_identities"]):
            episode_rows = np.flatnonzero(ids[:, 0] == episode)
            positions = frame + np.arange(data["sample_targets"].shape[1])
            valid = positions < len(episode_rows)
            expected = projected[episode_rows[np.minimum(positions, len(episode_rows) - 1)]]
            np.testing.assert_array_equal(data["sample_targets"][i], expected)
            np.testing.assert_array_equal(data["sample_valid_mask"][i], valid)
        np.testing.assert_allclose(data["timestamps"], ids[:, 1] / 50, atol=1e-5, rtol=0)
        chunk_values = int(data["sample_targets"].size)
    pts = json.loads((bundle / "pts/result.json").read_text())
    assert len(pts["records"]) == 45
    pts_count = 0
    for record in pts["records"]:
        expected = record["from_timestamp"] + np.arange(500) / 50
        np.testing.assert_allclose(record["pts_seconds"], expected, atol=1e-5, rtol=0)
        assert record["grid_passed"] and record["duplicate_pts"] == 0
        pts_count += len(expected)
    pixels = json.loads((bundle / "pixels/result.json").read_text())
    with np.load(bundle / "pixels/images.npz", allow_pickle=False) as images:
        assert len(images.files) == len(pixels["records"]) == 135
        for record in pixels["records"]:
            value = images[record["array_key"]]
            assert value.dtype == np.uint8 and value.shape == (3, 480, 640)
            assert hashlib.sha256(value.tobytes()).hexdigest() == record["sha256"]
            assert record["sha256"] == record["prior_native_sha256"]
    report = json.loads((bundle / "supervision/result.json").read_text())
    groups = json.loads((bundle / "data/result.json").read_text())["groups"]
    with np.load(bundle / "supervision/arrays.npz", allow_pickle=False) as arrays:
        episodes, target = arrays["episodes"], arrays["projected_targets"]
        scaled = arrays["coordinates"] / [640, 480, 640, 480]
        expected_neighbors, predicted, means, medians = [], [], [], []
        for query in range(45):
            allowed = [i for i in range(40) if i != query]
            distances = {i: float(sum((scaled[i] - scaled[query]) ** 2)) for i in allowed}
            selected = sorted(allowed, key=lambda i: (distances[i], int(episodes[i])))[:3]
            expected_neighbors.append(selected)
            predicted.append(sum(target[i] for i in selected) / 3)
            means.append(sum(target[i] for i in allowed) / len(allowed))
            medians.append(np.median(target[allowed], axis=0))
        np.testing.assert_array_equal(arrays["neighbors"], expected_neighbors)
        np.testing.assert_allclose(
            arrays["geometry_3nn_predictions"], predicted, atol=1e-12, rtol=0
        )
        np.testing.assert_allclose(arrays["train_mean"], means, atol=1e-12, rtol=0)
        np.testing.assert_allclose(arrays["train_median"], medians, atol=1e-12, rtol=0)
        count = 0
        for row in report["records"]:
            begin = row["frame"]
            end = begin + (1 if row["window"] == "first" else 50)
            cohort = range(40) if row["split"] == "train_loo" else range(40, 45)
            dimensions = groups[row["group"]]
            for field, predictions in (
                ("geometry_3nn_mae", predicted),
                ("train_mean_mae", means),
                ("train_median_mae", medians),
            ):
                values = []
                for i in cohort:
                    residual = (
                        predictions[i][begin:end, dimensions] - target[i, begin:end][:, dimensions]
                    )
                    expected = float(np.abs(residual).mean())
                    assert abs(row[field]["per_episode"][str(int(episodes[i]))] - expected) <= 1e-12
                    values.append(expected)
                    count += 1
                assert abs(row[field]["mean"] - sum(values) / len(values)) <= 1e-12
    provenance_rows = 0
    exposure_checks = 0
    coverage = {}
    if amendment is not None:
        metadata = json.loads((amendment / "result.json").read_text())
        assert (
            hashlib.sha256(schedule_path.read_bytes()).hexdigest()
            == metadata["training_schedule_sha256"]
        )
        assert (
            hashlib.sha256(ingress_path.read_bytes()).hexdigest()
            == metadata["training_ingress_sha256"]
        )
        schedule = json.loads(schedule_path.read_text())["sample_identities"]
        ingress = json.loads(ingress_path.read_text())
        assert [[r["episode"], r["frame"]] for r in ingress["records"]] == schedule
        assert ingress["optimizer_steps"] == 5000
        expected_seen, expected_counts = [], []
        row_by_pair = {tuple(pair): i for i, pair in enumerate(ids.tolist())}
        for stop in (10000, 20000):
            seen_array = np.zeros(22500, dtype=bool)
            count_array = np.zeros(22500, dtype=np.int64)
            for episode, frame in schedule[:stop]:
                seen_array[row_by_pair[episode, frame]] = True
                for slot in range(min(50, 500 - frame)):
                    count_array[row_by_pair[episode, frame + slot]] += 1
            expected_seen.append(seen_array)
            expected_counts.append(count_array)
        with np.load(amendment / "arrays.npz", allow_pickle=False) as amended:
            np.testing.assert_array_equal(amended["raw_actions"], raw)
            np.testing.assert_array_equal(amended["identities"], ids)
            np.testing.assert_array_equal(amended["input_seen_at_checkpoint"], expected_seen)
            np.testing.assert_array_equal(
                amended["valid_target_supervision_count_at_checkpoint"], expected_counts
            )
            exposure_checks = int(np.asarray(expected_counts).size * 2)
        sources = metadata["source_files_sha256"]
        coverage["training_exposure"] = metadata["training_exposure"]
        coverage["seen_main_grid_by_offset"] = {
            str(step): {
                str(frame): int(seen_array[ids[:, 1] == frame].sum())
                for frame in (0, 125, 250, 375)
            }
            for step, seen_array in zip((2500, 5000), expected_seen, strict=True)
        }
        coverage["projection_changed_rows"] = int(np.any(raw != projected, axis=1).sum())
        coverage["command_aperture_crossings"] = {
            side: {
                "total": sum(
                    len(r["crossing_frames"])
                    for r in metadata["command_events"]
                    if r["side"] == side
                ),
                "episodes_with_crossing": sum(
                    bool(r["crossing_frames"])
                    for r in metadata["command_events"]
                    if r["side"] == side
                ),
                "meaning": "command aperture crossing 0.5; not physical task phases",
            }
            for side in ("left", "right")
        }
        seen = {"2500": 0, "5000": 0}
        with (amendment / "samples.jsonl").open() as stream:
            for i, line in enumerate(stream):
                row = json.loads(line)
                assert [row["episode"], row["frame"]] == ids[i].tolist()
                assert row["source_parquet_sha256"] == sources[row["source_parquet"]]
                for j, step in enumerate(seen):
                    seen[step] += int(row["input_seen_at_checkpoint"][step])
                    assert row["input_seen_at_checkpoint"][step] == bool(expected_seen[j][i])
                    assert row["valid_target_supervision_count_at_checkpoint"][step] == int(
                        expected_counts[j][i]
                    )
                provenance_rows += 1
        assert provenance_rows == 22500 and seen == {"2500": 10000, "5000": 20000}
    if decoded_path is not None:
        assert hashlib.sha256(decoded_path.read_bytes()).hexdigest() == (
            "e6ada1fc54fdb82d4ef2508b5df9f6c2b6747a2578012962ac6512da92d9cbb1"
        )
        decoded = json.loads(decoded_path.read_text())
        assert sorted((r["episode"], r["frame"]) for r in decoded) == sorted(
            map(tuple, ids.tolist())
        )
        training = set(map(int, ids[:20000, 0]))
        coverage["decoded_rgb_unique_hashes"] = {
            split: len({r["sha256"] for r in decoded if (r["episode"] in training) == is_train})
            for split, is_train in (("train", True), ("development", False))
        }
        coverage["frame0_unique_rgb_hashes"] = len(
            {r["sha256"] for r in decoded if r["frame"] == 0}
        )
        coverage["pixel_uniqueness_is_not_scene_independence"] = True
    return {
        "status": "passed",
        "manifest_members": checked_files,
        "raw_action_values": int(raw.size),
        "sample_chunk_values": chunk_values,
        "video_timestamps": pts_count,
        "independent_image_hashes": 135,
        "supervision_prediction_values": int(np.asarray(predicted).size),
        "per_episode_metrics": count,
        "precise_provenance_rows": provenance_rows,
        "exposure_array_values": exposure_checks,
        "data_descriptives": coverage,
        "model_loaded": False,
        "optimizer_steps": 0,
        "gate_measured": False,
        "physical_alignment": "not established",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-amendment", type=Path)
    parser.add_argument("--training-schedule", type=Path)
    parser.add_argument("--training-ingress", type=Path)
    parser.add_argument("--decoded-samples", type=Path)
    args = parser.parse_args()
    result = verify(
        args.input,
        args.data_amendment,
        args.training_schedule,
        args.training_ingress,
        args.decoded_samples,
    )
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
