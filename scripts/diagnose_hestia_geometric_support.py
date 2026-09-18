"""Check observed socket coverage and fixed geometric nearest-neighbor readouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from rosetta_reality.vla import visual_fit as fit
from scripts.diagnose_kv_full_chunk import chunk_metrics
from scripts.diagnose_socket_position import locate_blue


def convex_hull(points):
    values = sorted(set(map(tuple, np.asarray(points, dtype=np.float64))))

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    lower, upper = [], []
    for target, order in ((lower, values), (upper, reversed(values))):
        for point in order:
            while len(target) >= 2 and cross(target[-2], target[-1], point) <= 0:
                target.pop()
            target.append(point)
    hull = np.asarray(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        raise ValueError("Two-dimensional support hull is degenerate")
    return hull


def inside_hull(points, hull):
    points = np.asarray(points, dtype=np.float64)
    following = np.roll(hull, -1, axis=0)
    edge = following - hull
    offset = points[:, None, :] - hull[None]
    cross = edge[None, :, 0] * offset[..., 1] - edge[None, :, 1] * offset[..., 0]
    return (cross >= -1e-8).all(axis=1)


def nearest_geometry(features, train_count):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all() or not 2 <= train_count < len(x):
        raise ValueError("Finite geometry with separate train/development rows required")
    mean, scale = x[:train_count].mean(0), x[:train_count].std(0)
    constant = scale == 0
    scale = np.where(constant, 1, scale)
    z = (x - mean) / scale
    distance = np.linalg.norm(z[:, None] - z[None, :train_count], axis=2)
    distance[np.arange(train_count), np.arange(train_count)] = np.inf
    donor = distance.argmin(axis=1)
    selected = distance[np.arange(len(x)), donor]
    return (
        donor,
        selected,
        {
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "constant_columns": np.flatnonzero(constant).tolist(),
            "train_loo_distance_q95": float(
                np.quantile(selected[:train_count], 0.95, method="linear")
            ),
        },
    )


def verify_geometry(plan, metadata):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from rosetta_reality.vla.vision_diagnostics import load_frame_zero_context

    context = load_frame_zero_context(Path.cwd(), "non_hidden")
    if context["episodes"] != metadata["episodes"] or context["episodes"] != plan["episodes"]:
        raise ValueError("Sample/split order differs")
    if fit.file_hash(context["root"] / "manifest.json") != plan["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest differs")
    reference = json.loads(Path(plan["input_preflight"]).read_text())
    old = json.loads(Path(plan["position_result"]).read_text())
    cfg = context["config"]
    dataset = LeRobotDataset(
        cfg.repo_id,
        root=context["root"],
        episodes=context["episodes"],
        revision=cfg.revision,
        download_videos=False,
        return_uint8=True,
    )
    starts = dict(
        zip(
            dataset.meta.episodes["episode_index"],
            dataset.meta.episodes["dataset_from_index"],
            strict=True,
        )
    )
    features, shapes = [], []
    for index, episode in enumerate(context["episodes"]):
        sample = dataset[dataset.absolute_to_relative_idx[int(starts[episode])]]
        image = sample[cfg.cameras["top"]].contiguous().numpy()
        if (
            int(sample[cfg.fields.episode_index]) != episode
            or int(sample[cfg.fields.frame_index]) != 0
        ):
            raise ValueError("Wrong image identity")
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("Unexpected raw image contract")
        if hashlib.sha256(image.tobytes()).hexdigest() != reference["image_sha256"][index]:
            raise ValueError("Image differs from local geometry extraction")
        if not np.array_equal(
            sample[cfg.fields.state].numpy(), context["states"][index]
        ) or not np.array_equal(sample[cfg.fields.action].numpy(), context["actions"][index]):
            raise ValueError("Frame-zero numeric context differs")
        matches = []
        for batched in (image[None], image[None, None]):
            value = hashlib.sha256(
                b"torch.uint8" + fit.array_hash(batched.astype(np.float64)).encode()
            ).hexdigest()
            if value == metadata["raw_image_hashes"][index]:
                matches.append(list(batched.shape))
        if len(matches) != 1:
            raise ValueError("Raw image does not match exactly one historical Hestia input layout")
        shapes.append(matches[0])
        position, record = locate_blue(image.transpose(1, 2, 0))
        expected = old["extraction"][index]
        if (
            expected["episode"] != episode
            or position.tolist() != expected["centroid_pixels"]
            or any(record[key] != expected[key] for key in record)
        ):
            raise ValueError("Frozen geometry extraction differs")
        x0, y0, x1, y1 = record["bbox"]
        features.append([*position.tolist(), x1 - x0, y1 - y0, record["areas"][0]])
    if len({tuple(s) for s in shapes}) != 1:
        raise ValueError("Historical raw input layout varies")
    return np.asarray(features), {
        "images_verified": len(features),
        "hestia_raw_shape": shapes[0],
        "both_image_hash_schemes_exact": True,
        "geometry_exact": True,
        "separate_numeric_context_preflight": True,
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
        raise ValueError("Registered offline container required")
    for name, sha in plan["sha256"].items():
        if fit.file_hash(Path(name)) != sha:
            raise ValueError(f"Input/code drift: {name}")
    saved, metadata = fit.read_bundle(Path(plan["candidate_bundle"]))
    geometry, identity = verify_geometry(plan, metadata)
    target = saved["standard_targets"]
    loo_mean = (target[:40].sum(0) - target[:40]) / 39
    loo_median = np.stack([np.median(np.delete(target[:40], i, axis=0), axis=0) for i in range(40)])
    hull = convex_hull(geometry[:40, :2])
    inside = inside_hull(geometry[40:, :2], hull)
    box = (
        (geometry[40:, :2] >= geometry[:40, :2].min(0))
        & (geometry[40:, :2] <= geometry[:40, :2].max(0))
    ).all(1)
    groups = {
        "joint": [],
        "gripper": [],
        "left_joint": [],
        "right_joint": [],
        "left_gripper": [],
        "right_gripper": [],
    }
    for index, dimension in enumerate(metadata["dimensions"]):
        kind = "joint" if dimension["unit"] == "radian" else "gripper"
        groups[kind].append(index)
        groups[dimension["name"].split("_")[0] + "_" + kind].append(index)
    windows = dict(
        full=(0, 50), first=(0, 1), early=(0, 10), middle=(10, 25), late=(25, 50), last=(49, 50)
    )
    records, arrays = {}, {"geometry": geometry, "hull": hull}
    for name, features in (("centroid_2d", geometry[:, :2]), ("geometry_proxy_5d", geometry)):
        donors, distances, scale = nearest_geometry(features, 40)
        predictions = target[donors]
        records[name] = {
            "scaling": scale,
            "donor_episodes": [metadata["episodes"][i] for i in donors],
            "distances": distances.tolist(),
            "dev_within_train_q95": (distances[40:] <= scale["train_loo_distance_q95"]).tolist(),
            "development": {},
            "train_loo": {},
        }
        for window, (start, stop) in windows.items():
            records[name]["development"][window] = chunk_metrics(
                predictions[40:], target[40:], target[:40], groups, start, stop
            )
            for group, dims in groups.items():
                pred, y = predictions[:40, start:stop, dims], target[:40, start:stop, dims]
                records[name]["train_loo"].setdefault(window, {})[group] = {
                    "mae": float(np.abs(pred - y).mean()),
                    "mse": float(np.square(pred - y).mean()),
                    "mean_baseline_mae": float(np.abs(loo_mean[:, start:stop, dims] - y).mean()),
                    "median_baseline_mae": float(
                        np.abs(loo_median[:, start:stop, dims] - y).mean()
                    ),
                }
        arrays[name + "_predictions"] = predictions
        arrays[name + "_donors"] = donors
    native = {
        str(seed): {
            window: chunk_metrics(
                saved["standard_predictions"][index, 40:],
                target[40:],
                target[:40],
                groups,
                start,
                stop,
            )
            for window, (start, stop) in windows.items()
        }
        for index, seed in enumerate(metadata["noise_conditions"])
    }
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    with (output / "arrays.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(output / "arrays.npz", allow_pickle=False) as loaded:
        assert all(np.array_equal(value, loaded[key]) for key, value in arrays.items())
    report = {
        "id": plan["id"],
        "status": "completed",
        "plan_sha256": fit.file_hash(args.plan),
        "input_identity": identity,
        "dev_episodes": metadata["views"]["dev5"],
        "dev_in_train_centroid_box": box.tolist(),
        "dev_in_train_centroid_hull": inside.tolist(),
        "nearest_neighbor": records,
        "native_C": native,
        "array_sha256": fit.file_hash(output / "arrays.npz"),
        "array_reload_exact": True,
        "model_forwards": 0,
        "optimizer_steps": 0,
        "hidden_test_loaded": False,
        "m2_complete": False,
    }
    with (output / "result.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "status": "completed",
                "input_identity": identity,
                "dev_in_hull": inside.tolist(),
                "dev_within_nn_q95": {k: v["dev_within_train_q95"] for k, v in records.items()},
            }
        )
    )


if __name__ == "__main__":
    main()
