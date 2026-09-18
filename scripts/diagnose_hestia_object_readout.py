"""Fixed two-object image-coordinate controls for recovered Hestia predictions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from rosetta_reality.vla import visual_fit as fit
from rosetta_reality.vla.vision_diagnostics import load_frame_zero_context, ridge_predict
from scripts.diagnose_kv_full_chunk import chunk_metrics
from scripts.diagnose_kv_regularization import nested_predictions, select_alpha
from scripts.diagnose_socket_position import locate_blue


def quadratic_coordinates(coordinates):
    x = np.asarray(coordinates, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] not in (2, 4) or not np.isfinite(x).all():
        raise ValueError("Expected finite one/two-object pixel coordinates")
    x = x / np.tile([640, 480], x.shape[1] // 2)
    return np.column_stack(
        [x] + [x[:, i] * x[:, j] for i in range(x.shape[1]) for j in range(i, x.shape[1])]
    )


def extract(plan, output, metadata):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from PIL import Image, ImageDraw

    context = load_frame_zero_context(Path.cwd(), "non_hidden")
    if context["episodes"] != plan["episodes"] or metadata["episodes"] != plan["episodes"]:
        raise ValueError("Sample order differs")
    if fit.file_hash(context["root"] / "manifest.json") != plan["dataset_manifest_sha256"]:
        raise ValueError("Cache differs")
    cfg = context["config"]
    dataset = LeRobotDataset(
        cfg.repo_id,
        root=context["root"],
        episodes=plan["episodes"],
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
    old = json.loads(Path(plan["blue_result"]).read_text())["extraction"]
    rows, coordinates = [], []
    for index, episode in enumerate(plan["episodes"]):
        sample = dataset[dataset.absolute_to_relative_idx[int(starts[episode])]]
        chw = sample[cfg.cameras["top"]].contiguous().numpy()
        import hashlib

        sha = hashlib.sha256(
            b"torch.uint8" + fit.array_hash(chw[None].astype(np.float64)).encode()
        ).hexdigest()
        if sha != metadata["raw_image_hashes"][index] or int(sample[cfg.fields.frame_index]) != 0:
            raise ValueError("Historical C image identity differs")
        rgb = chw.transpose(1, 2, 0)
        blue, b = locate_blue(rgb)
        red, r = locate_blue(rgb[..., ::-1])
        if old[index]["centroid_pixels"] != blue.tolist() or old[index]["episode"] != episode:
            raise ValueError("Blue identity differs")
        coordinates.append([*blue, *red])
        rows.append(
            {
                "episode": episode,
                "blue": {"centroid": blue.tolist(), **b},
                "red": {"centroid": red.tolist(), **r},
            }
        )
        if index % 15 == 0:
            sheet = Image.new("RGB", (1600, 810), "#171717")
        frame = Image.fromarray(rgb)
        draw = ImageDraw.Draw(frame)
        draw.rectangle(b["bbox"], outline="yellow", width=2)
        draw.rectangle(r["bbox"], outline="lime", width=2)
        col, row = index % 5, (index % 15) // 5
        sheet.paste(frame.resize((320, 240)), (col * 320, row * 270))
        ImageDraw.Draw(sheet).text(
            (col * 320 + 8, row * 270 + 245), f"episode {episode} | frame 0", fill="white"
        )
        if index % 15 == 14:
            sheet.save(output / f"objects-{index // 15 + 1}.png")
    with (output / "coordinates.npz").open("xb") as stream:
        np.savez_compressed(stream, coordinates=np.asarray(coordinates))
    (output / "extraction.json").write_text(
        json.dumps({"rows": rows, "candidate_image_hashes_exact": True}, indent=2) + "\n"
    )


def compact(metrics):
    return {
        group: {
            key: value
            for key, value in row.items()
            if key not in ("dimensions", "per_dimension_mae")
        }
        for group, row in metrics.items()
    }


def analyze(plan, output, saved, metadata):
    review = json.loads((output / "visual-review.json").read_text())
    if review["accepted_images"] != 45 or review["coordinates_sha256"] != fit.file_hash(
        output / "coordinates.npz"
    ):
        raise ValueError("Visual review required")
    for name, sha in review["sha256"].items():
        if fit.file_hash(output / name) != sha:
            raise ValueError("Reviewed image/extraction changed")
    with np.load(output / "coordinates.npz", allow_pickle=False) as data:
        coords = data["coordinates"]
    targets = saved["standard_targets"]
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
    for index, d in enumerate(metadata["dimensions"]):
        kind = "joint" if d["unit"] == "radian" else "gripper"
        groups[kind].append(index)
        groups[d["name"].split("_")[0] + "_" + kind].append(index)
    lower = np.asarray([d["minimum"] for d in metadata["dimensions"]])
    upper = np.asarray([d["maximum"] for d in metadata["dimensions"]])
    windows = dict(
        full=(0, 50), first=(0, 1), early=(0, 10), middle=(10, 25), late=(25, 50), last=(49, 50)
    )
    ty = targets[:40].reshape(40, -1)
    records, arrays = {}, {}
    for name, raw in (("blue", coords[:, :2]), ("red", coords[:, 2:]), ("both", coords)):
        x = quadratic_coordinates(raw)
        alpha, scores = select_alpha(x[:40], ty, plan["alphas"], plan["seed"])
        prediction = ridge_predict(x[:40], ty, x, alpha).reshape(targets.shape)
        nested, means, medians, folds = nested_predictions(x[:40], ty, plan["alphas"], plan["seed"])
        nested, means, medians = (a.reshape(targets[:40].shape) for a in (nested, means, medians))
        clipped, nested_clipped = np.clip(prediction, lower, upper), np.clip(nested, lower, upper)
        arrays.update(
            {
                name + "_raw": prediction,
                name + "_clipped": clipped,
                name + "_nested_raw": nested,
                name + "_nested_clipped": nested_clipped,
            }
        )
        record = {
            "alpha": alpha,
            "cv_scores": scores,
            "folds": folds,
            "features": x.shape[1],
            "clip_elements": int(np.count_nonzero(prediction != clipped)),
            "development": {},
            "nested_train": {},
        }
        for window, (start, stop) in windows.items():
            record["development"][window] = compact(
                chunk_metrics(clipped[40:], targets[40:], targets[:40], groups, start, stop)
            )
            record["nested_train"][window] = {}
            for group, dims in groups.items():
                y = targets[:40, start:stop, dims]
                record["nested_train"][window][group] = {
                    key: float(np.abs(a[:, start:stop, dims] - y).mean())
                    for key, a in (
                        ("mae", nested_clipped),
                        ("mean_baseline_mae", means),
                        ("median_baseline_mae", medians),
                    )
                }
        records[name] = record
    native = {
        str(seed): {
            w: compact(
                chunk_metrics(
                    saved["standard_predictions"][i, 40:],
                    targets[40:],
                    targets[:40],
                    groups,
                    start,
                    stop,
                )
            )
            for w, (start, stop) in windows.items()
        }
        for i, seed in enumerate(metadata["noise_conditions"])
    }
    with (output / "readouts.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(output / "readouts.npz", allow_pickle=False) as restored:
        if not all(np.array_equal(restored[k], a) for k, a in arrays.items()):
            raise ValueError("Output reload differs")
    result = {
        "id": plan["id"],
        "records": records,
        "native": native,
        "array_sha256": fit.file_hash(output / "readouts.npz"),
        "array_reload_exact": True,
        "model_forwards": 0,
        "policy_optimizer_steps": 0,
        "hidden_loaded": False,
        "development_selection": False,
    }
    with (output / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                name: {
                    "alpha": row["alpha"],
                    "development_full": {g: row["development"]["full"][g]["mae"] for g in groups},
                }
                for name, row in records.items()
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--stage", choices=("extract", "readout"), required=True)
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
    output = Path(plan["output"])
    if args.stage == "extract":
        output.mkdir(parents=True, exist_ok=False)
        extract(plan, output, metadata)
    else:
        analyze(plan, output, saved, metadata)


if __name__ == "__main__":
    main()
