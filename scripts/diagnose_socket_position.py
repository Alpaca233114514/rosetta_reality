"""Read frozen KV features into a fixed blue-socket pixel-position control."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from scripts.diagnose_kv_regularization import compare, digest


def locate_blue(image):
    """Fixed color predicate; reject empty, dispersed or threshold-unstable masks."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected HWC uint8 RGB.")
    rgb = image.astype(np.int16)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    positions = []
    bounds = []
    counts = []
    for minimum, difference in ((80, 40), (100, 60)):
        mask = (blue >= minimum) & (blue - red >= difference) & (blue - green >= difference)
        rows, cols = np.nonzero(mask)
        if not 40 <= len(rows) <= 10000:
            raise ValueError("Blue-mask area outside the fixed extraction contract.")
        box = [int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())]
        if not (4 <= box[2] - box[0] <= 180 and 4 <= box[3] - box[1] <= 180):
            raise ValueError("Blue-mask extent outside the fixed extraction contract.")
        positions.append(np.array([cols.mean(), rows.mean()]))
        bounds.append(box)
        counts.append(int(len(rows)))
    shift = float(np.max(np.abs(positions[0] - positions[1])))
    if shift > 2:
        raise ValueError("Blue centroid is unstable across the two fixed thresholds.")
    return positions[0], {"bbox": bounds[0], "areas": counts, "threshold_shift_pixels": shift}


def collect(plan, output):
    """Only the registered frame-zero samples; no policy construction or forward."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from PIL import Image, ImageDraw

    from rosetta_reality.vla.vision_diagnostics import load_frame_zero_context

    context = load_frame_zero_context(Path.cwd(), "non_hidden")
    if context["episodes"] != plan["episodes"]:
        raise ValueError("Registered split/order differs from local context.")
    if digest(context["root"] / "manifest.json") != plan["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest identity drift.")
    reference = json.loads(Path(plan["input_preflight"]).read_text())
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
    positions, records = [], []
    sheet = None
    for index, episode in enumerate(context["episodes"]):
        sample = dataset[dataset.absolute_to_relative_idx[int(starts[episode])]]
        chw = sample[cfg.cameras["top"]].contiguous().numpy()
        if (
            int(sample[cfg.fields.episode_index]) != episode
            or int(sample[cfg.fields.frame_index]) != 0
            or hashlib.sha256(chw.tobytes()).hexdigest() != reference["image_sha256"][index]
            or not np.array_equal(sample[cfg.fields.state].numpy(), context["states"][index])
            or not np.array_equal(sample[cfg.fields.action].numpy(), context["actions"][index])
        ):
            raise ValueError("Image/state/action/frame identity differs from the KV extraction.")
        rgb = chw.transpose(1, 2, 0)
        position, record = locate_blue(rgb)
        positions.append(position)
        records.append({"episode": episode, "centroid_pixels": position.tolist(), **record})
        # Inspection artifact only: full frame, raw-coordinate box and centroid.
        frame = Image.fromarray(rgb)
        draw = ImageDraw.Draw(frame)
        draw.rectangle(record["bbox"], outline="yellow", width=2)
        x, y = position
        draw.line((x - 7, y, x + 7, y), fill="white", width=2)
        draw.line((x, y - 7, x, y + 7), fill="white", width=2)
        if index % 15 == 0:
            sheet = Image.new("RGB", (1600, 810), "#171717")
        col, row = (index % 15) % 5, (index % 15) // 5
        sheet.paste(frame.resize((320, 240)), (col * 320, row * 270))
        ImageDraw.Draw(sheet).text(
            (col * 320 + 8, row * 270 + 245), f"episode {episode} | frame 0", fill="white"
        )
        if index % 15 == 14:
            sheet.save(output / f"position-check-{index // 15 + 1}.png")
    return np.stack(positions), records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8-sig"))
    if (
        os.environ.get("ROSETTA_CONTAINER_IMAGE_ID") != plan["runtime"]["image"]
        or os.environ.get("HF_HUB_OFFLINE") != "1"
    ):
        raise ValueError("Registered offline container required.")
    for path, sha in plan["input_and_code_sha256"].items():
        if digest(path) != sha:
            raise ValueError(f"Identity drift: {path}")
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    positions, records = collect(plan, output)
    with np.load(plan["features"], allow_pickle=False) as saved:
        if saved["episodes"].tolist() != plan["episodes"]:
            raise ValueError("Saved KV row order differs.")
        features = saved["early"].copy()
    result, arrays = compare(
        features,
        positions,
        train_count=40,
        groups={"pixel_x": [0], "pixel_y": [1]},
        grids={"position": plan["alphas"]},
        seed=plan["seed"],
    )
    arrays["targets"] = positions
    criteria = {}
    for group in ("pixel_x", "pixel_y"):
        metrics = result["position"]["development"][group]
        nested = result["position"]["nested_train_oof"][group]
        criteria[group] = {
            "development_below_half_both_constants": metrics["mae"]
            < 0.5 * min(metrics["train_mean_baseline_mae"], metrics["train_median_baseline_mae"]),
            "nested_below_half_both_constants": nested["mae"]
            < 0.5 * min(nested["fold_train_mean_mae"], nested["fold_train_median_mae"]),
            "development_positive_image_gain": metrics["paired_mae_gain"] > 1e-8,
        }
    if time.monotonic() - started > plan["runtime"]["maximum_seconds"]:
        raise TimeoutError("Registered budget exceeded.")
    with (output / "position-readout.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(output / "position-readout.npz", allow_pickle=False) as saved:
        assert all(np.array_equal(v, saved[k]) for k, v in arrays.items())
    report = {
        "id": plan["id"],
        "status": "completed",
        "plan_sha256": digest(args.plan),
        "readout": result["position"],
        "extraction": records,
        "criteria": criteria,
        "numeric_control_passed": all(all(v.values()) for v in criteria.values()),
        "visual_extraction_review": "pending",
        "array_reload_exact": True,
        "elapsed_seconds": time.monotonic() - started,
        "new_model_forwards": 0,
        "hidden_rows_materialized": False,
        "development_used_for_selection": False,
        "policy_improvement": "not measured",
        "task_success": "not measured",
        "m2_complete": False,
        "evidence_sha256": {str(p): digest(p) for p in output.iterdir()},
    }
    with (output / "result.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {"status": "completed", "criteria": criteria, "seconds": report["elapsed_seconds"]}
        )
    )


if __name__ == "__main__":
    main()
