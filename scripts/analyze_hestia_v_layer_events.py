"""Censor-aware gripper timing from saved predictions; no model or raw images."""

import json
import os
from pathlib import Path

import numpy as np

from scripts.analyze_hestia_v_layer import IMAGE, OUTPUT, ROOT
from scripts.diagnose_hestia_checkpoint_localization import file_hash, read_json, require
from scripts.hestia_v_layer import CONDITIONS


def first_opening(values, threshold):
    a = np.asarray(values)
    require(a.ndim == 3 and np.isfinite(a).all(), "Finite noise/scene/time required")
    crossed = a >= threshold
    present = crossed.any(-1)
    # -1 is a censoring marker, not an imputed opening time.
    slots = np.where(present, crossed.argmax(-1), -1)
    return slots, crossed[..., 0]


def main():
    require(os.environ.get("ROSETTA_CONTAINER_IMAGE_ID") == IMAGE, "Pinned container required")
    primary = read_json(OUTPUT / "result.json")
    require(
        primary["source_manifest_sha256"] == file_hash(ROOT / "handoff-manifest.json"),
        "Primary drift",
    )
    plan = read_json("reports/training/m2-smolvla-hestia-v-layer-events-plan-2026-09-12.json")
    for name, sha in plan["sha256"].items():
        require(file_hash(name) == sha, "Event source/input drift")
    old_plan = read_json("reports/training/m2-smolvla-hestia-scene-phase-plan-2026-09-12.json")
    require(old_plan["threshold"] == plan["threshold"] == 0.5, "Historical threshold differs")
    meta = read_json("runs/hestia-recovered-20260911-001/C-first/manifest.json")["metadata"]
    with np.load(plan["scene_arrays"], allow_pickle=False) as a:
        target = a["projected_targets"].astype(np.float64)
    previous = Path("runs/hestia-kv-split-recovered-20260912-003/verified")
    arrays = {}
    for name in (*CONDITIONS, "base1280_v640", "base640_v1280"):
        root = ROOT if name in CONDITIONS else previous
        p = root / name / "arrays.npz"
        manifest = read_json(root / "handoff-manifest.json")
        require(
            file_hash(p) == manifest["files"][f"{name}/arrays.npz"]["sha256"], "Saved array drift"
        )
        with np.load(p, allow_pickle=False) as a:
            require(
                np.allclose(a["standard_targets"], target[:, :50], atol=1e-6, rtol=0),
                "Timing target drift",
            )
            native_target = a["standard_targets"].astype(np.float64)
            require(
                np.array_equal(native_target >= 0.5, target[:, :50] >= 0.5), "Threshold label drift"
            )
            arrays[name] = a["standard_predictions"].astype(np.float64)
    shape = (len(meta["noise_conditions"]), *native_target.shape)
    arrays.update(
        train_mean=np.broadcast_to(native_target[:40].mean(0), shape),
        train_median=np.broadcast_to(np.median(native_target[:40], axis=0), shape),
    )
    result = {}
    for arm in ("left", "right"):
        dim = next(i for i, d in enumerate(meta["dimensions"]) if d["name"] == arm + "_gripper")
        truth, truth_initial = first_opening(target[None, ..., dim], plan["threshold"])
        truth = truth[0]
        require(not truth_initial.any(), "Historical target initially open")
        result[arm] = {}
        for split, rows in (("train40", list(range(40))), ("dev5", list(range(40, 45)))):
            records = {}
            t = truth[rows]
            for name, values in arrays.items():
                pred, initial = first_opening(values[..., dim][:, rows], plan["threshold"])
                present = pred >= 0
                early = present & ((t[None] < 0) | (pred < t[None]))
                opens_in_chunk = (t >= 0) & (t < 50)
                missed = opens_in_chunk[None] & ~present
                pairs = present & opens_in_chunk[None] & ~initial
                records[name] = {
                    "opening_slots_by_noise_episode": [
                        [None if v < 0 else int(v) for v in row] for row in pred
                    ],
                    "initially_open_by_noise": initial.sum(1).tolist(),
                    "premature_opening_by_noise": early.sum(1).tolist(),
                    "missed_observed_opening_by_noise": missed.sum(1).tolist(),
                    "observed_pair_count_by_noise": pairs.sum(1).tolist(),
                    "conditional_absolute_timing_error_by_noise": [
                        float(np.abs(pred[n, pairs[n]] - t[pairs[n]]).mean())
                        if pairs[n].any()
                        else None
                        for n in range(len(pred))
                    ],
                }
            result[arm][split] = {
                "episodes": [meta["episodes"][i] for i in rows],
                "target_opening_slots_100": [None if v < 0 else int(v) for v in t],
                "conditions": records,
            }
    out = OUTPUT.parent / "events.json"
    with out.open("x") as f:
        json.dump(
            {
                "threshold": plan["threshold"],
                "prediction_horizon": 50,
                "target_horizon": 100,
                "primary_analysis_sha256": file_hash(OUTPUT / "result.json"),
                "censored_times_imputed": False,
                "results": result,
            },
            f,
            indent=2,
            allow_nan=False,
        )
    print(json.dumps({"status": "passed", "output": str(out)}))


if __name__ == "__main__":
    main()
