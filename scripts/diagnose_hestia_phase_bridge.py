"""Test phase sufficiency against native outputs and a common clock-time target."""

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
from scripts.diagnose_hestia_command_pose import CommandKinematics, errors, runtime_identity
from scripts.diagnose_hestia_object_readout import quadratic_coordinates
from scripts.diagnose_hestia_pose_readout import crossfit, describe, pose_metrics
from scripts.diagnose_hestia_scene_phase import first_upcross


def resample(values, locations):
    y, x = np.asarray(values, dtype=float), np.asarray(locations, dtype=float)
    require(y.ndim == 3 and x.ndim == 2 and len(y) == len(x), "Scene/time layout mismatch")
    require(np.isfinite(y).all() and np.isfinite(x).all(), "Nonfinite interpolation")
    require(np.all(x >= 0) and np.all(x <= y.shape[1] - 1), "Unsupported time extrapolation")
    low = np.floor(x).astype(int)
    high = np.minimum(low + 1, y.shape[1] - 1)
    fraction = (x - low)[..., None]
    index = np.arange(len(y))[:, None]
    return y[index, low] * (1 - fraction) + y[index, high] * fraction


def phase_locations(onsets, *, inverse):
    e = np.asarray(onsets, dtype=float)
    require(
        e.ndim == 1 and np.isfinite(e).all() and np.all((e > 0) & (e < 99)),
        "Observed interior onsets required",
    )
    if inverse:
        return np.stack([np.interp(np.arange(50), [0, v, 99], [0, 50, 100]) for v in e])
    return np.stack([np.interp(np.arange(101), [0, 50, 100], [0, v, 99]) for v in e])


def event_window(values, gripper_column):
    a = np.asarray(values, dtype=float)
    require(a.ndim == 3, "Scene/time/action required")
    onset = first_upcross(a[:, :, gripper_column], 0.5)
    require(all(v is not None and v >= 19 for v in onset), "Censored or too-early native event")
    ix = np.asarray(onset)[:, None] + np.arange(-19, 1)
    return a[np.arange(len(a))[:, None], ix], onset


def score(value, target, groups, fk, eps):
    pose, _ = pose_metrics(value, fk.poses(target), fk, eps)
    return {
        "pose": pose,
        "action": {
            g: describe(np.abs(value[..., cols] - target[..., cols]).mean(-1), eps)
            for g, cols in groups.items()
        },
        "first_action": {
            g: describe(np.abs(value[:, :1, cols] - target[:, :1, cols]).mean(-1), eps)
            for g, cols in groups.items()
        },
    }


def run(plan, plan_path):
    started = time.monotonic()
    require(
        os.environ.get("ROSETTA_CONTAINER_IMAGE_ID") == plan["image"]
        and os.environ.get("HF_HUB_OFFLINE") == "1",
        "Pinned offline container required",
    )
    for path, sha in plan["sha256"].items():
        require(file_hash(path) == sha, f"Source drift: {path}")
    require(runtime_identity() == read_json(plan["runtime_identity"]), "Asset/runtime drift")
    meta = read_json(plan["metadata"])["metadata"]
    rows = split_rows(meta)
    require(
        rows["train40"] == list(range(40)) and rows["dev5"] == list(range(40, 45)), "Order drift"
    )
    eps = meta["episodes"]
    fk = CommandKinematics(meta["dimensions"])
    groups = read_json(plan["scene_result"])["groups"]
    groups = {
        key: groups[key] for key in ("left_joint", "right_joint", "left_gripper", "right_gripper")
    }
    with np.load(plan["scene_arrays"], allow_pickle=False) as a:
        y, coords = a["projected_targets"], a["coordinates"]
        require(a["episodes"].tolist() == eps, "Scene order drift")
    with np.load(plan["native_arrays"], allow_pickle=False) as a:
        native, target = a["standard_predictions"], a["standard_targets"]
    require(np.allclose(y[:, :50], target, atol=1e-6, rtol=0), "Native target identity differs")
    require(native.shape == (4, 45, 50, len(meta["dimensions"])), "Native layout differs")
    onset = first_upcross(y[:, :, groups["left_gripper"][0]], 0.5)
    require(all(v is not None for v in onset), "Censored ground truth")
    onset = np.asarray(onset, dtype=float)
    with np.load(plan["readout_arrays"], allow_pickle=False) as a:
        old = {key: a[key] for key in a.files}
    arrays, native_scores, native_onsets = {}, {}, []
    with np.load(plan["aligned_arrays"], allow_pickle=False) as aligned:
        for coord, cut in (("clock", slice(0, 20)), ("event_oracle", slice(1, 21))):
            actual = aligned[coord + "_targets"][:, cut]
            tp = fk.poses(actual[40:])
            conditions = []
            for condition in native:
                if coord == "clock":
                    v = condition[40:, 30:50]
                else:
                    v, predicted_onsets = event_window(condition[40:], groups["left_gripper"][0])
                    native_onsets.append(predicted_onsets)
                conditions.append(v)
            values = np.stack(conditions)
            pe = np.stack([errors(fk.poses(v), tp) for v in values])
            arrays[coord + "_native_pose_errors"] = pe
            arrays[coord + "_native_matched"] = values
            native_scores[coord] = {
                "native": {
                    "pose": {
                        arm: {
                            metric: {
                                "mean": float(pe[:, :, :, side, mi].mean()),
                                "by_noise": pe[:, :, :, side, mi].mean((1, 2)).tolist(),
                                "per_episode": {
                                    str(ep): float(pe[:, j, :, side, mi].mean())
                                    for j, ep in enumerate(eps[40:])
                                },
                            }
                            for mi, metric in enumerate(("position_m", "orientation_rad"))
                        }
                        for side, arm in enumerate(("left", "right"))
                    },
                    "action": {
                        g: float(np.abs(values[..., cols] - actual[40:][..., cols][None]).mean())
                        for g, cols in groups.items()
                    },
                }
            }
            # Explicit staged indexing avoids moving the action axis before the scene axis.
            for method in ("ridge", "neighbor", "constant_mean", "constant_median"):
                native_scores[coord][method] = score(
                    old[coord + "_" + method][:, cut], actual, groups, fk, eps
                )
    allowed = [np.asarray([j for j in range(40) if j != i]) for i in range(45)]
    canonical = resample(y, phase_locations(onset, inverse=False))
    x = quadratic_coordinates(coords)
    raw, folds = crossfit(x, canonical, 40, groups, plan["alphas"], plan["seed"])
    lower = [d["minimum"] for d in meta["dimensions"]]
    upper = [d["maximum"] for d in meta["dimensions"]]
    prediction = np.clip(raw, lower, upper)
    methods = {
        "ridge": prediction,
        "constant_mean": np.stack([canonical[d].mean(0) for d in allowed]),
        "constant_median": np.stack([np.median(canonical[d], axis=0) for d in allowed]),
    }
    anchors = {
        "oracle": onset,
        "estimated": old["onset_ridge"][:, 0, 0],
        "constant": np.array([onset[d].mean() for d in allowed]),
    }
    restored_scores = {}
    for label, anchor in anchors.items():
        locations = phase_locations(anchor, inverse=True)
        restored_scores[label] = {}
        for method, v in methods.items():
            restored = resample(v, locations)
            arrays[label + "_" + method] = restored
            restored_scores[label][method] = score(restored, target, groups, fk, eps)
    controls = {
        "direct_ridge": old["full_ridge"],
        "clock_mean": np.stack([target[d].mean(0) for d in allowed]),
        "clock_median": np.stack([np.median(target[d], axis=0) for d in allowed]),
    }
    control_scores = {k: score(v, target, groups, fk, eps) for k, v in controls.items()}
    roundtrip = resample(canonical, phase_locations(onset, inverse=True))
    arrays.update(
        canonical=canonical,
        canonical_raw=raw,
        roundtrip=roundtrip,
        onsets=onset,
        estimated_onsets=anchors["estimated"],
        constant_onsets=anchors["constant"],
    )
    out = Path(plan["output"])
    require(out.is_dir() and not any(out.iterdir()), "Fresh empty output required")
    with (out / "arrays.npz").open("xb") as f:
        np.savez_compressed(f, **arrays)
    with np.load(out / "arrays.npz", allow_pickle=False) as saved:
        require(
            set(saved.files) == set(arrays)
            and all(np.array_equal(saved[k], v) for k, v in arrays.items()),
            "Reload differs",
        )
    elapsed = time.monotonic() - started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    require(
        elapsed < plan["maximum_seconds"] and rss < plan["maximum_rss_bytes"], "Resource budget"
    )
    for path, sha in plan["sha256"].items():
        require(file_hash(path) == sha, f"Post-run drift: {path}")
    r = {
        "id": plan["id"],
        "status": "completed",
        "plan_sha256": file_hash(plan_path),
        "array_sha256": file_hash(out / "arrays.npz"),
        "array_reload_exact": True,
        "native": native_scores,
        "native_left_onsets": native_onsets,
        "restored": restored_scores,
        "controls": control_scores,
        "folds": folds,
        "clipped_elements": int(np.count_nonzero(raw != prediction)),
        "roundtrip": score(roundtrip, target, groups, fk, eps),
        "seconds": elapsed,
        "peak_rss_bytes": rss,
        "policy_forwards": 0,
        "policy_optimizer_steps": 0,
        "raw_rows_materialized": 0,
        "hidden_loaded": False,
        "simulation_steps": 0,
        "ssh_used": False,
        "unique_root_cause_established": False,
        "gate3": "not measured",
        "gate4": "not measured",
        "m2_complete": False,
    }
    with (out / "result.json").open("x") as f:
        json.dump(r, f, indent=2, allow_nan=False)
        f.write("\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in r.items()
                if k not in ("native", "restored", "controls", "folds", "roundtrip")
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    run(read_json(args.plan), args.plan)
