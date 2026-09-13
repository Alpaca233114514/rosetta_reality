"""Separate native pose error from fixed-coordinate readout-family limitations."""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import numpy as np

from rosetta_reality.vla.vision_diagnostics import ridge_predict
from scripts.diagnose_hestia_checkpoint_localization import (
    file_hash,
    read_json,
    require,
    split_rows,
)
from scripts.diagnose_hestia_command_pose import CommandKinematics, errors, runtime_identity
from scripts.diagnose_hestia_object_readout import quadratic_coordinates
from scripts.diagnose_hestia_scene_phase import first_upcross, geometry_neighbors
from scripts.diagnose_kv_regularization import select_alpha


def crossfit(x, targets, train_count, groups, alphas, seed):
    """Outer LOO; each query's alpha and normalization exclude its target."""
    x, y = np.asarray(x, dtype=float), np.asarray(targets, dtype=float)
    require(x.ndim == 2 and y.ndim == 3 and len(x) == len(y), "Aligned input/target rows required")
    require(np.isfinite(x).all() and np.isfinite(y).all(), "Nonfinite inputs")
    require(10 <= train_count < len(y), "Disjoint training/development required")
    require(
        sorted(d for cols in groups.values() for d in cols) == list(range(y.shape[-1])),
        "Groups must partition output dimensions",
    )
    prediction = np.empty_like(y)
    records = []
    for query in range(train_count + 1):
        held = np.array([query]) if query < train_count else np.arange(train_count, len(y))
        donors = np.array([i for i in range(train_count) if i != query])
        record = {"query_rows": held.tolist(), "donor_rows": donors.tolist(), "groups": {}}
        for name, cols in groups.items():
            target = y[donors][..., cols].reshape(len(donors), -1)
            alpha, scores = select_alpha(
                x[donors], target, alphas, seed + query if query < train_count else seed
            )
            value = ridge_predict(x[donors], target, x[held], alpha)
            for h, q in enumerate(held):
                prediction[q][:, cols] = value[h].reshape(y.shape[1], len(cols))
            record["groups"][name] = {"alpha": alpha, "inner_mae": scores}
        records.append(record)
    return prediction, records


def describe(a, episodes, train_count=40):
    """Keep scene as sampling unit; extra axes are condition/time only."""
    a = np.asarray(a, dtype=float)
    require(a.ndim == 2 and a.shape[0] == len(episodes), "Expected scene by observations")
    require(np.isfinite(a).all(), "Nonfinite errors")
    return {
        "train_loo": float(a[:train_count].mean()),
        "dev5": float(a[train_count:].mean()),
        "dev_per_episode": {
            str(ep): float(row.mean())
            for ep, row in zip(episodes[train_count:], a[train_count:], strict=True)
        },
    }


def pose_metrics(prediction, target_pose, fk, episodes):
    e = errors(fk.poses(prediction), target_pose)
    metrics = {
        arm: {
            metric: describe(e[:, :, side, i], episodes)
            for i, metric in enumerate(("position_m", "orientation_rad"))
        }
        for side, arm in enumerate(("left", "right"))
    }
    return metrics, e


def run(plan, plan_path):
    started = time.monotonic()
    require(
        os.environ.get("ROSETTA_CONTAINER_IMAGE_ID") == plan["image"]
        and os.environ.get("HF_HUB_OFFLINE") == "1",
        "Pinned offline runtime required",
    )
    for path, sha in plan["sha256"].items():
        require(file_hash(path) == sha, f"Input/source drift: {path}")
    require(runtime_identity() == read_json(plan["runtime_identity"]), "Runtime/asset drift")
    meta = read_json(plan["metadata"])["metadata"]
    rows = split_rows(meta)
    require(
        rows["train40"] == list(range(40)) and rows["dev5"] == list(range(40, 45)),
        "Train-first order required",
    )
    eps = meta["episodes"]
    with np.load(plan["scene_arrays"], allow_pickle=False) as a:
        y, coords, recorded, episodes = (
            a[k] for k in ("projected_targets", "coordinates", "neighbors", "episodes")
        )
    require(
        episodes.tolist() == eps and y.shape == (45, 100, len(meta["dimensions"])),
        "Saved scene identity differs",
    )
    _, nearest, allowed = geometry_neighbors(coords, eps, rows["train40"], 3)
    require(np.array_equal(nearest, recorded), "Historical donors differ")
    fk = CommandKinematics(meta["dimensions"])
    lower = np.array([d["minimum"] for d in meta["dimensions"]])
    upper = np.array([d["maximum"] for d in meta["dimensions"]])
    groups = {
        name: [
            i
            for i, d in enumerate(meta["dimensions"])
            if d["name"].startswith(arm + "_") and (d["unit"] == "radian") == joint
        ]
        for arm in ("left", "right")
        for joint in (True, False)
        for name in [arm + ("_joint" if joint else "_gripper")]
    }
    with np.load(plan["native_1280"], allow_pickle=False) as a:
        native_y = a["standard_targets"]
    require(np.allclose(y[:, :50], native_y, atol=1e-6, rtol=0), "Historical target drift")
    targets = {"full": native_y}
    with np.load(plan["aligned_arrays"], allow_pickle=False) as a:
        for coordinate in ("clock", "event_oracle"):
            targets[coordinate] = a[coordinate + "_targets"]
            require(
                np.array_equal(
                    targets[coordinate][nearest].mean(1), a[coordinate + "_predictions"]
                ),
                "Old readout differs",
            )
    x = quadratic_coordinates(coords)
    arrays, metrics, fits = {}, {}, {}
    prior = read_json(plan["pose_result"])
    reproductions = 0
    for coordinate, target in targets.items():
        raw, fitted = crossfit(x, target, 40, groups, plan["alphas"], plan["seed"])
        predicted = np.clip(raw, lower, upper)
        fits[coordinate] = {
            "folds": fitted,
            "clipped_elements": int(np.count_nonzero(raw != predicted)),
        }
        methods = {
            "ridge": predicted,
            "neighbor": target[nearest].mean(1),
            "constant_mean": np.stack([target[d].mean(0) for d in allowed]),
            "constant_median": np.stack([np.median(target[d], axis=0) for d in allowed]),
        }
        true_pose = fk.poses(target)
        metrics[coordinate] = {}
        arrays[coordinate + "_raw_ridge"] = raw
        for method, value in methods.items():
            pm, pe = pose_metrics(value, true_pose, fk, eps)
            record = {"pose": pm, "action": {}}
            for group, cols in groups.items():
                record["action"][group] = describe(
                    np.abs(value[..., cols] - target[..., cols]).mean(-1), eps
                )
                if coordinate == "full":
                    record["action"][group]["first_action"] = describe(
                        np.abs(value[:, :1, cols] - target[:, :1, cols]).mean(-1), eps
                    )
            metrics[coordinate][method] = record
            arrays[coordinate + "_" + method] = value
            arrays[coordinate + "_" + method + "_pose_errors"] = pe
            if coordinate != "full" and method != "ridge":
                old_method = {
                    "neighbor": "neighbor_joint_mean",
                    "constant_mean": "constant_joint_mean",
                    "constant_median": "constant_joint_median",
                }[method]
                for side, arm in enumerate(("left", "right")):
                    for mi, old_metric in enumerate(("position_l2_m", "orientation_geodesic_rad")):
                        for view, indices in rows.items():
                            old = prior["metrics"][coordinate][old_method][view][arm][old_metric][
                                "mean"
                            ]
                            require(
                                abs(float(pe[indices, :, side, mi].mean()) - old) < 1e-12,
                                "Historical pose metric drift",
                            )
                            reproductions += 1
    onset = np.stack(
        [first_upcross(y[:, :, groups[arm + "_gripper"][0]], 0.5) for arm in ("left", "right")],
        axis=-1,
    )
    require(onset.dtype != object, "Censored onset; no imputation allowed")
    onset = onset[:, None, :].astype(float)
    pred, fits["onset"] = crossfit(
        x, onset, 40, {"left": [0], "right": [1]}, plan["alphas"], plan["seed"]
    )
    metrics["onset"] = {}
    for name, v in (
        ("ridge", pred),
        ("neighbor", onset[nearest].mean(1)),
        ("constant_mean", np.stack([onset[d].mean(0) for d in allowed])),
        ("constant_median", np.stack([np.median(onset[d], axis=0) for d in allowed])),
    ):
        metrics["onset"][name] = {
            arm: describe(np.abs(v[:, :, a] - onset[:, :, a]), eps)
            for a, arm in enumerate(("left", "right"))
        }
        arrays["onset_" + name] = v
    # Native predictions never share an average-before-FK path with label readouts.
    native = {}
    native_errors = {}
    true_pose = fk.poses(native_y)
    for step in (640, 1280):
        with np.load(plan[f"native_{step}"], allow_pickle=False) as a:
            p = a["standard_predictions"]
            require(np.array_equal(a["standard_targets"], native_y), "Native targets differ")
        require(p.shape == (4, 45, 50, len(meta["dimensions"])), "Native condition layout differs")
        e = np.stack([errors(fk.poses(condition), true_pose) for condition in p])
        native_errors[step] = e
        arrays[f"native_{step}_pose_errors"] = e
        native[str(step)] = {}
        for window, (start, stop) in plan["windows"].items():
            native[str(step)][window] = {
                arm: {
                    metric: {
                        "train_fit": float(e[:, :40, start:stop, side, mi].mean()),
                        "dev5": float(e[:, 40:, start:stop, side, mi].mean()),
                        "dev_by_noise": e[:, 40:, start:stop, side, mi].mean((1, 2)).tolist(),
                        "dev_per_episode": {
                            str(ep): float(e[:, 40 + j, start:stop, side, mi].mean())
                            for j, ep in enumerate(eps[40:])
                        },
                    }
                    for mi, metric in enumerate(("position_m", "orientation_rad"))
                }
                for side, arm in enumerate(("left", "right"))
            }
    delta = native_errors[1280][:, 40:] - native_errors[640][:, 40:]
    contributions = delta.mean((0, 2)) / 5
    require(
        np.allclose(contributions.sum(0), delta.mean((0, 1, 2)), atol=1e-14, rtol=0),
        "Scene contribution identity failed",
    )
    arrays["native_pose_delta_contribution"] = contributions
    # Compare first-action pose as well as action for deployable-coordinate readouts.
    for method in ("ridge", "neighbor", "constant_mean", "constant_median"):
        metrics["full"][method]["first_pose"], _ = pose_metrics(
            arrays["full_" + method][:, :1], tuple(v[:, :1] for v in true_pose), fk, eps
        )
    out = Path(plan["output"])
    require(out.is_dir() and not any(out.iterdir()), "Fresh empty output required")
    with (out / "arrays.npz").open("xb") as f:
        np.savez_compressed(f, **arrays)
    with np.load(out / "arrays.npz", allow_pickle=False) as a:
        require(
            set(a.files) == set(arrays) and all(np.array_equal(a[k], v) for k, v in arrays.items()),
            "Saved output reload differs",
        )
    for path, sha in plan["sha256"].items():
        require(file_hash(path) == sha, f"Post-run source drift: {path}")
    elapsed = time.monotonic() - started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    require(
        elapsed < plan["maximum_seconds"] and rss < plan["maximum_rss_bytes"], "Budget exceeded"
    )
    result = {
        "id": plan["id"],
        "status": "completed",
        "plan_sha256": file_hash(plan_path),
        "array_sha256": file_hash(out / "arrays.npz"),
        "array_reload_exact": True,
        "historical_pose_metrics_reproduced": reproductions,
        "metrics": metrics,
        "native": native,
        "fits": fits,
        "seconds": elapsed,
        "peak_rss_bytes": rss,
        "sanity": fk.sanity,
        "policy_forwards": 0,
        "policy_optimizer_steps": 0,
        "raw_rows_materialized": 0,
        "hidden_loaded": False,
        "ssh_used": False,
        "simulation_steps": 0,
        "checkpoint_selection": False,
        "gate3": "not measured",
        "gate4": "not measured",
        "m2_complete": False,
    }
    with (out / "result.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("metrics", "native", "fits")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    run(read_json(args.plan), args.plan)
