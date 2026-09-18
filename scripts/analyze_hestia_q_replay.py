"""Offline action-effect analysis; Q replay is a diagnostic, never a deployable policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.hestia_q_replay import CONDITIONS, MODES, RUN_ID, validate_protocol
from scripts.hestia_scene_kv import file_hash


def analyze(root, historical):
    metadata = json.loads((historical / "manifest.json").read_text())["metadata"]
    dims = metadata["dimensions"]
    groups = {}
    for side in ("left", "right"):
        for kind in ("joint", "gripper"):
            groups[f"{side}_{kind}"] = [
                i
                for i, d in enumerate(dims)
                if side in d["name"] and (d["unit"] == "radian") == (kind == "joint")
            ]
    if any(not g for g in groups.values()):
        raise ValueError("Action dimension semantics unavailable")
    result = {
        "id": RUN_ID,
        "metrics": {},
        "primary": {},
        "source_sha256": {},
        "groups": groups,
        "hidden_loaded": False,
        "m2_complete": False,
        "gate3": "not measured",
        "gate4": "not measured",
    }
    for base in (640, 1280):
        folder = root / f"base{base}"
        report = json.loads((folder / "result.json").read_text())
        if (
            report["model_forwards"] != 1080
            or report["exact_control_forwards"] != 900
            or not report["all_parameters_unchanged"]
            or report["optimizer_steps"] != 0
        ):
            raise ValueError("Incomplete native/replay evidence")
        if file_hash(folder / "arrays.npz") != report["array_sha256"]:
            raise ValueError("Array identity changed")
        result["source_sha256"][str(base)] = report["array_sha256"]
        with np.load(folder / "arrays.npz", allow_pickle=False) as data:
            for space in ("normalized", "standard"):
                target = data[f"{space}_targets"]
                if target.shape[:2] != (45, 50) or not np.isfinite(target).all():
                    raise ValueError("Target shape or finite contract differs")
                for mode in MODES:
                    prediction = data[f"{mode}_{space}_predictions"]
                    if prediction.shape != (4, *target.shape) or not np.isfinite(prediction).all():
                        raise ValueError("Prediction shape or finite contract differs")
            for space in ("normalized", "standard"):
                y = data[f"{space}_targets"]
                for split, rows in (("train40", np.arange(40)), ("dev5", np.arange(40, 45))):
                    for window, stop in (("full", 50), ("first", 1)):
                        for group, indices in groups.items():
                            target = np.take(y[rows, :stop], indices, axis=-1)
                            t = np.take(y[:40, :stop], indices, axis=-1)
                            constants = {}
                            for method in ("mean", "median"):
                                fn = np.mean if method == "mean" else np.median
                                constants[method] = (
                                    np.stack([fn(np.delete(t, r, axis=0), axis=0) for r in rows])
                                    if split == "train40"
                                    else np.broadcast_to(fn(t, axis=0), target.shape)
                                )
                            for metric in ("mae", "mse"):
                                transform = np.abs if metric == "mae" else np.square
                                errors = {
                                    mode: transform(
                                        np.take(
                                            data[f"{mode}_{space}_predictions"][:, rows, :stop],
                                            indices,
                                            axis=-1,
                                        )
                                        - target
                                    ).mean((-2, -1))
                                    for mode in MODES
                                }
                                entry = {
                                    "by_noise_scene": {m: e.tolist() for m, e in errors.items()},
                                    "by_noise": {m: e.mean(1).tolist() for m, e in errors.items()},
                                    "constants": {
                                        m: float(transform(c - target).mean())
                                        for m, c in constants.items()
                                    },
                                    "gain": {
                                        m: (errors["native"] - e).mean(1).tolist()
                                        for m, e in errors.items()
                                    },
                                    "leave_one_scene_out_gain": {
                                        m: np.stack(
                                            [
                                                np.delete(errors["native"] - e, r, axis=1).mean(1)
                                                for r in range(len(rows))
                                            ]
                                        ).tolist()
                                        for m, e in errors.items()
                                    },
                                    "feedback_gain": (errors["kmean_qnative"] - errors["kmean"])
                                    .mean(1)
                                    .tolist(),
                                }
                                result["metrics"][
                                    f"{base}/{space}/{split}/{window}/{group}/{metric}"
                                ] = entry
                            p = {
                                m: np.take(
                                    data[f"{m}_{space}_predictions"][:, rows, :stop],
                                    indices,
                                    axis=-1,
                                )
                                for m in MODES
                            }
                            if split == "dev5":
                                terms = {}
                                for m, v in p.items():
                                    cy = target - target.mean(0)
                                    cp = v - v.mean(1, keepdims=True)
                                    terms[m] = {
                                        "bias": ((v.mean(1) - target.mean(0)) ** 2)
                                        .mean((1, 2))
                                        .tolist(),
                                        "variance": (cp**2).mean((1, 2, 3)).tolist(),
                                        "minus_twice_covariance": (-2 * cp * cy)
                                        .mean((1, 2, 3))
                                        .tolist(),
                                    }
                                result["metrics"][f"{base}/{space}/{split}/{window}/{group}/mse"][
                                    "decomposition"
                                ] = terms
        primary = [
            result["metrics"][f"{base}/standard/dev5/full/left_joint/{metric}"]
            for metric in ("mae", "mse")
        ]
        result["primary"][str(base)] = {
            "fixed_q_consistent_improvement": all(
                np.all(np.array(e["gain"]["kmean_qnative"]) > 0)
                and np.all(np.array(e["leave_one_scene_out_gain"]["kmean_qnative"]) > 0)
                for e in primary
            ),
            "live_q_consistent_improvement": all(
                np.all(np.array(e["gain"]["kmean"]) > 0)
                and np.all(np.array(e["leave_one_scene_out_gain"]["kmean"]) > 0)
                for e in primary
            ),
            "feedback_consistently_enhances": all(
                np.all(np.array(e["feedback_gain"]) > 0) for e in primary
            ),
            "feedback_consistently_offsets": all(
                np.all(np.array(e["feedback_gain"]) < 0) for e in primary
            ),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    args = parser.parse_args()
    template = json.loads(args.template.read_text(encoding="utf-8-sig"))
    validate_protocol(template)
    registration = json.loads((args.root / "registration.json").read_text())
    worker = json.loads((args.root / "worker-exited.json").read_text())
    if (
        registration["template_sha256"] != file_hash(args.template)
        or worker["error"] is not None
        or worker["completed_conditions"] != list(CONDITIONS)
    ):
        raise ValueError("Run registration or completion differs")
    manifest = json.loads((args.root / "handoff-manifest.json").read_text())
    from scripts.verify_hestia_q_replay_results import validate_manifest

    for name, entry in validate_manifest(manifest).items():
        path = args.root / name
        if path.is_symlink() or not path.resolve().is_relative_to(args.root.resolve()):
            raise ValueError("Unsafe evidence path")
        if path.stat().st_size != entry["bytes"] or file_hash(path) != entry["sha256"]:
            raise ValueError("Evidence manifest differs")
    meta_path = args.historical / "manifest.json"
    if file_hash(meta_path) != template["sha256"][meta_path.as_posix()]:
        raise ValueError("Historical metadata seal differs")
    result = analyze(args.root, args.historical)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result["primary"]))


if __name__ == "__main__":
    main()
