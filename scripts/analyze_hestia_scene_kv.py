"""Independent saved-output interpretation of registered same-checkpoint K/V ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from scripts.analyze_hestia_root_evidence import factorial, summarize
from scripts.hestia_scene_kv import CONDITIONS, condition_spec


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def scene_terms(prediction, target, train):
    p, y, t = (np.asarray(x, dtype=np.float64) for x in (prediction, target, train))
    require(p.ndim == 4 and p.shape[1:] == y.shape and t.shape[1:] == y.shape[1:], "Scene shape")
    require(all(np.isfinite(x).all() for x in (p, y, t)), "Nonfinite scene arrays")
    centered_p, centered_y = p - p.mean(1, keepdims=True), y - y.mean(0)
    mean = t.mean(0)
    bias = ((p.mean(1) - y.mean(0)) ** 2).mean((1, 2)) - ((mean - y.mean(0)) ** 2).mean()
    variance = (centered_p**2).mean((1, 2, 3))
    covariance_term = -2 * (centered_p * centered_y).mean((1, 2, 3))
    gap = ((p - y) ** 2).mean((1, 2, 3)) - ((mean - y) ** 2).mean()
    require(
        np.allclose(bias + variance + covariance_term, gap, atol=1e-12, rtol=1e-12),
        "MSE decomposition",
    )
    return {
        "bias": bias.tolist(),
        "variance": variance.tolist(),
        "minus_twice_covariance": covariance_term.tolist(),
        "gap_to_train_mean": gap.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    args = parser.parse_args()
    require(
        os.environ.get("HF_HUB_OFFLINE") == "1" and Path("/.dockerenv").exists(),
        "Offline container required",
    )
    require(not args.output.exists(), "Preserve prior analysis")
    template = read(args.template)
    manifest = read(args.root / "handoff-manifest.json")
    for name, entry in manifest["files"].items():
        path = args.root / name
        require(
            path.resolve().is_relative_to(args.root.resolve()) and not path.is_symlink(),
            "Unsafe path",
        )
        require(
            path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"], "Source drift"
        )
    worker = read(args.root / "worker-exited.json")
    require(
        worker["completed_conditions"] == list(CONDITIONS) and worker["error"] is None,
        "Incomplete CUDA run",
    )
    require(worker["optimizer_steps"] == 0, "Optimizer scope")
    registration = read(args.root / "registration.json")
    require(registration["template_sha256"] == sha(args.template), "Registration differs")
    metadata_path = Path(template["historical_bundle"]) / "manifest.json"
    require(sha(metadata_path) == template["sha256"][metadata_path.as_posix()], "Metadata drift")
    meta = read(metadata_path)["metadata"]
    require(
        meta["episodes"] == template["episodes"] and not meta["hidden_test_loaded"],
        "Episode metadata",
    )
    rows = {
        name: [meta["episodes"].index(i) for i in meta["views"][name]]
        for name in ("train40", "dev5")
    }
    require(
        rows["train40"] == list(range(40)) and rows["dev5"] == list(range(40, 45)), "Split layout"
    )
    arrays, records = {}, {}
    for condition in CONDITIONS:
        record = records[condition] = read(args.root / condition / "result.json")
        require(
            record["condition"] == condition
            and record["all_parameters_unchanged"]
            and record["raw_images_exact"],
            "Collection integrity",
        )
        require(
            record["model_forwards"] == 180
            and record["optimizer_steps"] == 0
            and not record["hidden_loaded"],
            "Collection scope",
        )
        path = args.root / condition / "arrays.npz"
        require(record["array_sha256"] == sha(path), "Array receipt drift")
        with np.load(path, allow_pickle=False) as saved:
            arrays[condition] = {key: saved[key] for key in saved.files}
        a = arrays[condition]
        for space in ("standard", "normalized"):
            require(
                a[space + "_predictions"].shape == (4, 45, 50, len(meta["dimensions"])),
                "Array layout",
            )
        require(all(np.isfinite(x).all() for x in a.values()), "Nonfinite output")
        for key in ("standard_targets", "normalized_targets", "noise"):
            require(np.array_equal(a[key], arrays[CONDITIONS[0]][key]), "Unpaired conditions")
        base, mode = condition_spec(condition)
        if mode in ("native", "self"):
            if mode == "native":
                reference = Path(template["reference_endpoints"][str(base)])
                require(
                    sha(reference) == template["sha256"][reference.as_posix()],
                    "Original reference drift",
                )
                with np.load(reference, allow_pickle=False) as old:
                    for key in ("normalized_predictions", "standard_predictions"):
                        require(np.array_equal(a[key], old[key]), "Native control not exact")
            else:
                for key in ("normalized_predictions", "standard_predictions"):
                    require(
                        np.array_equal(a[key], arrays[f"base{base}_native"][key]),
                        "Self copy not exact",
                    )
    groups = {}
    for side in ("left", "right"):
        for unit, kind in (("radian", "joint"), ("normalized", "gripper")):
            groups[f"{side}_{kind}"] = [
                i
                for i, d in enumerate(meta["dimensions"])
                if d["name"].startswith(side + "_") and d["unit"] == unit
            ]
    result = {
        "id": template["id"],
        "model_forwards": 0,
        "historical_cuda_forwards_verified": 1800,
        "source_manifest_sha256": sha(args.root / "handoff-manifest.json"),
        "template_sha256": sha(args.template),
        "episodes": meta["views"],
        "metrics": {},
        "m2_complete": False,
        "gate3": "not measured",
        "gate4": "not measured",
        "hidden_loaded": False,
    }
    for base in (640, 1280):
        for space in ("standard", "normalized"):
            target = arrays[f"base{base}_native"][space + "_targets"].astype(np.float64)
            for view, indices in rows.items():
                for window, (start, stop) in {"full": (0, 50), "first": (0, 1)}.items():
                    for group, dims in groups.items():
                        y = target[indices, start:stop][..., dims]
                        train = target[rows["train40"], start:stop][..., dims]
                        pred = {
                            mode: arrays[f"base{base}_{mode}"][space + "_predictions"].astype(
                                np.float64
                            )[:, indices, start:stop][..., dims]
                            for mode in ("native", "self", "kmean", "vmean", "kvmean")
                        }
                        terms = {mode: scene_terms(p, y, train) for mode, p in pred.items()}
                        for metric in ("mae", "mse"):

                            def error(p):
                                d = p - y
                                return (np.abs(d) if metric == "mae" else d**2).mean((-2, -1))

                            errors = {mode: error(p) for mode, p in pred.items()}
                            constants = {
                                "mean": float(error(train.mean(0)).mean()),
                                "median": float(error(np.median(train, axis=0)).mean()),
                            }
                            result["metrics"][
                                f"{base}/{space}/{view}/{window}/{group}/{metric}"
                            ] = {
                                "errors": {mode: summarize(e) for mode, e in errors.items()},
                                "constants": constants,
                                "gain_from_native": {
                                    mode: summarize(errors["native"] - errors[mode])
                                    for mode in ("kmean", "vmean", "kvmean")
                                },
                                "factorial": factorial(
                                    errors["native"],
                                    errors["kmean"],
                                    errors["vmean"],
                                    errors["kvmean"],
                                ),
                                "beats_both_constants": {
                                    mode: (e.mean(1) < min(constants.values())).tolist()
                                    for mode, e in errors.items()
                                },
                                "mse_scene_decomposition": terms if metric == "mse" else None,
                            }
    result["primary_640_consistent_improvement"] = {
        mode: all(
            all(
                x > 0
                for x in result["metrics"][f"640/standard/dev5/full/left_joint/{metric}"][
                    "gain_from_native"
                ][mode]["by_noise"]
            )
            for metric in ("mae", "mse")
        )
        for mode in ("kmean", "vmean", "kvmean")
    }
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "primary": result["primary_640_consistent_improvement"],
                "metric_groups": len(result["metrics"]),
            }
        )
    )


if __name__ == "__main__":
    main()
