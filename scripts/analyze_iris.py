"""Fixed Iris acceptance; dev remains development, never a closed-loop result."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rosetta_reality.vla.visual_coverage import action_groups, score_view
from scripts.iris_runtime import save, sha, verify_reload


def compare(control, treatment, dimensions):
    if set(control) != set(treatment):
        raise ValueError("Arm array sets differ")
    for key in control:
        if (
            control[key].shape != treatment[key].shape
            or not np.isfinite(control[key]).all()
            or not np.isfinite(treatment[key]).all()
        ):
            raise ValueError("Arm arrays have invalid shape or finite values")
    for key in ("normalized_targets", "standard_targets", "noise", "valid_mask"):
        if not np.array_equal(control[key], treatment[key]):
            raise ValueError("Arms changed target, mask or noise identity")
    targets = control["standard_targets"]
    if targets.shape != (45, 50, len(dimensions)) or control["standard_predictions"].shape != (
        4,
        *targets.shape,
    ):
        raise ValueError("Registered 40+5 / four-noise / 50-slot evaluation required")
    native_groups = action_groups(dimensions)
    groups = {}
    for side in ("left", "right"):
        for kind, source in (("joint", "joint_radian"), ("gripper", "gripper_normalized")):
            groups[side + "_" + kind] = [
                i for i in native_groups[source] if dimensions[i]["name"].startswith(side + "_")
            ]
    if any(not value for value in groups.values()):
        raise ValueError("Missing physical action group")
    fit = {}
    for arm, arrays in (("control", control), ("treatment", treatment)):
        fit[arm] = [
            score_view(
                arrays["normalized_predictions"][noise, :40],
                arrays["normalized_targets"][:40],
                arrays["valid_mask"][:40],
                dimensions,
            )
            for noise in range(4)
        ]
    fit_pass = all(
        sum(row[group]["passed"] for row in rows) >= 3
        for rows in fit.values()
        for group in native_groups
    )
    metrics, checks = {}, []
    for window, slots in (("first", slice(0, 1)), ("full", slice(None))):
        for group, indices in groups.items():
            target = targets[40:, slots, :][..., indices]
            train = targets[:40, slots, :][..., indices]
            for metric in ("mae", "mse"):
                transform = np.abs if metric == "mae" else np.square
                values = {}
                scene = {}
                for arm, arrays in (("control", control), ("treatment", treatment)):
                    prediction = arrays["standard_predictions"][:, 40:, slots, :][..., indices]
                    scene[arm] = transform(prediction - target).mean(axis=(2, 3))
                    values[arm] = scene[arm].mean(axis=1)
                constants = {
                    method: float(transform(fn(train, axis=0) - target).mean())
                    for method, fn in (("mean", np.mean), ("median", np.median))
                }
                nonregression = bool(np.all(values["treatment"] <= values["control"]))
                beats_constants = bool(np.all(values["treatment"] < min(constants.values())))
                checks.extend([nonregression, beats_constants])
                row = {
                    "by_noise": {key: value.tolist() for key, value in values.items()},
                    "constants": constants,
                    "nonregression": nonregression,
                    "beats_both_constants": beats_constants,
                }
                if group == "left_joint" and window == "full":
                    gains = scene["control"] - scene["treatment"]
                    leave_one_out = np.stack(
                        [np.delete(gains, i, axis=1).mean(axis=1) for i in range(5)]
                    )
                    improved = bool(np.all(gains.mean(axis=1) > 0) and np.all(leave_one_out > 0))
                    checks.append(improved)
                    row.update(
                        leave_one_scene_out_gain=leave_one_out.tolist(),
                        strict_left_improvement=improved,
                    )
                metrics[f"{window}/{group}/{metric}"] = row
    passed = fit_pass and all(checks)
    return {
        "status": "offline_candidate_passed" if passed else "negative_result",
        "passed": passed,
        "train_fit_passed": fit_pass,
        "train_fit": fit,
        "metrics": metrics,
        "development_is_independent_test": False,
        "fixed_endpoint": 1280,
        "gate3": "not measured",
        "gate4": "not measured",
        "m2_complete": False,
    }


def analyze(job):
    job = Path(job)
    arrays, metadata, reloads = {}, {}, {}
    for arm in ("control", "treatment"):
        first, second = job / (arm + "-first.npz"), job / (arm + "-reload.npz")
        reloads[arm] = verify_reload(first, second)
        with np.load(first, allow_pickle=False) as data:
            arrays[arm] = {key: data[key] for key in data.files}
        metadata[arm] = json.loads(Path(str(first) + ".json").read_text())
    for key in ("episodes", "image_sha256", "noise_seeds", "action_dimensions"):
        if metadata["control"][key] != metadata["treatment"][key]:
            raise ValueError("Arm input or Action Contract differs")
    result = compare(
        arrays["control"], arrays["treatment"], metadata["control"]["action_dimensions"]
    )
    result["reload"] = reloads
    result["source_arrays_sha256"] = {arm: sha(job / (arm + "-first.npz")) for arm in arrays}
    save(job / "result.json", result)
