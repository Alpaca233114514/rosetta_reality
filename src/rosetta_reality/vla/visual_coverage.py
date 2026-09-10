"""Frame-zero all-nonself scoring and complete prediction evidence.

Only numerical arrays and metadata are handled here. No model, dataset, network,
optimizer or environment access occurs at import. The protocol is a development
diagnostic; passing it never implies task success or M2 acceptance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

PROTOCOL = "frame0_all_nonself_mean_error_v1"
ARRAY_NAMES = {
    "normalized_predictions", "standard_predictions", "internal_grippers",
    "normalized_targets", "standard_targets", "valid_mask", "noise",
}


def array_hash(value: np.ndarray) -> str:
    """Bind shape, dtype and exact contiguous values, not just flattened bytes."""
    value = np.asarray(value)
    if value.dtype.hasobject:
        raise ValueError("Object arrays are forbidden.")
    header = json.dumps([value.dtype.str, list(value.shape)], separators=(",", ":"))
    return hashlib.sha256(header.encode() + b"\n" + value.tobytes(order="C")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def action_groups(dimensions: list[dict]) -> dict[str, list[int]]:
    names = [d["name"] for d in dimensions]
    if not names or len(names) != len(set(names)):
        raise ValueError("Action names must be nonempty and unique.")
    groups = {"all_valid": list(range(len(names))), "joint_radian": [], "gripper_normalized": []}
    for index, d in enumerate(dimensions):
        if d["unit"] == "radian":
            groups["joint_radian"].append(index)
        elif d["unit"] == "normalized" and d.get("encoding") == "0_closed_1_open":
            groups["gripper_normalized"].append(index)
        else:
            raise ValueError("Unsupported action unit/encoding.")
        if not np.isfinite([d["minimum"], d["maximum"]]).all() or d["minimum"] >= d["maximum"]:
            raise ValueError("Invalid action bounds.")
    if any(not values for values in groups.values()):
        raise ValueError("Both joint and gripper groups are required.")
    return groups


def _finite(value, label: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "fi" or not np.isfinite(raw).all():
        raise ValueError(f"{label} must contain only finite numeric values.")
    return raw.astype(np.float64)


def score_view(prediction, target, valid_mask, dimensions: list[dict]) -> dict:
    """Rows are destination labels, columns are image-donor predictions."""
    p, y = _finite(prediction, "Prediction"), _finite(target, "Target")
    mask = np.asarray(valid_mask)
    groups = action_groups(dimensions)
    if p.ndim != 3 or p.shape != y.shape or min(p.shape) <= 0:
        raise ValueError("Predictions and labels must share [N,H,D].")
    n, _, d = p.shape
    if n < 2 or d != len(dimensions):
        raise ValueError("Need at least two scenes and exactly the valid action dimensions.")
    if mask.dtype != np.bool_ or mask.shape != y.shape or not mask.all():
        raise ValueError("Frame-zero protocol requires complete identical valid chunks.")
    result = {}
    for name, indices in groups.items():
        a, b = p[..., indices], y[..., indices]
        matrix = np.square(b[:, None] - a[None, :]).mean(axis=(2, 3))
        per_correct = np.diag(matrix)
        per_wrong = (matrix.sum(axis=1) - per_correct) / (n - 1)
        per_floor = np.square(b - b.mean(axis=0)).mean(axis=(1, 2))
        correct, wrong, floor = map(float, (per_correct.mean(), per_wrong.mean(), per_floor.mean()))
        epsilon = 1e-8 * max(1.0, abs(wrong), abs(floor))
        measurable = floor > epsilon
        result[name] = {
            "dimensions": indices, "correct": correct, "mismatch": wrong,
            "mean_floor": floor, "epsilon": epsilon,
            "visual_gain": wrong - correct, "below_mean_gain": floor - correct,
            "measurable": measurable,
            "status": "measured" if measurable else "not measurable",
            "passed": bool(measurable and wrong - correct > epsilon and floor - correct > epsilon),
            "error_matrix": matrix.tolist(),
            "per_episode_correct": per_correct.tolist(),
            "per_episode_mismatch": per_wrong.tolist(),
            "per_episode_mean_floor": per_floor.tolist(),
            "per_episode_visual_gain": (per_wrong - per_correct).tolist(),
            "per_episode_below_mean_gain": (per_floor - per_correct).tolist(),
        }
    return result


def standard_diagnostics(prediction, target, dimensions: list[dict]) -> dict:
    p, y = _finite(prediction, "Standard prediction"), _finite(target, "Standard target")
    if p.ndim != 3 or p.shape != y.shape or p.shape[-1] != len(dimensions):
        raise ValueError("Standard action shapes disagree.")
    groups = action_groups(dimensions)
    result = {}
    for horizon, sl in (("chunk", slice(None)), ("first_action", slice(0, 1))):
        error = p[:, sl] - y[:, sl]
        result[horizon] = {
            name: {"mae": float(np.abs(error[..., indices]).mean()),
                   "mse": float(np.square(error[..., indices]).mean())}
            for name, indices in groups.items() if name != "all_valid"
        }
        result[horizon]["per_dimension_mae"] = np.abs(error).mean(axis=(0, 1)).tolist()
        result[horizon]["per_dimension_mse"] = np.square(error).mean(axis=(0, 1)).tolist()
    return result


def validate_bundle(arrays: dict[str, np.ndarray], metadata: dict) -> None:
    if set(arrays) != ARRAY_NAMES or metadata.get("protocol") != PROTOCOL:
        raise ValueError("Missing full prediction evidence or wrong protocol.")
    if metadata.get("hidden_test_loaded") is not False or metadata.get("target_conditioning") is not False:
        raise ValueError("Hidden access or teacher-forced conditioning is forbidden.")
    if metadata.get("native_denoising_steps") != 10 or metadata.get("model_mode") != "eval_inference":
        raise ValueError("Native inference contract changed.")
    if metadata.get("noise_conditions") != [None, 20260905, 20260906, 20260907]:
        raise ValueError("Noise registration changed.")
    if metadata.get("arm") not in {"A", "B"}:
        raise ValueError("Unknown experimental arm.")
    if type(metadata.get("chunk_length")) is not int or metadata["chunk_length"] <= 0:
        raise ValueError("Invalid action horizon.")
    if metadata.get("max_action_dim", 0) < len(metadata["dimensions"]):
        raise ValueError("Padded action width is smaller than the valid contract.")
    episodes = metadata["episodes"]
    if any(type(ep) is not int for ep in episodes) or len(set(episodes)) != len(episodes):
        raise ValueError("Duplicate or invalid episode identity.")
    views = metadata["views"]
    if set(views) != {"train8", "train40", "dev5"}:
        raise ValueError("All three registered views are required.")
    if episodes != views["train40"] + views["dev5"]:
        raise ValueError("Bundle episode order differs from its registered views.")
    if not set(views["train8"]) <= set(views["train40"]) or set(views["train40"]) & set(views["dev5"]):
        raise ValueError("Donor scope or split mismatch.")
    if any(len(v) < 2 or len(v) != len(set(v)) for v in views.values()):
        raise ValueError("Each view needs distinct scenes.")
    if set(episodes) & set(metadata["hidden_episodes"]):
        raise ValueError("Hidden episode in evidence.")
    groups = action_groups(metadata["dimensions"])
    shape = (4, len(episodes), metadata["chunk_length"], len(metadata["dimensions"]))
    for name in ("normalized_predictions", "standard_predictions"):
        if arrays[name].shape != shape:
            raise ValueError("Full prediction chunk shape differs from the contract.")
    for name in ("normalized_targets", "standard_targets", "valid_mask"):
        if arrays[name].shape != shape[1:]:
            raise ValueError("Target/mask shape mismatch.")
    if arrays["valid_mask"].dtype != np.bool_ or not arrays["valid_mask"].all():
        raise ValueError("Incomplete frame-zero target chunks.")
    if arrays["internal_grippers"].shape != (*shape[:-1], len(groups["gripper_normalized"])):
        raise ValueError("Missing full internal gripper evidence.")
    if arrays["noise"].shape != (4, 1, shape[2], metadata["max_action_dim"]):
        raise ValueError("Noise shape mismatch.")
    for name, array in arrays.items():
        if name != "valid_mask":
            _finite(array, name)
    if np.count_nonzero(arrays["noise"][0]):
        raise ValueError("The zero-noise condition is not zero.")
    if [array_hash(x) for x in arrays["noise"]] != metadata["noise_hashes"]:
        raise ValueError("Noise values differ from the saved identity.")
    for key in ("nonvisual_hashes", "image_hashes"):
        if len(metadata[key]) != len(episodes) or any(
            not isinstance(h, str) or len(h) != 64 for h in metadata[key]
        ):
            raise ValueError("Per-scene conditioning hashes are missing.")
    if len(set(metadata["nonvisual_hashes"])) != 1:
        raise ValueError("Nonvisual conditioning differs across images.")
    if len(set(metadata["image_hashes"])) != len(episodes):
        raise ValueError("Image duplicate found; preserve and review the registered scope.")
    for key in ("common_identity", "arm_identity", "process"):
        if not isinstance(metadata.get(key), dict) or not metadata[key]:
            raise ValueError("Missing execution provenance.")
    if (type(metadata["process"].get("pid")) is not int
            or not metadata["process"].get("invocation_id")):
        raise ValueError("Missing independent process identity.")


def summarize_bundle(arrays: dict[str, np.ndarray], metadata: dict) -> dict:
    validate_bundle(arrays, metadata)
    dims = metadata["dimensions"]
    results = {}
    for view, episodes in metadata["views"].items():
        indices = [metadata["episodes"].index(ep) for ep in episodes]
        y = arrays["normalized_targets"][indices]
        ys = arrays["standard_targets"][indices]
        mask = arrays["valid_mask"][indices]
        conditions = []
        for noise in range(4):
            g = arrays["internal_grippers"][noise, indices]
            conditions.append({
                "noise_seed": metadata["noise_conditions"][noise],
                "normalized": score_view(arrays["normalized_predictions"][noise, indices], y, mask, dims),
                "standard": standard_diagnostics(arrays["standard_predictions"][noise, indices], ys, dims),
                "gripper_internal_outside_support_rate": float((np.abs(g) > np.pi / 2).mean()),
            })
        results[view] = conditions
    train = [metadata["episodes"].index(ep) for ep in metadata["views"]["train40"]]
    dev = [metadata["episodes"].index(ep) for ep in metadata["views"]["dev5"]]
    train_mean = arrays["normalized_targets"][train].astype(np.float64).mean(axis=0)
    error = np.square(arrays["normalized_targets"][dev] - train_mean)
    baseline = {name: float(error[..., indices].mean()) for name, indices in action_groups(dims).items()}
    fit_view = "train8" if metadata["arm"] == "A" else "train40"
    fit = sum(row["normalized"]["all_valid"]["passed"] for row in results[fit_view])
    dev_pass = [all(g["passed"] for g in row["normalized"].values()) for row in results["dev5"]]
    return {"views": results, "train40_mean_on_dev5_mse": baseline,
            "training_fit_conditions": fit, "training_fit_passed": fit >= 3,
            "development_conditions": dev_pass, "development_passed": sum(dev_pass) >= 3,
            "m2_complete": False, "task_success": "not measured"}


def write_bundle(directory: Path, arrays: dict[str, np.ndarray], metadata: dict) -> None:
    validate_bundle(arrays, metadata)
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "metadata": metadata, "arrays": {}}
    for name, value in sorted(arrays.items()):
        path = directory / f"{name}.npy"
        with path.open("xb") as stream:
            np.save(stream, value, allow_pickle=False)
        manifest["arrays"][name] = {"path": path.name, "sha256": file_hash(path),
                                    "array_sha256": array_hash(value)}
    with (directory / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)


def read_bundle(directory: Path) -> tuple[dict, dict]:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or set(manifest["arrays"]) != ARRAY_NAMES:
        raise ValueError("Incomplete bundle manifest.")
    arrays = {}
    for name, info in manifest["arrays"].items():
        if info["path"] != f"{name}.npy":
            raise ValueError("Unexpected bundle path.")
        path = directory / info["path"]
        if path.is_symlink() or file_hash(path) != info["sha256"]:
            raise ValueError("Bundle file identity changed.")
        value = np.load(path, allow_pickle=False)
        if array_hash(value) != info["array_sha256"]:
            raise ValueError("Bundle array identity changed.")
        arrays[name] = value
    validate_bundle(arrays, manifest["metadata"])
    return arrays, manifest["metadata"]


def compare_reload(first: Path, second: Path) -> dict:
    a, am = read_bundle(first)
    b, bm = read_bundle(second)
    if (first.resolve() == second.resolve()
            or am["process"]["pid"] == bm["process"]["pid"]
            or am["process"]["invocation_id"] == bm["process"]["invocation_id"]):
        raise ValueError("Reload must come from an independent invocation.")
    for key in set(am) | set(bm):
        if key != "process" and am.get(key) != bm.get(key):
            raise ValueError(f"Reload metadata drift: {key}.")
    equality = {name: bool(a[name].dtype == b[name].dtype and np.array_equal(a[name], b[name]))
                for name in sorted(ARRAY_NAMES)}
    return {"passed": all(equality.values()), "exact_arrays": equality,
            "compared_entire_chunk": True, "metric_only_comparison": False}


def compare_arms(a: dict, am: dict, b: dict, bm: dict) -> dict[str, Any]:
    validate_bundle(a, am)
    validate_bundle(b, bm)
    if am["arm"] != "A" or bm["arm"] != "B":
        raise ValueError("Comparison requires A control then B candidate.")
    for key in set(am) | set(bm):
        if key not in {"arm", "arm_identity", "process"} and am.get(key) != bm.get(key):
            raise ValueError(f"Between-arm protocol drift: {key}.")
    for key in ("normalized_targets", "standard_targets", "valid_mask", "noise"):
        if a[key].dtype != b[key].dtype or not np.array_equal(a[key], b[key]):
            raise ValueError(f"Between-arm evidence mismatch: {key}.")
    ar, br = summarize_bundle(a, am), summarize_bundle(b, bm)
    gains = []
    for ac, bc in zip(ar["views"]["dev5"], br["views"]["dev5"], strict=True):
        x, y = ac["normalized"]["all_valid"], bc["normalized"]["all_valid"]
        epsilon = 1e-8 * max(1., abs(x["correct"]), abs(y["correct"]), abs(x["visual_gain"]), abs(y["visual_gain"]))
        gains.append(y["correct"] < x["correct"] - epsilon and y["visual_gain"] > x["visual_gain"] + epsilon)
    nonregression = {}
    for horizon in ("chunk", "first_action"):
        for group in ("joint_radian", "gripper_normalized"):
            old = float(np.mean([v["standard"][horizon][group]["mse"] for v in ar["views"]["dev5"]]))
            new = float(np.mean([v["standard"][horizon][group]["mse"] for v in br["views"]["dev5"]]))
            nonregression[f"{horizon}/{group}"] = new <= old + 1e-8 * max(1., abs(old), abs(new))
    return {"A": ar, "B": br, "coverage_gain_conditions": gains,
            "coverage_gain_passed": sum(gains) >= 3, "physical_nonregression": nonregression,
            "offline_metric_criteria_passed": bool(ar["training_fit_passed"] and br["training_fit_passed"]
                and br["development_passed"] and sum(gains) >= 3 and all(nonregression.values())),
            "full_acceptance_requires_external_integrity_and_reload": True,
            "m2_complete": False, "task_success": "not measured"}
