"""B/C fit-strength evidence; the historical A/B protocol remains separate.

Numerical criteria are development diagnostics, never M2 or task-success claims.
An observed negative B control is valid; candidate C must fit all action groups.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from rosetta_reality.vla.visual_coverage import (
    ARRAY_NAMES,
    _summarize_prediction_evidence,
    _validate_prediction_evidence,
    array_hash,
    file_hash,
)
from rosetta_reality.vla.visual_coverage import (
    PROTOCOL as FORMULA_PROTOCOL,
)

PROTOCOL = "frame0_all_nonself_mean_error_fit40_v1"
CONTROL_MODEL_SHA256 = "4e91624b34c3a16e22e0cbba19d8bcf3352a0d3835522ea6973ee22f0397accf"
RUN_NAMES = {
    "B": "m2-smolvla450m-visual-coverage40-main256-003",
    "C": "m2-smolvla450m-visual-hestia-fit40-001",
}
UPDATES = {"B": 256, "C": 1280}
TRAIN40 = [
    49,
    4,
    23,
    43,
    21,
    37,
    18,
    34,
    0,
    47,
    38,
    29,
    3,
    26,
    14,
    17,
    44,
    30,
    15,
    42,
    10,
    35,
    25,
    32,
    19,
    36,
    41,
    28,
    8,
    27,
    16,
    11,
    2,
    20,
    9,
    39,
    46,
    48,
    12,
    40,
]
DEV5 = [22, 13, 7, 33, 45]
HIDDEN5 = [31, 6, 1, 24, 5]
GROUPS = ("all_valid", "joint_radian", "gripper_normalized")
REVISIONS = {
    "base_revision": "c83c3163b8ca9b7e67c509fffd9121e66cb96205",
    "vlm_revision": "7b375e1b73b11138ff12fe22c8f2822d8fe03467",
    "data_revision": "cc571a3c661df81b566dbfde3d5c1e85fcdf7884",
    "upstream_revision": "c903b114a90e703b3f7d0c46cb38727c328c55ff",
}
OPTIMIZER = {
    "type": "adamw",
    "lr": 1e-4,
    "betas": [0.9, 0.95],
    "eps": 1e-8,
    "weight_decay": 1e-10,
    "grad_clip_norm": 10.0,
}


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_bundle(arrays: dict, metadata: dict) -> None:
    _validate_prediction_evidence(arrays, metadata, protocol=PROTOCOL, arms={"B", "C"})
    if metadata.get("formula_protocol") != FORMULA_PROTOCOL:
        raise ValueError("Fit-strength scoring must retain the registered numerical formula.")
    views = metadata["views"]
    if (
        views != {"train8": TRAIN40[:8], "train40": TRAIN40, "dev5": DEV5}
        or metadata["episodes"] != TRAIN40 + DEV5
        or metadata["hidden_episodes"] != HIDDEN5
        or metadata["chunk_length"] != 50
        or metadata["max_action_dim"] != 32
    ):
        raise ValueError("Fit-strength scene/horizon registration changed.")
    arm = metadata["arm"]
    common = metadata["common_identity"]
    if any(common.get(key) != revision for key, revision in REVISIONS.items()):
        raise ValueError("The pinned base/VLM/data/upstream identity changed.")
    for key in (
        "physical_contract_sha256",
        "normalization_sha256",
        "processor_identity_sha256",
        "execution_contract_sha256",
    ):
        if not _sha(common.get(key)):
            raise ValueError(f"Missing common identity: {key}.")
    identity = metadata["arm_identity"]
    recipe = identity.get("training_recipe", {})
    if identity.get("run_name") != RUN_NAMES[arm]:
        raise ValueError("Fit-strength run identity changed.")
    files = identity.get("checkpoint_files", {})
    if (
        not isinstance(files, dict)
        or not {"model.safetensors", "config.json", "train_config.json"} <= files.keys()
    ):
        raise ValueError("Fit-strength checkpoint identity is incomplete.")
    if not isinstance(recipe, dict):
        raise ValueError("Fit-strength training recipe is missing.")
    if any(not _sha(value) for value in files.values()):
        raise ValueError("Fit-strength checkpoint hashes are not sealed.")
    if arm == "B" and files["model.safetensors"] != CONTROL_MODEL_SHA256:
        raise ValueError("The historical B control checkpoint changed.")
    if arm == "C" and files["model.safetensors"] == CONTROL_MODEL_SHA256:
        raise ValueError("Candidate C cannot be the unchanged B checkpoint.")
    for key, expected in {
        "optimizer_steps": UPDATES[arm],
        "scheduler_decay_steps": UPDATES[arm],
        "scheduler_warmup_steps": 16,
        "batch_size": 4,
        "gradient_accumulation_steps": 1,
        "seed": 20260809,
        "completed_sample_exposures": 4 * UPDATES[arm],
    }.items():
        if type(recipe.get(key)) is not int or recipe[key] != expected:
            raise ValueError(f"Fit-strength training recipe drift: {key}.")
    if (
        recipe.get("episodes") != views["train40"]
        or type(recipe.get("frame")) is not int
        or recipe["frame"] != 0
    ):
        raise ValueError("Fit-strength active sample scope changed.")
    for key, expected in {
        "fresh_pinned_base": True,
        "optimizer_resumed": False,
        "native_loss_only": True,
        "freeze_vision_encoder": True,
        "train_expert_only": True,
        "train_state_proj": True,
        "use_amp": False,
    }.items():
        if recipe.get(key) is not expected:
            raise ValueError(f"Fit-strength adaptation/initialization drift: {key}.")
    if recipe.get("train_config_sha256") != files["train_config.json"]:
        raise ValueError("Training recipe is not bound to the saved native config.")
    if recipe.get("optimizer") != OPTIMIZER or recipe.get("scheduler_decay_lr") != 2.5e-6:
        raise ValueError("The pinned native optimizer/scheduler recipe changed.")


def summarize_bundle(arrays: dict, metadata: dict) -> dict:
    validate_bundle(arrays, metadata)
    summary = _summarize_prediction_evidence(arrays, metadata, fit_view="train40")
    training = [
        all(row["normalized"][group]["passed"] for group in GROUPS)
        for row in summary["views"]["train40"]
    ]
    summary.update(
        training_aggregate_conditions=summary["training_fit_conditions"],
        training_fit_conditions=sum(training),
        training_fit_all_group_conditions=training,
        training_fit_passed=sum(training) >= 3,
        training_group_pass_counts={
            group: sum(row["normalized"][group]["passed"] for row in summary["views"]["train40"])
            for group in GROUPS
        },
    )
    return summary


def compare_arms(b: dict, bm: dict, c: dict, cm: dict) -> dict:
    validate_bundle(b, bm)
    validate_bundle(c, cm)
    if bm["arm"] != "B" or cm["arm"] != "C":
        raise ValueError("Fit-strength comparison requires B control then C candidate.")
    for key in set(bm) | set(cm):
        if key not in {"arm", "arm_identity", "process"} and bm.get(key) != cm.get(key):
            raise ValueError(f"Between-arm protocol drift: {key}.")
    for key in ("normalized_targets", "standard_targets", "valid_mask", "noise"):
        if b[key].dtype != c[key].dtype or not np.array_equal(b[key], c[key]):
            raise ValueError(f"Between-arm evidence mismatch: {key}.")
    old, new = summarize_bundle(b, bm), summarize_bundle(c, cm)
    gains = []
    for before, after in zip(old["views"]["dev5"], new["views"]["dev5"], strict=True):
        x, y = before["normalized"]["all_valid"], after["normalized"]["all_valid"]
        epsilon = 1e-8 * max(
            1.0, abs(x["correct"]), abs(y["correct"]), abs(x["visual_gain"]), abs(y["visual_gain"])
        )
        gains.append(
            y["correct"] < x["correct"] - epsilon and y["visual_gain"] > x["visual_gain"] + epsilon
        )
    nonregression = {}
    for horizon in ("chunk", "first_action"):
        for group in GROUPS[1:]:
            before = float(
                np.mean([row["standard"][horizon][group]["mse"] for row in old["views"]["dev5"]])
            )
            after = float(
                np.mean([row["standard"][horizon][group]["mse"] for row in new["views"]["dev5"]])
            )
            nonregression[f"{horizon}/{group}"] = after <= before + 1e-8 * max(
                1.0, abs(before), abs(after)
            )
    return {
        "B": old,
        "C": new,
        "fit_strength_gain_conditions": gains,
        "fit_strength_gain_passed": sum(gains) >= 3,
        "physical_nonregression": nonregression,
        "control_aggregate_fit_passed": old["training_aggregate_conditions"] >= 3,
        "negative_control_gripper_fit_is_not_a_candidate_gate": True,
        "offline_metric_criteria_passed": bool(
            old["training_aggregate_conditions"] >= 3
            and new["training_fit_passed"]
            and new["development_passed"]
            and sum(gains) >= 3
            and all(nonregression.values())
        ),
        "full_acceptance_requires_external_integrity_and_reload": True,
        "m2_complete": False,
        "task_success": "not measured",
    }


def write_bundle(directory: Path, arrays: dict, metadata: dict) -> None:
    validate_bundle(arrays, metadata)
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "metadata": metadata, "arrays": {}}
    for name, value in sorted(arrays.items()):
        path = directory / f"{name}.npy"
        with path.open("xb") as stream:
            np.save(stream, value, allow_pickle=False)
        manifest["arrays"][name] = {
            "path": path.name,
            "sha256": file_hash(path),
            "array_sha256": array_hash(value),
        }
    with (directory / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)


def read_bundle(directory: Path) -> tuple[dict, dict]:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or set(manifest["arrays"]) != ARRAY_NAMES:
        raise ValueError("Incomplete fit-strength bundle manifest.")
    arrays = {}
    for name, info in manifest["arrays"].items():
        if info["path"] != f"{name}.npy":
            raise ValueError("Unexpected bundle path.")
        path = directory / info["path"]
        if path.is_symlink() or file_hash(path) != info["sha256"]:
            raise ValueError("Bundle file identity changed.")
        arrays[name] = np.load(path, allow_pickle=False)
        if array_hash(arrays[name]) != info["array_sha256"]:
            raise ValueError("Bundle array identity changed.")
    validate_bundle(arrays, manifest["metadata"])
    return arrays, manifest["metadata"]


def compare_reload(first: Path, second: Path) -> dict:
    """Verify distinct paths/processes and all seven complete arrays."""
    if first.resolve() == second.resolve():
        raise ValueError("Reload must use distinct bundle paths.")
    a, am = read_bundle(first)
    b, bm = read_bundle(second)
    validate_bundle(a, am)
    validate_bundle(b, bm)
    if (
        am["process"]["pid"] == bm["process"]["pid"]
        or am["process"]["invocation_id"] == bm["process"]["invocation_id"]
    ):
        raise ValueError("Reload must come from an independent invocation.")
    for key in set(am) | set(bm):
        if key != "process" and am.get(key) != bm.get(key):
            raise ValueError(f"Reload metadata drift: {key}.")
    equality = {
        name: bool(a[name].dtype == b[name].dtype and np.array_equal(a[name], b[name]))
        for name in sorted(ARRAY_NAMES)
    }
    return {
        "passed": all(equality.values()),
        "exact_arrays": equality,
        "compared_entire_chunk": True,
        "metric_only_comparison": False,
    }
