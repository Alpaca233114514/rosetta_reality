"""Fit-strength acceptance must expose candidate gripper failure and preserve B."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from rosetta_reality.vla import visual_coverage, visual_fit


def evidence(arm="C"):
    target = np.repeat(
        np.stack([np.linspace(-1, 1, 45), np.linspace(0.1, 0.9, 45)], axis=-1)[:, None],
        50,
        axis=1,
    )
    noise = np.zeros((4, 1, 50, 32), dtype=np.float32)
    for index in range(1, 4):
        noise[index] = np.random.default_rng(index).standard_normal((1, 50, 32))
    arrays = {
        "normalized_predictions": np.repeat(target[None], 4, axis=0),
        "standard_predictions": np.repeat(target[None], 4, axis=0),
        "internal_grippers": np.zeros((4, 45, 50, 1)),
        "normalized_targets": target.copy(),
        "standard_targets": target.copy(),
        "valid_mask": np.ones_like(target, bool),
        "noise": noise,
    }
    steps = visual_fit.UPDATES[arm]
    metadata = {
        "protocol": visual_fit.PROTOCOL,
        "formula_protocol": visual_coverage.PROTOCOL,
        "arm": arm,
        "episodes": list(range(45)),
        "views": {
            "train8": list(range(8)),
            "train40": list(range(40)),
            "dev5": list(range(40, 45)),
        },
        "hidden_episodes": list(range(45, 50)),
        "hidden_test_loaded": False,
        "target_conditioning": False,
        "noise_conditions": [None, 20260905, 20260906, 20260907],
        "noise_hashes": [visual_coverage.array_hash(n) for n in noise],
        "chunk_length": 50,
        "max_action_dim": 32,
        "native_denoising_steps": 10,
        "model_mode": "eval_inference",
        "dimensions": [
            {"name": "joint", "unit": "radian", "minimum": -2, "maximum": 2},
            {
                "name": "gripper",
                "unit": "normalized",
                "encoding": "0_closed_1_open",
                "minimum": 0,
                "maximum": 1,
            },
        ],
        "nonvisual_hashes": ["a" * 64] * 45,
        "image_hashes": [f"{index:064x}" for index in range(45)],
        "common_identity": {
            **visual_fit.REVISIONS,
            "physical_contract_sha256": "1" * 64,
            "normalization_sha256": "2" * 64,
            "processor_identity_sha256": "3" * 64,
            "execution_contract_sha256": "4" * 64,
        },
        "arm_identity": {
            "run_name": visual_fit.RUN_NAMES[arm],
            "checkpoint_files": {
                "model.safetensors": visual_fit.CONTROL_MODEL_SHA256 if arm == "B" else "c" * 64,
                "config.json": "d" * 64,
                "train_config.json": "e" * 64,
            },
            "training_recipe": {
                "optimizer_steps": steps,
                "scheduler_decay_steps": steps,
                "scheduler_warmup_steps": 16,
                "batch_size": 4,
                "gradient_accumulation_steps": 1,
                "seed": 20260809,
                "completed_sample_exposures": 4 * steps,
                "episodes": list(range(40)),
                "frame": 0,
                "fresh_pinned_base": True,
                "optimizer_resumed": False,
                "native_loss_only": True,
                "freeze_vision_encoder": True,
                "train_expert_only": True,
                "train_state_proj": True,
                "use_amp": False,
                "optimizer": copy.deepcopy(visual_fit.OPTIMIZER),
                "scheduler_decay_lr": 2.5e-6,
                "train_config_sha256": "e" * 64,
            },
        },
        "process": {"pid": 1, "invocation_id": "first"},
    }
    return arrays, metadata


def test_negative_control_does_not_make_successful_candidate_impossible():
    b, bm = evidence("B")
    c, cm = evidence("C")
    b["normalized_predictions"][..., 1] = b["normalized_targets"][..., 1].mean(axis=0)
    b["standard_predictions"][..., 1] = b["standard_targets"][..., 1].mean(axis=0)
    result = visual_fit.compare_arms(b, bm, c, cm)
    assert not result["B"]["training_fit_passed"]
    assert result["control_aggregate_fit_passed"]
    assert result["C"]["training_fit_passed"] and result["offline_metric_criteria_passed"]
    assert not result["m2_complete"] and result["task_success"] == "not measured"


def test_candidate_training_gripper_failure_cannot_hide_in_aggregate_fit():
    arrays, metadata = evidence()
    arrays["normalized_predictions"][:, :40, :, 1] = arrays["normalized_targets"][:40, :, 1].mean(
        axis=0
    )
    result = visual_fit.summarize_bundle(arrays, metadata)
    assert result["training_aggregate_conditions"] == 4
    assert result["training_group_pass_counts"]["joint_radian"] == 4
    assert result["training_group_pass_counts"]["gripper_normalized"] == 0
    assert not result["training_fit_passed"]


def test_each_group_passing_three_different_conditions_is_insufficient():
    arrays, metadata = evidence()
    for condition, dimension in [(0, 0), (1, 1)]:
        arrays["normalized_predictions"][condition, :40, :, dimension] = arrays[
            "normalized_targets"
        ][:40, :, dimension].mean(axis=0)
    result = visual_fit.summarize_bundle(arrays, metadata)
    assert result["training_group_pass_counts"]["joint_radian"] == 3
    assert result["training_group_pass_counts"]["gripper_normalized"] == 3
    assert result["training_fit_conditions"] == 2 and not result["training_fit_passed"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("optimizer_steps", 256),
        ("scheduler_decay_steps", 256),
        ("completed_sample_exposures", 1024),
        ("batch_size", 8),
        ("optimizer_resumed", True),
        ("native_loss_only", False),
        ("train_expert_only", False),
        ("frame", False),
        ("train_config_sha256", "f" * 64),
    ],
)
def test_changed_candidate_recipe_is_rejected(key, value):
    arrays, metadata = evidence()
    metadata["arm_identity"]["training_recipe"][key] = value
    with pytest.raises(ValueError):
        visual_fit.validate_bundle(arrays, metadata)


def test_legacy_protocol_does_not_accept_new_fit_evidence():
    arrays, metadata = evidence("B")
    with pytest.raises(ValueError):
        visual_coverage.validate_bundle(arrays, metadata)
    metadata["protocol"] = visual_coverage.PROTOCOL
    metadata["arm"] = "C"
    with pytest.raises(ValueError):
        visual_coverage.validate_bundle(arrays, metadata)


def test_changed_control_weight_and_fake_candidate_are_rejected():
    for arm, digest in [("B", "c" * 64), ("C", visual_fit.CONTROL_MODEL_SHA256)]:
        arrays, metadata = evidence(arm)
        metadata["arm_identity"]["checkpoint_files"]["model.safetensors"] = digest
        with pytest.raises(ValueError):
            visual_fit.validate_bundle(arrays, metadata)


def test_reload_compares_final_chunk_values_and_independent_processes(tmp_path):
    arrays, metadata = evidence()
    other = copy.deepcopy(arrays)
    reloaded = copy.deepcopy(metadata)
    first = tmp_path / "first"
    visual_fit.write_bundle(first, arrays, metadata)
    with pytest.raises(ValueError, match="distinct"):
        visual_fit.compare_reload(first, first)
    same_process = tmp_path / "same-process"
    visual_fit.write_bundle(same_process, other, reloaded)
    with pytest.raises(ValueError, match="independent"):
        visual_fit.compare_reload(first, same_process)
    reloaded["process"] = {"pid": 2, "invocation_id": "reload"}
    second = tmp_path / "second"
    visual_fit.write_bundle(second, other, reloaded)
    assert visual_fit.compare_reload(first, second)["passed"]
    other["normalized_predictions"][3, 44, 49, 0] += 1e-3
    changed = tmp_path / "changed"
    visual_fit.write_bundle(changed, other, reloaded)
    result = visual_fit.compare_reload(first, changed)
    assert not result["passed"] and len(result["exact_arrays"]) == 7


def test_bundle_is_create_only_and_detects_tampering(tmp_path):
    arrays, metadata = evidence()
    output = tmp_path / "bundle"
    visual_fit.write_bundle(output, arrays, metadata)
    with pytest.raises(FileExistsError):
        visual_fit.write_bundle(output, arrays, metadata)
    with (output / "standard_predictions.npy").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="file identity"):
        visual_fit.read_bundle(output)


def test_image_invariant_control_still_fails_the_historical_aggregate_fit_guard():
    b, bm = evidence("B")
    c, cm = evidence("C")
    b["normalized_predictions"][:] = b["normalized_targets"].mean(axis=0)
    result = visual_fit.compare_arms(b, bm, c, cm)
    assert not result["control_aggregate_fit_passed"]
    assert not result["offline_metric_criteria_passed"]


def test_first_action_regression_blocks_otherwise_better_normalized_scores():
    b, bm = evidence("B")
    c, cm = evidence("C")
    b["normalized_predictions"] *= 0.9
    b["standard_predictions"] *= 0.9
    c["standard_predictions"][:, 40:, 0, 0] += 0.5
    result = visual_fit.compare_arms(b, bm, c, cm)
    assert result["C"]["development_passed"] and result["fit_strength_gain_passed"]
    assert not result["physical_nonregression"]["first_action/joint_radian"]
    assert not result["offline_metric_criteria_passed"]


def test_common_revision_and_native_optimizer_drift_are_rejected():
    arrays, metadata = evidence()
    metadata["common_identity"]["base_revision"] = "f" * 40
    with pytest.raises(ValueError, match="pinned"):
        visual_fit.validate_bundle(arrays, metadata)
    arrays, metadata = evidence()
    metadata["arm_identity"]["training_recipe"]["optimizer"]["lr"] *= 2
    with pytest.raises(ValueError, match="optimizer"):
        visual_fit.validate_bundle(arrays, metadata)
