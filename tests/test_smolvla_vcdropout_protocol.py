"""Focused preregistration tests for the visual-conditioning candidate (no weights).

These act as the boot-time canary on the training host: every case either
proves the frozen post-training invariants still bind the working tree or
fails closed before any paid CUDA work starts.  They mirror the Zen protocol
tests and never load model weights, datasets or accelerators.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for candidate in (str(REPOSITORY_ROOT / "src"), str(SCRIPTS_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import smolvla_vcdropout_protocol as protocol  # type: ignore[import-not-found]  # noqa: E402

from rosetta_reality.experiment import file_sha256  # noqa: E402

# Frozen Zen-uniform offset-250 baseline, read from the immutable diagnostic
# report (preregistration section 4 approximates these to 0.953/0.047/0.905).
UNIFORM_STATE_RATIOS = {
    "action_expert": 2.387,
    "state_projector": 3.378,
    "action_io_projections": 2.162,
}
UNIFORM_IMAGE_RATIOS = {
    "action_expert": 0.972,
    "state_projector": 0.918,
    "action_io_projections": 0.972,
}


def _candidate_plan() -> dict:
    implementation = {
        relative: file_sha256(REPOSITORY_ROOT / relative)
        for relative in protocol.REQUIRED_IMPLEMENTATION_FILES
    }
    implementation[protocol.VISUAL_CONDITIONING_IMPLEMENTATION] = (
        protocol.VISUAL_CONDITIONING_SHA256
    )
    return {
        "schema_version": 2,
        "role": "vla",
        "status": "preregistered",
        "plan_id": protocol.VCD_PLAN_ID,
        "run_name": protocol.VCD_RUN_NAME,
        "parent_experiment": {
            "config": protocol.PARENT_CONFIG,
            "sha256": protocol.PARENT_SHA256,
            "experiment_id": protocol.EXPERIMENT_ID,
        },
        "runtime_profile": {
            "path": "configs/runtime/autodl_rtx4090.yaml",
            "sha256": protocol.RUNTIME_PROFILE_SHA256,
            "profile_id": protocol.RUNTIME_PROFILE_ID,
            "nested_docker_used": False,
        },
        "initialization": {
            "source": "revision_pinned_base_model",
            "faust_checkpoint_used": False,
            "aster_checkpoint_used": False,
            "way_checkpoint_used": False,
            "zen_checkpoint_used": False,
            "optimizer_state_reused": False,
        },
        "training": {
            "episodes": list(protocol.TRAIN_EPISODES),
            "batch_size": 64,
            "steps": protocol.FORMAL_STEPS,
            "save_freq": 79,
            "log_freq": 79,
            "checkpoint_steps": list(protocol.CHECKPOINT_STEPS),
            "eval_split": 0.0,
            "validation_gradients": False,
            "hidden_test_loaded": False,
            "num_workers": 0,
            "persistent_workers": False,
            "policy": {
                "empty_cameras": 2,
                "compile_model": False,
                "compile_mode": "default",
                "skip_fully_masked_camera_encoding": True,
            },
            "optimizer": {
                "type": "adamw",
                "lr": 1.0e-4,
                "betas": [0.9, 0.95],
                "eps": 1.0e-8,
                "weight_decay": 1.0e-10,
                "grad_clip_norm": 10.0,
            },
            "scheduler": {
                "type": "cosine_decay_with_warmup",
                "num_warmup_steps": 31,
                "num_decay_steps": 316,
                "peak_lr": 1.0e-4,
                "decay_lr": 2.5e-6,
            },
        },
        "resources": {
            "memory_limit": "autodl_platform_container",
            "memory_swap_limit": "autodl_platform_container",
            "cpu_limit": 16,
            "mixed_precision": "bf16",
            "checkpoint_memory_trim": True,
        },
        "features": [
            {"name": "train_only_statistics"},
            {"name": "action_boundary_projection"},
            {"name": "checkpoint_memory_trim"},
            {"name": "state_conditioning_dropout"},
            {"name": "gradient_clip_diagnostics"},
            {"name": "checkpoint_metric_snapshot"},
        ],
        "validation": {
            "episodes": list(protocol.VALIDATION_EPISODES),
            "frame_offsets": [0],
            "total_samples": len(protocol.VALIDATION_EPISODES),
            "hidden_test_loaded": False,
        },
        "visual_conditioning_contract": dict(protocol.VCD_CONTRACT),
        "monitoring": {
            "policy": "sleep_between_quarter_checkpoints",
            "wake_fractions": [0.25, 0.5, 0.75, 1.0],
            "wake_steps": list(protocol.CHECKPOINT_STEPS),
            "blocking_command": "sleep",
            "sleep_poll_seconds": 300,
            "estimated_total_minutes": 90,
            "hidden_test_loaded": False,
        },
        "prerequisites": {},
        "normalization": {
            "source_split": "train",
            "report": protocol.NORMALIZATION_REPORT_RELATIVE,
            "report_sha256": protocol.NORMALIZATION_REPORT_SHA256,
            "dataset_view_manifest": protocol.VIEW_MANIFEST_RELATIVE,
            "dataset_view_manifest_sha256": protocol.VIEW_MANIFEST_SHA256,
            "validation_episodes_loaded": False,
            "hidden_test_loaded": False,
        },
        "implementation_files": implementation,
        "stop_conditions": [
            "nonfinite_loss_gradient_or_action",
            "out_of_memory_no_automatic_retry",
            "hidden_test_or_validation_episode_used_for_gradients",
            "feature_install_rollback_failure",
        ],
        "hidden_test_loaded": False,
    }


def test_gate_thresholds_match_the_preregistration() -> None:
    assert protocol.GATE_NORMAL_FLOW_LOSS_MAX == 0.22005
    assert protocol.GATE_STATE_SENSITIVITY_MAX == 0.70
    assert protocol.GATE_IMAGE_SENSITIVITY_MIN == 0.10
    assert protocol.GATE_STATE_DOMINANCE_MAX == 0.453
    assert protocol.CONTROL_NORMAL_MEAN_FLOW_LOSS == 0.1467
    assert protocol.GATE_FRAME_OFFSET == 250
    assert protocol.GATE_EPISODES == protocol.VALIDATION_EPISODES
    assert protocol.GATE_SHUFFLE_SEED == 20260812
    assert protocol.GATE_FLOW_TIME == 0.5


def test_control_baseline_reproduces_the_uniform_diagnostic() -> None:
    baseline = protocol.control_baseline()
    assert baseline["normal_mean_flow_loss"] == pytest.approx(0.1467, abs=1e-9)
    assert round(baseline["state_sensitivity"], 3) == 0.953
    assert round(baseline["image_sensitivity"], 3) == 0.047
    assert round(baseline["state_dominance_score"], 3) == 0.905
    assert baseline["artifact_manifest_sha256"] == (
        protocol.CONTROL_ARTIFACT_MANIFEST_SHA256
    )


def test_gate_sensitivity_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="missing for trainable group"):
        protocol.gate_sensitivity({"action_expert": 2.0})
    with pytest.raises(ValueError, match="not numeric"):
        protocol.gate_sensitivity(
            {
                "action_expert": 2.0,
                "state_projector": None,
                "action_io_projections": 1.5,
            }
        )
    with pytest.raises(ValueError, match="finite and positive"):
        protocol.gate_sensitivity(
            {
                "action_expert": 2.0,
                "state_projector": -1.0,
                "action_io_projections": 1.5,
            }
        )


def _passing_measurement() -> dict:
    return {
        "normal_mean_flow_loss": 0.15,
        "state_shuffle_gradient_ratios": {
            "action_expert": 1.4,
            "state_projector": 1.6,
            "action_io_projections": 1.3,
        },
        "image_shuffle_gradient_ratios": {
            "action_expert": 1.5,
            "state_projector": 1.4,
            "action_io_projections": 1.6,
        },
        "normal_trainable_gradient_norms": {
            "action_expert": 0.5,
            "state_projector": 0.3,
            "action_io_projections": 0.2,
        },
        "pairwise_sample_state_difference": 0.8,
        "pairwise_sample_image_difference": 220.0,
    }


def test_evaluate_gate_criteria_passes_a_compliant_measurement() -> None:
    gate = protocol.evaluate_gate_criteria(**_passing_measurement())
    assert gate["passed"] is True
    assert len(gate["criteria"]) == 6
    assert [criterion["passed"] for criterion in gate["criteria"]] == [True] * 6


def test_each_gate_criterion_fails_alone() -> None:
    # criterion 1: a trainable group lost its normal gradient
    measurement = _passing_measurement()
    measurement["normal_trainable_gradient_norms"]["state_projector"] = 0.0
    gate = protocol.evaluate_gate_criteria(**measurement)
    assert gate["passed"] is False
    assert gate["criteria"][0]["passed"] is False

    # criterion 2: degenerate states at the nonzero offset
    measurement = _passing_measurement()
    measurement["pairwise_sample_state_difference"] = 0.0
    gate = protocol.evaluate_gate_criteria(**measurement)
    assert gate["criteria"][1]["passed"] is False

    # criterion 3: input destruction must not win the gate
    measurement = _passing_measurement()
    measurement["normal_mean_flow_loss"] = 0.22006
    gate = protocol.evaluate_gate_criteria(**measurement)
    assert gate["criteria"][2]["passed"] is False

    # criterion 4: state sensitivity ceiling
    measurement = _passing_measurement()
    measurement["state_shuffle_gradient_ratios"] = dict(UNIFORM_STATE_RATIOS)
    gate = protocol.evaluate_gate_criteria(**measurement)
    assert gate["criteria"][3]["passed"] is False

    # criterion 5: image sensitivity floor
    measurement = _passing_measurement()
    measurement["image_shuffle_gradient_ratios"] = dict(UNIFORM_IMAGE_RATIOS)
    gate = protocol.evaluate_gate_criteria(**measurement)
    assert gate["criteria"][4]["passed"] is False

    # criterion 6: dominance ceiling
    measurement = _passing_measurement()
    measurement["state_shuffle_gradient_ratios"] = {
        "action_expert": 2.1,
        "state_projector": 2.3,
        "action_io_projections": 2.0,
    }
    measurement["image_shuffle_gradient_ratios"] = dict(UNIFORM_IMAGE_RATIOS)
    gate = protocol.evaluate_gate_criteria(**measurement)
    assert gate["criteria"][5]["passed"] is False


def test_treatment_contract_is_consumable_by_the_feature() -> None:
    from rosetta_reality.vla.visual_conditioning import profile_from_plan

    profile = profile_from_plan(
        {"visual_conditioning_contract": dict(protocol.VCD_CONTRACT)}
    )
    assert profile.dropout_probability == 0.5
    assert profile.generator_seed == 20260828


def test_candidate_plan_validates_against_the_current_tree() -> None:
    assert protocol.validate_vcdropout_plan(_candidate_plan()) == protocol.VCD_PLAN_ID


def test_candidate_identity_is_fail_closed() -> None:
    mutations = (
        lambda plan: plan.update({"plan_id": "m2-smolvla450m-other-001"}),
        lambda plan: plan.update({"run_name": "other-run"}),
        lambda plan: plan.update({"status": "registered"}),
    )
    for mutate in mutations:
        plan = _candidate_plan()
        mutate(plan)
        with pytest.raises(ValueError):
            protocol.validate_vcdropout_plan(plan)


def test_treatment_stack_is_single_axis() -> None:
    plan = _candidate_plan()
    plan["features"].append({"name": "state_robustness_jitter"})
    plan["state_robustness_contract"] = {
        "profile": "normalized_gaussian_state_jitter"
    }
    with pytest.raises(ValueError, match="forbidden"):
        protocol.validate_vcdropout_plan(plan)

    plan = _candidate_plan()
    plan["features"].remove({"name": "state_conditioning_dropout"})
    with pytest.raises(ValueError, match="single-axis"):
        protocol.validate_vcdropout_plan(plan)

    plan = _candidate_plan()
    plan["features"].append({"name": "horizon_weight_profile"})
    plan["loss_contract"] = {"profile": "first_action_only"}
    with pytest.raises(ValueError, match="forbidden"):
        protocol.validate_vcdropout_plan(plan)

    plan = _candidate_plan()
    plan["tracking"] = {"project": "p", "space_id": "s"}
    plan["features"].append({"name": "trackio_logging"})
    with pytest.raises(ValueError, match="outside the registered stack"):
        protocol.validate_vcdropout_plan(plan)

    plan = _candidate_plan()
    plan["features"] = [
        {"name": name}
        for name in (
            "train_only_statistics",
            "action_boundary_projection",
            "checkpoint_memory_trim",
            "gradient_clip_diagnostics",
            "checkpoint_metric_snapshot",
        )
    ]
    with pytest.raises(ValueError, match="missing"):
        protocol.validate_vcdropout_plan(plan)


def test_treatment_contract_drift_fails_closed() -> None:
    plan = _candidate_plan()
    plan["visual_conditioning_contract"]["dropout_probability"] = 0.3
    with pytest.raises(ValueError, match="treatment contract"):
        protocol.validate_vcdropout_plan(plan)

    plan = _candidate_plan()
    plan["visual_conditioning_contract"]["generator_seed"] = 1
    with pytest.raises(ValueError, match="treatment contract"):
        protocol.validate_vcdropout_plan(plan)


def test_training_contract_drift_fails_closed() -> None:
    plan = _candidate_plan()
    plan["training"]["batch_size"] = 32
    with pytest.raises(ValueError, match="training contract"):
        protocol.validate_vcdropout_plan(plan)

    plan = _candidate_plan()
    plan["training"]["optimizer"]["lr"] = 5.0e-5
    with pytest.raises(ValueError):
        protocol.validate_vcdropout_plan(plan)


def test_validation_protocol_drift_fails_closed() -> None:
    plan = _candidate_plan()
    plan["validation"]["episodes"] = list(protocol.TRAIN_EPISODES[:5])
    plan["validation"]["total_samples"] = 5
    with pytest.raises(ValueError, match="validation protocol"):
        protocol.validate_vcdropout_plan(plan)


def test_implementation_inventory_is_fail_closed() -> None:
    plan = _candidate_plan()
    del plan["implementation_files"][
        "scripts/gate_smolvla_vcdropout_visual_conditioning.py"
    ]
    with pytest.raises(ValueError, match="missing"):
        protocol.validate_vcdropout_plan(plan)

    plan = _candidate_plan()
    plan["implementation_files"]["scripts/run_smolvla_v2.py"] = "0" * 64
    with pytest.raises(ValueError, match="changed"):
        protocol.validate_vcdropout_plan(plan)

    plan = _candidate_plan()
    plan["implementation_files"][protocol.VISUAL_CONDITIONING_IMPLEMENTATION] = "0" * 64
    with pytest.raises(ValueError, match="changed"):
        protocol.validate_vcdropout_plan(plan)


def test_post_training_entry_points_bind_the_shared_protocol() -> None:
    import importlib

    for name in (
        "smolvla_vcdropout_validate",
        "select_smolvla_vcdropout_checkpoint",
        "export_smolvla_vcdropout",
        "gate_smolvla_vcdropout_visual_conditioning",
    ):
        module = importlib.import_module(name)
        assert module.protocol.VCD_PLAN_ID == protocol.VCD_PLAN_ID
