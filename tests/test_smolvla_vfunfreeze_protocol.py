"""Focused preregistration tests for the vision-front-end candidate (no weights).

These act as the boot-time canary on the training host: every case either
proves the frozen post-training invariants still bind the working tree or
fails closed before any paid CUDA work starts.  They mirror the vcdropout
protocol tests and never load model weights, datasets or accelerators.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for candidate in (str(REPOSITORY_ROOT / "src"), SCRIPTS_ROOT):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import smolvla_vcdropout_protocol as shared  # type: ignore[import-not-found]  # noqa: E402
import smolvla_vfunfreeze_protocol as protocol  # type: ignore[import-not-found]  # noqa: E402

from rosetta_reality.experiment import file_sha256  # noqa: E402

PLAN_PATH = (
    REPOSITORY_ROOT / "configs/vla/smolvla_450m_aloha_insertion_vfunfreeze_cuda_b32_003.yaml"
)


def _candidate_plan() -> dict:
    implementation = {
        relative: file_sha256(REPOSITORY_ROOT / relative)
        for relative in protocol.REQUIRED_IMPLEMENTATION_FILES
    }
    return {
        "schema_version": 2,
        "role": "vla",
        "status": "preregistered",
        "plan_id": protocol.VFU_PLAN_ID,
        "run_name": protocol.VFU_RUN_NAME,
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
            "vcdropout_checkpoint_used": False,
            "optimizer_state_reused": False,
        },
        "training": {
            "episodes": list(protocol.TRAIN_EPISODES),
            "batch_size": 32,
            "steps": protocol.FORMAL_STEPS,
            "save_freq": 158,
            "log_freq": 158,
            "checkpoint_steps": list(protocol.CHECKPOINT_STEPS),
            "eval_split": 0.0,
            "hidden_test_loaded": False,
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
                "num_warmup_steps": 62,
                "num_decay_steps": 632,
                "peak_lr": 1.0e-4,
                "decay_lr": 2.5e-6,
            },
        },
        "features": [
            {"name": "train_only_statistics"},
            {"name": "action_boundary_projection"},
            {"name": "checkpoint_memory_trim"},
            {"name": "vision_front_end_unfreeze"},
            {"name": "gradient_clip_diagnostics"},
            {"name": "checkpoint_metric_snapshot"},
        ],
        "vision_front_end_contract": copy.deepcopy(protocol.VFU_CONTRACT),
        "resources": {
            "memory_limit": "autodl_platform_container",
            "memory_swap_limit": "autodl_platform_container",
            "cpu_limit": 16,
            "mixed_precision": "bf16",
            "checkpoint_memory_trim": True,
        },
        "validation": {
            "episodes": list(protocol.VALIDATION_EPISODES),
            "frame_offsets": [0],
            "total_samples": len(protocol.VALIDATION_EPISODES),
            "hidden_test_loaded": False,
        },
        "monitoring": {
            "policy": "sleep_between_quarter_checkpoints",
            "wake_steps": list(protocol.CHECKPOINT_STEPS),
            "blocking_command": "sleep",
            "sleep_poll_seconds": 300,
            "hidden_test_loaded": False,
        },
        "prerequisites": {},
        "normalization": {
            "source_split": "train",
            "report": protocol.NORMALIZATION_REPORT_RELATIVE,
            "report_sha256": protocol.NORMALIZATION_REPORT_SHA256,
            "dataset_view_manifest": protocol.VIEW_MANIFEST_RELATIVE,
            "dataset_view_manifest_sha256": protocol.VIEW_MANIFEST_SHA256,
            "hidden_test_loaded": False,
        },
        "implementation_files": implementation,
        "stop_conditions": ["nonfinite_loss_gradient_or_action"],
        "hidden_test_loaded": False,
    }


def test_shared_immutable_identities_match_the_frozen_vcdropout_protocol() -> None:
    assert protocol.EXPERIMENT_ID == shared.EXPERIMENT_ID
    assert protocol.PARENT_CONFIG == shared.PARENT_CONFIG
    assert protocol.PARENT_SHA256 == shared.PARENT_SHA256
    assert protocol.RUNTIME_PROFILE_SHA256 == shared.RUNTIME_PROFILE_SHA256
    assert protocol.RUNTIME_PROFILE_ID == shared.RUNTIME_PROFILE_ID
    assert protocol.NORMALIZATION_REPORT_SHA256 == shared.NORMALIZATION_REPORT_SHA256
    assert protocol.NORMALIZATION_REPORT_RELATIVE == shared.NORMALIZATION_REPORT_RELATIVE
    assert protocol.VIEW_MANIFEST_SHA256 == shared.VIEW_MANIFEST_SHA256
    assert protocol.VIEW_MANIFEST_RELATIVE == shared.VIEW_MANIFEST_RELATIVE
    assert protocol.TRAIN_EPISODES == shared.TRAIN_EPISODES
    assert protocol.VALIDATION_EPISODES == shared.VALIDATION_EPISODES
    assert protocol.HIDDEN_TEST_EPISODES == shared.HIDDEN_TEST_EPISODES
    # The registered batch-32 fallback (amendment 2026-09-05) intentionally
    # changes the schedule length and checkpoint grid while preserving every
    # optimization coefficient; document the exact deviation instead of
    # asserting blind equality.
    assert protocol.FORMAL_STEPS == 632
    assert protocol.CHECKPOINT_STEPS == [158, 316, 474, 632]
    assert protocol.PINNED_OPTIMIZER_CONTRACT["optimizer"] == (
        shared.PINNED_OPTIMIZER_CONTRACT["optimizer"]
    )
    assert protocol.PINNED_OPTIMIZER_CONTRACT["scheduler"] == {
        **shared.PINNED_OPTIMIZER_CONTRACT["scheduler"],
        "num_warmup_steps": 62,
        "num_decay_steps": 632,
    }


def test_registered_repository_plan_validates() -> None:
    plan, plan_id = protocol.resolve_plan(PLAN_PATH)
    assert plan_id == protocol.VFU_PLAN_ID
    assert plan["run_name"] == protocol.VFU_RUN_NAME


def test_candidate_plan_accepts_the_registered_stack() -> None:
    assert protocol.validate_vfunfreeze_plan(_candidate_plan()) == protocol.VFU_PLAN_ID


def test_candidate_plan_rejects_forbidden_co_treatments() -> None:
    plan = _candidate_plan()
    plan["features"].append({"name": "state_conditioning_dropout"})
    with pytest.raises(ValueError, match="forbidden co-treatment"):
        protocol.validate_vfunfreeze_plan(plan)


def test_candidate_plan_rejects_a_changed_treatment_contract() -> None:
    plan = _candidate_plan()
    contract = copy.deepcopy(protocol.VFU_CONTRACT)
    contract["language_model"] = "trainable"
    plan["vision_front_end_contract"] = contract
    with pytest.raises(ValueError, match="treatment contract changed"):
        protocol.validate_vfunfreeze_plan(plan)


def test_candidate_plan_rejects_a_state_dropout_contract() -> None:
    plan = _candidate_plan()
    plan["visual_conditioning_contract"] = {"profile": "samplewise"}
    with pytest.raises(ValueError, match="state-dropout contract"):
        protocol.validate_vfunfreeze_plan(plan)


def test_candidate_plan_rejects_a_stale_implementation_pin() -> None:
    plan = _candidate_plan()
    plan["implementation_files"]["src/rosetta_reality/vla/vision_front_end.py"] = (
        "0" * 64
    )
    with pytest.raises(ValueError, match="Implementation file changed"):
        protocol.validate_vfunfreeze_plan(plan)


def test_candidate_plan_rejects_wrong_identity() -> None:
    plan = _candidate_plan()
    plan["plan_id"] = "m2-smolvla450m-something-else-002"
    with pytest.raises(ValueError, match="Unknown candidate plan identity"):
        protocol.validate_vfunfreeze_plan(plan)
