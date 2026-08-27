"""Version-2 plan schema tests (no model weights, no data, no training)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from rosetta_reality.experiment import file_sha256
from rosetta_reality.vla.training.features import FEATURE_FACTORIES
from rosetta_reality.vla.training.plan import (
    load_v2_plan,
    validate_optimizer_contract,
    validate_plan_structure,
)

OPTIMIZER = {
    "type": "adamw",
    "lr": 1.0e-4,
    "betas": [0.9, 0.95],
    "eps": 1.0e-8,
    "weight_decay": 1.0e-10,
    "grad_clip_norm": 10.0,
}
SCHEDULER = {
    "type": "cosine_decay_with_warmup",
    "num_warmup_steps": 125,
    "num_decay_steps": 2500,
    "peak_lr": 1.0e-4,
    "decay_lr": 2.5e-6,
}


def _hex(seed: int) -> str:
    return f"{seed:064x}"


def _base_plan() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "role": "vla",
        "status": "preregistered",
        "plan_id": "m2-smolvla450m-v2-example-001",
        "run_name": "m2-smolvla450m-v2-example-001",
        "parent_experiment": {
            "config": "configs/vla/example.yaml",
            "sha256": _hex(1),
            "experiment_id": "m2-example",
        },
        "training": {
            "episodes": [49, 4, 23],
            "batch_size": 8,
            "steps": 2500,
            "save_freq": 500,
            "save_checkpoint": True,
            "log_freq": 10,
            "num_workers": 0,
            "persistent_workers": False,
            "checkpoint_steps": [500, 1000, 1500, 2000, 2500],
            "eval_split": 0.0,
            "validation_gradients": False,
            "hidden_test_loaded": False,
            "optimizer": copy.deepcopy(OPTIMIZER),
            "scheduler": copy.deepcopy(SCHEDULER),
            "policy": {
                "empty_cameras": 2,
                "compile_model": False,
                "compile_mode": "default",
                "skip_fully_masked_camera_encoding": True,
            },
        },
        "validation": {
            "episodes": [22, 13],
            "frame_offsets": [0, 125],
            "total_samples": 4,
            "hidden_test_loaded": False,
        },
        "resources": {
            "memory_limit": "8g",
            "memory_swap_limit": "8g",
            "mixed_precision": "bf16",
            "cpu_limit": 2,
            "checkpoint_memory_trim": True,
        },
        "features": [
            {"name": "trackio_logging"},
            {"name": "train_only_statistics"},
            {"name": "masked_camera_skip"},
            {"name": "action_boundary_projection"},
            {"name": "checkpoint_memory_trim"},
        ],
        "tracking": {"project": "rosetta-reality-vla", "space_id": "unit/space"},
        "prerequisites": {
            "gate1": {"path": "runs/example/gate1.json", "sha256": _hex(2)},
        },
        "normalization": {
            "source_split": "train",
            "report": "runs/example/normalization.json",
            "report_sha256": _hex(3),
            "dataset_view_manifest": "runs/example/view_manifest.json",
            "dataset_view_manifest_sha256": _hex(4),
            "validation_episodes_loaded": False,
            "hidden_test_loaded": False,
        },
        "implementation_files": {
            "src/rosetta_reality/vla/training/plan.py": _hex(5),
        },
        "stop_conditions": ["nonfinite_loss_gradient_or_action"],
        "hidden_test_loaded": False,
    }


def _validated(plan: dict[str, Any]) -> list[str]:
    return validate_plan_structure(plan, known_features=FEATURE_FACTORIES)


def test_base_plan_is_valid_and_reports_feature_order() -> None:
    names = _validated(_base_plan())
    assert names == [
        "trackio_logging",
        "train_only_statistics",
        "masked_camera_skip",
        "action_boundary_projection",
        "checkpoint_memory_trim",
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda p: p.update(schema_version=1), "schema_version"),
        (lambda p: p.update(status="draft"), "preregistered"),
        (lambda p: p.update(run_name="Bad Name"), "path-safe"),
        (lambda p: p["training"].update(save_freq=495), "multiple of the log frequency"),
        (lambda p: p["training"].update(steps=2502), "divisible by four"),
        (lambda p: p["training"].update(checkpoint_steps=[500]), "save-frequency grid"),
        (lambda p: p["training"]["optimizer"].update(lr=-1.0), "optimizer contract"),
        (lambda p: p["training"]["scheduler"].update(num_decay_steps=2499), "schedule"),
        (lambda p: p["training"]["scheduler"].update(num_warmup_steps=2500), "schedule"),
        (lambda p: p["training"].update(validation_gradients=True), "validation gradients"),
        (lambda p: p["training"].update(hidden_test_loaded=True), "hidden-test"),
        (lambda p: p["training"].update(num_workers=0, persistent_workers=True), "worker"),
        (lambda p: p["validation"].update(total_samples=5), "total samples"),
        (lambda p: p["resources"].update(memory_swap_limit="4g"), "memory"),
        (lambda p: p["resources"].update(checkpoint_memory_trim="yes"), "checkpoint_memory_trim"),
        (lambda p: p["features"].append({"name": "not_a_feature"}), "Unknown training feature"),
        (lambda p: p["features"].append({"name": "trackio_logging"}), "declared twice"),
        (
            lambda p: p["features"].append({"name": "horizon_weight_profile"}),
            "requires a loss contract",
        ),
        (
            lambda p: p["features"].append({"name": "state_robustness_jitter"}),
            "state-robustness contract",
        ),
        (lambda p: p["features"].append({"name": "fixed_frame_sampler"}), "'phase'"),
        (lambda p: p.update(hidden_test_loaded=True), "hidden-test"),
        (lambda p: p["normalization"].update(report="/absolute/path.json"), "repository-relative"),
        (
            lambda p: p["prerequisites"]["gate1"].update(path="../escape.json"),
            "repository-relative",
        ),
        (lambda p: p["prerequisites"]["gate1"].update(sha256="short"), "SHA-256"),
        (lambda p: p.update(implementation_files={}), "implementation checksum inventory"),
        (lambda p: p.update(stop_conditions=[]), "stop conditions"),
    ],
)
def test_invalid_mutations_fail_closed(mutation, message) -> None:
    plan = _base_plan()
    mutation(plan)
    with pytest.raises(ValueError, match=message):
        _validated(plan)


def test_masked_camera_feature_requires_policy_flag() -> None:
    plan = _base_plan()
    plan["training"]["policy"]["skip_fully_masked_camera_encoding"] = False
    with pytest.raises(ValueError, match="skip_fully_masked_camera_encoding"):
        _validated(plan)


def test_checkpoint_trim_feature_requires_resource_flag() -> None:
    plan = _base_plan()
    plan["resources"]["checkpoint_memory_trim"] = False
    with pytest.raises(ValueError, match="checkpoint_memory_trim"):
        _validated(plan)


def test_trackio_feature_requires_tracking_section() -> None:
    plan = _base_plan()
    del plan["tracking"]
    with pytest.raises(ValueError, match="'tracking'"):
        _validated(plan)


def test_fixed_frame_sampler_validates_phase_parameter() -> None:
    plan = _base_plan()
    plan["features"].append({"name": "fixed_frame_sampler", "phase": "formal"})
    with pytest.raises(ValueError, match="phase"):
        _validated(plan)
    plan["features"][-1] = {"name": "fixed_frame_sampler", "phase": "smoke"}
    _validated(plan)


def test_monitoring_must_keep_the_quarter_sleep_policy() -> None:
    plan = _base_plan()
    plan["monitoring"] = {
        "policy": "sleep_between_quarter_checkpoints",
        "estimated_total_minutes": 100,
        "wake_fractions": [0.25, 0.5, 0.75, 1.0],
        "wake_steps": [500, 1000, 1500, 2000, 2500],
        "blocking_command": "sleep",
        "sleep_poll_seconds": 60,
        "hidden_test_loaded": False,
    }
    with pytest.raises(ValueError, match="quarter-only sleep policy"):
        _validated(plan)
    plan["monitoring"]["sleep_poll_seconds"] = 300
    plan["monitoring"]["wake_steps"] = [625, 1250, 1875, 2500]
    plan["training"]["save_freq"] = 625
    plan["training"]["log_freq"] = 25
    plan["training"]["checkpoint_steps"] = [625, 1250, 1875, 2500]
    _validated(plan)


def test_plan_inheritance_is_checksum_pinned_and_shallow(tmp_path: Path) -> None:
    base = {"schema_version": 2, "shared": {"a": 1}}
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(base), encoding="utf-8")
    child = {
        "extends": {"config": "base.yaml", "sha256": file_sha256(base_path)},
        "shared": {"b": 2},
    }
    child_path = tmp_path / "child.yaml"
    child_path.write_text(yaml.safe_dump(child), encoding="utf-8")

    merged = load_v2_plan(child_path, tmp_path)
    assert merged["shared"] == {"a": 1, "b": 2}
    assert merged["plan_inheritance"] == {
        "config": "base.yaml",
        "sha256": file_sha256(base_path),
    }

    stale = copy.deepcopy(child)
    stale["extends"]["sha256"] = _hex(9)
    stale_path = tmp_path / "stale.yaml"
    stale_path.write_text(yaml.safe_dump(stale), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum changed"):
        load_v2_plan(stale_path, tmp_path)

    nested = {
        "extends": {"config": "child.yaml", "sha256": file_sha256(child_path)},
        "x": 1,
    }
    nested_path = tmp_path / "nested.yaml"
    nested_path.write_text(yaml.safe_dump(nested), encoding="utf-8")
    with pytest.raises(ValueError, match="Nested"):
        load_v2_plan(nested_path, tmp_path)


def test_optimizer_contract_matches_the_frozen_faust_validator() -> None:
    from scripts.run_smolvla_formal import _optimizer_contract

    training = {
        "steps": 2500,
        "optimizer": copy.deepcopy(OPTIMIZER),
        "scheduler": copy.deepcopy(SCHEDULER),
    }
    assert validate_optimizer_contract(training) == _optimizer_contract(training)
    assert validate_optimizer_contract({"steps": 5}) is None
