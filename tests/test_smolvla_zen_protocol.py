"""Focused preregistration tests for the Zen formal program (no weights, no GPU).

These act as the boot-time canary on the training host: every case either
proves the frozen invariants still bind the working tree or fails closed
before any paid CUDA work starts.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for candidate in (str(REPOSITORY_ROOT / "src"), str(SCRIPTS_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import smolvla_zen_protocol as protocol  # type: ignore[import-not-found]  # noqa: E402

UNIFORM_PLAN = (
    REPOSITORY_ROOT
    / "configs/vla/smolvla_450m_aloha_insertion_zen_cuda_b64_uniform_002.yaml"
)
FIRSTACTION_PLAN = (
    REPOSITORY_ROOT
    / "configs/vla/smolvla_450m_aloha_insertion_zen_cuda_b64_firstaction_001.yaml"
)


def test_registry_roles_are_distinct_and_complete() -> None:
    roles = {spec["role"] for spec in protocol.ZEN_SPECS.values()}
    assert roles == {"control", "treatment"}
    assert all(spec["horizon_feature_declared"] is not None for spec in protocol.ZEN_SPECS.values())


def test_uniform_plan_binds_the_current_tree() -> None:
    plan, plan_id = protocol.resolve_plan(UNIFORM_PLAN)
    assert plan_id == "m2-smolvla450m-zen-uniform-002"
    assert protocol.ZEN_SPECS[plan_id]["role"] == "control"


def test_firstaction_plan_binds_the_current_tree() -> None:
    plan, plan_id = protocol.resolve_plan(FIRSTACTION_PLAN)
    assert plan_id == "m2-smolvla450m-zen-firstaction-001"
    assert protocol.ZEN_SPECS[plan_id]["role"] == "treatment"


def _features(plan: dict) -> list[str]:
    return [
        declaration["name"]
        for declaration in plan["features"]
        if isinstance(declaration, dict)
    ]


def test_control_must_not_declare_the_horizon_feature() -> None:
    plan, _ = protocol.resolve_plan(UNIFORM_PLAN)
    features = _features(plan) + ["horizon_weight_profile"]
    with pytest.raises(ValueError, match="contradicts"):
        protocol.validate_zen_plan(plan, feature_names=features)


def test_treatment_loss_contract_is_fail_closed() -> None:
    plan, _ = protocol.resolve_plan(FIRSTACTION_PLAN)
    broken = copy.deepcopy(plan)
    broken["loss_contract"]["profile"] = "uniform_everything"
    with pytest.raises(ValueError, match="loss contract"):
        protocol.validate_zen_plan(broken)


def test_implementation_pin_drift_fails_closed() -> None:
    plan, _ = protocol.resolve_plan(UNIFORM_PLAN)
    broken = copy.deepcopy(plan)
    first_key = sorted(broken["implementation_files"])[0]
    broken["implementation_files"][first_key] = "0" * 64
    with pytest.raises(ValueError, match="drifted|changed"):
        protocol.validate_zen_plan(broken)


def test_validation_split_stays_disjoint() -> None:
    assert set(protocol.VALIDATION_EPISODES).isdisjoint(protocol.TRAIN_EPISODES)
    assert set(protocol.HIDDEN_TEST_EPISODES).isdisjoint(protocol.TRAIN_EPISODES)
    assert set(protocol.HIDDEN_TEST_EPISODES).isdisjoint(protocol.VALIDATION_EPISODES)
