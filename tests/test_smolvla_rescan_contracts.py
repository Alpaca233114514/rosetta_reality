"""Fail-closed counterexamples for declared versus executed training behavior."""

import copy

import pytest
from test_smolvla_training_plan_schema import _base_plan

from rosetta_reality.vla.training.features import FEATURE_FACTORIES
from rosetta_reality.vla.training.plan import validate_plan_structure


def validate(plan):
    return validate_plan_structure(plan, known_features=FEATURE_FACTORIES)


def test_repaired_training_cannot_omit_action_boundary():
    plan = _base_plan()
    plan["features"] = [f for f in plan["features"] if f["name"] != "action_boundary_projection"]
    with pytest.raises(ValueError, match="action_boundary_projection"):
        validate(plan)


@pytest.mark.parametrize(
    "contract",
    [
        "loss_contract",
        "state_robustness_contract",
        "visual_conditioning_contract",
        "vision_front_end_contract",
        "image_key_scene_contract",
    ],
)
def test_learning_contract_cannot_be_silently_ignored(contract):
    plan = _base_plan()
    plan[contract] = {"profile": "declared_but_not_installed"}
    with pytest.raises(ValueError, match="feature"):
        validate(plan)


@pytest.mark.parametrize("field", ["optimizer", "scheduler", "policy"])
def test_smoke_overlay_cannot_be_silently_ignored(field):
    plan = _base_plan()
    plan["optimizer_smoke"] = {field: copy.deepcopy(plan["training"][field])}
    with pytest.raises(ValueError, match="optimizer_smoke"):
        validate(plan)


def test_smoke_episode_identity_cannot_be_truncated_from_float():
    plan = _base_plan()
    plan["optimizer_smoke"] = {
        "episodes": [49.8],
        "batch_size": 1,
        "steps": 2,
        "run_name": "unit-smoke",
    }
    with pytest.raises(ValueError, match="episode"):
        validate(plan)


def test_training_cannot_disable_required_checkpoints():
    plan = _base_plan()
    plan["training"]["save_checkpoint"] = False
    with pytest.raises(ValueError, match="checkpoint"):
        validate(plan)


@pytest.mark.parametrize("key", ["chunk_size", "n_obs_steps", "n_action_steps", "video_backend"])
def test_unimplemented_training_policy_overlay_cannot_change_frame_contract(key):
    plan = _base_plan()
    plan["training"]["policy"][key] = 2
    with pytest.raises(ValueError, match="policy overlay"):
        validate(plan)
