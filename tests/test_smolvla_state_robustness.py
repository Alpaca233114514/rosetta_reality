"""Training-only SmolVLA state-robustness contract tests."""

from __future__ import annotations

import types
from pathlib import Path

import pytest
import torch

from rosetta_reality.experiment import file_sha256
from rosetta_reality.vla.state_robustness import (
    PROFILE_NORMALIZED_GAUSSIAN_STATE_JITTER,
    UPSTREAM_IMPLEMENTATION_SHA256,
    install_state_robustness_profile,
    profile_from_plan,
    restore_state_robustness_profile,
)


def _plan(**overrides) -> dict:
    contract = {
        "profile": PROFILE_NORMALIZED_GAUSSIAN_STATE_JITTER,
        "normalized_standard_deviation": 0.05,
        "input_space": "train_normalized_observation_state",
        "training_only": True,
        "target_semantics": "unchanged_absolute_expert_action",
        "upstream_implementation_sha256": UPSTREAM_IMPLEMENTATION_SHA256,
    }
    contract.update(overrides)
    return {"state_robustness_contract": contract}


def _module(tmp_path: Path) -> types.ModuleType:
    source = tmp_path / "fake_modeling.py"
    source.write_text("# fake pinned upstream\n", encoding="utf-8")

    class SmolVLAPolicy:
        def __init__(self) -> None:
            self.training = True
            self.seen_state = None

        def forward(self, batch, noise=None, time=None, reduction="mean"):
            del noise, time, reduction
            self.seen_state = batch["observation.state"].clone()
            loss = self.seen_state.square().mean()
            return loss, {"loss": float(loss)}

    module = types.ModuleType("fake_modeling")
    module.__file__ = str(source)
    module.SmolVLAPolicy = SmolVLAPolicy
    return module


def test_profile_from_plan_is_training_only_and_normalized() -> None:
    profile = profile_from_plan(_plan())
    assert profile.name == PROFILE_NORMALIZED_GAUSSIAN_STATE_JITTER
    assert profile.normalized_standard_deviation == pytest.approx(0.05)
    assert profile.training_only is True

    with pytest.raises(ValueError, match="different upstream"):
        profile_from_plan(_plan(upstream_implementation_sha256="0" * 64))
    with pytest.raises(ValueError, match="normalized state space"):
        profile_from_plan(_plan(input_space="raw_joint_position"))
    with pytest.raises(ValueError, match="finite and in"):
        profile_from_plan(_plan(normalized_standard_deviation=0.0))


def test_jitter_changes_only_training_state_and_preserves_input_batch(tmp_path: Path) -> None:
    module = _module(tmp_path)
    profile = profile_from_plan(_plan())
    install_state_robustness_profile(
        module,
        profile,
        upstream_sha256=file_sha256(Path(module.__file__)),
    )
    policy = module.SmolVLAPolicy()
    state = torch.zeros(2, 14)
    batch = {"observation.state": state}
    torch.manual_seed(7)
    policy.forward(batch)
    assert policy.seen_state is not None
    assert not torch.equal(policy.seen_state, state)
    assert torch.equal(batch["observation.state"], state)

    policy.training = False
    policy.forward(batch)
    assert torch.equal(policy.seen_state, state)
    restore_state_robustness_profile(module)


def test_install_fails_closed_on_duplicate_or_changed_upstream(tmp_path: Path) -> None:
    module = _module(tmp_path)
    profile = profile_from_plan(_plan())
    with pytest.raises(ValueError, match="differs from the registered"):
        install_state_robustness_profile(module, profile, upstream_sha256="0" * 64)
    source_sha = file_sha256(Path(module.__file__))
    install_state_robustness_profile(module, profile, upstream_sha256=source_sha)
    with pytest.raises(RuntimeError, match="already installed"):
        install_state_robustness_profile(module, profile, upstream_sha256=source_sha)
    restore_state_robustness_profile(module)
