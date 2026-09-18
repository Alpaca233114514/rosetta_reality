"""Training-only SmolVLA visual-conditioning contract tests."""

from __future__ import annotations

import types
from pathlib import Path

import pytest
import torch

from rosetta_reality.experiment import file_sha256
from rosetta_reality.vla.visual_conditioning import (
    PROFILE_SAMPLEWISE_NORMALIZED_STATE_DROPOUT,
    UPSTREAM_IMPLEMENTATION_SHA256,
    install_visual_conditioning_profile,
    profile_from_plan,
    restore_visual_conditioning_profile,
)


def _plan(**overrides) -> dict:
    contract = {
        "profile": PROFILE_SAMPLEWISE_NORMALIZED_STATE_DROPOUT,
        "dropout_probability": 0.5,
        "generator_seed": 20260828,
        "input_space": "train_normalized_observation_state",
        "granularity": "whole_sample",
        "replacement": "normalized_zero",
        "rescale_retained_state": False,
        "generator": "dedicated_cpu_generator",
        "training_only": True,
        "target_semantics": "unchanged_absolute_expert_action",
        "upstream_implementation_sha256": UPSTREAM_IMPLEMENTATION_SHA256,
    }
    contract.update(overrides)
    return {"visual_conditioning_contract": contract}


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

    module = types.ModuleType("fake_visual_conditioning_modeling")
    module.__file__ = str(source)
    module.SmolVLAPolicy = SmolVLAPolicy
    return module


def test_profile_requires_the_exact_single_axis_contract() -> None:
    profile = profile_from_plan(_plan())
    assert profile.dropout_probability == pytest.approx(0.5)
    assert profile.granularity == "whole_sample"
    assert profile.rescale_retained_state is False

    with pytest.raises(ValueError, match="different upstream"):
        profile_from_plan(_plan(upstream_implementation_sha256="0" * 64))
    with pytest.raises(ValueError, match="finite and in"):
        profile_from_plan(_plan(dropout_probability=1.0))
    with pytest.raises(ValueError, match="granularity"):
        profile_from_plan(_plan(granularity="per_dimension"))
    with pytest.raises(ValueError, match="must not be rescaled"):
        profile_from_plan(_plan(rescale_retained_state=True))
    with pytest.raises(ValueError, match="generator seed"):
        profile_from_plan(_plan(generator_seed=-1))


def test_dropout_is_whole_sample_training_only_and_preserves_global_rng(
    tmp_path: Path,
) -> None:
    module = _module(tmp_path)
    profile = profile_from_plan(_plan())
    install_visual_conditioning_profile(
        module,
        profile,
        upstream_sha256=file_sha256(Path(module.__file__)),
    )
    policy = module.SmolVLAPolicy()
    state = torch.arange(1, 57, dtype=torch.float32).reshape(4, 14)
    batch = {"observation.state": state}
    global_rng_before = torch.random.get_rng_state().clone()

    policy.forward(batch)

    assert torch.equal(torch.random.get_rng_state(), global_rng_before)
    assert policy.seen_state is not None
    dropped = (policy.seen_state == 0.0).all(dim=1)
    assert int(dropped.sum()) == 2
    assert torch.equal(policy.seen_state[~dropped], state[~dropped])
    assert torch.equal(batch["observation.state"], state)

    policy.training = False
    policy.forward(batch)
    assert torch.equal(policy.seen_state, state)
    restore_visual_conditioning_profile(module)


def test_dropout_fails_closed_on_degenerate_batch_or_duplicate_install(
    tmp_path: Path,
) -> None:
    module = _module(tmp_path)
    profile = profile_from_plan(_plan())
    source_sha = file_sha256(Path(module.__file__))
    install_visual_conditioning_profile(module, profile, upstream_sha256=source_sha)
    with pytest.raises(RuntimeError, match="already installed"):
        install_visual_conditioning_profile(module, profile, upstream_sha256=source_sha)
    policy = module.SmolVLAPolicy()
    with pytest.raises(ValueError, match="at least two"):
        policy.forward({"observation.state": torch.ones(1, 14)})
    restore_visual_conditioning_profile(module)


def test_dropout_fails_closed_on_changed_upstream(tmp_path: Path) -> None:
    module = _module(tmp_path)
    with pytest.raises(ValueError, match="differs from the registered"):
        install_visual_conditioning_profile(
            module,
            profile_from_plan(_plan()),
            upstream_sha256="0" * 64,
        )
