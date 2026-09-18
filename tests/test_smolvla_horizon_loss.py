"""Temporal horizon-weighting contract tests (no model weights, no data)."""

from __future__ import annotations

import types
from pathlib import Path

import pytest
import torch

from rosetta_reality.experiment import file_sha256
from rosetta_reality.vla.horizon_loss import (
    NORMALIZATION_SELECTED_VALID_MEAN,
    PROFILE_FIRST_ACTION_ONLY,
    UPSTREAM_IMPLEMENTATION_SHA256,
    HorizonWeightProfile,
    install_horizon_weight_profile,
    profile_from_plan,
    restore_horizon_weight_profile,
)
from scripts.select_smolvla_aster_checkpoint import (
    _apply_control_acceptance,
    _mark_public_sync_complete,
)


def _plan(overrides: dict | None = None) -> dict:
    plan = {
        "loss_contract": {
            "profile": PROFILE_FIRST_ACTION_ONLY,
            "chunk_size": 50,
            "normalization": NORMALIZATION_SELECTED_VALID_MEAN,
            "upstream_implementation_sha256": UPSTREAM_IMPLEMENTATION_SHA256,
        }
    }
    if overrides:
        plan["loss_contract"].update(overrides)
    return plan


def test_first_action_profile_contract() -> None:
    profile = profile_from_plan(_plan(), 50)
    assert profile.name == PROFILE_FIRST_ACTION_ONLY
    assert profile.chunk_size == 50
    assert profile.normalization == NORMALIZATION_SELECTED_VALID_MEAN
    assert len(profile.weights) == 50
    assert profile.weights[0] == 1.0
    assert sum(profile.weights[1:]) == 0.0


def test_profile_rejects_unregistered_contracts() -> None:
    with pytest.raises(ValueError, match="no temporal loss contract"):
        profile_from_plan({}, 50)
    with pytest.raises(ValueError, match="different upstream"):
        profile_from_plan(_plan({"upstream_implementation_sha256": "0" * 64}), 50)
    with pytest.raises(ValueError, match="chunk size"):
        profile_from_plan(_plan({"chunk_size": 49}), 50)
    with pytest.raises(ValueError, match="Unsupported temporal weight profile"):
        profile_from_plan(_plan({"profile": "uniform"}), 50)
    with pytest.raises(ValueError, match="normalization"):
        profile_from_plan(_plan({"normalization": "upstream_uniform_mean"}), 50)


def test_profile_value_validation() -> None:
    with pytest.raises(ValueError, match="all zero"):
        HorizonWeightProfile(
            name=PROFILE_FIRST_ACTION_ONLY, chunk_size=3, weights=(0.0, 0.0, 0.0)
        )
    with pytest.raises(ValueError, match="finite"):
        HorizonWeightProfile(
            name=PROFILE_FIRST_ACTION_ONLY, chunk_size=3, weights=(1.0, -0.5, 0.0)
        )
    with pytest.raises(ValueError, match="full action chunk"):
        HorizonWeightProfile(
            name=PROFILE_FIRST_ACTION_ONLY, chunk_size=3, weights=(1.0, 0.0)
        )


def _fake_modeling_module(tmp_path: Path) -> tuple[types.ModuleType, type]:
    source = tmp_path / "fake_modeling.py"
    source.write_text("# fake pinned upstream implementation\n", encoding="utf-8")

    class VLAFlowMatching:
        def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise, time):
            del images, img_masks, lang_tokens, lang_masks, state, noise, time
            return actions.square()

    class SmolVLAPolicy:
        def __init__(self) -> None:
            self.model = VLAFlowMatching()

        def forward(self, batch, noise=None, time=None, reduction="mean"):
            losses = self.model.forward(
                None,
                None,
                None,
                None,
                None,
                batch["action"],
                noise,
                time,
            )
            action_is_pad = batch.get("action_is_pad")
            if action_is_pad is not None:
                losses = losses * (~action_is_pad).unsqueeze(-1)
            if reduction == "none":
                if action_is_pad is None:
                    loss = losses.mean(dim=(1, 2))
                else:
                    denominator = ((~action_is_pad).sum(dim=1) * losses.shape[-1]).clamp_min(1)
                    loss = losses.sum(dim=(1, 2)) / denominator
            else:
                if action_is_pad is None:
                    loss = losses.mean()
                else:
                    denominator = ((~action_is_pad).sum() * losses.shape[-1]).clamp_min(1)
                    loss = losses.sum() / denominator
            return loss, {"loss": float(loss.detach().mean().item())}

    module = types.ModuleType("fake_modeling")
    module.__file__ = str(source)
    module.VLAFlowMatching = VLAFlowMatching
    module.SmolVLAPolicy = SmolVLAPolicy
    return module, VLAFlowMatching


def test_weighted_forward_keeps_only_first_action(tmp_path: Path) -> None:
    module, model_class = _fake_modeling_module(tmp_path)
    profile = HorizonWeightProfile(
        name=PROFILE_FIRST_ACTION_ONLY, chunk_size=3, weights=(1.0, 0.0, 0.0)
    )
    install_horizon_weight_profile(
        module, profile, upstream_sha256=file_sha256(Path(module.__file__))
    )
    actions = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    losses = model_class().forward(None, None, None, None, None, actions, None, None)
    assert torch.equal(losses[0, 0], torch.tensor([1.0, 4.0]))
    assert bool((losses[0, 1:] == 0.0).all())
    loss, loss_dict = module.SmolVLAPolicy().forward({"action": actions})
    assert loss.item() == pytest.approx(2.5)
    assert loss_dict["loss"] == pytest.approx(2.5)
    restore_horizon_weight_profile(module)


def test_first_action_mean_preserves_scale_with_padding_and_gradients(
    tmp_path: Path,
) -> None:
    module, _ = _fake_modeling_module(tmp_path)
    profile = HorizonWeightProfile(
        name=PROFILE_FIRST_ACTION_ONLY, chunk_size=3, weights=(1.0, 0.0, 0.0)
    )
    install_horizon_weight_profile(
        module, profile, upstream_sha256=file_sha256(Path(module.__file__))
    )
    actions = torch.tensor(
        [[[1.0], [10.0], [20.0]], [[2.0], [30.0], [40.0]]],
        requires_grad=True,
    )
    action_is_pad = torch.tensor(
        [[False, False, False], [False, True, True]], dtype=torch.bool
    )
    policy = module.SmolVLAPolicy()
    loss, _ = policy.forward(
        {"action": actions, "action_is_pad": action_is_pad}, reduction="mean"
    )
    per_sample, _ = policy.forward(
        {"action": actions, "action_is_pad": action_is_pad}, reduction="none"
    )
    assert loss.item() == pytest.approx((1.0**2 + 2.0**2) / 2.0)
    assert torch.equal(per_sample, torch.tensor([1.0, 4.0]))
    loss.backward()
    assert torch.equal(
        actions.grad,
        torch.tensor([[[1.0], [0.0], [0.0]], [[2.0], [0.0], [0.0]]]),
    )
    restore_horizon_weight_profile(module)


def test_install_guards_double_install_and_upstream_change(tmp_path: Path) -> None:
    module, _ = _fake_modeling_module(tmp_path)
    source_sha = file_sha256(Path(module.__file__))
    profile = HorizonWeightProfile(
        name=PROFILE_FIRST_ACTION_ONLY, chunk_size=3, weights=(1.0, 0.0, 0.0)
    )
    with pytest.raises(ValueError, match="differs from the registered"):
        install_horizon_weight_profile(module, profile, upstream_sha256="0" * 64)
    install_horizon_weight_profile(module, profile, upstream_sha256=source_sha)
    with pytest.raises(RuntimeError, match="already installed"):
        install_horizon_weight_profile(module, profile, upstream_sha256=source_sha)
    restore_horizon_weight_profile(module)
    with pytest.raises(RuntimeError, match="No temporal weight profile"):
        restore_horizon_weight_profile(module)


def test_aster_selection_requires_improvement_over_faust_control() -> None:
    plan = {
        "validation": {"primary_selection_metric": "first_action_mae"},
        "control_reference": {
            "control_run": "m2-smolvla450m-faust-b8-002",
            "plan_sha256": "a" * 64,
            "selection_report_sha256": "b" * 64,
        },
    }
    control_plan = {"run_name": "m2-smolvla450m-faust-b8-002"}
    control_report = {
        "status": "passed",
        "stage": "smolvla_formal_checkpoint_selection",
        "experiment_id": "experiment",
        "formal_plan_sha256": "a" * 64,
        "selected": {"metrics": {"first_action_mae": 0.027}},
    }
    payload = {
        "status": "passed",
        "experiment_id": "experiment",
        "selected": {"metrics": {"first_action_mae": 0.022}},
        "acceptance": {"generic_gate": True},
    }

    assert _apply_control_acceptance(payload, plan, control_plan, control_report)
    assert payload["status"] == "passed"
    assert payload["control_comparison"]["control_value"] == pytest.approx(0.027)

    payload["selected"]["metrics"]["first_action_mae"] = 0.028
    assert not _apply_control_acceptance(payload, plan, control_plan, control_report)
    assert payload["status"] == "rejected"
    assert (
        payload["acceptance"][
            "validation_first_action_mae_improves_over_faust_control"
        ]
        is False
    )


def test_aster_selection_records_real_public_sync_provenance() -> None:
    payload = {
        "trackio_sync_report_sha256": "a" * 64,
        "trackio_project_snapshot_sha256": "b" * 64,
        "trackio_synced_run": {"run_name": "aster"},
    }
    plan = {
        "loss_contract": {
            "profile": PROFILE_FIRST_ACTION_ONLY,
            "normalization": NORMALIZATION_SELECTED_VALID_MEAN,
        }
    }

    _mark_public_sync_complete(payload, plan)

    assert payload["trackio_sync_report_sha256"] == "a" * 64
    assert payload["public_sync_performed"] is True
    assert payload["trackio_delivery_status"] == "public_checkpoint_sync_complete"

    with pytest.raises(ValueError, match="public sync provenance"):
        _mark_public_sync_complete({}, plan)
