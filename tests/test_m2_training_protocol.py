from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rosetta_reality.experiment import load_experiment_config
from rosetta_reality.train.losses import (
    globally_normalized_scoped_first_action_loss,
    smooth_l1_action_loss,
)
from rosetta_reality.train.m2 import build_cached_policy
from scripts.train_m2 import (
    _checkpoint_due,
    _configured_action_loss,
    _early_phase_first_action_protocol,
    _model_action_loss,
    _to_cpu_tree,
    _training_device,
    _training_state_with_noise,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint_schedule_preserves_periodic_best_and_terminal_epochs() -> None:
    assert _checkpoint_due(5, every_epochs=5, improved=False, terminal_epoch=False)
    assert _checkpoint_due(3, every_epochs=5, improved=True, terminal_epoch=False)
    assert _checkpoint_due(2, every_epochs=5, improved=False, terminal_epoch=True)
    assert not _checkpoint_due(4, every_epochs=5, improved=False, terminal_epoch=False)


def test_checkpoint_schedule_rejects_nonpositive_interval() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _checkpoint_due(1, every_epochs=0, improved=False, terminal_epoch=False)


def test_training_device_defaults_to_cpu() -> None:
    context = {"experiment": {"resources": {}}}

    assert _training_device(context) == torch.device("cpu")


def test_checkpoint_tensor_tree_is_portable_cpu_data() -> None:
    source = {"nested": [torch.ones(2), (torch.zeros(1),)]}

    converted = _to_cpu_tree(source)

    assert converted is not source
    assert converted["nested"][0].device.type == "cpu"
    assert converted["nested"][1][0].device.type == "cpu"


def test_v007_state_dropout_is_training_only_and_preserves_tensor_contract() -> None:
    experiment = load_experiment_config(
        REPOSITORY_ROOT
        / "configs"
        / "experiments"
        / "m2_qwen08b_frozen_007_state_dropout_xpu.yaml",
        REPOSITORY_ROOT,
    )
    assert experiment["action_expert"]["state_dropout"] == 0.1
    policy = build_cached_policy(
        experiment,
        feature_dim=5120,
        state_dim=14,
        action_dim=14,
        chunk_size=100,
    )
    observations = {"features": torch.randn(2, 5120)}
    robot_state = torch.randn(2, 14)

    policy.train()
    torch.manual_seed(1)
    first = policy(observations, robot_state)
    torch.manual_seed(2)
    second = policy(observations, robot_state)
    assert first.shape == (2, 100, 14)
    assert bool(torch.isfinite(first).all())
    assert not torch.equal(first, second)

    policy.eval()
    first_eval = policy(observations, robot_state)
    second_eval = policy(observations, robot_state)
    assert torch.equal(first_eval, second_eval)


def test_training_state_noise_is_seeded_finite_and_zero_is_identity() -> None:
    state = torch.zeros(4, 14)
    assert _training_state_with_noise(state, {}) is state

    torch.manual_seed(7)
    first = _training_state_with_noise(
        state,
        {"state_noise_std_normalized": 0.05},
    )
    torch.manual_seed(7)
    second = _training_state_with_noise(
        state,
        {"state_noise_std_normalized": 0.05},
    )

    assert torch.equal(first, second)
    assert not torch.equal(first, state)
    assert bool(torch.isfinite(first).all())


@pytest.mark.parametrize("value", (-0.1, 1.0, float("inf"), float("nan")))
def test_training_state_noise_rejects_unsafe_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite and in"):
        _training_state_with_noise(
            torch.zeros(1, 14),
            {"state_noise_std_normalized": value},
        )


def test_execution_aligned_loss_preserves_legacy_and_matches_equal_mix() -> None:
    prediction = torch.tensor([[[2.0], [1.0]]], requires_grad=True)
    target = torch.zeros_like(prediction)
    legacy = smooth_l1_action_loss(prediction, target)
    zero_weight = smooth_l1_action_loss(
        prediction, target, first_action_weight=0.0
    )
    expected_first = torch.nn.functional.smooth_l1_loss(
        prediction[:, :1], target[:, :1]
    )
    equal_mix = smooth_l1_action_loss(
        prediction, target, first_action_weight=1.0
    )

    assert torch.equal(zero_weight, legacy)
    assert torch.equal(equal_mix, (legacy + expected_first) / 2.0)

    equal_mix.backward()
    assert prediction.grad is not None
    assert prediction.grad[0, 0].abs().item() > prediction.grad[0, 1].abs().item()
    assert prediction.grad[0, 1].abs().item() > 0.0


@pytest.mark.parametrize("weight", (-1.0, float("inf"), float("nan")))
def test_execution_aligned_loss_rejects_unsafe_weight(weight: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        smooth_l1_action_loss(
            torch.zeros(1, 2, 1),
            torch.zeros(1, 2, 1),
            first_action_weight=weight,
        )


def test_configured_action_loss_reads_experiment_weight() -> None:
    prediction = torch.tensor([[[2.0], [1.0]]])
    target = torch.zeros_like(prediction)

    configured = _configured_action_loss(
        prediction, target, {"first_action_loss_weight": 1.0}
    )
    direct = smooth_l1_action_loss(
        prediction, target, first_action_weight=1.0
    )

    assert torch.equal(configured, direct)


def test_scoped_first_action_loss_uses_fixed_global_normalization() -> None:
    prediction = torch.tensor(
        [[[2.0], [9.0]], [[4.0], [9.0]], [[6.0], [9.0]], [[8.0], [9.0]]],
        requires_grad=True,
    )
    target = torch.zeros_like(prediction)
    mask = torch.tensor([True, False, True, False])

    actual = globally_normalized_scoped_first_action_loss(
        prediction,
        target,
        mask,
        global_scale=2.0,
    )
    per_sample = torch.nn.functional.smooth_l1_loss(
        prediction[:, 0], target[:, 0], reduction="none"
    ).mean(dim=1)

    assert torch.equal(actual, per_sample[mask].mean())
    actual.backward()
    assert prediction.grad is not None
    assert prediction.grad[0, 0].abs().item() > 0.0
    assert prediction.grad[1, 0].abs().item() == 0.0
    assert prediction.grad[:, 1].abs().sum().item() == 0.0


def test_scoped_model_loss_keeps_empty_batch_finite_and_full_chunk_gradients() -> None:
    model = _StateEchoPolicy()
    observations = {"features": torch.tensor([[10.0], [20.0]])}
    state = torch.tensor([[1.0], [3.0]], requires_grad=True)
    target = torch.zeros(2, 1, 1)
    protocol = {
        "weight": 1.0,
        "maximum_frame_index_exclusive": 50,
        "global_scale": 9.88,
    }

    prediction, loss = _model_action_loss(
        model,
        observations,
        state,
        target,
        {},
        early_phase_first_action=protocol,
        frame_indices=torch.tensor([50, 100]),
    )

    full = smooth_l1_action_loss(prediction, target)
    assert torch.equal(loss, full / 2.0)
    loss.backward()
    assert state.grad is not None
    assert bool(state.grad.abs().gt(0).all())


class _EarlyPhaseDataset:
    def __init__(self) -> None:
        self.episode_ids = torch.tensor([1, 1, 1, 2, 2, 2])
        self.frame_indices = torch.tensor([0, 2, 4, 0, 2, 4])

    def __len__(self) -> int:
        return len(self.frame_indices)


def test_early_phase_protocol_binds_exact_episode_frame_scope() -> None:
    experiment = {
        "dataset": {"frame_stride": 2, "split": {"train": [1, 2]}},
        "training": {
            "early_phase_first_action_loss": {
                "weight": 1.0,
                "maximum_frame_index_exclusive": 4,
                "expected_selected_train_samples": 4,
            }
        },
    }

    protocol = _early_phase_first_action_protocol(experiment, _EarlyPhaseDataset())

    assert protocol is not None
    assert protocol["selected_train_samples"] == 4
    assert protocol["total_train_samples"] == 6
    assert protocol["global_scale"] == 1.5
    assert protocol["selected_frames_per_episode"] == [0, 2]


class _StateEchoPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(
        self, observations: dict[str, torch.Tensor], state: torch.Tensor
    ) -> torch.Tensor:
        self.calls.append((observations["features"].clone(), state.clone()))
        return state[:, None, :1]


def test_paired_loss_uses_same_features_and_fixed_global_subset_scale() -> None:
    model = _StateEchoPolicy()
    observations = {"features": torch.tensor([[10.0], [20.0]])}
    recorded_state = torch.tensor([[1.0], [3.0]])
    paired_state = torch.tensor([[2.0], [99.0]])
    target = torch.zeros(2, 1, 1)

    prediction, loss = _model_action_loss(
        model,
        observations,
        recorded_state,
        target,
        {},
        state_pairing={"pairing_scale": 2.0, "weight": 1.0},
        paired_state=paired_state,
        pairing_mask=torch.tensor([True, False]),
    )

    recorded = torch.nn.functional.smooth_l1_loss(prediction, target)
    paired = torch.nn.functional.smooth_l1_loss(
        torch.tensor([[[2.0]]]), torch.zeros(1, 1, 1)
    )
    assert torch.equal(loss, (recorded + paired) / 2.0)
    assert len(model.calls) == 2
    assert torch.equal(model.calls[0][0], observations["features"])
    assert torch.equal(model.calls[1][0], observations["features"][:1])
    assert torch.equal(model.calls[1][1], paired_state[:1])


def test_unconfigured_pairing_is_exact_legacy_and_does_not_run_pair_branch() -> None:
    model = _StateEchoPolicy()
    observations = {"features": torch.tensor([[10.0], [20.0]])}
    state = torch.tensor([[1.0], [3.0]])
    target = torch.zeros(2, 1, 1)

    prediction, loss = _model_action_loss(
        model,
        observations,
        state,
        target,
        {},
        state_pairing=None,
        paired_state=None,
        pairing_mask=None,
    )

    assert torch.equal(loss, smooth_l1_action_loss(prediction, target))
    assert len(model.calls) == 1
