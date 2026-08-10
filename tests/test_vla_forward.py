"""End-to-end dummy VLA policy tests."""

import torch

from rosetta_reality.models import ContinuousActionHead, StateEncoder, VLAPolicy
from rosetta_reality.models.backbones import DummyBackbone
from rosetta_reality.train import smooth_l1_action_loss


def test_vla_forward_and_backward() -> None:
    """DummyBackbone supports an offline action-chunk gradient path."""

    batch_size = 2
    policy = VLAPolicy(
        backbone=DummyBackbone(input_dim=16, hidden_size=32),
        state_encoder=StateEncoder(state_dim=9, hidden_dim=32),
        action_head=ContinuousActionHead(input_dim=32, action_dim=7, chunk_size=8),
    )
    predicted = policy(
        {"features": torch.randn(batch_size, 16)},
        torch.randn(batch_size, 9),
    )
    target = torch.randn(batch_size, 8, 7)
    loss = smooth_l1_action_loss(predicted, target)
    loss.backward()

    assert predicted.shape == (batch_size, 8, 7)
    assert any(parameter.grad is not None for parameter in policy.parameters())


def test_vla_residual_parameterization_adds_current_state_in_action_units() -> None:
    policy = VLAPolicy(
        backbone=DummyBackbone(input_dim=3, hidden_size=4),
        state_encoder=StateEncoder(state_dim=2, hidden_dim=4),
        action_head=ContinuousActionHead(input_dim=4, action_dim=2, chunk_size=3),
        state_to_action_scale=torch.tensor([2.0, 3.0]),
        state_to_action_offset=torch.tensor([0.5, -0.5]),
    )
    for parameter in policy.parameters():
        parameter.detach().zero_()
    normalized_state = torch.tensor([[1.0, 2.0]])

    predicted = policy({"features": torch.zeros(1, 3)}, normalized_state)

    expected = torch.tensor([2.5, 5.5]).view(1, 1, 2).expand(1, 3, 2)
    assert torch.equal(predicted, expected)
