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

