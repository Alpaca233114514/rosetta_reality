"""Continuous action head shape tests."""

import torch

from rosetta_reality.models.action_head import ContinuousActionHead


def test_action_head_shape() -> None:
    """Action head preserves configurable action and chunk dimensions."""

    head = ContinuousActionHead(input_dim=32, action_dim=7, chunk_size=8)
    actions = head(torch.randn(3, 32))

    assert actions.shape == (3, 8, 7)

