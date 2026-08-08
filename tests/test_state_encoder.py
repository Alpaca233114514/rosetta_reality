"""State encoder shape tests."""

import torch

from rosetta_reality.models.state_encoder import StateEncoder


def test_state_encoder_shape() -> None:
    """StateEncoder maps configurable state width to hidden width."""

    encoder = StateEncoder(state_dim=11, hidden_dim=24, num_layers=3, dropout=0.1)
    encoded = encoder(torch.randn(4, 11))

    assert encoded.shape == (4, 24)

