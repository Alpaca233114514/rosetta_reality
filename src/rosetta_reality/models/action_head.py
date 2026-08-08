"""Continuous action prediction heads."""

from __future__ import annotations

from torch import Tensor, nn


class ContinuousActionHead(nn.Module):
    """Predict a configurable chunk of continuous actions with an MLP."""

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        chunk_size: int,
        *,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or action_dim <= 0 or chunk_size <= 0:
            raise ValueError("input_dim, action_dim, and chunk_size must be positive.")
        head_hidden_dim = input_dim if hidden_dim is None else hidden_dim
        if head_hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive when provided.")

        self.input_dim = input_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.network = nn.Sequential(
            nn.Linear(input_dim, head_hidden_dim),
            nn.GELU(),
            nn.Linear(head_hidden_dim, chunk_size * action_dim),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        """Map ``[batch, input_dim]`` to ``[batch, chunk_size, action_dim]``."""

        if hidden.ndim != 2 or hidden.shape[-1] != self.input_dim:
            raise ValueError(
                f"hidden must have shape [batch, {self.input_dim}], "
                f"but received {tuple(hidden.shape)}."
            )
        actions = self.network(hidden)
        return actions.reshape(hidden.shape[0], self.chunk_size, self.action_dim)
