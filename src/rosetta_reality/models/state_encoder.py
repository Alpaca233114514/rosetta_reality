"""Robot-state encoders."""

from __future__ import annotations

from torch import Tensor, nn


class StateEncoder(nn.Module):
    """Encode arbitrary fixed-width robot state with a configurable MLP."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        *,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or hidden_dim <= 0:
            raise ValueError("state_dim and hidden_dim must be positive.")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.state_dim = state_dim
        self.output_dim = hidden_dim
        layers: list[nn.Module] = []
        input_dim = state_dim
        for layer_index in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            if layer_index < num_layers - 1:
                layers.extend((nn.GELU(), nn.Dropout(dropout)))
            input_dim = hidden_dim
        layers.append(nn.LayerNorm(hidden_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, robot_state: Tensor) -> Tensor:
        """Map ``[batch, state_dim]`` to ``[batch, hidden_dim]``."""

        if robot_state.ndim != 2 or robot_state.shape[-1] != self.state_dim:
            raise ValueError(
                f"robot_state must have shape [batch, {self.state_dim}], "
                f"but received {tuple(robot_state.shape)}."
            )
        return self.network(robot_state)

