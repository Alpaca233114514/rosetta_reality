"""Small offline backbone used by tests and smoke training."""

from __future__ import annotations

from torch import Tensor, nn

from rosetta_reality.models.backbones.base import BackboneBatch, VLABackbone


class DummyBackbone(VLABackbone):
    """Project synthetic input features without models, downloads, or a GPU."""

    def __init__(self, input_dim: int, hidden_size: int) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_size <= 0:
            raise ValueError("input_dim and hidden_size must be positive.")
        self.input_dim = input_dim
        self._hidden_size = hidden_size
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )

    @property
    def hidden_size(self) -> int:
        """Width of the generated synthetic representation."""

        return self._hidden_size

    def encode(self, observations: BackboneBatch) -> Tensor:
        """Encode the batch's ``features`` tensor.

        ``features`` must have shape ``[batch, input_dim]``. Requiring explicit
        features keeps tests deterministic and prevents accidental assumptions
        about a real backbone's image preprocessing.
        """

        features = observations.get("features")
        if not isinstance(features, Tensor):
            raise TypeError("DummyBackbone observations must contain a Tensor named 'features'.")
        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise ValueError(
                "DummyBackbone features must have shape "
                f"[batch, {self.input_dim}], but received {tuple(features.shape)}."
            )
        return self.projection(features)

