"""Backbone-compatible access to immutable precomputed representations."""

from __future__ import annotations

from torch import Tensor

from rosetta_reality.models.backbones.base import BackboneBatch, VLABackbone


class CachedBackbone(VLABackbone):
    """Expose already-pooled features through the generic backbone contract."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        self._hidden_size = hidden_size

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def encode(self, observations: BackboneBatch) -> Tensor:
        features = observations.get("features")
        if not isinstance(features, Tensor):
            raise TypeError("CachedBackbone expects a Tensor named 'features'.")
        if features.ndim != 2 or features.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Cached features must have shape [batch, {self.hidden_size}], "
                f"received {tuple(features.shape)}."
            )
        return features
