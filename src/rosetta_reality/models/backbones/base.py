"""Generic interface for vision-language backbones."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn

BackboneBatch = Mapping[str, Any]


class VLABackbone(nn.Module, ABC):
    """Convert a multimodal observation batch into one vector per sample.

    The M0 contract intentionally exposes a pooled tensor with shape
    ``[batch, hidden_size]``. Model-family-specific tokenization, image
    processing, and loading remain inside concrete adapters.
    """

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """Width of the representation returned by :meth:`encode`."""

    @abstractmethod
    def encode(self, observations: BackboneBatch) -> Tensor:
        """Encode observations into a ``[batch, hidden_size]`` tensor."""

    def forward(self, observations: BackboneBatch) -> Tensor:
        """Delegate the standard module call to :meth:`encode`."""

        encoded = self.encode(observations)
        if encoded.ndim != 2:
            raise ValueError(
                "A VLA backbone must return [batch, hidden_size], "
                f"but received shape {tuple(encoded.shape)}."
            )
        if encoded.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Backbone declared hidden_size={self.hidden_size}, "
                f"but returned width {encoded.shape[-1]}."
            )
        return encoded

