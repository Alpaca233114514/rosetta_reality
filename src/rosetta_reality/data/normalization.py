"""Small tensor normalization primitives for future dataset adapters."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True, slots=True)
class NormalizationStats:
    """Per-feature mean and standard deviation."""

    mean: Tensor
    std: Tensor

    def __post_init__(self) -> None:
        if self.mean.shape != self.std.shape:
            raise ValueError("mean and std must have identical shapes.")
        if (self.std <= 0).any().item():
            raise ValueError("std entries must be positive.")


def normalize(values: Tensor, stats: NormalizationStats) -> Tensor:
    """Apply feature-wise standardization."""

    return (values - stats.mean) / stats.std


def denormalize(values: Tensor, stats: NormalizationStats) -> Tensor:
    """Invert feature-wise standardization."""

    return values * stats.std + stats.mean

