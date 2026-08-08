"""Online population statistics and reversible tensor normalization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from rosetta_reality.data.adapters import DatasetAdapter

MIN_STD = 1e-6


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

    def to_dict(self) -> dict[str, list[float]]:
        """Return a JSON-serializable representation."""

        return {
            "mean": self.mean.detach().cpu().tolist(),
            "std": self.std.detach().cpu().tolist(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> NormalizationStats:
        """Construct statistics from a JSON-compatible mapping."""

        return cls(
            mean=torch.as_tensor(value["mean"], dtype=torch.float32),
            std=torch.as_tensor(value["std"], dtype=torch.float32),
        )


@dataclass(frozen=True, slots=True)
class DatasetStatistics:
    """Separate state and action statistics with their observation counts."""

    state: NormalizationStats
    action: NormalizationStats
    state_count: int
    action_count: int

    def __post_init__(self) -> None:
        if self.state_count <= 0 or self.action_count <= 0:
            raise ValueError("Statistic counts must be positive.")

    def to_dict(self) -> dict[str, Any]:
        """Return a versioned JSON-serializable representation."""

        return {
            "version": 1,
            "population_std": True,
            "minimum_std": MIN_STD,
            "state_count": self.state_count,
            "action_count": self.action_count,
            "state": self.state.to_dict(),
            "action": self.action.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DatasetStatistics:
        """Restore a version-one statistics payload."""

        if value.get("version") != 1:
            raise ValueError(f"Unsupported statistics version: {value.get('version')!r}.")
        return cls(
            state=NormalizationStats.from_dict(value["state"]),
            action=NormalizationStats.from_dict(value["action"]),
            state_count=int(value["state_count"]),
            action_count=int(value["action_count"]),
        )


class RunningMoments:
    """Numerically stable online population moments for feature vectors."""

    def __init__(self) -> None:
        self.count = 0
        self.mean: Tensor | None = None
        self.m2: Tensor | None = None

    def update(self, values: Tensor) -> None:
        """Merge one vector or a batch of vectors into the running moments."""

        if values.ndim == 0:
            raise ValueError("values must include a feature dimension.")
        matrix = values.detach().to(dtype=torch.float64, device="cpu")
        matrix = matrix.reshape(-1, matrix.shape[-1])
        batch_count = matrix.shape[0]
        batch_mean = matrix.mean(dim=0)
        batch_m2 = ((matrix - batch_mean) ** 2).sum(dim=0)
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        assert self.mean is not None
        assert self.m2 is not None
        if batch_mean.shape != self.mean.shape:
            raise ValueError(
                "Feature dimension changed while computing statistics: "
                f"{tuple(self.mean.shape)} versus {tuple(batch_mean.shape)}."
            )
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (batch_count / total)
        self.m2 = self.m2 + batch_m2 + delta.square() * self.count * batch_count / total
        self.count = total

    def finalize(self, minimum_std: float = MIN_STD) -> NormalizationStats:
        """Return float32 mean and population standard deviation."""

        if self.count == 0 or self.mean is None or self.m2 is None:
            raise ValueError("Cannot finalize empty running moments.")
        if minimum_std <= 0:
            raise ValueError("minimum_std must be positive.")
        variance = self.m2 / self.count
        return NormalizationStats(
            mean=self.mean.to(torch.float32),
            std=variance.sqrt().clamp_min(minimum_std).to(torch.float32),
        )


def normalize(values: Tensor, stats: NormalizationStats) -> Tensor:
    """Apply feature-wise standardization."""

    return (values - stats.mean) / stats.std


def denormalize(values: Tensor, stats: NormalizationStats) -> Tensor:
    """Invert feature-wise standardization."""

    return values * stats.std + stats.mean


def compute_dataset_statistics(adapter: DatasetAdapter) -> DatasetStatistics:
    """Compute state/action moments once per selected source frame."""

    state_moments = RunningMoments()
    action_moments = RunningMoments()
    for index in range(len(adapter)):
        state_moments.update(adapter.state_at(index))
        action_moments.update(adapter.action_at(index))
    return DatasetStatistics(
        state=state_moments.finalize(),
        action=action_moments.finalize(),
        state_count=state_moments.count,
        action_count=action_moments.count,
    )


def save_dataset_statistics(path: Path, statistics: DatasetStatistics) -> None:
    """Persist statistics once; validate and reuse an identical existing file."""

    payload = statistics.to_dict()
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise FileExistsError(f"Refusing to overwrite different statistics at {path}.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def load_dataset_statistics(path: Path) -> DatasetStatistics:
    """Load a persisted statistics payload without modifying it."""

    return DatasetStatistics.from_dict(json.loads(path.read_text(encoding="utf-8")))
