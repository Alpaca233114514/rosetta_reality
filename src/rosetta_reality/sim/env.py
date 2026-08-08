"""Simulator-neutral environment contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from torch import Tensor


class SimulationEnvironment(ABC):
    """Minimal interface future MuJoCo or other adapters must implement."""

    @abstractmethod
    def reset(self, *, seed: int | None = None) -> Mapping[str, Any]:
        """Reset the environment and return the first observation."""

    @abstractmethod
    def step(self, action: Tensor) -> tuple[Mapping[str, Any], float, bool, dict[str, Any]]:
        """Advance the simulator by one action and return transition data."""

    @abstractmethod
    def close(self) -> None:
        """Release simulator resources."""

