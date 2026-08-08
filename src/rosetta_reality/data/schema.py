"""Internal observation and action sample schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torch import Tensor


@dataclass(slots=True)
class RosettaSample:
    """A robot-agnostic sample used at Rosetta Reality boundaries.

    Dataset-specific adapters should convert their records into this schema.
    M0 deliberately does not implement large dataset loaders.
    """

    instruction: str
    robot_state: Tensor
    actions: Tensor
    episode_id: str
    timestamp: float
    embodiment: str
    images: Tensor | list[Tensor] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

