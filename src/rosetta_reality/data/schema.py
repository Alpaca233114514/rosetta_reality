"""Robot-agnostic frame, sample, and batch contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torch import Tensor


def _validate_vector(name: str, value: Tensor) -> None:
    if value.ndim != 1:
        raise ValueError(f"{name} must have shape [feature_dim], received {tuple(value.shape)}.")


def _validate_images(images: dict[str, Tensor]) -> None:
    for camera_name, image in images.items():
        if not camera_name:
            raise ValueError("Camera names must be non-empty.")
        if image.ndim != 3:
            raise ValueError(
                f"Image '{camera_name}' must have shape [channels, height, width], "
                f"received {tuple(image.shape)}."
            )
        if not image.is_floating_point():
            raise TypeError(f"Image '{camera_name}' must be a floating-point tensor.")


@dataclass(frozen=True, slots=True)
class RosettaFrame:
    """One source frame before temporal action chunking."""

    instruction: str
    robot_state: Tensor
    action: Tensor
    episode_id: str
    frame_index: int
    timestamp: float
    embodiment: str
    images: dict[str, Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_vector("robot_state", self.robot_state)
        _validate_vector("action", self.action)
        _validate_images(self.images)
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative.")


@dataclass(frozen=True, slots=True)
class RosettaSample:
    """One current observation paired with a future action chunk."""

    instruction: str
    robot_state: Tensor
    actions: Tensor
    episode_id: str
    frame_index: int
    timestamp: float
    embodiment: str
    images: dict[str, Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_vector("robot_state", self.robot_state)
        if self.actions.ndim != 2:
            raise ValueError(
                "actions must have shape [chunk_size, action_dim], "
                f"received {tuple(self.actions.shape)}."
            )
        _validate_images(self.images)
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative.")


@dataclass(frozen=True, slots=True)
class RosettaBatch:
    """DataLoader output with tensors stacked along the batch dimension."""

    instructions: tuple[str, ...]
    robot_state: Tensor
    actions: Tensor
    images: dict[str, Tensor]
    episode_ids: tuple[str, ...]
    frame_indices: Tensor
    timestamps: Tensor
    embodiments: tuple[str, ...]
    metadata: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        batch_size = self.robot_state.shape[0]
        if self.robot_state.ndim != 2:
            raise ValueError("robot_state must have shape [batch, state_dim].")
        if self.actions.ndim != 3 or self.actions.shape[0] != batch_size:
            raise ValueError("actions must have shape [batch, chunk_size, action_dim].")
        if any(image.ndim != 4 or image.shape[0] != batch_size for image in self.images.values()):
            raise ValueError("Batched images must have shape [batch, channels, height, width].")
        lengths = (
            len(self.instructions),
            len(self.episode_ids),
            len(self.embodiments),
            len(self.metadata),
            self.frame_indices.shape[0],
            self.timestamps.shape[0],
        )
        if any(length != batch_size for length in lengths):
            raise ValueError("All batch fields must share the same batch dimension.")
