"""Dataset adapter boundary used by the generic M1 pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor
from torch.utils.data import Dataset

from rosetta_reality.data.schema import RosettaFrame


@dataclass(frozen=True, slots=True)
class FrameReference:
    """Minimal ordering metadata that can be read without decoding images."""

    episode_id: str
    frame_index: int


class DatasetAdapter(Dataset[RosettaFrame], ABC):
    """Convert a source dataset into Rosetta frames without policy assumptions."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of selected source frames."""

    @abstractmethod
    def __getitem__(self, index: int) -> RosettaFrame:
        """Return one fully materialized frame."""

    def frame_reference(self, index: int) -> FrameReference:
        """Return episode ordering metadata, avoiding image decoding when overridden."""

        frame = self[index]
        return FrameReference(episode_id=frame.episode_id, frame_index=frame.frame_index)

    def state_at(self, index: int) -> Tensor:
        """Return one state vector, avoiding image decoding when overridden."""

        return self[index].robot_state

    def action_at(self, index: int) -> Tensor:
        """Return one action vector, avoiding image decoding when overridden."""

        return self[index].action
