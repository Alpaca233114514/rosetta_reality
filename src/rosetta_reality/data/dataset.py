"""Dataset-independent temporal chunking and batching."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from rosetta_reality.data.adapters import DatasetAdapter, FrameReference
from rosetta_reality.data.schema import RosettaBatch, RosettaSample


class ActionChunkDataset(Dataset[RosettaSample]):
    """Pair each observation with consecutive actions from the same episode."""

    def __init__(self, adapter: DatasetAdapter, chunk_size: int = 8) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        self.adapter = adapter
        self.chunk_size = chunk_size
        references = [adapter.frame_reference(index) for index in range(len(adapter))]
        self._references = tuple(references)
        self._anchors = tuple(self._find_anchors(references, chunk_size))

    @staticmethod
    def _find_anchors(references: list[FrameReference], chunk_size: int) -> list[int]:
        anchors: list[int] = []
        for anchor in range(max(0, len(references) - chunk_size + 1)):
            window = references[anchor : anchor + chunk_size]
            first = window[0]
            if all(
                reference.episode_id == first.episode_id
                and reference.frame_index == first.frame_index + offset
                for offset, reference in enumerate(window)
            ):
                anchors.append(anchor)
        return anchors

    def __len__(self) -> int:
        return len(self._anchors)

    @property
    def anchor_indices(self) -> tuple[int, ...]:
        """Return immutable source indices for valid action-chunk anchors."""

        return self._anchors

    def anchor_reference(self, index: int) -> FrameReference:
        """Return the episode/frame identity without decoding images."""

        return self._references[self._anchors[index]]

    def __getitem__(self, index: int) -> RosettaSample:
        anchor = self._anchors[index]
        frame = self.adapter[anchor]
        actions = torch.stack(
            [
                self.adapter.action_at(frame_index)
                for frame_index in range(anchor, anchor + self.chunk_size)
            ]
        )
        return RosettaSample(
            instruction=frame.instruction,
            robot_state=frame.robot_state,
            actions=actions,
            images=frame.images,
            episode_id=frame.episode_id,
            frame_index=frame.frame_index,
            timestamp=frame.timestamp,
            embodiment=frame.embodiment,
            metadata=frame.metadata,
        )


def collate_rosetta(samples: list[RosettaSample]) -> RosettaBatch:
    """Stack homogeneous Rosetta samples while retaining text and metadata."""

    if not samples:
        raise ValueError("Cannot collate an empty sample list.")
    camera_names = tuple(samples[0].images)
    if any(tuple(sample.images) != camera_names for sample in samples[1:]):
        raise ValueError("All samples in a batch must contain the same ordered camera names.")
    return RosettaBatch(
        instructions=tuple(sample.instruction for sample in samples),
        robot_state=torch.stack([sample.robot_state for sample in samples]),
        actions=torch.stack([sample.actions for sample in samples]),
        images={
            camera_name: torch.stack([sample.images[camera_name] for sample in samples])
            for camera_name in camera_names
        },
        episode_ids=tuple(sample.episode_id for sample in samples),
        frame_indices=torch.tensor([sample.frame_index for sample in samples], dtype=torch.long),
        timestamps=torch.tensor([sample.timestamp for sample in samples], dtype=torch.float32),
        embodiments=tuple(sample.embodiment for sample in samples),
        metadata=tuple(sample.metadata for sample in samples),
    )
