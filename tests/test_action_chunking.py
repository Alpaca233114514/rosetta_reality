"""Dataset-independent action chunking tests."""

import torch

from rosetta_reality.data import ActionChunkDataset
from rosetta_reality.data.adapters import DatasetAdapter
from rosetta_reality.data.schema import RosettaFrame


class MemoryAdapter(DatasetAdapter):
    """Small adapter whose frame values expose ordering mistakes."""

    def __init__(self, episode_lengths: tuple[int, ...], feature_dim: int = 2) -> None:
        self.frames: list[RosettaFrame] = []
        action_value = 0
        for episode_id, episode_length in enumerate(episode_lengths):
            for frame_index in range(episode_length):
                self.frames.append(
                    RosettaFrame(
                        instruction="test",
                        robot_state=torch.full((feature_dim,), float(frame_index)),
                        action=torch.full((feature_dim,), float(action_value)),
                        episode_id=str(episode_id),
                        frame_index=frame_index,
                        timestamp=float(frame_index),
                        embodiment="test",
                    )
                )
                action_value += 1

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> RosettaFrame:
        return self.frames[index]


def test_action_chunks_do_not_cross_episodes_and_drop_tails() -> None:
    adapter = MemoryAdapter((5, 4))
    dataset = ActionChunkDataset(adapter, chunk_size=3)

    assert len(dataset) == 5
    assert dataset.anchor_indices == (0, 1, 2, 5, 6)
    assert dataset.anchor_reference(3).episode_id == "1"
    assert dataset.anchor_reference(3).frame_index == 0
    assert [dataset[index].episode_id for index in range(len(dataset))] == [
        "0",
        "0",
        "0",
        "1",
        "1",
    ]
    assert dataset[2].frame_index == 2
    assert dataset[2].actions[:, 0].tolist() == [2.0, 3.0, 4.0]
    assert dataset[3].frame_index == 0
    assert dataset[3].actions[:, 0].tolist() == [5.0, 6.0, 7.0]


def test_action_chunk_size_must_be_positive() -> None:
    adapter = MemoryAdapter((2,))

    try:
        ActionChunkDataset(adapter, chunk_size=0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("Expected an invalid chunk size to fail.")
