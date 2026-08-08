"""Rosetta DataLoader collation contract tests."""

import torch

from rosetta_reality.data import RosettaSample, collate_rosetta


def _sample(frame_index: int) -> RosettaSample:
    return RosettaSample(
        instruction="Insert the peg into the socket.",
        robot_state=torch.zeros(14),
        actions=torch.zeros(8, 14),
        images={"top": torch.zeros(3, 480, 640)},
        episode_id="0",
        frame_index=frame_index,
        timestamp=frame_index / 50,
        embodiment="aloha",
    )


def test_collate_shapes_match_vla_contract() -> None:
    batch = collate_rosetta([_sample(0), _sample(1)])

    assert batch.robot_state.shape == (2, 14)
    assert batch.actions.shape == (2, 8, 14)
    assert batch.images["top"].shape == (2, 3, 480, 640)
    assert batch.frame_indices.tolist() == [0, 1]
    assert batch.instructions == (
        "Insert the peg into the socket.",
        "Insert the peg into the socket.",
    )
