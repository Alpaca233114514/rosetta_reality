"""Offline field-mapping tests for the LeRobot v3 adapter."""

import pytest
import torch

from rosetta_reality.data.adapters import LeRobotV3Adapter
from rosetta_reality.data.config import FieldMapping

REVISION = "a" * 40
FIELDS = FieldMapping(
    state="observation.state",
    action="action",
    timestamp="timestamp",
    instruction="task",
    episode_index="episode_index",
    frame_index="frame_index",
)


class FakeLeRobotDataset:
    def __init__(self, *, include_action_feature: bool = True) -> None:
        self.features = {
            "observation.state": {},
            "timestamp": {},
            "episode_index": {},
            "frame_index": {},
            "observation.images.top": {},
        }
        if include_action_feature:
            self.features["action"] = {}
        self.record = {
            "observation.state": torch.arange(14),
            "action": torch.arange(14) + 1,
            "timestamp": torch.tensor(0.0),
            "episode_index": torch.tensor(0),
            "frame_index": torch.tensor(0),
            "index": 0,
        }

    def __len__(self) -> int:
        return 1

    def get_raw_item(self, index: int) -> dict:
        assert index == 0
        return self.record

    def __getitem__(self, index: int) -> dict:
        return {
            **self.get_raw_item(index),
            "task": "Insert the peg into the socket.",
            "observation.images.top": torch.zeros(3, 480, 640),
        }


def _adapter(dataset: FakeLeRobotDataset) -> LeRobotV3Adapter:
    return LeRobotV3Adapter(
        repo_id="lerobot/example",
        revision=REVISION,
        root=None,
        episodes=(0,),
        cameras={"top": "observation.images.top"},
        fields=FIELDS,
        embodiment="aloha",
        license_name="MIT",
        dataset=dataset,
    )


def test_lerobot_adapter_maps_required_fields() -> None:
    frame = _adapter(FakeLeRobotDataset())[0]

    assert frame.instruction == "Insert the peg into the socket."
    assert frame.robot_state.shape == (14,)
    assert frame.action.shape == (14,)
    assert frame.images["top"].shape == (3, 480, 640)
    assert frame.images["top"].dtype == torch.float32
    assert frame.episode_id == "0"
    assert frame.frame_index == 0
    assert frame.metadata["source_revision"] == REVISION


def test_lerobot_adapter_reports_missing_dataset_feature() -> None:
    with pytest.raises(KeyError, match="action"):
        _adapter(FakeLeRobotDataset(include_action_feature=False))


def test_lerobot_adapter_reports_missing_record_field() -> None:
    dataset = FakeLeRobotDataset()
    del dataset.record["action"]
    adapter = _adapter(dataset)

    with pytest.raises(KeyError, match="action"):
        adapter.action_at(0)
