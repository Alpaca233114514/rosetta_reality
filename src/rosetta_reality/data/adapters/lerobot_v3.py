"""LeRobot v3 to Rosetta frame adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from rosetta_reality.data.adapters.base import DatasetAdapter, FrameReference
from rosetta_reality.data.config import FieldMapping
from rosetta_reality.data.manifest import require_commit_sha
from rosetta_reality.data.schema import RosettaFrame


def _required(record: dict[str, Any], key: str) -> Any:
    try:
        return record[key]
    except KeyError as error:
        raise KeyError(f"LeRobot record is missing required field '{key}'.") from error


def _scalar(value: Any, key: str) -> int | float:
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError(f"LeRobot field '{key}' must be scalar.")
        return value.item()
    if isinstance(value, int | float):
        return value
    tensor = torch.as_tensor(value)
    if tensor.numel() != 1:
        raise ValueError(f"LeRobot field '{key}' must be scalar.")
    return tensor.item()


def _vector(value: Any, key: str) -> Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim != 1:
        raise ValueError(
            f"LeRobot field '{key}' must have shape [feature_dim], received {tuple(tensor.shape)}."
        )
    return tensor


def _image(value: Any, key: str) -> Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim != 3:
        raise ValueError(
            f"LeRobot image '{key}' must be rank three, received {tuple(tensor.shape)}."
        )
    if tensor.shape[0] not in (1, 3, 4) and tensor.shape[-1] in (1, 3, 4):
        tensor = tensor.permute(2, 0, 1)
    if tensor.shape[0] not in (1, 3, 4):
        raise ValueError(
            f"LeRobot image '{key}' has no recognizable channel dimension: {tuple(tensor.shape)}."
        )
    if tensor.dtype == torch.uint8:
        return tensor.to(torch.float32).div(255)
    return tensor.to(torch.float32)


class LeRobotV3Adapter(DatasetAdapter):
    """Map one immutable LeRobot v3 episode selection into Rosetta frames."""

    def __init__(
        self,
        *,
        repo_id: str,
        revision: str,
        root: Path,
        episodes: tuple[int, ...],
        cameras: dict[str, str],
        fields: FieldMapping,
        embodiment: str,
        license_name: str,
        video_backend: str = "pyav",
        dataset: Any | None = None,
    ) -> None:
        self.repo_id = repo_id
        self.revision = require_commit_sha(revision)
        self.root = root
        self.episodes = episodes
        self.cameras = cameras
        self.fields = fields
        self.embodiment = embodiment
        self.license_name = license_name
        if dataset is None:
            from lerobot.datasets import LeRobotDataset

            dataset = LeRobotDataset(
                repo_id=repo_id,
                root=root,
                episodes=list(episodes),
                revision=self.revision,
                download_videos=True,
                video_backend=video_backend,
                return_uint8=False,
            )
        self._dataset = dataset
        self._validate_features()

    def _validate_features(self) -> None:
        features = getattr(self._dataset, "features", {})
        required = {
            self.fields.state,
            self.fields.action,
            self.fields.timestamp,
            self.fields.episode_index,
            self.fields.frame_index,
            *self.cameras.values(),
        }
        missing = sorted(key for key in required if key not in features)
        if missing:
            raise KeyError(f"LeRobot dataset is missing required features: {missing}.")

    def __len__(self) -> int:
        return len(self._dataset)

    def _raw_item(self, index: int) -> dict[str, Any]:
        getter = getattr(self._dataset, "get_raw_item", None)
        if getter is not None:
            return getter(index)
        return self._dataset.hf_dataset[index]

    def frame_reference(self, index: int) -> FrameReference:
        record = self._raw_item(index)
        episode_index = _scalar(
            _required(record, self.fields.episode_index),
            self.fields.episode_index,
        )
        frame_index = _scalar(
            _required(record, self.fields.frame_index),
            self.fields.frame_index,
        )
        return FrameReference(
            episode_id=str(int(episode_index)),
            frame_index=int(frame_index),
        )

    def state_at(self, index: int) -> Tensor:
        record = self._raw_item(index)
        return _vector(_required(record, self.fields.state), self.fields.state)

    def action_at(self, index: int) -> Tensor:
        record = self._raw_item(index)
        return _vector(_required(record, self.fields.action), self.fields.action)

    @property
    def state_dim(self) -> int:
        return self.state_at(0).shape[0]

    @property
    def action_dim(self) -> int:
        return self.action_at(0).shape[0]

    def __getitem__(self, index: int) -> RosettaFrame:
        record = self._dataset[index]
        episode_id = str(
            int(_scalar(_required(record, self.fields.episode_index), self.fields.episode_index))
        )
        frame_index = int(
            _scalar(_required(record, self.fields.frame_index), self.fields.frame_index)
        )
        instruction = str(_required(record, self.fields.instruction))
        return RosettaFrame(
            instruction=instruction,
            robot_state=_vector(_required(record, self.fields.state), self.fields.state),
            action=_vector(_required(record, self.fields.action), self.fields.action),
            images={
                camera_name: _image(_required(record, source_key), source_key)
                for camera_name, source_key in self.cameras.items()
            },
            episode_id=episode_id,
            frame_index=frame_index,
            timestamp=float(
                _scalar(_required(record, self.fields.timestamp), self.fields.timestamp)
            ),
            embodiment=self.embodiment,
            metadata={
                "source_repo": self.repo_id,
                "source_revision": self.revision,
                "source_index": record.get("index"),
                "license": self.license_name,
            },
        )
