"""Third-party dataset adapters behind the generic Rosetta contract."""

from rosetta_reality.data.adapters.base import DatasetAdapter, FrameReference
from rosetta_reality.data.adapters.lerobot_v3 import LeRobotV3Adapter

__all__ = ["DatasetAdapter", "FrameReference", "LeRobotV3Adapter"]
