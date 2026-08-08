"""Typed loading for dataset preparation configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class FieldMapping:
    """Source field names consumed by an adapter."""

    state: str
    action: str
    timestamp: str
    instruction: str
    episode_index: str
    frame_index: str


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Configuration for one bounded dataset preparation target."""

    name: str
    source: str
    repo_id: str
    revision: str
    episodes: tuple[int, ...]
    cameras: dict[str, str]
    fields: FieldMapping
    embodiment: str
    chunk_size: int
    cache_root: Path
    license: str
    expected_frames: int | None = None
    expected_state_dim: int | None = None
    expected_action_dim: int | None = None
    expected_instruction: str | None = None

    def __post_init__(self) -> None:
        if not self.episodes:
            raise ValueError("At least one episode must be configured.")
        if not self.cameras:
            raise ValueError("At least one named camera must be configured.")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")


def _required(mapping: dict[str, Any], key: str) -> Any:
    try:
        return mapping[key]
    except KeyError as error:
        raise ValueError(f"Dataset configuration is missing '{key}'.") from error


def load_dataset_config(path: Path) -> DatasetConfig:
    """Read and validate a YAML dataset configuration without network access."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Dataset configuration must contain a YAML mapping.")
    raw_fields = _required(raw, "fields")
    if not isinstance(raw_fields, dict):
        raise ValueError("'fields' must be a mapping.")
    raw_cameras = _required(raw, "cameras")
    if not isinstance(raw_cameras, dict):
        raise ValueError("'cameras' must be a mapping of camera name to source key.")
    expected = raw.get("expected", {})
    if not isinstance(expected, dict):
        raise ValueError("'expected' must be a mapping when provided.")
    return DatasetConfig(
        name=str(_required(raw, "name")),
        source=str(_required(raw, "source")),
        repo_id=str(_required(raw, "repo_id")),
        revision=str(_required(raw, "revision")),
        episodes=tuple(int(episode) for episode in _required(raw, "episodes")),
        cameras={str(name): str(key) for name, key in raw_cameras.items()},
        fields=FieldMapping(
            state=str(_required(raw_fields, "state")),
            action=str(_required(raw_fields, "action")),
            timestamp=str(_required(raw_fields, "timestamp")),
            instruction=str(_required(raw_fields, "instruction")),
            episode_index=str(_required(raw_fields, "episode_index")),
            frame_index=str(_required(raw_fields, "frame_index")),
        ),
        embodiment=str(_required(raw, "embodiment")),
        chunk_size=int(_required(raw, "chunk_size")),
        cache_root=Path(_required(raw, "cache_root")),
        license=str(_required(raw, "license")),
        expected_frames=expected.get("frames"),
        expected_state_dim=expected.get("state_dim"),
        expected_action_dim=expected.get("action_dim"),
        expected_instruction=expected.get("instruction"),
    )
