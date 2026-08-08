"""Offline tests for dataset preparation cache-root selection."""

from pathlib import Path

import pytest

from rosetta_reality.data.config import (
    CACHE_ROOT_ENVIRONMENT_VARIABLE,
    load_dataset_config,
)
from scripts.prepare_data import _cache_root, _selected_episode_prefixes

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "data" / "aloha_sim_insertion.yaml"


def test_absolute_environment_cache_root_overrides_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = load_dataset_config(CONFIG_PATH)
    monkeypatch.setenv(CACHE_ROOT_ENVIRONMENT_VARIABLE, str(tmp_path))

    assert _cache_root(config) == tmp_path


def test_relative_environment_cache_root_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_dataset_config(CONFIG_PATH)
    monkeypatch.setenv(CACHE_ROOT_ENVIRONMENT_VARIABLE, "relative/data")

    with pytest.raises(ValueError, match="absolute path"):
        _cache_root(config)


class FakeMetadata:
    video_keys = ("observation.images.primary", "observation.images.secondary")

    def get_data_file_path(self, episode: int) -> Path:
        return Path(f"data/chunk-000/file-{episode // 2:03d}.parquet")

    def get_video_file_path(self, episode: int, video_key: str) -> Path:
        return Path(f"videos/{video_key}/chunk-000/file-{episode // 2:03d}.mp4")


def test_episode_prefixes_include_all_metadata_video_keys_and_deduplicate_files() -> None:
    prefixes = _selected_episode_prefixes(FakeMetadata(), (0, 1))

    assert prefixes == (
        "data/chunk-000/file-000.parquet",
        "videos/observation.images.primary/chunk-000/file-000.mp4",
        "videos/observation.images.secondary/chunk-000/file-000.mp4",
    )
