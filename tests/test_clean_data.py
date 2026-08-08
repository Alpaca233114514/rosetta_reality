"""Offline tests for revision-safe dataset cache selection."""

from dataclasses import asdict, replace
from pathlib import Path

import pytest

from rosetta_reality.data.config import load_dataset_config
from rosetta_reality.data.manifest import (
    DatasetManifest,
    revision_cache_root,
    save_dataset_manifest,
)
from scripts.clean_data import _matching_manifest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "data" / "aloha_sim_insertion.yaml"


def _save_manifest(cache_root: Path, requested: str, resolved: str) -> Path:
    config = load_dataset_config(CONFIG_PATH)
    root = revision_cache_root(cache_root, config.repo_id, resolved)
    return save_dataset_manifest(
        root,
        DatasetManifest(
            repo_id=config.repo_id,
            requested_revision=requested,
            resolved_revision=resolved,
            episodes=config.episodes,
            cameras=config.cameras,
            fields=asdict(config.fields),
            license=config.license,
        ),
    )


def test_pinned_revision_selects_only_the_configured_manifest(tmp_path: Path) -> None:
    first_revision = "1" * 40
    configured_revision = "2" * 40
    _save_manifest(tmp_path, "main", first_revision)
    expected_path = _save_manifest(tmp_path, "main", configured_revision)
    config = replace(load_dataset_config(CONFIG_PATH), revision=configured_revision)

    path, manifest = _matching_manifest(config, tmp_path)

    assert path == expected_path
    assert manifest.resolved_revision == configured_revision


def test_mutable_revision_rejects_multiple_matching_manifests(tmp_path: Path) -> None:
    _save_manifest(tmp_path, "main", "1" * 40)
    _save_manifest(tmp_path, "main", "2" * 40)
    config = load_dataset_config(CONFIG_PATH)

    with pytest.raises(ValueError, match="Multiple prepared manifests"):
        _matching_manifest(config, tmp_path)
