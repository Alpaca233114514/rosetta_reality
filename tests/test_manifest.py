"""Immutable dataset revision and cache manifest tests."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from rosetta_reality.data.manifest import (
    DatasetManifest,
    compute_cache_checksums,
    load_dataset_manifest,
    resolve_hub_revision,
    revision_cache_root,
    save_cache_checksums,
    save_dataset_manifest,
    validate_cache_checksums,
)

REVISION = "b" * 40


class FakeHubApi:
    def dataset_info(self, *, repo_id: str, revision: str) -> SimpleNamespace:
        assert repo_id == "lerobot/example"
        assert revision == "main"
        return SimpleNamespace(sha=REVISION)


def test_hub_revision_resolves_to_commit_and_cache_is_revision_scoped(tmp_path) -> None:
    resolved = resolve_hub_revision("lerobot/example", "main", api=FakeHubApi())

    assert resolved == REVISION
    assert revision_cache_root(tmp_path, "lerobot/example", resolved) == (
        tmp_path / "lerobot--example" / REVISION
    )


def test_manifest_round_trip_reuses_identical_file(tmp_path) -> None:
    manifest = DatasetManifest(
        repo_id="lerobot/example",
        requested_revision="main",
        resolved_revision=REVISION,
        episodes=(0,),
        cameras={"top": "observation.images.top"},
        license="MIT",
    )
    root = Path(tmp_path) / REVISION

    path = save_dataset_manifest(root, manifest)
    assert save_dataset_manifest(root, manifest) == path
    assert load_dataset_manifest(path) == manifest


def test_manifest_requires_immutable_revision() -> None:
    with pytest.raises(ValueError, match="commit SHA"):
        DatasetManifest(
            repo_id="lerobot/example",
            requested_revision="main",
            resolved_revision="main",
            episodes=(0,),
            cameras={"top": "observation.images.top"},
            license="MIT",
        )


def test_cache_checksums_detect_changed_content(tmp_path) -> None:
    root = Path(tmp_path)
    metadata = root / "meta" / "info.json"
    metadata.parent.mkdir()
    metadata.write_text('{"version": "v3.0"}', encoding="utf-8")
    checksums = compute_cache_checksums(root)
    save_cache_checksums(root, checksums)

    assert validate_cache_checksums(root) == 1
    metadata.write_text('{"version": "changed"}', encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_cache_checksums(root)
