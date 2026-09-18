from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from rosetta_reality.experiment import file_sha256, stable_hash
from rosetta_reality.features import CachedFeatureDataset, create_json, save_tensor_shard


def _payload(identity: str, *, episode: int = 1) -> dict[str, object]:
    return {
        "identity_hash": identity,
        "split": "train",
        "episode": episode,
        "features": torch.randn(2, 4),
        "robot_state": torch.randn(2, 3),
        "actions": torch.randn(2, 2, 3),
        "episode_ids": torch.tensor([episode, episode]),
        "frame_indices": torch.tensor([0, 2]),
    }


def test_create_only_feature_cache_round_trip(tmp_path: Path) -> None:
    identity = {"model": "test"}
    identity_hash = stable_hash(identity)
    shard_path = tmp_path / "shards" / "train" / "episode-001.pt"
    save_tensor_shard(shard_path, _payload(identity_hash))
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "identity": identity,
        "identity_hash": identity_hash,
        "shards": {
            "train": [
                {
                    "episode": 1,
                    "path": shard_path.relative_to(tmp_path).as_posix(),
                    "samples": 2,
                    "sha256": file_sha256(shard_path),
                }
            ]
        },
        "samples": {"train": 2},
    }
    manifest_path = create_json(tmp_path / "manifest.json", manifest)

    dataset = CachedFeatureDataset(manifest_path, "train")

    assert len(dataset) == 2
    assert dataset[0]["features"].shape == (4,)
    assert dataset[0]["actions"].shape == (2, 3)


def test_feature_shards_are_never_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "episode.pt"
    save_tensor_shard(path, _payload("one"))

    with pytest.raises(FileExistsError, match="overwrite"):
        save_tensor_shard(path, _payload("two"))


def test_json_identity_is_immutable(tmp_path: Path) -> None:
    path = tmp_path / "identity.json"
    create_json(path, {"identity": "one"})
    create_json(path, {"identity": "one"})

    with pytest.raises(FileExistsError, match="overwrite"):
        create_json(path, {"identity": "two"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"identity": "one"}


def test_feature_cache_rejects_duplicate_episode_shards(tmp_path: Path) -> None:
    identity = {"model": "test", "split": {"train": [1]}}
    identity_hash = stable_hash(identity)
    records = []
    for name in ("first.pt", "second.pt"):
        shard_path = tmp_path / name
        save_tensor_shard(shard_path, _payload(identity_hash))
        records.append(
            {
                "episode": 1,
                "path": name,
                "samples": 2,
                "sha256": file_sha256(shard_path),
            }
        )
    manifest_path = create_json(
        tmp_path / "manifest.json",
        {
            "schema_version": 1,
            "status": "complete",
            "identity": identity,
            "identity_hash": identity_hash,
            "shards": {"train": records},
            "samples": {"train": 4},
        },
    )

    with pytest.raises(ValueError, match="repeats episode 1"):
        CachedFeatureDataset(manifest_path, "train")


def test_feature_cache_rejects_tampered_manifest_identity(tmp_path: Path) -> None:
    identity = {"model": "test", "split": {"train": [1]}}
    manifest_path = create_json(
        tmp_path / "manifest.json",
        {
            "schema_version": 1,
            "status": "complete",
            "identity": identity,
            "identity_hash": stable_hash({"model": "different"}),
            "shards": {"train": []},
            "samples": {"train": 0},
        },
    )

    with pytest.raises(ValueError, match="identity hash mismatch"):
        CachedFeatureDataset(manifest_path, "train")


def test_feature_cache_rejects_reused_shard_path(tmp_path: Path) -> None:
    identity = {"model": "test", "split": {"train": [1, 2]}}
    identity_hash = stable_hash(identity)
    shard_path = tmp_path / "shared.pt"
    save_tensor_shard(shard_path, _payload(identity_hash, episode=1))
    record = {
        "path": shard_path.name,
        "samples": 2,
        "sha256": file_sha256(shard_path),
    }
    manifest_path = create_json(
        tmp_path / "manifest.json",
        {
            "schema_version": 1,
            "status": "complete",
            "identity": identity,
            "identity_hash": identity_hash,
            "shards": {
                "train": [
                    {**record, "episode": 1},
                    {**record, "episode": 2},
                ]
            },
            "samples": {"train": 4},
        },
    )

    with pytest.raises(ValueError, match="repeats shard path shared.pt"):
        CachedFeatureDataset(manifest_path, "train")


def test_feature_cache_requires_declared_episode_order(tmp_path: Path) -> None:
    identity = {"model": "test", "split": {"train": [2, 1]}}
    identity_hash = stable_hash(identity)
    records = []
    for episode in (1, 2):
        shard_path = tmp_path / f"episode-{episode:03d}.pt"
        save_tensor_shard(shard_path, _payload(identity_hash, episode=episode))
        records.append(
            {
                "episode": episode,
                "path": shard_path.name,
                "samples": 2,
                "sha256": file_sha256(shard_path),
            }
        )
    manifest_path = create_json(
        tmp_path / "manifest.json",
        {
            "schema_version": 1,
            "status": "complete",
            "identity": identity,
            "identity_hash": identity_hash,
            "shards": {"train": records},
            "samples": {"train": 4},
        },
    )

    with pytest.raises(ValueError, match="split order differs"):
        CachedFeatureDataset(manifest_path, "train")
