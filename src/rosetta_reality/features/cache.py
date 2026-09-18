"""Create-only storage and validated loading for pooled feature shards."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from rosetta_reality.experiment import file_sha256, stable_hash


def create_json(path: Path, payload: dict[str, Any]) -> Path:
    """Create JSON once, or validate that an existing file is identical."""

    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise FileExistsError(f"Refusing to overwrite different JSON content at {path}.")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    return path


def save_tensor_shard(path: Path, payload: dict[str, Any]) -> Path:
    """Save one new shard through a uniquely named partial file."""

    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing feature shard: {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.partial")
    with temporary.open("xb") as file:
        torch.save(payload, file)
        file.flush()
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(f"Feature shard appeared concurrently: {path}.") from error
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_feature_manifest(path: Path) -> dict[str, Any]:
    """Load a complete cache manifest and validate its schema marker."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or value.get("status") != "complete":
        raise ValueError(f"Feature cache is not a complete schema-v1 cache: {path}.")
    identity = value.get("identity")
    if not isinstance(identity, dict) or value.get("identity_hash") != stable_hash(
        identity
    ):
        raise ValueError(f"Feature cache identity hash mismatch: {path}.")
    return value


class CachedFeatureDataset(Dataset[dict[str, Tensor]]):
    """Load small pooled feature shards for one declared episode split."""

    def __init__(self, manifest_path: Path, split: str) -> None:
        manifest = load_feature_manifest(manifest_path)
        raw_shards = manifest.get("shards", {}).get(split)
        if not isinstance(raw_shards, list) or not raw_shards:
            raise ValueError(f"Feature cache has no shards for split {split!r}.")
        cache_root = manifest_path.parent
        pieces: dict[str, list[Tensor]] = {
            "features": [],
            "robot_state": [],
            "actions": [],
            "episode_ids": [],
            "frame_indices": [],
        }
        expected_identity = str(manifest["identity_hash"])
        expected_transform = (
            manifest.get("identity", {}).get("selection", {}).get("action_transform")
        )
        loaded_episodes: list[int] = []
        loaded_episode_set: set[int] = set()
        loaded_paths: set[str] = set()
        for shard in raw_shards:
            if not isinstance(shard, dict):
                raise ValueError("Feature cache shard records must be mappings.")
            relative_text = str(shard["path"])
            relative = Path(relative_text)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe feature shard path: {relative}.")
            episode = int(shard["episode"])
            if episode in loaded_episode_set:
                raise ValueError(f"Feature cache repeats episode {episode} for split {split!r}.")
            relative_key = relative.as_posix()
            if relative_key in loaded_paths:
                raise ValueError(
                    f"Feature cache repeats shard path {relative} for split {split!r}."
                )
            loaded_episodes.append(episode)
            loaded_episode_set.add(episode)
            loaded_paths.add(relative_key)
            path = cache_root / relative
            if file_sha256(path) != shard["sha256"]:
                raise ValueError(f"Feature shard checksum mismatch: {relative}.")
            value = torch.load(path, map_location="cpu", weights_only=True)
            if value.get("identity_hash") != expected_identity or value.get("split") != split:
                raise ValueError(f"Feature shard identity mismatch: {relative}.")
            if int(value.get("episode", -1)) != episode:
                raise ValueError(f"Feature shard episode identity mismatch: {relative}.")
            if expected_transform:
                payload_transform = value.get("action_transform")
                if (
                    not isinstance(payload_transform, dict)
                    or payload_transform.get("type") != expected_transform
                    or payload_transform != shard.get("action_transform")
                ):
                    raise ValueError(
                        f"Feature shard action-transform provenance mismatch: {relative}."
                    )
            for key in pieces:
                tensor = value.get(key)
                if not isinstance(tensor, Tensor):
                    raise ValueError(f"Feature shard {relative} is missing tensor {key!r}.")
                if not bool(torch.isfinite(tensor).all()):
                    raise ValueError(f"Feature shard {relative} has non-finite tensor {key!r}.")
                pieces[key].append(tensor)
            if int(shard["samples"]) != value["features"].shape[0]:
                raise ValueError(f"Feature shard declared sample count mismatch: {relative}.")
            if not bool(value["episode_ids"].eq(episode).all()):
                raise ValueError(f"Feature shard episode tensor mismatch: {relative}.")
        self.manifest = manifest
        self.split = split
        self.features = torch.cat(pieces["features"]).to(torch.float32)
        self.robot_state = torch.cat(pieces["robot_state"]).to(torch.float32)
        self.actions = torch.cat(pieces["actions"]).to(torch.float32)
        self.episode_ids = torch.cat(pieces["episode_ids"]).to(torch.long)
        self.frame_indices = torch.cat(pieces["frame_indices"]).to(torch.long)
        count = self.features.shape[0]
        if any(
            tensor.shape[0] != count
            for tensor in (
                self.robot_state,
                self.actions,
                self.episode_ids,
                self.frame_indices,
            )
        ):
            raise ValueError("Feature shard tensors do not share a sample dimension.")
        if self.features.ndim != 2 or self.actions.ndim != 3 or self.robot_state.ndim != 2:
            raise ValueError(
                "Feature cache tensor ranks violate the cached representation contract."
            )
        declared_total = int(manifest.get("samples", {}).get(split, -1))
        if declared_total != count:
            raise ValueError("Feature manifest total differs from loaded shard samples.")
        declared_episodes = manifest.get("identity", {}).get("split", {}).get(split)
        if declared_episodes is not None:
            if not isinstance(declared_episodes, list):
                raise ValueError("Feature manifest episode split must be an ordered list.")
            declared_episode_values = [int(value) for value in declared_episodes]
            if len(declared_episode_values) != len(set(declared_episode_values)):
                raise ValueError("Feature manifest episode split contains duplicates.")
            if declared_episode_values != loaded_episodes:
                raise ValueError(
                    "Feature manifest episode split order differs from loaded shards."
                )

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {
            "features": self.features[index],
            "robot_state": self.robot_state[index],
            "actions": self.actions[index],
            "episode_id": self.episode_ids[index],
            "frame_index": self.frame_indices[index],
        }
