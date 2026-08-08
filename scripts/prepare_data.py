"""Idempotent preparation and read-only inspection for the M1 dataset."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from rosetta_reality.data import ActionChunkDataset
from rosetta_reality.data.adapters import LeRobotV3Adapter
from rosetta_reality.data.config import DatasetConfig, load_dataset_config
from rosetta_reality.data.hub import download_dataset_snapshot
from rosetta_reality.data.manifest import (
    DatasetManifest,
    compute_cache_checksums,
    find_dataset_manifests,
    load_dataset_manifest,
    resolve_hub_revision,
    revision_cache_root,
    save_cache_checksums,
    save_dataset_manifest,
    validate_cache_checksums,
)
from rosetta_reality.data.normalization import (
    DatasetStatistics,
    compute_dataset_statistics,
    load_dataset_statistics,
    save_dataset_statistics,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "data" / "aloha_sim_insertion.yaml"


def _cache_root(config: DatasetConfig) -> Path:
    if config.cache_root.is_absolute():
        return config.cache_root
    return REPOSITORY_ROOT / config.cache_root


def _directory_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _validate_expected(config: DatasetConfig, adapter: LeRobotV3Adapter) -> None:
    checks = (
        ("frames", len(adapter), config.expected_frames),
        ("state_dim", adapter.state_dim, config.expected_state_dim),
        ("action_dim", adapter.action_dim, config.expected_action_dim),
    )
    for name, actual, expected in checks:
        if expected is not None and actual != expected:
            raise ValueError(f"Expected {name}={expected}, received {actual}.")


def _validate_statistics(
    statistics: DatasetStatistics,
    adapter: LeRobotV3Adapter,
) -> None:
    if statistics.state_count != len(adapter) or statistics.action_count != len(adapter):
        raise ValueError("Cached statistics counts do not match the selected source frames.")
    if statistics.state.mean.shape != (adapter.state_dim,):
        raise ValueError("Cached state statistics dimension does not match the adapter.")
    if statistics.action.mean.shape != (adapter.action_dim,):
        raise ValueError("Cached action statistics dimension does not match the adapter.")


def prepare(config: DatasetConfig) -> int:
    """Resolve, cache, validate, and summarize the configured dataset."""

    resolved_revision = resolve_hub_revision(config.repo_id, config.revision)
    root = revision_cache_root(_cache_root(config), config.repo_id, resolved_revision)
    manifest = DatasetManifest(
        repo_id=config.repo_id,
        requested_revision=config.revision,
        resolved_revision=resolved_revision,
        episodes=config.episodes,
        cameras=config.cameras,
        license=config.license,
        fields=asdict(config.fields),
    )
    manifest_path = save_dataset_manifest(root, manifest)
    checksum_path = root / "cache_checksums.json"
    if checksum_path.exists():
        validate_cache_checksums(root)
    video_prefixes = tuple(f"videos/{source_key}/" for source_key in config.cameras.values())
    downloaded_paths = download_dataset_snapshot(
        repo_id=config.repo_id,
        revision=resolved_revision,
        root=root,
        prefixes=("meta/", "data/", *video_prefixes),
    )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    adapter = LeRobotV3Adapter(
        repo_id=config.repo_id,
        revision=resolved_revision,
        root=root,
        episodes=config.episodes,
        cameras=config.cameras,
        fields=config.fields,
        embodiment=config.embodiment,
        license_name=config.license,
    )
    _validate_expected(config, adapter)
    chunked_dataset = ActionChunkDataset(adapter, chunk_size=config.chunk_size)
    statistics_path = root / "statistics.json"
    if statistics_path.exists():
        statistics = load_dataset_statistics(statistics_path)
    else:
        statistics = compute_dataset_statistics(adapter)
        save_dataset_statistics(statistics_path, statistics)
    _validate_statistics(statistics, adapter)
    sample = chunked_dataset[0]
    if (
        config.expected_instruction is not None
        and sample.instruction != config.expected_instruction
    ):
        raise ValueError(
            f"Expected instruction {config.expected_instruction!r}, "
            f"received {sample.instruction!r}."
        )
    checksums = compute_cache_checksums(root)
    cache_file_count = len(checksums)
    save_cache_checksums(root, checksums)
    summary = {
        "instruction": sample.instruction,
        "episode_id": sample.episode_id,
        "frame_index": sample.frame_index,
        "timestamp": sample.timestamp,
        "state_shape": list(sample.robot_state.shape),
        "action_chunk_shape": list(sample.actions.shape),
        "image_shapes": {name: list(image.shape) for name, image in sample.images.items()},
    }
    print(f"Repository: {config.repo_id}")
    print(f"Requested revision: {config.revision}")
    print(f"Resolved revision: {resolved_revision}")
    print(f"License: {config.license}")
    print(f"Cache path: {root}")
    print(f"Cache bytes: {_directory_size(root)}")
    print(f"Manifest: {manifest_path}")
    print(f"Snapshot files: {len(downloaded_paths)}")
    print(f"Checksum files: {cache_file_count}")
    print(f"Checksums: {checksum_path}")
    print(f"Frames: {len(adapter)}")
    print(f"Samples: {len(chunked_dataset)}")
    print(f"State dimension: {adapter.state_dim}")
    print(f"Action dimension: {adapter.action_dim}")
    print(f"Chunk size: {config.chunk_size}")
    print(f"State statistic count: {statistics.state_count}")
    print(f"Action statistic count: {statistics.action_count}")
    print(f"Statistics: {statistics_path}")
    print(f"Sample: {json.dumps(summary, sort_keys=True)}")
    return 0


def inspect(config: DatasetConfig) -> int:
    """Inspect cached manifests and files without network access or writes."""

    manifests = find_dataset_manifests(_cache_root(config), config.repo_id)
    if not manifests:
        print(f"No prepared cache found for {config.repo_id}.")
        return 1
    for manifest_path in manifests:
        manifest = load_dataset_manifest(manifest_path)
        root = manifest_path.parent
        statistics_path = root / "statistics.json"
        checksum_path = root / "cache_checksums.json"
        metadata_path = root / "meta" / "info.json"
        print(f"Repository: {manifest.repo_id}")
        print(f"Resolved revision: {manifest.resolved_revision}")
        print(f"License: {manifest.license}")
        print(f"Episodes: {list(manifest.episodes)}")
        print(f"Cameras: {json.dumps(manifest.cameras, sort_keys=True)}")
        print(f"Cache path: {root}")
        print(f"Cache bytes: {_directory_size(root)}")
        print(f"Metadata present: {metadata_path.is_file()}")
        print(f"Statistics present: {statistics_path.is_file()}")
        print(
            "Checksum files: "
            f"{validate_cache_checksums(root) if checksum_path.is_file() else 'not recorded'}"
        )
    return 0


def main() -> int:
    """Run the default prepare command or the read-only cache inspector."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("prepare", "inspect"), default="prepare")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_dataset_config(args.config.resolve())
    if args.command == "inspect":
        return inspect(config)
    return prepare(config)


if __name__ == "__main__":
    raise SystemExit(main())
