"""Read-only resolution of one exact prepared dataset cache."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rosetta_reality.data.config import DatasetConfig, resolve_dataset_cache_root
from rosetta_reality.data.manifest import (
    COMMIT_SHA_PATTERN,
    DatasetManifest,
    find_dataset_manifests,
    load_dataset_manifest,
    validate_cache_checksums,
)


def resolve_prepared_cache(
    config: DatasetConfig,
    repository_root: Path,
    *,
    validate_checksums: bool = True,
) -> tuple[Path, DatasetManifest]:
    """Resolve exactly one manifest matching every configured identity field."""

    cache_root = resolve_dataset_cache_root(config, repository_root)
    revision_is_pinned = bool(COMMIT_SHA_PATTERN.fullmatch(config.revision.lower()))
    expected_fields = asdict(config.fields)
    matches: list[tuple[Path, DatasetManifest]] = []
    for manifest_path in find_dataset_manifests(cache_root, config.repo_id):
        manifest = load_dataset_manifest(manifest_path)
        revision_matches = (
            manifest.resolved_revision == config.revision.lower()
            if revision_is_pinned
            else manifest.requested_revision == config.revision
        )
        if (
            revision_matches
            and manifest.episodes == config.episodes
            and manifest.cameras == config.cameras
            and manifest.fields == expected_fields
        ):
            matches.append((manifest_path.parent, manifest))
    if not matches:
        raise FileNotFoundError(
            "No prepared cache exactly matches the configured revision, episodes, cameras, "
            "and field mapping."
        )
    if len(matches) != 1:
        raise ValueError("Multiple prepared caches match; use an immutable dataset revision.")
    root, manifest = matches[0]
    if validate_checksums:
        validate_cache_checksums(root)
    return root, manifest


def ordered_feature_names(root: Path, feature_key: str) -> tuple[str, ...]:
    """Read ordered scalar names from list- or group-shaped LeRobot metadata."""

    info_path = root / "meta" / "info.json"
    info: dict[str, Any] = json.loads(info_path.read_text(encoding="utf-8"))
    try:
        feature = info["features"][feature_key]
    except KeyError as error:
        raise ValueError(f"Dataset metadata does not declare feature {feature_key!r}.") from error

    raw_names = feature.get("names")
    if isinstance(raw_names, list):
        names = raw_names
    elif isinstance(raw_names, dict) and raw_names:
        names = []
        for group, members in raw_names.items():
            if not isinstance(members, list) or not members:
                raise ValueError(
                    f"Dataset metadata name group {group!r} for {feature_key!r} is invalid."
                )
            names.extend(members)
    else:
        raise ValueError(
            f"Dataset metadata does not declare ordered scalar names for {feature_key!r}."
        )

    ordered = tuple(str(name) for name in names)
    shape = feature.get("shape")
    if not isinstance(shape, list) or len(shape) != 1 or shape[0] != len(ordered):
        raise ValueError(
            f"Dataset metadata names for {feature_key!r} do not match its vector shape."
        )
    return ordered
