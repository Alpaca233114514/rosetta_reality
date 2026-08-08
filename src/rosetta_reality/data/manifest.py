"""Immutable Hub revision resolution and ignored cache manifests."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosetta_reality.data.hub import resolve_dataset_revision

COMMIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")

CACHE_CONTENT_DIRECTORIES = ("meta", "data", "videos", "images")


def require_commit_sha(revision: str) -> str:
    """Reject mutable branches or tags at the adapter boundary."""

    normalized = revision.lower()
    if not COMMIT_SHA_PATTERN.fullmatch(normalized):
        raise ValueError(f"Expected an immutable 40-character commit SHA, received {revision!r}.")
    return normalized


def resolve_hub_revision(repo_id: str, revision: str, api: Any | None = None) -> str:
    """Resolve a dataset branch or tag to its current immutable Hub commit."""

    if api is None:
        return require_commit_sha(resolve_dataset_revision(repo_id, revision))
    info = api.dataset_info(repo_id=repo_id, revision=revision)
    if not info.sha:
        raise RuntimeError(f"Hub did not return a commit SHA for {repo_id}@{revision}.")
    return require_commit_sha(str(info.sha))


def repository_cache_name(repo_id: str) -> str:
    """Return a filesystem-safe, collision-resistant Hub repository name."""

    parts = repo_id.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("repo_id must have the form 'owner/name'.")
    return "--".join(parts)


def revision_cache_root(cache_root: Path, repo_id: str, revision: str) -> Path:
    """Keep each immutable revision in a distinct local directory."""

    return cache_root / repository_cache_name(repo_id) / require_commit_sha(revision)


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Reproducibility metadata stored beside ignored dataset files."""

    repo_id: str
    requested_revision: str
    resolved_revision: str
    episodes: tuple[int, ...]
    cameras: dict[str, str]
    license: str
    fields: dict[str, str] | None = None
    source_format: str = "lerobot-v3"
    version: int = 2

    def __post_init__(self) -> None:
        require_commit_sha(self.resolved_revision)
        if self.version not in (1, 2):
            raise ValueError(f"Unsupported dataset manifest version: {self.version!r}.")
        if self.version == 2 and self.fields is None:
            raise ValueError("Dataset manifest version 2 requires a field mapping.")

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON payload."""

        payload = asdict(self)
        payload["episodes"] = list(self.episodes)
        if self.fields is None:
            payload.pop("fields")
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DatasetManifest:
        """Restore a supported manifest payload."""

        version = value.get("version")
        if version not in (1, 2):
            raise ValueError(f"Unsupported dataset manifest version: {value.get('version')!r}.")
        raw_fields = value.get("fields")
        if raw_fields is not None and not isinstance(raw_fields, dict):
            raise ValueError("Dataset manifest fields must be a mapping.")
        if version == 2 and raw_fields is None:
            raise ValueError("Dataset manifest version 2 requires a field mapping.")
        return cls(
            repo_id=str(value["repo_id"]),
            requested_revision=str(value["requested_revision"]),
            resolved_revision=str(value["resolved_revision"]),
            episodes=tuple(int(episode) for episode in value["episodes"]),
            cameras={str(name): str(key) for name, key in value["cameras"].items()},
            license=str(value["license"]),
            fields=(
                None
                if raw_fields is None
                else {str(name): str(key) for name, key in raw_fields.items()}
            ),
            source_format=str(value["source_format"]),
            version=int(version),
        )


def save_dataset_manifest(root: Path, manifest: DatasetManifest) -> Path:
    """Create a manifest once, or validate an identical existing one."""

    path = root / "manifest.json"
    if path.exists():
        existing = DatasetManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if existing != manifest:
            raise FileExistsError(f"Refusing to overwrite a different dataset manifest at {path}.")
        return path
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"Refusing to manage non-empty cache directory without a manifest: {root}."
        )
    root.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(manifest.to_dict(), file, indent=2, sort_keys=True)
        file.write("\n")
    return path


def load_dataset_manifest(path: Path) -> DatasetManifest:
    """Read a manifest without contacting the Hub or modifying the cache."""

    return DatasetManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))


def find_dataset_manifests(cache_root: Path, repo_id: str) -> list[Path]:
    """List cached revision manifests for read-only inspection."""

    repository_root = cache_root / repository_cache_name(repo_id)
    if not repository_root.exists():
        return []
    return sorted(repository_root.glob("*/manifest.json"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compute_cache_checksums(root: Path) -> dict[str, str]:
    """Hash immutable dataset content, excluding local manifests and statistics."""

    checksums: dict[str, str] = {}
    for directory_name in CACHE_CONTENT_DIRECTORIES:
        directory = root / directory_name
        if not directory.exists():
            continue
        for path in sorted(
            file
            for file in directory.rglob("*")
            if file.is_file() and not file.name.endswith(".partial")
        ):
            checksums[path.relative_to(root).as_posix()] = _sha256(path)
    if not checksums:
        raise ValueError(f"No dataset content files found under {root}.")
    return checksums


def save_cache_checksums(root: Path, checksums: dict[str, str]) -> Path:
    """Persist an integrity inventory once, or verify an identical inventory."""

    path = root / "cache_checksums.json"
    payload = {"version": 1, "algorithm": "sha256", "files": checksums}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise FileExistsError(f"Refusing to overwrite different cache checksums at {path}.")
        return path
    with path.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    return path


def validate_cache_checksums(root: Path) -> int:
    """Validate every recorded cache file without changing the cache."""

    path = root / "cache_checksums.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != 1 or payload.get("algorithm") != "sha256":
        raise ValueError(f"Unsupported cache checksum manifest at {path}.")
    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"Cache checksum manifest contains no files: {path}.")
    for relative_path, expected in files.items():
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe cache checksum path: {relative_path!r}.")
        content_path = root / relative
        if not content_path.is_file():
            raise FileNotFoundError(f"Cached dataset file is missing: {content_path}.")
        actual = _sha256(content_path)
        if actual != expected:
            raise ValueError(f"Cached dataset checksum mismatch: {content_path}.")
    return len(files)
