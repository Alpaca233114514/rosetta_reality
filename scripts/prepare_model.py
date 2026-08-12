"""Revision-pinned, resumable preparation and offline inspection of a local model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "models" / "qwen35_08b_base.yaml"
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rosetta_reality.experiment import file_sha256  # noqa: E402
from rosetta_reality.features import create_json  # noqa: E402


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping.")
    return value


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate the immutable source identity and expected model contract."""

    value = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "Model config")
    if value.get("source") != "huggingface":
        raise ValueError("Only the explicit Hugging Face model source is supported.")
    repo_id = str(value.get("repo_id", ""))
    if repo_id.count("/") != 1 or any(part in ("", ".", "..") for part in repo_id.split("/")):
        raise ValueError("Model repo_id must be a namespace/name identifier.")
    revision = str(value.get("revision", "")).lower()
    if not COMMIT_PATTERN.fullmatch(revision):
        raise ValueError("Model revision must be an immutable 40-character commit SHA.")
    expected = _mapping(value.get("expected"), "Model config expected")
    files = expected.get("files")
    if not isinstance(files, list) or not files or len(set(map(str, files))) != len(files):
        raise ValueError("Model expected.files must be a non-empty unique list.")
    upstream_digests = expected.get("upstream_digests", {})
    if not isinstance(upstream_digests, dict):
        raise ValueError("Model expected.upstream_digests must be a mapping.")
    for relative, record in upstream_digests.items():
        if relative not in set(map(str, files)) or not isinstance(record, dict):
            raise ValueError("Upstream digest entries must identify declared model files.")
        algorithm = record.get("algorithm")
        digest = str(record.get("digest", "")).lower()
        expected_length = {"sha256": 64, "git_blob_sha1": 40}.get(algorithm)
        if expected_length is None or not re.fullmatch(
            rf"[0-9a-f]{{{expected_length}}}", digest
        ):
            raise ValueError("Unsupported or malformed upstream model digest.")
    return value


def _models_root(config: dict[str, Any]) -> Path:
    override = os.environ.get("ROSETTA_MODELS_ROOT")
    root = Path(override) if override else REPOSITORY_ROOT / str(config["cache_root"])
    if override and not root.is_absolute():
        raise ValueError("ROSETTA_MODELS_ROOT must be absolute.")
    return root


def snapshot_root(config: dict[str, Any]) -> Path:
    """Return a revision-scoped destination without machine-specific identity fields."""

    namespace, name = str(config["repo_id"]).split("/", maxsplit=1)
    return _models_root(config) / f"{namespace}--{name}" / str(config["revision"])


def _config_contract(root: Path, expected: dict[str, Any]) -> dict[str, Any]:
    raw = json.loads((root / "config.json").read_text(encoding="utf-8"))
    text_config = raw.get("text_config", raw)
    received = {
        "model_type": raw.get("model_type"),
        "architectures": raw.get("architectures"),
        "hidden_size": text_config.get("hidden_size"),
    }
    declared = {
        "model_type": expected["model_type"],
        "architectures": expected["architectures"],
        "hidden_size": expected["hidden_size"],
    }
    if received != declared:
        raise ValueError(
            f"Downloaded model config differs from the declared Base contract: {received!r}."
        )
    return received


def _file_records(root: Path, expected_files: list[str]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for relative_text in expected_files:
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe expected model path: {relative_text!r}.")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Expected model file is missing: {relative.as_posix()}.")
        records[relative.as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    return records


def _git_blob_sha1(path: Path) -> str:
    """Return the Git object identity for an existing working-tree file."""

    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_upstream_digests(root: Path, expected: dict[str, Any]) -> dict[str, Any]:
    """Verify configured upstream object identities without network access."""

    configured = expected.get("upstream_digests", {})
    result: dict[str, Any] = {}
    for relative_text, record in sorted(configured.items()):
        relative = Path(relative_text)
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Upstream-verified model file is missing: {relative}.")
        algorithm = str(record["algorithm"])
        received = (
            file_sha256(path) if algorithm == "sha256" else _git_blob_sha1(path)
        )
        expected_digest = str(record["digest"]).lower()
        if received != expected_digest:
            raise ValueError(
                f"Local model file differs from its pinned upstream identity: {relative}."
            )
        result[relative.as_posix()] = {
            "algorithm": algorithm,
            "digest": received,
        }
    return result


def validate_snapshot(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Validate repository identity, config contract, exact files, and checksums."""

    manifest_path = root / str(config["manifest"])
    if not manifest_path.is_file():
        raise FileNotFoundError("The revision-scoped model manifest is missing.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "validated"
        or manifest.get("source") != config["source"]
        or manifest.get("repo_id") != config["repo_id"]
        or manifest.get("revision") != config["revision"]
    ):
        raise ValueError("Model manifest source identity does not match the pinned config.")
    contract = _config_contract(root, _mapping(config["expected"], "expected"))
    if manifest.get("model_contract") != contract:
        raise ValueError("Model manifest contract differs from config.json.")
    expected_files = [str(value) for value in config["expected"]["files"]]
    if set(manifest.get("files", {})) != set(expected_files):
        raise ValueError("Model manifest file set differs from the pinned source declaration.")
    for relative, record in manifest["files"].items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Manifest model file is missing: {relative}.")
        if path.stat().st_size != int(record["bytes"]) or file_sha256(path) != record["sha256"]:
            raise ValueError(f"Model file checksum mismatch: {relative}.")
    upstream_verification = _verify_upstream_digests(
        root,
        _mapping(config["expected"], "expected"),
    )
    if manifest.get("upstream_verification", {}) != upstream_verification:
        raise ValueError("Model manifest upstream verification differs from pinned config.")
    return manifest


def prepare(config: dict[str, Any], *, attempts: int) -> int:
    """Download one pinned snapshot with bounded retries and create an immutable manifest."""

    root = snapshot_root(config)
    manifest_path = root / str(config["manifest"])
    if manifest_path.is_file():
        manifest = validate_snapshot(root, config)
        print(json.dumps({"status": "already_validated", **manifest}, indent=2, sort_keys=True))
        return 0
    root.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            print(f"Download attempt {attempt}/{attempts}", flush=True)
            snapshot_download(
                repo_id=str(config["repo_id"]),
                revision=str(config["revision"]),
                local_dir=root,
                max_workers=1,
            )
            error = None
            break
        except Exception as caught:  # noqa: BLE001 - retry at the process boundary
            error = caught
            print(
                f"Download attempt {attempt} failed: {type(caught).__name__}: {caught}",
                flush=True,
            )
            if attempt < attempts:
                time.sleep(min(30, attempt * 5))
    if error is not None:
        raise RuntimeError(f"Pinned model download failed after {attempts} attempts.") from error

    expected = _mapping(config["expected"], "expected")
    contract = _config_contract(root, expected)
    records = _file_records(root, [str(value) for value in expected["files"]])
    upstream_verification = _verify_upstream_digests(root, expected)
    manifest = {
        "schema_version": 1,
        "status": "validated",
        "source": config["source"],
        "repo_id": config["repo_id"],
        "revision": config["revision"],
        "model_contract": contract,
        "files": records,
        "total_bytes": sum(int(record["bytes"]) for record in records.values()),
        "upstream_verification": upstream_verification,
    }
    create_json(manifest_path, manifest)
    validate_snapshot(root, config)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def adopt(root: Path, config: dict[str, Any]) -> int:
    """Create-only validation of an existing offline snapshot at an explicit path."""

    if not root.is_absolute() or not root.is_dir():
        raise ValueError("Adopted model root must be an existing absolute directory.")
    manifest_path = root / str(config["manifest"])
    if manifest_path.is_file():
        manifest = validate_snapshot(root, config)
        print(json.dumps({"status": "already_validated", **manifest}, indent=2, sort_keys=True))
        return 0
    expected = _mapping(config["expected"], "expected")
    contract = _config_contract(root, expected)
    records = _file_records(root, [str(value) for value in expected["files"]])
    upstream_verification = _verify_upstream_digests(root, expected)
    manifest = {
        "schema_version": 1,
        "status": "validated",
        "source": config["source"],
        "repo_id": config["repo_id"],
        "revision": config["revision"],
        "model_contract": contract,
        "files": records,
        "total_bytes": sum(int(record["bytes"]) for record in records.values()),
        "upstream_verification": upstream_verification,
        "local_adoption": {
            "origin": str(config.get("local_snapshot_origin", "unspecified")),
            "provider_extra_files": list(config.get("provider_extra_files", [])),
            "provider_packaging_variations": list(
                config.get("provider_packaging_variations", [])
            ),
        },
    }
    create_json(manifest_path, manifest)
    validate_snapshot(root, config)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def inspect(config: dict[str, Any]) -> int:
    """Verify a prepared snapshot without network access or file mutation."""

    root = snapshot_root(config)
    manifest = validate_snapshot(root, config)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "repo_id": manifest["repo_id"],
                "revision": manifest["revision"],
                "total_bytes": manifest["total_bytes"],
                "files": len(manifest["files"]),
                "model_contract": manifest["model_contract"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "inspect", "adopt"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--local-root", type=Path)
    args = parser.parse_args()
    if args.attempts <= 0:
        raise ValueError("--attempts must be positive.")
    config = load_config(args.config.resolve())
    if args.command == "adopt":
        if args.local_root is None:
            parser.error("adopt requires --local-root")
        return adopt(args.local_root.resolve(), config)
    if args.command == "inspect":
        return inspect(config)
    return prepare(config, attempts=args.attempts)


if __name__ == "__main__":
    raise SystemExit(main())
