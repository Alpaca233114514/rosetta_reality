"""Prepare and verify one revision-pinned local SmolVLA policy snapshot."""

from __future__ import annotations

import argparse
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
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs/models/smolvla_450m.yaml"
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
    value = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "Model config")
    if value.get("schema_version") != 1 or value.get("source") != "huggingface":
        raise ValueError("SmolVLA model config must be schema-v1 Hugging Face source.")
    repo_id = str(value.get("repo_id", ""))
    if repo_id.count("/") != 1:
        raise ValueError("SmolVLA repo_id must be a namespace/name identifier.")
    revision = str(value.get("revision", "")).lower()
    if not COMMIT_PATTERN.fullmatch(revision):
        raise ValueError("SmolVLA revision must be an immutable commit SHA.")
    expected = _mapping(value.get("expected"), "expected")
    files = expected.get("files")
    if not isinstance(files, list) or not files or len(set(files)) != len(files):
        raise ValueError("expected.files must be a non-empty unique list.")
    _mapping(expected.get("config"), "expected.config")
    dependencies = _mapping(value.get("dependencies"), "dependencies")
    dependency = _mapping(dependencies.get("vlm"), "dependencies.vlm")
    dependency_repo_id = str(dependency.get("repo_id", ""))
    if dependency_repo_id.count("/") != 1:
        raise ValueError("VLM dependency repo_id must be a namespace/name identifier.")
    dependency_revision = str(dependency.get("revision", "")).lower()
    if not COMMIT_PATTERN.fullmatch(dependency_revision):
        raise ValueError("VLM dependency revision must be an immutable commit SHA.")
    dependency_files = dependency.get("files")
    if (
        not isinstance(dependency_files, list)
        or not dependency_files
        or len(set(dependency_files)) != len(dependency_files)
    ):
        raise ValueError("dependencies.vlm.files must be a non-empty unique list.")
    if expected["config"].get("vlm_model_name") != dependency_repo_id:
        raise ValueError("The SmolVLA model contract and VLM dependency differ.")
    if not str(dependency.get("license", "")).strip():
        raise ValueError("The VLM dependency license must be recorded.")
    if not str(value.get("dependency_manifest", "")).strip():
        raise ValueError("dependency_manifest must be configured.")
    return value


def _models_root(config: dict[str, Any]) -> Path:
    raw = os.environ.get("ROSETTA_MODELS_ROOT")
    root = Path(raw) if raw else REPOSITORY_ROOT / str(config["cache_root"])
    if raw and not root.is_absolute():
        raise ValueError("ROSETTA_MODELS_ROOT must be absolute.")
    return root


def snapshot_root(config: dict[str, Any]) -> Path:
    namespace, name = str(config["repo_id"]).split("/", maxsplit=1)
    return _models_root(config) / f"{namespace}--{name}" / str(config["revision"])


def _hf_home(config: dict[str, Any]) -> Path:
    raw = os.environ.get("HF_HOME")
    root = Path(raw) if raw else _models_root(config) / "hf_home"
    if raw and not root.is_absolute():
        raise ValueError("HF_HOME must be absolute.")
    return root


def dependency_snapshot_root(config: dict[str, Any]) -> Path:
    dependency = _mapping(config["dependencies"]["vlm"], "dependencies.vlm")
    namespace, name = str(dependency["repo_id"]).split("/", maxsplit=1)
    return (
        _hf_home(config)
        / "hub"
        / f"models--{namespace}--{name}"
        / "snapshots"
        / str(dependency["revision"])
    )


def _dependency_reference(config: dict[str, Any]) -> Path:
    dependency = _mapping(config["dependencies"]["vlm"], "dependencies.vlm")
    namespace, name = str(dependency["repo_id"]).split("/", maxsplit=1)
    return _hf_home(config) / "hub" / f"models--{namespace}--{name}" / "refs" / "main"


def _contract(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    raw = _mapping(json.loads((root / "config.json").read_text(encoding="utf-8")), "config.json")
    expected = _mapping(config["expected"]["config"], "expected.config")
    received = {key: raw.get(key) for key in expected}
    if received != expected:
        raise ValueError(f"SmolVLA config contract differs: {received!r}.")
    return received


def _file_records(root: Path, files: list[str]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for relative_text in files:
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe expected model path: {relative_text!r}.")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Expected SmolVLA file is missing: {relative.as_posix()}.")
        records[relative.as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    return records


def validate_snapshot(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = root / str(config["manifest"])
    if not manifest_path.is_file():
        raise FileNotFoundError("The SmolVLA model manifest is missing.")
    manifest = _mapping(json.loads(manifest_path.read_text(encoding="utf-8")), "manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "validated"
        or manifest.get("repo_id") != config["repo_id"]
        or manifest.get("revision") != config["revision"]
    ):
        raise ValueError("SmolVLA manifest identity differs from the pinned config.")
    contract = _contract(root, config)
    if manifest.get("model_contract") != contract:
        raise ValueError("SmolVLA manifest contract differs from config.json.")
    expected_files = [str(value) for value in config["expected"]["files"]]
    current = _file_records(root, expected_files)
    if manifest.get("files") != current:
        raise ValueError("SmolVLA file identities differ from the immutable manifest.")
    return manifest


def _model_card_license(path: Path) -> str | None:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return None
    _, frontmatter, _ = text.split("---", maxsplit=2)
    value = yaml.safe_load(frontmatter)
    if not isinstance(value, dict) or value.get("license") is None:
        return None
    return str(value["license"])


def validate_dependency(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    dependency = _mapping(config["dependencies"]["vlm"], "dependencies.vlm")
    manifest_path = root / str(config["dependency_manifest"])
    if not manifest_path.is_file():
        raise FileNotFoundError("The pinned VLM dependency manifest is missing.")
    manifest = _mapping(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        "dependency manifest",
    )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "validated"
        or manifest.get("repo_id") != dependency["repo_id"]
        or manifest.get("revision") != dependency["revision"]
        or manifest.get("license") != dependency["license"]
    ):
        raise ValueError("VLM dependency manifest identity differs from the pinned config.")
    reference = _dependency_reference(config)
    if (
        not reference.is_file()
        or reference.read_text(encoding="utf-8").strip() != dependency["revision"]
    ):
        raise ValueError(
            "The offline Hugging Face main reference is not pinned to the VLM revision."
        )
    expected_files = [str(value) for value in dependency["files"]]
    current = _file_records(dependency_snapshot_root(config), expected_files)
    if manifest.get("files") != current:
        raise ValueError("VLM dependency file identities differ from the immutable manifest.")
    if _model_card_license(dependency_snapshot_root(config) / "README.md") != dependency["license"]:
        raise ValueError("VLM dependency model-card license differs from the pinned config.")
    return manifest


def prepare_dependency(root: Path, config: dict[str, Any], attempts: int) -> dict[str, Any]:
    manifest_path = root / str(config["dependency_manifest"])
    if manifest_path.is_file():
        return validate_dependency(root, config)
    dependency = _mapping(config["dependencies"]["vlm"], "dependencies.vlm")
    from huggingface_hub import snapshot_download

    error: Exception | None = None
    resolved: Path | None = None
    for attempt in range(1, attempts + 1):
        try:
            print(f"VLM dependency download attempt {attempt}/{attempts}", flush=True)
            resolved = Path(
                snapshot_download(
                    repo_id=dependency["repo_id"],
                    revision=dependency["revision"],
                    cache_dir=_hf_home(config) / "hub",
                    allow_patterns=[str(value) for value in dependency["files"]],
                    max_workers=1,
                )
            )
            if resolved.name != dependency["revision"]:
                raise RuntimeError(
                    "The resolved VLM dependency revision moved from the preregistered commit."
                )
            error = None
            break
        except Exception as caught:  # noqa: BLE001 - retry at the process boundary
            error = caught
            print(f"VLM dependency attempt {attempt} failed: {type(caught).__name__}", flush=True)
            if attempt < attempts:
                time.sleep(min(30, attempt * 5))
    if error is not None or resolved is None:
        raise RuntimeError("Pinned VLM dependency download failed.") from error
    expected_root = dependency_snapshot_root(config)
    if resolved.resolve() != expected_root.resolve():
        raise RuntimeError("The VLM dependency cache resolved outside its pinned snapshot root.")
    files = _file_records(expected_root, [str(value) for value in dependency["files"]])
    if _model_card_license(expected_root / "README.md") != dependency["license"]:
        raise ValueError("VLM dependency model-card license differs from the pinned config.")
    manifest = {
        "schema_version": 1,
        "status": "validated",
        "source": "huggingface",
        "repo_id": dependency["repo_id"],
        "revision": dependency["revision"],
        "license": dependency["license"],
        "cache_layout": (
            Path("hub")
            / f"models--{str(dependency['repo_id']).replace('/', '--')}"
            / "snapshots"
            / str(dependency["revision"])
        ).as_posix(),
        "files": files,
        "total_bytes": sum(record["bytes"] for record in files.values()),
    }
    create_json(manifest_path, manifest)
    return validate_dependency(root, config)


def prepare(config: dict[str, Any], attempts: int) -> int:
    root = snapshot_root(config)
    manifest_path = root / str(config["manifest"])
    if manifest_path.is_file():
        model_manifest = validate_snapshot(root, config)
        dependency_manifest = prepare_dependency(root, config, attempts)
        payload = {
            "status": "already_validated",
            "model": model_manifest,
            "vlm_dependency": dependency_manifest,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    root.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            print(f"Download attempt {attempt}/{attempts}", flush=True)
            snapshot_download(
                repo_id=config["repo_id"],
                revision=config["revision"],
                local_dir=root,
                allow_patterns=[str(value) for value in config["expected"]["files"]],
                max_workers=1,
            )
            error = None
            break
        except Exception as caught:  # noqa: BLE001 - retry at the process boundary
            error = caught
            print(f"Download attempt {attempt} failed: {type(caught).__name__}", flush=True)
            if attempt < attempts:
                time.sleep(min(30, attempt * 5))
    if error is not None:
        raise RuntimeError("Pinned SmolVLA download failed.") from error
    contract = _contract(root, config)
    files = _file_records(root, [str(value) for value in config["expected"]["files"]])
    manifest = {
        "schema_version": 1,
        "status": "validated",
        "source": "huggingface",
        "repo_id": config["repo_id"],
        "revision": config["revision"],
        "model_contract": contract,
        "files": files,
        "total_bytes": sum(record["bytes"] for record in files.values()),
    }
    create_json(manifest_path, manifest)
    validate_snapshot(root, config)
    dependency_manifest = prepare_dependency(root, config, attempts)
    print(
        json.dumps(
            {"model": manifest, "vlm_dependency": dependency_manifest},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def inspect(config: dict[str, Any]) -> int:
    manifest = validate_snapshot(snapshot_root(config), config)
    dependency = validate_dependency(snapshot_root(config), config)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "repo_id": manifest["repo_id"],
                "revision": manifest["revision"],
                "total_bytes": manifest["total_bytes"],
                "model_contract": manifest["model_contract"],
                "vlm_dependency": {
                    "repo_id": dependency["repo_id"],
                    "revision": dependency["revision"],
                    "license": dependency["license"],
                    "total_bytes": dependency["total_bytes"],
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "inspect"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--attempts", type=int, default=5)
    args = parser.parse_args()
    if args.attempts <= 0:
        raise ValueError("--attempts must be positive.")
    config = load_config(args.config.resolve())
    return prepare(config, args.attempts) if args.command == "prepare" else inspect(config)


if __name__ == "__main__":
    raise SystemExit(main())
