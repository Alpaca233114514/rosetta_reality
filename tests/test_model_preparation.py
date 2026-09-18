from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from rosetta_reality.experiment import file_sha256
from scripts.prepare_model import adopt, load_config, validate_snapshot


def _config() -> dict[str, object]:
    return {
        "source": "huggingface",
        "repo_id": "Qwen/Qwen3.5-0.8B-Base",
        "revision": "a" * 40,
        "manifest": "model_manifest.json",
        "expected": {
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "hidden_size": 1024,
            "files": ["config.json", "model.safetensors"],
        },
    }


def test_model_config_requires_immutable_revision(tmp_path: Path) -> None:
    value = _config()
    value["revision"] = "main"
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")

    with pytest.raises(ValueError, match="immutable"):
        load_config(path)


def test_model_manifest_binds_base_repo_revision_and_checksums(tmp_path: Path) -> None:
    config = _config()
    model_config = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {"hidden_size": 1024},
    }
    (tmp_path / "config.json").write_text(json.dumps(model_config), encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    files = {
        name: {
            "bytes": (tmp_path / name).stat().st_size,
            "sha256": file_sha256(tmp_path / name),
        }
        for name in config["expected"]["files"]  # type: ignore[index]
    }
    manifest = {
        "schema_version": 1,
        "status": "validated",
        "source": config["source"],
        "repo_id": config["repo_id"],
        "revision": config["revision"],
        "model_contract": {
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "hidden_size": 1024,
        },
        "files": files,
        "total_bytes": sum(record["bytes"] for record in files.values()),
    }
    (tmp_path / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert validate_snapshot(tmp_path, config)["repo_id"].endswith("-Base")

    (tmp_path / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_snapshot(tmp_path, config)


def test_adopt_validates_upstream_digest_before_creating_manifest(tmp_path: Path) -> None:
    config = _config()
    model_config = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {"hidden_size": 1024},
    }
    (tmp_path / "config.json").write_text(json.dumps(model_config), encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"existing local weights")
    config["local_snapshot_origin"] = "test_cache"
    expected = config["expected"]
    assert isinstance(expected, dict)
    expected["upstream_digests"] = {
        "model.safetensors": {
            "algorithm": "sha256",
            "digest": file_sha256(tmp_path / "model.safetensors"),
        }
    }

    assert adopt(tmp_path, config) == 0
    manifest = validate_snapshot(tmp_path, config)

    assert manifest["local_adoption"]["origin"] == "test_cache"
    assert manifest["upstream_verification"]["model.safetensors"]["algorithm"] == (
        "sha256"
    )
