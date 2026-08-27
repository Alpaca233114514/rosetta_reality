from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosetta_reality.experiment import file_sha256
from scripts.prepare_smolvla import remanifest, snapshot_root, validate_snapshot


def _config() -> dict:
    return {
        "schema_version": 1,
        "source": "huggingface",
        "repo_id": "lerobot/smolvla_base",
        "revision": "a" * 40,
        "manifest": "model_manifest.json",
        "dependency_manifest": "vlm_dependency_manifest.json",
        "cache_root": "models",
        "expected": {
            "files": [
                "config.json",
                "model.safetensors",
                "tokenizer/tokenizer.json",
            ],
            "config": {
                "type": "smolvla",
                "chunk_size": 50,
                "n_action_steps": 50,
                "max_action_dim": 32,
                "max_state_dim": 32,
                "vlm_model_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
            },
        },
        "dependencies": {
            "vlm": {
                "repo_id": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
                "revision": "b" * 40,
                "license": "apache-2.0",
                "files": ["config.json"],
            }
        },
    }


def _record(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": file_sha256(path)}


def _snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, Path]:
    config = _config()
    monkeypatch.setenv("ROSETTA_MODELS_ROOT", str(tmp_path))
    root = snapshot_root(config)
    (root / "tokenizer").mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps(config["expected"]["config"]), encoding="utf-8"
    )
    (root / "model.safetensors").write_bytes(b"weights")
    (root / "tokenizer/tokenizer.json").write_text("{}", encoding="utf-8")
    previous_files = {
        name: _record(root / name)
        for name in ("config.json", "model.safetensors")
    }
    manifest = {
        "schema_version": 1,
        "status": "validated",
        "source": "huggingface",
        "repo_id": config["repo_id"],
        "revision": config["revision"],
        "model_contract": config["expected"]["config"],
        "files": previous_files,
        "total_bytes": sum(int(value["bytes"]) for value in previous_files.values()),
        "provenance_extension": {"preserve": True},
    }
    (root / "model_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return config, root


def test_remanifest_is_a_strict_provenance_preserving_superset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, root = _snapshot(tmp_path, monkeypatch)

    assert remanifest(config) == 0
    manifest = validate_snapshot(root, config)

    assert manifest["provenance_extension"] == {"preserve": True}
    assert set(manifest["files"]) == {
        "config.json",
        "model.safetensors",
        "tokenizer/tokenizer.json",
    }


def test_remanifest_refuses_changed_record_without_touching_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, root = _snapshot(tmp_path, monkeypatch)
    manifest_path = root / "model_manifest.json"
    before = manifest_path.read_bytes()
    (root / "model.safetensors").write_bytes(b"changed")

    with pytest.raises(ValueError, match="refuses to change"):
        remanifest(config)

    assert manifest_path.read_bytes() == before
