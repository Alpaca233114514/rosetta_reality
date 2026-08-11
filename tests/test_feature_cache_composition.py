from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from rosetta_reality.experiment import file_sha256, stable_hash
from rosetta_reality.features import CachedFeatureDataset, create_json, save_tensor_shard
from rosetta_reality.sim import load_action_contract
from scripts import cache_features

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _identity(
    *,
    experiment_id: str,
    pooling: str,
    normalization: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "experiment_config_sha256": f"config-{experiment_id}",
        "code": {"workspace_tree_sha256": f"tree-{experiment_id}"},
        "dataset": {"repo_id": "dataset", "revision": "revision"},
        "split": {"train": [1], "validation": [2], "test": [3]},
        "selection": {
            "frame_stride": 1,
            "action_chunk_length": 100,
            "action_transform": "clip_to_rosetta_contract_v1",
        },
        "model": {
            "identifier": "Qwen/Qwen3.5-0.8B-Base",
            "revision": "model-revision",
            "hidden_size": 2,
        },
        "processor": {"prompt": "Act: {instruction}"},
        "feature": {
            "layer": "final_hidden_state",
            "pooling": pooling,
            "storage_dtype": "float16",
        },
        "normalization_sha256": stable_hash(normalization),
        "action_contract_sha256": "action-contract",
    }


def test_model_identity_hashes_every_manifest_declared_file(tmp_path: Path) -> None:
    configured = {
        "family": "qwen35",
        "identifier": "Qwen/Qwen3.5-0.8B-Base",
        "scale": "0.8B",
        "adaptation": "frozen",
        "manifest": "model_manifest.json",
    }
    files = {
        "config.json": json.dumps({"text_config": {"hidden_size": 2}}).encode(),
        "model.safetensors": b"weights",
        "chat_template.jinja": b"{{ messages }}",
    }
    for name, value in files.items():
        (tmp_path / name).write_bytes(value)
    records = {
        name: {"bytes": len(value), "sha256": file_sha256(tmp_path / name)}
        for name, value in files.items()
    }
    manifest = {
        "schema_version": 1,
        "status": "validated",
        "source": "huggingface",
        "repo_id": configured["identifier"],
        "revision": "a" * 40,
        "model_contract": {"hidden_size": 2},
        "files": records,
    }
    (tmp_path / "model_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    identity = cache_features._model_identity(tmp_path, configured)

    assert set(identity["files"]) == set(files)
    (tmp_path / "chat_template.jinja").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="chat_template.jinja"):
        cache_features._model_identity(tmp_path, configured)


def test_feature_cache_rejects_source_overshoot_above_contract_tolerance() -> None:
    contract = load_action_contract(
        REPOSITORY_ROOT / "configs" / "sim" / "aloha_insertion.yaml"
    )
    within = contract.source_overshoot_tolerances.clone()
    cache_features._validate_source_overshoot(contract, within, context="test frame")
    beyond = within.clone()
    beyond[0] = 0.01

    with pytest.raises(ValueError, match="left_waist"):
        cache_features._validate_source_overshoot(contract, beyond, context="test frame")


def _source_cache(
    root: Path,
    *,
    experiment_id: str,
    pooling: str,
    feature_dim: int,
    normalization: dict[str, Any],
    state_offset: float = 0.0,
) -> Path:
    identity = _identity(
        experiment_id=experiment_id,
        pooling=pooling,
        normalization=normalization,
    )
    identity_hash = stable_hash(identity)
    create_json(root / "normalization.json", normalization)
    shards: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    transform = {"type": "clip_to_rosetta_contract_v1", "clipped_elements": 0}
    for split, episode, count in (
        ("train", 1, 2),
        ("validation", 2, 1),
        ("test", 3, 1),
    ):
        relative = Path("shards") / split / f"episode-{episode:03d}.pt"
        payload = {
            "schema_version": 1,
            "identity_hash": identity_hash,
            "split": split,
            "episode": episode,
            "features": torch.arange(count * feature_dim, dtype=torch.float16).reshape(
                count, feature_dim
            ),
            "robot_state": torch.full((count, 14), state_offset, dtype=torch.float32),
            "actions": torch.zeros(count, 100, 14),
            "episode_ids": torch.full((count,), episode, dtype=torch.long),
            "frame_indices": torch.arange(count, dtype=torch.long),
            "action_transform": transform,
        }
        path = root / relative
        save_tensor_shard(path, payload)
        shards[split].append(
            {
                "episode": episode,
                "path": relative.as_posix(),
                "samples": count,
                "sha256": file_sha256(path),
                "action_transform": transform,
            }
        )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "identity_hash": identity_hash,
        "identity": identity,
        "normalization_path": "normalization.json",
        "normalization_sha256": file_sha256(root / "normalization.json"),
        "shards": shards,
        "samples": {"train": 2, "validation": 1, "test": 1},
    }
    return create_json(root / "manifest.json", manifest)


def _target_context(
    normalization: dict[str, Any],
    *,
    pooling: str = cache_features.COMBINED_POOLING,
) -> dict[str, Any]:
    identity = _identity(
        experiment_id="target",
        pooling=pooling,
        normalization=normalization,
    )
    return {
        "experiment": {
            "experiment_id": "target",
            "backbone": {"pooling": pooling},
        },
        "identity": identity,
        "normalization": normalization,
        "anchors": {1: [0, 1], 2: [0], 3: [0]},
        "split_lookup": {1: "train", 2: "validation", 3: "test"},
        "contract": load_action_contract(
            REPOSITORY_ROOT / "configs" / "sim" / "aloha_insertion.yaml"
        ),
    }


def _visible_derivation_case(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Any, Path]:
    split = {
        "train": list(range(40)),
        "validation": list(range(40, 45)),
        "test": list(range(45, 50)),
    }
    source_path = tmp_path / "source" / "manifest.json"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("{}\n", encoding="utf-8")
    config_path = tmp_path / "target.yaml"
    config_path.write_text("target\n", encoding="utf-8")
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text("contract\n", encoding="utf-8")
    processor = {"prompt": "Act: {instruction}"}
    materialization = {
        "schema_version": 1,
        "type": "direct_visible_feature_cache_build_v1",
        "materialized_splits": ["train", "validation"],
        "withheld_splits": ["test"],
        "materialized_episodes": {
            "train": split["train"],
            "validation": split["validation"],
        },
        "withheld_episodes": {"test": split["test"]},
        "adapter_episodes": split["train"] + split["validation"],
        "hidden_test_loaded": False,
        "hidden_test_materialized": False,
    }
    identity = {
        "schema_version": 1,
        "experiment_id": "source-visible",
        "experiment_config_sha256": "source-config",
        "code": {"workspace_tree_sha256": "source-tree"},
        "dataset": {
            "repo_id": "dataset",
            "revision": "revision",
            "episodes": list(range(50)),
        },
        "split": split,
        "selection": {"frame_stride": 2, "action_chunk_length": 8},
        "model": {
            "identifier": "model",
            "adaptation": "frozen",
            "hidden_size": 2,
        },
        "processor": processor,
        "feature": {"pooling": "image_spatial_2x2", "layer": "final"},
        "normalization_sha256": "normalization",
        "action_contract_sha256": file_sha256(contract_path),
        "materialization": materialization,
    }
    records = {
        name: [
            {
                "episode": episode,
                "path": f"shards/{name}/episode-{episode:03d}.pt",
                "samples": 1,
                "sha256": f"sha-{name}-{episode}",
                "action_transform": {"type": "clip_to_rosetta_contract_v1"},
            }
            for episode in split[name]
        ]
        for name in ("train", "validation")
    }
    records["test"] = []
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "identity_hash": stable_hash(identity),
        "identity": identity,
        "normalization_path": "normalization.json",
        "normalization_sha256": "normalization",
        "shards": records,
        "samples": {"train": 40, "validation": 5, "test": 0},
        "hidden_test_loaded": False,
        "hidden_test_materialized": False,
        "materialized_splits": ["train", "validation"],
        "withheld_splits": ["test"],
    }
    source = {
        "path": source_path.resolve(),
        "root": source_path.parent,
        "manifest": manifest,
        "identity": identity,
        "pooling": "image_spatial_2x2",
        "normalization": {"source_split": "train", "statistics": {"marker": 1}},
    }
    experiment = {
        "experiment_id": "target-visible",
        "dataset": {
            "config": str(tmp_path / "dataset.yaml"),
            "frame_stride": 2,
            "split": split,
        },
        "backbone": {
            "identifier": "model",
            "adaptation": "frozen",
            "pooling": "image_spatial_2x2",
            "feature_layer": "final",
            "processor": processor,
        },
        "action_contract": str(contract_path),
        "controlled_change": {
            "changed_axis": "training.first_action_loss_weight",
            "reference_experiment": "source-visible",
            "feature_derivation": "verified_identity_rebind_v1",
            "feature_source_manifest": str(source_path.resolve()),
        },
    }
    dataset_config = SimpleNamespace(
        repo_id="dataset",
        revision="revision",
        episodes=tuple(range(50)),
        chunk_size=8,
    )
    return experiment, source, dataset_config, config_path


def test_feature_cache_composition_concatenates_verified_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normalization = {"source_split": "train", "statistics": {"marker": 1}}
    global_manifest = _source_cache(
        tmp_path / "global",
        experiment_id="global",
        pooling="attention_masked_mean",
        feature_dim=2,
        normalization=normalization,
    )
    spatial_manifest = _source_cache(
        tmp_path / "spatial",
        experiment_id="spatial",
        pooling="image_spatial_2x2",
        feature_dim=8,
        normalization=normalization,
    )
    context = _target_context(normalization)
    target_root = tmp_path / "target-cache"
    monkeypatch.setenv("ROSETTA_FEATURE_ROOT", str(target_root))
    monkeypatch.setattr(cache_features, "_context", lambda _: deepcopy(context))

    cache_features.compose(Path("unused.yaml"), [spatial_manifest, global_manifest])

    manifest_path = next(target_root.glob("target/*/manifest.json"))
    train = CachedFeatureDataset(manifest_path, "train")
    assert train.features.shape == (2, 10)
    torch.testing.assert_close(
        train.features[:, :2],
        torch.tensor([[0, 1], [2, 3]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        train.features[:, 2:],
        torch.arange(16, dtype=torch.float32).reshape(2, 8),
    )


def test_feature_cache_composition_rejects_nonfeature_tensor_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normalization = {"source_split": "train", "statistics": {"marker": 1}}
    global_manifest = _source_cache(
        tmp_path / "global",
        experiment_id="global",
        pooling="attention_masked_mean",
        feature_dim=2,
        normalization=normalization,
    )
    spatial_manifest = _source_cache(
        tmp_path / "spatial",
        experiment_id="spatial",
        pooling="image_spatial_2x2",
        feature_dim=8,
        normalization=normalization,
        state_offset=1.0,
    )
    context = _target_context(normalization)
    monkeypatch.setenv("ROSETTA_FEATURE_ROOT", str(tmp_path / "target-cache"))
    monkeypatch.setattr(cache_features, "_context", lambda _: deepcopy(context))

    with pytest.raises(ValueError, match="robot_state.*differs"):
        cache_features.compose(Path("unused.yaml"), [global_manifest, spatial_manifest])


def test_feature_cache_derivation_copies_exact_verified_tensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normalization = {"source_split": "train", "statistics": {"marker": 1}}
    source_manifest = _source_cache(
        tmp_path / "source",
        experiment_id="source",
        pooling="image_spatial_2x2",
        feature_dim=8,
        normalization=normalization,
    )
    context = _target_context(normalization, pooling="image_spatial_2x2")
    target_root = tmp_path / "target-cache"
    monkeypatch.setenv("ROSETTA_FEATURE_ROOT", str(target_root))
    monkeypatch.setattr(cache_features, "_context", lambda _: deepcopy(context))

    cache_features.derive(Path("unused.yaml"), [source_manifest])

    manifest_path = next(target_root.glob("target/*/manifest.json"))
    derived = CachedFeatureDataset(manifest_path, "train")
    source = CachedFeatureDataset(source_manifest, "train")
    assert torch.equal(derived.features, source.features)
    assert torch.equal(derived.robot_state, source.robot_state)
    assert torch.equal(derived.actions, source.actions)
    manifest = cache_features.load_feature_manifest(manifest_path)
    assert manifest["derivation"]["tensor_transform"] == "identity"


def test_visible_source_records_require_exact_45_episode_scope(tmp_path: Path) -> None:
    experiment, source, _, _ = _visible_derivation_case(tmp_path)

    records = cache_features._visible_source_shard_records(source, experiment)

    assert len(records) == 45
    assert set(records) == {
        (split, episode)
        for split in ("train", "validation")
        for episode in experiment["dataset"]["split"][split]
    }
    assert all(split != "test" for split, _ in records)


@pytest.mark.parametrize(
    ("changed_axis", "allowed"),
    [
        ("training.first_action_loss_weight", True),
        ("action_expert.fusion_dim.extra", False),
        ("action_expert.head_hidden_dim", False),
        ("backbone.pooling", False),
        ("dataset.frame_stride", False),
        ("resources.training_device", False),
    ],
)
def test_visible_derivation_allows_only_tensor_invariant_axis_families(
    changed_axis: str,
    allowed: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment, source, _, config_path = _visible_derivation_case(tmp_path)
    experiment["controlled_change"]["changed_axis"] = changed_axis
    monkeypatch.setattr(cache_features, "load_experiment_config", lambda *_: experiment)
    monkeypatch.setattr(cache_features, "_validated_source_manifest", lambda _: source)
    if allowed:
        monkeypatch.setattr(
            cache_features,
            "_visible_source_shard_records",
            lambda *_: pytest.fail("allowed axis passed the pre-I/O axis gate"),
        )
        with pytest.raises(pytest.fail.Exception, match="allowed axis"):
            cache_features.derive_visible(config_path, [source["path"]])
    else:
        monkeypatch.setattr(
            cache_features,
            "_visible_source_shard_records",
            lambda *_: pytest.fail("unsafe axis reached source shard discovery"),
        )
        with pytest.raises(ValueError, match="only supports training axes"):
            cache_features.derive_visible(config_path, [source["path"]])


def test_visible_derivation_accepts_exact_single_fusion_width_axis() -> None:
    target_path = (
        REPOSITORY_ROOT
        / "configs"
        / "experiments"
        / "m2_qwen08b_frozen_015_fusion512_xpu.yaml"
    )
    reference_path = (
        REPOSITORY_ROOT
        / "configs"
        / "experiments"
        / "m2_qwen08b_frozen_012_stride2_xpu.yaml"
    )
    target = cache_features.load_experiment_config(target_path, REPOSITORY_ROOT)
    source = {
        "identity": {
            "experiment_id": "m2-qwen08b-frozen-012-stride2-xpu",
            "experiment_config_sha256": cache_features.file_sha256(reference_path),
        }
    }

    controlled = cache_features._validated_visible_controlled_change(target, source)

    assert controlled["changed_axis"] == "action_expert.fusion_dim"
    assert controlled["reference_value"] == 256
    assert controlled["candidate_value"] == 512


def test_visible_derivation_rejects_second_axis_with_fusion_width() -> None:
    target_path = (
        REPOSITORY_ROOT
        / "configs"
        / "experiments"
        / "m2_qwen08b_frozen_015_fusion512_xpu.yaml"
    )
    reference_path = (
        REPOSITORY_ROOT
        / "configs"
        / "experiments"
        / "m2_qwen08b_frozen_012_stride2_xpu.yaml"
    )
    target = cache_features.load_experiment_config(target_path, REPOSITORY_ROOT)
    target["action_expert"]["head_hidden_dim"] = 512
    source = {
        "identity": {
            "experiment_id": "m2-qwen08b-frozen-012-stride2-xpu",
            "experiment_config_sha256": cache_features.file_sha256(reference_path),
        }
    }

    with pytest.raises(ValueError, match="another action_expert axis"):
        cache_features._validated_visible_controlled_change(target, source)


@pytest.mark.parametrize("forgery", ["legacy_test_shard", "identity_hidden_flag"])
def test_visible_derivation_rejects_hidden_source_before_target_write(
    forgery: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment, source, _, config_path = _visible_derivation_case(tmp_path)
    if forgery == "legacy_test_shard":
        source["manifest"]["shards"]["test"] = [
            {
                "episode": 45,
                "path": "shards/test/episode-045.pt",
                "samples": 1,
            }
        ]
        source["manifest"]["samples"]["test"] = 1
    else:
        source["identity"]["materialization"]["hidden_test_loaded"] = True
    target_root = tmp_path / "target-cache"
    writes: list[str] = []
    monkeypatch.setattr(cache_features, "load_experiment_config", lambda *_: experiment)
    monkeypatch.setattr(cache_features, "_validated_source_manifest", lambda _: source)
    monkeypatch.setattr(cache_features, "_feature_root", lambda: target_root)
    monkeypatch.setattr(
        cache_features,
        "load_dataset_config",
        lambda *_: pytest.fail("hidden source reached dataset config loading"),
    )
    monkeypatch.setattr(
        cache_features,
        "create_json",
        lambda *_args, **_kwargs: writes.append("json"),
    )
    monkeypatch.setattr(
        cache_features,
        "save_tensor_shard",
        lambda *_args, **_kwargs: writes.append("tensor"),
    )

    with pytest.raises(ValueError, match="hidden-test"):
        cache_features.derive_visible(config_path, [source["path"]])

    assert writes == []
    assert not target_root.exists()


def test_visible_derivation_rejects_unmanifested_test_file(tmp_path: Path) -> None:
    experiment, source, _, _ = _visible_derivation_case(tmp_path)
    hidden_path = source["root"] / "shards" / "test" / "episode-045.pt"
    hidden_path.parent.mkdir(parents=True)
    hidden_path.write_bytes(b"must not be opened")

    with pytest.raises(ValueError, match="unmanifested hidden-test files"):
        cache_features._visible_source_shard_records(source, experiment)


def test_visible_derivation_preflights_all_source_shards_before_target_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment, source, dataset_config, config_path = _visible_derivation_case(tmp_path)
    target_root = tmp_path / "target-cache"
    calls: list[tuple[str, int]] = []
    writes: list[str] = []
    monkeypatch.setattr(cache_features, "load_experiment_config", lambda *_: experiment)
    monkeypatch.setattr(cache_features, "_validated_source_manifest", lambda _: source)
    monkeypatch.setattr(cache_features, "load_dataset_config", lambda *_: dataset_config)
    monkeypatch.setattr(cache_features, "load_action_contract", lambda *_: object())
    monkeypatch.setattr(cache_features, "_feature_root", lambda: target_root)

    def fail_last_source(
        _source: dict[str, Any],
        _record: dict[str, Any],
        *,
        split: str,
        episode: int,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        calls.append((split, episode))
        if len(calls) == 45:
            raise ValueError("last source shard is corrupt")
        return {}

    monkeypatch.setattr(cache_features, "_load_composition_shard", fail_last_source)
    monkeypatch.setattr(
        cache_features,
        "create_json",
        lambda *_args, **_kwargs: writes.append("json"),
    )
    monkeypatch.setattr(
        cache_features,
        "save_tensor_shard",
        lambda *_args, **_kwargs: writes.append("tensor"),
    )

    with pytest.raises(ValueError, match="last source shard is corrupt"):
        cache_features.derive_visible(config_path, [source["path"]])

    assert len(calls) == 45
    assert all(split != "test" for split, _ in calls)
    assert writes == []
    assert not target_root.exists()


def test_visible_derivation_emits_only_train_validation_identity_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment, source, dataset_config, config_path = _visible_derivation_case(tmp_path)
    target_root = tmp_path / "target-cache"
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(cache_features, "load_experiment_config", lambda *_: experiment)
    monkeypatch.setattr(cache_features, "_validated_source_manifest", lambda _: source)
    monkeypatch.setattr(cache_features, "load_dataset_config", lambda *_: dataset_config)
    monkeypatch.setattr(cache_features, "load_action_contract", lambda *_: object())
    monkeypatch.setattr(cache_features, "_feature_root", lambda: target_root)
    monkeypatch.setattr(
        cache_features,
        "workspace_code_identity",
        lambda *_: {"workspace_tree_sha256": "target-tree"},
    )

    def fake_source_shard(
        _source: dict[str, Any],
        record: dict[str, Any],
        *,
        split: str,
        episode: int,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        calls.append((split, episode))
        return {
            "features": torch.full((1, 8), float(episode), dtype=torch.float16),
            "robot_state": torch.zeros(1, 14),
            "actions": torch.zeros(1, 8, 14),
            "episode_ids": torch.tensor([episode]),
            "frame_indices": torch.tensor([episode]),
            "action_transform": record["action_transform"],
        }

    monkeypatch.setattr(cache_features, "_load_composition_shard", fake_source_shard)

    assert cache_features.derive_visible(config_path, [source["path"]]) == 0

    manifests = list((target_root / "target-visible").glob("*/manifest.json"))
    assert len(manifests) == 1
    manifest = cache_features.load_feature_manifest(manifests[0])
    assert len(calls) == 45
    assert manifest["samples"] == {"train": 40, "validation": 5, "test": 0}
    assert manifest["shards"]["test"] == []
    assert manifest["hidden_test_loaded"] is False
    assert manifest["hidden_test_materialized"] is False
    assert manifest["derivation"]["tensor_transform"] == "identity"
    assert manifest["derivation"]["hidden_test_loaded"] is False
    assert manifest["derivation"]["hidden_test_materialized"] is False
    assert manifest["derivation"]["source"]["identity_hash"] == source["manifest"][
        "identity_hash"
    ]


@pytest.mark.parametrize(
    "entrypoint",
    [cache_features.build_visible, cache_features.smoke_visible],
)
def test_visible_cache_rejects_test_overlap_before_payload_io(
    entrypoint: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = {
        "dataset": {
            "split": {
                "train": [1],
                "validation": [2],
                "test": [1],
            }
        }
    }
    downstream_calls: list[str] = []
    monkeypatch.setattr(
        cache_features,
        "load_experiment_config",
        lambda *_: experiment,
    )

    def record(name: str) -> Any:
        def fake(*_args: Any, **_kwargs: Any) -> None:
            downstream_calls.append(name)

        return fake

    for name in (
        "load_dataset_config",
        "resolve_prepared_cache",
        "_model_identity",
        "LeRobotV3Adapter",
        "_anchors_by_episode",
        "_backbone",
        "save_tensor_shard",
    ):
        monkeypatch.setattr(cache_features, name, record(name))

    with pytest.raises(ValueError, match="scopes overlap"):
        entrypoint(Path("unused.yaml"))

    assert downstream_calls == []


@pytest.mark.parametrize(
    ("visible_only", "expected_validate_checksums"),
    [(True, False), (False, True)],
)
def test_visible_context_does_not_scan_prepared_payload_checksums(
    visible_only: bool,
    expected_validate_checksums: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = {
        "dataset": {
            "config": "unused-dataset.yaml",
            "split": {
                "train": [1],
                "validation": [2],
                "test": [3],
            },
        },
        "action_contract": "unused-contract.yaml",
    }
    dataset_config = object()
    checksum_modes: list[bool] = []
    monkeypatch.setattr(cache_features, "load_experiment_config", lambda *_: experiment)
    monkeypatch.setattr(cache_features, "load_dataset_config", lambda *_: dataset_config)

    def fake_resolve(
        actual_config: Any,
        _root: Path,
        *,
        validate_checksums: bool,
    ) -> tuple[Path, object]:
        assert actual_config is dataset_config
        checksum_modes.append(validate_checksums)
        return Path("unused-cache"), object()

    monkeypatch.setattr(cache_features, "resolve_prepared_cache", fake_resolve)
    monkeypatch.setattr(
        cache_features,
        "load_action_contract",
        lambda *_: (_ for _ in ()).throw(RuntimeError("stop after cache resolution")),
    )

    with pytest.raises(RuntimeError, match="stop after cache resolution"):
        cache_features._context(Path("unused.yaml"), visible_only=visible_only)

    assert checksum_modes == [expected_validate_checksums]


def test_visible_cache_rejects_hidden_anchor_before_any_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialization = cache_features._visible_materialization_scope(
        {
            "dataset": {
                "split": {
                    "train": [1],
                    "validation": [2],
                    "test": [3],
                }
            }
        }
    )
    cache_root = tmp_path / "must-not-exist"
    writes: list[str] = []
    monkeypatch.setattr(
        cache_features,
        "create_json",
        lambda *_args, **_kwargs: writes.append("json"),
    )
    monkeypatch.setattr(
        cache_features,
        "save_tensor_shard",
        lambda *_args, **_kwargs: writes.append("tensor"),
    )
    context = {
        "materialization": materialization,
        "anchors": {1: [0], 2: [1], 3: [2]},
        "split_lookup": {1: "train", 2: "validation", 3: "test"},
        "cache_root": cache_root,
    }

    with pytest.raises(ValueError, match="before writes"):
        cache_features._build_cache(context)

    assert writes == []
    assert not cache_root.exists()


def test_visible_cache_manifest_withholds_test_and_preserves_stride_identity() -> None:
    materialization = cache_features._visible_materialization_scope(
        {
            "dataset": {
                "split": {
                    "train": [3, 1],
                    "validation": [4],
                    "test": [2],
                }
            }
        }
    )
    identity = {
        "schema_version": 1,
        "selection": {"frame_stride": 2},
        "materialization": materialization,
    }
    context = {
        "identity_hash": stable_hash(identity),
        "identity": identity,
        "materialization": materialization,
    }
    records = {
        "train": [
            {"episode": 1, "samples": 2},
            {"episode": 3, "samples": 3},
        ],
        "validation": [{"episode": 4, "samples": 1}],
        "test": [],
    }

    manifest = cache_features._feature_manifest_payload(
        context,
        records,
        normalization_sha256="normalization-sha256",
    )

    assert manifest["status"] == "complete"
    assert manifest["materialized_splits"] == ["train", "validation"]
    assert manifest["withheld_splits"] == ["test"]
    assert manifest["hidden_test_loaded"] is False
    assert manifest["hidden_test_materialized"] is False
    assert manifest["shards"]["test"] == []
    assert manifest["samples"] == {"train": 5, "validation": 1, "test": 0}
    assert manifest["identity"]["selection"]["frame_stride"] == 2
    assert manifest["identity"]["materialization"] == materialization

    records_with_test = deepcopy(records)
    records_with_test["test"] = [{"episode": 2, "samples": 1}]
    with pytest.raises(ValueError, match="hidden-test shards"):
        cache_features._feature_manifest_payload(
            context,
            records_with_test,
            normalization_sha256="normalization-sha256",
        )


def test_smoke_visible_uses_one_first_train_anchor_without_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = SimpleNamespace(images={"camera": object()}, instruction="insert")

    class FakeChunked:
        def __init__(self) -> None:
            self.accesses: list[int] = []

        def __getitem__(self, index: int) -> Any:
            self.accesses.append(index)
            if index != 42:
                raise AssertionError("smoke-visible accessed more than the first train anchor")
            return sample

    class FakeBackbone:
        hidden_size = 2

        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def __call__(self, value: dict[str, Any]) -> torch.Tensor:
            self.calls.append(value)
            return torch.ones(1, self.hidden_size)

        def parameters(self) -> tuple[()]:
            return ()

    chunked = FakeChunked()
    backbone = FakeBackbone()
    context = {
        "experiment": {"dataset": {"split": {"train": [7]}}},
        "chunked": chunked,
        "anchors": {7: [42, 43], 8: [99]},
        "identity_hash": "visible-smoke-identity",
    }

    def fake_context(_path: Path, *, visible_only: bool = False) -> dict[str, Any]:
        assert visible_only is True
        return context

    monkeypatch.setattr(cache_features, "_context", fake_context)
    monkeypatch.setattr(cache_features, "_backbone", lambda _: backbone)
    monkeypatch.setattr(
        cache_features,
        "create_json",
        lambda *_args, **_kwargs: pytest.fail("smoke-visible wrote JSON"),
    )
    monkeypatch.setattr(
        cache_features,
        "save_tensor_shard",
        lambda *_args, **_kwargs: pytest.fail("smoke-visible wrote a tensor"),
    )

    assert cache_features.smoke_visible(Path("unused.yaml")) == 0
    assert chunked.accesses == [42]
    assert len(backbone.calls) == 1
    assert backbone.calls[0] == {
        "images": sample.images,
        "instruction": sample.instruction,
    }
