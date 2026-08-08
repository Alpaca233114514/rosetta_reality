"""Explicit offline smoke test over the prepared ALOHA episode."""

import math
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from rosetta_reality.data import ActionChunkDataset, collate_rosetta
from rosetta_reality.data.adapters import LeRobotV3Adapter
from rosetta_reality.data.config import load_dataset_config
from rosetta_reality.data.manifest import find_dataset_manifests, load_dataset_manifest
from rosetta_reality.data.normalization import load_dataset_statistics, normalize
from rosetta_reality.models import ContinuousActionHead, StateEncoder, VLAPolicy
from rosetta_reality.models.backbones import DummyBackbone
from rosetta_reality.train import train_step

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "data" / "aloha_sim_insertion.yaml"


@pytest.mark.data
def test_prepared_episode_supports_one_cpu_optimizer_step(monkeypatch) -> None:
    config = load_dataset_config(CONFIG_PATH)
    cache_root = (
        config.cache_root
        if config.cache_root.is_absolute()
        else REPOSITORY_ROOT / config.cache_root
    )
    manifests = find_dataset_manifests(cache_root, config.repo_id)
    expected_fields = asdict(config.fields)
    complete = [
        (path, manifest)
        for path in manifests
        if (manifest := load_dataset_manifest(path)).fields == expected_fields
        and (path.parent / "meta" / "info.json").is_file()
        and (path.parent / "statistics.json").is_file()
    ]
    if not complete:
        pytest.skip(
            "Run 'python scripts/prepare_data.py' in WSL to prepare the current field mapping."
        )
    manifest_path, manifest = complete[-1]
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    adapter = LeRobotV3Adapter(
        repo_id=config.repo_id,
        revision=manifest.resolved_revision,
        root=manifest_path.parent,
        episodes=config.episodes,
        cameras=config.cameras,
        fields=config.fields,
        embodiment=config.embodiment,
        license_name=config.license,
    )
    dataset = ActionChunkDataset(adapter, config.chunk_size)
    batch = next(
        iter(DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_rosetta))
    )

    assert len(adapter) == 500
    assert batch.instructions[0] == "Insert the peg into the socket."
    assert batch.robot_state.shape == (2, 14)
    assert batch.actions.shape == (2, 8, 14)
    assert batch.images["top"].shape == (2, 3, 480, 640)

    statistics = load_dataset_statistics(manifest_path.parent / "statistics.json")
    robot_state = normalize(batch.robot_state, statistics.state)
    target_actions = normalize(batch.actions, statistics.action)
    dummy_features = batch.images["top"].mean(dim=(-2, -1))
    policy = VLAPolicy(
        backbone=DummyBackbone(input_dim=dummy_features.shape[-1], hidden_size=32),
        state_encoder=StateEncoder(state_dim=robot_state.shape[-1], hidden_dim=32),
        action_head=ContinuousActionHead(
            input_dim=32,
            action_dim=target_actions.shape[-1],
            chunk_size=target_actions.shape[-2],
        ),
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    result = train_step(
        policy,
        optimizer,
        {"features": dummy_features},
        robot_state,
        target_actions,
    )

    assert result.prediction_shape == (2, 8, 14)
    assert math.isfinite(result.loss)
