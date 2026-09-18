"""Opt-in, read-only real-cache checks for the registered temporal diagnostic."""

from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.mark.data
def test_registered_train_frames_decode_with_matching_state_action_and_task():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from rosetta_reality.data import resolve_prepared_cache
    from rosetta_reality.data.config import load_dataset_config
    from rosetta_reality.vla.action_space import load_smolvla_experiment
    from rosetta_reality.vla.visual_grounding import TRAIN_EPISODES, read_temporal_rows

    repository = Path(__file__).resolve().parents[1]
    cfg = load_dataset_config(repository / "configs/data/aloha_sim_insertion_m2.yaml")
    experiment = load_smolvla_experiment(
        repository
        / "configs/vla/smolvla_450m_aloha_insertion_action_repair_bounded_gripper_003.yaml",
        repository,
    )
    root, _ = resolve_prepared_cache(cfg, repository, validate_checksums=True)
    rows = read_temporal_rows(
        root, cfg, experiment["dataset"]["train_episodes"], experiment["dataset"]["test_episodes"]
    )
    dataset = LeRobotDataset(
        cfg.repo_id,
        root=root,
        episodes=list(TRAIN_EPISODES),
        revision=cfg.revision,
        download_videos=False,
        return_uint8=True,
    )
    starts = dict(
        zip(
            dataset.meta.episodes["episode_index"],
            dataset.meta.episodes["dataset_from_index"],
            strict=True,
        )
    )
    for (episode, offset), row in rows.items():
        sample = dataset[dataset.absolute_to_relative_idx[int(starts[episode]) + offset]]
        assert int(sample[cfg.fields.episode_index]) == episode
        assert int(sample[cfg.fields.frame_index]) == offset
        np.testing.assert_array_equal(sample[cfg.fields.state].numpy(), row[cfg.fields.state])
        np.testing.assert_array_equal(sample[cfg.fields.action].numpy(), row[cfg.fields.action])
        image = sample[cfg.cameras["top"]]
        assert image.ndim == 3 and image.shape[0] == 3 and image.dtype == torch.uint8
        assert image.max() > image.min()
        assert sample["task"] == cfg.expected_instruction
