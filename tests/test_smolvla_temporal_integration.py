"""Sealed feature -> native DataLoader/Accelerate -> update observer integration."""

import json
from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator
from lerobot.scripts import lerobot_train
from torch.utils.data import DataLoader, Dataset

from rosetta_reality.experiment import file_sha256
from rosetta_reality.vla.training import temporal_sampler
from rosetta_reality.vla.training.features import FeatureStack
from rosetta_reality.vla.training.observed_launch import run_observed_launch


class TemporalDataset(Dataset):
    def __len__(self):
        return 20

    def __getitem__(self, index):
        episode, frame = (0 if index < 10 else 2), index % 10
        slots = torch.arange(3) + frame
        return {
            "episode_index": episode,
            "frame_index": frame,
            "action": (episode * 100 + slots.clamp(max=9)).float().unsqueeze(-1),
            "action_is_pad": slots >= 10,
        }


def context_and_declaration(tmp_path, monkeypatch, samples, batch_size, **overrides):
    monkeypatch.setattr(
        temporal_sampler,
        "__file__",
        str(tmp_path / "src/rosetta_reality/vla/training/temporal_sampler.py"),
    )
    path = tmp_path / "schedule.json"
    path.write_text(
        json.dumps(
            {
                "status": "preregistered",
                "training_authorized": True,
                "seed": 17,
                "sample_identities": samples,
                **overrides,
            }
        )
    )
    declaration = {
        "name": "explicit_sample_schedule",
        "path": path.name,
        "sha256": file_sha256(path),
    }
    context = SimpleNamespace(
        phase="smoke",
        plan={
            "scope": "bounded_temporal_sampling",
            "optimizer_smoke": {
                "batch_size": batch_size,
                "steps": len(samples) // batch_size,
                "episodes": [2, 0],
            },
        },
        experiment={
            "seed": 17,
            "dataset": {"train_episodes": [0, 2], "validation_episodes": [1], "test_episodes": [3]},
        },
    )
    return context, declaration


def test_draft_temporal_schedule_cannot_be_installed(tmp_path, monkeypatch):
    context, declaration = context_and_declaration(
        tmp_path,
        monkeypatch,
        [[2, 0], [0, 9]],
        1,
        status="draft_not_launchable",
        training_authorized=False,
    )
    original = lerobot_train.EpisodeAwareSampler
    stack = FeatureStack.from_plan({"features": [declaration]})
    try:
        with pytest.raises(ValueError, match="preregistered|draft|authorized"):
            stack.install_all(context)
    finally:
        if stack.installed:
            stack.restore_all(context)
    assert lerobot_train.EpisodeAwareSampler is original


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("workers", [0, 2])
def test_temporal_frames_reach_native_updates_with_tail_masks(
    tmp_path,
    monkeypatch,
    batch_size,
    workers,
):
    samples = [[2, 0], [0, 9], [2, 8], [0, 4], [2, 9], [0, 0], [2, 5], [0, 8]]
    context, declaration = context_and_declaration(tmp_path, monkeypatch, samples, batch_size)
    original = lerobot_train.EpisodeAwareSampler
    stack = FeatureStack.from_plan({"features": [declaration]})
    seen = []

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def forward(self, batch):
            for ep, frame, action, pad in zip(
                batch["episode_index"],
                batch["frame_index"],
                batch["action"],
                batch["action_is_pad"],
                strict=True,
            ):
                ep, frame = int(ep), int(frame)
                slots = torch.arange(3) + frame
                assert torch.equal(pad, slots >= 10)
                assert torch.equal(action[:, 0], ep * 100 + slots.clamp(max=9))
                seen.append([ep, frame])
            valid = ~batch["action_is_pad"]
            return ((self.weight - batch["action"].squeeze(-1) / 210).square()[valid]).mean(), {}

    def launch():
        stack.install_all(context)
        try:
            mapping = {
                absolute: relative for relative, absolute in enumerate([*range(10), *range(20, 30)])
            }
            sampler = lerobot_train.EpisodeAwareSampler(
                [0, 10, 20],
                [10, 20, 30],
                [2, 0],
                shuffle=True,
                seed=17,
                absolute_to_relative_idx=mapping,
            )
            loader = DataLoader(
                TemporalDataset(),
                sampler=sampler,
                batch_size=batch_size,
                num_workers=workers,
                drop_last=False,
            )
            accelerator = Accelerator(cpu=True, gradient_accumulation_steps=1)
            policy = Policy()
            optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
            policy, optimizer, loader = accelerator.prepare(policy, optimizer, loader)
            iterator = lerobot_train.cycle(loader)
            for _ in range(len(samples) // batch_size):
                lerobot_train.update_policy(
                    SimpleNamespace(), policy, next(iterator), optimizer, 10, accelerator
                )
            iterator.close()
            return 0
        finally:
            stack.restore_all(context)

    output = tmp_path / "observed"
    run_observed_launch(
        lerobot_train, launch, expected_samples=samples, batch_size=batch_size, output=output
    )
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "passed"
    assert result["observation"]["unique_completed_input_frames"] == 8
    assert result["observation"]["completed_samples"] == 8
    assert seen == samples
    assert lerobot_train.EpisodeAwareSampler is original
