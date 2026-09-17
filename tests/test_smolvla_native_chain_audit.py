"""Local CPU integration: native update/sampler/AdamW/scheduler, no SmolVLA weights.

This proves only the unaugmented tiny-policy path, not formal model resume.
"""

import copy
import itertools
from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator
from lerobot.datasets.sampler import EpisodeAwareSampler, compute_sampler_state
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.scripts.lerobot_train import update_policy


@pytest.mark.parametrize("batch_size", [1, 4])
def test_native_update_stop_restore_matches_continuous(tmp_path, batch_size):
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(2, 2)

        def forward(self, batch):
            prediction = self.layer(batch) + torch.randn_like(batch) * 0.01
            return prediction.square().mean(), {}

    def create():
        accelerator = Accelerator(cpu=True, gradient_accumulation_steps=1)
        policy = Policy()
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4, betas=(0.9, 0.95))
        scheduler = CosineDecayWithWarmupSchedulerConfig(2, 8, 1e-4, 2.5e-6).build(optimizer, 8)
        policy, optimizer, scheduler = accelerator.prepare(policy, optimizer, scheduler)
        sampler = EpisodeAwareSampler([0], [16], [0], shuffle=True, seed=17)
        return accelerator, policy, optimizer, scheduler, sampler

    def batches(sampler):
        while True:
            yield from iter(sampler)

    def run(interrupt):
        torch.manual_seed(123)
        accelerator, policy, optimizer, scheduler, sampler = create()
        iterator = batches(sampler)
        traces = []
        for step in range(8):
            indices = list(itertools.islice(iterator, batch_size))
            value = torch.tensor([[x / 16, 1.0] for x in indices])
            lr_before = optimizer.param_groups[0]["lr"]
            metrics, _ = update_policy(
                SimpleNamespace(), policy, value, optimizer, 10, accelerator, lr_scheduler=scheduler
            )
            traces.append((indices, metrics.loss, metrics.grad_norm, lr_before))
            assert all(p.grad is None or not bool(p.grad.any()) for p in policy.parameters())
            if interrupt and step == 3:
                checkpoint = {
                    "policy": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "rng": torch.get_rng_state(),
                    "sampler": compute_sampler_state(step + 1, 16, batch_size, 1),
                }
                path = tmp_path / f"tiny-{batch_size}.pt"
                torch.save(checkpoint, path)
                loaded = torch.load(path, weights_only=True)
                accelerator, policy, optimizer, scheduler, sampler = create()
                policy.load_state_dict(loaded["policy"])
                optimizer.load_state_dict(loaded["optimizer"])
                scheduler.load_state_dict(loaded["scheduler"])
                sampler.load_state_dict(loaded["sampler"])
                torch.set_rng_state(loaded["rng"])
                iterator = batches(sampler)
        return (
            copy.deepcopy(policy.state_dict()),
            copy.deepcopy(optimizer.state_dict()),
            traces,
            torch.get_rng_state(),
        )

    expected, actual = run(False), run(True)
    assert expected[2] == actual[2]
    assert torch.equal(expected[3], actual[3])
    assert all(torch.equal(expected[0][k], actual[0][k]) for k in expected[0])
    for key, state in expected[1]["state"].items():
        assert all(
            torch.equal(value, actual[1]["state"][key][name]) for name, value in state.items()
        )


def test_pinned_scheduler_is_nominal_scale_not_true_warmup_peak():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    scheduler = CosineDecayWithWarmupSchedulerConfig(16, 1280, 1e-4, 2.5e-6).build(optimizer, 1280)
    observed = []
    for _ in range(1280):
        observed.append(optimizer.param_groups[0]["lr"])
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
    assert max(observed) < 1e-4
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2.5e-6)
