"""CPU native trainer integration with synthetic tensors; no pretrained policy."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator
from lerobot.scripts import lerobot_train
from test_smolvla_temporal_integration import TemporalDataset, context_and_declaration
from torch.utils.data import DataLoader

from rosetta_reality.vla.training.features import FeatureStack
from rosetta_reality.vla.training.observation import TrainingObservation, observe_native_training
from rosetta_reality.vla.training.trajectory_probe import (
    PINNED_NATIVE_LOSS_SHA256,
    observe_effective_runtime,
)


@pytest.mark.parametrize("drift", [False, True])
def test_actual_native_factory_accelerator_and_update_contract(tmp_path, monkeypatch, drift):
    samples = [[2, 0], [0, 9], [2, 0], [0, 9]]
    context, declaration = context_and_declaration(tmp_path, monkeypatch, samples, 2)
    stack = FeatureStack.from_plan({"features": [declaration]})
    ledger = TrainingObservation(samples, 2)
    source = Path(lerobot_train.__file__).parents[1] / "policies/smolvla/modeling_smolvla.py"

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def forward(self, batch):
            valid = ~batch["action_is_pad"]
            return (self.weight - batch["action"].squeeze(-1) / 210).square()[valid].mean(), {}

    policy = Policy()
    config = SimpleNamespace(
        steps=2,
        save_freq=2,
        use_policy_training_preset=False,
        optimizer=SimpleNamespace(
            build=lambda params: torch.optim.AdamW(
                params,
                lr=0.001,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=0.0,
            )
        ),
        scheduler=SimpleNamespace(
            build=lambda optimizer, steps: torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda step: 1.0 - step / (steps + 1),
            )
        ),
    )
    expected = {
        "trainable_parameters": ["weight"],
        "optimizer_groups": [
            {
                "parameters": ["weight"],
                "lr": 0.001,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.0,
            }
        ],
        "features": ["explicit_sample_schedule"],
        "actual_lr": [0.001],
        "save_grid": [2],
        "loss_source_path": source,
        "loss_source_sha256": PINNED_NATIVE_LOSS_SHA256,
        "loss_mask": {
            "reduction": "global_valid_action_entries",
            "upstream_source_sha256": PINNED_NATIVE_LOSS_SHA256,
            "valid_steps_per_sample": [3, 1],
            "denominator": 4,
            "action_dimension": 1,
        },
    }
    if drift:
        expected["loss_mask"]["denominator"] = 6
    original = lerobot_train.update_policy, lerobot_train.make_optimizer_and_scheduler
    output = tmp_path / "effective-runtime-contract.json"

    def execute():
        with observe_native_training(lerobot_train, ledger):
            stack.install_all(context)
            try:
                with observe_effective_runtime(
                    lerobot_train,
                    expected=expected,
                    ledger=ledger,
                    output=output,
                    stack=stack,
                ):
                    mapping = {
                        absolute: relative
                        for relative, absolute in enumerate([*range(10), *range(20, 30)])
                    }
                    sampler = lerobot_train.EpisodeAwareSampler(
                        [0, 10, 20],
                        [10, 20, 30],
                        [2, 0],
                        shuffle=True,
                        seed=17,
                        absolute_to_relative_idx=mapping,
                    )
                    loader = DataLoader(TemporalDataset(), sampler=sampler, batch_size=2)
                    optimizer, scheduler = lerobot_train.make_optimizer_and_scheduler(
                        config, policy
                    )
                    accelerator = Accelerator(cpu=True)
                    active, optimizer, loader = accelerator.prepare(policy, optimizer, loader)
                    iterator = lerobot_train.cycle(loader)
                    try:
                        for _ in range(2):
                            lerobot_train.update_policy(
                                SimpleNamespace(),
                                active,
                                next(iterator),
                                optimizer,
                                10,
                                accelerator,
                                scheduler,
                            )
                    finally:
                        iterator.close()
            finally:
                stack.restore_all(context)

    if drift:
        with pytest.raises(ValueError, match="denominator"):
            execute()
        assert not output.exists()
        assert ledger.snapshot()["successful_optimizer_step_calls"] == 0
        assert policy.weight.item() == 0
    else:
        execute()
        result = json.loads(output.read_text())
        assert result["feature_evidence_scope"] == "live_feature_stack"
        assert result["loss_mask"]["denominator"] == 4
        assert result["ledger"]["completed_updates"] == 0
        final = ledger.snapshot()
        assert final["status"] == "complete"
        assert final["completed_samples"] == 4
        assert final["unique_completed_input_frames"] == 2
        assert final["successful_optimizer_step_calls"] == 2
        assert policy.weight.item() != 0
    assert (lerobot_train.update_policy, lerobot_train.make_optimizer_and_scheduler) == original
