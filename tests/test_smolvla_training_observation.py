"""Exercise real CPU optimizer equivalence and incomplete-update accounting."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from rosetta_reality.vla.training.observation import (
    TrainingObservation,
    observe_native_training,
)


def cycle(loader):
    while True:
        yield from loader


class Optimizer:
    def __init__(self):
        self.param_groups = [{"lr": 1e-4}]
        self.calls = 0
        self.step_was_skipped = False

    def step(self):
        self.calls += 1


def update(_metrics, _policy, _batch, optimizer):
    optimizer.step()
    return "native-result"


def batch(episodes=(3, 5)):
    return {"episode_index": list(episodes), "frame_index": [0, 0]}


def test_real_adamw_and_scheduler_preserve_weights_rng_and_outputs():
    import torch

    def run(observed):
        torch.manual_seed(71)
        parameter = torch.nn.Parameter(torch.tensor([0.5]))
        optimizer = torch.optim.AdamW([parameter], lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda index: 1 / (index + 1))
        original_step = optimizer.step

        def native(_metrics, _policy, _batch, optimizer):
            loss = (parameter * torch.randn(1)).square().sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            return float(loss.detach())

        module = SimpleNamespace(cycle=cycle, update_policy=native)
        ledger = TrainingObservation([(3, 0), (5, 0)] * 2, 2)
        scope = observe_native_training(module, ledger) if observed else nullcontext()
        with scope:
            iterator = module.cycle([batch()])
            outputs = [
                module.update_policy(None, None, next(iterator), optimizer) for _ in range(2)
            ]
        assert optimizer.step is original_step
        assert module.update_policy is native and module.cycle is cycle
        return parameter.detach().clone(), torch.get_rng_state(), outputs, ledger.snapshot()

    plain, observed = run(False), run(True)
    assert torch.equal(plain[0], observed[0]) and torch.equal(plain[1], observed[1])
    assert plain[2] == observed[2]
    assert observed[3]["status"] == "complete"
    assert observed[3]["completed_updates"] == 2
    assert observed[3]["learning_rates"] == [[1e-4], [5e-5]]


@pytest.mark.parametrize("bad", [(5, 3), (3, 3), (3, 7)])
def test_wrong_sample_order_is_rejected_before_delivery(bad):
    module = SimpleNamespace(cycle=cycle, update_policy=update)
    ledger = TrainingObservation([(3, 0), (5, 0)], 2)
    optimizer = Optimizer()
    with pytest.raises(ValueError, match="registered order"):
        with observe_native_training(module, ledger):
            next(module.cycle([batch(bad)]))
    assert optimizer.calls == 0 and ledger.snapshot()["completed_samples"] == 0


def test_preprocessor_failure_is_not_counted_as_a_completed_update():
    module = SimpleNamespace(cycle=cycle, update_policy=update)
    ledger = TrainingObservation([(3, 0), (5, 0)], 2)
    with pytest.raises(ValueError, match="original preprocessor failure"):
        with observe_native_training(module, ledger):
            next(module.cycle([batch()]))
            raise ValueError("original preprocessor failure")
    result = ledger.snapshot()
    assert result["delivered_samples"] == 2 and result["completed_samples"] == 0
    assert result["successful_optimizer_step_calls"] == 0


@pytest.mark.parametrize("failure", ["before_step", "after_step", "skip", "twice", "no_step"])
def test_update_failure_keeps_actual_step_count_and_original_error(failure):
    optimizer = Optimizer()

    def native(_metrics, _policy, _batch, optimizer):
        if failure == "before_step":
            raise ValueError("original update failure")
        if failure == "no_step":
            return None
        optimizer.step_was_skipped = failure == "skip"
        optimizer.step()
        if failure == "after_step":
            raise ValueError("original update failure")
        if failure == "twice":
            optimizer.step()

    module = SimpleNamespace(cycle=cycle, update_policy=native)
    ledger = TrainingObservation([(3, 0), (5, 0)], 2)
    expected_exception = ValueError if failure in {"before_step", "after_step"} else RuntimeError
    with pytest.raises(expected_exception):
        with observe_native_training(module, ledger):
            delivered = next(module.cycle([batch()]))
            module.update_policy(None, None, delivered, optimizer)
    result = ledger.snapshot()
    assert result["status"] == "incomplete" and result["completed_updates"] == 0
    assert result["successful_optimizer_step_calls"] == int(failure in {"after_step", "twice"})
    assert "step" not in vars(optimizer)
    assert module.update_policy is native and module.cycle is cycle


def test_extra_batch_and_update_without_delivery_are_rejected():
    module = SimpleNamespace(cycle=cycle, update_policy=update)
    ledger = TrainingObservation([(3, 0), (5, 0)], 2)
    with pytest.raises(RuntimeError, match="delivered batch"):
        with observe_native_training(module, ledger):
            module.update_policy(None, None, batch(), Optimizer())
    ledger = TrainingObservation([(3, 0), (5, 0)], 2)
    with pytest.raises(ValueError, match="registered order"):
        with observe_native_training(module, ledger):
            iterator = module.cycle([batch()])
            module.update_policy(None, None, next(iterator), Optimizer())
            next(iterator)
    assert ledger.snapshot()["status"] == "incomplete"


def test_outer_feature_wrapper_can_restore_without_losing_observation():
    module = SimpleNamespace(cycle=cycle, update_policy=update)
    ledger = TrainingObservation([(3, 0), (5, 0)], 2)
    with observe_native_training(module, ledger):
        observed = module.update_policy

        def feature(*args, **kwargs):
            return observed(*args, **kwargs)

        module.update_policy = feature
        assert module.update_policy(None, None, next(module.cycle([batch()])), Optimizer()) == (
            "native-result"
        )
        module.update_policy = observed
    assert module.update_policy is update and ledger.snapshot()["status"] == "complete"


def test_caught_extra_delivery_still_invalidates_a_complete_run():
    module = SimpleNamespace(cycle=cycle, update_policy=update)
    ledger = TrainingObservation([(3, 0), (5, 0)], 2)
    with observe_native_training(module, ledger):
        iterator = module.cycle([batch()])
        module.update_policy(None, None, next(iterator), Optimizer())
        with pytest.raises(ValueError):
            next(iterator)
    assert ledger.snapshot()["status"] == "incomplete"


@pytest.mark.parametrize("frames", [[1, 0], [0, -1], [0], [0.0, 0], 0])
def test_wrong_frame_or_malformed_identity_is_rejected(frames):
    ledger = TrainingObservation([(3, 0), (5, 0)], 2)
    delivered = batch()
    delivered["frame_index"] = frames
    with pytest.raises(ValueError):
        ledger.deliver(delivered)
    assert ledger.snapshot()["status"] == "incomplete"
    assert ledger.snapshot()["delivered_samples"] == 0


def test_observer_yields_the_original_batch_and_snapshot_is_detached():
    delivered = batch()
    module = SimpleNamespace(cycle=cycle, update_policy=update)
    ledger = TrainingObservation([(3, 0), (5, 0)], 2)
    with observe_native_training(module, ledger):
        observed = next(module.cycle([delivered]))
        assert observed is delivered and delivered == batch()
        module.update_policy(None, None, observed, Optimizer())
    ledger.snapshot()["learning_rates"][0][0] = 0
    assert ledger.snapshot()["learning_rates"] == [[1e-4]]


def test_incomplete_run_and_gradient_accumulation_are_not_completed_updates():
    module = SimpleNamespace(cycle=cycle, update_policy=update)
    ledger = TrainingObservation([(3, 0), (5, 0)] * 2, 2)
    with observe_native_training(module, ledger):
        module.update_policy(None, None, next(module.cycle([batch()])), Optimizer())
    assert ledger.snapshot()["status"] == "incomplete"
    optimizer = Optimizer()
    optimizer.gradient_state = SimpleNamespace(sync_gradients=False)
    ledger = TrainingObservation([(3, 0), (5, 0)], 2)
    with pytest.raises(RuntimeError, match="every batch"):
        with observe_native_training(module, ledger):
            module.update_policy(None, None, next(module.cycle([batch()])), optimizer)
    assert optimizer.calls == ledger.snapshot()["successful_optimizer_step_calls"] == 0


def test_real_native_cpu_update_and_accelerated_optimizer_are_unchanged():
    import torch
    from accelerate import Accelerator

    native_module = pytest.importorskip("lerobot.scripts.lerobot_train")

    class ScalarPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([0.5]))

        def forward(self, _batch):
            return (self.weight * torch.randn(1)).square().sum(), {}

    def run(observed):
        torch.manual_seed(71)
        accelerator = Accelerator(cpu=True, gradient_accumulation_steps=1)
        policy = ScalarPolicy()
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda index: 1 / (index + 1))
        policy, optimizer, scheduler = accelerator.prepare(policy, optimizer, scheduler)
        ledger = TrainingObservation([(3, 0), (5, 0)] * 2, 2)
        original_cycle, original_update = native_module.cycle, native_module.update_policy
        original_step = optimizer.step
        scope = observe_native_training(native_module, ledger) if observed else nullcontext()
        with scope:
            iterator = native_module.cycle([batch()])
            losses = []
            for _ in range(2):
                metrics, output = native_module.update_policy(
                    SimpleNamespace(),
                    policy,
                    next(iterator),
                    optimizer,
                    10,
                    accelerator,
                    lr_scheduler=scheduler,
                )
                losses.append((metrics.loss, metrics.lr, output))
        assert optimizer.step == original_step
        assert native_module.cycle is original_cycle
        assert native_module.update_policy is original_update
        return policy.weight.detach().clone(), torch.get_rng_state(), losses, ledger.snapshot()

    plain, observed = run(False), run(True)
    assert torch.equal(plain[0], observed[0]) and torch.equal(plain[1], observed[1])
    assert plain[2] == observed[2]
    assert observed[3]["status"] == "complete"
    assert observed[3]["learning_rates"] == [[1e-4], [5e-5]]
