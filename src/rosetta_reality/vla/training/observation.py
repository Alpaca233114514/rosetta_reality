"""Observe delivered sample identities and completed native updates without changing batches."""

from __future__ import annotations

import math
import sys
from contextlib import contextmanager
from functools import wraps


class TrainingObservation:
    """An incomplete or failed update can never count as completed sample exposure."""

    def __init__(self, expected_samples, batch_size):
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("Observation requires a positive batch size")
        self.expected = [tuple(sample) for sample in expected_samples]
        if not self.expected or len(self.expected) % batch_size:
            raise ValueError("Expected samples must contain whole batches")
        if any(
            len(sample) != 2 or any(type(x) is not int or x < 0 for x in sample)
            for sample in self.expected
        ):
            raise ValueError("Expected identities must be nonnegative episode/frame pairs")
        self.batch_size = batch_size
        self.delivered = []
        self.completed = []
        self.learning_rates = []
        self.pending = None
        self.errors = []

    def deliver(self, batch):
        try:
            self._deliver(batch)
        except BaseException as exc:
            self.errors.append({"stage": "delivery", "type": type(exc).__name__})
            raise

    def _deliver(self, batch):
        if self.pending is not None or self.errors:
            raise RuntimeError("Cannot deliver another batch after an incomplete update")
        columns = []
        for key in ("episode_index", "frame_index"):
            value = batch[key]
            if hasattr(value, "detach"):
                value = value.detach().cpu()
            if hasattr(value, "tolist"):
                value = value.tolist()
            if not isinstance(value, list) or len(value) != self.batch_size:
                raise ValueError("Batch identity columns must match batch size")
            if any(type(x) is not int or x < 0 for x in value):
                raise ValueError("Batch identities must be nonnegative integers")
            columns.append(value)
        actual = list(zip(*columns, strict=True))
        start = len(self.delivered)
        if actual != self.expected[start : start + self.batch_size]:
            raise ValueError("Delivered samples differ from the registered order")
        self.delivered.extend(actual)
        self.pending = actual

    def snapshot(self):
        complete = (
            not self.errors
            and self.pending is None
            and self.completed == self.expected
            and self.delivered == self.expected
            and len(self.learning_rates) * self.batch_size == len(self.expected)
        )
        return {
            "status": "complete" if complete else "incomplete",
            "expected_samples": len(self.expected),
            "delivered_samples": len(self.delivered),
            "completed_samples": len(self.completed),
            "completed_updates": len(self.completed) // self.batch_size,
            "successful_optimizer_step_calls": len(self.learning_rates),
            "learning_rates": [list(rates) for rates in self.learning_rates],
            "pending_samples": [] if self.pending is None else list(self.pending),
            "errors": [dict(error) for error in self.errors],
        }


@contextmanager
def observe_native_training(module, observation):
    """Install before native feature wrappers; restore after their reverse teardown.

    The received optimizer's bound step is wrapped only during one native update.
    This preserves PyTorch scheduler/profiler wrappers and Accelerate's optimizer.
    A returned but skipped step is rejected; a later failure retains the successful
    step count while leaving its sample batch incomplete.
    """
    original_cycle = module.cycle
    original_update = module.update_policy

    @wraps(original_cycle)
    def cycle(*args, **kwargs):
        for batch in original_cycle(*args, **kwargs):
            observation.deliver(batch)
            yield batch

    @wraps(original_update)
    def update(*args, **kwargs):
        if observation.pending is None or observation.errors:
            observation.errors.append({"stage": "update_entry", "type": "MissingBatch"})
            raise RuntimeError("Native update requires exactly one delivered batch")
        optimizer = kwargs.get("optimizer", args[3] if len(args) > 3 else None)
        if optimizer is None:
            observation.errors.append({"stage": "update_entry", "type": "MissingOptimizer"})
            raise ValueError("Native optimizer argument is missing")
        missing = object()
        local_step = vars(optimizer).get("step", missing)
        original_step = optimizer.step
        initial_steps = len(observation.learning_rates)
        attempted = False

        @wraps(original_step)
        def step(*step_args, **step_kwargs):
            nonlocal attempted
            if attempted:
                observation.errors.append({"stage": "optimizer_step", "type": "MultipleSteps"})
                raise RuntimeError("Native update attempted multiple optimizer steps")
            attempted = True
            gradient_state = getattr(optimizer, "gradient_state", None)
            if gradient_state is not None and not gradient_state.sync_gradients:
                raise RuntimeError("Observation requires an optimizer update on every batch")
            rates = [float(group["lr"]) for group in optimizer.param_groups]
            if not rates or any(not math.isfinite(rate) or rate < 0 for rate in rates):
                raise ValueError("Optimizer learning rate is not finite and nonnegative")
            result = original_step(*step_args, **step_kwargs)
            if getattr(optimizer, "step_was_skipped", False):
                raise RuntimeError("Native optimizer step was skipped")
            observation.learning_rates.append(rates)
            return result

        optimizer.step = step
        try:
            result = original_update(*args, **kwargs)
            if observation.errors or len(observation.learning_rates) != initial_steps + 1:
                raise RuntimeError("Native update did not complete one optimizer step")
            observation.completed.extend(observation.pending)
            observation.pending = None
            return result
        except BaseException as exc:
            observation.errors.append({"stage": "native_update", "type": type(exc).__name__})
            raise
        finally:
            if vars(optimizer).get("step") is not step:
                observation.errors.append({"stage": "optimizer_restore", "type": "HookChanged"})
                if sys.exc_info()[0] is None:
                    raise RuntimeError("Optimizer step hook changed during observation")
            elif local_step is missing:
                del optimizer.step
            else:
                optimizer.step = local_step

    module.cycle, module.update_policy = cycle, update
    try:
        yield observation
    except BaseException as exc:
        observation.errors.append({"stage": "training_scope", "type": type(exc).__name__})
        raise
    finally:
        drifted = []
        for name, installed, original in (
            ("cycle", cycle, original_cycle),
            ("update_policy", update, original_update),
        ):
            if getattr(module, name) is installed:
                setattr(module, name, original)
            else:
                drifted.append(name)
        if drifted:
            observation.errors.append({"stage": "module_restore", "type": "HookChanged"})
            if sys.exc_info()[0] is None:
                raise RuntimeError("Native observation hooks changed during execution")
