"""Observe the existing online adapter without changing its sampling or actions."""

from __future__ import annotations

import copy
import os
import random
import threading
from contextlib import contextmanager
from pathlib import Path

from .gate_diagnostic_io import seal, write_json, write_tree
from .rollout_trace import run_traced_rollout

_LOCK = threading.Lock()


def clone_tree(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: clone_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(clone_tree(v) for v in value)
    return copy.deepcopy(value)


@contextmanager
def rng_scope(online, *, restore_always=False):
    """Successful live calls advance normally; errors/replays restore entry RNGs."""
    import numpy as np
    import torch

    py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    generator = online._noise_generator
    own = generator.get_state()
    failed = True
    try:
        yield
        failed = False
    finally:
        if failed or restore_always:
            random.setstate(py)
            np.random.set_state(np_state)
            torch.set_rng_state(cpu)
            if cuda is not None:
                torch.cuda.set_rng_state_all(cuda)
            generator.set_state(own)


def capture_prediction(online, observation, instruction, *, forced_noise=None):
    """Capture actual model ingress/egress. Forced noise is replay-only."""
    model = online.policy
    method = model.predict_action_chunk
    had_override = "predict_action_chunk" in vars(model)
    old_override = vars(model).get("predict_action_chunk")
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError("Concurrent diagnostic model binding")
    captured = {}
    calls = 0

    def observed(batch, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls != 1 or args or "noise" not in kwargs:
            raise ValueError("Expected exactly one explicit-noise native model call")
        if forced_noise is not None:
            original = kwargs["noise"]
            if forced_noise.shape != original.shape or forced_noise.dtype != original.dtype:
                raise ValueError("Replay noise contract differs")
            kwargs["noise"] = forced_noise.to(original.device).clone()
        captured["batch"] = clone_tree(batch)
        captured["noise"] = clone_tree(kwargs["noise"])
        result = method(batch, **kwargs)
        captured["normalized"] = clone_tree(result)
        return result

    try:
        model.predict_action_chunk = observed
        captured.update(observation=clone_tree(observation), instruction=instruction)
        with rng_scope(online, restore_always=forced_noise is not None):
            result = online.predict(observation, instruction)
            if calls != 1:
                raise ValueError("Native model ingress was not observed")
            captured["decoded"], captured["projected"] = clone_tree(result)
            decoders = [
                s
                for s in online.postprocessor.steps
                if getattr(type(s), "_registry_name", None) == "rosetta_pi_aloha_postprocessor"
            ]
            if len(decoders) != 1:
                raise ValueError("Unique saved action decoder required")
            captured["internal"] = clone_tree(decoders[0].last_model_action)
        return result, captured
    finally:
        if had_override:
            model.predict_action_chunk = old_override
        else:
            delattr(model, "predict_action_chunk")
        _LOCK.release()


def collect_episode(
    engine,
    online,
    contract,
    output,
    identity,
    *,
    maximum_steps=500,
    maximum_bytes=4 * 1024**3,
    guard=lambda: None,
    check_unchanged=lambda: True,
):
    """Single Seed 3 episode; no training, retry, checkpoint selection or model loading."""
    if not 1 <= maximum_steps <= 500 or contract.frequency_hz != 50:
        raise ValueError("Seed 3 requires at most 500 steps at 50 Hz")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "identity.json",
        {
            "schema_version": 1,
            "stage": "collect",
            "identity": identity,
            "pid": os.getpid(),
            "dimension_names": list(contract.dimension_names),
            "chunk_length": contract.chunk_length,
            "maximum_steps": maximum_steps,
            "maximum_bytes": maximum_bytes,
            "seed": 3,
            "policy_noise_seed": 3,
            "diagnostic_only": True,
            "optimizer_steps": 0,
        },
    )
    count = 0
    result = {
        "stage": "collect",
        "status": "incomplete",
        "diagnostic_only": True,
        "optimizer_steps": 0,
        "task_success": None,
        "error_type": None,
    }

    class Observed:
        postprocessor = online.postprocessor

        def configure_noise(self, *args, **kwargs):
            return online.configure_noise(*args, **kwargs)

        def predict(self, observation, instruction):
            nonlocal count
            guard()
            returned, captured = capture_prediction(online, observation, instruction)
            write_tree(
                output / "predictions" / f"{count:04d}",
                captured,
                budget_root=output,
                maximum_bytes=maximum_bytes,
                reserve_bytes=min(32 * 1024**2, maximum_bytes // 4),
            )
            count += 1
            guard()
            return returned

    try:
        guard()
        with rng_scope(online):
            metrics = run_traced_rollout(
                engine,
                Observed(),
                contract,
                "Insert the peg into the socket.",
                output=output / "trace",
                identity=identity,
                episode_index=0,
                seed=3,
                policy_noise_seed=3,
                maximum_steps=maximum_steps,
                noise_mode="seeded_standard_normal",
                project_policy_output=True,
            )
            guard()
            if not check_unchanged():
                raise ValueError("Policy parameters changed")
        result.update(
            status="complete",
            task_success=metrics["success"],
            metrics=metrics,
            parameters_unchanged=True,
        )
        return result
    except BaseException as exc:
        result["error_type"] = type(exc).__name__
        raise
    finally:
        result["predictions_saved"] = count
        seal(output, result)
