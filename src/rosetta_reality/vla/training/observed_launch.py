"""Wrap an already-authorized v2 launcher without replacing its native loop.

The outer job must validate its sealed source/plan/schedule and resource contract
before calling this adapter. This module neither grants execution permission nor
loads any model/data. Install the observer before invoking the v2 launcher, whose
feature stack tears down before the observer. Every output is create-only.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from rosetta_reality.vla.training.observation import (
    TrainingObservation,
    observe_native_training,
)


def _save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def run_observed_launch(module, launch, *, expected_samples, batch_size, output: Path):
    """Return the native launcher result; persist failures without masking their cause.

    ``launch`` must be the guarded v2 launcher, not a second training loop. The
    expected episode/frame order is supplied by a separately hash-bound native
    sampler inspection; this adapter never consumes RNG to derive a schedule.
    """
    ledger = TrainingObservation(expected_samples, batch_size)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    schedule_sha = hashlib.sha256(
        json.dumps(ledger.expected, separators=(",", ":")).encode()
    ).hexdigest()
    _save(
        output / "started.json",
        {
            "status": "started_not_complete",
            "expected_samples": len(ledger.expected),
            "batch_size": batch_size,
            "expected_schedule_sha256": schedule_sha,
        },
    )
    failed = None
    try:
        with observe_native_training(module, ledger):
            result = launch()
            if result is not None and (type(result) is not int or result != 0):
                raise RuntimeError("The native launcher did not return success")
        if ledger.snapshot()["status"] != "complete":
            raise RuntimeError(
                "The native launcher did not complete the registered sample schedule"
            )
        return result
    except BaseException as exc:
        failed = type(exc).__name__
        raise
    finally:
        report = {
            "status": "passed" if failed is None else "failed",
            "error_type": failed,
            "expected_schedule_sha256": schedule_sha,
            "observation": ledger.snapshot(),
            "m2_complete": False,
            "task_success": "not measured",
        }
        try:
            _save(output / "result.json", report)
        except Exception:
            if failed is None:
                raise
            logging.error("Could not save observation evidence; preserving the native failure")
