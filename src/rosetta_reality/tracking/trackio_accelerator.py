"""AutoDL Trackio logger with device-neutral accelerator memory evidence."""

from __future__ import annotations

import os
from typing import Any

from rosetta_reality.vla.accelerator_memory import tracking_memory_metrics

from .public_payload import sanitize_metrics
from .trackio_lerobot import TrackioLogger as HistoricalTrackioLogger


class AcceleratorTrackioLogger(HistoricalTrackioLogger):
    """Preserve the historical logger contract while adding CUDA metrics."""

    def log_dict(
        self,
        values: dict[str, Any],
        step: int | None = None,
        mode: str = "train",
        custom_step_key: str | None = None,
    ) -> None:
        if custom_step_key is not None:
            raise ValueError("Custom Trackio step keys are not approved for this VLA pipeline.")
        if step is None or isinstance(step, bool) or step < 0:
            raise ValueError("Trackio metrics require a non-negative integer step.")
        public_values = dict(values)
        if os.environ.get("ROSETTA_VLA_PHASE") in {"performance_benchmark", "formal"}:
            import torch

            public_values.update(
                tracking_memory_metrics(
                    torch,
                    os.environ.get("ROSETTA_TORCH_DEVICE", "cpu"),
                )
            )
        payload = sanitize_metrics(public_values, mode=mode)
        if payload:
            self._trackio.log(payload, step=step)
            self._last_step = max(self._last_step, step)
