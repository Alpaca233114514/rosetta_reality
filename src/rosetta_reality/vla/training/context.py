"""Immutable runtime context handed to every installed training feature."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rosetta_reality.vla.action_space import SmolVLAActionSpace

PHASE_SMOKE = "smoke"
PHASE_OVERFIT = "overfit"
PHASE_OVERFIT_RESUME = "overfit_resume"
PHASE_FORMAL = "formal"
PHASE_PERFORMANCE_BENCHMARK = "performance_benchmark"
TRAINING_PHASES = frozenset(
    {
        PHASE_SMOKE,
        PHASE_OVERFIT,
        PHASE_OVERFIT_RESUME,
        PHASE_FORMAL,
        PHASE_PERFORMANCE_BENCHMARK,
    }
)


@dataclass(frozen=True)
class TrainingContext:
    """Everything a feature may rely on when it installs or restores itself.

    The context is built once by the launcher-side entry point after the plan
    has been schema-validated and its parent experiment resolved.  Features
    must treat it as read-only: their only mutable surface is the pinned
    upstream module they wrap.
    """

    plan: dict[str, Any]
    experiment: dict[str, Any]
    action_space: SmolVLAActionSpace
    plan_path: Path
    experiment_path: Path
    contract_path: Path
    normalization_report: Path
    phase: str
    device: str
    run_name: str

    def __post_init__(self) -> None:
        if self.phase not in TRAINING_PHASES:
            raise ValueError(f"Unsupported v2 training phase: {self.phase!r}.")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("The v2 training context requires a device identifier.")
        if not isinstance(self.run_name, str) or not self.run_name:
            raise ValueError("The v2 training context requires a run name.")
