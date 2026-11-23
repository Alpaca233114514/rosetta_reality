"""Experimental Q0 teacher: bounded shared translation of INSERT IK targets.

This is a new candidate, not a qualified provider. The existing geometric
teacher and all protocol thresholds remain unchanged. No model/data is loaded.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from rosetta_reality.sim.geometric_teacher import MinkGeometricInsertionTeacher
from rosetta_reality.sim.scripted_teacher import ScriptedInsertionStage


CANDIDATE_ID = "mink-coupled-translation-teacher-20260927-001"
# World-frame metres. No downward candidate; order is the exact tie-break.
TRANSLATIONS_M = (
    (0.0, 0.0, 0.0),
    (0.02, 0.0, 0.0),
    (-0.02, 0.0, 0.0),
    (0.0, 0.02, 0.0),
    (0.0, -0.02, 0.0),
    (0.0, 0.0, 0.02),
)


class _SharedTranslationSolver:
    """Finite target comparison for each seed of the unchanged IK family."""

    def __init__(self, solver: Any) -> None:
        self.solver = solver
        self.branch_diagnostics: list[dict[str, Any]] = []

    def solve(self, current_qpos: Any, **targets: Any) -> Any:
        choices: list[tuple[tuple[float, float, int], Any]] = []
        trials: list[dict[str, Any]] = []
        for index, translation in enumerate(TRANSLATIONS_M):
            delta = np.asarray(translation, dtype=np.float64)
            shifted = dict(targets)
            shifted["left_position"] = np.asarray(targets["left_position"]) + delta
            shifted["right_position"] = np.asarray(targets["right_position"]) + delta
            try:
                result = self.solver.solve(np.array(current_qpos, copy=True), **shifted)
                position = max(float(value) for value in result.position_errors_m)
                orientation = max(float(value) for value in result.orientation_errors_rad)
                if not (math.isfinite(position) and math.isfinite(orientation)):
                    raise ValueError("non-finite IK residual")
                if position < 0.0 or orientation < 0.0:
                    raise ValueError("negative IK residual")
                if not np.isfinite(result.qpos).all():
                    raise ValueError("non-finite IK configuration")
                choices.append(((position, orientation, index), result))
                trials.append({"index": index, "position_m": position,
                               "orientation_rad": orientation, "solved": True})
            except (RuntimeError, ValueError) as error:
                trials.append({"index": index, "solved": False,
                               "error": str(error)})
        if not choices:
            self.branch_diagnostics.append({"selected_index": None, "trials": trials})
            raise RuntimeError("all six registered shared-translation targets failed")
        score, selected = min(choices, key=lambda item: item[0])
        self.branch_diagnostics.append({"selected_index": score[2], "trials": trials})
        return selected


class CoupledTranslationInsertionTeacher(MinkGeometricInsertionTeacher):
    """Same task/safety/action layer; INSERT jointly relocates both targets.

The parent bounds every resulting joint command and owns all refusals and
stage transitions. The lattice is anchored to the parent's target each call,
not accumulated into a moving reference. Target relative pose is preserved;
actual object relative pose is measured by the simulator, never assumed.
"""

    identity = CANDIDATE_ID

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.translation_diagnostics: list[dict[str, Any]] = []

    def reset(self) -> None:
        super().reset()
        self.translation_diagnostics = []

    def _task_target_action(self, observation: Any, left_target: Any,
                            right_target: Any, gripper: float) -> Any:
        self.translation_diagnostics = []
        if self._phase is not ScriptedInsertionStage.INSERT:
            return super()._task_target_action(
                observation, left_target, right_target, gripper
            )
        original_solver = self._solver
        adapter = _SharedTranslationSolver(original_solver)
        self._solver = adapter
        try:
            return super()._task_target_action(
                observation, left_target, right_target, gripper
            )
        finally:
            self._solver = original_solver
            self.translation_diagnostics = adapter.branch_diagnostics


def build_teacher() -> CoupledTranslationInsertionTeacher:
    """Create this fixed candidate; registration supplies no tunable kwargs."""
    return CoupledTranslationInsertionTeacher()
