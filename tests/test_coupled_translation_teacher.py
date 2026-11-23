"""Synthetic contract tests; these do not establish live teacher qualification."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rosetta_reality.sim.coupled_translation_teacher import (  # noqa: E402
    CANDIDATE_ID,
    TRANSLATIONS_M,
    CoupledTranslationInsertionTeacher,
    _SharedTranslationSolver,
)
from rosetta_reality.sim.geometric_teacher import GeometricEscapeSettings  # noqa: E402
from rosetta_reality.sim.mink_aloha_ik import MinkAlohaIkResult  # noqa: E402
from rosetta_reality.sim.scripted_teacher import ScriptedInsertionStage  # noqa: E402
from rosetta_reality.sim.teacher_gate import probe_observation  # noqa: E402


class RecordingSolver:
    def __init__(self, *, prefer_x: bool = False, fail: bool = False) -> None:
        self.calls: list[dict] = []
        self.prefer_x = prefer_x
        self.fail = fail

    def solve(self, current_qpos, **targets):
        self.calls.append(targets)
        if self.fail:
            raise RuntimeError("synthetic unsupported target")
        qpos = np.array(current_qpos, copy=True)
        qpos[0] += 0.6
        residual = abs(float(targets["left_position"][0]) - 0.02) if self.prefer_x else 0.0
        return MinkAlohaIkResult(qpos, 1, (residual, residual), (0.0, 0.0))


def targets():
    return {"left_position": np.array([0.0, 0.1, 0.2]),
            "right_position": np.array([0.3, 0.4, 0.5]),
            "left_quaternion_wxyz": np.array([1., 0., 0., 0.]),
            "right_quaternion_wxyz": np.array([1., 0., 0., 0.])}


def test_lattice_preserves_relative_targets_and_has_fixed_tie_break():
    solver = RecordingSolver()
    wrapper = _SharedTranslationSolver(solver)
    original = targets()
    wrapper.solve(np.zeros(16), **original)
    assert len(solver.calls) == 6
    for call, delta in zip(solver.calls, TRANSLATIONS_M):
        np.testing.assert_allclose(call["left_position"], original["left_position"] + delta)
        np.testing.assert_allclose(call["right_position"] - call["left_position"],
                                   original["right_position"] - original["left_position"])
        np.testing.assert_array_equal(call["left_quaternion_wxyz"], original["left_quaternion_wxyz"])
    assert wrapper.branch_diagnostics[0]["selected_index"] == 0


def test_selection_uses_actual_solved_residual():
    wrapper = _SharedTranslationSolver(RecordingSolver(prefer_x=True))
    wrapper.solve(np.zeros(16), **targets())
    assert wrapper.branch_diagnostics[0]["selected_index"] == 1


def test_all_failed_targets_remain_failure():
    wrapper = _SharedTranslationSolver(RecordingSolver(fail=True))
    with pytest.raises(RuntimeError, match="all six"):
        wrapper.solve(np.zeros(16), **targets())
    assert wrapper.branch_diagnostics[0]["selected_index"] is None


def test_insert_command_keeps_joint_bound_and_restores_solver():
    solver = RecordingSolver()
    teacher = CoupledTranslationInsertionTeacher(
        solver=solver, escape_settings=GeometricEscapeSettings(hold_enabled=False)
    )
    teacher._phase = ScriptedInsertionStage.INSERT
    observation = probe_observation()
    decision = teacher._task_target_action(observation, observation.left_eef,
                                           observation.right_eef, 0.0)
    assert decision.refusal is None
    assert float(decision.action[0]) <= teacher.settings.maximum_joint_target_delta_rad + 1e-7
    assert len(solver.calls) == 6
    assert teacher._solver is solver
    assert teacher.identity == CANDIDATE_ID
    assert len(teacher.translation_diagnostics) == 1
    teacher.reset()
    assert teacher.translation_diagnostics == []


def test_non_insert_phase_does_not_search_translation_grid():
    solver = RecordingSolver()
    teacher = CoupledTranslationInsertionTeacher(
        solver=solver, escape_settings=GeometricEscapeSettings(hold_enabled=False)
    )
    observation = probe_observation()
    teacher._task_target_action(observation, observation.left_eef, observation.right_eef, 0.0)
    assert len(solver.calls) == 1
    assert teacher.translation_diagnostics == []


def test_all_failed_insert_branches_refuse_with_bounded_work():
    solver = RecordingSolver(fail=True)
    teacher = CoupledTranslationInsertionTeacher(
        solver=solver, escape_settings=GeometricEscapeSettings(hold_enabled=False)
    )
    teacher._phase = ScriptedInsertionStage.INSERT
    observation = probe_observation()
    decision = teacher._task_target_action(observation, observation.left_eef,
                                           observation.right_eef, 0.0)
    assert decision.action is None
    assert decision.refusal is not None
    assert len(solver.calls) == 54
    assert teacher._solver is solver
    assert len(teacher.translation_diagnostics) == 9
