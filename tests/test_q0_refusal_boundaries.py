"""Synthetic boundary harness checks; Linux container only, no qualification."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/check_q0_refusal_boundaries.py"
SPEC = importlib.util.spec_from_file_location("q0_refusal_boundaries", PATH)
assert SPEC is not None and SPEC.loader is not None
diagnostic = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = diagnostic
SPEC.loader.exec_module(diagnostic)


def test_registered_pairs_and_determinism_ledger():
    ledger = diagnostic.run_diagnostic()
    assert len(ledger["inventory"]) == len(ledger["cases"]) == 32
    assert len({case["case_id"] for case in ledger["inventory"]}) == 32
    assert all(len(case["outcomes"]) == 2 for case in ledger["cases"])
    assert ledger["engineering_checks_passed"]
    assert ledger["status"] == "passed"
    assert ledger["infrastructure_failure"] is False
    assert ledger["provider_qualified"] is False
    assert ledger["candidate_g1_passed"] is False
    assert ledger["formal_gate_verdict"] is None
    assert ledger["simulation_steps"] == 0
    assert not any(case["silent_action_on_negative"] for case in ledger["cases"])


def test_unexpected_decision_exception_is_not_accepted(monkeypatch):
    def fail(self, observation):
        raise ValueError("unrelated implementation failure")
    monkeypatch.setattr(diagnostic.CoupledTranslationInsertionTeacher, "decide", fail)
    boundary = next(item for item in diagnostic.BOUNDARIES if item.identity == "reported_collision")
    outcome = diagnostic.observe(boundary, True)
    assert outcome == {"kind": "unexpected_exception", "layer": "decision", "exception_type": "ValueError"}


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError, TypeError])
def test_constructor_exception_whitelist_is_specific(monkeypatch, error_type):
    def fail(identity, negative):
        raise error_type("unrelated constructor failure")
    monkeypatch.setattr(diagnostic, "build_observation", fail)
    assert diagnostic.observe(diagnostic.BOUNDARIES[0], True)["kind"] == "unexpected_exception"


def test_positive_constructor_failure_is_not_boundary_success(monkeypatch):
    def fail(identity, negative):
        raise ValueError("positive control must remain constructible")
    monkeypatch.setattr(diagnostic, "build_observation", fail)
    assert diagnostic.observe(diagnostic.BOUNDARIES[0], False)["kind"] == "unexpected_exception"


def test_logical_failure_has_batch_status_and_does_not_claim_infrastructure(monkeypatch):
    monkeypatch.setattr(diagnostic, "observe", lambda boundary, negative: {
        "kind": "unexpected_exception", "layer": "decision", "exception_type": "ValueError"})
    ledger = diagnostic.run_diagnostic()
    assert ledger["engineering_checks_passed"] is False
    assert ledger["status"] == "check_failed"
    assert ledger["infrastructure_failure"] is False


@pytest.mark.parametrize("axis", "xyz")
@pytest.mark.parametrize("side", ["lower", "upper"])
def test_workspace_controls_stay_inside_float32_bounds(axis, side):
    observation = diagnostic.build_observation(f"workspace_{axis}_{side}", False)
    settings = diagnostic.ScriptedTeacherSettings()
    pose = observation.socket if side == "lower" else observation.peg
    actual = float(pose.position["xyz".index(axis)])
    assert getattr(settings, f"workspace_{axis}_min_m") < actual
    assert actual < getattr(settings, f"workspace_{axis}_max_m")


@pytest.mark.parametrize("budget", [0, -1, 121, float("nan")])
def test_budget_cannot_expand(budget):
    with pytest.raises(ValueError):
        diagnostic.run_diagnostic(max_seconds=budget)


def test_create_only_output_preflight(tmp_path, monkeypatch):
    target = tmp_path / "existing.json"
    target.write_text("preserved", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [str(PATH), "--output", str(target)])
    with pytest.raises(SystemExit):
        diagnostic.main()
    assert target.read_text(encoding="utf-8") == "preserved"
