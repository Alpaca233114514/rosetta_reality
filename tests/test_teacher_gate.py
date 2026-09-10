"""T2 teacher-gate contract and runner tests (no weights, no simulator).

Covers the frozen teacher contract of
``reports/training/m2-smolvla-t2-teacher-gate-protocol-preregistration-2026-08-29.json``:
the observation field set is structurally closed against forbidden inputs,
teacher decisions carry exactly one action or explicit refusal, the G0
conformance suite detects non-determinism and silent defaults, and the staged
runner writes create-only evidence and fails closed without declared
capabilities or earlier passing stages.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for candidate in (str(REPOSITORY_ROOT / "src"), str(REPOSITORY_ROOT / "scripts")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import rosetta_reality.sim.teacher_gate as teacher_gate  # noqa: E402
from rosetta_reality.sim.teacher_gate import (  # noqa: E402
    ConformanceProbeTeacher,
    TeacherDecision,
    TeacherRefusal,
    TeacherRefusalReason,
    assert_observation_contract,
    probe_observation,
    run_conformance_probe,
)


def test_observation_field_set_is_frozen() -> None:
    assert_observation_contract()
    fields = set(teacher_gate.observation_fields())
    assert not fields & {
        "time",
        "step",
        "timestamp",
        "frame_index",
        "episode",
        "seed",
        "dataset",
        "policy",
    }


def test_decision_carries_exactly_one_payload() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        TeacherDecision()
    with pytest.raises(ValueError, match="exactly one"):
        TeacherDecision(
            action=torch.zeros(14),
            refusal=TeacherRefusal(
                reason=TeacherRefusalReason.OUT_OF_SUPPORT, detail="both"
            ),
        )
    action_decision = TeacherDecision(action=torch.tensor([0.5]))
    assert action_decision.action is not None
    assert action_decision.refusal is None
    with pytest.raises(ValueError, match="finite"):
        TeacherDecision(action=torch.tensor([float("nan")]))


def test_refusal_requires_registered_reason_and_detail() -> None:
    with pytest.raises(ValueError, match="registered class"):
        TeacherRefusal(reason="made_up", detail="x")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        TeacherRefusal(reason=TeacherRefusalReason.UNSAFE_STATE, detail="  ")


def test_probe_is_deterministic_and_refuses_off_support() -> None:
    probe = ConformanceProbeTeacher()
    observations = [
        probe_observation(),
        probe_observation(observed_reward=2.0),
    ]
    result = run_conformance_probe(
        probe,
        probe_observations=observations,
        off_support_observation=dataclasses.replace(
            probe_observation(), unexpected_collision_count=3
        ),
    )
    assert result.deterministic is True
    assert result.refusal_region_non_degenerate is True
    replay = run_conformance_probe(
        ConformanceProbeTeacher(),
        probe_observations=observations,
        off_support_observation=dataclasses.replace(
            probe_observation(), unexpected_collision_count=3
        ),
    )
    assert replay.decision_digest == result.decision_digest


def test_conformance_suite_detects_silent_default() -> None:
    class SilentTeacher(ConformanceProbeTeacher):
        identity = "silent-default-bad"

        def decide(self, observation):  # type: ignore[no-untyped-def]
            return TeacherDecision(action=self._action.clone())

    result = run_conformance_probe(
        SilentTeacher(),
        probe_observations=[probe_observation()],
        off_support_observation=dataclasses.replace(
            probe_observation(), unexpected_collision_count=1
        ),
    )
    assert result.refusal_region_non_degenerate is False


def test_conformance_suite_detects_nondeterminism() -> None:
    import random

    class DriftingTeacher(ConformanceProbeTeacher):
        identity = "drifting-bad"

        def decide(self, observation):  # type: ignore[no-untyped-def]
            decision = super().decide(observation)
            if decision.action is None:
                return decision
            noise = random.random()
            return TeacherDecision(action=decision.action + noise * 1e-3)

    result = run_conformance_probe(
        DriftingTeacher(),
        probe_observations=[probe_observation()],
        off_support_observation=dataclasses.replace(
            probe_observation(), unexpected_collision_count=1
        ),
    )
    assert result.deterministic is False


def test_conformance_suite_records_legacy_exception_refusals() -> None:
    from rosetta_reality.sim.geometry_teacher import GeometryTeacherError

    class LegacyExceptionTeacher:
        identity = "legacy-exception-teacher"

        def reset(self) -> None:
            return None

        def decide(self, observation):  # type: ignore[no-untyped-def]
            raise GeometryTeacherError("legacy refusal region")

    result = run_conformance_probe(
        LegacyExceptionTeacher(),
        probe_observations=[probe_observation()],
        off_support_observation=dataclasses.replace(
            probe_observation(), unexpected_collision_count=1
        ),
    )
    # The probe replays every decision through the refusal-compat boundary:
    # a legacy exception teacher is measured as explicit refusals (and stays
    # deterministic) instead of crashing the suite.
    assert result.deterministic is True
    assert result.refusal_region_non_degenerate is True


def test_runner_g0_writes_create_only_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ROSETTA_RUN_ROOT", str(tmp_path))
    runner = importlib.import_module("run_teacher_gate")
    assert runner.PREREGISTRATION.endswith(
        "2026-08-29.json"
    ), "runner must bind the frozen preregistration"
    registration_dir = runner._gate_root() / "registrations"
    registration_dir.mkdir(parents=True, exist_ok=True)
    registration = {
        "schema_version": 1,
        "candidate_id": "gate-conformance-probe-001",
        "module": "rosetta_reality.sim.teacher_gate",
        "factory": "ConformanceProbeTeacher",
        "constructor": {},
        "capabilities": {"simulator_execution": False},
    }
    (registration_dir / "gate-conformance-probe-001.json").write_text(
        json.dumps(registration), encoding="utf-8"
    )
    namespace = argparse_namespace("g0", "gate-conformance-probe-001")
    exit_code = runner.cmd_g0(namespace)
    assert exit_code == 0
    stage_root = runner._gate_root() / "gate-conformance-probe-001"
    evidence_files = sorted(stage_root.glob("g0-*.json"))
    assert len(evidence_files) == 1
    first_text = evidence_files[0].read_text(encoding="utf-8")
    payload = json.loads(first_text)
    assert payload["passed"] is True
    assert payload["harness_self_test"] is True
    assert payload["checks"]["deterministic_under_replay"] is True
    assert payload["checks"]["refusal_region_non_degenerate"] is True

    # Attempts increment create-only; prior evidence is never overwritten.
    exit_code = runner.cmd_g0(namespace)
    assert exit_code == 0
    attempts = sorted(stage_root.glob("g0-*.json"))
    assert len(attempts) == 2
    assert attempts[0].read_text(encoding="utf-8") == first_text


def test_runner_rollout_stage_fails_closed_without_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ROSETTA_RUN_ROOT", str(tmp_path))
    runner = importlib.import_module("run_teacher_gate")
    registration_dir = runner._gate_root() / "registrations"
    registration_dir.mkdir(parents=True, exist_ok=True)
    (registration_dir / "gate-conformance-probe-001.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidate_id": "gate-conformance-probe-001",
                "module": "rosetta_reality.sim.teacher_gate",
                "capabilities": {"simulator_execution": False},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="simulator_execution"):
        runner._require_simulator_capability(
            json.loads(
                (
                    registration_dir / "gate-conformance-probe-001.json"
                ).read_text(encoding="utf-8")
            )
        )


def test_runner_stage_guard_requires_earlier_passing_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ROSETTA_RUN_ROOT", str(tmp_path))
    runner = importlib.import_module("run_teacher_gate")
    namespace = argparse_namespace("g2", "any-candidate")
    with pytest.raises(FileNotFoundError, match="not registered"):
        runner._load_registration("any-candidate")
    stage_root = runner._gate_root() / "any-candidate"
    stage_root.mkdir(parents=True, exist_ok=True)
    registration_dir = runner._gate_root() / "registrations"
    registration_dir.mkdir(parents=True, exist_ok=True)
    registration = {
        "schema_version": 1,
        "candidate_id": "any-candidate",
        "capabilities": {"simulator_execution": True},
        "observer_module": "some_observer",
    }
    (registration_dir / "any-candidate.json").write_text(
        json.dumps(registration), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="fail-closed"):
        runner._stage_guard(namespace, "g2")


def test_runner_rollout_records_action_contract_clipping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clipping observability: an out-of-contract teacher action must surface
    in the episode evidence instead of silently executing a clipped target."""

    runner = importlib.import_module("run_teacher_gate")

    class StubEnvironment:
        def __init__(self, contract: Any, maximum_episode_steps: int) -> None:
            self.maximum_episode_steps = maximum_episode_steps

        def reset(self, seed: int | None = None) -> dict[str, Any]:
            return {}

        def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
            return {}, 0.0, True, {"is_success": False, "terminated": False}

        def state_limit_violation_count(self) -> int:
            return 0

        def unexpected_collision_count(self) -> int:
            return 0

        def close(self) -> None:
            return None

    stub_module = types.ModuleType("rosetta_reality.sim.gym_aloha")
    stub_module.GymAlohaEnvironment = StubEnvironment  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rosetta_reality.sim.gym_aloha", stub_module)

    class ClippingTeacher:
        identity = "clipping-teacher"

        def reset(self) -> None:
            return None

        def decide(self, observation):  # type: ignore[no-untyped-def]
            action = torch.zeros(14, dtype=torch.float32)
            action[0] = 99.0  # far beyond the registered arm-joint bound
            return TeacherDecision(action=action)

    class StubObserver:
        @staticmethod
        def build_observation(environment: Any, contract: Any):  # type: ignore[no-untyped-def]
            return probe_observation()

    episodes = runner._rollout_episodes(
        ClippingTeacher(), StubObserver(), [10], 500
    )
    assert len(episodes) == 1
    assert episodes[0]["action_clip_entry_count"] == 1
    assert episodes[0]["action_clipped_fields"] == ["left_waist"]


def argparse_namespace(stage: str, candidate_id: str):
    import argparse as _argparse

    return _argparse.Namespace(command=stage, candidate_id=candidate_id)
