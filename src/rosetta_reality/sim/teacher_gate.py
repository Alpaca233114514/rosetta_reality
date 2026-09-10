"""T2 state-conditioned teacher contract and gate conformance primitives.

Implements the teacher contract of
``reports/training/m2-smolvla-t2-state-conditioned-teacher-gate-design-2026-08-28.md``
section 3: a teacher consumes only robot state, object geometry and
reward-derived event state, emits one absolute standard-space action under the
unchanged Action Contract or an explicit refusal, is deterministic given its
inputs, and refuses off-support states instead of degrading silently.

The observation type is the existing frozen :class:`InsertionGeometry`, which
structurally excludes wall-clock time, trajectory/frame indices, episode and
seed identity, dataset identity and any policy state — the forbidden-input
half of the contract is enforced by construction, not by convention.  The
:class:`ConformanceProbeTeacher` in this module is the gate harness's own
self-test target; it is not a task teacher and cannot pass any rollout stage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from rosetta_reality.sim.geometry_teacher import (
    GeometryTeacherError,
    InsertionGeometry,
)


class TeacherGateError(RuntimeError):
    """Raised for teacher-gate contract violations."""


class TeacherRefusalReason(str, Enum):
    """Registered refusal classes; every refusal names exactly one."""

    OUT_OF_SUPPORT = "out_of_support"
    CORRUPTED_GEOMETRY = "corrupted_geometry"
    UNSAFE_STATE = "unsafe_state"
    WORKSPACE_VIOLATION = "workspace_violation"


@dataclass(frozen=True, slots=True)
class TeacherRefusal:
    """An explicit fail-closed refusal; never a silent default action."""

    reason: TeacherRefusalReason
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.reason, TeacherRefusalReason):
            raise ValueError("Teacher refusal reason must be a registered class.")
        if not isinstance(self.detail, str) or not self.detail.strip():
            raise ValueError("Teacher refusal detail must be a non-empty string.")


@dataclass(frozen=True, slots=True)
class TeacherDecision:
    """Exactly one of an absolute action or an explicit refusal."""

    action: Tensor | None = None
    refusal: TeacherRefusal | None = None

    def __post_init__(self) -> None:
        if (self.action is None) == (self.refusal is None):
            raise ValueError(
                "A teacher decision must carry exactly one action or refusal."
            )
        if self.action is not None:
            action = torch.as_tensor(self.action, dtype=torch.float32).detach().cpu()
            if action.ndim != 1 or not bool(torch.isfinite(action).all()):
                raise ValueError(
                    "Teacher action must be a finite one-dimensional standard-space "
                    "action vector."
                )
            object.__setattr__(self, "action", action)


@runtime_checkable
class TeacherCandidate(Protocol):
    """The registered teacher surface every candidate must expose."""

    identity: str

    def reset(self) -> None:
        """Reset teacher-internal history only; the simulator stays caller-owned."""

    def decide(self, observation: InsertionGeometry) -> TeacherDecision:
        """Return one action or one explicit refusal for the observed state."""


def observation_fields() -> tuple[str, ...]:
    """The exhaustive registered observation field set (contract conformance)."""

    return tuple(
        field for field in InsertionGeometry.__dataclass_fields__  # type: ignore[attr-defined]
    )


def assert_observation_contract() -> None:
    """Fail closed if the observation type ever grows forbidden inputs.

    The registered observation must remain exactly the robot-state, geometry
    and event fields; any added field (a timestamp, an index, a seed, a policy
    handle) changes the teacher contract and must fail the gate.
    """

    expected = {
        "robot_state",
        "left_eef",
        "right_eef",
        "socket",
        "peg",
        "observed_reward",
        "socket_grasp_contact",
        "peg_grasp_contact",
        "socket_on_table",
        "peg_on_table",
        "peg_socket_contact",
        "pin_contact",
        "unexpected_collision_count",
    }
    actual = set(observation_fields())
    if actual != expected:
        raise TeacherGateError(
            "Teacher observation contract changed: registered fields "
            f"{sorted(expected)}, observed {sorted(actual)}."
        )


def probe_observation(
    *, robot_state: Tensor | None = None, observed_reward: float = 0.0
) -> InsertionGeometry:
    """Build one deterministic, workspace-valid synthetic observation.

    Used only by the G0 conformance probe and its tests; a rollout stage must
    build observations from the live simulator instead.
    """

    from rosetta_reality.sim.geometry_teacher import GeometryPose

    state = (
        torch.zeros(14, dtype=torch.float32)
        if robot_state is None
        else torch.as_tensor(robot_state, dtype=torch.float32).detach().cpu()
    )
    table = 0.0

    def pose(x: float, y: float, z: float) -> GeometryPose:
        return GeometryPose(
            position=torch.tensor([x, y, z], dtype=torch.float64),
            quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        )

    return InsertionGeometry(
        robot_state=state,
        left_eef=pose(-0.10, 0.10, table + 0.05),
        right_eef=pose(0.10, 0.10, table + 0.05),
        socket=pose(-0.10, 0.10, table),
        peg=pose(0.10, 0.10, table),
        observed_reward=observed_reward,
        socket_grasp_contact=False,
        peg_grasp_contact=False,
        socket_on_table=True,
        peg_on_table=True,
        peg_socket_contact=False,
        pin_contact=False,
        unexpected_collision_count=0,
    )


@dataclass(frozen=True, slots=True)
class ConformanceProbeResult:
    """G0 evidence for one candidate: determinism and refusal honesty."""

    deterministic: bool
    refusal_region_non_degenerate: bool
    decision_digest: str


def _decision_digest(decisions: list[TeacherDecision]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for decision in decisions:
        if decision.action is not None:
            digest.update(b"A")
            digest.update(decision.action.numpy().tobytes())
        else:
            assert decision.refusal is not None
            digest.update(b"R")
            digest.update(
                f"{decision.refusal.reason.value}:{decision.refusal.detail}".encode()
            )
    return digest.hexdigest()


def _decide_with_refusal_compat(
    candidate: TeacherCandidate, observation: InsertionGeometry
) -> TeacherDecision:
    """Decide once, converting legacy exception refusals into explicit ones.

    Legacy teachers signal refusal by raising :class:`GeometryTeacherError`;
    the gate records the class and message as the explicit refusal trace for
    every probe decision, not only the off-support site, so a legacy teacher
    is measured instead of crashing the probe.
    """

    try:
        return candidate.decide(observation)
    except GeometryTeacherError as error:
        return TeacherDecision(
            refusal=TeacherRefusal(
                reason=TeacherRefusalReason.OUT_OF_SUPPORT, detail=str(error)
            )
        )


def run_conformance_probe(
    candidate: TeacherCandidate,
    *,
    probe_observations: list[InsertionGeometry],
    off_support_observation: InsertionGeometry,
) -> ConformanceProbeResult:
    """Execute the G0 property suite against one candidate instance.

    Determinism: replaying the same observation sequence from the same reset
    must reproduce byte-identical decisions.  Refusal honesty: the registered
    off-support observation must produce an explicit refusal, never an action.
    """

    candidate.reset()
    first: list[TeacherDecision] = []
    for observation in probe_observations:
        first.append(_decide_with_refusal_compat(candidate, observation))
    candidate.reset()
    second: list[TeacherDecision] = []
    for observation in probe_observations:
        second.append(_decide_with_refusal_compat(candidate, observation))
    digest_first = _decision_digest(first)
    deterministic = digest_first == _decision_digest(second)

    off_decision = _decide_with_refusal_compat(candidate, off_support_observation)
    refusal_region_non_degenerate = off_decision.refusal is not None

    return ConformanceProbeResult(
        deterministic=deterministic,
        refusal_region_non_degenerate=refusal_region_non_degenerate,
        decision_digest=digest_first,
    )


class ConformanceProbeTeacher:
    """Deterministic harness self-test target; not a task teacher.

    It holds a fixed joint-space home action, refuses on any observation that
    reports an unexpected collision or an off-workspace reward, and is
    byte-deterministic.  Its only role is to prove the G0 suite can detect
    non-determinism and silent defaults.
    """

    identity = "gate-conformance-probe-001"

    def __init__(self, dimension: int = 14) -> None:
        if dimension < 1:
            raise ValueError("Probe dimension must be positive.")
        self._action = torch.zeros(dimension, dtype=torch.float32)

    def reset(self) -> None:
        return None

    def decide(self, observation: InsertionGeometry) -> TeacherDecision:
        if observation.unexpected_collision_count:
            return TeacherDecision(
                refusal=TeacherRefusal(
                    reason=TeacherRefusalReason.UNSAFE_STATE,
                    detail="conformance probe refuses under unexpected collisions",
                )
            )
        if not math.isfinite(observation.observed_reward):
            return TeacherDecision(
                refusal=TeacherRefusal(
                    reason=TeacherRefusalReason.CORRUPTED_GEOMETRY,
                    detail="conformance probe refuses non-finite observed reward",
                )
            )
        return TeacherDecision(action=self._action.clone())
