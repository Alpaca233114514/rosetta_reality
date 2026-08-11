"""Dependency-free ActionPlan v1 value objects and JSON-compatible conversion."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    return value


def _optional_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field)


@dataclass(frozen=True)
class ActionTarget:
    """A grounded target whose coordinates always carry frame and unit identity."""

    ALLOWED_KINDS: ClassVar[frozenset[str]] = frozenset(
        {"pixel", "point_3d", "region", "joint"}
    )

    kind: str
    values: tuple[float, ...]
    reference_frame: str
    unit: str

    def __post_init__(self) -> None:
        if self.kind not in self.ALLOWED_KINDS:
            raise ValueError(f"Unsupported target kind: {self.kind!r}.")
        if not self.values:
            raise ValueError("target.values must contain at least one coordinate.")
        if not all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
            for value in self.values
        ):
            raise ValueError("target.values must contain only finite numbers.")
        _required_text(self.reference_frame, "target.reference_frame")
        _required_text(self.unit, "target.unit")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ActionTarget:
        expected = {"kind", "values", "reference_frame", "unit"}
        if set(payload) != expected:
            raise ValueError(
                f"ActionTarget fields differ: missing={sorted(expected - set(payload))}, "
                f"extra={sorted(set(payload) - expected)}."
            )
        values = payload["values"]
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError("target.values must be an array of finite numbers.")
        return cls(
            kind=_required_text(payload["kind"], "target.kind"),
            values=tuple(values),
            reference_frame=_required_text(payload["reference_frame"], "target.reference_frame"),
            unit=_required_text(payload["unit"], "target.unit"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "values": list(self.values),
            "reference_frame": self.reference_frame,
            "unit": self.unit,
        }


@dataclass(frozen=True)
class ActionPlan:
    """The complete version-1 wire contract from ER to VLA."""

    schema_version: int
    subtask: str
    object: str | None
    target: ActionTarget | None
    motion_hint: str | None
    constraints: tuple[str, ...]
    success_condition: str
    replan_condition: str

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("ActionPlan.schema_version must be 1.")
        _required_text(self.subtask, "subtask")
        _optional_text(self.object, "object")
        _optional_text(self.motion_hint, "motion_hint")
        _required_text(self.success_condition, "success_condition")
        _required_text(self.replan_condition, "replan_condition")
        for index, constraint in enumerate(self.constraints):
            _required_text(constraint, f"constraints[{index}]")
        if len(set(self.constraints)) != len(self.constraints):
            raise ValueError("constraints must not contain duplicates.")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ActionPlan:
        expected = {
            "schema_version",
            "subtask",
            "object",
            "target",
            "motion_hint",
            "constraints",
            "success_condition",
            "replan_condition",
        }
        if set(payload) != expected:
            raise ValueError(
                f"ActionPlan fields differ: missing={sorted(expected - set(payload))}, "
                f"extra={sorted(set(payload) - expected)}."
            )
        constraints = payload["constraints"]
        if not isinstance(constraints, Sequence) or isinstance(constraints, (str, bytes)):
            raise ValueError("constraints must be an array of strings.")
        raw_target = payload["target"]
        if raw_target is not None and not isinstance(raw_target, Mapping):
            raise ValueError("target must be null or an object.")
        return cls(
            schema_version=payload["schema_version"],
            subtask=_required_text(payload["subtask"], "subtask"),
            object=_optional_text(payload["object"], "object"),
            target=ActionTarget.from_dict(raw_target) if raw_target is not None else None,
            motion_hint=_optional_text(payload["motion_hint"], "motion_hint"),
            constraints=tuple(constraints),
            success_condition=_required_text(payload["success_condition"], "success_condition"),
            replan_condition=_required_text(payload["replan_condition"], "replan_condition"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "subtask": self.subtask,
            "object": self.object,
            "target": self.target.to_dict() if self.target is not None else None,
            "motion_hint": self.motion_hint,
            "constraints": list(self.constraints),
            "success_condition": self.success_condition,
            "replan_condition": self.replan_condition,
        }
