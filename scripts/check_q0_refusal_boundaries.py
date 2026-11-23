"""Synthetic engineering input-boundary ledger; never a G4/provider verdict.

No simulator step, weights, datasets, or real IK. Run only in the registered
Linux container. Each case is reconstructed twice with a fresh candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from rosetta_reality.sim.coupled_translation_teacher import (
    CANDIDATE_ID, CoupledTranslationInsertionTeacher,
)
from rosetta_reality.sim.geometry_teacher import GeometryPose, InsertionGeometry
from rosetta_reality.sim.scripted_teacher import ScriptedTeacherSettings
from rosetta_reality.sim.teacher_gate import probe_observation

SOURCE_FILES = (
    "scripts/check_q0_refusal_boundaries.py",
    "src/rosetta_reality/sim/teacher_gate.py",
    "src/rosetta_reality/sim/geometry_teacher.py",
    "src/rosetta_reality/sim/scripted_teacher.py",
    "src/rosetta_reality/sim/geometric_teacher.py",
    "src/rosetta_reality/sim/coupled_translation_teacher.py",
)


@dataclass(frozen=True)
class Boundary:
    identity: str
    layer: str
    expected: str
    source: str


# Explicit pre-registration, not a retrospective classification or G4 extension.
BOUNDARIES = (
    Boundary("state_nan", "construction", "ValueError", "InsertionGeometry._finite_vector"),
    Boundary("state_shape", "construction", "ValueError", "InsertionGeometry._finite_vector"),
    Boundary("reward_nan", "construction", "ValueError", "InsertionGeometry.__post_init__"),
    Boundary("position_nan", "construction", "ValueError", "GeometryPose._finite_vector"),
    Boundary("position_shape", "construction", "ValueError", "GeometryPose._finite_vector"),
    Boundary("quaternion_shape", "construction", "ValueError", "GeometryPose._normalized_quaternion"),
    Boundary("quaternion_zero", "construction", "ValueError", "GeometryPose._normalized_quaternion"),
    Boundary("missing_robot_state", "construction", "TypeError", "InsertionGeometry constructor signature"),
    Boundary("negative_collision_count", "construction", "ValueError", "InsertionGeometry.__post_init__"),
    Boundary("reported_collision", "decision", "unsafe_state", "ScriptedInsertionTeacher._validate_support"),
    *(Boundary(f"workspace_{axis}_{side}", "decision", "workspace_violation",
               "ScriptedInsertionTeacher._validate_support")
      for axis in "xyz" for side in ("lower", "upper")),
)

# Class alone is insufficient: unrelated ValueError/TypeError must fail too.
CONSTRUCTION_ERROR_SIGNATURES = {
    "state_nan": "Insertion robot state contains NaN or Inf.",
    "state_shape": "Insertion robot state must have shape (14,),",
    "reward_nan": "Observed insertion reward must be finite.",
    "position_nan": "Pose position contains NaN or Inf.",
    "position_shape": "Pose position must have shape (3,),",
    "quaternion_shape": "Pose quaternion must have shape (4,),",
    "quaternion_zero": "Pose quaternion must have non-zero finite norm.",
    "missing_robot_state": "InsertionGeometry.__init__() missing 1 required positional argument: 'robot_state'",
    "negative_collision_count": "Unexpected collision count must be non-negative.",
}


class NoSolveStub:
    """All valid controls stay in the existing initial HOLD branch."""
    def solve(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("boundary diagnostic unexpectedly reached IK")


def inventory() -> list[dict[str, Any]]:
    return [dict(asdict(boundary), case_id=f"{boundary.identity}:{polarity}",
                 polarity=polarity, expected_kind=("action" if polarity == "positive"
                 else "clear_exception" if boundary.layer == "construction" else "refusal"))
            for boundary in BOUNDARIES for polarity in ("positive", "negative")]


def build_observation(identity: str, negative: bool) -> InsertionGeometry:
    base = probe_observation()
    values = {name: getattr(base, name) for name in base.__dataclass_fields__}
    if identity.startswith("workspace_"):
        _, axis, side = identity.split("_")
        settings = ScriptedTeacherSettings()
        bound = getattr(settings, f"workspace_{axis}_{'min' if side == 'lower' else 'max'}_m")
        # GeometryPose uses float32; keep the positive 1e-6 m inside to avoid
        # rounding e.g. 0.55 upward across the unchanged Python-float bound.
        position = base.socket.position.clone() if side == "lower" else base.peg.position.clone()
        offset = (-0.01 if side == "lower" else 0.01) if negative else (1e-6 if side == "lower" else -1e-6)
        position["xyz".index(axis)] = bound + offset
        values["socket" if side == "lower" else "peg"] = GeometryPose(
            position=position, quaternion=base.socket.quaternion)
    elif negative:
        if identity == "state_nan":
            values["robot_state"] = base.robot_state.clone()
            values["robot_state"][0] = float("nan")
        elif identity == "state_shape":
            values["robot_state"] = torch.zeros(13)
        elif identity == "reward_nan":
            values["observed_reward"] = float("nan")
        elif identity == "missing_robot_state":
            del values["robot_state"]
        elif identity == "negative_collision_count":
            values["unexpected_collision_count"] = -1
        elif identity == "reported_collision":
            values["unexpected_collision_count"] = 1
        else:
            position, quaternion = base.socket.position.clone(), base.socket.quaternion.clone()
            if identity == "position_nan":
                position[0] = float("nan")
            elif identity == "position_shape":
                position = torch.zeros(2)
            elif identity == "quaternion_shape":
                quaternion = torch.zeros(3)
            elif identity == "quaternion_zero":
                quaternion = torch.zeros(4)
            else:
                raise AssertionError("unknown registered case")
            values["socket"] = GeometryPose(position=position, quaternion=quaternion)
    return InsertionGeometry(**values)


def observe(boundary: Boundary, negative: bool) -> dict[str, Any]:
    layer = "construction"
    try:
        observation = build_observation(boundary.identity, negative)
        layer = "candidate_construction"
        teacher = CoupledTranslationInsertionTeacher(solver=NoSolveStub())
        layer = "decision"
        decision = teacher.decide(observation)
        if decision.refusal is not None:
            return {"kind": "refusal", "layer": layer, "reason": decision.refusal.reason.value}
        action = decision.action
        if action is None or action.shape != (14,) or not bool(torch.isfinite(action).all()):
            return {"kind": "invalid_action", "layer": layer}
        return {"kind": "action", "layer": layer, "action": action.tolist()}
    except Exception as error:
        # Never persist arbitrary exception messages/tracebacks or host paths.
        expected = (negative and boundary.layer == "construction" and layer == "construction"
                    and type(error).__name__ == boundary.expected
                    and type(error) in (ValueError, TypeError)
                    and str(error).startswith(CONSTRUCTION_ERROR_SIGNATURES[boundary.identity]))
        return {"kind": "clear_exception" if expected else "unexpected_exception",
                "layer": layer, "exception_type": type(error).__name__}


def run_diagnostic(*, max_seconds: float = 120.0) -> dict[str, Any]:
    if not 0 < max_seconds <= 120:
        raise ValueError("budget must be within (0, 120] seconds")
    start = time.monotonic()
    rows = []
    for boundary in BOUNDARIES:
        for negative in (False, True):
            if time.monotonic() - start >= max_seconds:
                raise TimeoutError("registered refusal-boundary budget exceeded")
            outcomes = [observe(boundary, negative) for _ in range(2)]
            kind = "action" if not negative else "clear_exception" if boundary.layer == "construction" else "refusal"
            expected = all(outcome["kind"] == kind and
                           (kind != "refusal" or outcome.get("reason") == boundary.expected)
                           for outcome in outcomes)
            rows.append({"case_id": f"{boundary.identity}:{'negative' if negative else 'positive'}",
                         "outcomes": outcomes, "deterministic": outcomes[0] == outcomes[1],
                         "expected_boundary_observed": expected,
                         "silent_action_on_negative": negative and any(o["kind"] == "action" for o in outcomes)})
    elapsed = time.monotonic() - start
    if elapsed > max_seconds:
        raise TimeoutError("registered refusal-boundary budget exceeded")
    manifest = inventory()
    checks_passed = all(
        r["deterministic"] and r["expected_boundary_observed"] for r in rows)
    return {"schema": "rosetta.q0_refusal_boundaries.v1", "candidate_id": CANDIDATE_ID,
            "status": "passed" if checks_passed else "check_failed",
            "infrastructure_failure": False,
            "evidence_kind": "synthetic_engineering_diagnostic", "provider_qualified": False,
            "formal_gate_verdict": None, "candidate_g1_passed": False,
            "solver": "NoSolveStub; IK must not be reached; real IK unverified",
            "simulation_steps": 0, "model_loads": 0, "dataset_loads": 0,
            "inventory": manifest,
            "inventory_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
            "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES},
            "budget": {"max_cases": 32, "replicates_per_case": 2, "max_decisions": 64, "max_seconds": max_seconds},
            "elapsed_seconds": elapsed, "cases": rows,
            "engineering_checks_passed": checks_passed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new JSON ledger; existing file is refused")
    parser.add_argument("--max-seconds", type=float, default=120.0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new file")
    result = run_diagnostic(max_seconds=args.max_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({"engineering_checks_passed": result["engineering_checks_passed"],
                      "provider_qualified": False, "case_count": len(result["cases"])}))
    return 0 if result["engineering_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
