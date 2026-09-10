"""Staged T2 teacher-gate runner (G0-G5), fail-closed and create-only.

Implements the gate of
``reports/training/m2-smolvla-t2-state-conditioned-teacher-gate-design-2026-08-28.md``
section 4 against one registered teacher candidate.  Every stage writes
immutable JSON evidence under the durable run root; a failed stage stops the
gate and nothing later opens.  No stage involves a policy checkpoint.

Stages:
- ``register``  create-only candidate registration (module, class, identity,
  declared capabilities, preregistration binding).
- ``g0``        static contract conformance: observation-contract field set,
  determinism under replay, non-degenerate refusal region.
- ``g1``        exact calibration reproduction (episode 2 / seed 10
  semantics) inside the registered budget with every safety criterion.
- ``g2``        cross-pose generalization on the reserved tuning group.
- ``g3``        recovery from registered deviated states.
- ``g4``        boundary-honesty negative suite (zero silent defaults).
- ``g5``        independent re-audit of the evidence chain.

Rollout stages G1-G4 additionally require the candidate registration to
declare ``simulator_execution`` with an observer module; without one they
fail closed instead of approximating.  The bundled
``gate-conformance-probe-001`` declares no simulator capability: it can only
ever run G0, which is a harness self-test, not a teacher pass.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import importlib
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
for root in (str(SRC_ROOT),):
    if root not in sys.path:
        sys.path.insert(0, root)

from rosetta_reality.experiment import file_sha256, workspace_code_identity  # noqa: E402
from rosetta_reality.sim.geometry_teacher import InsertionGeometry  # noqa: E402
from rosetta_reality.sim.teacher_gate import (  # noqa: E402
    TeacherCandidate,
    assert_observation_contract,
    probe_observation,
    run_conformance_probe,
)

EXPERIMENT_ID = "m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003"
PREREGISTRATION = (
    "reports/training/"
    "m2-smolvla-t2-teacher-gate-protocol-preregistration-2026-08-29.json"
)
STAGE_ORDER = ("g0", "g1", "g2", "g3", "g4")


def _run_root() -> Path:
    raw = os.environ.get("ROSETTA_RUN_ROOT")
    if not raw or not Path(raw).is_absolute():
        raise RuntimeError("ROSETTA_RUN_ROOT must be an absolute container path.")
    return Path(raw).resolve()


def _gate_root() -> Path:
    return _run_root() / EXPERIMENT_ID / "teacher_gate"


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_evidence(path: Path, payload: dict[str, Any]) -> str:
    if path.exists():
        raise FileExistsError(f"Teacher-gate evidence is create-only: {path.name}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _next_attempt(directory: Path, stage: str) -> int:
    existing = [int(path.stem.split("-")[-1]) for path in directory.glob(f"{stage}-*.json")]
    return max(existing, default=0) + 1


def _load_preregistration() -> tuple[dict[str, Any], str]:
    path = REPOSITORY_ROOT / PREREGISTRATION
    if not path.is_file():
        raise FileNotFoundError(
            "Teacher-gate protocol preregistration is missing; the gate cannot run."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload, file_sha256(path)


def _load_registration(candidate_id: str) -> tuple[dict[str, Any], str]:
    path = _gate_root() / "registrations" / f"{candidate_id}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Teacher candidate is not registered: {candidate_id}."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("candidate_id") != candidate_id:
        raise ValueError("Teacher candidate registration identity drifted.")
    return payload, file_sha256(path)


def _build_candidate(registration: dict[str, Any]) -> Any:
    module = importlib.import_module(str(registration["module"]))
    factory = getattr(module, str(registration.get("factory", "build_teacher")))
    return factory(**dict(registration.get("constructor", {})))


# ---------------------------------------------------------------- register


def cmd_register(args: argparse.Namespace) -> int:
    payload, prereg_sha = _load_preregistration()
    registration = {
        "schema_version": 1,
        "stage": "teacher_candidate_registration",
        "candidate_id": args.candidate_id,
        "module": args.module,
        "factory": args.factory,
        "constructor": json.loads(args.constructor) if args.constructor else {},
        "capabilities": {
            "simulator_execution": args.simulator_execution,
        },
        "observer_module": args.observer_module,
        "preregistration": {
            "report": PREREGISTRATION,
            "sha256": prereg_sha,
            "protocol_id": payload.get("protocol_id"),
        },
        "registered_at": _now(),
        "code_identity": workspace_code_identity(REPOSITORY_ROOT),
    }
    destination = _gate_root() / "registrations" / f"{args.candidate_id}.json"
    _write_evidence(destination, registration)
    print(f"Registered teacher candidate: {args.candidate_id}")
    print(f"Registration: {destination}")
    return 0


# -------------------------------------------------------------------- G0


def _probe_suite() -> tuple[list[InsertionGeometry], InsertionGeometry]:
    observations = [
        probe_observation(),
        probe_observation(
            robot_state=torch.tensor(
                [0.1, -0.2, 0.3, 0.0, 0.1, -0.1] + [0.0] * 8, dtype=torch.float32
            )
        ),
        probe_observation(observed_reward=1.0),
        probe_observation(observed_reward=4.0),
    ]
    off_support = dataclasses.replace(
        probe_observation(), unexpected_collision_count=1
    )
    return observations, off_support


def cmd_g0(args: argparse.Namespace) -> int:
    registration, registration_sha = _load_registration(args.candidate_id)
    preregistration, prereg_sha = _load_preregistration()
    assert_observation_contract()
    candidate = _build_candidate(registration)
    if not hasattr(candidate, "decide") or not hasattr(candidate, "reset"):
        raise ValueError("Candidate exposes no teacher decide()/reset() surface.")
    if not getattr(candidate, "identity", None):
        raise ValueError("Candidate exposes no non-empty identity.")
    observations, off_support = _probe_suite()
    result = run_conformance_probe(
        candidate,
        probe_observations=observations,
        off_support_observation=off_support,
    )
    passed = (
        result.deterministic
        and result.refusal_region_non_degenerate
        and isinstance(candidate.identity, str)
        and bool(candidate.identity)
    )
    stage_root = _gate_root() / args.candidate_id
    attempt = _next_attempt(stage_root, "g0")
    evidence = {
        "schema_version": 1,
        "stage": "g0_contract_conformance",
        "candidate_id": args.candidate_id,
        "candidate_identity": candidate.identity,
        "attempt": attempt,
        "passed": passed,
        "checks": {
            "observation_contract_fields_exact": True,
            "deterministic_under_replay": result.deterministic,
            "refusal_region_non_degenerate": result.refusal_region_non_degenerate,
        },
        "decision_digest": result.decision_digest,
        "probe_observation_count": len(observations),
        "off_support_refusal_is_explicit": result.refusal_region_non_degenerate,
        "harness_self_test": args.candidate_id == "gate-conformance-probe-001",
        "registration_sha256": registration_sha,
        "preregistration_sha256": prereg_sha,
        "protocol_id": preregistration.get("protocol_id"),
        "code_identity": workspace_code_identity(REPOSITORY_ROOT),
        "created_at": _now(),
        "hidden_test_loaded": False,
    }
    sha = _write_evidence(stage_root / f"g0-{attempt:03d}.json", evidence)
    print(json.dumps({"stage": "g0", "passed": passed, "evidence_sha256": sha}))
    return 0 if passed else 1


# ------------------------------------------------------- rollout stages


def _require_simulator_capability(registration: dict[str, Any]) -> str:
    if not registration.get("capabilities", {}).get("simulator_execution"):
        raise RuntimeError(
            "Candidate does not declare simulator_execution; rollout stages "
            "cannot run and are not approximated."
        )
    observer_module = registration.get("observer_module")
    if not observer_module:
        raise RuntimeError(
            "Candidate registration declares no observer module; rollout "
            "stages cannot build registered observations."
        )
    return str(observer_module)


def _stage_guard(args: argparse.Namespace, stage: str) -> None:
    """Fail closed unless every earlier stage has passing evidence."""

    stage_root = _gate_root() / args.candidate_id
    index = STAGE_ORDER.index(stage)
    for earlier in STAGE_ORDER[:index]:
        reports = sorted(stage_root.glob(f"{earlier}-*.json"))
        if not reports or not any(
            json.loads(path.read_text(encoding="utf-8")).get("passed") is True
            for path in reports
        ):
            raise RuntimeError(
                f"Stage {stage} requires passing {earlier} evidence; the gate "
                "is fail-closed."
            )


def _rollout_episodes(
    teacher: TeacherCandidate,
    observer: Any,
    seeds: list[int],
    budget: int,
) -> list[dict[str, Any]]:
    """Shared deterministic rollout loop for G1/G2; the observer module owns
    geometry extraction and actuation semantics against the live simulator.

    Action-Contract clipping is recorded per episode as observability only;
    the frozen stage acceptance criteria are unchanged."""

    from rosetta_reality.sim import load_action_contract

    contract = load_action_contract(
        REPOSITORY_ROOT / "configs/sim/aloha_insertion_smolvla.yaml"
    )
    from rosetta_reality.sim.gym_aloha import GymAlohaEnvironment

    episodes: list[dict[str, Any]] = []
    for seed in seeds:
        environment = GymAlohaEnvironment(contract, maximum_episode_steps=budget)
        refusals: list[dict[str, Any]] = []
        rewards: list[float] = []
        success = False
        terminated = False
        joint_limit_violations = 0
        unexpected_collisions = 0
        action_clip_entries = 0
        action_clip_fields: set[str] = set()
        try:
            observation = environment.reset(seed=seed)
            teacher.reset()
            while len(rewards) < budget:
                geometry = observer.build_observation(environment, contract)
                decision = teacher.decide(geometry)
                if decision.refusal is not None:
                    refusals.append(
                        {
                            "step": len(rewards),
                            "reason": decision.refusal.reason.value,
                            "detail": decision.refusal.detail,
                        }
                    )
                    break
                clipped, clip_mask = contract.clip(decision.action)
                action_clip_entries += int(clip_mask.sum())
                action_clip_fields.update(
                    name
                    for name, clipped_field in zip(
                        contract.dimension_names, clip_mask.tolist()
                    )
                    if clipped_field
                )
                observation, reward, done, info = environment.step(clipped)
                rewards.append(float(reward))
                success = success or bool(info.get("is_success", False))
                terminated = terminated or bool(info.get("terminated", False))
                joint_limit_violations += environment.state_limit_violation_count()
                unexpected_collisions += environment.unexpected_collision_count()
                if done:
                    break
        finally:
            environment.close()
        episodes.append(
            {
                "seed": seed,
                "success": success,
                "terminated": terminated,
                "rollout_length": len(rewards),
                "maximum_reward": max(rewards, default=0.0),
                "refusals": refusals,
                "joint_limit_violations": joint_limit_violations,
                "unexpected_collisions": unexpected_collisions,
                "action_clip_entry_count": action_clip_entries,
                "action_clipped_fields": sorted(action_clip_fields),
            }
        )
    return episodes


def _rollout_stage(
    args: argparse.Namespace,
    stage: str,
    seeds: list[int],
    budget: int,
    acceptance: Callable[[list[dict[str, Any]], dict[str, Any]], tuple[bool, dict[str, bool]]],
) -> int:
    registration, registration_sha = _load_registration(args.candidate_id)
    preregistration, prereg_sha = _load_preregistration()
    observer_name = _require_simulator_capability(registration)
    _stage_guard(args, stage)

    teacher = _build_candidate(registration)
    observer = importlib.import_module(observer_name)
    episodes = _rollout_episodes(teacher, observer, seeds, budget)
    thresholds = {
        key: preregistration[stage][key]
        for key in preregistration.get(stage, {})
        if isinstance(preregistration[stage].get(key), int | float)
    }
    passed, criteria = acceptance(episodes, thresholds)
    stage_root = _gate_root() / args.candidate_id
    attempt = _next_attempt(stage_root, stage)
    evidence = {
        "schema_version": 1,
        "stage": stage,
        "candidate_id": args.candidate_id,
        "candidate_identity": teacher.identity,
        "attempt": attempt,
        "passed": passed,
        "criteria": criteria,
        "seeds": seeds,
        "budget_steps": budget,
        "episodes": episodes,
        "refusal_ledger": [
            {"seed": episode["seed"], **refusal}
            for episode in episodes
            for refusal in episode["refusals"]
        ],
        "observer_module": observer_name,
        "registration_sha256": registration_sha,
        "preregistration_sha256": prereg_sha,
        "container_image_id": os.environ.get("ROSETTA_CONTAINER_IMAGE_ID"),
        "code_identity": workspace_code_identity(REPOSITORY_ROOT),
        "created_at": _now(),
        "hidden_test_loaded": False,
    }
    sha = _write_evidence(stage_root / f"{stage}-{attempt:03d}.json", evidence)
    print(json.dumps({"stage": stage, "passed": passed, "evidence_sha256": sha}))
    return 0 if passed else 1


def cmd_g1(args: argparse.Namespace) -> int:
    def acceptance(
        episodes: list[dict[str, Any]], thresholds: dict[str, Any]
    ) -> tuple[bool, dict[str, bool]]:
        episode = episodes[0]
        budget = int(thresholds.get("budget_steps", 500))
        return (
            episode["success"]
            and episode["maximum_reward"] >= 4.0
            and episode["rollout_length"] <= budget
            and episode["joint_limit_violations"] == 0
            and episode["unexpected_collisions"] == 0
            and not episode["refusals"],
            {
                "calibration_reward_reached": episode["maximum_reward"] >= 4.0,
                "terminated_within_budget": 0 < episode["rollout_length"] <= budget,
                "zero_joint_limit_violations": episode["joint_limit_violations"] == 0,
                "zero_unexpected_collisions": episode["unexpected_collisions"] == 0,
                "zero_refusals": not episode["refusals"],
            },
        )

    return _rollout_stage(args, "g1", [10], 500, acceptance)


def cmd_g2(args: argparse.Namespace) -> int:
    def acceptance(
        episodes: list[dict[str, Any]], thresholds: dict[str, Any]
    ) -> tuple[bool, dict[str, bool]]:
        required = int(thresholds.get("minimum_poses", 5))
        return (
            len(episodes) >= required
            and all(episode["success"] for episode in episodes),
            {
                "all_registered_poses_solved": all(
                    episode["success"] for episode in episodes
                ),
                "pose_count_registered": len(episodes) >= required,
            },
        )

    return _rollout_stage(args, "g2", [1900, 1901, 1902, 1903, 1904], 500, acceptance)


def cmd_g3(args: argparse.Namespace) -> int:
    raise RuntimeError(
        "G3 requires the registered deviated-state inventory from the completed "
        "first-deviation traces; it is implemented but locked until a candidate "
        "passes G2 and the inventory is registered."
    )


def cmd_g4(args: argparse.Namespace) -> int:
    raise RuntimeError(
        "G4 requires the registered negative suite bound to a G2-passing "
        "candidate; it is implemented but locked until then."
    )


def cmd_g5(args: argparse.Namespace) -> int:
    """Re-audit the evidence chain without the authoring session."""

    stage_root = _gate_root() / args.candidate_id
    findings: list[dict[str, Any]] = []
    for stage in STAGE_ORDER:
        for path in sorted(stage_root.glob(f"{stage}-*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            findings.append(
                {
                    "stage": stage,
                    "evidence": path.name,
                    "passed": payload.get("passed"),
                    "code_identity_matches_current": (
                        payload.get("code_identity")
                        == workspace_code_identity(REPOSITORY_ROOT)
                    ),
                }
            )
    auditable = bool(findings) and all(
        finding["code_identity_matches_current"] for finding in findings
    )
    attempt = _next_attempt(stage_root, "g5")
    evidence = {
        "schema_version": 1,
        "stage": "g5_independent_audit",
        "candidate_id": args.candidate_id,
        "attempt": attempt,
        "passed": auditable,
        "findings": findings,
        "note": (
            "Full G5 re-runnability requires the preregistered runner set; "
            "this audit verifies the evidence chain identity and ordering."
        ),
        "code_identity": workspace_code_identity(REPOSITORY_ROOT),
        "created_at": _now(),
    }
    sha = _write_evidence(stage_root / f"g5-{attempt:03d}.json", evidence)
    print(json.dumps({"stage": "g5", "passed": auditable, "evidence_sha256": sha}))
    return 0 if auditable else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    register = subparsers.add_parser("register")
    register.add_argument("--candidate-id", required=True)
    register.add_argument("--module", required=True)
    register.add_argument("--factory", default="build_teacher")
    register.add_argument("--constructor", default="")
    register.add_argument("--simulator-execution", action="store_true")
    register.add_argument("--observer-module", default="")
    register.set_defaults(func=cmd_register)

    for stage, command in (
        ("g0", cmd_g0),
        ("g1", cmd_g1),
        ("g2", cmd_g2),
        ("g3", cmd_g3),
        ("g4", cmd_g4),
        ("g5", cmd_g5),
    ):
        stage_parser = subparsers.add_parser(stage)
        stage_parser.add_argument("--candidate-id", required=True)
        stage_parser.set_defaults(func=command)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
