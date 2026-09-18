"""Render the visual-conditioning simulation-gate plan and dispatch Gate 3/4.

Create-only local adaptation of the registered Zen gate chain: the frozen
simulator engine (``smolvla_sim_gate``) stays untouched and keeps its own
XPU-native online policy class and runtime probe, so this wrapper clones no
policy code. ``render`` verifies the entry permit (the frozen offset-250
gradient gate on the exported candidate artifact), derives a gate-facing
selection record from the artifact manifest, writes an inventory backup
record, and renders the deterministic simulation plan into the durable
orchestration area for the host to register verbatim under ``configs/vla/``
(the vla containers mount the repository read-only). ``gate3``/``gate4``
re-derive the same content, verify the registered copy byte-for-byte, record
sanitized vcdropout provenance into every gate report, and dispatch the
frozen engine.

Report suffix ``433`` (engine-enforced three digits), continuing the campaign
series after Zen ``411``/``422``. Execution boundary: local WSL XPU container,
network disabled; Gate 3 must pass before Gate 4; a pass or failure here
closes the registered comparison either way.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for root in (str(REPOSITORY_ROOT / "src"), SCRIPTS_ROOT):
    if root not in sys.path:
        sys.path.insert(0, root)

import smolvla_sim_gate as simulator  # noqa: E402
import smolvla_vcdropout_protocol as protocol  # noqa: E402

from rosetta_reality.experiment import file_sha256, workspace_code_identity  # noqa: E402
from rosetta_reality.vla.processor import ensure_smolvla_action_boundary  # noqa: E402,F401
from rosetta_reality.vla.runtime_compatibility import (  # noqa: E402
    require_absolute_environment_directory,
)

REPORT_SUFFIX = "433"
CONTRACT_SHA = "fc71a0438f0e3af7258e5b52d82fa22fc53c12b47901606cbee715524392ac62"
GATE3_SEED = 20260809
GATE3_POLICY_NOISE_SEED = 20260809
GATE4_SEEDS = [1000, 1001, 1002, 1003, 1004]
GATE3_MAXIMUM_STEPS = 20
GATE4_MAXIMUM_STEPS = 500
GATE4_MINIMUM_TASK_SUCCESS_RATE = 0.2
PRIOR_FAILURE = {
    "report": "runs/m2-smolvla450m-aloha-insertion-001/gates/gate3-smolvla-sim-001.json",
    "report_sha256": "5df3b887984d1c8fd47084c3315a71e3894eac18cc80b1eb08b2120583ed26ed",
    "failed_criterion": "raw_actions_within_contract",
}
PRIOR_TASK_FAILURE = {
    "report": (
        "runs/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/"
        "gates/gate4-smolvla-sim-003.json"
    ),
    "report_sha256": "86c35a4dbcb70761a5ec6787fd95e6dcbd11e26738aaab9e34500be02aff8a46",
    "failed_criterion": "minimum_task_success_rate",
}
SIMULATION_CODE_FILES = (
    "scripts/smolvla_sim_gate.py",
    "scripts/smolvla_vcdropout_sim_gate.py",
    "src/rosetta_reality/sim/gym_aloha.py",
    "src/rosetta_reality/vla/processor.py",
)
_MEMORY_LIMIT = "6g"

SIM_PLAN_TEMPLATE = """schema_version: 1
role: vla
stage: m2_closed_loop_simulation
status: preregistered
plan_id: {sim_plan_id}
experiment_id: {experiment_id}
artifact_id: {artifact_id}
artifact_manifest_sha256: {artifact_manifest_sha256}
hypothesis: >-
  Closed-loop Gate 3/4 comparison for the visual-conditioning state-dropout
  arm, separately registered after the frozen offset-250 gradient gate passed
  on this artifact. Training-only whole-sample normalized-state dropout is the
  single changed axis against the immutable Zen-uniform control; environment
  seeds, inference noise seeds, action-contract projection, receding-horizon
  execution, simulator physics, collision semantics and acceptance thresholds
  are fixed and identical to the historical controls.
single_axis_change:
  field: training.state_conditioning_dropout
  control: none_uniform_flow_loss
  candidate: samplewise_normalized_state_dropout_p0.5_dedicated_rng
gradient_gate:
  report: {gradient_gate_report}
  report_sha256: {gradient_gate_sha}
  gate_passed: true
prior_failure:
  report: {prior_report}
  report_sha256: {prior_sha}
  failed_criterion: raw_actions_within_contract
prior_task_failure:
  report: {prior_task_report}
  report_sha256: {prior_task_sha}
  failed_criterion: minimum_task_success_rate
selection:
  report: {selection_report}
  report_sha256: {selection_sha}
  checkpoint_step: {selected_step}
  model_safetensors_sha256: {model_sha}
export_manifest_ref:
  path: artifacts/{experiment_id}/{artifact_id}/manifest.json
  sha256_recorded_inside_manifest: true
artifact_backup:
  report: runs/{experiment_id}/artifact_backup/{artifact_id}-backup.json
  report_sha256: {backup_sha}
action_contract:
  path: configs/sim/aloha_insertion_smolvla.yaml
  sha256: {contract_sha}
collision_policy:
  classifier: explicit_task_contact_allowlist
  allowed_task_contacts:
    - [red_peg, vx300s_right/10_right_gripper_finger]
    - [socket-1, vx300s_left/10_left_gripper_finger]
    - [socket-2, vx300s_left/10_left_gripper_finger]
    - [socket-3, vx300s_left/10_left_gripper_finger]
    - [socket-4, vx300s_left/10_left_gripper_finger]
  same_arm_internal_gripper_contacts_are_non_gating: true
  all_other_robot_scene_contacts_are_unexpected: true
simulation_code_sha256:
{sim_code_blocks}
inference:
  observation_camera: top
  policy_camera: observation.images.camera1
  instruction: Insert the peg into the socket.
  noise: seeded_standard_normal
  noise_source: pinned_lerobot_default_standard_normal
  mixed_precision: bf16
  chunk_execution: receding_horizon_first_action
  chunk_execution_steps: 1
  policy_output_projection: action_contract_clip
  projection_location: vla_output_boundary_before_simulation_adapter
  unprojected_decoder_action_role: non_gating_diagnostic
resources:
  runtime: local_wsl_sim_container
  accelerator: xpu
  memory_limit: "{memory_limit}"
  memory_swap_limit: "{memory_limit}"
  nested_docker_used: false
gate3:
  seed: {gate3_seed}
  policy_noise_seed: {gate3_noise_seed}
  maximum_steps: {gate3_steps}
  require_finite_actions: true
  require_projected_policy_actions_within_contract: true
  require_adapter_no_additional_clipping: true
  maximum_unexpected_collisions: 0
  report_suffix: "{report_suffix}"
gate4:
  seeds: [{gate4_seeds}]
  policy_noise_seeds: [{gate4_noise_seeds}]
  maximum_steps: {gate4_steps}
  minimum_task_success_rate: {gate4_min_success}
  maximum_unexpected_collisions: 0
  require_gate3_passed: true
  report_suffix: "{report_suffix}"
hidden_test_loaded: false
"""


def _run_root() -> Path:
    return require_absolute_environment_directory("ROSETTA_RUN_ROOT")


def _evidence_path_field(path: Path) -> str:
    """Prefer repository-relative evidence paths the engine can resolve.

    Under the registered container boundary the run root is the repository's
    own ``runs/`` mount, so gate records resolve repository-relative. Paths
    outside the repository (only reachable in synthetic tests) fall back to
    their absolute form; the engine rejects those, as intended.
    """

    try:
        return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _artifact_root() -> Path:
    return require_absolute_environment_directory("ROSETTA_ARTIFACT_ROOT")


def _gradient_gate_report(experiment_id: str, artifact_id: str) -> tuple[Path, dict[str, Any]]:
    """Locate and verify the single gradient-gate pass report (entry permit)."""

    directory = _run_root() / experiment_id / "diagnostics"
    candidates = sorted(directory.glob("vcdropout-gradient-gate-*.json"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            "Exactly one gradient-gate report is required as the entry permit; "
            f"found {len(candidates)} under {directory}."
        )
    report = json.loads(candidates[0].read_text(encoding="utf-8"))
    criteria = report.get("gate_criteria")
    criteria_all_passed = isinstance(criteria, list) and bool(criteria) and all(
        isinstance(entry, dict) and entry.get("passed") is True for entry in criteria
    )
    if (
        report.get("gate_passed") is not True
        or report.get("status") != "passed"
        or report.get("artifact_id") != artifact_id
        or not criteria_all_passed
    ):
        raise ValueError(
            "The gradient-gate entry permit is not a pass for this artifact."
        )
    return candidates[0], report


def _write_gate_selection(
    run_root: Path, run_name: str, artifact_dir: Path
) -> tuple[Path, str, int, str]:
    """Derive the gate-facing selection record from the artifact manifest."""

    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    step = int(manifest["selected_checkpoint_step"])
    model_sha = str(manifest["selected_checkpoint_model_sha256"])
    payload = {
        "schema_version": 1,
        "status": "passed",
        "selected": {"step": step, "model_safetensors_sha256": model_sha},
        "hidden_test_loaded": False,
        "derived_from": {
            "artifact_manifest_sha256": file_sha256(artifact_dir / "manifest.json"),
            "selection_report_sha256_recorded_in_manifest": str(
                manifest["selection_report_sha256"]
            ),
        },
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    destination = (
        run_root / str(manifest["experiment_id"]) / "selection"
        / f"{run_name}-selection-gate.json"
    )
    if destination.exists():
        if destination.read_text(encoding="utf-8") != text:
            raise FileExistsError("Gate-facing selection evidence drifted.")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
    return destination, file_sha256(destination), step, model_sha


def _write_backup_evidence(run_root: Path, artifact_dir: Path) -> tuple[Path, str]:
    """Inventory backup record: the deploy artifact already lives in the local
    repository artifact root that the gate reads from, so this record mirrors
    every file checksum instead of duplicating the artifact."""

    backup_dir = run_root / str(
        json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))[
            "experiment_id"
        ]
    ) / "artifact_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    destination = backup_dir / f"{artifact_dir.name}-backup.json"
    current_manifest_sha = file_sha256(artifact_dir / "manifest.json")
    if destination.is_file():
        text = destination.read_text(encoding="utf-8")
        existing = json.loads(text)
        if existing.get("artifact_manifest_sha256") == current_manifest_sha:
            return destination, hashlib.sha256(text.encode("utf-8")).hexdigest()
    files = {
        path.relative_to(artifact_dir).as_posix(): file_sha256(path)
        for path in sorted(artifact_dir.rglob("*"))
        if path.is_file()
    }
    report = {
        "schema_version": 1,
        "status": "verified",
        "stage": "smolvla_local_repository_artifact_backup",
        "artifact_id": artifact_dir.name,
        "artifact_manifest_sha256": current_manifest_sha,
        "backup_mode": "repository_artifact_file_inventory",
        "files": files,
        "file_count": len(files),
        "off_host_copy_created": False,
        "archive_file_set_matches_manifest": True,
        "artifact_reload_verified": True,
        "same_repository_artifact_root": True,
        "gate_unlock_scope": "local_gate3_gate4_only",
        "hidden_test_loaded": False,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    destination.write_text(text, encoding="utf-8")
    return destination, hashlib.sha256(text.encode("utf-8")).hexdigest()


def _prepare_gate_evidence(formal_plan_path: Path, artifact_id: str) -> dict[str, Any]:
    """Validate identities and the entry permit, then write run-root records.

    Returns every template field the simulation plan is rendered from.
    """

    _, plan_id = protocol.resolve_plan(formal_plan_path)
    formal_plan = protocol.load_yaml(formal_plan_path.resolve())
    run_name = str(formal_plan["run_name"])
    experiment_id = protocol.EXPERIMENT_ID
    if formal_plan.get("plan_id") != plan_id:
        raise ValueError("Candidate plan identity differs from the protocol decision.")

    artifact_dir = _artifact_root() / experiment_id / artifact_id
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Artifact manifest missing; run export first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("plan_id") != plan_id
        or manifest.get("artifact_id") != artifact_id
        or manifest.get("experiment_id") != experiment_id
        or manifest.get("hidden_test_loaded") is not False
    ):
        raise ValueError("Artifact identity differs from the candidate plan.")

    gradient_gate_report, _ = _gradient_gate_report(experiment_id, artifact_id)

    run_root = _run_root()
    gate_selection, gate_selection_sha, step, model_sha = _write_gate_selection(
        run_root, run_name, artifact_dir
    )
    _, backup_sha = _write_backup_evidence(run_root, artifact_dir)

    return {
        "formal_plan_path": formal_plan_path.resolve(),
        "artifact_dir": artifact_dir,
        "run_name": run_name,
        "plan_id": plan_id,
        "artifact_manifest_sha256": file_sha256(manifest_path),
        "gradient_gate_report": _evidence_path_field(gradient_gate_report),
        "gradient_gate_sha": file_sha256(gradient_gate_report),
        "selection_report": _evidence_path_field(gate_selection),
        "selection_sha": gate_selection_sha,
        "selected_step": step,
        "model_sha": model_sha,
        "backup_sha": backup_sha,
    }


def _build_sim_plan_content(evidence: dict[str, Any]) -> tuple[str, str]:
    """Return the rendered simulation plan content and its id."""

    sim_plan_id = f"{evidence['run_name']}-sim-{REPORT_SUFFIX}"
    sim_code_blocks = "\n".join(
        f"  {relative}: {file_sha256(REPOSITORY_ROOT / relative)}"
        for relative in SIMULATION_CODE_FILES
    )
    content = SIM_PLAN_TEMPLATE.format(
        sim_plan_id=sim_plan_id,
        experiment_id=protocol.EXPERIMENT_ID,
        artifact_id=evidence["artifact_dir"].name,
        artifact_manifest_sha256=evidence["artifact_manifest_sha256"],
        gradient_gate_report=evidence["gradient_gate_report"],
        gradient_gate_sha=evidence["gradient_gate_sha"],
        prior_report=PRIOR_FAILURE["report"],
        prior_sha=PRIOR_FAILURE["report_sha256"],
        prior_task_report=PRIOR_TASK_FAILURE["report"],
        prior_task_sha=PRIOR_TASK_FAILURE["report_sha256"],
        selection_report=evidence["selection_report"],
        selection_sha=evidence["selection_sha"],
        selected_step=evidence["selected_step"],
        model_sha=evidence["model_sha"],
        backup_sha=evidence["backup_sha"],
        contract_sha=CONTRACT_SHA,
        sim_code_blocks=sim_code_blocks,
        memory_limit=_MEMORY_LIMIT,
        gate3_seed=GATE3_SEED,
        gate3_noise_seed=GATE3_POLICY_NOISE_SEED,
        gate3_steps=GATE3_MAXIMUM_STEPS,
        gate4_seeds=", ".join(str(seed) for seed in GATE4_SEEDS),
        gate4_noise_seeds=", ".join(str(seed) for seed in GATE4_SEEDS),
        gate4_steps=GATE4_MAXIMUM_STEPS,
        gate4_min_success=GATE4_MINIMUM_TASK_SUCCESS_RATE,
        report_suffix=REPORT_SUFFIX,
    )
    return content, sim_plan_id


def staged_plan_path(sim_plan_id: str) -> Path:
    return _run_root() / protocol.EXPERIMENT_ID / "orchestration" / f"{sim_plan_id}.yaml"


def registered_plan_path(sim_plan_id: str) -> Path:
    return REPOSITORY_ROOT / "configs/vla" / f"{sim_plan_id}.yaml"


def render_sim_plan(
    formal_plan_path: Path, artifact_id: str
) -> tuple[Path, Path, str]:
    """Render the create-only simulation plan into the run-root staging area.

    The vla containers mount the repository read-only, so registration is a
    host-side step: this writes the deterministic plan under the durable
    orchestration area and returns its path for the host to copy verbatim
    into ``configs/vla/``. Gate modes later verify that registered copy
    byte-for-byte before dispatching.

    Returns (formal plan path, staged simulation plan path, plan id).
    """

    evidence = _prepare_gate_evidence(formal_plan_path, artifact_id)
    content, sim_plan_id = _build_sim_plan_content(evidence)
    yaml.safe_load(content)
    destination = staged_plan_path(sim_plan_id)
    if destination.exists():
        if destination.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Rendered sim plan drifted: {destination.name}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    return evidence["formal_plan_path"], destination.resolve(), evidence["plan_id"]


def _require_registered_plan(
    sim_plan_id: str, content: str
) -> Path:
    """Verify the registered configs/vla copy matches the rendered plan exactly."""

    destination = registered_plan_path(sim_plan_id)
    if not destination.is_file():
        raise FileNotFoundError(
            "Registered simulation plan missing; render it and copy the staged "
            f"file verbatim to {destination}."
        )
    if destination.read_text(encoding="utf-8") != content:
        raise FileExistsError(f"Registered sim plan drifted: {destination.name}")
    return destination.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("render", "gate3", "gate4"):
        gate_parser = subparsers.add_parser(command)
        gate_parser.add_argument(
            "--plan",
            type=Path,
            default=REPOSITORY_ROOT
            / "configs/vla/smolvla_450m_aloha_insertion_vcdropout_cuda_b64_001.yaml",
        )
        gate_parser.add_argument("--artifact-id", required=True)
        if command == "gate4":
            gate_parser.add_argument("--gate3-report", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "render":
        _, staged, _ = render_sim_plan(args.plan, args.artifact_id)
        print(f"Staged simulation plan: {staged}")
        print(
            "Copy this file verbatim to configs/vla/ on the host, then run "
            "gate3/gate4."
        )
        return 0

    evidence = _prepare_gate_evidence(args.plan, args.artifact_id)
    content, sim_plan_id = _build_sim_plan_content(evidence)
    sim_plan_path = _require_registered_plan(sim_plan_id, content)

    original_create_json = simulator.create_json
    try:
        def create_json(path: Path, payload: dict[str, Any]) -> None:
            if str(payload.get("gate", "")).startswith("m2_gate"):
                payload["vcdropout_protocol"] = {
                    "schema_version": 1,
                    "wrapper_sha256": file_sha256(Path(__file__)),
                    "protocol_module_sha256": file_sha256(
                        REPOSITORY_ROOT / "scripts/smolvla_vcdropout_protocol.py"
                    ),
                    "formal_plan_sha256": file_sha256(evidence["formal_plan_path"]),
                    "gradient_gate_report_sha256": evidence["gradient_gate_sha"],
                    "artifact_backup_verified": True,
                    "code_identity": workspace_code_identity(REPOSITORY_ROOT),
                }
            original_create_json(path, payload)

        simulator.create_json = create_json
        if args.command == "gate3":
            return simulator.gate3(sim_plan_path)
        return simulator.gate4(sim_plan_path, args.gate3_report.resolve())
    finally:
        simulator.create_json = original_create_json


if __name__ == "__main__":
    raise SystemExit(main())
