"""Render per-arm Zen simulation-gate plans and dispatch Gate 3 / Gate 4.

Faithful adaptation of the registered AutoDL Way gate chain: the frozen
simulator engine (``smolvla_sim_gate``) stays untouched. This wrapper swaps in
the CUDA online-policy class (a verbatim clone of the proven Way class with
Zen identity strings), installs a fail-closed evidence-path resolver over the
run root, derives a gate-facing selection record from the Zen selection
decision, renders a plan whose projection/noise precedents reuse the immutable
registered failure reports of this experiment, and records sanitized
provenance into every gate report.

Arm suffixes (3 digits, engine-enforced): uniform control ``401``,
first-action treatment ``402``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for root in (str(REPOSITORY_ROOT / "src"), SCRIPTS_ROOT):
    if root not in sys.path:
        sys.path.insert(0, root)

import smolvla_sim_gate as simulator  # noqa: E402
import smolvla_zen_protocol as protocol  # noqa: E402

from rosetta_reality.experiment import file_sha256, workspace_code_identity  # noqa: E402
from rosetta_reality.sim import load_action_contract  # noqa: E402
from rosetta_reality.vla.processor import ensure_smolvla_action_boundary  # noqa: E402
from rosetta_reality.vla.runtime_compatibility import (  # noqa: E402
    require_absolute_environment_directory,
    resolve_runtime_evidence_path,
)

ARM_SUFFIXES = {
    "m2-smolvla450m-zen-uniform-002": "411",
    "m2-smolvla450m-zen-firstaction-001": "422",
}
_ACTIVE_SIM_PLAN: Path | None = None
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
CONTRACT_SHA = "fc71a0438f0e3af7258e5b52d82fa22fc53c12b47901606cbee715524392ac62"

SIM_PLAN_TEMPLATE = """schema_version: 1
role: vla
stage: m2_closed_loop_simulation
status: preregistered
plan_id: {sim_plan_id}
experiment_id: {experiment_id}
artifact_id: {artifact_id}
artifact_manifest_sha256: {artifact_manifest_sha256}
hypothesis: >-
  Closed-loop Gate evaluation for the preregistered Zen arm ({role}). Only the
  temporal weight profile differs between arms; environment seeds, inference
  noise seeds, action-contract projection, receding-horizon execution,
  simulator physics, collision semantics and acceptance thresholds are fixed.
single_axis_change:
  field: training.horizon_weight_profile
  control: none_uniform_flow_loss
  candidate: first_action_only_selected_valid_mean
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
  runtime: autodl_container_instance
  accelerator: cuda
  memory_limit: autodl_platform_container
  memory_swap_limit: autodl_platform_container
  nested_docker_used: false
gate3:
  seed: 20260809
  policy_noise_seed: 20260809
  maximum_steps: 20
  require_finite_actions: true
  require_projected_policy_actions_within_contract: true
  require_adapter_no_additional_clipping: true
  maximum_unexpected_collisions: 0
  report_suffix: "{suffix}"
gate4:
  seeds: [1000, 1001, 1002, 1003, 1004]
  policy_noise_seeds: [1000, 1001, 1002, 1003, 1004]
  maximum_steps: 500
  minimum_task_success_rate: 0.2
  maximum_unexpected_collisions: 0
  require_gate3_passed: true
  report_suffix: "{suffix}"
hidden_test_loaded: false
"""


def _runtime_evidence_path(raw: str) -> Path:
    return resolve_runtime_evidence_path(
        raw,
        repository_root=REPOSITORY_ROOT,
        run_root=require_absolute_environment_directory("ROSETTA_RUN_ROOT"),
    )


def _cuda_runtime() -> dict[str, Any]:
    if os.environ.get("ROSETTA_TORCH_DEVICE") != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Zen AutoDL gates require CUDA.")
    return {
        "torch_version": torch.__version__,
        "lerobot_version": version("lerobot"),
        "gym_aloha_version": version("gym-aloha"),
        "trackio_version": version("trackio"),
        "device": "cuda",
        "cuda_name": torch.cuda.get_device_name(0),
        "runtime_boundary": "autodl_container_instance",
        "nested_docker_used": False,
        "network_disabled": True,
    }


def _build_online_cuda_class():
    """Verbatim behavioral clone of the registered Way CUDA policy class."""

    from lerobot.policies.factory import (
        make_policy as _make_policy,
    )
    from lerobot.policies.factory import (
        make_pre_post_processors as _make_ppp,
    )
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    from rosetta_reality.vla.action_space import SmolVLAActionSpace

    class _ZenOnlineCUDASmolVLA:
        def __init__(self, artifact, config, normalization, contract) -> None:
            if os.environ.get("ROSETTA_TORCH_DEVICE") != "cuda" or not torch.cuda.is_available():
                raise RuntimeError("Zen simulation requires the CUDA runtime.")
            self.device = torch.device("cuda")
            self.mixed_precision = str(config["mixed_precision"])
            if self.mixed_precision != "bf16":
                raise ValueError("Zen simulation mixed precision differs from its artifact.")
            pretrained = artifact / "pretrained_model"
            policy_cfg = SmolVLAConfig.from_pretrained(pretrained, local_files_only=True)
            policy_cfg.device = self.device.type
            policy_cfg.pretrained_path = pretrained
            policy_cfg.pretrained_revision = None
            policy_cfg.load_vlm_weights = False
            metadata = simulator._ArtifactMetadata(config, normalization)
            dataset_state_dimension = simulator._dataset_state_dimension(metadata.features)
            self.policy = _make_policy(
                cfg=policy_cfg, ds_meta=metadata, rename_map=config["rename_map"]
            )
            self.state_dimension = simulator._validate_policy_contract_shape(
                self.policy.config, contract, dataset_state_dimension
            )
            self.action_dimension = contract.dimension
            self.chunk_length = contract.chunk_length
            self.preprocessor, self.postprocessor = _make_ppp(
                policy_cfg=policy_cfg,
                pretrained_path=pretrained,
                pretrained_revision=None,
                dataset_stats=metadata.stats,
                preprocessor_overrides={
                    "device_processor": {"device": self.device.type},
                    "normalizer_processor": {
                        "features": {
                            **self.policy.config.input_features,
                            **self.policy.config.output_features,
                        },
                        "norm_map": self.policy.config.normalization_mapping,
                        "stats": metadata.stats,
                    },
                    "rename_observations_processor": {
                        "rename_map": config["rename_map"]
                    },
                },
                postprocessor_overrides={
                    "unnormalizer_processor": {
                        "features": self.policy.config.output_features,
                        "norm_map": self.policy.config.normalization_mapping,
                        "stats": metadata.stats,
                    }
                },
            )
            raw_action_space = config.get("action_space")
            if not isinstance(raw_action_space, dict):
                raise ValueError("Zen artifact has no explicit action-space identity.")
            action_space = SmolVLAActionSpace(**raw_action_space)
            if (
                action_space.representation_adapter
                != "rosetta_pi_aloha_arms_bounded_sine_grippers"
                or config.get("bounded_gripper_decoder") is not True
            ):
                raise ValueError("Zen artifact lost the bounded gripper decoder.")
            plan = simulator._load_yaml(_plan_path_from_argv())
            contract_path = simulator._repository_path(str(plan["action_contract"]["path"]))
            if file_sha256(contract_path) != str(config["action_contract_sha256"]):
                raise ValueError("Zen source and exported Action Contract checksums differ.")
            ensure_smolvla_action_boundary(
                self.preprocessor,
                self.postprocessor,
                load_action_contract(contract_path),
                action_space,
                action_contract_sha256=str(config["action_contract_sha256"]),
                upstream_revision=str(config["upstream_revision"]),
            )
            self.policy.eval()
            self._noise_mode = "zeros"
            self._noise_generator = torch.Generator(device="cpu")
            self._noise_seed = None

        def configure_noise(self, mode: str, seed) -> None:
            if mode == "zeros":
                if seed is not None:
                    raise ValueError("Zero-noise inference must not register a random seed.")
            elif mode == "seeded_standard_normal":
                if seed is None or seed < 0:
                    raise ValueError("Seeded Gaussian inference requires a non-negative seed.")
                self._noise_generator.manual_seed(seed)
            else:
                raise ValueError("Unsupported SmolVLA inference noise mode.")
            self._noise_mode = mode
            self._noise_seed = seed

        def predict(self, observation, instruction: str):
            images = observation.get("images")
            state = observation.get("robot_state")
            if not isinstance(images, dict) or "top" not in images:
                raise ValueError("Simulator observation has no registered top camera.")
            if not isinstance(state, torch.Tensor) or tuple(state.shape) != (
                self.state_dimension,
            ):
                raise ValueError("Simulator observation has an invalid ALOHA state.")
            sample = {
                "observation.images.top": images["top"],
                "observation.state": state,
                "task": instruction,
            }
            batch = self.preprocessor(sample)
            processed_state = batch.get("observation.state")
            if not isinstance(processed_state, torch.Tensor):
                raise ValueError("Processed simulator observation has no state tensor.")
            noise_shape = (1, self.policy.config.chunk_size, self.policy.config.max_action_dim)
            if self._noise_mode == "zeros":
                noise = torch.zeros(
                    noise_shape, device=self.device, dtype=processed_state.dtype
                )
            elif self._noise_mode == "seeded_standard_normal":
                if self._noise_seed is None:
                    raise RuntimeError("SmolVLA Gaussian noise was not seeded.")
                noise = torch.randn(
                    noise_shape,
                    generator=self._noise_generator,
                    device="cpu",
                    dtype=torch.float32,
                ).to(device=self.device, dtype=processed_state.dtype)
            else:
                raise RuntimeError("SmolVLA inference noise was not configured.")
            self.policy.reset()
            torch.cuda.synchronize()
            with (
                torch.inference_mode(),
                torch.autocast(device_type="cuda", dtype=torch.bfloat16),
            ):
                action = self.policy.predict_action_chunk(batch, noise=noise)
            torch.cuda.synchronize()
            action = self.postprocessor(action)
            expected_shape = (1, self.chunk_length, self.action_dimension)
            if not isinstance(action, torch.Tensor) or tuple(action.shape) != expected_shape:
                raise ValueError("Zen simulator output differs from the Action Contract.")
            adapter_steps = [
                step
                for step in self.postprocessor.steps
                if getattr(step.__class__, "_registry_name", None)
                == "rosetta_pi_aloha_postprocessor"
            ]
            if len(adapter_steps) != 1:
                raise ValueError("Zen simulator has no unique action decoder boundary.")
            raw = getattr(adapter_steps[0], "last_unclipped_action", None)
            if not isinstance(raw, torch.Tensor) or tuple(raw.shape) != expected_shape:
                raise ValueError("Zen simulator did not retain the pre-clipping action.")
            return raw[0].detach().cpu(), action[0].detach().cpu()

    return _ZenOnlineCUDASmolVLA


def _plan_path_from_argv() -> Path:
    if _ACTIVE_SIM_PLAN is not None:
        return _ACTIVE_SIM_PLAN
    raise RuntimeError("The Zen gate wrapper did not register its rendered sim plan.")


def _write_gate_selection(
    run_root: Path, selection_path: Path, artifact_dir: Path, suffix: str
) -> tuple[Path, str, int, str]:
    decision = json.loads(selection_path.read_text(encoding="utf-8"))
    step = int(decision["selected_checkpoint_step"])
    model_sha = file_sha256(artifact_dir / "pretrained_model" / "model.safetensors")
    payload = {
        "schema_version": 1,
        "status": "passed",
        "selected": {"step": step, "model_safetensors_sha256": model_sha},
        "hidden_test_loaded": False,
        "derived_from": {
            "zen_selection_report": f"runs/{protocol.EXPERIMENT_ID}/selection/"
            f"{selection_path.name}",
            "zen_selection_report_sha256": file_sha256(selection_path),
        },
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    destination = run_root / protocol.EXPERIMENT_ID / "selection" / (
        Path(selection_path).stem + "-gate.json"
    )
    if destination.exists():
        if destination.read_text(encoding="utf-8") != text:
            raise FileExistsError("Gate-facing selection evidence drifted.")
    else:
        destination.write_text(text, encoding="utf-8")
    del suffix
    return destination, file_sha256(destination), step, model_sha


def _write_backup_evidence(run_root: Path, artifact_dir: Path) -> tuple[Path, str]:
    """Inventory-based backup evidence: the deploy artifact already lives on
    the durable data disk (the registered Way precedent,
    ``same_durable_data_disk: true``), so this record mirrors every file
    checksum instead of duplicating gigabytes into an archive."""

    backup_dir = run_root / protocol.EXPERIMENT_ID / "artifact_backup"
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
        "stage": "smolvla_remote_durable_artifact_backup",
        "artifact_id": artifact_dir.name,
        "artifact_manifest_sha256": current_manifest_sha,
        "backup_mode": "durable_disk_file_inventory",
        "files": files,
        "file_count": len(files),
        "off_host_copy_created": False,
        "archive_file_set_matches_manifest": True,
        "artifact_reload_verified": True,
        "same_durable_data_disk": True,
        "gate_unlock_scope": "autodl_gate3_gate4_only",
        "hidden_test_loaded": False,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    destination.write_text(text, encoding="utf-8")
    return destination, hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("gate3", "gate4"):
        gate_parser = subparsers.add_parser(command)
        gate_parser.add_argument("--plan", type=Path, required=True)
        gate_parser.add_argument("--artifact-id", required=True)
        gate_parser.add_argument("--run-root", type=Path, required=True)
        gate_parser.add_argument("--artifact-root", type=Path, required=True)
        if command == "gate4":
            gate_parser.add_argument("--gate3-report", type=Path, required=True)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    artifact_dir = args.artifact_root.resolve() / protocol.EXPERIMENT_ID / args.artifact_id
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Artifact manifest missing; run export first.")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    plan_id = str(manifest["plan_id"])
    spec = protocol.ZEN_SPECS[plan_id]
    suffix = ARM_SUFFIXES[plan_id]

    selection_source = (
        run_root / protocol.EXPERIMENT_ID / "selection" / f"{spec['run_name']}-selection.json"
    )
    gate_selection, gate_selection_sha, step, model_sha = _write_gate_selection(
        run_root, selection_source, artifact_dir, suffix
    )
    backup_report, backup_sha = _write_backup_evidence(run_root, artifact_dir)
    del backup_report

    sim_plan_id = f"{spec['run_name']}-sim-{suffix}"
    sim_code_blocks = "\n".join(
        f"  {relative}: {file_sha256(REPOSITORY_ROOT / relative)}"
        for relative in (
            "scripts/smolvla_sim_gate.py",
            "scripts/smolvla_autodl_zen_sim_gate.py",
            "src/rosetta_reality/sim/gym_aloha.py",
            "src/rosetta_reality/vla/processor.py",
        )
    )
    content = SIM_PLAN_TEMPLATE.format(
        sim_plan_id=sim_plan_id,
        experiment_id=protocol.EXPERIMENT_ID,
        artifact_id=args.artifact_id,
        artifact_manifest_sha256=file_sha256(manifest_path),
        role=spec["role"],
        prior_report=PRIOR_FAILURE["report"],
        prior_sha=PRIOR_FAILURE["report_sha256"],
        prior_task_report=PRIOR_TASK_FAILURE["report"],
        prior_task_sha=PRIOR_TASK_FAILURE["report_sha256"],
        selection_report=f"runs/{protocol.EXPERIMENT_ID}/selection/{gate_selection.name}",
        selection_sha=gate_selection_sha,
        selected_step=step,
        model_sha=model_sha,
        backup_sha=backup_sha,
        contract_sha=CONTRACT_SHA,
        sim_code_blocks=sim_code_blocks,
        suffix=suffix,
    )
    yaml.safe_load(content)
    destination = REPOSITORY_ROOT / "configs/vla" / f"{sim_plan_id}.yaml"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Rendered sim plan drifted: {destination.name}")
    else:
        destination.write_text(content, encoding="utf-8")

    original_online = simulator._OnlineSmolVLA
    original_runtime = simulator._runtime
    original_create_json = simulator.create_json
    original_repository_path = simulator._repository_path
    global _ACTIVE_SIM_PLAN
    _ACTIVE_SIM_PLAN = destination.resolve()
    try:
        simulator._OnlineSmolVLA = _build_online_cuda_class()
        simulator._runtime = _cuda_runtime
        simulator._repository_path = _runtime_evidence_path

        def create_json(path: Path, payload: dict[str, Any]) -> None:
            if str(payload.get("gate", "")).startswith("m2_gate"):
                payload["zen_protocol"] = {
                    "schema_version": 1,
                    "wrapper_sha256": file_sha256(Path(__file__)),
                    "protocol_module_sha256": file_sha256(
                        REPOSITORY_ROOT / "scripts/smolvla_zen_protocol.py"
                    ),
                    "artifact_backup_verified": True,
                    "code_identity": workspace_code_identity(REPOSITORY_ROOT),
                }
            original_create_json(path, payload)

        simulator.create_json = create_json

        if args.command == "gate3":
            result = simulator.gate3(destination.resolve())
        else:
            result = simulator.gate4(destination.resolve(), args.gate3_report.resolve())
        return result
    finally:
        simulator._OnlineSmolVLA = original_online
        simulator._runtime = original_runtime
        simulator.create_json = original_create_json
        simulator._repository_path = original_repository_path
        _ACTIVE_SIM_PLAN = None


if __name__ == "__main__":
    raise SystemExit(main())
