"""Trace the selected Zen first-action artifact against the registered expert replay.

Create-only Zen-native adaptation of the registered Aster trajectory diagnostic
(``diagnose_smolvla_aster_trajectory.py``): the frozen historical stack stays
untouched. This script derives a gate-facing selection record from the Zen
selection decision, renders a trace simulation plan whose projection/noise
precedents reuse the immutable registered failure reports of this experiment,
validates the exported deploy artifact through the frozen
``smolvla_sim_gate._load_artifact`` boundary, and executes the same dual-
environment step-aligned expert/policy trace from the exact train episode 2 /
simulator seed 10 reset.

The trace is a non-gating diagnostic. After divergence the time-indexed expert
action is a reference, not a state-conditioned recovery oracle.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import sim_gate as dataset_gate  # noqa: E402
import smolvla_sim_gate as simulator  # noqa: E402
import smolvla_zen_protocol as protocol  # noqa: E402

from rosetta_reality.data import resolve_prepared_cache  # noqa: E402
from rosetta_reality.data.config import load_dataset_config  # noqa: E402
from rosetta_reality.experiment import (  # noqa: E402
    file_sha256,
    stable_hash,
    workspace_code_identity,
)
from rosetta_reality.features import create_json  # noqa: E402
from rosetta_reality.sim import GymAlohaEnvironment, load_action_contract  # noqa: E402
from rosetta_reality.vla.action_space import SmolVLAActionSpace  # noqa: E402
from rosetta_reality.vla.processor import ensure_smolvla_action_boundary  # noqa: E402

EXPERIMENT_ID = protocol.EXPERIMENT_ID
DEFAULT_ARTIFACT_ID = "m2-smolvla450m-zen-cuda-b64-firstaction-001-step0316-deploy-001"
ZEN_FIRSTACTION_PLAN_ID = "m2-smolvla450m-zen-firstaction-001"
ZEN_FIRSTACTION_PLAN = (
    REPOSITORY_ROOT / "configs/vla/smolvla_450m_aloha_insertion_zen_cuda_b64_firstaction_001.yaml"
)
DATASET_CONFIG = REPOSITORY_ROOT / "configs/data/aloha_sim_insertion_m2.yaml"
STATE_MAE_THRESHOLDS = (0.005, 0.01, 0.025, 0.05, 0.1)
HIDDEN_TEST_EPISODES = frozenset({31, 6, 1, 24, 5})
SIM_PLAN_ID = "m2-smolvla450m-zen-firstaction-trace-sim-001"
TRACE_SELECTION_STEM = "m2-smolvla450m-zen-cuda-b64-firstaction-001-selection-trace-gate"

PRIOR_FAILURE = {
    "report": "runs/m2-smolvla450m-aloha-insertion-001/gates/gate3-smolvla-sim-001.json",
    "report_sha256": "5df3b887984d1c8fd47084c3315a71e3894eac18cc80b1eb08b2120583ed26ed",
    "failed_criterion": "raw_actions_within_contract",
}
PRIOR_TASK_FAILURE = {
    "report": (
        "runs/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/"
        "gates/gate4-smolvla-sim-422.json"
    ),
    "report_sha256": "21a851abdda6b9f642a6021a768905bf0878ff37d3d13ccd75145affabf98003",
    "failed_criterion": "minimum_task_success_rate",
}
CONTRACT_SHA = "fc71a0438f0e3af7258e5b52d82fa22fc53c12b47901606cbee715524392ac62"
CONTRACT_PATH = "configs/sim/aloha_insertion_smolvla.yaml"
CONTAINER_IMAGE_ID = "sha256:f4a71c4020cd54d2a878f01628d591af9572f0784458f4c821008f8aea30393c"
MEMORY_LIMIT = "6g"

SIM_PLAN_TEMPLATE = """schema_version: 1
role: vla
stage: m2_closed_loop_simulation
status: preregistered
plan_id: {sim_plan_id}
experiment_id: {experiment_id}
artifact_id: {artifact_id}
artifact_manifest_sha256: {artifact_manifest_sha256}
hypothesis: >-
  A step-aligned comparison from the exact train episode 2 / simulator seed 10
  reset will locate where the selected Zen first-action treatment artifact
  first departs from the registered expert replay, and will compare its
  step-zero action error and early state-MAE crossings against the immutable
  Aster trace baseline. The selected artifact, action contract, seeded Gaussian
  policy noise, dataset, simulator physics and expert actions remain read-only
  and fixed. Non-gating diagnostic only.
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
action_contract:
  path: {contract_path}
  sha256: {contract_sha}
diagnostic_protocol:
  dataset_episode: 2
  simulator_seed: 10
  policy_noise_seed: 10
  maximum_steps: 320
  expert_reference: time_indexed_not_state_conditioned
  recovery_oracle: false
  state_mae_thresholds: [0.005, 0.01, 0.025, 0.05, 0.1]
  hidden_test_loaded: false
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
  runtime: docker_linux_from_wsl
  accelerator: xpu
  container_image_id: {container_image_id}
  memory_limit: {memory_limit}
  memory_swap_limit: {memory_limit}
gate3:
  require_projected_policy_actions_within_contract: true
  require_adapter_no_additional_clipping: true
gate4:
  require_gate3_passed: true
hidden_test_loaded: false
"""

SIMULATION_CODE_FILES = (
    "scripts/smolvla_sim_gate.py",
    "scripts/diagnose_smolvla_zen_trajectory.py",
    "scripts/smolvla_zen_protocol.py",
    "src/rosetta_reality/sim/gym_aloha.py",
    "src/rosetta_reality/vla/processor.py",
    "src/rosetta_reality/vla/action_space.py",
)


def _state(observation: dict[str, Any], dimension: int) -> torch.Tensor:
    value = observation.get("robot_state")
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != (dimension,):
        raise ValueError("Trajectory observation violates the registered state contract.")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError("Trajectory state contains NaN or Inf.")
    return value.to(torch.float32).cpu()


def _new_crossings(
    value: float,
    step: int,
    crossings: dict[str, int | None],
) -> bool:
    """Record each first state-MAE threshold crossing and report any change."""

    changed = False
    for threshold in STATE_MAE_THRESHOLDS:
        key = f"{threshold:g}"
        if crossings[key] is None and value >= threshold:
            crossings[key] = step
            changed = True
    return changed


def _pose_delta(
    expert: dict[str, Any], policy: dict[str, Any], body: str
) -> dict[str, Any] | None:
    expert_pose = expert.get("bodies", {}).get(body)
    policy_pose = policy.get("bodies", {}).get(body)
    if not isinstance(expert_pose, dict) or not isinstance(policy_pose, dict):
        return None
    expert_position = torch.tensor(expert_pose["position"], dtype=torch.float64)
    policy_position = torch.tensor(policy_pose["position"], dtype=torch.float64)
    difference = policy_position - expert_position
    return {
        "expert_position": expert_position.tolist(),
        "policy_position": policy_position.tolist(),
        "policy_minus_expert": difference.tolist(),
        "position_l2": float(difference.square().sum().sqrt()),
    }


class _ZenTraceOnlineSmolVLA(simulator._OnlineSmolVLA):
    """The registered XPU online policy plus the serialized bounded boundary.

    Behavioral equivalent of the frozen ``_ActionRepairOnlineSmolVLA`` minus its
    ``sys.argv`` plan lookup: the trace plan path is passed explicitly, exactly
    like the remote Zen gate class receives its rendered plan.
    """

    def __init__(
        self,
        artifact: Path,
        config: dict[str, Any],
        normalization: dict[str, Any],
        contract: Any,
        plan_path: Path,
    ) -> None:
        super().__init__(artifact, config, normalization, contract)
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
        plan = simulator._load_yaml(plan_path)
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


def _write_trace_selection(
    run_root: Path, selection_source: Path, artifact_dir: Path
) -> tuple[Path, str, int, str]:
    """Derive the gate-facing selection record from the Zen selection decision."""

    decision = json.loads(selection_source.read_text(encoding="utf-8"))
    if decision.get("status") != "selected" or decision.get("hidden_test_loaded") is not False:
        raise ValueError("Zen trace selection source is not a completed clean decision.")
    step = int(decision["selected_checkpoint_step"])
    model_sha = file_sha256(artifact_dir / "pretrained_model" / "model.safetensors")
    payload = {
        "schema_version": 1,
        "status": "passed",
        "selected": {"step": step, "model_safetensors_sha256": model_sha},
        "hidden_test_loaded": False,
        "derived_from": {
            "zen_selection_report": f"runs/{EXPERIMENT_ID}/selection/{selection_source.name}",
            "zen_selection_report_sha256": file_sha256(selection_source),
        },
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    destination = run_root / EXPERIMENT_ID / "selection" / f"{TRACE_SELECTION_STEM}.json"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != text:
            raise FileExistsError("Trace-facing selection evidence drifted.")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
    return destination, file_sha256(destination), step, model_sha


def _render_sim_plan(
    run_root: Path,
    artifact_dir: Path,
    gate_selection: Path,
    gate_selection_sha: str,
    selected_step: int,
    model_sha: str,
) -> Path:
    sim_code_blocks = "\n".join(
        f"  {relative}: {file_sha256(REPOSITORY_ROOT / relative)}"
        for relative in SIMULATION_CODE_FILES
    )
    content = SIM_PLAN_TEMPLATE.format(
        sim_plan_id=SIM_PLAN_ID,
        experiment_id=EXPERIMENT_ID,
        artifact_id=artifact_dir.name,
        artifact_manifest_sha256=file_sha256(artifact_dir / "manifest.json"),
        prior_report=PRIOR_FAILURE["report"],
        prior_sha=PRIOR_FAILURE["report_sha256"],
        prior_task_report=PRIOR_TASK_FAILURE["report"],
        prior_task_sha=PRIOR_TASK_FAILURE["report_sha256"],
        selection_report=f"runs/{EXPERIMENT_ID}/selection/{gate_selection.name}",
        selection_sha=gate_selection_sha,
        selected_step=selected_step,
        model_sha=model_sha,
        contract_path=CONTRACT_PATH,
        contract_sha=CONTRACT_SHA,
        sim_code_blocks=sim_code_blocks,
        container_image_id=CONTAINER_IMAGE_ID,
        memory_limit=MEMORY_LIMIT,
    )
    yaml.safe_load(content)
    destination = run_root / EXPERIMENT_ID / "plans" / f"{SIM_PLAN_ID}.yaml"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Rendered trace plan drifted: {destination.name}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    return destination


def _main(args: argparse.Namespace) -> int:
    run_root = simulator._absolute_root("ROSETTA_RUN_ROOT")
    artifact_root = simulator._absolute_root("ROSETTA_ARTIFACT_ROOT")
    artifact_dir = artifact_root / EXPERIMENT_ID / args.artifact_id
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Artifact manifest missing; transfer the selected deploy artifact first."
        )
    manifest = simulator._load_json(manifest_path)
    if str(manifest.get("plan_id")) != ZEN_FIRSTACTION_PLAN_ID:
        raise ValueError("Zen trace requires the first-action treatment artifact.")

    zen_plan = simulator._load_yaml(ZEN_FIRSTACTION_PLAN)
    if (
        args.episode not in zen_plan["training"]["episodes"]
        or args.episode in zen_plan["validation"]["episodes"]
        or args.episode in HIDDEN_TEST_EPISODES
    ):
        raise ValueError("Trajectory trace must use a registered Zen train episode.")

    selection_source = (
        run_root
        / EXPERIMENT_ID
        / "selection"
        / "m2-smolvla450m-zen-cuda-b64-firstaction-001-selection.json"
    )
    gate_selection, gate_selection_sha, selected_step, model_sha = _write_trace_selection(
        run_root, selection_source, artifact_dir
    )
    plan_path = _render_sim_plan(
        run_root,
        artifact_dir,
        gate_selection,
        gate_selection_sha,
        selected_step,
        model_sha,
    )

    compiler_cache = (
        run_root / "compiler_cache" / f"zen-trace-{file_sha256(plan_path)[:12]}"
    )
    triton_cache = compiler_cache / "triton"
    inductor_cache = compiler_cache / "inductor"
    triton_cache.mkdir(parents=True, exist_ok=True)
    inductor_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(triton_cache)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_cache)

    plan, artifact, artifact_manifest, config, normalization = simulator._load_artifact(
        plan_path
    )
    contract_path = simulator._repository_path(plan["action_contract"]["path"])
    contract = load_action_contract(contract_path)
    dataset_config = load_dataset_config(DATASET_CONFIG)
    dataset_root, dataset_manifest = resolve_prepared_cache(
        dataset_config,
        REPOSITORY_ROOT,
        validate_checksums=True,
    )
    rows = dataset_gate._dataset_rows(dataset_root, args.episode, dataset_config.fields)
    maximum_steps = min(args.maximum_steps, len(rows))
    if maximum_steps < 3:
        raise ValueError("Trajectory trace requires at least three expert rows.")

    policy = _ZenTraceOnlineSmolVLA(artifact, config, normalization, contract, plan_path)
    policy.configure_noise("seeded_standard_normal", args.policy_noise_seed)
    expert_environment = GymAlohaEnvironment(
        contract, maximum_episode_steps=maximum_steps
    )
    policy_environment = GymAlohaEnvironment(
        contract, maximum_episode_steps=maximum_steps
    )
    crossings = {f"{value:g}": None for value in STATE_MAE_THRESHOLDS}
    first_events: dict[str, int | None] = {
        "expert_nonzero_reward": None,
        "policy_nonzero_reward": None,
        "expert_done": None,
        "policy_done": None,
        "policy_joint_limit_violation": None,
        "policy_unexpected_collision": None,
    }
    trace_steps: list[dict[str, Any]] = []
    state_mae_values: list[float] = []
    action_mae_values: list[float] = []
    expert_maximum_reward = 0.0
    policy_maximum_reward = 0.0

    def first(name: str, step: int, condition: bool) -> bool:
        if condition and first_events[name] is None:
            first_events[name] = step
            return True
        return False

    try:
        expert_observation = dict(expert_environment.reset(seed=args.seed))
        policy_observation = dict(policy_environment.reset(seed=args.seed))
        reset_state_mae = float(
            (
                _state(policy_observation, policy.state_dimension)
                - _state(expert_observation, policy.state_dimension)
            )
            .abs()
            .mean()
        )
        reset_snapshot = {
            "expert": expert_environment.diagnostic_snapshot(),
            "policy": policy_environment.diagnostic_snapshot(),
        }
        if reset_state_mae > 1e-7:
            raise RuntimeError("Independent simulator resets are not deterministic.")

        for step, row in enumerate(rows[:maximum_steps]):
            expert_pre_state = _state(expert_observation, policy.state_dimension)
            policy_pre_state = _state(policy_observation, policy.state_dimension)
            expert_raw = torch.as_tensor(
                row[dataset_config.fields.action], dtype=torch.float32
            )
            contract.validate_tensor(expert_raw, allow_chunk=False)
            expert_action, expert_clip = contract.clip(expert_raw)
            policy_raw_chunk, policy_processed_chunk = policy.predict(
                policy_observation,
                plan["inference"]["instruction"],
            )
            policy_raw = policy_raw_chunk[0]
            policy_processed = policy_processed_chunk[0]
            policy_action, policy_clip = contract.clip(policy_processed)

            expert_observation, expert_reward, expert_done, expert_info = (
                expert_environment.step(expert_action)
            )
            policy_observation, policy_reward, policy_done, policy_info = (
                policy_environment.step(policy_action)
            )
            expert_observation = dict(expert_observation)
            policy_observation = dict(policy_observation)
            expert_state = _state(expert_observation, policy.state_dimension)
            policy_state = _state(policy_observation, policy.state_dimension)
            state_difference = policy_state - expert_state
            action_difference = policy_action - expert_action
            state_mae = float(state_difference.abs().mean())
            action_mae = float(action_difference.abs().mean())
            state_mae_values.append(state_mae)
            action_mae_values.append(action_mae)
            expert_maximum_reward = max(expert_maximum_reward, float(expert_reward))
            policy_maximum_reward = max(policy_maximum_reward, float(policy_reward))

            expert_snapshot = expert_environment.diagnostic_snapshot()
            policy_snapshot = policy_environment.diagnostic_snapshot()
            policy_unexpected = sum(
                policy_environment.is_unexpected_collision_pair(first_name, second_name)
                for first_name, second_name in policy_environment.contact_pairs()
            )
            event = _new_crossings(state_mae, step, crossings)
            event |= first("expert_nonzero_reward", step, expert_reward != 0.0)
            event |= first("policy_nonzero_reward", step, policy_reward != 0.0)
            event |= first("expert_done", step, expert_done)
            event |= first("policy_done", step, policy_done)
            event |= first(
                "policy_joint_limit_violation",
                step,
                bool(policy_snapshot["joint_limit_violations"]),
            )
            event |= first(
                "policy_unexpected_collision", step, policy_unexpected > 0
            )
            if step < 12 or step % 25 == 0 or event or step + 1 == maximum_steps:
                trace_steps.append(
                    {
                        "step": step,
                        "dataset_frame": int(row[dataset_config.fields.frame_index]),
                        "expert_reference": {
                            "state_conditioned": False,
                            "recovery_oracle": False,
                        },
                        "state": {
                            "expert_pre": expert_pre_state.tolist(),
                            "policy_pre": policy_pre_state.tolist(),
                            "expert_post": expert_state.tolist(),
                            "policy_post": policy_state.tolist(),
                            "post_mae": state_mae,
                            "post_l2": float(state_difference.square().sum().sqrt()),
                        },
                        "action": {
                            "expert_raw": expert_raw.tolist(),
                            "expert_executed": expert_action.tolist(),
                            "expert_clipped_elements": int(expert_clip.sum()),
                            "policy_raw": policy_raw.tolist(),
                            "policy_processed": policy_processed.tolist(),
                            "policy_executed": policy_action.tolist(),
                            "policy_clipped_elements": int(policy_clip.sum()),
                            "executed_mae_vs_time_indexed_expert": action_mae,
                        },
                        "outcome": {
                            "expert_reward": float(expert_reward),
                            "policy_reward": float(policy_reward),
                            "expert_done": bool(expert_done),
                            "policy_done": bool(policy_done),
                            "expert_success": bool(expert_info.get("is_success", False)),
                            "policy_success": bool(policy_info.get("is_success", False)),
                            "policy_unexpected_collisions": policy_unexpected,
                        },
                        "pose_delta": {
                            body: _pose_delta(expert_snapshot, policy_snapshot, body)
                            for body in (
                                "peg",
                                "socket",
                                "vx300s_left/gripper_link",
                                "vx300s_right/gripper_link",
                            )
                        },
                        "expert_snapshot": expert_snapshot,
                        "policy_snapshot": policy_snapshot,
                    }
                )
            if expert_done or policy_done:
                break

        report = {
            "schema_version": 1,
            "status": "complete",
            "diagnostic": "zen_train_pose_expert_policy_trajectory_divergence",
            "experiment_id": plan["experiment_id"],
            "artifact_id": artifact_manifest["artifact_id"],
            "artifact_manifest_sha256": file_sha256(artifact / "manifest.json"),
            "simulation_plan_sha256": file_sha256(plan_path),
            "dataset_revision": dataset_manifest.resolved_revision,
            "dataset_manifest_sha256": file_sha256(dataset_root / "manifest.json"),
            "episode": args.episode,
            "seed": args.seed,
            "policy_noise_seed": args.policy_noise_seed,
            "maximum_steps": maximum_steps,
            "protocol": {
                "expert": "time_indexed_dataset_action_in_independent_environment",
                "policy": "receding_horizon_first_action",
                "warning": (
                    "After divergence the time-indexed expert action is a reference, "
                    "not a state-conditioned recovery oracle."
                ),
                "test_split_opened": False,
                "aster_baseline": (
                    "runs/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/"
                    "diagnostics/aster-trajectory-520c8ec87c1618fc.json"
                ),
            },
            "reset": {
                "cross_environment_state_mae": reset_state_mae,
                "snapshot": reset_snapshot,
            },
            "summary": {
                "steps_executed": len(state_mae_values),
                "first_state_mae_crossings": crossings,
                "first_events": first_events,
                "step_zero_action_mae": action_mae_values[0],
                "step_zero_post_state_mae": state_mae_values[0],
                "maximum_state_mae": max(state_mae_values),
                "final_state_mae": state_mae_values[-1],
                "mean_time_indexed_action_mae": sum(action_mae_values)
                / len(action_mae_values),
                "expert_maximum_reward": expert_maximum_reward,
                "policy_maximum_reward": policy_maximum_reward,
            },
            "trace_steps": trace_steps,
            "zen_protocol": {
                "schema_version": 1,
                "trace_script_sha256": file_sha256(Path(__file__)),
                "protocol_module_sha256": file_sha256(
                    REPOSITORY_ROOT / "scripts/smolvla_zen_protocol.py"
                ),
                "code_identity": workspace_code_identity(REPOSITORY_ROOT),
            },
            "code_identity": workspace_code_identity(REPOSITORY_ROOT),
            "hidden_test_loaded": False,
        }
        json.dumps(report, allow_nan=False)
        destination = (
            run_root
            / plan["experiment_id"]
            / "diagnostics"
            / f"zen-trajectory-{stable_hash(report)[:16]}.json"
        )
        create_json(destination, report)
        print(json.dumps(report["summary"], indent=2, sort_keys=True))
        print(f"Report: {destination.relative_to(run_root)}")
        return 0
    finally:
        expert_environment.close()
        policy_environment.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-id", default=DEFAULT_ARTIFACT_ID)
    parser.add_argument("--episode", type=int, default=2)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--policy-noise-seed", type=int, default=10)
    parser.add_argument("--maximum-steps", type=int, default=320)
    args = parser.parse_args()
    if args.maximum_steps <= 0 or args.seed < 0 or args.policy_noise_seed < 0:
        raise ValueError("Trajectory limits and seeds must be non-negative.")
    if not math.isfinite(float(args.maximum_steps)):
        raise ValueError("Trajectory maximum steps must be finite.")
    return _main(args)


if __name__ == "__main__":
    raise SystemExit(main())
