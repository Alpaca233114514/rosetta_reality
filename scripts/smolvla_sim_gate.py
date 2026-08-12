"""Run SmolVLA Gate 3 and Gate 4 through the simulator-neutral ALOHA adapter."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
import yaml
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
DEFAULT_PLAN = REPOSITORY_ROOT / "configs/vla/smolvla_450m_aloha_insertion_sim_001.yaml"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rosetta_reality.experiment import (  # noqa: E402
    file_sha256,
    stable_hash,
    workspace_code_identity,
)
from rosetta_reality.features import create_json  # noqa: E402
from rosetta_reality.sim import (  # noqa: E402
    ActionContract,
    GymAlohaEnvironment,
    load_action_contract,
)
from rosetta_reality.tracking import sanitize_metrics, validate_public_payload  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}.")
    json.dumps(value, allow_nan=False)
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path.name}.")
    return value


def _repository_path(raw: str) -> Path:
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Simulation plan paths must be safe repository-relative paths.")
    path = (REPOSITORY_ROOT / relative).resolve()
    if not path.is_relative_to(REPOSITORY_ROOT) or not path.is_file():
        raise ValueError("Simulation plan path is missing or outside the repository.")
    return path


def _absolute_root(environment: str) -> Path:
    raw = os.environ.get(environment)
    if not raw or not Path(raw).is_absolute():
        raise ValueError(f"{environment} must be an absolute container path.")
    return Path(raw).resolve()


def _convert_statistics(value: dict[str, Any]) -> dict[str, dict[str, torch.Tensor]]:
    result: dict[str, dict[str, torch.Tensor]] = {}
    for feature, statistics in value.items():
        if not isinstance(statistics, dict):
            raise ValueError("Artifact statistics must be mappings.")
        result[feature] = {
            statistic: torch.tensor(
                raw_value,
                dtype=torch.int64 if statistic == "count" else torch.float64,
            )
            for statistic, raw_value in statistics.items()
        }
    return result


class _ArtifactMetadata:
    def __init__(self, config: dict[str, Any], normalization: dict[str, Any]) -> None:
        self.features = config["dataset_features"]
        self.fps = int(config["dataset_fps"])
        combined = dict(normalization["effective_stats"])
        combined.update(
            {
                feature: normalization["visual_statistics"]
                for feature in normalization["visual_features"]
            }
        )
        self.stats = _convert_statistics(combined)


def _validate_policy_contract_shape(
    policy_config: Any, contract: ActionContract
) -> int:
    output_feature = policy_config.output_features.get("action")
    output_shape = getattr(output_feature, "shape", None)
    if (
        policy_config.chunk_size != contract.chunk_length
        or not isinstance(output_shape, tuple | list)
        or tuple(output_shape) != (contract.dimension,)
    ):
        raise ValueError("Artifact policy dimensions differ from the Action Contract.")
    state_feature = policy_config.input_features.get("observation.state")
    state_shape = getattr(state_feature, "shape", None)
    if (
        not isinstance(state_shape, tuple | list)
        or len(state_shape) != 1
        or isinstance(state_shape[0], bool)
        or not isinstance(state_shape[0], int)
        or state_shape[0] <= 0
    ):
        raise ValueError("Artifact policy has no one-dimensional robot-state contract.")
    return int(state_shape[0])


class _OnlineSmolVLA:
    def __init__(
        self,
        artifact: Path,
        config: dict[str, Any],
        normalization: dict[str, Any],
        contract: ActionContract,
    ):
        device_name = os.environ.get("ROSETTA_TORCH_DEVICE")
        if device_name != "xpu" or not torch.xpu.is_available():
            raise RuntimeError("SmolVLA simulation requires the registered XPU runtime.")
        self.device = torch.device(device_name)
        self.mixed_precision = str(config["mixed_precision"])
        if self.mixed_precision != "bf16":
            raise ValueError("Simulation mixed precision differs from the registered artifact.")
        pretrained = artifact / "pretrained_model"
        policy_cfg = SmolVLAConfig.from_pretrained(pretrained, local_files_only=True)
        policy_cfg.device = self.device.type
        policy_cfg.pretrained_path = pretrained
        policy_cfg.pretrained_revision = None
        policy_cfg.load_vlm_weights = False
        metadata = _ArtifactMetadata(config, normalization)
        self.policy = make_policy(
            cfg=policy_cfg,
            ds_meta=metadata,
            rename_map=config["rename_map"],
        )
        self.state_dimension = _validate_policy_contract_shape(
            self.policy.config, contract
        )
        self.action_dimension = contract.dimension
        self.chunk_length = contract.chunk_length
        self.preprocessor, self.postprocessor = make_pre_post_processors(
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
                "rename_observations_processor": {"rename_map": config["rename_map"]},
            },
            postprocessor_overrides={
                "unnormalizer_processor": {
                    "features": self.policy.config.output_features,
                    "norm_map": self.policy.config.normalization_mapping,
                    "stats": metadata.stats,
                }
            },
        )
        self.policy.eval()
        self._noise_mode = "zeros"
        self._noise_generator = torch.Generator(device="cpu")
        self._noise_seed: int | None = None

    def configure_noise(self, mode: str, seed: int | None) -> None:
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

    def predict(
        self, observation: dict[str, Any], instruction: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
            noise = torch.zeros(noise_shape, device=self.device, dtype=processed_state.dtype)
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
        torch.xpu.synchronize()
        with (
            torch.inference_mode(),
            torch.autocast(device_type="xpu", dtype=torch.bfloat16),
        ):
            action = self.policy.predict_action_chunk(batch, noise=noise)
        torch.xpu.synchronize()
        action = self.postprocessor(action)
        expected_shape = (1, self.chunk_length, self.action_dimension)
        if not isinstance(action, torch.Tensor) or tuple(action.shape) != expected_shape:
            raise ValueError("SmolVLA simulator output differs from the Action Contract.")
        adapter_steps = [
            step
            for step in self.postprocessor.steps
            if getattr(step.__class__, "_registry_name", None)
            == "rosetta_pi_aloha_postprocessor"
        ]
        if len(adapter_steps) != 1:
            raise ValueError("SmolVLA simulator has no unique action decoder boundary.")
        raw = getattr(adapter_steps[0], "last_unclipped_action", None)
        if not isinstance(raw, torch.Tensor) or tuple(raw.shape) != expected_shape:
            raise ValueError("SmolVLA simulator did not retain the pre-clipping action.")
        return raw[0].detach().cpu(), action[0].detach().cpu()


def _load_artifact(plan_path: Path) -> tuple[dict[str, Any], Path, dict[str, Any], Any, Any]:
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("HF_DATASETS_OFFLINE") != "1":
        raise RuntimeError("SmolVLA simulation must run with networking disabled.")
    plan = _load_yaml(plan_path)
    resources = plan.get("resources", {})
    if (
        plan.get("status") != "preregistered"
        or plan.get("stage") != "m2_closed_loop_simulation"
        or plan.get("hidden_test_loaded") is not False
        or os.environ.get("ROSETTA_DOCKER_MEMORY_LIMIT") != resources.get("memory_limit")
        or os.environ.get("ROSETTA_DOCKER_MEMORY_SWAP_LIMIT")
        != resources.get("memory_swap_limit")
        or (
            resources.get("container_image_id") is not None
            and os.environ.get("ROSETTA_CONTAINER_IMAGE_ID")
            != resources.get("container_image_id")
        )
    ):
        raise ValueError("SmolVLA simulation plan or resource boundary is invalid.")
    selection_path = _repository_path(plan["selection"]["report"])
    selection = _load_json(selection_path)
    if (
        file_sha256(selection_path) != plan["selection"]["report_sha256"]
        or selection.get("status") != "passed"
        or selection.get("selected", {}).get("step") != plan["selection"]["checkpoint_step"]
        or selection.get("selected", {}).get("model_safetensors_sha256")
        != plan["selection"]["model_safetensors_sha256"]
        or selection.get("hidden_test_loaded") is not False
    ):
        raise ValueError("SmolVLA simulation selection identity is invalid.")
    contract_path = _repository_path(plan["action_contract"]["path"])
    if file_sha256(contract_path) != plan["action_contract"]["sha256"]:
        raise ValueError("SmolVLA simulation Action Contract checksum changed.")
    contract = load_action_contract(contract_path)
    inference = plan["inference"]
    projection = inference.get("policy_output_projection", "none")
    if (
        contract.chunk_execution != inference["chunk_execution"]
        or contract.chunk_execution_steps != inference["chunk_execution_steps"]
        or inference["noise"] not in {"zeros", "seeded_standard_normal"}
        or projection not in {"none", "action_contract_clip"}
    ):
        raise ValueError("SmolVLA simulation inference differs from the Action Contract.")
    if projection == "action_contract_clip":
        prior_registration = plan.get("prior_failure")
        if not isinstance(prior_registration, dict):
            raise ValueError("Projected policy output requires a registered prior failure.")
        prior_path = _repository_path(str(prior_registration.get("report", "")))
        prior = _load_json(prior_path)
        if (
            file_sha256(prior_path) != prior_registration.get("report_sha256")
            or prior.get("status") != "failed"
            or prior.get("gate") != "m2_gate_3_small_policy_rollout"
            or prior.get("acceptance_criteria", {}).get("raw_actions_within_contract") is not False
            or prior.get("hidden_test_loaded") is not False
        ):
            raise ValueError("Projected policy output prior-failure identity is invalid.")
        if (
            prior_registration.get("failed_criterion") != "raw_actions_within_contract"
            or inference.get("projection_location")
            != "vla_output_boundary_before_simulation_adapter"
            or inference.get("unprojected_decoder_action_role") != "non_gating_diagnostic"
            or plan.get("gate3", {}).get("require_projected_policy_actions_within_contract")
            is not True
            or plan.get("gate3", {}).get("require_adapter_no_additional_clipping") is not True
            or plan.get("gate4", {}).get("require_gate3_passed") is not True
        ):
            raise ValueError("Projected policy output registration is incomplete.")
    if inference["noise"] == "seeded_standard_normal":
        task_failure_registration = plan.get("prior_task_failure")
        if not isinstance(task_failure_registration, dict):
            raise ValueError("Seeded Gaussian inference requires a registered task failure.")
        task_failure_path = _repository_path(
            str(task_failure_registration.get("report", ""))
        )
        task_failure = _load_json(task_failure_path)
        if (
            file_sha256(task_failure_path)
            != task_failure_registration.get("report_sha256")
            or task_failure.get("status") != "failed"
            or task_failure.get("gate") != "m2_gate_4_development_task_evaluation"
            or task_failure.get("acceptance_criteria", {}).get("minimum_task_success_rate")
            is not False
            or task_failure.get("hidden_test_loaded") is not False
            or task_failure_registration.get("failed_criterion")
            != "minimum_task_success_rate"
            or inference.get("noise_source") != "pinned_lerobot_default_standard_normal"
        ):
            raise ValueError("Seeded Gaussian prior task-failure identity is invalid.")

    artifact = (
        _absolute_root("ROSETTA_ARTIFACT_ROOT")
        / str(plan["experiment_id"])
        / str(plan["artifact_id"])
    )
    manifest_path = artifact / "manifest.json"
    manifest = _load_json(manifest_path)
    config = _load_json(artifact / "config.json")
    normalization = _load_json(artifact / "normalization.json")
    if (
        manifest.get("status") != "verified"
        or manifest.get("artifact_id") != plan["artifact_id"]
        or manifest.get("experiment_id") != plan["experiment_id"]
        or manifest.get("selected_checkpoint_step") != plan["selection"]["checkpoint_step"]
        or manifest.get("selected_checkpoint_model_sha256")
        != plan["selection"]["model_safetensors_sha256"]
        or manifest.get("reload", {}).get("exact_tensor_equality") is not True
        or manifest.get("hidden_test_loaded") is not False
        or config.get("hidden_test_loaded") is not False
        or normalization.get("source_split") != "train"
        or normalization.get("hidden_test_loaded") is not False
    ):
        raise ValueError("SmolVLA simulation artifact identity is invalid.")
    for relative, expected in manifest["files"].items():
        path = artifact / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"SmolVLA artifact checksum changed: {relative}.")
    artifact_contract = _load_json(artifact / "action_contract.json")
    if artifact_contract != json.loads(json.dumps(asdict(contract), allow_nan=False)):
        raise ValueError("Exported and registered Action Contracts differ.")
    return plan, artifact, manifest, config, normalization


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def _rollout(
    policy: _OnlineSmolVLA,
    contract: Any,
    instruction: str,
    *,
    seed: int,
    maximum_steps: int,
    project_policy_output: bool,
    noise_mode: str,
    policy_noise_seed: int | None,
) -> dict[str, Any]:
    configure_noise = getattr(policy, "configure_noise", None)
    if callable(configure_noise):
        configure_noise(noise_mode, policy_noise_seed)
    elif noise_mode != "zeros" or policy_noise_seed is not None:
        raise ValueError("Policy does not expose the registered inference-noise interface.")
    environment = GymAlohaEnvironment(contract, maximum_episode_steps=maximum_steps)
    inference_latencies: list[float] = []
    simulation_latencies: list[float] = []
    executed: list[torch.Tensor] = []
    rewards: list[float] = []
    raw_limit_violations = 0
    policy_output_limit_violations = 0
    executed_limit_violations = 0
    raw_dimension_violations = torch.zeros(contract.dimension, dtype=torch.int64)
    raw_maximum_overshoot = torch.zeros(contract.dimension, dtype=torch.float64)
    raw_minimum = torch.full((contract.dimension,), float("inf"), dtype=torch.float64)
    raw_maximum = torch.full((contract.dimension,), float("-inf"), dtype=torch.float64)
    invalid_actions = 0
    joint_limit_violations = 0
    unexpected_collisions = 0
    unexpected_pairs: Counter[str] = Counter()
    success = False
    terminated = False
    truncated = False
    try:
        observation = environment.reset(seed=seed)
        reset_contacts = environment.contact_pairs()
        while len(executed) < maximum_steps:
            started = time.perf_counter()
            prediction = policy.predict(observation, instruction)
            if isinstance(prediction, tuple):
                raw_prediction, processed_prediction = prediction
            else:
                raw_prediction = processed_prediction = prediction
            raw_chunk = raw_prediction[: contract.chunk_execution_steps]
            processed_chunk = processed_prediction[: contract.chunk_execution_steps]
            inference_latencies.append(time.perf_counter() - started)
            if not bool(torch.isfinite(raw_chunk).all()):
                invalid_actions += 1
                break
            done = False
            for raw_action, processed_action in zip(
                raw_chunk, processed_chunk, strict=True
            ):
                clipped_raw, raw_mask = contract.clip(raw_action)
                raw_limit_violations += int(raw_mask.sum().item())
                raw_dimension_violations += raw_mask.to(torch.int64)
                raw64 = raw_action.to(torch.float64)
                lower = contract.lower_bounds.to(torch.float64)
                upper = contract.upper_bounds.to(torch.float64)
                raw_minimum = torch.minimum(raw_minimum, raw64)
                raw_maximum = torch.maximum(raw_maximum, raw64)
                overshoot = torch.maximum(lower - raw64, raw64 - upper).clamp_min(0)
                raw_maximum_overshoot = torch.maximum(raw_maximum_overshoot, overshoot)
                policy_action = (
                    clipped_raw if project_policy_output else processed_action
                )
                clipped, policy_mask = contract.clip(policy_action)
                policy_output_limit_violations += int(policy_mask.sum().item())
                started = time.perf_counter()
                observation, reward, done, info = environment.step(clipped)
                simulation_latencies.append(time.perf_counter() - started)
                executed.append(clipped)
                rewards.append(reward)
                executed_limit_violations += int(environment.last_clip_mask.sum().item())
                state = observation["robot_state"]
                joint_limit_violations += int(
                    (
                        (state < contract.lower_bounds - 1e-5)
                        | (state > contract.upper_bounds + 1e-5)
                    ).sum()
                )
                for first, second in environment.contact_pairs():
                    if environment.is_unexpected_collision_pair(first, second):
                        unexpected_collisions += 1
                        unexpected_pairs[" <-> ".join(sorted((first, second)))] += 1
                success = success or bool(info.get("is_success", False))
                terminated = terminated or bool(info.get("terminated", False))
                truncated = truncated or bool(info.get("truncated", False))
                if done or len(executed) >= maximum_steps:
                    break
            if done:
                break
        smoothness = 0.0
        if len(executed) > 1:
            actions = torch.stack(executed)
            smoothness = float((actions[1:] - actions[:-1]).square().sum(-1).sqrt().mean())
        action_elements = max(1, len(executed) * contract.dimension)
        dimension_diagnostics = {}
        for index, name in enumerate(contract.dimension_names):
            dimension_diagnostics[name] = {
                "strict_violation_count": int(raw_dimension_violations[index].item()),
                "maximum_overshoot": float(raw_maximum_overshoot[index].item()),
                "minimum_predicted": (
                    float(raw_minimum[index].item()) if executed else None
                ),
                "maximum_predicted": (
                    float(raw_maximum[index].item()) if executed else None
                ),
            }
        return {
            "seed": seed,
            "noise_mode": noise_mode,
            "policy_noise_seed": policy_noise_seed,
            "success": success,
            "terminated": terminated,
            "truncated": truncated,
            "rollout_length": len(executed),
            "maximum_reward": max(rewards, default=0.0),
            "invalid_action_rate": invalid_actions / max(1, len(executed) + invalid_actions),
            "raw_limit_violation_rate": raw_limit_violations / action_elements,
            "unprojected_limit_violation_rate": raw_limit_violations / action_elements,
            "policy_output_limit_violation_rate": policy_output_limit_violations
            / action_elements,
            "executed_limit_violation_rate": executed_limit_violations / action_elements,
            "unprojected_dimension_diagnostics": dimension_diagnostics,
            "joint_limit_violations": joint_limit_violations,
            "unexpected_collisions": unexpected_collisions,
            "unexpected_collision_pairs": dict(sorted(unexpected_pairs.items())),
            "reset_contact_events": len(reset_contacts),
            "action_smoothness_l2": smoothness,
            "policy_inference_calls": len(inference_latencies),
            "mean_policy_inference_seconds": sum(inference_latencies)
            / max(1, len(inference_latencies)),
            "p95_policy_inference_seconds": _p95(inference_latencies),
            "mean_simulation_step_seconds": sum(simulation_latencies)
            / max(1, len(simulation_latencies)),
            "p95_simulation_step_seconds": _p95(simulation_latencies),
        }
    finally:
        environment.close()


def _trackio(
    plan: dict[str, Any],
    plan_path: Path,
    gate: str,
    metrics: dict[str, int | float],
) -> None:
    import trackio

    config = {
        "experiment_id": plan["experiment_id"],
        "role": "vla",
        "phase": gate,
        "artifact_id": plan["artifact_id"],
        "noise_mode": plan["inference"]["noise"],
        "simulation_plan_sha256": file_sha256(plan_path),
        "test_split_loaded": False,
    }
    validate_public_payload(config, context="smolvla_sim_config")
    trackio.init(
        project="rosetta-reality-vla",
        name=f"{plan['plan_id']}-{gate}",
        group=f"{plan['experiment_id']}-simulation",
        config=config,
        resume="never",
        embed=False,
        auto_log_cpu=False,
        auto_log_gpu=False,
    )
    try:
        trackio.log(sanitize_metrics(metrics, mode="eval"), step=0)
    finally:
        trackio.finish()


def _runtime() -> dict[str, Any]:
    return {
        "torch_version": torch.__version__,
        "lerobot_version": version("lerobot"),
        "gym_aloha_version": version("gym-aloha"),
        "trackio_version": version("trackio"),
        "device": "xpu",
        "xpu_name": torch.xpu.get_device_name(0),
        "container_image_id": os.environ.get("ROSETTA_CONTAINER_IMAGE_ID"),
        "network_disabled": True,
    }


def gate3(plan_path: Path) -> int:
    plan, artifact, manifest, config, normalization = _load_artifact(plan_path)
    contract = load_action_contract(_repository_path(plan["action_contract"]["path"]))
    policy = _OnlineSmolVLA(artifact, config, normalization, contract)
    registered = plan["gate3"]
    noise_mode = str(plan["inference"]["noise"])
    policy_noise_seed = (
        int(registered["policy_noise_seed"])
        if noise_mode == "seeded_standard_normal"
        else None
    )
    metrics = _rollout(
        policy,
        contract,
        str(plan["inference"]["instruction"]),
        seed=int(registered["seed"]),
        maximum_steps=int(registered["maximum_steps"]),
        project_policy_output=plan["inference"].get("policy_output_projection")
        == "action_contract_clip",
        noise_mode=noise_mode,
        policy_noise_seed=policy_noise_seed,
    )
    criteria: dict[str, bool] = {
        "completed_nonempty_rollout": metrics["rollout_length"] > 0,
        "finite_actions": metrics["invalid_action_rate"] == 0,
        "executed_actions_within_contract": metrics["executed_limit_violation_rate"] == 0,
        "joint_limits_respected": metrics["joint_limit_violations"] == 0,
        "maximum_unexpected_collisions": metrics["unexpected_collisions"]
        <= int(registered["maximum_unexpected_collisions"]),
        "artifact_reload_verified": manifest["reload"]["verified"] is True,
    }
    projection = plan["inference"].get("policy_output_projection", "none")
    if projection == "action_contract_clip":
        criteria.update(
            policy_output_projection_registered=True,
            projected_policy_actions_within_contract=(
                metrics["policy_output_limit_violation_rate"] == 0
            ),
        )
    else:
        criteria["raw_actions_within_contract"] = metrics["raw_limit_violation_rate"] == 0
    passed = all(criteria.values())
    public_metrics = {
        key: value
        for key, value in metrics.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }
    _trackio(plan, plan_path, "gate3", public_metrics)
    report = {
        "schema_version": 1,
        "gate": "m2_gate_3_small_policy_rollout",
        "status": "passed" if passed else "failed",
        "experiment_id": plan["experiment_id"],
        "artifact_id": plan["artifact_id"],
        "artifact_manifest_sha256": file_sha256(artifact / "manifest.json"),
        "simulation_plan_sha256": file_sha256(plan_path),
        "artifact_reload_verified": True,
        "chunk_execution": contract.chunk_execution,
        "chunk_execution_steps": contract.chunk_execution_steps,
        "policy_output_projection": projection,
        "noise_mode": noise_mode,
        "non_gating_diagnostics": {
            "unprojected_decoder_actions_within_contract": (
                metrics["unprojected_limit_violation_rate"] == 0
            )
        },
        "acceptance_criteria": criteria,
        "metrics": metrics,
        "runtime": _runtime(),
        "hidden_test_loaded": False,
        "code_identity": workspace_code_identity(REPOSITORY_ROOT),
    }
    report_suffix = str(registered.get("report_suffix", "001"))
    if len(report_suffix) != 3 or not report_suffix.isdigit():
        raise ValueError("Gate 3 report suffix must contain exactly three digits.")
    destination = (
        _absolute_root("ROSETTA_RUN_ROOT")
        / str(plan["experiment_id"])
        / "gates"
        / f"gate3-smolvla-sim-{report_suffix}.json"
    )
    create_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report: {destination.relative_to(REPOSITORY_ROOT).as_posix()}")
    return 0 if passed else 1


def _gate4_episode_workspace_matches(
    episode_report: dict[str, Any], current_code_identity: dict[str, Any]
) -> bool:
    return episode_report.get("code_identity") == current_code_identity


def gate4(plan_path: Path, gate3_report: Path) -> int:
    plan, artifact, manifest, config, normalization = _load_artifact(plan_path)
    gate3_report = gate3_report.resolve()
    prior = _load_json(gate3_report)
    code_identity = workspace_code_identity(REPOSITORY_ROOT)
    if (
        prior.get("status") != "passed"
        or prior.get("gate") != "m2_gate_3_small_policy_rollout"
        or prior.get("artifact_manifest_sha256") != file_sha256(artifact / "manifest.json")
        or prior.get("simulation_plan_sha256") != file_sha256(plan_path)
        or prior.get("code_identity") != code_identity
        or prior.get("hidden_test_loaded") is not False
    ):
        raise ValueError("Gate 4 requires the matching passed Gate 3 report.")
    registered = plan["gate4"]
    report_suffix = str(registered.get("report_suffix", "001"))
    if len(report_suffix) != 3 or not report_suffix.isdigit():
        raise ValueError("Gate 4 report suffix must contain exactly three digits.")
    plan_sha256 = file_sha256(plan_path)
    artifact_sha256 = file_sha256(artifact / "manifest.json")
    gate3_sha256 = file_sha256(gate3_report)
    episode_root = (
        _absolute_root("ROSETTA_RUN_ROOT")
        / str(plan["experiment_id"])
        / "gates"
        / f"gate4-smolvla-sim-{report_suffix}-episodes"
    )
    noise_mode = str(plan["inference"]["noise"])
    environment_seeds = [int(value) for value in registered["seeds"]]
    if noise_mode == "seeded_standard_normal":
        policy_noise_seeds: list[int | None] = [
            int(value) for value in registered["policy_noise_seeds"]
        ]
        if len(policy_noise_seeds) != len(environment_seeds):
            raise ValueError("Gate 4 policy-noise seeds must align with environment seeds.")
    else:
        policy_noise_seeds = [None] * len(environment_seeds)
    episode_registrations: list[tuple[int, int, int | None, Path]] = []
    pending = False
    for index, (seed, policy_noise_seed) in enumerate(
        zip(environment_seeds, policy_noise_seeds, strict=True)
    ):
        path = episode_root / f"episode-{index:02d}-seed-{seed}.json"
        episode_registrations.append((index, seed, policy_noise_seed, path))
        if not path.is_file():
            pending = True
    contract = load_action_contract(_repository_path(plan["action_contract"]["path"]))
    policy = (
        _OnlineSmolVLA(artifact, config, normalization, contract) if pending else None
    )
    episodes: list[dict[str, Any]] = []
    for index, seed, policy_noise_seed, path in episode_registrations:
        if path.is_file():
            episode_report = _load_json(path)
            if (
                episode_report.get("status") != "complete"
                or episode_report.get("episode_index") != index
                or episode_report.get("seed") != seed
                or episode_report.get("policy_noise_seed") != policy_noise_seed
                or episode_report.get("simulation_plan_sha256") != plan_sha256
                or episode_report.get("artifact_manifest_sha256") != artifact_sha256
                or episode_report.get("gate3_report_sha256") != gate3_sha256
                or not _gate4_episode_workspace_matches(episode_report, code_identity)
                or episode_report.get("hidden_test_loaded") is not False
                or not isinstance(episode_report.get("metrics"), dict)
            ):
                raise ValueError("Existing Gate 4 episode report identity is invalid.")
            episodes.append(episode_report["metrics"])
            continue
        if policy is None:
            raise RuntimeError("Gate 4 policy was not loaded for a pending episode.")
        metrics = _rollout(
            policy,
            contract,
            str(plan["inference"]["instruction"]),
            seed=seed,
            maximum_steps=int(registered["maximum_steps"]),
            project_policy_output=plan["inference"].get("policy_output_projection")
            == "action_contract_clip",
            noise_mode=noise_mode,
            policy_noise_seed=policy_noise_seed,
        )
        episode_report = {
            "schema_version": 1,
            "status": "complete",
            "gate": "m2_gate_4_episode",
            "episode_index": index,
            "seed": seed,
            "noise_mode": noise_mode,
            "policy_noise_seed": policy_noise_seed,
            "experiment_id": plan["experiment_id"],
            "artifact_id": plan["artifact_id"],
            "simulation_plan_sha256": plan_sha256,
            "artifact_manifest_sha256": artifact_sha256,
            "gate3_report_sha256": gate3_sha256,
            "metrics": metrics,
            "runtime": _runtime(),
            "hidden_test_loaded": False,
            "code_identity": code_identity,
        }
        create_json(path, episode_report)
        episodes.append(metrics)
        print(
            json.dumps(
                {
                    "gate4_episode_complete": index,
                    "seed": seed,
                    "policy_noise_seed": policy_noise_seed,
                    "rollout_length": metrics["rollout_length"],
                    "success": metrics["success"],
                },
                sort_keys=True,
            )
        )
    total_steps = sum(int(episode["rollout_length"]) for episode in episodes)
    aggregate = {
        "task_success_rate": sum(bool(episode["success"]) for episode in episodes)
        / len(episodes),
        "mean_rollout_length": total_steps / len(episodes),
        "invalid_action_rate": sum(
            float(episode["invalid_action_rate"]) * max(1, int(episode["rollout_length"]))
            for episode in episodes
        )
        / max(1, total_steps),
        "raw_limit_violation_rate": sum(
            float(episode["raw_limit_violation_rate"])
            * max(1, int(episode["rollout_length"]))
            for episode in episodes
        )
        / max(1, total_steps),
        "unprojected_limit_violation_rate": sum(
            float(episode["unprojected_limit_violation_rate"])
            * max(1, int(episode["rollout_length"]))
            for episode in episodes
        )
        / max(1, total_steps),
        "policy_output_limit_violation_rate": sum(
            float(episode["policy_output_limit_violation_rate"])
            * max(1, int(episode["rollout_length"]))
            for episode in episodes
        )
        / max(1, total_steps),
        "executed_limit_violation_rate": sum(
            float(episode["executed_limit_violation_rate"])
            * max(1, int(episode["rollout_length"]))
            for episode in episodes
        )
        / max(1, total_steps),
        "joint_limit_violations": sum(int(ep["joint_limit_violations"]) for ep in episodes),
        "unexpected_collisions": sum(int(ep["unexpected_collisions"]) for ep in episodes),
        "mean_action_smoothness_l2": sum(float(ep["action_smoothness_l2"]) for ep in episodes)
        / len(episodes),
        "mean_policy_inference_seconds": sum(
            float(ep["mean_policy_inference_seconds"]) for ep in episodes
        )
        / len(episodes),
        "mean_simulation_step_seconds": sum(
            float(ep["mean_simulation_step_seconds"]) for ep in episodes
        )
        / len(episodes),
    }
    criteria: dict[str, bool] = {
        "completed_nonempty_rollouts": total_steps > 0,
        "finite_actions": aggregate["invalid_action_rate"] == 0,
        "executed_actions_within_contract": aggregate["executed_limit_violation_rate"] == 0,
        "joint_limits_respected": aggregate["joint_limit_violations"] == 0,
        "minimum_task_success_rate": aggregate["task_success_rate"]
        >= float(registered["minimum_task_success_rate"]),
        "maximum_unexpected_collisions": aggregate["unexpected_collisions"]
        <= int(registered["maximum_unexpected_collisions"]),
    }
    projection = plan["inference"].get("policy_output_projection", "none")
    if projection == "action_contract_clip":
        criteria.update(
            policy_output_projection_registered=True,
            projected_policy_actions_within_contract=(
                aggregate["policy_output_limit_violation_rate"] == 0
            ),
        )
    else:
        criteria["raw_actions_within_contract"] = aggregate["raw_limit_violation_rate"] == 0
    passed = all(criteria.values())
    _trackio(plan, plan_path, "gate4", aggregate)
    report = {
        "schema_version": 1,
        "gate": "m2_gate_4_development_task_evaluation",
        "status": "passed" if passed else "failed",
        "experiment_id": plan["experiment_id"],
        "artifact_id": plan["artifact_id"],
        "artifact_manifest_sha256": artifact_sha256,
        "simulation_plan_sha256": plan_sha256,
        "gate3_report_sha256": gate3_sha256,
        "policy_output_projection": projection,
        "noise_mode": noise_mode,
        "non_gating_diagnostics": {
            "unprojected_decoder_actions_within_contract": (
                aggregate["unprojected_limit_violation_rate"] == 0
            )
        },
        "acceptance_criteria": criteria,
        "aggregate": aggregate,
        "episodes": episodes,
        "runtime": _runtime(),
        "hidden_test_loaded": False,
        "code_identity": code_identity,
    }
    destination = (
        _absolute_root("ROSETTA_RUN_ROOT")
        / str(plan["experiment_id"])
        / "gates"
        / f"gate4-smolvla-sim-{report_suffix}.json"
    )
    create_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report: {destination.relative_to(REPOSITORY_ROOT).as_posix()}")
    return 0 if passed else 1


def execution_strategy_diagnostic(
    plan_path: Path,
    *,
    seeds: list[int],
    maximum_steps: int,
    actions_per_inference: int,
) -> int:
    """Measure short open-loop chunk execution without changing Gate evidence.

    The exported artifact remains bound to its one-action receding-horizon
    contract.  This diagnostic deliberately executes more actions from each
    predicted chunk, records the deviation, and may only be promoted through a
    new versioned Action Contract followed by Gates 1 through 4.
    """

    if not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("Execution diagnostic seeds must be explicit integers.")
    if maximum_steps <= 0:
        raise ValueError("Execution diagnostic maximum_steps must be positive.")
    plan, artifact, manifest, config, normalization = _load_artifact(plan_path)
    contract = load_action_contract(_repository_path(plan["action_contract"]["path"]))
    if not 1 <= actions_per_inference <= contract.chunk_length:
        raise ValueError("actions_per_inference must be within the model action chunk.")
    diagnostic_contract = replace(
        contract,
        chunk_execution=f"diagnostic_first_{actions_per_inference}_then_reobserve",
        chunk_execution_steps=actions_per_inference,
    )
    noise_mode = str(plan["inference"]["noise"])
    if noise_mode == "seeded_standard_normal":
        registered_noise = plan.get("gate4", {}).get("policy_noise_seeds", [])
        if len(registered_noise) < len(seeds):
            raise ValueError(
                "Seeded diagnostic requires one registered policy-noise seed per episode."
            )
        policy_noise_seeds: list[int | None] = [
            int(value) for value in registered_noise[: len(seeds)]
        ]
    else:
        policy_noise_seeds = [None] * len(seeds)
    policy = _OnlineSmolVLA(artifact, config, normalization, diagnostic_contract)
    episodes: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, (seed, policy_noise_seed) in enumerate(
        zip(seeds, policy_noise_seeds, strict=True), start=1
    ):
        print(
            f"Chunk diagnostic episode {index}/{len(seeds)} seed={seed} "
            f"actions_per_inference={actions_per_inference}",
            flush=True,
        )
        episodes.append(
            _rollout(
                policy,
                diagnostic_contract,
                str(plan["inference"]["instruction"]),
                seed=seed,
                maximum_steps=maximum_steps,
                project_policy_output=plan["inference"].get("policy_output_projection")
                == "action_contract_clip",
                noise_mode=noise_mode,
                policy_noise_seed=policy_noise_seed,
            )
        )
    wall_seconds = time.perf_counter() - started
    total_steps = sum(int(episode["rollout_length"]) for episode in episodes)
    total_policy_calls = sum(int(episode["policy_inference_calls"]) for episode in episodes)
    first_action_policy_calls = total_steps
    aggregate = {
        "task_success_rate": sum(bool(episode["success"]) for episode in episodes)
        / len(episodes),
        "maximum_reward": max(float(episode["maximum_reward"]) for episode in episodes),
        "rollout_steps": total_steps,
        "policy_inference_calls": total_policy_calls,
        "first_action_reference_policy_calls": first_action_policy_calls,
        "policy_call_reduction_fraction": 1.0
        - total_policy_calls / max(1, first_action_policy_calls),
        "policy_call_reduction_factor": first_action_policy_calls / max(1, total_policy_calls),
        "wall_seconds": wall_seconds,
        "steps_per_second": total_steps / max(wall_seconds, 1e-12),
        "unexpected_collisions": sum(int(ep["unexpected_collisions"]) for ep in episodes),
        "invalid_action_rate": sum(
            float(ep["invalid_action_rate"]) * max(1, int(ep["rollout_length"]))
            for ep in episodes
        )
        / max(1, total_steps),
    }
    report = {
        "schema_version": 1,
        "status": "complete",
        "diagnostic": "smolvla_non_gate_chunk_execution_strategy",
        "experiment_id": plan["experiment_id"],
        "artifact_id": plan["artifact_id"],
        "artifact_manifest_sha256": file_sha256(artifact / "manifest.json"),
        "simulation_plan_sha256": file_sha256(plan_path),
        "declared_contract": {
            "chunk_execution": contract.chunk_execution,
            "chunk_execution_steps": contract.chunk_execution_steps,
            "action_contract_sha256": file_sha256(
                _repository_path(plan["action_contract"]["path"])
            ),
        },
        "diagnostic_deviation": {
            "chunk_execution": diagnostic_contract.chunk_execution,
            "chunk_execution_steps": diagnostic_contract.chunk_execution_steps,
            "warning": (
                "Non-gate evidence only. Promotion requires a versioned Action Contract and "
                "fresh Gates 1 through 4."
            ),
        },
        "noise_mode": noise_mode,
        "seeds": seeds,
        "maximum_steps": maximum_steps,
        "aggregate": aggregate,
        "episodes": episodes,
        "runtime": _runtime(),
        "hidden_test_loaded": False,
        "code_identity": workspace_code_identity(REPOSITORY_ROOT),
    }
    digest = stable_hash(report)[:12]
    destination = (
        _absolute_root("ROSETTA_RUN_ROOT")
        / str(plan["experiment_id"])
        / "diagnostics"
        / f"smolvla-chunk-execution-{digest}.json"
    )
    create_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report: {destination.relative_to(REPOSITORY_ROOT).as_posix()}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    gate3_parser = subparsers.add_parser("gate3")
    gate3_parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    gate4_parser = subparsers.add_parser("gate4")
    gate4_parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    gate4_parser.add_argument("--gate3-report", type=Path, required=True)
    execution_parser = subparsers.add_parser("execution-diagnostic")
    execution_parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    execution_parser.add_argument("--seeds", type=int, nargs="+", required=True)
    execution_parser.add_argument("--maximum-steps", type=int, required=True)
    execution_parser.add_argument("--actions-per-inference", type=int, required=True)
    args = parser.parse_args()
    if args.command == "gate3":
        return gate3(args.plan.resolve())
    if args.command == "gate4":
        return gate4(args.plan.resolve(), args.gate3_report)
    return execution_strategy_diagnostic(
        args.plan.resolve(),
        seeds=args.seeds,
        maximum_steps=args.maximum_steps,
        actions_per_inference=args.actions_per_inference,
    )


if __name__ == "__main__":
    raise SystemExit(main())
