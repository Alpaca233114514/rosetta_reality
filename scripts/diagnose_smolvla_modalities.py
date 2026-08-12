"""Run teacher-forced SmolVLA image/state ablations on fixed validation contexts."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import default_collate

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import evaluate_smolvla_validation as evaluator  # noqa: E402
import run_smolvla_formal as formal_runner  # noqa: E402
import run_smolvla_phase as phase_runner  # noqa: E402

from rosetta_reality.eval.diagnostics import (  # noqa: E402
    action_dimension_diagnostics,
    action_error_summary,
    cross_episode_shuffle_indices,
)
from rosetta_reality.experiment import (  # noqa: E402
    file_sha256,
    stable_hash,
    workspace_code_identity,
)
from rosetta_reality.features import create_json  # noqa: E402
from rosetta_reality.sim import load_action_contract  # noqa: E402

DEFAULT_PLAN = (
    REPOSITORY_ROOT / "configs/vla/smolvla_450m_aloha_insertion_moby_dick_001.yaml"
)


def _clone_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    return value


def _load_historical_plan(plan_path: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Keep historical identity checks while allowing a new non-gating diagnostic script."""

    plan = formal_runner._load_formal_plan(plan_path)
    parent = plan.get("parent_experiment")
    if not isinstance(parent, dict):
        raise ValueError("Historical formal plan has no parent experiment identity.")
    base_path = formal_runner._repository_path(str(parent.get("config", "")))
    if file_sha256(base_path) != parent.get("sha256"):
        raise ValueError("Historical parent experiment checksum changed.")
    experiment = formal_runner._load_yaml(base_path)
    if experiment.get("experiment_id") != parent.get("experiment_id"):
        raise ValueError("Historical parent experiment identifier changed.")
    return plan, base_path, experiment


def _load_uncompiled_policy_and_dataset(
    plan: dict[str, Any],
    experiment: dict[str, Any],
    dataset_root: Path,
    checkpoint_step: int,
) -> tuple[Any, Any, Any, Any, dict[str, Any], dict[str, str]]:
    """Disable the checkpoint's performance-only compile flag for this diagnostic."""

    config_class = evaluator.SmolVLAConfig

    class DiagnosticConfigLoader:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> Any:
            config = config_class.from_pretrained(*args, **kwargs)
            config.compile_model = False
            return config

    evaluator.SmolVLAConfig = DiagnosticConfigLoader
    try:
        return evaluator._load_policy_and_dataset(
            plan, experiment, dataset_root, checkpoint_step
        )
    finally:
        evaluator.SmolVLAConfig = config_class


def _perturb_sample(
    samples: list[dict[str, Any]],
    destination: int,
    source: int,
    condition: str,
    camera_keys: list[str],
) -> dict[str, Any]:
    sample = _clone_tree(samples[destination])
    if condition == "normal":
        return sample
    if condition == "image_shuffle":
        for key in camera_keys:
            if key not in samples[source]:
                raise ValueError(f"Image-shuffle source is missing {key}.")
            sample[key] = _clone_tree(samples[source][key])
        return sample
    if condition == "state_shuffle":
        state = samples[source].get("observation.state")
        if not isinstance(state, torch.Tensor):
            raise ValueError("State-shuffle source has no robot state tensor.")
        sample["observation.state"] = state.clone()
        return sample
    if condition == "image_zero":
        for key in camera_keys:
            image = sample.get(key)
            if not isinstance(image, torch.Tensor):
                raise ValueError(f"Zero-image sample is missing {key}.")
            sample[key] = torch.zeros_like(image)
        return sample
    raise ValueError(f"Unsupported SmolVLA modality condition: {condition}.")


def _evaluate_condition(
    condition: str,
    samples: list[dict[str, Any]],
    shuffle: torch.Tensor,
    camera_keys: list[str],
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    *,
    flow_time_value: float,
    mixed_precision: str,
) -> tuple[torch.Tensor, list[float], list[float]]:
    device = torch.device(str(os.environ["ROSETTA_TORCH_DEVICE"]))
    autocast_dtype = evaluator._autocast_dtype(mixed_precision)
    predictions: list[torch.Tensor] = []
    losses: list[float] = []
    latencies: list[float] = []
    policy.eval()
    for destination in range(len(samples)):
        source = int(shuffle[destination])
        sample = _perturb_sample(samples, destination, source, condition, camera_keys)
        batch = default_collate([sample])
        for camera_key in camera_keys:
            value = batch.get(camera_key)
            if isinstance(value, torch.Tensor) and value.dtype == torch.uint8:
                batch[camera_key] = (
                    value.to(torch.get_default_dtype()) / torch.iinfo(value.dtype).max
                )
        batch = preprocessor(batch)
        action = batch.get("action")
        if not isinstance(action, torch.Tensor):
            raise ValueError("Teacher-forced diagnostic batch has no action tensor.")
        noise = torch.zeros(
            (1, policy.config.chunk_size, policy.config.max_action_dim),
            device=device,
            dtype=action.dtype,
        )
        flow_time = torch.full(
            (1,), flow_time_value, device=device, dtype=action.dtype
        )
        inference_batch = _clone_tree(batch)
        loss_batch = _clone_tree(batch)
        policy.reset()
        evaluator._sync(device)
        started = time.perf_counter()
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ),
        ):
            prediction = policy.predict_action_chunk(inference_batch, noise=noise)
            loss, _ = policy(loss_batch, noise=noise, time=flow_time)
        evaluator._sync(device)
        latencies.append(time.perf_counter() - started)
        prediction = postprocessor(prediction)
        if not isinstance(prediction, torch.Tensor) or not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError("Teacher-forced diagnostic produced an invalid action.")
        loss_value = float(loss.detach().cpu())
        if not math.isfinite(loss_value):
            raise FloatingPointError("Teacher-forced diagnostic produced a non-finite loss.")
        predictions.append(prediction.detach().cpu().to(torch.float64))
        losses.append(loss_value)
    return torch.cat(predictions), losses, latencies


def diagnose(plan_path: Path, checkpoint_step: int, shuffle_seed: int) -> Path:
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("HF_DATASETS_OFFLINE") != "1":
        raise RuntimeError("SmolVLA modality diagnostics require networking disabled.")
    plan, base_path, experiment = _load_historical_plan(plan_path)
    contract_path = REPOSITORY_ROOT / str(experiment["action_contract"]["derived"])
    contract = load_action_contract(contract_path)
    contract_sha256 = file_sha256(contract_path)
    normalization_path, view_manifest_path, dataset_root = formal_runner._validate_normalization(
        plan, experiment, base_path, contract_sha256
    )
    policy, preprocessor, postprocessor, dataset, source, processor_hashes = (
        _load_uncompiled_policy_and_dataset(
            plan, experiment, dataset_root, checkpoint_step
        )
    )
    validation = plan["validation"]
    episodes = [int(value) for value in validation["episodes"]]
    offsets = [int(value) for value in validation["frame_offsets"]]
    indices = evaluator._validation_indices(dataset, episodes, offsets)
    if len(indices) != int(validation["total_samples"]):
        raise ValueError("Teacher-forced sample count differs from the historical protocol.")
    samples = [_clone_tree(dataset[relative]) for _, _, relative in indices]
    episode_tensor = torch.tensor([episode for episode, _, _ in indices], dtype=torch.int64)
    frame_tensor = torch.tensor([offset for _, offset, _ in indices], dtype=torch.int64)
    shuffle = cross_episode_shuffle_indices(
        episode_tensor,
        frame_indices=frame_tensor,
        seed=shuffle_seed,
    )
    targets = torch.cat(
        [
            sample["action"].detach().cpu().to(torch.float64).unsqueeze(0)
            for sample in samples
        ]
    )
    camera_keys = [str(value) for value in dataset.meta.camera_keys]
    if not camera_keys:
        raise ValueError("Teacher-forced diagnostic dataset has no camera features.")
    conditions = ["normal", "image_shuffle", "state_shuffle", "image_zero"]
    predictions: dict[str, torch.Tensor] = {}
    condition_reports: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for condition in conditions:
        predicted, losses, latencies = _evaluate_condition(
            condition,
            samples,
            shuffle,
            camera_keys,
            policy,
            preprocessor,
            postprocessor,
            flow_time_value=float(validation["flow_time"]),
            mixed_precision=str(plan["resources"]["mixed_precision"]),
        )
        predictions[condition] = predicted
        condition_reports[condition] = {
            **action_error_summary(predicted, targets),
            "fixed_flow_loss": sum(losses) / len(losses),
            "mean_inference_and_loss_seconds": sum(latencies) / len(latencies),
        }
    normal = predictions["normal"]
    for condition in conditions[1:]:
        shift = action_error_summary(predictions[condition], normal)
        condition_reports[condition]["prediction_shift_from_normal"] = shift
        condition_reports[condition]["chunk_mae_delta_from_normal"] = (
            condition_reports[condition]["chunk_mae"]
            - condition_reports["normal"]["chunk_mae"]
        )

    lower = contract.lower_bounds.to(torch.float64)
    upper = contract.upper_bounds.to(torch.float64)
    report = {
        "schema_version": 1,
        "status": "complete",
        "stage": "smolvla_teacher_forced_modality_diagnostic",
        "gating_evidence": False,
        "experiment_id": experiment["experiment_id"],
        "historical_plan_id": plan["plan_id"],
        "historical_plan_sha256": file_sha256(plan_path),
        "historical_plan_implementation_hash_enforced": False,
        "historical_parent_experiment_sha256": file_sha256(base_path),
        "checkpoint_step": checkpoint_step,
        "model_source": source,
        "processor_statistics": processor_hashes,
        "action_contract_sha256": contract_sha256,
        "normalization_report_sha256": file_sha256(normalization_path),
        "dataset_view_manifest_sha256": file_sha256(view_manifest_path),
        "dataset_revision": experiment["dataset"]["revision"],
        "validation_episodes": episodes,
        "frame_offsets": offsets,
        "sample_count": len(samples),
        "teacher_forced_observations": True,
        "next_observation_source": "expert_dataset_not_policy_rollout",
        "checkpoint_compile_model": bool(plan["training"]["policy"]["compile_model"]),
        "diagnostic_compile_model": False,
        "shuffle": {
            "seed": shuffle_seed,
            "policy": "cross_episode_same_frame_offset_derangement",
            "source_indices": shuffle.tolist(),
            "source_episodes": episode_tensor[shuffle].tolist(),
        },
        "conditions": condition_reports,
        "normal_action_diagnostics": action_dimension_diagnostics(
            normal,
            targets,
            lower,
            upper,
            contract.dimension_names,
        ),
        "gradients_enabled": False,
        "optimizer_created": False,
        "hidden_test_loaded": False,
        "network_disabled": True,
        "elapsed_seconds": time.perf_counter() - started,
        "diagnostic_script_sha256": file_sha256(Path(__file__)),
        "code_identity": workspace_code_identity(REPOSITORY_ROOT),
    }
    json.dumps(report, allow_nan=False)
    identity = {
        "plan": file_sha256(plan_path),
        "checkpoint_step": checkpoint_step,
        "shuffle_seed": shuffle_seed,
        "script": file_sha256(Path(__file__)),
    }
    digest = stable_hash(identity)[:16]
    destination = (
        phase_runner._absolute_root("ROSETTA_RUN_ROOT")
        / str(experiment["experiment_id"])
        / "diagnostics"
        / f"teacher-forced-modalities-step{checkpoint_step:06d}-{digest}.json"
    )
    create_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report: {destination.relative_to(REPOSITORY_ROOT).as_posix()}")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--checkpoint-step", type=int, default=1260)
    parser.add_argument("--shuffle-seed", type=int, default=20260812)
    args = parser.parse_args()
    diagnose(args.plan.resolve(), args.checkpoint_step, args.shuffle_seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
