"""Evaluate the SmolVLA base or a formal checkpoint on the fixed validation protocol."""

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
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from safetensors.torch import load_file
from torch.utils.data import default_collate

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
DEFAULT_PLAN = REPOSITORY_ROOT / "configs/vla/smolvla_450m_aloha_insertion_formal_001.yaml"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_smolvla_formal as formal_runner  # noqa: E402
import run_smolvla_phase as phase_runner  # noqa: E402

from rosetta_reality.experiment import file_sha256, workspace_code_identity  # noqa: E402
from rosetta_reality.features import create_json  # noqa: E402
from rosetta_reality.tracking import sanitize_metrics, validate_public_payload  # noqa: E402


def _autocast_dtype(name: str) -> torch.dtype | None:
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "float32": None,
        "no": None,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported configured mixed precision: {name}.")
    return mapping[name]


def _checkpoint_source(
    plan: dict[str, Any],
    experiment: dict[str, Any],
    step: int,
    dataset_root: Path,
) -> tuple[Path, dict[str, Any]]:
    allowed = {value for value in plan["validation"]["checkpoints"] if isinstance(value, int)}
    if step not in allowed:
        raise ValueError("Checkpoint step is outside the preregistered validation protocol.")
    checkpoint_root = phase_runner._absolute_root("ROSETTA_CHECKPOINT_ROOT")
    step_dir = (
        checkpoint_root
        / str(experiment["experiment_id"])
        / "formal"
        / str(plan["run_name"])
        / "checkpoints"
        / f"{step:06d}"
    )
    pretrained_dir = step_dir / "pretrained_model"
    training_state_dir = step_dir / "training_state"
    required = [
        pretrained_dir / "config.json",
        pretrained_dir / "model.safetensors",
        pretrained_dir / "policy_preprocessor.json",
        pretrained_dir / "policy_postprocessor.json",
        pretrained_dir / "policy_preprocessor_step_5_normalizer_processor.safetensors",
        pretrained_dir / "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        pretrained_dir / "train_config.json",
        training_state_dir / "training_step.json",
    ]
    if any(not path.is_file() or path.stat().st_size <= 0 for path in required):
        raise FileNotFoundError("Formal checkpoint files are missing or empty.")
    train_config = formal_runner._load_json(pretrained_dir / "train_config.json")
    training_step = formal_runner._load_json(training_state_dir / "training_step.json")
    policy = train_config.get("policy", {})
    dataset = train_config.get("dataset", {})
    training = plan["training"]
    expected_output = step_dir.parents[1]
    if (
        training_step.get("step") != step
        or dataset.get("repo_id") != experiment["dataset"]["identifier"]
        or dataset.get("revision") != experiment["dataset"]["revision"]
        or dataset.get("episodes") != training["episodes"]
        or Path(str(dataset.get("root"))).resolve() != dataset_root
        or train_config.get("job_name") != plan["run_name"]
        or train_config.get("seed") != experiment["seed"]
        or train_config.get("steps") != training["steps"]
        or train_config.get("save_freq") != training["save_freq"]
        or train_config.get("batch_size") != training["batch_size"]
        or Path(str(train_config.get("output_dir"))).resolve() != expected_output
        or policy.get("type") != "smolvla"
        or policy.get("pretrained_revision") != experiment["model"]["revision"]
        or policy.get("load_vlm_weights") is not False
    ):
        raise ValueError("Formal checkpoint identity differs from the preregistered run.")
    return pretrained_dir, {
        "kind": "checkpoint",
        "step": step,
        "path": step_dir.relative_to(checkpoint_root).as_posix(),
        "model_safetensors_sha256": file_sha256(pretrained_dir / "model.safetensors"),
        "policy_config_sha256": file_sha256(pretrained_dir / "config.json"),
        "preprocessor_config_sha256": file_sha256(
            pretrained_dir / "policy_preprocessor.json"
        ),
        "postprocessor_config_sha256": file_sha256(
            pretrained_dir / "policy_postprocessor.json"
        ),
    }


def _assert_tensor_equal(actual: torch.Tensor, expected: Any, name: str) -> None:
    expected_tensor = torch.as_tensor(expected, dtype=actual.dtype)
    if actual.shape != expected_tensor.shape or not torch.equal(actual.cpu(), expected_tensor):
        raise ValueError(f"Saved checkpoint processor statistic differs: {name}.")


def _validate_checkpoint_statistics(
    pretrained_dir: Path, normalization_report: dict[str, Any]
) -> dict[str, str]:
    pre_path = pretrained_dir / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    post_path = pretrained_dir / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    pre = load_file(pre_path, device="cpu")
    post = load_file(post_path, device="cpu")
    expected = dict(normalization_report["effective_stats"])
    expected.update(
        {
            feature: normalization_report["visual_statistics"]
            for feature in normalization_report["visual_features"]
        }
    )
    for feature, statistics in expected.items():
        if not isinstance(statistics, dict):
            raise ValueError("Normalization report statistics must be mappings.")
        for statistic, value in statistics.items():
            key = f"{feature}.{statistic}"
            if key not in pre:
                raise ValueError(f"Saved checkpoint processor statistic is missing: {key}.")
            _assert_tensor_equal(pre[key], value, key)
    for statistic, value in normalization_report["effective_stats"]["action"].items():
        key = f"action.{statistic}"
        if key not in post:
            raise ValueError(f"Saved checkpoint postprocessor statistic is missing: {key}.")
        _assert_tensor_equal(post[key], value, key)
    return {
        "preprocessor_statistics_sha256": file_sha256(pre_path),
        "postprocessor_statistics_sha256": file_sha256(post_path),
    }


def _validation_indices(
    dataset: LeRobotDataset, episodes: list[int], offsets: list[int]
) -> list[tuple[int, int, int]]:
    episode_ids = [int(value) for value in dataset.meta.episodes["episode_index"]]
    starts = [int(value) for value in dataset.meta.episodes["dataset_from_index"]]
    lengths = [int(value) for value in dataset.meta.episodes["length"]]
    metadata = {
        episode: (start, length)
        for episode, start, length in zip(episode_ids, starts, lengths, strict=True)
    }
    selected: list[tuple[int, int, int]] = []
    for episode in episodes:
        if episode not in metadata:
            raise ValueError("A validation episode is absent from dataset metadata.")
        start, length = metadata[episode]
        for offset in offsets:
            if offset < 0 or offset >= length:
                raise ValueError("A validation frame offset is outside its episode.")
            absolute = start + offset
            if absolute not in dataset.absolute_to_relative_idx:
                raise ValueError("A validation frame is absent from the selected dataset view.")
            selected.append((episode, offset, dataset.absolute_to_relative_idx[absolute]))
    return selected


def _load_policy_and_dataset(
    plan: dict[str, Any],
    experiment: dict[str, Any],
    dataset_root: Path,
    checkpoint_step: int | None,
) -> tuple[Any, Any, Any, LeRobotDataset, dict[str, Any], dict[str, str]]:
    device_name = os.environ.get("ROSETTA_TORCH_DEVICE")
    if not device_name:
        raise ValueError("ROSETTA_TORCH_DEVICE must be set by the Docker runner.")
    device = torch.device(device_name)
    if device.type == "xpu" and not torch.xpu.is_available():
        raise RuntimeError("The requested XPU is unavailable.")

    if checkpoint_step is None:
        source_dir = phase_runner._model_root(experiment)
        source_identity = {
            "kind": "base",
            "model_id": experiment["model"]["identifier"],
            "model_revision": experiment["model"]["revision"],
            "model_manifest_sha256": file_sha256(source_dir / "model_manifest.json"),
            "model_safetensors_sha256": file_sha256(source_dir / "model.safetensors"),
        }
        processor_hashes: dict[str, str] = {}
        revision: str | None = experiment["model"]["revision"]
    else:
        source_dir, source_identity = _checkpoint_source(
            plan, experiment, checkpoint_step, dataset_root
        )
        normalization = formal_runner._load_json(
            formal_runner._repository_path(plan["normalization"]["report"])
        )
        processor_hashes = _validate_checkpoint_statistics(source_dir, normalization)
        revision = None

    policy_cfg = SmolVLAConfig.from_pretrained(
        source_dir,
        revision=revision,
        local_files_only=True,
    )
    registered_policy = experiment["model"]["policy"]
    adaptation = experiment["model"]["adaptation"]
    policy_cfg.device = device.type
    policy_cfg.pretrained_path = source_dir
    policy_cfg.pretrained_revision = revision
    policy_cfg.chunk_size = int(registered_policy["chunk_size"])
    policy_cfg.n_action_steps = int(registered_policy["n_action_steps"])
    policy_cfg.empty_cameras = int(registered_policy["empty_cameras"])
    policy_cfg.load_vlm_weights = bool(registered_policy["load_vlm_weights"])
    policy_cfg.freeze_vision_encoder = bool(adaptation["freeze_vision_encoder"])
    policy_cfg.train_expert_only = bool(adaptation["train_expert_only"])
    policy_cfg.train_state_proj = bool(adaptation["train_state_proj"])

    metadata = LeRobotDatasetMetadata(
        experiment["dataset"]["identifier"],
        root=dataset_root,
        revision=experiment["dataset"]["revision"],
    )
    delta_timestamps = resolve_delta_timestamps(policy_cfg, metadata)
    dataset = LeRobotDataset(
        experiment["dataset"]["identifier"],
        root=dataset_root,
        episodes=[int(value) for value in plan["validation"]["episodes"]],
        delta_timestamps=delta_timestamps,
        revision=experiment["dataset"]["revision"],
        download_videos=False,
        return_uint8=True,
    )
    policy = make_policy(
        cfg=policy_cfg,
        ds_meta=dataset.meta,
        rename_map=experiment["dataset"]["rename_map"],
    )
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=source_dir,
        pretrained_revision=revision,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
                "stats": dataset.meta.stats,
            },
            "rename_observations_processor": {
                "rename_map": experiment["dataset"]["rename_map"]
            },
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
                "stats": dataset.meta.stats,
            }
        },
    )
    return policy, preprocessor, postprocessor, dataset, source_identity, processor_hashes


def _sync(device: torch.device) -> None:
    if device.type == "xpu":
        torch.xpu.synchronize()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("Cannot summarize an empty metric list.")
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1)
    return ordered[max(index, 0)]


def _log_validation(
    experiment: dict[str, Any],
    plan: dict[str, Any],
    run_name: str,
    source_kind: str,
    source_step: int,
    plan_sha256: str,
    metrics: dict[str, int | float],
) -> None:
    import trackio

    public_config = {
        "experiment_id": experiment["experiment_id"],
        "role": "vla",
        "phase": "formal_validation",
        "source_kind": source_kind,
        "source_step": source_step,
        "model_id": experiment["model"]["identifier"],
        "model_revision": experiment["model"]["revision"],
        "dataset_id": experiment["dataset"]["identifier"],
        "dataset_revision": experiment["dataset"]["revision"],
        "formal_plan_sha256": plan_sha256,
        "normalization_sha256": plan["normalization"]["report_sha256"],
        "validation_sample_count": plan["validation"]["total_samples"],
        "test_split_loaded": False,
    }
    validate_public_payload(public_config, context="formal_validation_config")
    payload = sanitize_metrics(metrics, mode="eval")
    trackio.init(
        project=experiment["tracking"]["project"],
        name=run_name,
        group=f"{experiment['experiment_id']}-formal-validation",
        config=public_config,
        resume="never",
        embed=False,
        auto_log_cpu=False,
        auto_log_gpu=False,
    )
    try:
        trackio.log(payload, step=source_step)
    finally:
        trackio.finish()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int)
    args = parser.parse_args()
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("HF_DATASETS_OFFLINE") != "1":
        raise RuntimeError("Formal validation must run with networking disabled.")

    plan_path = args.plan.resolve()
    plan, base_path, experiment = formal_runner._validate_plan(plan_path)
    resources = plan["resources"]
    if (
        os.environ.get("ROSETTA_DOCKER_MEMORY_LIMIT") != resources["memory_limit"]
        or os.environ.get("ROSETTA_DOCKER_MEMORY_SWAP_LIMIT")
        != resources["memory_swap_limit"]
    ):
        raise ValueError("The active Docker memory limits differ from the formal plan.")
    contract_path = REPOSITORY_ROOT / str(experiment["action_contract"]["derived"])
    contract = formal_runner._load_yaml(contract_path)
    contract_sha256 = file_sha256(contract_path)
    formal_runner._validate_prerequisites(plan, experiment, base_path, contract_sha256)
    normalization_path, view_manifest_path, dataset_root = formal_runner._validate_normalization(
        plan, experiment, base_path, contract_sha256
    )
    formal_runner._validate_preflight(
        args.preflight_report.resolve(),
        plan,
        experiment,
        base_path,
        contract_sha256,
        file_sha256(normalization_path),
        file_sha256(plan_path),
    )

    validation = plan["validation"]
    source_label = "base" if args.checkpoint_step is None else f"step-{args.checkpoint_step:06d}"
    run_name = f"{validation['run_name_prefix']}-{source_label}"
    run_root = phase_runner._absolute_root("ROSETTA_RUN_ROOT")
    destination = (
        run_root / str(experiment["experiment_id"]) / "validation" / f"{run_name}.json"
    )
    if destination.exists():
        raise FileExistsError("The formal validation report is create-only.")

    started = time.perf_counter()
    torch.manual_seed(int(experiment["seed"]))
    policy, preprocessor, postprocessor, dataset, source, processor_hashes = (
        _load_policy_and_dataset(plan, experiment, dataset_root, args.checkpoint_step)
    )
    device = torch.device(str(os.environ["ROSETTA_TORCH_DEVICE"]))
    episodes = [int(value) for value in validation["episodes"]]
    hidden_test = {int(value) for value in experiment["dataset"]["test_episodes"]}
    if set(episodes) & hidden_test:
        raise ValueError("The validation protocol intersects the sealed hidden-test split.")
    indices = _validation_indices(
        dataset,
        episodes,
        [int(value) for value in validation["frame_offsets"]],
    )
    if len(indices) != int(validation["total_samples"]):
        raise ValueError("The selected validation sample count differs from the formal plan.")

    action_spec = contract["action"]
    action_dimension = int(action_spec["dimension"])
    chunk_length = int(action_spec["chunk_length"])
    limits = torch.tensor(
        [
            [float(dimension["minimum"]), float(dimension["maximum"])]
            for dimension in action_spec["dimensions"]
        ],
        dtype=torch.float64,
    )
    if limits.shape != (action_dimension, 2):
        raise ValueError("The Action Contract limit table has an invalid shape.")

    policy.eval()
    mixed_precision = str(resources["mixed_precision"])
    autocast_dtype = _autocast_dtype(mixed_precision)
    if device.type == "xpu":
        torch.xpu.reset_peak_memory_stats()
    absolute_error = 0.0
    squared_error = 0.0
    first_action_error = 0.0
    error_elements = 0
    first_action_elements = 0
    invalid_actions = 0
    limit_violations = 0
    predicted_elements = 0
    smoothness_sum = 0.0
    smoothness_elements = 0
    losses: list[float] = []
    latencies: list[float] = []
    materialized_episodes: set[int] = set()

    for expected_episode, _, relative_index in indices:
        sample = dataset[relative_index]
        batch = default_collate([sample])
        episode_value = batch.get("episode_index")
        raw_action = batch.get("action")
        action_is_pad = batch.get("action_is_pad")
        if (
            not isinstance(episode_value, torch.Tensor)
            or int(episode_value.item()) != expected_episode
            or not isinstance(raw_action, torch.Tensor)
            or list(raw_action.shape) != [1, chunk_length, action_dimension]
            or not isinstance(action_is_pad, torch.Tensor)
            or bool(action_is_pad.any().item())
        ):
            raise ValueError("A fixed validation sample differs from the registered contract.")
        materialized_episodes.add(expected_episode)
        raw_action = raw_action.detach().cpu().to(torch.float64).clone()
        for camera_key in dataset.meta.camera_keys:
            if camera_key in batch and batch[camera_key].dtype == torch.uint8:
                maximum = torch.iinfo(batch[camera_key].dtype).max
                batch[camera_key] = (
                    batch[camera_key].to(dtype=torch.get_default_dtype()) / maximum
                )
        batch = preprocessor(batch)
        normalized_action = batch.get("action")
        if not isinstance(normalized_action, torch.Tensor):
            raise ValueError("The processed validation batch has no action tensor.")
        noise = torch.zeros(
            (1, chunk_length, policy.config.max_action_dim),
            device=device,
            dtype=normalized_action.dtype,
        )
        flow_time = torch.full(
            (1,),
            float(validation["flow_time"]),
            device=device,
            dtype=normalized_action.dtype,
        )
        policy.reset()
        _sync(device)
        inference_started = time.perf_counter()
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ),
        ):
            predicted = policy.predict_action_chunk(batch, noise=noise)
            loss, _ = policy(batch, noise=noise, time=flow_time)
        _sync(device)
        latencies.append(time.perf_counter() - inference_started)
        predicted = postprocessor(predicted)
        if not isinstance(predicted, torch.Tensor) or list(predicted.shape) != [
            1,
            chunk_length,
            action_dimension,
        ]:
            raise ValueError("SmolVLA produced an invalid validation action chunk shape.")
        predicted = predicted.detach().cpu().to(torch.float64)
        finite = torch.isfinite(predicted)
        invalid_actions += int((~finite).sum().item())
        if not bool(finite.all().item()):
            raise FloatingPointError("SmolVLA produced a non-finite validation action.")
        error = predicted - raw_action
        absolute_error += float(error.abs().sum().item())
        squared_error += float(error.square().sum().item())
        first_action_error += float(error[:, 0].abs().sum().item())
        error_elements += error.numel()
        first_action_elements += error[:, 0].numel()
        below = predicted < limits[:, 0].view(1, 1, -1)
        above = predicted > limits[:, 1].view(1, 1, -1)
        limit_violations += int((below | above).sum().item())
        predicted_elements += predicted.numel()
        differences = predicted[:, 1:] - predicted[:, :-1]
        smoothness_sum += float(differences.abs().sum().item())
        smoothness_elements += differences.numel()
        loss_value = float(loss.detach().cpu().item())
        if not math.isfinite(loss_value):
            raise FloatingPointError("SmolVLA produced a non-finite fixed-flow validation loss.")
        losses.append(loss_value)

    metrics: dict[str, int | float] = {
        "action_mae": absolute_error / error_elements,
        "action_rmse": math.sqrt(squared_error / error_elements),
        "first_action_mae": first_action_error / first_action_elements,
        "fixed_flow_loss": sum(losses) / len(losses),
        "invalid_action_rate": invalid_actions / predicted_elements,
        "joint_limit_violation_rate": limit_violations / predicted_elements,
        "action_smoothness_mean_abs_delta": smoothness_sum / smoothness_elements,
        "inference_latency_mean_seconds": sum(latencies) / len(latencies),
        "inference_latency_p95_seconds": _percentile(latencies, 0.95),
        "sample_count": len(indices),
        "test_split_loaded": 0,
    }
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise FloatingPointError("Formal validation metrics contain a non-finite value.")
    source_step = int(args.checkpoint_step or 0)
    _log_validation(
        experiment,
        plan,
        run_name,
        str(source["kind"]),
        source_step,
        file_sha256(plan_path),
        metrics,
    )
    accelerator_memory = (
        {
            "allocated_bytes": int(torch.xpu.memory_allocated()),
            "reserved_bytes": int(torch.xpu.memory_reserved()),
            "maximum_allocated_bytes": int(torch.xpu.max_memory_allocated()),
        }
        if device.type == "xpu"
        else {}
    )
    report = {
        "schema_version": 1,
        "status": "complete",
        "stage": "smolvla_fixed_validation",
        "experiment_id": experiment["experiment_id"],
        "run_name": run_name,
        "formal_plan_sha256": file_sha256(plan_path),
        "experiment_config_sha256": file_sha256(base_path),
        "action_contract_sha256": contract_sha256,
        "normalization_report_sha256": file_sha256(normalization_path),
        "dataset_view_manifest_sha256": file_sha256(view_manifest_path),
        "evaluation_script_sha256": file_sha256(Path(__file__)),
        "code_identity": workspace_code_identity(REPOSITORY_ROOT),
        "model_source": source,
        "processor_statistics": processor_hashes,
        "model_revision": experiment["model"]["revision"],
        "dataset_revision": experiment["dataset"]["revision"],
        "validation_episodes": episodes,
        "frame_offsets": [int(value) for value in validation["frame_offsets"]],
        "materialized_episodes": sorted(materialized_episodes),
        "sample_count": len(indices),
        "hidden_test_loaded": False,
        "network_disabled": True,
        "gradients_enabled": False,
        "optimizer_created": False,
        "fixed_input": {
            "noise": validation["noise"],
            "flow_time": float(validation["flow_time"]),
        },
        "metrics": metrics,
        "device": device.type,
        "mixed_precision": mixed_precision,
        "accelerator_memory": accelerator_memory,
        "trackio_local_logged": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    json.dumps(report, allow_nan=False)
    create_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report: {destination.relative_to(REPOSITORY_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
