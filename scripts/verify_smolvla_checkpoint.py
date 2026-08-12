"""Independently reload a saved SmolVLA checkpoint and verify its real-data contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401
from torch.utils.data import DataLoader, Subset

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs/vla/smolvla_450m_aloha_insertion.yaml"
RUN_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{2,79}")
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rosetta_reality.eval.diagnostics import action_dimension_diagnostics  # noqa: E402
from rosetta_reality.experiment import file_sha256  # noqa: E402
from rosetta_reality.features import create_json  # noqa: E402
from rosetta_reality.sim import load_action_contract as load_physical_contract  # noqa: E402
from rosetta_reality.vla import (  # noqa: E402
    load_smolvla_action_space,
    load_smolvla_experiment,
)
from rosetta_reality.vla.fixed_samples import (  # noqa: E402
    load_fixed_frame_protocol,
    resolve_fixed_dataset_indices,
)
from rosetta_reality.vla.processor import (  # noqa: E402
    BOUNDED_SINE_ACTION_ADAPTER,
    PI_ALOHA_POSTPROCESSOR_REGISTRY_NAME,
    PI_ALOHA_PREPROCESSOR_REGISTRY_NAME,
    REGISTRY_NAME,
    ensure_smolvla_action_boundary,
    model_action_to_standard,
    processor_state_path,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping: {path.name}.")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object: {path.name}.")
    json.dumps(value, allow_nan=False)
    return value


def _load_action_contract(experiment: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = REPOSITORY_ROOT / str(experiment["action_contract"]["derived"])
    value = _load_yaml(path)
    if not isinstance(value.get("action"), dict):
        raise ValueError("The derived Action Contract must contain an action mapping.")
    return path, value


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


def _feature_dimension(dataset: Any, key: str) -> int:
    feature = dataset.meta.features.get(key)
    if not isinstance(feature, dict):
        raise ValueError(f"Dataset metadata is missing {key}.")
    shape = feature.get("shape")
    if not isinstance(shape, list | tuple) or not shape:
        raise ValueError(f"Dataset metadata has no usable shape for {key}.")
    dimension = math.prod(int(value) for value in shape)
    if dimension <= 0:
        raise ValueError(f"Dataset metadata has an invalid dimension for {key}.")
    return dimension


def _absolute_root(environment: str) -> Path:
    raw = os.environ.get(environment)
    if not raw:
        raise ValueError(f"{environment} must be set by the Docker runner.")
    root = Path(raw)
    if not root.is_absolute():
        raise ValueError(f"{environment} must be absolute.")
    return root.resolve()


def _checkpoint_paths(
    checkpoint: Path,
    experiment_id: str,
    phase: str,
    *,
    require_last: bool,
) -> tuple[Path, Path, Path]:
    checkpoint_root = _absolute_root("ROSETTA_CHECKPOINT_ROOT")
    candidate = checkpoint.resolve()
    step_dir = candidate.parent if candidate.name == "pretrained_model" else candidate
    if not step_dir.is_relative_to(checkpoint_root):
        raise ValueError("Checkpoint must remain inside the mounted checkpoint root.")
    relative = step_dir.relative_to(checkpoint_root)
    if (
        len(relative.parts) != 5
        or relative.parts[0] != experiment_id
        or relative.parts[1] != phase
        or relative.parts[3] != "checkpoints"
        or not relative.parts[4].isdigit()
    ):
        raise ValueError("Checkpoint does not match the registered phase-run layout.")
    pretrained_dir = step_dir / "pretrained_model"
    training_state_dir = step_dir / "training_state"
    if not pretrained_dir.is_dir() or not training_state_dir.is_dir():
        raise NotADirectoryError("Checkpoint model or training-state directory is missing.")
    last = step_dir.parent / "last"
    if require_last and (not last.is_symlink() or last.resolve() != step_dir):
        raise ValueError("The checkpoint is not the run's immutable last checkpoint.")
    return step_dir, pretrained_dir, training_state_dir


def _validate_checkpoint_files(pretrained_dir: Path, training_state_dir: Path) -> list[str]:
    preprocessor_state = processor_state_path(
        pretrained_dir,
        pipeline_config_filename="policy_preprocessor.json",
        registry_name="normalizer_processor",
    )
    postprocessor_state = processor_state_path(
        pretrained_dir,
        pipeline_config_filename="policy_postprocessor.json",
        registry_name="unnormalizer_processor",
    )
    required = [
        pretrained_dir / "config.json",
        pretrained_dir / "model.safetensors",
        pretrained_dir / "policy_preprocessor.json",
        pretrained_dir / "policy_postprocessor.json",
        preprocessor_state,
        postprocessor_state,
        pretrained_dir / "tokenizer/tokenizer.json",
        pretrained_dir / "tokenizer/tokenizer_config.json",
        pretrained_dir / "train_config.json",
        training_state_dir / "optimizer_param_groups.json",
        training_state_dir / "optimizer_state.safetensors",
        training_state_dir / "rng_state.safetensors",
        training_state_dir / "scheduler_state.json",
        training_state_dir / "training_step.json",
    ]
    missing = [path.name for path in required if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise FileNotFoundError(f"Checkpoint files are missing or empty: {sorted(missing)}.")
    return [path.relative_to(pretrained_dir.parent).as_posix() for path in required]


def _validate_saved_identity(
    train_config: dict[str, Any],
    policy_config: dict[str, Any],
    training_step: dict[str, Any],
    experiment: dict[str, Any],
    step_dir: Path,
    action_dimension: int,
    chunk_length: int,
    phase: str,
    expected_step: int,
) -> int:
    dataset = train_config.get("dataset", {})
    policy = train_config.get("policy", {})
    action_space = load_smolvla_action_space(experiment)
    phase_config = experiment["phases"][phase]
    step = training_step.get("step")
    if (
        dataset.get("repo_id") != experiment["dataset"]["identifier"]
        or dataset.get("revision") != experiment["dataset"]["revision"]
        or dataset.get("episodes") != phase_config["episodes"]
        or train_config.get("rename_map") != experiment["dataset"]["rename_map"]
        or train_config.get("seed") != experiment["seed"]
        or train_config.get("batch_size") != phase_config["batch_size"]
        or train_config.get("steps") != phase_config["steps"]
        or train_config.get("save_freq") != phase_config["save_freq"]
        or policy.get("type") != "smolvla"
        or policy.get("pretrained_revision") != experiment["model"]["revision"]
        or policy.get("chunk_size") != chunk_length
        or policy.get("n_action_steps") != experiment["model"]["policy"]["n_action_steps"]
        or policy.get("load_vlm_weights") is not False
        or bool(policy.get("adapt_to_pi_aloha", False)) != action_space.adapt_to_pi_aloha
        or bool(policy_config.get("adapt_to_pi_aloha", False))
        != action_space.adapt_to_pi_aloha
        or policy_config.get("output_features", {}).get("action", {}).get("shape")
        != [action_dimension]
        or int(policy_config.get("max_action_dim", 0)) < action_dimension
        or int(policy_config.get("max_state_dim", 0)) <= 0
        or step != expected_step
        or training_step.get("batch_size") != phase_config["batch_size"]
        or Path(str(train_config.get("output_dir"))).resolve() != step_dir.parents[1]
    ):
        raise ValueError("Saved checkpoint identity differs from the registered smoke run.")
    return int(step)


def _episode_indices(batch: dict[str, Any]) -> list[int]:
    value = batch.get("episode_index")
    if not isinstance(value, torch.Tensor):
        raise ValueError("The real-data batch has no tensor episode_index.")
    return sorted({int(item) for item in value.detach().cpu().reshape(-1).tolist()})


def _tensor_contract(batch: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key, value in sorted(batch.items()):
        if isinstance(value, torch.Tensor):
            result[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype).removeprefix("torch."),
                "device": value.device.type,
                "finite": (
                    bool(torch.isfinite(value).all().item()) if value.is_floating_point() else True
                ),
            }
    return result


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _summary(value: torch.Tensor) -> dict[str, float]:
    tensor = value.detach().to(device="cpu")
    result = {
        "minimum": float(tensor.min().item()),
        "maximum": float(tensor.max().item()),
        "mean": float(tensor.mean().item()),
        "standard_deviation": float(tensor.std(unbiased=False).item()),
    }
    if not all(math.isfinite(number) for number in result.values()):
        raise FloatingPointError("Reloaded checkpoint produced a non-finite action summary.")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase", choices=("smoke", "overfit"), required=True)
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--fixed-sample-scope",
        choices=("first", "all"),
        default="first",
        help="Evaluate the first registered anchor or the complete fixed set.",
    )
    args = parser.parse_args()
    if not RUN_NAME_PATTERN.fullmatch(args.run_name):
        raise ValueError("--run-name must be a lower-case path-safe identifier.")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("HF_DATASETS_OFFLINE") != "1":
        raise RuntimeError("Checkpoint verification must run with networking disabled.")

    config_path = args.config.resolve()
    experiment = load_smolvla_experiment(config_path, REPOSITORY_ROOT)
    action_space = load_smolvla_action_space(experiment)
    contract_path, action_contract = _load_action_contract(experiment)
    physical_contract = load_physical_contract(contract_path)
    action_spec = action_contract["action"]
    action_dimension = int(action_spec["dimension"])
    chunk_length = int(action_spec["chunk_length"])
    dimensions = action_spec.get("dimensions")
    if not isinstance(dimensions, list) or len(dimensions) != action_dimension:
        raise ValueError("Action Contract dimension metadata is inconsistent.")
    phase_config = experiment["phases"][args.phase]
    expected_step = (
        int(args.expected_step) if args.expected_step is not None else int(phase_config["steps"])
    )
    if expected_step <= 0 or expected_step > int(phase_config["steps"]):
        raise ValueError("--expected-step must fall inside the registered phase.")
    step_dir, pretrained_dir, training_state_dir = _checkpoint_paths(
        args.checkpoint,
        str(experiment["experiment_id"]),
        args.phase,
        require_last=expected_step == int(phase_config["steps"]),
    )
    required_files = _validate_checkpoint_files(pretrained_dir, training_state_dir)
    train_config = _load_json(pretrained_dir / "train_config.json")
    policy_config = _load_json(pretrained_dir / "config.json")
    training_step = _load_json(training_state_dir / "training_step.json")
    step = _validate_saved_identity(
        train_config,
        policy_config,
        training_step,
        experiment,
        step_dir,
        action_dimension,
        chunk_length,
        args.phase,
        expected_step,
    )

    expected_device = os.environ.get("ROSETTA_TORCH_DEVICE")
    if not expected_device:
        raise ValueError("ROSETTA_TORCH_DEVICE must be set by the Docker runner.")
    device = torch.device(expected_device)
    if device.type == "xpu" and not torch.xpu.is_available():
        raise RuntimeError("The requested XPU is unavailable.")
    started = time.perf_counter()
    torch.manual_seed(int(experiment["seed"]))
    cfg = TrainPipelineConfig.from_pretrained(pretrained_dir)
    if cfg.policy is None or cfg.policy.type != "smolvla":
        raise ValueError("The saved train configuration does not contain SmolVLA.")
    cfg.policy.device = device.type
    cfg.policy.pretrained_path = pretrained_dir
    cfg.policy.pretrained_revision = None
    dataset = make_dataset(cfg)
    registered_batch_size = int(phase_config["batch_size"])
    evaluation_batch_size = registered_batch_size
    state_dimension = _feature_dimension(dataset, "observation.state")
    dataset_action_dimension = _feature_dimension(dataset, "action")
    if dataset_action_dimension != action_dimension:
        raise ValueError("Dataset action dimension differs from the Action Contract.")
    if cfg.policy.chunk_size != chunk_length:
        raise ValueError("Policy chunk size differs from the Action Contract.")
    if cfg.policy.max_state_dim < state_dimension:
        raise ValueError("Policy max state dimension cannot contain the dataset state.")
    fixed_sample_context: dict[str, Any] | None = None
    fixed_sample_raw = experiment.get("repair_protocol", {}).get(
        "fixed_sample_overfit"
    )
    verification_dataset: Any = dataset
    if isinstance(fixed_sample_raw, dict):
        protocol = load_fixed_frame_protocol(experiment, args.phase)
        fixed_indices = resolve_fixed_dataset_indices(
            protocol,
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            dataset.episodes,
            dataset.absolute_to_relative_idx,
        )
        if registered_batch_size > len(fixed_indices):
            raise ValueError("Verification batch exceeds the registered fixed sample set.")
        selected_indices = (
            fixed_indices
            if args.fixed_sample_scope == "all"
            else fixed_indices[:registered_batch_size]
        )
        evaluation_batch_size = len(selected_indices)
        verification_dataset = Subset(dataset, selected_indices)
        fixed_sample_context = {
            "protocol": protocol.as_dict(),
            "scope": args.fixed_sample_scope,
            "selected_dataset_indices": selected_indices,
            "selected_frame_indices": (
                list(protocol.frame_indices)
                if args.fixed_sample_scope == "all"
                else list(protocol.frame_indices[:registered_batch_size])
            ),
        }
    elif args.fixed_sample_scope != "first":
        raise ValueError("All-anchor verification requires a fixed-sample experiment.")
    loader = DataLoader(
        verification_dataset,
        batch_size=evaluation_batch_size,
        shuffle=False,
        num_workers=0,
    )
    batch = next(iter(loader))
    raw_target = batch.get("action")
    if not isinstance(raw_target, torch.Tensor):
        raise ValueError("Checkpoint verification batch has no raw target action tensor.")
    raw_target = raw_target.clone()
    episodes_loaded = _episode_indices(batch)
    if episodes_loaded != phase_config["episodes"]:
        raise ValueError("Checkpoint verification loaded an unregistered episode.")
    if set(episodes_loaded) & set(experiment["dataset"]["test_episodes"]):
        raise ValueError("Checkpoint verification crossed the sealed hidden-test boundary.")

    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=pretrained_dir,
        preprocessor_overrides={"device_processor": {"device": device.type}},
    )
    processor_contract: dict[str, Any] = {"explicit": action_space.explicit}
    if action_space.explicit:
        ensure_smolvla_action_boundary(
            preprocessor,
            postprocessor,
            physical_contract,
            action_space,
            action_contract_sha256=file_sha256(contract_path),
            upstream_revision=str(experiment["upstream"]["revision"]),
        )
        preprocessor_steps = [
            getattr(step.__class__, "_registry_name", None)
            for step in preprocessor.steps
        ]
        postprocessor_steps = [
            getattr(step.__class__, "_registry_name", None)
            for step in postprocessor.steps
        ]
        projection_index = preprocessor_steps.index(REGISTRY_NAME)
        representation_index = preprocessor_steps.index(
            PI_ALOHA_PREPROCESSOR_REGISTRY_NAME
        )
        normalizer_index = preprocessor_steps.index("normalizer_processor")
        unnormalizer_index = postprocessor_steps.index("unnormalizer_processor")
        inverse_index = postprocessor_steps.index(PI_ALOHA_POSTPROCESSOR_REGISTRY_NAME)
        if not (
            projection_index + 1 == representation_index
            and representation_index + 1 == normalizer_index
            and unnormalizer_index + 1 == inverse_index
        ):
            raise ValueError("Reloaded action-boundary processor ordering is invalid.")
        processor_contract = {
            "explicit": True,
            "preprocessor_steps": preprocessor_steps,
            "postprocessor_steps": postprocessor_steps,
            "projection_before_representation_before_normalization": True,
            "unnormalization_before_inverse_and_clamp": True,
        }
    for camera_key in dataset.meta.camera_keys:
        if camera_key in batch and batch[camera_key].dtype == torch.uint8:
            maximum = torch.iinfo(batch[camera_key].dtype).max
            batch[camera_key] = batch[camera_key].to(dtype=torch.get_default_dtype()) / maximum
    batch = preprocessor(batch)
    tensor_contract = _tensor_contract(batch)
    if not tensor_contract or any(not item["finite"] for item in tensor_contract.values()):
        raise FloatingPointError("Reloaded checkpoint received a non-finite processed batch.")

    action = batch.get("action")
    state = batch.get("observation.state")
    expected_action_shape = [evaluation_batch_size, chunk_length, action_dimension]
    expected_state_shape = [
        evaluation_batch_size,
        cfg.policy.n_obs_steps,
        state_dimension,
    ]
    if not isinstance(action, torch.Tensor) or list(action.shape) != expected_action_shape:
        raise ValueError("Reloaded checkpoint action input differs from the Action Contract.")
    if not isinstance(state, torch.Tensor) or list(state.shape) != expected_state_shape:
        raise ValueError("Reloaded checkpoint state input differs from dataset metadata.")

    noise = torch.zeros(
        (evaluation_batch_size, chunk_length, cfg.policy.max_action_dim),
        device=device,
        dtype=action.dtype,
    )
    flow_time = torch.full(
        (evaluation_batch_size,), 0.5, device=device, dtype=action.dtype
    )
    mixed_precision = str(experiment["resources"]["mixed_precision"])
    autocast_dtype = _autocast_dtype(mixed_precision)
    autocast_enabled = autocast_dtype is not None
    policy.eval()
    policy.reset()
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=autocast_enabled,
        ),
    ):
        normalized_action_chunk = policy.predict_action_chunk(batch, noise=noise)
        loss, loss_details = policy(batch, noise=noise, time=flow_time)
    raw_action_diagnostics: dict[str, Any] | None = None
    if action_space.explicit:
        unnormalizers = [
            step
            for step in postprocessor.steps
            if getattr(step.__class__, "_registry_name", None)
            == "unnormalizer_processor"
        ]
        if len(unnormalizers) != 1:
            raise ValueError("Reloaded action boundary has an ambiguous unnormalizer.")
        unnormalized = unnormalizers[0]({"action": normalized_action_chunk})
        raw_pi_action = unnormalized.get("action")
        if not isinstance(raw_pi_action, torch.Tensor):
            raise ValueError("Reloaded unnormalizer did not produce a tensor action.")
        raw_standard_action = model_action_to_standard(
            raw_pi_action, action_space.representation_adapter
        )
        lower = physical_contract.lower_bounds.to(raw_target).view(1, 1, -1)
        upper = physical_contract.upper_bounds.to(raw_target).view(1, 1, -1)
        projected_target = torch.maximum(torch.minimum(raw_target, upper), lower)
        raw_action_diagnostics = action_dimension_diagnostics(
            raw_standard_action.detach().cpu(),
            projected_target.detach().cpu(),
            physical_contract.lower_bounds,
            physical_contract.upper_bounds,
            physical_contract.dimension_names,
        )
        raw_action_diagnostics["source_target_projection_rate"] = float(
            projected_target.ne(raw_target).to(torch.float64).mean()
        )
        if action_space.representation_adapter == BOUNDED_SINE_ACTION_ADAPTER:
            support_lower = -math.pi / 2
            support_upper = math.pi / 2
            support: dict[str, Any] = {}
            for name, index in (("left_gripper", 6), ("right_gripper", 13)):
                values = raw_pi_action[..., index].detach().to(torch.float64).cpu()
                outside = (values < support_lower) | (values > support_upper)
                support[name] = {
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                    "outside_training_support_rate": float(outside.to(torch.float64).mean()),
                    "training_support": [support_lower, support_upper],
                }
            raw_action_diagnostics["model_internal_gripper_support"] = support
        if fixed_sample_context is not None and args.fixed_sample_scope == "all":
            per_anchor: list[dict[str, Any]] = []
            for position, frame_index in enumerate(
                fixed_sample_context["selected_frame_indices"]
            ):
                diagnostics = action_dimension_diagnostics(
                    raw_standard_action[position : position + 1].detach().cpu(),
                    projected_target[position : position + 1].detach().cpu(),
                    physical_contract.lower_bounds,
                    physical_contract.upper_bounds,
                    physical_contract.dimension_names,
                )
                per_anchor.append(
                    {
                        "episode_index": episodes_loaded[0],
                        "frame_index": frame_index,
                        "left_gripper": diagnostics["dimensions"]["left_gripper"],
                        "right_gripper": diagnostics["dimensions"]["right_gripper"],
                    }
                )
            raw_action_diagnostics["fixed_sample_per_anchor"] = per_anchor
    action_chunk = postprocessor(normalized_action_chunk)
    if (
        not isinstance(action_chunk, torch.Tensor)
        or list(action_chunk.shape) != expected_action_shape
    ):
        received = list(action_chunk.shape) if isinstance(action_chunk, torch.Tensor) else None
        raise ValueError(f"Reloaded checkpoint produced an invalid action chunk: {received}.")
    if not torch.isfinite(action_chunk).all():
        raise FloatingPointError("Reloaded checkpoint produced a non-finite action chunk.")
    loss_value = float(loss.detach().cpu().item())
    if not math.isfinite(loss_value):
        raise FloatingPointError("Reloaded checkpoint produced a non-finite fixed-input loss.")
    numeric_loss_details = {
        str(key): float(value)
        for key, value in loss_details.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }
    if not all(math.isfinite(value) for value in numeric_loss_details.values()):
        raise FloatingPointError("Reloaded checkpoint produced non-finite loss details.")

    if device.type == "xpu":
        torch.xpu.synchronize()
        accelerator_memory = {
            "allocated_bytes": int(torch.xpu.memory_allocated()),
            "maximum_allocated_bytes": int(torch.xpu.max_memory_allocated()),
            "reserved_bytes": int(torch.xpu.memory_reserved()),
        }
    else:
        accelerator_memory = {}
    checkpoint_root = _absolute_root("ROSETTA_CHECKPOINT_ROOT")
    report = {
        "schema_version": 1,
        "status": "passed",
        "stage": "smolvla_checkpoint_independent_reload",
        "experiment_id": experiment["experiment_id"],
        "phase": args.phase,
        "run_name": args.run_name,
        "experiment_config_sha256": file_sha256(config_path),
        "verification_script_sha256": file_sha256(Path(__file__)),
        "action_contract_sha256": file_sha256(contract_path),
        "action_space": action_space.as_dict(),
        "serialized_action_boundary": processor_contract,
        "action_dimension": action_dimension,
        "chunk_length": chunk_length,
        "state_dimension": state_dimension,
        "checkpoint": step_dir.relative_to(checkpoint_root).as_posix(),
        "checkpoint_step": step,
        "checkpoint_files": required_files,
        "checkpoint_hashes": {
            "model_safetensors_sha256": file_sha256(pretrained_dir / "model.safetensors"),
            "optimizer_state_sha256": file_sha256(
                training_state_dir / "optimizer_state.safetensors"
            ),
            "policy_config_sha256": file_sha256(pretrained_dir / "config.json"),
            "preprocessor_config_sha256": file_sha256(pretrained_dir / "policy_preprocessor.json"),
            "postprocessor_config_sha256": file_sha256(
                pretrained_dir / "policy_postprocessor.json"
            ),
            "train_config_sha256": file_sha256(pretrained_dir / "train_config.json"),
        },
        "model_revision": experiment["model"]["revision"],
        "dataset_revision": experiment["dataset"]["revision"],
        "episodes_loaded": episodes_loaded,
        "registered_training_batch_size": registered_batch_size,
        "evaluation_batch_size": evaluation_batch_size,
        "fixed_sample_context": fixed_sample_context,
        "hidden_test_loaded": False,
        "network_disabled": True,
        "device": device.type,
        "mixed_precision": mixed_precision,
        "tensor_contract": tensor_contract,
        "fixed_input": {"noise": "zeros", "flow_time": 0.5},
        "fixed_input_loss": loss_value,
        "loss_details": numeric_loss_details,
        "action_chunk": {
            "shape": list(action_chunk.shape),
            "dtype": str(action_chunk.dtype).removeprefix("torch."),
            "sha256": _tensor_sha256(action_chunk),
            **_summary(action_chunk),
        },
        "raw_standard_action_diagnostics": raw_action_diagnostics,
        "parameters": {
            "total": sum(parameter.numel() for parameter in policy.parameters()),
            "trainable": sum(
                parameter.numel() for parameter in policy.parameters() if parameter.requires_grad
            ),
        },
        "accelerator_memory": accelerator_memory,
        "elapsed_seconds": time.perf_counter() - started,
    }
    json.dumps(report, allow_nan=False)
    run_root = _absolute_root("ROSETTA_RUN_ROOT")
    verification_root = run_root / str(experiment["experiment_id"]) / "verification"
    destination = verification_root / f"{args.run_name}.json"
    create_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report: {destination.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
