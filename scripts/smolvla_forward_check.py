"""Run one real-data SmolVLA forward pass without gradients or optimizer state."""

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import IMAGENET_STATS, make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rosetta_reality.experiment import file_sha256  # noqa: E402
from rosetta_reality.features import create_json  # noqa: E402


@parser.wrap()
def _parse_config(cfg: TrainPipelineConfig) -> TrainPipelineConfig:
    return cfg


def _load_experiment() -> tuple[Path, dict[str, Any]]:
    raw = os.environ.get("ROSETTA_VLA_EXPERIMENT_CONFIG")
    if not raw:
        raise ValueError("ROSETTA_VLA_EXPERIMENT_CONFIG must be set by the gated launcher.")
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError("The experiment config path must be absolute inside the container.")
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("The experiment config must contain a mapping.")
    return path, value


def _load_action_contract(experiment: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = REPOSITORY_ROOT / str(experiment["action_contract"]["derived"])
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("action"), dict):
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


def _episode_indices(batch: dict[str, Any]) -> set[int]:
    value = batch.get("episode_index")
    if not isinstance(value, torch.Tensor):
        raise ValueError("The real-data batch has no tensor episode_index.")
    return {int(item) for item in value.detach().cpu().reshape(-1).tolist()}


def _tensor_contract(batch: dict[str, Any]) -> dict[str, dict[str, Any]]:
    contract: dict[str, dict[str, Any]] = {}
    for key, value in sorted(batch.items()):
        if isinstance(value, torch.Tensor):
            contract[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype).removeprefix("torch."),
                "device": value.device.type,
                "finite": (
                    bool(torch.isfinite(value).all().item()) if value.is_floating_point() else True
                ),
            }
    return contract


def _numeric_details(values: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            number = float(value.detach().cpu().item())
        elif isinstance(value, int | float) and not isinstance(value, bool):
            number = float(value)
        else:
            continue
        if not math.isfinite(number):
            raise FloatingPointError(f"Non-finite SmolVLA forward detail: {key}.")
        result[str(key)] = number
    return result


def _validate_train_only_statistics(dataset: Any) -> dict[str, Any] | None:
    raw_path = os.environ.get("ROSETTA_VLA_TRAIN_STATS_REPORT")
    if raw_path is None:
        return None
    path = Path(raw_path)
    report = json.loads(path.read_text(encoding="utf-8"))
    expected = report.get("effective_stats", {})
    if (
        not path.is_absolute()
        or report.get("status") != "complete"
        or report.get("stage") != "smolvla_train_only_normalization"
        or report.get("source_split") != "train"
        or os.environ.get("ROSETTA_VLA_NORMALIZATION_SHA256") != file_sha256(path)
        or not os.environ.get("ROSETTA_VLA_FORMAL_PLAN_SHA256")
        or report.get("validation_episodes_loaded") is not False
        or report.get("hidden_test_loaded") is not False
        or report.get("visual_statistics_policy") != "imagenet_constants"
        or report.get("visual_statistics_source") != "fixed_constants_not_dataset_rows"
        or report.get("visual_statistics") != IMAGENET_STATS
        or not isinstance(expected, dict)
    ):
        raise ValueError("Formal preflight train-only normalization identity is invalid.")
    for feature, raw_statistics in expected.items():
        actual_statistics = dataset.meta.stats.get(feature)
        if not isinstance(raw_statistics, dict) or not isinstance(actual_statistics, dict):
            raise ValueError("Formal preflight feature statistics must be mappings.")
        if set(actual_statistics) != set(raw_statistics):
            raise ValueError("Formal preflight feature statistic keys differ.")
        for statistic, raw_value in raw_statistics.items():
            actual = torch.as_tensor(actual_statistics[statistic])
            if not isinstance(raw_value, list):
                raise ValueError("Formal preflight statistics must be backed by lists.")
            expected_tensor = torch.tensor(raw_value, dtype=actual.dtype)
            if actual.shape != expected_tensor.shape or not torch.equal(
                actual.cpu(), expected_tensor
            ):
                raise ValueError(
                    "Formal preflight dataset stats differ from the train-only report."
                )
    visual_features = set(report.get("visual_features", []))
    if set(dataset.meta.stats) != set(expected) | visual_features:
        raise ValueError("Formal preflight dataset stats contain an unexpected feature.")
    for feature in visual_features:
        actual_statistics = dataset.meta.stats.get(feature)
        if not isinstance(actual_statistics, dict) or set(actual_statistics) != set(IMAGENET_STATS):
            raise ValueError("Formal preflight visual statistics are not fixed ImageNet constants.")
        for statistic, raw_value in IMAGENET_STATS.items():
            actual = torch.as_tensor(actual_statistics[statistic])
            expected_tensor = torch.tensor(raw_value, dtype=actual.dtype)
            if not torch.equal(actual.cpu(), expected_tensor):
                raise ValueError("Formal preflight ImageNet statistic differs from the constant.")
    dataset_root = Path(dataset.root)
    view_manifest_path = dataset_root / "view_manifest.json"
    view_manifest = json.loads(view_manifest_path.read_text(encoding="utf-8"))
    if (
        view_manifest.get("status") != "complete"
        or view_manifest.get("stage") != "smolvla_train_only_dataset_view"
        or view_manifest.get("normalization_report_sha256") != file_sha256(path)
        or view_manifest.get("validation_episodes_loaded") is not False
        or view_manifest.get("hidden_test_loaded") is not False
    ):
        raise ValueError("Formal preflight dataset view manifest is invalid.")
    return {
        "source_split": "train",
        "normalization_report_sha256": file_sha256(path),
        "dataset_view_manifest_sha256": file_sha256(view_manifest_path),
        "features": sorted(expected),
        "visual_statistics_policy": "imagenet_constants",
        "rows": int(expected["action"]["count"][0]),
        "validation_episodes_loaded": False,
        "hidden_test_loaded": False,
    }


def _make_processors(
    cfg: TrainPipelineConfig,
    policy: torch.nn.Module,
    dataset: Any,
    device: torch.device,
) -> tuple[Any, Any]:
    if cfg.policy is None or cfg.policy.pretrained_path is None:
        raise ValueError("The forward check requires a local pretrained SmolVLA policy.")
    preprocessor_overrides = {
        "device_processor": {"device": device.type},
        "normalizer_processor": {
            "features": {**policy.config.input_features, **policy.config.output_features},
            "norm_map": policy.config.normalization_mapping,
            "stats": dataset.meta.stats,
        },
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }
    postprocessor_overrides = {
        "unnormalizer_processor": {
            "features": policy.config.output_features,
            "norm_map": policy.config.normalization_mapping,
            "stats": dataset.meta.stats,
        },
    }
    return make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        pretrained_revision=cfg.policy.pretrained_revision,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
    )


def main() -> int:
    config_path, experiment = _load_experiment()
    contract_path, action_contract = _load_action_contract(experiment)
    action_spec = action_contract["action"]
    action_dimension = int(action_spec["dimension"])
    chunk_length = int(action_spec["chunk_length"])
    dimensions = action_spec.get("dimensions")
    if not isinstance(dimensions, list) or len(dimensions) != action_dimension:
        raise ValueError("Action Contract dimension metadata is inconsistent.")
    run_name = os.environ.get("ROSETTA_VLA_RUN_NAME")
    if not run_name:
        raise ValueError("ROSETTA_VLA_RUN_NAME must be set by the gated launcher.")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("HF_DATASETS_OFFLINE") != "1":
        raise RuntimeError("The forward check must run with model and dataset networking disabled.")
    cfg = _parse_config()
    cfg.validate()
    if cfg.policy is None or cfg.policy.type != "smolvla":
        raise ValueError("The parsed policy is not SmolVLA.")
    if cfg.policy.load_vlm_weights:
        raise ValueError("Combined SmolVLA checkpoints must not reload duplicate base VLM weights.")
    expected_device = os.environ.get("ROSETTA_TORCH_DEVICE")
    if not expected_device:
        raise ValueError("ROSETTA_TORCH_DEVICE must be set by the Docker runner.")
    if cfg.policy.device != expected_device:
        raise ValueError(
            "The parsed policy device differs from the container accelerator contract."
        )
    device = torch.device(expected_device)
    if device.type == "xpu" and not torch.xpu.is_available():
        raise RuntimeError("The requested XPU is unavailable.")

    started = time.perf_counter()
    torch.manual_seed(int(experiment["seed"]))
    dataset = make_dataset(cfg)
    train_only_statistics = _validate_train_only_statistics(dataset)
    batch_size = int(experiment["phases"]["smoke"]["batch_size"])
    state_dimension = _feature_dimension(dataset, "observation.state")
    dataset_action_dimension = _feature_dimension(dataset, "action")
    if dataset_action_dimension != action_dimension:
        raise ValueError("Dataset action dimension differs from the Action Contract.")
    if cfg.policy.chunk_size != chunk_length:
        raise ValueError("Policy chunk size differs from the Action Contract.")
    allowed_episodes = {int(value) for value in experiment["phases"]["smoke"]["episodes"]}
    hidden_test = {int(value) for value in experiment["dataset"]["test_episodes"]}
    if allowed_episodes & hidden_test:
        raise ValueError("The preregistered forward-check episode intersects hidden test data.")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    episodes_loaded = _episode_indices(batch)
    if not episodes_loaded <= allowed_episodes:
        raise ValueError("The forward check materialized an episode outside the smoke scope.")
    raw_tensor_contract = _tensor_contract(batch)
    raw_action = batch.get("action")
    raw_state = batch.get("observation.state")
    expected_action_shape = [batch_size, chunk_length, action_dimension]
    expected_state_shape = [batch_size, cfg.policy.n_obs_steps, state_dimension]
    if not isinstance(raw_action, torch.Tensor) or list(raw_action.shape) != expected_action_shape:
        raise ValueError(
            "The raw real-data action tensor does not match the registered Action Contract."
        )
    if not isinstance(raw_state, torch.Tensor) or list(raw_state.shape) != expected_state_shape:
        received = list(raw_state.shape) if isinstance(raw_state, torch.Tensor) else None
        raise ValueError(
            f"The raw real-data state history differs from dataset metadata: {received}."
        )

    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    preprocessor, _ = _make_processors(cfg, policy, dataset, device)
    for camera_key in dataset.meta.camera_keys:
        if camera_key in batch and batch[camera_key].dtype == torch.uint8:
            maximum = torch.iinfo(batch[camera_key].dtype).max
            batch[camera_key] = batch[camera_key].to(dtype=torch.get_default_dtype()) / maximum
    batch = preprocessor(batch)
    tensor_contract = _tensor_contract(batch)
    if not tensor_contract or any(not entry["finite"] for entry in tensor_contract.values()):
        raise FloatingPointError("The preprocessed SmolVLA batch is empty or non-finite.")
    action = batch.get("action")
    state = batch.get("observation.state")
    if not isinstance(action, torch.Tensor) or list(action.shape) != expected_action_shape:
        received = list(action.shape) if isinstance(action, torch.Tensor) else None
        raise ValueError(f"The preprocessed action tensor has an unexpected shape: {received}.")
    if not isinstance(state, torch.Tensor) or list(state.shape) != expected_state_shape:
        received = list(state.shape) if isinstance(state, torch.Tensor) else None
        raise ValueError(f"The preprocessed state tensor has an unexpected shape: {received}.")

    padded_state = policy.prepare_state(batch)
    padded_action = policy.prepare_action(batch)
    padded_tensor_contract = _tensor_contract(
        {"observation.state": padded_state, "action": padded_action}
    )
    expected_padded_state = [batch_size, cfg.policy.max_state_dim]
    expected_padded_action = [batch_size, chunk_length, cfg.policy.max_action_dim]
    if (
        list(padded_state.shape) != expected_padded_state
        or list(padded_action.shape) != expected_padded_action
    ):
        raise ValueError("SmolVLA did not pad state/action to its declared max dimensions.")

    policy.train()
    mixed_precision = str(experiment["resources"]["mixed_precision"])
    autocast_dtype = _autocast_dtype(mixed_precision)
    autocast_enabled = autocast_dtype is not None
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=autocast_enabled,
        ),
    ):
        loss, details = policy(batch)
    loss_value = float(loss.detach().cpu().item())
    if not math.isfinite(loss_value):
        raise FloatingPointError("The real SmolVLA dummy forward produced a non-finite loss.")
    details = _numeric_details(details)
    if device.type == "xpu":
        torch.xpu.synchronize()
        accelerator_memory = {
            "allocated_bytes": int(torch.xpu.memory_allocated()),
            "reserved_bytes": int(torch.xpu.memory_reserved()),
            "maximum_allocated_bytes": int(torch.xpu.max_memory_allocated()),
        }
    else:
        accelerator_memory = {}

    run_root_raw = os.environ.get("ROSETTA_RUN_ROOT")
    if not run_root_raw:
        raise ValueError("ROSETTA_RUN_ROOT must be set by the Docker runner.")
    preflight_root = Path(run_root_raw) / str(experiment["experiment_id"]) / "preflight"
    report_path = preflight_root / f"{run_name}.json"
    model_root = Path(cfg.policy.pretrained_path)
    dataset_root = Path(cfg.dataset.root)
    dataset_manifest_path = (
        dataset_root / "view_manifest.json"
        if train_only_statistics is not None
        else dataset_root / "manifest.json"
    )
    report = {
        "schema_version": 1,
        "status": "passed",
        "stage": "real_smolvla_no_optimizer_forward",
        "experiment_id": experiment["experiment_id"],
        "run_name": run_name,
        "experiment_config_sha256": file_sha256(config_path),
        "formal_plan_sha256": os.environ.get("ROSETTA_VLA_FORMAL_PLAN_SHA256"),
        "action_contract_sha256": file_sha256(contract_path),
        "action_dimension": action_dimension,
        "chunk_length": chunk_length,
        "state_dimension": state_dimension,
        "model_revision": experiment["model"]["revision"],
        "model_manifest_sha256": file_sha256(model_root / "model_manifest.json"),
        "vlm_dependency_revision": experiment["model"]["vlm_dependency"]["revision"],
        "vlm_dependency_manifest_sha256": file_sha256(
            model_root / experiment["model"]["vlm_dependency"]["manifest"]
        ),
        "dataset_revision": experiment["dataset"]["revision"],
        "dataset_manifest_sha256": file_sha256(dataset_manifest_path),
        "normalization_report_sha256": (
            train_only_statistics["normalization_report_sha256"]
            if train_only_statistics is not None
            else None
        ),
        "train_only_statistics": train_only_statistics,
        "episodes_loaded": sorted(episodes_loaded),
        "hidden_test_loaded": False,
        "network_disabled": True,
        "optimizer_created": False,
        "gradients_enabled": False,
        "device": device.type,
        "mixed_precision": mixed_precision,
        "camera_keys": sorted(dataset.meta.camera_keys),
        "raw_tensor_contract": raw_tensor_contract,
        "tensor_contract": tensor_contract,
        "padded_tensor_contract": padded_tensor_contract,
        "loss": loss_value,
        "loss_details": details,
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
    create_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report: {report_path.relative_to(REPOSITORY_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
