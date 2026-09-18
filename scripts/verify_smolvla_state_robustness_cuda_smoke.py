"""Independently reload Way's final CUDA-smoke checkpoint on one clean sample."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import run_smolvla_phase as phase_runner  # noqa: E402
import run_smolvla_state_robustness_cuda_smoke as smoke_runner  # noqa: E402
import verify_smolvla_checkpoint as historical_verify  # noqa: E402

from rosetta_reality.experiment import (  # noqa: E402
    file_sha256,
    workspace_code_identity,
)
from rosetta_reality.features import create_json  # noqa: E402
from rosetta_reality.sim import load_action_contract  # noqa: E402
from rosetta_reality.vla import load_smolvla_action_space  # noqa: E402
from rosetta_reality.vla.accelerator_memory import (  # noqa: E402
    memory_snapshot,
    reset_peak_memory_stats,
    synchronize,
)
from rosetta_reality.vla.processor import ensure_smolvla_action_boundary  # noqa: E402


def _processors(
    cfg: TrainPipelineConfig,
    policy: Any,
    dataset: Any,
    pretrained_dir: Path,
    device: torch.device,
) -> tuple[Any, Any]:
    return make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=pretrained_dir,
        preprocessor_overrides={"device_processor": {"device": device.type}},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    if (
        os.environ.get("HF_HUB_OFFLINE") != "1"
        or os.environ.get("HF_DATASETS_OFFLINE") != "1"
        or os.environ.get("ROSETTA_TORCH_DEVICE") != "cuda"
        or not torch.cuda.is_available()
    ):
        raise RuntimeError("Way CUDA reload requires the offline AutoDL CUDA boundary.")
    plan_path = args.plan.resolve()
    plan, base_path, experiment = smoke_runner._validate_plan(plan_path)
    smoke = plan["optimizer_smoke"]
    step = int(smoke["steps"])
    run_name = str(plan["run_name"])
    checkpoint_root = phase_runner._absolute_root("ROSETTA_CHECKPOINT_ROOT")
    step_dir = (
        checkpoint_root
        / str(experiment["experiment_id"])
        / "smoke"
        / run_name
        / "checkpoints"
        / f"{step:06d}"
    )
    pretrained_dir = step_dir / "pretrained_model"
    training_state_dir = step_dir / "training_state"
    required_files = historical_verify._validate_checkpoint_files(
        pretrained_dir, training_state_dir
    )
    train_config = historical_verify._load_json(pretrained_dir / "train_config.json")
    policy_config = historical_verify._load_json(pretrained_dir / "config.json")
    training_step = historical_verify._load_json(
        training_state_dir / "training_step.json"
    )
    contract_path = REPOSITORY_ROOT / str(experiment["action_contract"]["derived"])
    contract = load_action_contract(contract_path)
    action_space = load_smolvla_action_space(experiment, require_explicit=True)
    runtime_experiment = copy.deepcopy(experiment)
    runtime_experiment["phases"]["smoke"] = dict(smoke)
    historical_verify._validate_saved_identity(
        train_config,
        policy_config,
        training_step,
        runtime_experiment,
        step_dir,
        contract.dimension,
        contract.chunk_length,
        "smoke",
        step,
    )
    last = step_dir.parent / "last"
    if not last.is_symlink() or last.resolve() != step_dir:
        raise ValueError("Way CUDA final smoke checkpoint is not the immutable last link.")

    started = time.perf_counter()
    device = torch.device("cuda")
    cfg = TrainPipelineConfig.from_pretrained(pretrained_dir)
    if cfg.policy is None or cfg.policy.type != "smolvla":
        raise ValueError("Way CUDA checkpoint has no saved SmolVLA policy config.")
    cfg.policy.device = device.type
    cfg.policy.pretrained_path = pretrained_dir
    cfg.policy.pretrained_revision = None
    dataset = make_dataset(cfg)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    episodes = historical_verify._episode_indices(batch)
    if episodes != smoke["episodes"] or set(episodes) & set(
        experiment["dataset"]["test_episodes"]
    ):
        raise ValueError("Way CUDA reload materialized an unregistered episode.")
    raw_action = batch.get("action")
    if not isinstance(raw_action, torch.Tensor) or list(raw_action.shape) != [
        1,
        contract.chunk_length,
        contract.dimension,
    ]:
        raise ValueError("Way CUDA reload target differs from the Action Contract.")
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    preprocessor, postprocessor = _processors(
        cfg, policy, dataset, pretrained_dir, device
    )
    ensure_smolvla_action_boundary(
        preprocessor,
        postprocessor,
        contract,
        action_space,
        action_contract_sha256=file_sha256(contract_path),
        upstream_revision=str(experiment["upstream"]["revision"]),
    )
    for camera_key in dataset.meta.camera_keys:
        if camera_key in batch and batch[camera_key].dtype == torch.uint8:
            maximum = torch.iinfo(batch[camera_key].dtype).max
            batch[camera_key] = (
                batch[camera_key].to(dtype=torch.get_default_dtype()) / maximum
            )
    batch = preprocessor(batch)
    normalized_action = batch.get("action")
    if not isinstance(normalized_action, torch.Tensor):
        raise ValueError("Way CUDA reload preprocessor produced no action tensor.")
    noise = torch.zeros(
        (1, contract.chunk_length, cfg.policy.max_action_dim),
        device=device,
        dtype=normalized_action.dtype,
    )
    flow_time = torch.full((1,), 0.5, device=device, dtype=normalized_action.dtype)
    autocast_dtype = historical_verify._autocast_dtype(
        str(plan["resources"]["mixed_precision"])
    )
    reset_peak_memory_stats(torch, device)
    policy.eval()
    policy.reset()
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=autocast_dtype is not None,
        ),
    ):
        prediction = policy.predict_action_chunk(batch, noise=noise)
        loss, details = policy(batch, noise=noise, time=flow_time)
    prediction = postprocessor(prediction)
    synchronize(torch, device)
    loss_value = float(loss.detach().cpu().item())
    if (
        not isinstance(prediction, torch.Tensor)
        or list(prediction.shape)
        != [1, contract.chunk_length, contract.dimension]
        or not bool(torch.isfinite(prediction).all())
        or not math.isfinite(loss_value)
    ):
        raise FloatingPointError("Way CUDA checkpoint reload produced invalid output.")
    lower = contract.lower_bounds.to(prediction).view(1, 1, -1)
    upper = contract.upper_bounds.to(prediction).view(1, 1, -1)
    violation_rate = float(
        ((prediction < lower) | (prediction > upper)).to(torch.float64).mean()
    )
    if violation_rate != 0.0:
        raise ValueError("Way CUDA reload violated the bounded action contract.")
    numeric_details = {
        str(key): float(value)
        for key, value in details.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }
    if not all(math.isfinite(value) for value in numeric_details.values()):
        raise FloatingPointError("Way CUDA reload loss details are non-finite.")
    report = {
        "schema_version": 1,
        "status": "passed",
        "stage": "smolvla_state_robustness_cuda_smoke_independent_reload",
        "experiment_id": experiment["experiment_id"],
        "run_name": run_name,
        "checkpoint_step": step,
        "formal_plan_sha256": file_sha256(plan_path),
        "experiment_config_sha256": file_sha256(base_path),
        "action_contract_sha256": file_sha256(contract_path),
        "required_checkpoint_files": required_files,
        "model_safetensors_sha256": file_sha256(
            pretrained_dir / "model.safetensors"
        ),
        "device": "cuda",
        "loss": loss_value,
        "loss_details": numeric_details,
        "prediction_shape": list(prediction.shape),
        "prediction_finite": True,
        "joint_limit_violation_rate": violation_rate,
        "accelerator_memory": memory_snapshot(torch, device),
        "code_identity": workspace_code_identity(REPOSITORY_ROOT),
        "hidden_test_loaded": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    run_root = phase_runner._absolute_root("ROSETTA_RUN_ROOT")
    destination = (
        run_root
        / str(experiment["experiment_id"])
        / "verification"
        / f"{run_name}-step{step:06d}.json"
    )
    if destination.exists():
        raise FileExistsError("Way CUDA smoke reload report is create-only.")
    create_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Reload report: {destination.relative_to(run_root).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
