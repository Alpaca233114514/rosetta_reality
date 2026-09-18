"""Minimal LeRobot logger that writes only public-safe data to local Trackio."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

from .public_payload import sanitize_metrics, validate_public_payload

_STEP_PATTERN = re.compile(r"(\d+)$")


def _experiment_config() -> dict[str, Any]:
    path = Path(
        os.environ.get(
            "ROSETTA_VLA_EXPERIMENT_CONFIG",
            "configs/vla/smolvla_450m_aloha_insertion.yaml",
        )
    )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("SmolVLA experiment config must be a mapping.")
    return raw


def _public_config(cfg: Any, experiment: dict[str, Any], phase: str) -> dict[str, Any]:
    policy = cfg.policy
    optimizer = cfg.optimizer
    scheduler = cfg.scheduler
    dataset = experiment["dataset"]
    model = experiment["model"]
    upstream = experiment["upstream"]
    resources = experiment["resources"]
    resume_from_step = os.environ.get("ROSETTA_VLA_RESUME_FROM_STEP")
    resume_source_run = os.environ.get("ROSETTA_VLA_RESUME_SOURCE_RUN")
    formal_plan_sha256 = os.environ.get("ROSETTA_VLA_FORMAL_PLAN_SHA256")
    normalization_sha256 = os.environ.get("ROSETTA_VLA_NORMALIZATION_SHA256")
    code_revision = os.environ.get("ROSETTA_VLA_CODE_REVISION")
    workspace_tree_sha256 = os.environ.get("ROSETTA_VLA_WORKSPACE_TREE_SHA256")
    workspace_dirty = os.environ.get("ROSETTA_VLA_WORKSPACE_DIRTY")
    workspace_file_count = os.environ.get("ROSETTA_VLA_WORKSPACE_FILE_COUNT")
    performance_plan_sha256 = os.environ.get("ROSETTA_VLA_PERFORMANCE_PLAN_SHA256")
    skip_masked_camera_encoding = (
        os.environ.get("ROSETTA_VLA_SKIP_FULLY_MASKED_CAMERA_ENCODING") == "1"
    )
    active_memory_limit = (
        os.environ.get("ROSETTA_DOCKER_MEMORY_LIMIT", resources["memory_limit"])
        if phase == "performance_benchmark"
        else resources["memory_limit"]
    )
    active_memory_swap_limit = (
        os.environ.get("ROSETTA_DOCKER_MEMORY_SWAP_LIMIT", resources["memory_swap_limit"])
        if phase == "performance_benchmark"
        else resources["memory_swap_limit"]
    )
    payload = {
        "experiment_id": experiment["experiment_id"],
        "role": "vla",
        "phase": phase,
        "upstream_repository": upstream["repository"],
        "upstream_revision": upstream["revision"],
        "model_id": model["identifier"],
        "model_revision": model["revision"],
        "dataset_id": dataset["identifier"],
        "dataset_revision": dataset["revision"],
        "seed": cfg.seed,
        "batch_size": cfg.batch_size,
        "steps": cfg.steps,
        "save_freq": cfg.save_freq,
        "log_freq": cfg.log_freq,
        "policy_type": policy.type,
        "chunk_size": policy.chunk_size,
        "action_steps_per_observation": policy.n_action_steps,
        "empty_cameras": policy.empty_cameras,
        "compile_model": bool(getattr(policy, "compile_model", False)),
        "compile_mode": getattr(policy, "compile_mode", None),
        "skip_fully_masked_camera_encoding": skip_masked_camera_encoding,
        "freeze_vision_encoder": policy.freeze_vision_encoder,
        "train_expert_only": policy.train_expert_only,
        "train_state_proj": policy.train_state_proj,
        "optimizer_type": None if optimizer is None else optimizer.type,
        "optimizer_lr": None if optimizer is None else optimizer.lr,
        "optimizer_betas": None
        if optimizer is None
        else list(getattr(optimizer, "betas", ())),
        "optimizer_eps": None if optimizer is None else getattr(optimizer, "eps", None),
        "optimizer_weight_decay": None
        if optimizer is None
        else optimizer.weight_decay,
        "optimizer_grad_clip_norm": None
        if optimizer is None
        else optimizer.grad_clip_norm,
        "scheduler_type": None if scheduler is None else scheduler.type,
        "scheduler_warmup_steps": None
        if scheduler is None
        else scheduler.num_warmup_steps,
        "scheduler_decay_steps": None
        if scheduler is None
        else getattr(scheduler, "num_decay_steps", None),
        "scheduler_peak_lr": None
        if scheduler is None
        else getattr(scheduler, "peak_lr", None),
        "scheduler_decay_lr": None
        if scheduler is None
        else getattr(scheduler, "decay_lr", None),
        "train_episode_count": len(cfg.dataset.episodes or []),
        "eval_split": cfg.dataset.eval_split,
        "accelerator": resources["accelerator"],
        "mixed_precision": resources["mixed_precision"],
        "memory_limit": active_memory_limit,
        "memory_swap_limit": active_memory_swap_limit,
        "resume": bool(cfg.resume),
        "resume_from_step": int(resume_from_step) if resume_from_step is not None else None,
        "resume_source_run": resume_source_run,
        "formal_plan_sha256": formal_plan_sha256,
        "performance_plan_sha256": performance_plan_sha256,
        "normalization_source_split": "train" if normalization_sha256 is not None else None,
        "normalization_sha256": normalization_sha256,
        "code_revision": code_revision,
        "workspace_tree_sha256": workspace_tree_sha256,
        "workspace_dirty": workspace_dirty == "true" if workspace_dirty is not None else None,
        "workspace_file_count": (
            int(workspace_file_count) if workspace_file_count is not None else None
        ),
        "test_split_loaded": False,
    }
    validate_public_payload(payload, context="trackio_config")
    return payload


class TrackioLogger:
    """Drop-in replacement for LeRobot's small ``WandBLogger`` interface."""

    def __init__(self, cfg: Any):
        import trackio

        experiment = _experiment_config()
        phase = os.environ.get("ROSETTA_VLA_PHASE", "")
        if phase not in {
            "space_smoke",
            "smoke",
            "overfit",
            "overfit_resume",
            "formal",
            "performance_benchmark",
        }:
            raise ValueError("ROSETTA_VLA_PHASE must identify an approved VLA phase.")
        project = str(experiment["tracking"]["project"])
        name = str(cfg.job_name)
        group = f"{experiment['experiment_id']}-{phase}"
        validate_public_payload(
            {"project": project, "name": name, "group": group},
            context="trackio_identity",
        )
        resume = "allow" if cfg.resume else "never"
        self._run = trackio.init(
            project=project,
            name=name,
            group=group,
            config=_public_config(cfg, experiment, phase),
            resume=resume,
            embed=False,
            auto_log_cpu=False,
            auto_log_gpu=False,
        )
        self._trackio = trackio
        self._last_step = 0
        cfg.wandb.run_id = str(getattr(self._run, "id", name))
        logging.info("Metrics are stored in local Trackio for sanitized static-Space sync.")

    def log_dict(
        self,
        values: dict[str, Any],
        step: int | None = None,
        mode: str = "train",
        custom_step_key: str | None = None,
    ) -> None:
        if custom_step_key is not None:
            raise ValueError("Custom Trackio step keys are not approved for this VLA pipeline.")
        if step is None or isinstance(step, bool) or step < 0:
            raise ValueError("Trackio metrics require a non-negative integer step.")
        public_values = dict(values)
        if os.environ.get("ROSETTA_VLA_PHASE") in {"performance_benchmark", "formal"}:
            import torch

            if torch.xpu.is_available():
                public_values.update(
                    {
                        "xpu_allocated_bytes": int(torch.xpu.memory_allocated()),
                        "xpu_reserved_bytes": int(torch.xpu.memory_reserved()),
                        "xpu_max_allocated_bytes": int(torch.xpu.max_memory_allocated()),
                    }
                )
        payload = sanitize_metrics(public_values, mode=mode)
        if payload:
            self._trackio.log(payload, step=step)
            self._last_step = max(self._last_step, step)

    def log_policy(self, checkpoint_dir: Path) -> None:
        match = _STEP_PATTERN.search(checkpoint_dir.name)
        step = int(match.group(1)) if match else self._last_step
        payload = sanitize_metrics({"checkpoint_saved": 1}, mode="system")
        self._trackio.log(payload, step=step)
        self._last_step = max(self._last_step, step)

    def log_video(self, video_path: str, step: int, mode: str = "train") -> None:
        del video_path, step, mode
        logging.warning("Trackio media upload is disabled by the public-payload policy.")


def finish_trackio() -> None:
    """Flush the current local run without importing Trackio during normal package import."""

    try:
        import trackio
    except ImportError:
        return
    trackio.finish()
