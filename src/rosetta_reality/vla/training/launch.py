"""Version-2 plan-to-CLI assembly for the pinned ``lerobot-train`` entry point.

This module is the single place that maps a validated version-2 plan plus its
parent experiment onto the ``lerobot-train`` command line.  The flag spellings
are byte-compatible with the historical launchers so the pinned upstream
trainer observes exactly the same interface; only the construction is
consolidated.  The runtime experiment written for the Trackio bridge is
composed here as well, replacing the historical in-process deepcopy patch.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from rosetta_reality.vla.training.plan import (
    FEATURE_TRACKIO_LOGGING,
    validate_optimizer_contract,
)

MODE_PREFLIGHT = "preflight"
MODE_SMOKE = "smoke"
MODE_TRAIN = "train"
LAUNCH_MODES = frozenset({MODE_PREFLIGHT, MODE_SMOKE, MODE_TRAIN})


def _phase_config(plan: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode == MODE_PREFLIGHT:
        section = plan.get("preflight")
    elif mode == MODE_SMOKE:
        section = plan.get("optimizer_smoke")
    else:
        section = plan.get("training")
    if not isinstance(section, dict):
        raise ValueError(f"The version-2 plan has no section for launch mode '{mode}'.")
    return section


def optimizer_arguments(training: dict[str, Any]) -> list[str]:
    """Build the pinned optimizer/scheduler CLI fragment from the contract."""

    contract = validate_optimizer_contract(training)
    if contract is None:
        return []
    optimizer = contract["optimizer"]
    scheduler = contract["scheduler"]
    return [
        f"--policy.optimizer_lr={optimizer['lr']}",
        f"--policy.optimizer_betas={json.dumps(optimizer['betas'], separators=(',', ':'))}",
        f"--policy.optimizer_eps={optimizer['eps']}",
        f"--policy.optimizer_weight_decay={optimizer['weight_decay']}",
        f"--policy.optimizer_grad_clip_norm={optimizer['grad_clip_norm']}",
        f"--policy.scheduler_warmup_steps={scheduler['num_warmup_steps']}",
        f"--policy.scheduler_decay_steps={scheduler['num_decay_steps']}",
        f"--policy.scheduler_decay_lr={scheduler['decay_lr']}",
    ]


def compose_runtime_experiment(
    experiment: dict[str, Any], plan: dict[str, Any]
) -> dict[str, Any]:
    """Merge plan resources into a serializable runtime experiment copy."""

    runtime_experiment = copy.deepcopy(experiment)
    resources = plan.get("resources")
    if isinstance(resources, dict):
        runtime_experiment.setdefault("resources", {}).update(resources)
    return runtime_experiment


def _declared_features(plan: dict[str, Any]) -> set[str]:
    declarations = plan.get("features")
    if not isinstance(declarations, list):
        return set()
    return {
        declaration.get("name")
        for declaration in declarations
        if isinstance(declaration, dict)
    }


def build_training_arguments(
    plan: dict[str, Any],
    experiment: dict[str, Any],
    *,
    mode: str,
    run_name: str,
    model_root: Path,
    dataset_root: Path,
    output_dir: Path,
    device: str,
) -> list[str]:
    """Map one validated plan onto the pinned ``lerobot-train`` CLI."""

    if mode not in LAUNCH_MODES:
        raise ValueError(f"Unsupported version-2 launch mode: {mode!r}.")
    if not isinstance(device, str) or not device:
        raise ValueError("Version-2 launches require a device identifier.")
    phase_config = _phase_config(plan, mode)
    episodes = [int(value) for value in phase_config["episodes"]]
    test_episodes = {int(value) for value in experiment["dataset"]["test_episodes"]}
    if set(episodes) & test_episodes:
        raise ValueError("A training phase attempted to load hidden-test episodes.")
    policy = dict(experiment["model"]["policy"])
    adaptation = experiment["model"]["adaptation"]
    mixed_precision = str(plan["resources"]["mixed_precision"])
    save_checkpoint = bool(phase_config.get("save_checkpoint", mode != MODE_PREFLIGHT))
    log_freq = int(phase_config.get("log_freq", 1))
    num_workers = int(phase_config.get("num_workers", 0))
    persistent_workers = bool(phase_config.get("persistent_workers", False))
    if log_freq <= 0:
        raise ValueError("Version-2 phases require a positive log frequency.")
    if num_workers < 0 or (num_workers == 0 and persistent_workers):
        raise ValueError("Persistent workers require a positive worker count.")
    if mode == MODE_TRAIN:
        overlay = plan["training"].get("policy", {})
        for key in ("empty_cameras", "compile_model", "compile_mode"):
            if key in overlay:
                policy[key] = overlay[key]
    tracking = plan.get("tracking", {})
    wandb_project = None
    if FEATURE_TRACKIO_LOGGING in _declared_features(plan):
        wandb_project = tracking.get("project")
        if not isinstance(wandb_project, str) or not wandb_project:
            raise ValueError("trackio_logging requires a tracking project in the plan.")
    arguments = [
        f"--policy.path={model_root}",
        f"--policy.pretrained_revision={experiment['model']['revision']}",
        f"--policy.device={device}",
        "--policy.push_to_hub=false",
        f"--policy.chunk_size={policy['chunk_size']}",
        f"--policy.n_action_steps={policy['n_action_steps']}",
        f"--policy.empty_cameras={policy['empty_cameras']}",
        f"--policy.load_vlm_weights={str(policy['load_vlm_weights']).lower()}",
        f"--policy.freeze_vision_encoder={str(adaptation['freeze_vision_encoder']).lower()}",
        f"--policy.train_expert_only={str(adaptation['train_expert_only']).lower()}",
        f"--policy.train_state_proj={str(adaptation['train_state_proj']).lower()}",
        f"--dataset.repo_id={experiment['dataset']['identifier']}",
        f"--dataset.root={dataset_root}",
        f"--dataset.revision={experiment['dataset']['revision']}",
        f"--dataset.episodes={json.dumps(episodes, separators=(',', ':'))}",
        "--dataset.eval_split=0.0",
        f"--rename_map={json.dumps(experiment['dataset']['rename_map'], separators=(',', ':'))}",
        f"--output_dir={output_dir}",
        f"--job_name={run_name}",
        f"--seed={experiment['seed']}",
        f"--batch_size={phase_config['batch_size']}",
        f"--steps={phase_config.get('steps', 1)}",
        f"--save_freq={phase_config.get('save_freq', phase_config.get('steps', 1))}",
        f"--save_checkpoint={str(save_checkpoint).lower()}",
        "--save_checkpoint_to_hub=false",
        f"--log_freq={log_freq}",
        "--eval_steps=0",
        "--env_eval_freq=0",
        f"--num_workers={num_workers}",
        f"--persistent_workers={str(persistent_workers).lower()}",
        "--dataloader_multiprocessing_context=null",
        f"--accelerator.mixed_precision={mixed_precision}",
        "--accelerator.gradient_accumulation.steps=1",
    ]
    if wandb_project is not None:
        arguments.extend(
            [
                "--wandb.enable=true",
                "--wandb.disable_artifact=true",
                f"--wandb.project={wandb_project}",
            ]
        )
    else:
        arguments.append("--wandb.enable=false")
    if policy.get("compile_model"):
        compile_mode = str(policy.get("compile_mode", "default"))
        arguments.append("--policy.compile_model=true")
        arguments.append(f"--policy.compile_mode={compile_mode}")
    elif "compile_model" in policy:
        arguments.append(f"--policy.compile_model={str(bool(policy['compile_model'])).lower()}")
    if num_workers > 0:
        prefetch_factor = int(phase_config.get("prefetch_factor", 2))
        if prefetch_factor <= 0:
            raise ValueError("Version-2 prefetch_factor must be positive.")
        arguments.append(f"--prefetch_factor={prefetch_factor}")
    if mode in {MODE_SMOKE, MODE_TRAIN}:
        arguments.extend(optimizer_arguments(plan["training"]))
    return arguments
