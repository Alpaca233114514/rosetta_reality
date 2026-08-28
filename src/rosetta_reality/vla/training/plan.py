"""Typed structure and contract validation for version-2 SmolVLA training plans.

A version-2 plan replaces the per-experiment launcher validators of the
historical ``run_smolvla_*`` stack with one schema that is checked field by
field.  The schema is deliberately close to the semantics of the completed
Faust plans: same optimizer/scheduler contract, same split guards, same
quarter-only monitoring policy.  It adds the two structural repairs the old
stack lacked:

- the ordered ``features`` list is the single declaration of every local
  extension the trainer installs, so composition can never depend on
  environment variables or private cross-script patching;
- ``save_freq`` must be a multiple of ``log_freq`` so every checkpoint has an
  exact metric row (audit finding T9).

This module validates structure only.  File existence and checksum binding of
prerequisites, normalization evidence and implementation files happen in the
launcher once, against the repository root.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Container
from pathlib import Path
from typing import Any

import yaml

PLAN_SCHEMA_VERSION = 2
RUN_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{2,79}")
ALLOWED_COMPILE_MODES = frozenset({"default", "reduce-overhead", "max-autotune"})
FEATURE_MASKED_CAMERA_SKIP = "masked_camera_skip"
FEATURE_CHECKPOINT_MEMORY_TRIM = "checkpoint_memory_trim"
FEATURE_HORIZON_WEIGHT_PROFILE = "horizon_weight_profile"
FEATURE_STATE_ROBUSTNESS_JITTER = "state_robustness_jitter"
FEATURE_STATE_CONDITIONING_DROPOUT = "state_conditioning_dropout"
FEATURE_TRACKIO_LOGGING = "trackio_logging"
FEATURE_FIXED_FRAME_SAMPLER = "fixed_frame_sampler"
SAMPLER_PHASES = frozenset({"smoke", "overfit", "overfit_resume"})


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping: {path.name}.")
    return value


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def repository_relative_path(raw: Any, *, context: str) -> Path:
    """Accept only safe repository-relative POSIX paths from plan declarations."""

    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{context} must be a non-empty repository-relative path.")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"{context} must be a safe repository-relative path.")
    return relative


def load_v2_plan(path: Path, repository_root: Path) -> dict[str, Any]:
    """Load one version-2 plan with checksum-pinned single-level inheritance."""

    plan = _load_yaml(path)
    inheritance = plan.pop("extends", None)
    if inheritance is None:
        return plan
    if not isinstance(inheritance, dict):
        raise ValueError("Version-2 plan inheritance must be a mapping.")
    base_relative = repository_relative_path(
        inheritance.get("config", ""),
        context="Version-2 plan inheritance",
    )
    base_path = repository_root / base_relative
    from rosetta_reality.experiment import file_sha256

    if file_sha256(base_path) != inheritance.get("sha256"):
        raise ValueError("Inherited version-2 plan checksum changed.")
    base = _load_yaml(base_path)
    if "extends" in base:
        raise ValueError("Nested version-2 plan inheritance is not supported.")
    merged = _deep_merge(base, plan)
    merged["plan_inheritance"] = {
        "config": base_relative.as_posix(),
        "sha256": file_sha256(base_path),
    }
    return merged


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping.")
    return value


def _required(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{context} is missing the required key '{key}'.")
    return mapping[key]


def _positive_int(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{context} must be a positive integer.")
    return value


def _non_negative_int(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer.")
    return value


def _boolean(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{context} must be a boolean.")
    return value


def _false(value: Any, context: str) -> None:
    if value is not False:
        raise ValueError(f"{context} must be false in a version-2 plan.")


def _episode_list(value: Any, context: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context} must be a non-empty episode list.")
    episodes: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise ValueError(f"{context} must contain non-negative integer episodes.")
        episodes.append(item)
    if len(set(episodes)) != len(episodes):
        raise ValueError(f"{context} must not repeat episodes.")
    return episodes


def validate_optimizer_contract(training: dict[str, Any]) -> dict[str, Any] | None:
    """Validate the registered AdamW plus cosine-warmup contract (Faust semantics)."""

    optimizer = training.get("optimizer")
    scheduler = training.get("scheduler")
    if optimizer is None and scheduler is None:
        return None
    if not isinstance(optimizer, dict) or not isinstance(scheduler, dict):
        raise ValueError("Optimizer and scheduler contracts must both be mappings.")

    betas = optimizer.get("betas")
    numeric_optimizer = {
        "lr": optimizer.get("lr"),
        "eps": optimizer.get("eps"),
        "weight_decay": optimizer.get("weight_decay"),
        "grad_clip_norm": optimizer.get("grad_clip_norm"),
    }
    if (
        optimizer.get("type") != "adamw"
        or not isinstance(betas, list)
        or len(betas) != 2
        or any(
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0.0 < float(value) < 1.0
            for value in betas
        )
        or any(
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in numeric_optimizer.values()
        )
        or float(numeric_optimizer["lr"]) <= 0.0
        or float(numeric_optimizer["eps"]) <= 0.0
        or float(numeric_optimizer["weight_decay"]) < 0.0
        or float(numeric_optimizer["grad_clip_norm"]) <= 0.0
    ):
        raise ValueError("The version-2 AdamW optimizer contract is invalid.")

    steps = training.get("steps")
    warmup_steps = scheduler.get("num_warmup_steps")
    decay_steps = scheduler.get("num_decay_steps")
    peak_lr = scheduler.get("peak_lr")
    decay_lr = scheduler.get("decay_lr")
    if (
        scheduler.get("type") != "cosine_decay_with_warmup"
        or not isinstance(steps, int)
        or isinstance(steps, bool)
        or not isinstance(warmup_steps, int)
        or isinstance(warmup_steps, bool)
        or not isinstance(decay_steps, int)
        or isinstance(decay_steps, bool)
        or not isinstance(peak_lr, int | float)
        or isinstance(peak_lr, bool)
        or not isinstance(decay_lr, int | float)
        or isinstance(decay_lr, bool)
        or not math.isfinite(float(peak_lr))
        or not math.isfinite(float(decay_lr))
        or not 0 <= warmup_steps < steps
        or decay_steps != steps
        or float(peak_lr) != float(numeric_optimizer["lr"])
        or not 0.0 < float(decay_lr) < float(peak_lr)
    ):
        raise ValueError(
            "The version-2 cosine LR schedule is invalid or not step-matched."
        )

    return {
        "optimizer": {
            "type": "adamw",
            "lr": float(numeric_optimizer["lr"]),
            "weight_decay": float(numeric_optimizer["weight_decay"]),
            "grad_clip_norm": float(numeric_optimizer["grad_clip_norm"]),
            "betas": [float(value) for value in betas],
            "eps": float(numeric_optimizer["eps"]),
        },
        "scheduler": {
            "type": "cosine_decay_with_warmup",
            "num_warmup_steps": warmup_steps,
            "num_decay_steps": decay_steps,
            "peak_lr": float(peak_lr),
            "decay_lr": float(decay_lr),
        },
    }


def training_coverage(training: dict[str, Any], train_rows: int) -> dict[str, int | float]:
    """Validate and summarize the planned observation exposure."""

    batch_size = _positive_int(training.get("batch_size"), "Training batch size")
    steps = _positive_int(training.get("steps"), "Training step count")
    _positive_int(train_rows, "Train row count")
    minimum_passes = training.get("minimum_dataset_passes")
    sample_exposures = batch_size * steps
    dataset_passes = sample_exposures / train_rows
    if minimum_passes is not None:
        if (
            isinstance(minimum_passes, bool)
            or not isinstance(minimum_passes, int | float)
            or not math.isfinite(float(minimum_passes))
            or float(minimum_passes) <= 0
        ):
            raise ValueError("minimum_dataset_passes must be a positive finite number.")
        if dataset_passes + 1e-12 < float(minimum_passes):
            raise ValueError(
                "Version-2 training exposure is below the registered minimum dataset "
                f"passes: planned={dataset_passes:.6f}, required={float(minimum_passes):.6f}."
            )
    return {
        "train_rows": train_rows,
        "batch_size": batch_size,
        "optimizer_steps": steps,
        "sample_exposures": sample_exposures,
        "dataset_passes": dataset_passes,
        "minimum_dataset_passes": (
            0.0 if minimum_passes is None else float(minimum_passes)
        ),
    }


def _validate_training(plan: dict[str, Any]) -> None:
    training = _mapping(plan.get("training"), "Plan section 'training'")
    _episode_list(training.get("episodes"), "Training episodes")
    _positive_int(training.get("batch_size"), "Training batch size")
    steps = _positive_int(training.get("steps"), "Training step count")
    save_freq = _positive_int(training.get("save_freq"), "Training save frequency")
    log_freq = _positive_int(training.get("log_freq"), "Training log frequency")
    if save_freq % log_freq:
        raise ValueError(
            "Training save frequency must be a multiple of the log frequency so "
            "every checkpoint has an exact metric row (audit finding T9)."
        )
    num_workers = _non_negative_int(
        training.get("num_workers", 0), "Training worker count"
    )
    persistent_workers = _boolean(
        training.get("persistent_workers", False), "Training persistent workers"
    )
    if num_workers == 0 and persistent_workers:
        raise ValueError("Persistent workers require a positive worker count.")
    if steps % 4:
        raise ValueError(
            "Version-2 training step counts must be divisible by four for the "
            "quarter-checkpoint monitoring policy."
        )
    expected_checkpoints = list(range(save_freq, steps + 1, save_freq))
    if training.get("checkpoint_steps") != expected_checkpoints:
        raise ValueError("Training checkpoint steps differ from the save-frequency grid.")
    if training.get("eval_split") != 0.0:
        raise ValueError("Version-2 training must not carve an eval split from train data.")
    _false(training.get("validation_gradients"), "Training validation gradients")
    _false(training.get("hidden_test_loaded"), "Training hidden-test boundary")

    policy = _mapping(training.get("policy"), "Training policy overlay")
    _non_negative_int(policy.get("empty_cameras"), "Training empty camera count")
    _boolean(policy.get("compile_model"), "Training compile flag")
    compile_mode = policy.get("compile_mode", "default")
    if compile_mode not in ALLOWED_COMPILE_MODES:
        raise ValueError("Unsupported torch.compile mode in the training policy overlay.")
    _boolean(
        policy.get("skip_fully_masked_camera_encoding"),
        "Training masked-camera skip flag",
    )
    validate_optimizer_contract(training)


def _validate_validation_section(plan: dict[str, Any]) -> None:
    validation = _mapping(plan.get("validation"), "Plan section 'validation'")
    episodes = _episode_list(validation.get("episodes"), "Validation episodes")
    offsets = validation.get("frame_offsets")
    if not isinstance(offsets, list) or not offsets:
        raise ValueError("Validation frame offsets must be a non-empty list.")
    for offset in offsets:
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError("Validation frame offsets must be non-negative integers.")
    if validation.get("total_samples") != len(episodes) * len(offsets):
        raise ValueError("Validation total samples differ from episodes times offsets.")
    _false(validation.get("hidden_test_loaded"), "Validation hidden-test boundary")


def _validate_resources(plan: dict[str, Any]) -> None:
    resources = _mapping(plan.get("resources"), "Plan section 'resources'")
    if resources.get("memory_limit") != resources.get("memory_swap_limit"):
        raise ValueError("Version-2 resources must pin memory limit and swap to one value.")
    if not isinstance(resources.get("mixed_precision"), str) or not resources["mixed_precision"]:
        raise ValueError("Version-2 resources must declare mixed precision.")
    _positive_int(resources.get("cpu_limit"), "Resource CPU limit")
    if not isinstance(resources.get("checkpoint_memory_trim"), bool):
        raise ValueError("Resources must declare checkpoint_memory_trim explicitly.")


def _validate_monitoring(plan: dict[str, Any]) -> None:
    monitoring = plan.get("monitoring")
    if monitoring is None:
        return
    monitoring = _mapping(monitoring, "Plan section 'monitoring'")
    training = _mapping(plan.get("training"), "Plan section 'training'")
    steps = _positive_int(training.get("steps"), "Training step count")
    expected_steps = [steps * fraction // 4 for fraction in range(1, 5)]
    estimated = monitoring.get("estimated_total_minutes")
    if (
        monitoring.get("policy") != "sleep_between_quarter_checkpoints"
        or monitoring.get("wake_fractions") != [0.25, 0.5, 0.75, 1.0]
        or monitoring.get("wake_steps") != expected_steps
        or monitoring.get("wake_steps") != training.get("checkpoint_steps")
        or monitoring.get("blocking_command") != "sleep"
        or not isinstance(monitoring.get("sleep_poll_seconds"), int)
        or isinstance(monitoring.get("sleep_poll_seconds"), bool)
        or int(monitoring["sleep_poll_seconds"]) != 300
        or not isinstance(estimated, int | float)
        or isinstance(estimated, bool)
        or not math.isfinite(float(estimated))
        or float(estimated) <= 0.0
        or monitoring.get("hidden_test_loaded") is not False
    ):
        raise ValueError("Version-2 monitoring differs from the quarter-only sleep policy.")


def _validate_features(plan: dict[str, Any], known_features: Container[str]) -> list[str]:
    declarations = plan.get("features")
    if not isinstance(declarations, list) or not declarations:
        raise ValueError("Version-2 plans must declare a non-empty ordered features list.")
    names: list[str] = []
    for declaration in declarations:
        declaration = _mapping(declaration, "Each feature declaration")
        name = _required(declaration, "name", "Each feature declaration")
        if not isinstance(name, str) or name not in known_features:
            raise ValueError(f"Unknown training feature declared: {name!r}.")
        if name in names:
            raise ValueError(f"Training feature declared twice: {name!r}.")
        names.append(name)
        if name == FEATURE_FIXED_FRAME_SAMPLER:
            phase = _required(declaration, "phase", "The fixed-frame sampler declaration")
            if phase not in SAMPLER_PHASES:
                raise ValueError("The fixed-frame sampler declared an unsupported phase.")

    training_policy = _mapping(
        _mapping(plan.get("training"), "Plan section 'training'").get("policy"),
        "Training policy overlay",
    )
    if FEATURE_MASKED_CAMERA_SKIP in names and training_policy.get(
        "skip_fully_masked_camera_encoding"
    ) is not True:
        raise ValueError(
            "Declaring masked_camera_skip requires "
            "training.policy.skip_fully_masked_camera_encoding: true."
        )
    resources = _mapping(plan.get("resources"), "Plan section 'resources'")
    if FEATURE_CHECKPOINT_MEMORY_TRIM in names and resources.get(
        "checkpoint_memory_trim"
    ) is not True:
        raise ValueError(
            "Declaring checkpoint_memory_trim requires resources.checkpoint_memory_trim: true."
        )
    if FEATURE_HORIZON_WEIGHT_PROFILE in names and not isinstance(
        plan.get("loss_contract"), dict
    ):
        raise ValueError("Declaring horizon_weight_profile requires a loss contract.")
    if FEATURE_STATE_ROBUSTNESS_JITTER in names and not isinstance(
        plan.get("state_robustness_contract"), dict
    ):
        raise ValueError(
            "Declaring state_robustness_jitter requires a state-robustness contract."
        )
    if FEATURE_STATE_CONDITIONING_DROPOUT in names and not isinstance(
        plan.get("visual_conditioning_contract"), dict
    ):
        raise ValueError(
            "Declaring state_conditioning_dropout requires a visual-conditioning "
            "contract."
        )
    if FEATURE_TRACKIO_LOGGING in names:
        tracking = _mapping(plan.get("tracking"), "Plan section 'tracking'")
        if not isinstance(tracking.get("project"), str) or not isinstance(
            tracking.get("space_id"), str
        ):
            raise ValueError("trackio_logging requires a tracking project and Space.")
    return names


def _validate_prerequisites(plan: dict[str, Any]) -> None:
    prerequisites = plan.get("prerequisites", {})
    prerequisites = _mapping(prerequisites, "Plan section 'prerequisites'")
    for name, declaration in prerequisites.items():
        declaration = _mapping(declaration, f"Prerequisite '{name}'")
        repository_relative_path(
            _required(declaration, "path", f"Prerequisite '{name}'"),
            context=f"Prerequisite '{name}'",
        )
        if not is_sha256(declaration.get("sha256")):
            raise ValueError(f"Prerequisite '{name}' must declare a SHA-256 checksum.")


def _validate_normalization_declarations(plan: dict[str, Any]) -> None:
    normalization = _mapping(plan.get("normalization"), "Plan section 'normalization'")
    if normalization.get("source_split") != "train":
        raise ValueError("Version-2 normalization must come from the train split only.")
    for key in ("report", "dataset_view_manifest"):
        repository_relative_path(
            _required(normalization, key, "Plan section 'normalization'"),
            context=f"Normalization {key}",
        )
    for key in ("report_sha256", "dataset_view_manifest_sha256"):
        if not is_sha256(normalization.get(key)):
            raise ValueError(f"Normalization {key} must be a SHA-256 checksum.")
    _false(normalization.get("validation_episodes_loaded"), "Normalization validation split")
    _false(normalization.get("hidden_test_loaded"), "Normalization hidden-test boundary")


def _validate_implementation_files(plan: dict[str, Any]) -> None:
    implementation = plan.get("implementation_files")
    if not isinstance(implementation, dict) or not implementation:
        raise ValueError("Version-2 plans must bind an implementation checksum inventory.")
    for raw_path, sha in implementation.items():
        repository_relative_path(raw_path, context="Implementation file")
        if not is_sha256(sha):
            raise ValueError(f"Implementation file {raw_path} must declare a SHA-256.")


def validate_plan_structure(
    plan: dict[str, Any],
    *,
    known_features: Container[str],
) -> list[str]:
    """Validate one version-2 plan mapping and return the ordered feature names."""

    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("Version-2 plans must set schema_version: 2.")
    if plan.get("role") != "vla":
        raise ValueError("Version-2 training plans must declare role: vla.")
    if plan.get("status") != "preregistered":
        raise ValueError("Version-2 training plans must be preregistered.")
    for key in ("plan_id", "run_name"):
        value = _required(plan, key, "Version-2 plan")
        if not isinstance(value, str) or not RUN_NAME_PATTERN.fullmatch(value):
            raise ValueError(f"Version-2 plan {key} must be a lower-case path-safe identifier.")

    parent = _mapping(plan.get("parent_experiment"), "Plan section 'parent_experiment'")
    repository_relative_path(
        _required(parent, "config", "Parent experiment"),
        context="Parent experiment config",
    )
    if not is_sha256(parent.get("sha256")):
        raise ValueError("Parent experiment must declare a SHA-256 checksum.")
    if not isinstance(parent.get("experiment_id"), str):
        raise ValueError("Parent experiment must declare its experiment id.")

    _validate_training(plan)
    _validate_validation_section(plan)
    _validate_resources(plan)
    _validate_monitoring(plan)
    names = _validate_features(plan, known_features)
    _validate_prerequisites(plan)
    _validate_normalization_declarations(plan)
    _validate_implementation_files(plan)

    stop_conditions = plan.get("stop_conditions")
    if not isinstance(stop_conditions, list) or not stop_conditions:
        raise ValueError("Version-2 plans must declare non-empty stop conditions.")
    for condition in stop_conditions:
        if not isinstance(condition, str) or not condition:
            raise ValueError("Stop conditions must be non-empty strings.")
    _false(plan.get("hidden_test_loaded"), "Plan-level hidden-test boundary")
    return names
