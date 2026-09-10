"""Shared preregistered identity for the visual-conditioning state-dropout axis.

The candidate furnace ``m2-smolvla450m-vcdropout-001`` is the single treatment
arm of the T4 visual-conditioning axis registered in
``reports/training/m2-smolvla-visual-conditioning-state-dropout-preregistration-2026-08-28.md``.
Its immutable control is the completed Zen-uniform run
``m2-smolvla450m-zen-cuda-b64-uniform-001-step0316-deploy-001``; the candidate
differs from that control by exactly the ``state_conditioning_dropout``
feature.  This module freezes the candidate identity, the treatment contract,
the offset-250 gradient-gate protocol and its preregistered thresholds so the
create-only post-training chain (validation, selection, export, exact reload
and the executable metric gate) can bind to the plan fail-closed.

Everything here is dependency-light (stdlib plus YAML) so post-training entry
points can validate plans without importing torch.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_ID = "m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003"
PARENT_CONFIG = (
    "configs/vla/smolvla_450m_aloha_insertion_action_repair_bounded_gripper_003.yaml"
)
PARENT_SHA256 = "0e9dd0499d0708939ac73cc5d517849f133cf6deab072d9cde09f2880ae22210"
RUNTIME_PROFILE_SHA256 = "be2bfc3ea2a518c85e56410ba3ea1da6f744236d51b6f4a5f6a7b73927e9f992"
RUNTIME_PROFILE_ID = "autodl-rtx4090-cuda-001"
NORMALIZATION_REPORT_SHA256 = (
    "263880ec3adfddb8517a50fa5483e7c8f32f0c208243c229cbc72f8e9cf8d988"
)
NORMALIZATION_REPORT_RELATIVE = (
    "runs/" + EXPERIMENT_ID + "/normalization/train-only-3e3c6b9d347e5e71.json"
)
VIEW_MANIFEST_SHA256 = "9853c191ae87016379fc1a16ebfbb87e05ab5147cd8a03d82f9c2a894c9b531e"
VIEW_MANIFEST_RELATIVE = (
    "runs/"
    + EXPERIMENT_ID
    + "/dataset_views/train-only-3e3c6b9d347e5e71/view_manifest.json"
)
ACTION_CONTRACT_RELATIVE = "configs/sim/aloha_insertion_smolvla.yaml"
ACTION_CONTRACT_SHA256 = (
    "fc71a0438f0e3af7258e5b52d82fa22fc53c12b47901606cbee715524392ac62"
)

NORMALIZATION_REPORT_UNDER_RUNROOT = (
    EXPERIMENT_ID + "/normalization/train-only-3e3c6b9d347e5e71.json"
)
VIEW_MANIFEST_UNDER_RUNROOT = (
    EXPERIMENT_ID + "/dataset_views/train-only-3e3c6b9d347e5e71/view_manifest.json"
)

TRAIN_EPISODES = [
    49, 4, 23, 43, 21, 37, 18, 34, 0, 47, 38, 29, 3, 26, 14, 17, 44, 30, 15, 42,
    10, 35, 25, 32, 19, 36, 41, 28, 8, 27, 16, 11, 2, 20, 9, 39, 46, 48, 12, 40,
]
VALIDATION_EPISODES = [22, 13, 7, 33, 45]
HIDDEN_TEST_EPISODES = [31, 6, 1, 24, 5]

FORMAL_STEPS = 316
CHECKPOINT_STEPS = [79, 158, 237, 316]

# --- Candidate identity (the plan YAML is created only after the post-training
# path bound to this module is verified; nothing here authorizes training). ---

VCD_PLAN_ID = "m2-smolvla450m-vcdropout-001"
VCD_RUN_NAME = "m2-smolvla450m-vcdropout-cuda-b64-001"
VCD_SMOKE_RUN_NAME = "m2-smolvla450m-vcdropout-smoke-001"
VCD_PREFLIGHT_RUN_NAME = "m2-smolvla450m-vcdropout-preflight-001"
VCD_VALIDATION_PREFIX = "m2-smolvla450m-vcdropout-val"
VCD_RELOAD_SOURCE_ENV = "ROSETTA_VCD_RELOAD_SOURCE"
VCD_VALIDATION_PREFIX_ENV = "ROSETTA_VCD_VALIDATION_PREFIX_OVERRIDE"

# Frozen treatment contract (preregistration section 3, table).  The plan's
# ``visual_conditioning_contract`` must equal this mapping exactly.
VCD_CONTRACT = {
    "profile": "samplewise_normalized_state_dropout",
    "dropout_probability": 0.5,
    "generator_seed": 20260828,
    "input_space": "train_normalized_observation_state",
    "granularity": "whole_sample",
    "replacement": "normalized_zero",
    "rescale_retained_state": False,
    "generator": "dedicated_cpu_generator",
    "training_only": True,
    "target_semantics": "unchanged_absolute_expert_action",
    "upstream_implementation_sha256": (
        "37b1d56f37510732a087cf5c32c05cd15d6234201a3f002f108ec4c53438cc7d"
    ),
}

VISUAL_CONDITIONING_IMPLEMENTATION = (
    "src/rosetta_reality/vla/visual_conditioning.py"
)
VISUAL_CONDITIONING_SHA256 = (
    "8113fb1bb0c0dcf99111b970d926fd39c3235ebe6b356e35441903463e6c369c"
)

REQUIRED_FEATURES = frozenset(
    {
        "train_only_statistics",
        "action_boundary_projection",
        "checkpoint_memory_trim",
        "state_conditioning_dropout",
    }
)
# No-learning-semantics instrumentation the audit explicitly allows to share a
# training change: per-step pre/post-clip gradient norms (O2) and the exact
# same-step checkpoint metric row (T9).  A plan may declare them optionally;
# nothing else beyond the required single-axis stack is accepted.
INSTRUMENTATION_FEATURES = frozenset(
    {
        "gradient_clip_diagnostics",
        "checkpoint_metric_snapshot",
    }
)
FORBIDDEN_FEATURES = frozenset(
    {
        "state_robustness_jitter",
        "horizon_weight_profile",
        "fixed_frame_sampler",
    }
)

PINNED_OPTIMIZER_CONTRACT = {
    "optimizer": {
        "type": "adamw",
        "lr": 1.0e-4,
        "weight_decay": 1.0e-10,
        "grad_clip_norm": 10.0,
        "betas": [0.9, 0.95],
        "eps": 1.0e-8,
    },
    "scheduler": {
        "type": "cosine_decay_with_warmup",
        "num_warmup_steps": 31,
        "num_decay_steps": 316,
        "peak_lr": 1.0e-4,
        "decay_lr": 2.5e-6,
    },
}

# The candidate plan must pin at least this implementation inventory; extra
# entries are allowed.  Digests are not frozen here — the plan declares them
# and every post-training entry point verifies them against the working tree.
REQUIRED_IMPLEMENTATION_FILES = (
    "scripts/run_smolvla_v2.py",
    "scripts/train_smolvla_v2.py",
    "scripts/smolvla_forward_check.py",
    "scripts/smolvla_vcdropout_protocol.py",
    "scripts/smolvla_vcdropout_validate.py",
    "scripts/select_smolvla_vcdropout_checkpoint.py",
    "scripts/export_smolvla_vcdropout.py",
    "scripts/gate_smolvla_vcdropout_visual_conditioning.py",
    "src/rosetta_reality/vla/training/plan.py",
    "src/rosetta_reality/vla/training/features.py",
    "src/rosetta_reality/vla/training/context.py",
    "src/rosetta_reality/vla/training/launch.py",
    "src/rosetta_reality/vla/action_space.py",
    "src/rosetta_reality/vla/processor.py",
    VISUAL_CONDITIONING_IMPLEMENTATION,
)

# --- Immutable control evidence (preregistration section 4). ---

CONTROL_ARTIFACT_ID = "m2-smolvla450m-zen-cuda-b64-uniform-001-step0316-deploy-001"
CONTROL_ARTIFACT_MANIFEST_SHA256 = (
    "ecc73b9e26f43b6dc85981e61e2ed3aaa651b208b1cd72ad26f93877599694ec"
)
CONTROL_DIAGNOSTIC_REPORT = (
    "reports/training/m2-smolvla-zen-module-gradient-diagnostic-2026-08-28.json"
)
CONTROL_DIAGNOSTIC_SHA256 = (
    "5ee9824e8bea8031094a7d650a9f9e594a5b55f96470f9ba7eda5065dc38190c"
)
CONTROL_NORMAL_MEAN_FLOW_LOSS = 0.1467

# --- Frozen offset-250 gradient gate (preregistration section 4). ---

GATE_FRAME_OFFSET = 250
GATE_EPISODES = [22, 13, 7, 33, 45]
GATE_SHUFFLE_SEED = 20260812
GATE_FLOW_TIME = 0.5
GATE_TRAINABLE_GROUPS = ("action_expert", "state_projector", "action_io_projections")
GATE_NORMAL_FLOW_LOSS_MAX = 0.22005
GATE_STATE_SENSITIVITY_MAX = 0.70
GATE_IMAGE_SENSITIVITY_MIN = 0.10
GATE_STATE_DOMINANCE_MAX = 0.453


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def _feature_names(plan: dict[str, Any]) -> list[str]:
    declarations = plan.get("features")
    if not isinstance(declarations, list):
        raise ValueError("Candidate plan features must be a list.")
    names = [
        declaration.get("name")
        for declaration in declarations
        if isinstance(declaration, dict)
    ]
    if len(set(names)) != len(names):
        raise ValueError("Candidate plan declares a feature twice.")
    return [str(name) for name in names]


def gate_sensitivity(ratios: Mapping[str, float]) -> float:
    """Mean ``abs(log(ratio))`` over the three trainable module groups."""

    values: list[float] = []
    for group in GATE_TRAINABLE_GROUPS:
        if group not in ratios:
            raise ValueError(f"Gradient ratio missing for trainable group: {group}.")
        raw = ratios[group]
        if raw is None or isinstance(raw, bool) or not isinstance(raw, int | float):
            raise ValueError(
                f"Gradient ratio is not numeric (degenerate normal gradient): {group}."
            )
        ratio = float(raw)
        if not math.isfinite(ratio) or ratio <= 0.0:
            raise ValueError(f"Gradient ratio is not finite and positive: {group}.")
        values.append(abs(math.log(ratio)))
    return sum(values) / len(values)


def evaluate_gate_criteria(
    *,
    normal_mean_flow_loss: float,
    state_shuffle_gradient_ratios: Mapping[str, float],
    image_shuffle_gradient_ratios: Mapping[str, float],
    normal_trainable_gradient_norms: Mapping[str, float],
    pairwise_sample_state_difference: float,
    pairwise_sample_image_difference: float,
) -> dict[str, Any]:
    """Evaluate the six preregistered gate criteria against one measurement."""

    state_sensitivity = gate_sensitivity(state_shuffle_gradient_ratios)
    image_sensitivity = gate_sensitivity(image_shuffle_gradient_ratios)
    dominance = state_sensitivity - image_sensitivity
    norms_finite_nonzero = all(
        math.isfinite(float(normal_trainable_gradient_norms[group]))
        and float(normal_trainable_gradient_norms[group]) > 0.0
        for group in GATE_TRAINABLE_GROUPS
    )
    criteria = [
        {
            "name": "trainable_normal_gradients_finite_nonzero",
            "preregistration": "criterion 1",
            "passed": bool(norms_finite_nonzero),
        },
        {
            "name": "sample_state_image_diversity_guards",
            "preregistration": "criterion 2",
            "measured": {
                "pairwise_sample_state_difference": pairwise_sample_state_difference,
                "pairwise_sample_image_difference": pairwise_sample_image_difference,
            },
            "passed": bool(
                pairwise_sample_state_difference > 0.0
                and pairwise_sample_image_difference > 0.0
            ),
        },
        {
            "name": "normal_mean_flow_loss_ceiling",
            "preregistration": "criterion 3",
            "measured": float(normal_mean_flow_loss),
            "threshold_max": GATE_NORMAL_FLOW_LOSS_MAX,
            "passed": bool(normal_mean_flow_loss <= GATE_NORMAL_FLOW_LOSS_MAX),
        },
        {
            "name": "state_sensitivity_ceiling",
            "preregistration": "criterion 4",
            "measured": state_sensitivity,
            "threshold_max": GATE_STATE_SENSITIVITY_MAX,
            "passed": bool(state_sensitivity <= GATE_STATE_SENSITIVITY_MAX),
        },
        {
            "name": "image_sensitivity_floor",
            "preregistration": "criterion 5",
            "measured": image_sensitivity,
            "threshold_min": GATE_IMAGE_SENSITIVITY_MIN,
            "passed": bool(image_sensitivity >= GATE_IMAGE_SENSITIVITY_MIN),
        },
        {
            "name": "state_dominance_ceiling",
            "preregistration": "criterion 6",
            "measured": dominance,
            "threshold_max": GATE_STATE_DOMINANCE_MAX,
            "passed": bool(dominance <= GATE_STATE_DOMINANCE_MAX),
        },
    ]
    return {
        "metrics": {
            "normal_mean_flow_loss": float(normal_mean_flow_loss),
            "state_sensitivity": state_sensitivity,
            "image_sensitivity": image_sensitivity,
            "state_dominance_score": dominance,
        },
        "criteria": criteria,
        "passed": all(criterion["passed"] for criterion in criteria),
    }


def control_baseline(*, file_sha256=None) -> dict[str, Any]:
    """Load the pinned Zen-uniform offset-250 baseline from immutable evidence."""

    from rosetta_reality.experiment import file_sha256 as default_hash

    digest = file_sha256 or default_hash
    report_path = Path(__file__).resolve().parents[1] / CONTROL_DIAGNOSTIC_REPORT
    if digest(report_path) != CONTROL_DIAGNOSTIC_SHA256:
        raise ValueError("The control gradient-diagnostic report checksum changed.")
    import json

    report = json.loads(report_path.read_text(encoding="utf-8"))
    uniform = report["offset_250_results"]["uniform"]
    state_ratios = uniform["state_shuffle"]["gradient_ratios"]
    image_ratios = uniform["image_shuffle"]["gradient_ratios"]
    return {
        "artifact_id": CONTROL_ARTIFACT_ID,
        "artifact_manifest_sha256": CONTROL_ARTIFACT_MANIFEST_SHA256,
        "report": CONTROL_DIAGNOSTIC_REPORT,
        "report_sha256": CONTROL_DIAGNOSTIC_SHA256,
        "normal_mean_flow_loss": float(uniform["normal_mean_loss"]),
        "state_shuffle_gradient_ratios": {
            group: float(state_ratios[group]) for group in GATE_TRAINABLE_GROUPS
        },
        "image_shuffle_gradient_ratios": {
            group: float(image_ratios[group]) for group in GATE_TRAINABLE_GROUPS
        },
        "state_sensitivity": gate_sensitivity(state_ratios),
        "image_sensitivity": gate_sensitivity(image_ratios),
        "state_dominance_score": gate_sensitivity(state_ratios)
        - gate_sensitivity(image_ratios),
    }


def validate_vcdropout_plan(
    plan: dict[str, Any],
    *,
    feature_names: list[str] | None = None,
    file_sha256=None,
) -> str:
    """Validate the frozen invariant surface of the candidate plan."""

    from rosetta_reality.experiment import file_sha256 as default_hash
    from rosetta_reality.vla.training.plan import validate_optimizer_contract

    digest = file_sha256 or default_hash
    repository_root = Path(__file__).resolve().parents[1]

    if plan.get("schema_version") != 2 or plan.get("role") != "vla":
        raise ValueError("The candidate plan must be role=vla schema_version=2.")
    if plan.get("plan_id") != VCD_PLAN_ID:
        raise ValueError(f"Unknown candidate plan identity: {plan.get('plan_id')!r}.")
    if plan.get("status") != "preregistered":
        raise ValueError("The candidate plan must remain preregistered.")
    if plan.get("run_name") != VCD_RUN_NAME:
        raise ValueError("The candidate run_name differs from the registration.")

    parent = plan.get("parent_experiment", {})
    parent_path = repository_root / PARENT_CONFIG
    if (
        parent.get("config") != PARENT_CONFIG
        or parent.get("experiment_id") != EXPERIMENT_ID
        or parent.get("sha256") != PARENT_SHA256
        or digest(parent_path) != PARENT_SHA256
    ):
        raise ValueError("Candidate parent experiment identity changed.")

    runtime = plan.get("runtime_profile", {})
    profile_path = repository_root / "configs/runtime/autodl_rtx4090.yaml"
    if (
        runtime.get("sha256") != RUNTIME_PROFILE_SHA256
        or digest(profile_path) != RUNTIME_PROFILE_SHA256
        or runtime.get("profile_id") != RUNTIME_PROFILE_ID
        or runtime.get("nested_docker_used") is not False
    ):
        raise ValueError("Candidate runtime profile identity changed.")

    initialization = plan.get("initialization", {})
    expected_initialization = {
        "source": "revision_pinned_base_model",
        "faust_checkpoint_used": False,
        "aster_checkpoint_used": False,
        "way_checkpoint_used": False,
        "zen_checkpoint_used": False,
        "optimizer_state_reused": False,
    }
    if initialization != expected_initialization:
        raise ValueError("Candidate initialization boundary changed.")

    training = plan.get("training", {})
    if (
        list(training.get("episodes", [])) != TRAIN_EPISODES
        or training.get("batch_size") != 64
        or training.get("steps") != FORMAL_STEPS
        or training.get("save_freq") != 79
        or training.get("log_freq") != 79
        or list(training.get("checkpoint_steps", [])) != CHECKPOINT_STEPS
        or training.get("eval_split") != 0.0
        or training.get("hidden_test_loaded") is not False
        or validate_optimizer_contract(training) != PINNED_OPTIMIZER_CONTRACT
    ):
        raise ValueError("Candidate training contract differs from the registration.")

    policy_overlay = training.get("policy", {})
    if (
        policy_overlay.get("empty_cameras") != 2
        or policy_overlay.get("compile_model") is not False
        or policy_overlay.get("compile_mode") != "default"
        or policy_overlay.get("skip_fully_masked_camera_encoding") is not True
    ):
        raise ValueError("Candidate policy overlay differs from the registration.")

    names = feature_names if feature_names is not None else _feature_names(plan)
    if FORBIDDEN_FEATURES & set(names):
        raise ValueError("Candidate plan declares a forbidden co-treatment feature.")
    declared = set(names)
    missing = sorted(REQUIRED_FEATURES - declared)
    if missing:
        raise ValueError(
            "Candidate features differ from the registered single-axis stack "
            f"(missing: {missing})."
        )
    unexpected = sorted(declared - REQUIRED_FEATURES - INSTRUMENTATION_FEATURES)
    if unexpected:
        raise ValueError(
            "Candidate plan declares features outside the registered stack: "
            f"{unexpected}."
        )
    if plan.get("loss_contract") is not None:
        raise ValueError("The candidate must keep the uniform flow loss (no contract).")
    if plan.get("state_robustness_contract") is not None:
        raise ValueError("The candidate must not declare a state-robustness contract.")
    if plan.get("visual_conditioning_contract") != VCD_CONTRACT:
        raise ValueError("The visual-conditioning treatment contract changed.")

    resources = plan.get("resources", {})
    if (
        resources.get("memory_limit") != "autodl_platform_container"
        or resources.get("memory_swap_limit") != resources.get("memory_limit")
        or resources.get("mixed_precision") != "bf16"
        or resources.get("checkpoint_memory_trim") is not True
    ):
        raise ValueError("Candidate resource boundary differs from the registration.")

    validation = plan.get("validation", {})
    if (
        list(validation.get("episodes", [])) != VALIDATION_EPISODES
        or not set(validation.get("episodes", [])).isdisjoint(TRAIN_EPISODES)
        or not set(validation.get("episodes", [])).isdisjoint(HIDDEN_TEST_EPISODES)
        or validation.get("frame_offsets") != [0]
        or validation.get("total_samples") != len(VALIDATION_EPISODES)
        or validation.get("hidden_test_loaded") is not False
    ):
        raise ValueError("Candidate validation protocol differs from the registration.")

    monitoring = plan.get("monitoring", {})
    if (
        monitoring.get("policy") != "sleep_between_quarter_checkpoints"
        or monitoring.get("wake_steps") != CHECKPOINT_STEPS
        or monitoring.get("blocking_command") != "sleep"
        or int(monitoring.get("sleep_poll_seconds", -1)) != 300
        or monitoring.get("hidden_test_loaded") is not False
    ):
        raise ValueError("Candidate monitoring differs from the quarter-only policy.")

    if not isinstance(plan.get("prerequisites", {}), dict):
        raise ValueError("Candidate prerequisites must be a mapping.")

    normalization = plan.get("normalization", {})
    report_relative = repository_root / NORMALIZATION_REPORT_RELATIVE
    view_relative = repository_root / VIEW_MANIFEST_RELATIVE
    if (
        normalization.get("source_split") != "train"
        or normalization.get("report") != NORMALIZATION_REPORT_RELATIVE
        or normalization.get("report_sha256") != NORMALIZATION_REPORT_SHA256
        or normalization.get("dataset_view_manifest") != VIEW_MANIFEST_RELATIVE
        or normalization.get("dataset_view_manifest_sha256") != VIEW_MANIFEST_SHA256
        or digest(report_relative) != NORMALIZATION_REPORT_SHA256
        or digest(view_relative) != VIEW_MANIFEST_SHA256
    ):
        raise ValueError("Candidate train-only normalization identity changed.")

    implementation = plan.get("implementation_files", {})
    if not isinstance(implementation, dict) or not implementation:
        raise ValueError("Candidate plan has no implementation inventory.")
    missing = sorted(set(REQUIRED_IMPLEMENTATION_FILES) - set(implementation))
    if missing:
        raise ValueError(f"Candidate implementation inventory is missing: {missing}.")
    for relative, expected in implementation.items():
        if not isinstance(expected, str) or not _is_sha256(expected):
            raise ValueError(f"Implementation pin is not a SHA-256: {relative}.")
        if digest(repository_root / relative) != expected:
            raise ValueError(f"Implementation file changed: {relative}.")
    if (
        implementation.get(VISUAL_CONDITIONING_IMPLEMENTATION)
        != VISUAL_CONDITIONING_SHA256
    ):
        raise ValueError("The state-dropout implementation differs from registration.")

    stop_conditions = plan.get("stop_conditions")
    if not isinstance(stop_conditions, list) or not stop_conditions:
        raise ValueError("Candidate stop conditions are missing.")
    if plan.get("hidden_test_loaded") is not False:
        raise ValueError("Candidate hidden-test boundary is open.")
    return VCD_PLAN_ID


def resolve_plan(path: Path, *, file_sha256=None) -> tuple[dict[str, Any], str]:
    """Load and validate the candidate plan, returning (plan, plan_id)."""

    plan = load_yaml(Path(path).resolve())
    return plan, validate_vcdropout_plan(plan, file_sha256=file_sha256)
