"""Frozen identity module for the vision-front-end-unfreeze candidate.

Candidate ``m2-smolvla450m-vfunfreeze-001`` tests one registered axis against
the immutable Zen-uniform control lineage: exactly the visual front-end
(``vlm.model.vision_model`` + ``vlm.model.connector``) joins the trainable set
inside the trainer process, while the language model, the saved policy flags
and every other training semantic stay on the frozen baseline.  The repaired
September-5 diagnostics motivate the axis (frame-0 image sensitivity without
expert alignment; no linearly decodable object placement in the frozen
tower/connector readouts).  Acceptance is not offline MAE: the registered
chain is the two-step optimizer smoke's non-zero vision-update verification,
validation-only selection, exact-reload export, the corrected frame-0 paired
alignment probe on the exported artifact, and a separately registered Gate
3/4 comparison.  Nothing in this module authorizes training by itself.

Shared immutable identities (parent experiment, runtime profile, split,
normalization, optimizer contract) intentionally duplicate the frozen
vcdropout protocol values; ``tests/test_smolvla_vfunfreeze_protocol.py``
fails closed if the two modules ever drift apart.
"""

from __future__ import annotations

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

FORMAL_STEPS = 632
CHECKPOINT_STEPS = [158, 316, 474, 632]

# --- Candidate identity ---

VFU_PLAN_ID = "m2-smolvla450m-vfunfreeze-003"
VFU_RUN_NAME = "m2-smolvla450m-vfunfreeze-cuda-b32-003"
VFU_SMOKE_RUN_NAME = "m2-smolvla450m-vfunfreeze-smoke-003"
VFU_PREFLIGHT_RUN_NAME = "m2-smolvla450m-vfunfreeze-preflight-003"
VFU_VALIDATION_PREFIX = "m2-smolvla450m-vfunfreeze-val"
VFU_RELOAD_SOURCE_ENV = "ROSETTA_VFU_RELOAD_SOURCE"
VFU_VALIDATION_PREFIX_ENV = "ROSETTA_VFU_VALIDATION_PREFIX_OVERRIDE"

# Frozen treatment contract.  The plan's ``vision_front_end_contract`` must
# equal this mapping exactly.
VFU_CONTRACT = {
    "profile": "visual_front_end_unfreeze",
    "scope": ["vlm.model.vision_model", "vlm.model.connector"],
    "language_model": "frozen",
    "scope_application": "trainer_process_only_after_make_policy",
    "activation_checkpointing": "per_module_nonreentrant_vision_front_end",
    "upstream_scope_flags": "unchanged_frozen_baseline",
    "saved_policy_config_flags": "unchanged_frozen_baseline",
    "deployment": "scope_not_installed",
    "training_only": True,
    "target_semantics": "unchanged_absolute_expert_action",
}

VISION_FRONT_END_IMPLEMENTATION = "src/rosetta_reality/vla/vision_front_end.py"

REQUIRED_FEATURES = frozenset(
    {
        "train_only_statistics",
        "action_boundary_projection",
        "checkpoint_memory_trim",
        "vision_front_end_unfreeze",
    }
)
INSTRUMENTATION_FEATURES = frozenset(
    {
        "gradient_clip_diagnostics",
        "checkpoint_metric_snapshot",
    }
)
FORBIDDEN_FEATURES = frozenset(
    {
        "state_robustness_jitter",
        "state_conditioning_dropout",
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
        "num_warmup_steps": 62,
        "num_decay_steps": 632,
        "peak_lr": 1.0e-4,
        "decay_lr": 2.5e-6,
    },
}

REQUIRED_IMPLEMENTATION_FILES = (
    "scripts/run_smolvla_v2.py",
    "scripts/train_smolvla_v2.py",
    "scripts/smolvla_forward_check.py",
    "scripts/smolvla_vfunfreeze_protocol.py",
    "scripts/smolvla_vfunfreeze_validate.py",
    "scripts/select_smolvla_vfunfreeze_checkpoint.py",
    "scripts/export_smolvla_vfunfreeze.py",
    "scripts/verify_vfunfreeze_smoke_updates.py",
    "src/rosetta_reality/vla/training/plan.py",
    "src/rosetta_reality/vla/training/features.py",
    "src/rosetta_reality/vla/training/context.py",
    "src/rosetta_reality/vla/training/launch.py",
    "src/rosetta_reality/vla/action_space.py",
    "src/rosetta_reality/vla/processor.py",
    VISION_FRONT_END_IMPLEMENTATION,
)

# --- Immutable context controls (read-only, never gates). ---

CONTROL_ARTIFACT_ID = "m2-smolvla450m-zen-cuda-b64-uniform-001-step0316-deploy-001"
CONTROL_ZEN_UNIFORM_FIRST_ACTION_MAE = 0.021572770214905695
CONTROL_VCDROPOUT_FIRST_ACTION_MAE = 0.029304
CONTROL_FRAME0_PAIRED_ALIGNMENT_REPORT = (
    "runs/"
    + EXPERIMENT_ID
    + "/diagnostics/frame0-paired-alignment-train-2026-09-05.json"
)


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


def validate_vfunfreeze_plan(
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
    if plan.get("plan_id") != VFU_PLAN_ID:
        raise ValueError(f"Unknown candidate plan identity: {plan.get('plan_id')!r}.")
    if plan.get("status") != "preregistered":
        raise ValueError("The candidate plan must remain preregistered.")
    if plan.get("run_name") != VFU_RUN_NAME:
        raise ValueError("Candidate run_name differs from the registration.")

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
        "vcdropout_checkpoint_used": False,
        "optimizer_state_reused": False,
    }
    if initialization != expected_initialization:
        raise ValueError("Candidate initialization boundary changed.")

    training = plan.get("training", {})
    if (
        list(training.get("episodes", [])) != TRAIN_EPISODES
        or training.get("batch_size") != 32
        or training.get("steps") != FORMAL_STEPS
        or training.get("save_freq") != 158
        or training.get("log_freq") != 158
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
    if plan.get("visual_conditioning_contract") is not None:
        raise ValueError("The candidate must not declare a state-dropout contract.")
    if plan.get("vision_front_end_contract") != VFU_CONTRACT:
        raise ValueError("The vision front-end treatment contract changed.")

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

    stop_conditions = plan.get("stop_conditions")
    if not isinstance(stop_conditions, list) or not stop_conditions:
        raise ValueError("Candidate stop conditions are missing.")
    if plan.get("hidden_test_loaded") is not False:
        raise ValueError("Candidate hidden-test boundary is open.")
    return VFU_PLAN_ID


def resolve_plan(path: Path, *, file_sha256=None) -> tuple[dict[str, Any], str]:
    """Load and validate the candidate plan, returning (plan, plan_id)."""

    plan = load_yaml(Path(path).resolve())
    return plan, validate_vfunfreeze_plan(plan, file_sha256=file_sha256)
