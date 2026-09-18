"""Shared preregistered identity and validation helpers for the Zen formal program.

Zen trains two arms of one registered comparison through the version-2
harness (see ``configs/vla/smolvla_450m_aloha_insertion_zen_cuda_b64_*.yaml``):

- ``m2-smolvla450m-zen-uniform-002`` (control, uniform flow loss);
- ``m2-smolvla450m-zen-firstaction-001`` (treatment,
  ``first_action_only`` temporal weight profile).

Everything in this module is deliberately dependency-light (stdlib plus the
existing checksum helpers) so the post-training chain can validate plans
without importing torch.
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

FORMAL_STEPS = 316
CHECKPOINT_STEPS = [79, 158, 237, 316]

LOSS_UPSTREAM_SHA256 = "37b1d56f37510732a087cf5c32c05cd15d6234201a3f002f108ec4c53438cc7d"

IMPLEMENTATION_FILES = {
    "scripts/run_smolvla_v2.py": (
        "d703b63b8ebf96b9579cb7af41e3ec84645a9a225c21a72698451ee782559327"
    ),
    "scripts/train_smolvla_v2.py": (
        "888cca61cbacee7ce72527cc78f232c0b783f8c6bb432d11bf7e75740e89c52a"
    ),
    "scripts/smolvla_forward_check.py": (
        "7d7dd79167c604bae13e525fa9a6f51a9a1fec5e331a1352120d49d60f04e3d0"
    ),
    "src/rosetta_reality/vla/training/plan.py": (
        "6b8afadca01b1cff2a1a7075b1884c0362f327810363cc589e42def336f9bcf9"
    ),
    "src/rosetta_reality/vla/training/features.py": (
        "3b312e5363c835381151042a2d92b7ca15c96a5757eb04120c89112b854811a0"
    ),
    "src/rosetta_reality/vla/training/context.py": (
        "3681409b19dbae403a22dbf5dcff64e8d767b0ffea59b4be1260c2f1538c68bd"
    ),
    "src/rosetta_reality/vla/training/launch.py": (
        "c7a3d2efa8b14a531c01a5c81df12ab47ea146c9048e89a21a1346f2525d4de8"
    ),
    "src/rosetta_reality/vla/action_space.py": (
        "4321d7d76e39db8644500be4c02f6de89caadfd58832047fde493450df1cfbeb"
    ),
    "src/rosetta_reality/vla/processor.py": (
        "6751d4dd901da27e0a299bd9426fa484540e85dc12f1f1a62694e063d07e2384"
    ),
}

ZEN_SPECS: dict[str, dict[str, Any]] = {
    "m2-smolvla450m-zen-uniform-002": {
        "role": "control",
        "run_name": "m2-smolvla450m-zen-cuda-b64-uniform-001",
        "validation_prefix": "m2-smolvla450m-zen-cuda-b64-uniform-val",
        "horizon_feature_declared": False,
    },
    "m2-smolvla450m-zen-firstaction-001": {
        "role": "treatment",
        "run_name": "m2-smolvla450m-zen-cuda-b64-firstaction-001",
        "validation_prefix": "m2-smolvla450m-zen-cuda-b64-firstaction-val",
        "horizon_feature_declared": True,
    },
}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def validate_zen_plan(
    plan: dict[str, Any],
    *,
    feature_names: list[str] | None = None,
    file_sha256=None,
) -> str:
    """Validate one frozen invariant surface shared by every Zen entry point."""

    from rosetta_reality.experiment import file_sha256 as default_hash
    from rosetta_reality.vla.training.plan import validate_optimizer_contract

    digest = file_sha256 or default_hash
    repository_root = Path(__file__).resolve().parents[1]

    if plan.get("schema_version") != 2 or plan.get("role") != "vla":
        raise ValueError("Zen plans must be role=vla schema_version=2.")
    plan_id = plan.get("plan_id")
    spec = ZEN_SPECS.get(str(plan_id))
    if spec is None:
        raise ValueError(f"Unknown Zen plan identity: {plan_id!r}.")
    if plan.get("status") != "preregistered":
        raise ValueError("Zen plans must remain preregistered.")
    if plan.get("run_name") != spec["run_name"]:
        raise ValueError("Zen run_name differs from the registered spec.")

    parent = plan.get("parent_experiment", {})
    parent_path = repository_root / PARENT_CONFIG
    if (
        parent.get("config") != PARENT_CONFIG
        or parent.get("experiment_id") != EXPERIMENT_ID
        or parent.get("sha256") != PARENT_SHA256
        or digest(parent_path) != PARENT_SHA256
    ):
        raise ValueError("Zen parent experiment identity changed.")

    runtime = plan.get("runtime_profile", {})
    profile_path = repository_root / "configs/runtime/autodl_rtx4090.yaml"
    if (
        runtime.get("sha256") != RUNTIME_PROFILE_SHA256
        or digest(profile_path) != RUNTIME_PROFILE_SHA256
        or runtime.get("profile_id") != "autodl-rtx4090-cuda-001"
        or runtime.get("nested_docker_used") is not False
    ):
        raise ValueError("Zen runtime profile identity changed.")

    initialization = plan.get("initialization", {})
    expected_initialization = {
        "source": "revision_pinned_base_model",
        "faust_checkpoint_used": False,
        "aster_checkpoint_used": False,
        "way_checkpoint_used": False,
        "prometheus_checkpoint_used": False,
        "optimizer_state_reused": False,
    }
    if initialization != expected_initialization:
        raise ValueError("Zen initialization boundary changed.")

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
        or not validate_optimizer_contract(training)
    ):
        raise ValueError("Zen training contract differs from the registration.")

    policy_overlay = training.get("policy", {})
    if (
        policy_overlay.get("empty_cameras") != 2
        or policy_overlay.get("compile_model") is not False
        or policy_overlay.get("compile_mode") != "default"
        or policy_overlay.get("skip_fully_masked_camera_encoding") is not True
    ):
        raise ValueError("Zen policy overlay differs from the registration.")

    resources = plan.get("resources", {})
    if (
        resources.get("memory_limit") != "autodl_platform_container"
        or resources.get("memory_swap_limit") != resources.get("memory_limit")
        or resources.get("mixed_precision") != "bf16"
        or resources.get("checkpoint_memory_trim") is not True
    ):
        raise ValueError("Zen resource boundary differs from the registration.")

    features = feature_names
    if features is None:
        declarations = plan.get("features")
        if not isinstance(declarations, list):
            raise ValueError("Zen plan features must be a list.")
        features = [
            declaration.get("name")
            for declaration in declarations
            if isinstance(declaration, dict)
        ]
    if bool("horizon_weight_profile" in features) is not bool(
        spec["horizon_feature_declared"]
    ):
        raise ValueError(f"{plan_id}: horizon feature declaration contradicts its role.")
    if bool(spec["horizon_feature_declared"]):
        contract = plan.get("loss_contract", {})
        if (
            contract.get("profile") != "first_action_only"
            or contract.get("chunk_size") != 50
            or contract.get("normalization") != "mean_over_selected_valid_entries"
            or contract.get("upstream_implementation_sha256") != LOSS_UPSTREAM_SHA256
        ):
            raise ValueError(f"{plan_id}: loss contract differs from the registration.")
    elif "loss_contract" in plan:
        raise ValueError(f"{plan_id}: control arms must not carry a loss contract.")
    if len(set(features)) != len(features):
        raise ValueError(f"{plan_id}: duplicate feature declarations.")

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
        raise ValueError(f"{plan_id}: train-only normalization identity changed.")

    implementation = plan.get("implementation_files", {})
    if set(implementation) != set(IMPLEMENTATION_FILES):
        raise ValueError(f"{plan_id}: implementation inventory differs from registration.")
    for relative, expected in IMPLEMENTATION_FILES.items():
        if implementation[relative] != expected:
            raise ValueError(f"{plan_id}: implementation pin drifted: {relative}.")
        if digest(repository_root / relative) != expected:
            raise ValueError(f"{plan_id}: implementation file changed: {relative}.")

    validation = plan.get("validation", {})
    if (
        list(validation.get("episodes", [])) != VALIDATION_EPISODES
        or not set(validation.get("episodes", [])).isdisjoint(TRAIN_EPISODES)
        or not set(validation.get("episodes", [])).isdisjoint(HIDDEN_TEST_EPISODES)
        or validation.get("frame_offsets") != [0]
        or validation.get("total_samples") != len(VALIDATION_EPISODES)
        or validation.get("hidden_test_loaded") is not False
    ):
        raise ValueError(f"{plan_id}: validation protocol differs from the registration.")

    stop_conditions = plan.get("stop_conditions")
    if not isinstance(stop_conditions, list) or not stop_conditions:
        raise ValueError(f"{plan_id}: stop conditions missing.")
    if plan.get("hidden_test_loaded") is not False:
        raise ValueError(f"{plan_id}: hidden-test boundary open.")
    return str(plan_id)


def resolve_plan(path: Path, *, file_sha256=None) -> tuple[dict[str, Any], str]:
    """Load and validate one Zen plan, returning (plan, plan_id)."""

    plan = load_yaml(Path(path).resolve())
    return plan, validate_zen_plan(plan, file_sha256=file_sha256)
