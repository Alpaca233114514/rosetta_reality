"""Prepare the fixed canonical endpoint without copying model weights.

This command is the runtime-side adapter for the registered post-training plan.
``prepare`` performs identity, storage, dataset metadata, and processor-file
checks only.  It creates a content-addressed artifact whose
``pretrained_model`` directory is a symlink to the verified step-5000
checkpoint.  It never loads a model, opens a video, creates an optimizer, or
starts a simulator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
DEFAULT_PLAN = (
    ROOT
    / "reports/training/m2-smolvla-canonical-fullframes-posttrain-gate-plan-004-2026-09-15.json"
)
DEFAULT_EXPERIMENT = "m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003"
DEFAULT_RUN_NAME = "canonical-fullframes-20260914-001"
DEFAULT_DATA_VIEW = (
    "m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/"
    "dataset_views/train-only-3e3c6b9d347e5e71"
)
DEFAULT_ARTIFACT_ID = "canonical-fullframes-20260914-001-step5000-zero-copy-001"
DIMENSION_NAMES = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
]


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    json.dumps(value, allow_nan=False)
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def resolve_input(raw: str, *, run_root: Path) -> Path:
    value = Path(raw)
    if value.is_absolute():
        return value.resolve()
    if value.parts and value.parts[0] == "runs":
        return (run_root / Path(*value.parts[1:])).resolve()
    return (ROOT / value).resolve()


def write_json_create_only(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Create-only output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")


def _action_contract_json(path: Path) -> dict[str, Any]:
    from dataclasses import asdict

    from rosetta_reality.sim import load_action_contract

    return json.loads(json.dumps(asdict(load_action_contract(path)), allow_nan=False))


def _all_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for directory, _dirs, names in os.walk(root, followlinks=True):
        for name in names:
            path = Path(directory) / name
            if path.is_file():
                files.append(path)
    return sorted(files)


def _artifact_files(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in _all_files(root):
        relative = path.relative_to(root).as_posix()
        if relative == "manifest.json":
            continue
        result[relative] = sha256(path)
    require(result, "Zero-copy artifact has no files")
    return result


def _build_config(
    plan: dict[str, Any],
    training_config: dict[str, Any],
    dataset_info: dict[str, Any],
    artifact_id: str,
    plan_sha256: str,
    normalization_sha256: str,
    view_manifest_sha256: str,
    selection_sha256: str,
) -> dict[str, Any]:
    model = training_config.get("model", {})
    policy = model.get("policy", {})
    adaptation = model.get("adaptation", {})
    action_space = model.get("action_space")
    require(isinstance(action_space, dict), "Training config has no explicit action-space identity")
    from rosetta_reality.vla.action_space import load_smolvla_action_space

    action_space = load_smolvla_action_space(training_config, require_explicit=True).as_dict()
    features = dataset_info.get("features")
    require(isinstance(features, dict), "Dataset metadata has no feature schema")
    return {
        "schema_version": 1,
        "artifact_id": artifact_id,
        "artifact_type": "smolvla_policy",
        "experiment_id": DEFAULT_EXPERIMENT,
        "selected_checkpoint_step": 5000,
        "selected_checkpoint_model_sha256": plan["endpoint"]["model_safetensors_sha256"],
        "formal_plan_sha256": plan_sha256,
        "selection_report_sha256": selection_sha256,
        "normalization_report_sha256": normalization_sha256,
        "dataset_view_manifest_sha256": view_manifest_sha256,
        "action_contract_sha256": plan["gate_protocol_authority"]["action_contract_sha256"],
        "dataset_id": "lerobot/aloha_sim_insertion_human",
        "dataset_revision": plan["data"]["dataset_revision"],
        "dataset_fps": int(dataset_info.get("fps", 50)),
        "dataset_features": features,
        "rename_map": training_config["dataset"]["rename_map"],
        "mixed_precision": "bf16",
        "instruction": "Insert the peg into the socket.",
        "policy": policy,
        "adaptation": adaptation,
        "action_space": action_space,
        "bounded_gripper_decoder": True,
        "upstream_repository": "huggingface/lerobot",
        "upstream_revision": "c903b114a90e703b3f7d0c46cb38727c328c55ff",
        "base_model": "lerobot/smolvla_base",
        "base_model_revision": plan["data"]["model_revision"],
        "hidden_test_loaded": False,
        "physical_robot_validated": False,
        "research_only": True,
        "zero_copy_pretrained_model": True,
        "optimizer_created": False,
        "optimizer_updates": 0,
    }


def prepare(args: argparse.Namespace) -> int:
    plan_path = args.plan.resolve()
    plan = load_json(plan_path)
    require(plan.get("status") == "preregistered", "Post-training plan is not preregistered")
    require(plan.get("fixed_endpoint_step") == 5000, "Endpoint must be fixed at step 5000")
    run_root = Path(args.run_root or os.environ.get("ROSETTA_RUN_ROOT", "")).resolve()
    artifact_root = Path(
        args.artifact_root or os.environ.get("ROSETTA_ARTIFACT_ROOT", "")
    ).resolve()
    require(
        run_root.is_absolute() and artifact_root.is_absolute(), "Runtime roots must be absolute"
    )
    pretrained = Path(args.pretrained_dir).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    from canonical_fullframes_contract import verify_files

    verify_files(pretrained, plan["endpoint"]["files"])
    require(pretrained.is_dir(), f"Pretrained checkpoint directory is missing: {pretrained}")
    require(dataset_root.is_dir(), f"Dataset view is missing: {dataset_root}")
    require((pretrained / "model.safetensors").is_file(), "Endpoint model.safetensors is missing")
    require(
        sha256(pretrained / "model.safetensors") == plan["endpoint"]["model_safetensors_sha256"],
        "Endpoint model SHA drift",
    )
    required_pretrained = {
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_7_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        "train_config.json",
    }
    actual = {path.name for path in pretrained.iterdir() if path.is_file()}
    require(required_pretrained <= actual, "Endpoint processor/config files are incomplete")
    info_path = dataset_root / "meta/info.json"
    require(info_path.is_file(), "Dataset metadata info.json is missing")
    dataset_info = load_json(info_path)
    require(
        dataset_info.get("total_episodes") == 50 and dataset_info.get("total_frames") == 25000,
        "Dataset metadata coverage drift",
    )
    features = dataset_info.get("features", {})
    require(features.get("observation.state", {}).get("shape") == [14], "Dataset state shape drift")
    require(features.get("action", {}).get("shape") == [14], "Dataset action shape drift")
    normalization = resolve_input(args.normalization_report, run_root=run_root)
    view_manifest = resolve_input(args.dataset_view_manifest, run_root=run_root)
    training_config_path = (
        ROOT / "configs/vla/smolvla_450m_aloha_insertion_action_repair_bounded_gripper_003.yaml"
    ).resolve()
    contract_path = (ROOT / "configs/sim/aloha_insertion_smolvla.yaml").resolve()
    for path in (normalization, view_manifest, training_config_path, contract_path):
        require(path.is_file(), f"Bound source is missing: {path}")
    require(
        sha256(training_config_path) == plan["source_identity"]["training_config_sha256"],
        "Training config SHA drift",
    )
    require(
        sha256(normalization) == plan["source_identity"]["normalization_report_sha256"],
        "Normalization report SHA drift",
    )
    require(
        sha256(view_manifest) == plan["source_identity"]["dataset_view_manifest_sha256"],
        "Dataset view manifest SHA drift",
    )
    require(
        sha256(contract_path) == plan["gate_protocol_authority"]["action_contract_sha256"],
        "Action Contract SHA drift",
    )
    recovery_preflight = load_json(Path(args.recovery_preflight).resolve())
    require(recovery_preflight.get("status") == "passed", "Recovery preflight is not passed")
    require(
        sha256(Path(args.recovery_preflight).resolve()) == plan["recovery_preflight_sha256"],
        "Recovery preflight seal drift",
    )
    free_bytes = shutil.disk_usage(artifact_root).free
    budget = plan["posttrain_runtime"]["storage_budget_basis"]
    required_free = int(budget["required_free_bytes"])
    require(
        free_bytes >= required_free,
        f"Durable free space below registered budget: {free_bytes} < {required_free}",
    )

    experiment_root = artifact_root / DEFAULT_EXPERIMENT
    artifact = experiment_root / str(plan["endpoint"]["artifact_id"])
    if artifact.exists():
        raise FileExistsError(f"Zero-copy artifact is create-only: {artifact}")
    selection = (
        run_root
        / DEFAULT_EXPERIMENT
        / "selection"
        / "canonical-fullframes-20260914-001-step5000-selection.json"
    )
    selection_payload = {
        "schema_version": 1,
        "status": "passed",
        "stage": "canonical_fullframes_fixed_endpoint_selection",
        "plan_id": plan["plan_id"],
        "plan_sha256": sha256(plan_path),
        "experiment_id": DEFAULT_EXPERIMENT,
        "selected": {
            "step": 5000,
            "model_safetensors_sha256": plan["endpoint"]["model_safetensors_sha256"],
            "selection_policy": "fixed_endpoint_no_checkpoint_search",
        },
        "retained_recovery_checkpoint_steps": [2500, 5000],
        "hidden_test_loaded": False,
        "optimizer_updates": 0,
        "recovery_preflight_sha256": sha256(Path(args.recovery_preflight).resolve()),
    }
    write_json_create_only(selection, selection_payload)
    selection_sha256 = sha256(selection)
    from rosetta_reality.vla.action_space import load_smolvla_experiment

    training_config = load_smolvla_experiment(training_config_path, ROOT)
    require(isinstance(training_config, dict), "Training config is invalid")
    artifact.mkdir(parents=True)
    os.symlink(pretrained, artifact / "pretrained_model", target_is_directory=True)
    shutil.copy2(normalization, artifact / "normalization.json")
    contract_json = _action_contract_json(contract_path)
    write_json_create_only(artifact / "action_contract.json", contract_json)
    config_payload = _build_config(
        plan,
        training_config,
        dataset_info,
        artifact.name,
        sha256(plan_path),
        sha256(normalization),
        sha256(view_manifest),
        selection_sha256,
    )
    write_json_create_only(artifact / "config.json", config_payload)
    model_card = (
        f"# {artifact.name}\n\n"
        "Research-only fixed endpoint artifact for canonical full-frame SmolVLA.\n"
        "The pretrained model directory is a zero-copy symlink to the verified\n"
        "step-5000 checkpoint; no physical-robot validation is claimed.\n"
    )
    with (artifact / "MODEL_CARD.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(model_card)
    files = _artifact_files(artifact)
    manifest = {
        "schema_version": 1,
        "status": "prepared",
        "artifact_id": artifact.name,
        "experiment_id": DEFAULT_EXPERIMENT,
        "selected_checkpoint_step": 5000,
        "selected_checkpoint_model_sha256": plan["endpoint"]["model_safetensors_sha256"],
        "plan_id": plan["plan_id"],
        "plan_sha256": sha256(plan_path),
        "selection_report_sha256": selection_sha256,
        "normalization_report_sha256": sha256(normalization),
        "dataset_view_manifest_sha256": sha256(view_manifest),
        "zero_copy_pretrained_model": True,
        "pretrained_model_bytes_not_duplicated": int(plan["endpoint"]["pretrained_model_bytes"]),
        "reload": {
            "verified": False,
            "exact_tensor_equality": False,
            "independent_processes": 2,
            "optimizer_created": False,
            "optimizer_updates": 0,
            "source": "new artifact loader reload pending",
            "recovery_preflight_sha256": sha256(Path(args.recovery_preflight).resolve()),
        },
        "hidden_test_loaded": False,
        "files": files,
    }
    write_json_create_only(artifact / "candidate-manifest.json", manifest)
    metadata_bytes = sum(path.stat().st_size for path in artifact.iterdir() if path.is_file())
    require(
        metadata_bytes <= int(budget["artifact_metadata_max_bytes"]),
        "Zero-copy artifact metadata exceeds sealed cap",
    )
    runtime_report = {
        "schema_version": 1,
        "status": "prepared",
        "stage": "canonical_fullframes_zero_copy_endpoint_prepare",
        "plan_id": plan["plan_id"],
        "plan_sha256": sha256(plan_path),
        "artifact_id": artifact.name,
        "artifact_manifest_sha256": sha256(artifact / "candidate-manifest.json"),
        "artifact_path": artifact.relative_to(ROOT).as_posix(),
        "pretrained_source": "checkpoints/005000/pretrained_model",
        "pretrained_model_bytes": int(plan["endpoint"]["pretrained_model_bytes"]),
        "artifact_metadata_bytes": int(metadata_bytes),
        "artifact_metadata_max_bytes": int(budget["artifact_metadata_max_bytes"]),
        "free_bytes_before": int(free_bytes),
        "required_free_bytes": required_free,
        "free_bytes_after": int(shutil.disk_usage(artifact_root).free),
        "zero_copy": True,
        "model_loaded": False,
        "optimizer_created": False,
        "hidden_test_loaded": False,
        "created_unix": time.time(),
    }
    report_path = (
        run_root
        / DEFAULT_EXPERIMENT
        / "posttrain"
        / "canonical-fullframes-20260914-001-zero-copy-prepare.json"
    )
    write_json_create_only(report_path, runtime_report)
    print(json.dumps(runtime_report, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare",))
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--pretrained-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--normalization-report", required=True)
    parser.add_argument("--dataset-view-manifest", required=True)
    parser.add_argument("--recovery-preflight", required=True)
    parser.add_argument("--run-root")
    parser.add_argument("--artifact-root")
    args = parser.parse_args()
    return prepare(args)


if __name__ == "__main__":
    raise SystemExit(main())
