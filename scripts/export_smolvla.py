"""Export and independently reload the selected formal SmolVLA policy artifact."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from torch.utils.data import default_collate

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
DEFAULT_PLAN = REPOSITORY_ROOT / "configs/vla/smolvla_450m_aloha_insertion_formal_001.yaml"
for root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import evaluate_smolvla_validation as evaluator  # noqa: E402
import run_smolvla_formal as formal_runner  # noqa: E402
import run_smolvla_phase as phase_runner  # noqa: E402
import select_smolvla_checkpoint as selector  # noqa: E402

from rosetta_reality.experiment import file_sha256, workspace_code_identity  # noqa: E402
from rosetta_reality.features import create_json  # noqa: E402
from rosetta_reality.sim import load_action_contract  # noqa: E402

ARTIFACT_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{2,79}")


def _validated_artifact_id(value: str) -> str:
    if ARTIFACT_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("--artifact-id must be one path-safe component.")
    return value


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _processors(policy_cfg: Any, policy: Any, dataset: Any, source: Path) -> tuple[Any, Any]:
    device = str(os.environ["ROSETTA_TORCH_DEVICE"])
    return make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=source,
        pretrained_revision=None,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "normalizer_processor": {
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
                "stats": dataset.meta.stats,
            },
            "rename_observations_processor": {
                "rename_map": {"observation.images.top": "observation.images.camera1"}
            },
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
                "stats": dataset.meta.stats,
            }
        },
    )


def _load_artifact_policy(source: Path, dataset: Any) -> tuple[Any, Any, Any]:
    device = str(os.environ["ROSETTA_TORCH_DEVICE"])
    cfg = SmolVLAConfig.from_pretrained(source, local_files_only=True)
    cfg.device = device
    cfg.pretrained_path = source
    cfg.pretrained_revision = None
    cfg.load_vlm_weights = False
    policy = make_policy(
        cfg=cfg,
        ds_meta=dataset.meta,
        rename_map={"observation.images.top": "observation.images.camera1"},
    )
    preprocessor, postprocessor = _processors(cfg, policy, dataset, source)
    return policy, preprocessor, postprocessor


def _fixed_prediction(
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    dataset: Any,
    relative_index: int,
    *,
    mixed_precision: str,
) -> torch.Tensor:
    batch = default_collate([dataset[relative_index]])
    for camera_key in dataset.meta.camera_keys:
        if camera_key in batch and batch[camera_key].dtype == torch.uint8:
            batch[camera_key] = batch[camera_key].float() / 255
    batch = preprocessor(batch)
    action = batch.get("action")
    if not isinstance(action, torch.Tensor):
        raise ValueError("The artifact reload sample has no action tensor.")
    noise = torch.zeros(
        (1, policy.config.chunk_size, policy.config.max_action_dim),
        device=action.device,
        dtype=action.dtype,
    )
    policy.eval()
    policy.reset()
    autocast_dtype = evaluator._autocast_dtype(mixed_precision)
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=action.device.type,
            dtype=autocast_dtype,
            enabled=autocast_dtype is not None,
        ),
    ):
        prediction = policy.predict_action_chunk(batch, noise=noise)
    prediction = postprocessor(prediction)
    if not isinstance(prediction, torch.Tensor) or not bool(torch.isfinite(prediction).all()):
        raise FloatingPointError("The independently loaded artifact produced an invalid action.")
    return prediction.detach().cpu()


def _copy_policy(source: Path, destination: Path) -> None:
    required = [
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ]
    destination.mkdir()
    for name in required:
        path = source / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Selected policy file is missing: {name}.")
        shutil.copy2(path, destination / name)
    tokenizer = source / "tokenizer"
    if not tokenizer.is_dir():
        raise FileNotFoundError("Selected policy tokenizer directory is missing.")
    shutil.copytree(tokenizer, destination / "tokenizer")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--artifact-id", required=True)
    args = parser.parse_args()
    artifact_id = _validated_artifact_id(args.artifact_id)
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("HF_DATASETS_OFFLINE") != "1":
        raise RuntimeError("SmolVLA export must run with networking disabled.")

    plan_path = args.plan.resolve()
    plan, base_path, experiment = formal_runner._validate_plan(plan_path)
    contract_path = REPOSITORY_ROOT / str(experiment["action_contract"]["derived"])
    contract_sha256 = file_sha256(contract_path)
    normalization_path, view_manifest_path, dataset_root = formal_runner._validate_normalization(
        plan, experiment, base_path, contract_sha256
    )
    selection_path = args.selection_report.resolve()
    selection = formal_runner._load_json(selection_path)
    selected = selection.get("selected", {})
    step = int(selected.get("step", 0))
    if (
        selection.get("status") != "passed"
        or selection.get("stage") != "smolvla_formal_checkpoint_selection"
        or selection.get("formal_plan_sha256") != file_sha256(plan_path)
        or selection.get("hidden_test_loaded") is not False
        or step not in plan["training"]["checkpoint_steps"]
    ):
        raise ValueError("SmolVLA selection report is invalid.")
    checkpoint_root = phase_runner._absolute_root("ROSETTA_CHECKPOINT_ROOT")
    source_dir = (
        checkpoint_root
        / str(experiment["experiment_id"])
        / "formal"
        / str(plan["run_name"])
        / "checkpoints"
        / f"{step:06d}"
        / "pretrained_model"
    )
    selected_files = {
        "model_safetensors_sha256": source_dir / "model.safetensors",
        "policy_config_sha256": source_dir / "config.json",
        "preprocessor_config_sha256": source_dir / "policy_preprocessor.json",
        "postprocessor_config_sha256": source_dir / "policy_postprocessor.json",
        "preprocessor_statistics_sha256": selector._processor_state_file(
            source_dir, "policy_preprocessor.json", "normalizer_processor"
        ),
        "postprocessor_statistics_sha256": selector._processor_state_file(
            source_dir, "policy_postprocessor.json", "unnormalizer_processor"
        ),
    }
    for hash_name, path in selected_files.items():
        if not path.is_file() or file_sha256(path) != selected.get(hash_name):
            raise ValueError(f"Selected checkpoint file changed before export: {path.name}.")
    if selector._tokenizer_hashes(source_dir) != selected.get("tokenizer_files_sha256"):
        raise ValueError("Selected checkpoint tokenizer changed before export.")

    artifact_root = phase_runner._absolute_root("ROSETTA_ARTIFACT_ROOT")
    destination = artifact_root / str(experiment["experiment_id"]) / artifact_id
    if destination.exists():
        raise FileExistsError("The SmolVLA artifact is create-only.")
    policy, preprocessor, postprocessor, dataset, _, _ = evaluator._load_policy_and_dataset(
        plan, experiment, dataset_root, step
    )
    first_index = evaluator._validation_indices(
        dataset,
        [int(plan["validation"]["episodes"][0])],
        [int(plan["validation"]["frame_offsets"][0])],
    )[0][2]
    reference = _fixed_prediction(
        policy,
        preprocessor,
        postprocessor,
        dataset,
        first_index,
        mixed_precision=str(experiment["resources"]["mixed_precision"]),
    )
    del policy, preprocessor, postprocessor
    gc.collect()
    if torch.xpu.is_available():
        torch.xpu.empty_cache()

    destination.mkdir(parents=True, exist_ok=False)
    pretrained_destination = destination / "pretrained_model"
    _copy_policy(source_dir, pretrained_destination)
    normalization = formal_runner._load_json(normalization_path)
    artifact_normalization = {
        "schema_version": 1,
        "source_split": "train",
        "train_rows": normalization["train_rows"],
        "effective_stats": normalization["effective_stats"],
        "visual_features": normalization["visual_features"],
        "visual_statistics": normalization["visual_statistics"],
        "visual_statistics_policy": normalization["visual_statistics_policy"],
        "hidden_test_loaded": False,
    }
    create_json(destination / "normalization.json", artifact_normalization)
    contract = load_action_contract(contract_path)
    create_json(destination / "action_contract.json", asdict(contract))
    info = formal_runner._load_json(dataset_root / "meta/info.json")
    artifact_config = {
        "schema_version": 1,
        "artifact_type": "smolvla_policy",
        "experiment_id": experiment["experiment_id"],
        "artifact_id": args.artifact_id,
        "base_model": experiment["model"]["identifier"],
        "base_model_revision": experiment["model"]["revision"],
        "upstream_repository": experiment["upstream"]["repository"],
        "upstream_revision": experiment["upstream"]["revision"],
        "dataset_id": experiment["dataset"]["identifier"],
        "dataset_revision": experiment["dataset"]["revision"],
        "dataset_fps": info["fps"],
        "dataset_features": {
            name: info["features"][name]
            for name in ("observation.images.top", "observation.state", "action")
        },
        "rename_map": experiment["dataset"]["rename_map"],
        "instruction": "Insert the peg into the socket.",
        "selected_checkpoint_step": step,
        "adaptation": experiment["model"]["adaptation"],
        "policy": experiment["model"]["policy"],
        "optimizer_contract": formal_runner._optimizer_contract(plan["training"]),
        "mixed_precision": experiment["resources"]["mixed_precision"],
        "inference_noise": "zeros",
        "formal_plan_sha256": file_sha256(plan_path),
        "selection_report_sha256": file_sha256(selection_path),
        "normalization_report_sha256": file_sha256(normalization_path),
        "dataset_view_manifest_sha256": file_sha256(view_manifest_path),
        "action_contract_sha256": contract_sha256,
        "hidden_test_loaded": False,
        "research_only": True,
        "physical_robot_validated": False,
    }
    create_json(destination / "config.json", artifact_config)

    reloaded_policy, reloaded_preprocessor, reloaded_postprocessor = _load_artifact_policy(
        pretrained_destination, dataset
    )
    reloaded = _fixed_prediction(
        reloaded_policy,
        reloaded_preprocessor,
        reloaded_postprocessor,
        dataset,
        first_index,
        mixed_precision=str(experiment["resources"]["mixed_precision"]),
    )
    maximum_difference = float((reference - reloaded).abs().max().item())
    reload_verified = bool(torch.equal(reference, reloaded))
    if not reload_verified:
        raise RuntimeError("Exported SmolVLA behavior differs from the selected checkpoint.")
    model_card = f"""# Rosetta Reality {args.artifact_id}

Experimental research-only SmolVLA development policy for simulated ALOHA insertion.
It has not been validated on a physical robot and must not be represented as an
autonomous real-robot controller.

- Base model: `{experiment['model']['identifier']}` at `{experiment['model']['revision']}`
- Selected training step: `{step}`
- Dataset: `{experiment['dataset']['identifier']}` at `{experiment['dataset']['revision']}`
- Inputs: top-camera image, language instruction, and 14-dimensional robot state
- Outputs: 50 absolute 14-dimensional joint-position target actions at 50 Hz
- Execution: receding-horizon first action, then observe again
- Selection validation action MAE: `{selected['metrics']['action_mae']:.8f}`
- Export reload: exact deterministic action equality verified

Limitations: development-scale data and simulation only; no physical-robot safety
validation, no cross-embodiment claim, and task success must be established separately.
"""
    with (destination / "MODEL_CARD.md").open("x", encoding="utf-8", newline="\n") as file:
        file.write(model_card)

    files = {
        path.relative_to(destination).as_posix(): file_sha256(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "schema_version": 1,
        "status": "verified",
        "artifact_type": "smolvla_policy",
        "artifact_id": args.artifact_id,
        "experiment_id": experiment["experiment_id"],
        "selected_checkpoint_step": step,
        "selected_checkpoint_model_sha256": selected["model_safetensors_sha256"],
        "selection_report_sha256": file_sha256(selection_path),
        "code_identity": workspace_code_identity(REPOSITORY_ROOT),
        "reload": {
            "verified": True,
            "exact_tensor_equality": True,
            "maximum_absolute_difference": maximum_difference,
            "reference_action_sha256": _tensor_sha256(reference),
            "reloaded_action_sha256": _tensor_sha256(reloaded),
            "noise": "zeros",
        },
        "files": files,
        "hidden_test_loaded": False,
    }
    create_json(destination / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Artifact: {args.artifact_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
