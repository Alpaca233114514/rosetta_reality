"""Executable offset-250 gradient gate for the state-dropout candidate.

This is the preregistered metric gate frozen in section 4 of
``reports/training/m2-smolvla-visual-conditioning-state-dropout-preregistration-2026-08-28.md``.
It consumes the exported candidate deploy artifact (verified manifest, exact
independent reload), re-runs the module-gradient measurement protocol exactly
as the completed Zen diagnostic did at validation frame offset 250 — the same
five validation episodes, cross-episode same-frame derangement, zero noise,
flow time 0.5, bf16 autocast, teacher-forced — and evaluates the six frozen
criteria:

1. freeze/trainable grouping exact and every trainable-group normal gradient
   finite and nonzero;
2. sample state/image diversity guards pass at the nonzero offset;
3. normal mean flow loss at most 0.22005 (input-destruction guard);
4. state sensitivity at most 0.70;
5. image sensitivity at least 0.10;
6. state-dominance score at most 0.453 (>= 50% below the Zen-uniform control).

A pass does not accept M2 and does not authorize training: it only permits a
separately registered Gate 3/4 comparison. Thresholds may not be relaxed after
seeing the candidate; a failed criterion stops the axis and is recorded
honestly. Teacher-forced only, no optimizer, no weight updates, no hidden-test
access, networking disabled.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import diagnose_smolvla_zen_module_gradients as gradient_diagnostic  # noqa: E402
import evaluate_smolvla_validation as evaluator  # noqa: E402
import smolvla_sim_gate as simulator  # noqa: E402
import smolvla_vcdropout_protocol as protocol  # noqa: E402
from lerobot.datasets.factory import resolve_delta_timestamps  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: E402
from torch.utils.data import default_collate  # noqa: E402

from rosetta_reality.data import resolve_prepared_cache  # noqa: E402
from rosetta_reality.data.config import load_dataset_config  # noqa: E402
from rosetta_reality.eval.diagnostics import cross_episode_shuffle_indices  # noqa: E402
from rosetta_reality.experiment import (  # noqa: E402
    file_sha256,
    stable_hash,
    workspace_code_identity,
)
from rosetta_reality.features import create_json  # noqa: E402
from rosetta_reality.sim import load_action_contract  # noqa: E402
from rosetta_reality.vla.action_space import SmolVLAActionSpace  # noqa: E402
from rosetta_reality.vla.processor import ensure_smolvla_action_boundary  # noqa: E402

DATASET_CONFIG = REPOSITORY_ROOT / "configs/data/aloha_sim_insertion_m2.yaml"


def _validate_candidate_artifact(
    artifact_dir: Path, plan_path: Path
) -> dict[str, Any]:
    """Bind the artifact to the validated candidate plan and verify every file."""

    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Candidate artifact has no manifest.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    normalization = json.loads(
        (artifact_dir / "normalization.json").read_text(encoding="utf-8")
    )
    if (
        manifest.get("status") != "verified"
        or manifest.get("artifact_id") != artifact_dir.name
        or manifest.get("experiment_id") != protocol.EXPERIMENT_ID
        or manifest.get("plan_id") != protocol.VCD_PLAN_ID
        or manifest.get("plan_sha256") != file_sha256(plan_path)
        or manifest.get("reload", {}).get("exact_tensor_equality") is not True
        or manifest.get("selected_checkpoint_step") not in protocol.CHECKPOINT_STEPS
        or manifest.get("hidden_test_loaded") is not False
        or config.get("hidden_test_loaded") is not False
        or config.get("training_only_treatment", {}).get("state_dropout_training_only")
        is not True
        or normalization.get("source_split") != "train"
        or normalization.get("hidden_test_loaded") is not False
    ):
        raise ValueError("Artifact identity is invalid for the gradient gate.")
    for relative, expected in manifest["files"].items():
        path = artifact_dir / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"Artifact checksum changed: {relative}.")
    return manifest


def _main(args: argparse.Namespace) -> int:
    if (
        os.environ.get("HF_HUB_OFFLINE") != "1"
        or os.environ.get("HF_DATASETS_OFFLINE") != "1"
    ):
        raise RuntimeError("The gradient gate requires networking disabled.")
    device = torch.device(str(os.environ["ROSETTA_TORCH_DEVICE"]))
    if device.type != "xpu" or not torch.xpu.is_available():
        raise RuntimeError(
            "The gradient gate requires the registered XPU runtime so the "
            "measurement stays comparable with the frozen Zen-uniform baseline."
        )

    plan_path = args.plan.resolve()
    plan, plan_id = protocol.resolve_plan(plan_path)
    artifact_dir = (
        simulator._absolute_root("ROSETTA_ARTIFACT_ROOT")
        / protocol.EXPERIMENT_ID
        / args.artifact_id
    )
    manifest = _validate_candidate_artifact(artifact_dir, plan_path)
    config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    normalization = json.loads(
        (artifact_dir / "normalization.json").read_text(encoding="utf-8")
    )

    episodes = list(protocol.GATE_EPISODES)
    hidden = set(protocol.HIDDEN_TEST_EPISODES)
    train_episodes = set(protocol.TRAIN_EPISODES)
    if (
        any(episode in train_episodes for episode in episodes)
        or any(episode in hidden for episode in episodes)
    ):
        raise ValueError("Gate samples must be the registered validation episodes only.")

    dataset_config = load_dataset_config(DATASET_CONFIG)
    dataset_root, dataset_manifest = resolve_prepared_cache(
        dataset_config, REPOSITORY_ROOT, validate_checksums=True
    )
    pretrained = artifact_dir / "pretrained_model"
    policy_cfg = SmolVLAConfig.from_pretrained(pretrained, local_files_only=True)
    policy_cfg.device = device.type
    policy_cfg.pretrained_path = pretrained
    policy_cfg.pretrained_revision = None
    policy_cfg.load_vlm_weights = False
    policy_cfg.freeze_vision_encoder = gradient_diagnostic.REGISTERED_ADAPTATION[
        "freeze_vision_encoder"
    ]
    policy_cfg.train_expert_only = gradient_diagnostic.REGISTERED_ADAPTATION[
        "train_expert_only"
    ]
    policy_cfg.train_state_proj = gradient_diagnostic.REGISTERED_ADAPTATION[
        "train_state_proj"
    ]

    metadata = LeRobotDatasetMetadata(
        dataset_config.repo_id,
        root=dataset_root,
        revision=dataset_config.revision,
    )
    delta_timestamps = resolve_delta_timestamps(policy_cfg, metadata)
    dataset = LeRobotDataset(
        dataset_config.repo_id,
        root=dataset_root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        revision=dataset_config.revision,
        download_videos=False,
        return_uint8=True,
    )
    artifact_metadata = simulator._ArtifactMetadata(config, normalization)
    policy = make_policy(
        cfg=policy_cfg,
        ds_meta=artifact_metadata,
        rename_map=config["rename_map"],
    )
    artifact_stats = artifact_metadata.stats
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=pretrained,
        pretrained_revision=None,
        dataset_stats=artifact_stats,
        preprocessor_overrides={
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "features": {
                    **policy.config.input_features,
                    **policy.config.output_features,
                },
                "norm_map": policy.config.normalization_mapping,
                "stats": artifact_stats,
            },
            "rename_observations_processor": {
                "rename_map": config["rename_map"]
            },
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
                "stats": artifact_stats,
            }
        },
    )
    raw_action_space = config.get("action_space")
    if not isinstance(raw_action_space, dict):
        raise ValueError("Candidate artifact has no explicit action-space identity.")
    action_space = SmolVLAActionSpace(**raw_action_space)
    contract_path = REPOSITORY_ROOT / protocol.ACTION_CONTRACT_RELATIVE
    if file_sha256(contract_path) != protocol.ACTION_CONTRACT_SHA256:
        raise ValueError("Registered Action Contract checksum changed.")
    ensure_smolvla_action_boundary(
        preprocessor,
        postprocessor,
        load_action_contract(contract_path),
        action_space,
        action_contract_sha256=str(config["action_contract_sha256"]),
        upstream_revision=str(config["upstream_revision"]),
    )
    freeze_verification = gradient_diagnostic._verify_requires_grad(policy)

    indices = evaluator._validation_indices(dataset, episodes, [protocol.GATE_FRAME_OFFSET])
    if len(indices) != len(episodes):
        raise ValueError("Sample count differs from the registered gate protocol.")
    samples = [
        gradient_diagnostic._clone_tree(dataset[relative])
        for _, _, relative in indices
    ]
    camera_keys = [str(value) for value in dataset.meta.camera_keys]
    if not camera_keys:
        raise ValueError("Gate dataset has no camera features.")

    def pairwise_max(stack: torch.Tensor) -> float:
        difference = 0.0
        for i in range(stack.shape[0]):
            for j in range(i + 1, stack.shape[0]):
                difference = max(
                    difference, float((stack[i] - stack[j]).abs().max())
                )
        return difference

    state_stack = torch.stack(
        [torch.as_tensor(sample["observation.state"]).reshape(-1) for sample in samples]
    ).to(torch.float64)
    pairwise_state_difference = pairwise_max(state_stack)
    image_stack = torch.stack(
        [
            torch.as_tensor(sample[key], dtype=torch.float64).reshape(-1)
            for sample in samples
            for key in camera_keys
        ]
    )
    pairwise_image_difference = pairwise_max(image_stack)
    if protocol.GATE_FRAME_OFFSET > 0 and pairwise_state_difference <= 0.0:
        raise ValueError(
            "Degenerate state_shuffle: selected samples share identical states; "
            "the gate cannot probe state sensitivity."
        )
    if pairwise_image_difference <= 0.0:
        raise ValueError("Degenerate image conditions: samples share identical images.")
    episode_tensor = torch.tensor(episodes, dtype=torch.int64)
    frame_tensor = torch.full(
        (len(episodes),), protocol.GATE_FRAME_OFFSET, dtype=torch.int64
    )
    shuffle = cross_episode_shuffle_indices(
        episode_tensor, frame_indices=frame_tensor, seed=protocol.GATE_SHUFFLE_SEED
    )

    autocast_dtype = evaluator._autocast_dtype("bf16")
    policy.eval()
    condition_reports: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for condition in gradient_diagnostic.CONDITIONS:
        per_sample_norms: dict[str, list[float]] = {
            group: [] for group, _ in gradient_diagnostic.GRADIENT_GROUPS
        }
        losses: list[float] = []
        for destination in range(len(samples)):
            source = int(shuffle[destination])
            sample = gradient_diagnostic._perturb_sample(
                samples, destination, source, condition, camera_keys
            )
            batch = default_collate([sample])
            for camera_key in camera_keys:
                value = batch.get(camera_key)
                if isinstance(value, torch.Tensor) and value.dtype == torch.uint8:
                    batch[camera_key] = (
                        value.to(torch.get_default_dtype())
                        / torch.iinfo(value.dtype).max
                    )
            batch = preprocessor(batch)
            action = batch.get("action")
            if not isinstance(action, torch.Tensor):
                raise ValueError("Gate batch has no action tensor.")
            noise = torch.zeros(
                (1, policy.config.chunk_size, policy.config.max_action_dim),
                device=device,
                dtype=action.dtype,
            )
            flow_time = torch.full(
                (1,), protocol.GATE_FLOW_TIME, device=device, dtype=action.dtype
            )
            policy.reset()
            policy.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ):
                loss, _ = policy(
                    gradient_diagnostic._clone_tree(batch), noise=noise, time=flow_time
                )
            loss.backward()
            torch.xpu.synchronize()
            norms = gradient_diagnostic._group_gradient_norms(policy)
            for group in per_sample_norms:
                per_sample_norms[group].append(norms[group])
            losses.append(float(loss.detach().cpu()))
            if not math.isfinite(losses[-1]):
                raise FloatingPointError("The gradient gate produced a non-finite loss.")
        condition_reports[condition] = {
            "mean_flow_loss": sum(losses) / len(losses),
            "per_group_mean_gradient_l2": {
                group: sum(values) / len(values)
                for group, values in per_sample_norms.items()
            },
            "per_group_max_gradient_l2": {
                group: max(values) for group, values in per_sample_norms.items()
            },
        }

    normal_groups = condition_reports["normal"]["per_group_mean_gradient_l2"]
    for condition in gradient_diagnostic.CONDITIONS[1:]:
        perturbed = condition_reports[condition]["per_group_mean_gradient_l2"]
        condition_reports[condition]["gradient_ratio_vs_normal"] = {
            group: (
                perturbed[group] / normal_groups[group]
                if normal_groups[group] > 0.0
                else None
            )
            for group in normal_groups
        }

    gate = protocol.evaluate_gate_criteria(
        normal_mean_flow_loss=condition_reports["normal"]["mean_flow_loss"],
        state_shuffle_gradient_ratios=condition_reports["state_shuffle"][
            "gradient_ratio_vs_normal"
        ],
        image_shuffle_gradient_ratios=condition_reports["image_shuffle"][
            "gradient_ratio_vs_normal"
        ],
        normal_trainable_gradient_norms=normal_groups,
        pairwise_sample_state_difference=pairwise_state_difference,
        pairwise_sample_image_difference=pairwise_image_difference,
    )
    baseline = protocol.control_baseline()

    report = {
        "schema_version": 1,
        "status": "passed" if gate["passed"] else "failed",
        "stage": "smolvla_vcdropout_offset250_gradient_gate",
        "gating": True,
        "training_authorized": False,
        "m2_acceptance": False,
        "experiment_id": protocol.EXPERIMENT_ID,
        "plan_id": plan_id,
        "plan_sha256": file_sha256(plan_path),
        "artifact_id": args.artifact_id,
        "artifact_manifest_sha256": file_sha256(artifact_dir / "manifest.json"),
        "selected_checkpoint_step": manifest["selected_checkpoint_step"],
        "registered_adaptation": gradient_diagnostic.REGISTERED_ADAPTATION,
        "freeze_verification": freeze_verification,
        "gate_protocol": {
            "frame_offset": protocol.GATE_FRAME_OFFSET,
            "episodes": episodes,
            "sample_count": len(samples),
            "shuffle_seed": protocol.GATE_SHUFFLE_SEED,
            "shuffle_policy": "cross_episode_same_frame_offset_derangement",
            "noise": "zeros",
            "flow_time": protocol.GATE_FLOW_TIME,
            "mixed_precision": "bf16",
            "policy_mode": "eval",
            "optimizer_created": False,
            "weight_updates": 0,
            "teacher_forced_observations": True,
            "offset_0_forbidden_for_state_sensitivity": True,
        },
        "pairwise_sample_state_max_difference": pairwise_state_difference,
        "pairwise_sample_image_max_difference": pairwise_image_difference,
        "gate_metrics": gate["metrics"],
        "gate_criteria": gate["criteria"],
        "gate_passed": gate["passed"],
        "control_baseline": baseline,
        "conditions": condition_reports,
        "dataset_revision": dataset_manifest.resolved_revision,
        "dataset_manifest_sha256": file_sha256(dataset_root / "manifest.json"),
        "action_contract_sha256": protocol.ACTION_CONTRACT_SHA256,
        "elapsed_seconds": time.perf_counter() - started,
        "gate_script_sha256": file_sha256(Path(__file__)),
        "protocol_module_sha256": file_sha256(
            REPOSITORY_ROOT / "scripts/smolvla_vcdropout_protocol.py"
        ),
        "code_identity": workspace_code_identity(REPOSITORY_ROOT),
        "hidden_test_loaded": False,
        "network_disabled": True,
    }
    json.dumps(report, allow_nan=False)
    identity = {
        "artifact": report["artifact_manifest_sha256"],
        "plan": report["plan_sha256"],
        "frame_offset": protocol.GATE_FRAME_OFFSET,
        "shuffle_seed": protocol.GATE_SHUFFLE_SEED,
        "script": report["gate_script_sha256"],
    }
    destination = (
        simulator._absolute_root("ROSETTA_RUN_ROOT")
        / protocol.EXPERIMENT_ID
        / "diagnostics"
        / f"vcdropout-gradient-gate-{stable_hash(identity)[:16]}.json"
    )
    create_json(destination, report)
    print(
        json.dumps(
            {
                "gate_passed": gate["passed"],
                "gate_metrics": gate["metrics"],
                "failed_criteria": [
                    criterion["name"]
                    for criterion in gate["criteria"]
                    if not criterion["passed"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"Report: {destination.name}")
    return 0 if gate["passed"] else 5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="the preregistered candidate plan this artifact belongs to",
    )
    parser.add_argument(
        "--artifact-id",
        required=True,
        help="exported candidate deploy artifact id (see export_smolvla_vcdropout)",
    )
    args = parser.parse_args()
    return _main(args)


if __name__ == "__main__":
    raise SystemExit(main())
