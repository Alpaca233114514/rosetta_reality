"""Per-module gradient norms of a selected Zen deploy artifact under modality conditions.

Create-only Zen-native diagnostic (architecture map section 9, finding T4):
loads a locally transferred Zen deploy artifact through the registered frozen
boundary, rebuilds the registered adaptation pattern (frozen VLM, trainable
expert / state projector / action projections), and measures per-module
gradient L2 norms of the teacher-forced flow-matching loss on the fixed
validation-episode samples under four conditions: normal, image_shuffle,
state_shuffle and image_zero. Non-gating: no optimizer, no weight updates, no
hidden-test access. If the trainable modules' gradients move massively under
state_shuffle but barely under image_shuffle / image_zero, the learning signal
itself is state-dominated at the registered reset distribution.
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
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import evaluate_smolvla_validation as evaluator  # noqa: E402
import smolvla_sim_gate as simulator  # noqa: E402
import smolvla_zen_protocol as protocol  # noqa: E402

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

EXPERIMENT_ID = protocol.EXPERIMENT_ID
DATASET_CONFIG = REPOSITORY_ROOT / "configs/data/aloha_sim_insertion_m2.yaml"
CONTRACT_PATH = REPOSITORY_ROOT / "configs/sim/aloha_insertion_smolvla.yaml"
CONTRACT_SHA = "fc71a0438f0e3af7258e5b52d82fa22fc53c12b47901606cbee715524392ac62"
ZEN_FIRSTACTION_PLAN = (
    REPOSITORY_ROOT / "configs/vla/smolvla_450m_aloha_insertion_zen_cuda_b64_firstaction_001.yaml"
)
ARTIFACT_MANIFEST_SHA = {
    "m2-smolvla450m-zen-cuda-b64-uniform-001-step0316-deploy-001": (
        "ecc73b9e26f43b6dc85981e61e2ed3aaa651b208b1cd72ad26f93877599694ec"
    ),
    "m2-smolvla450m-zen-cuda-b64-firstaction-001-step0316-deploy-001": (
        "d6b2a7ff922605daf04670dd8e57a582fc4f5f5dcb1efd78ff37aec3357d0653"
    ),
}
CONDITIONS = ("normal", "image_shuffle", "state_shuffle", "image_zero")
SHUFFLE_SEED = 20260812
FLOW_TIME = 0.5
REGISTERED_ADAPTATION = {
    "freeze_vision_encoder": True,
    "train_expert_only": True,
    "train_state_proj": True,
}
# First-match-wins parameter grouping over policy.named_parameters(). A name
# matches a group when ANY alternative matches, and an alternative matches
# when ALL of its markers appear in the name (substring, position
# independent): concrete runtime names carry a `model.` prefix and nested HF
# module names, so fixed prefixes are not stable across the pinned tree.
GRADIENT_GROUPS: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...] = (
    ("vision_encoder", (("vlm_with_expert.vlm.", "vision_model."),)),
    ("language_model", (("vlm_with_expert.vlm.",),)),
    ("action_expert", (("vlm_with_expert.lm_expert.",),)),
    ("state_projector", (("state_proj.",),)),
    (
        "action_io_projections",
        (
            ("action_in_proj.",),
            ("action_out_proj.",),
            ("action_time_mlp_in.",),
            ("action_time_mlp_out.",),
        ),
    ),
)


def _group_of(name: str) -> str:
    for group, alternatives in GRADIENT_GROUPS:
        if any(
            all(marker in name for marker in alternative)
            for alternative in alternatives
        ):
            return group
    raise ValueError(f"Unmatched policy parameter fails the group contract: {name}")


def _validate_artifact(artifact_dir: Path, expected_manifest_sha: str) -> dict[str, Any]:
    manifest_path = artifact_dir / "manifest.json"
    if file_sha256(manifest_path) != expected_manifest_sha:
        raise ValueError("Artifact manifest checksum differs from the registration.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    normalization = json.loads(
        (artifact_dir / "normalization.json").read_text(encoding="utf-8")
    )
    if (
        manifest.get("status") != "verified"
        or manifest.get("artifact_id") != artifact_dir.name
        or manifest.get("experiment_id") != EXPERIMENT_ID
        or manifest.get("reload", {}).get("exact_tensor_equality") is not True
        or manifest.get("hidden_test_loaded") is not False
        or config.get("hidden_test_loaded") is not False
        or normalization.get("source_split") != "train"
        or normalization.get("hidden_test_loaded") is not False
    ):
        raise ValueError("Artifact identity is invalid for the gradient diagnostic.")
    for relative, expected in manifest["files"].items():
        path = artifact_dir / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"Artifact checksum changed: {relative}.")
    return manifest


def _clone_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    return value


def _perturb_sample(
    samples: list[dict[str, Any]],
    destination: int,
    source: int,
    condition: str,
    camera_keys: list[str],
) -> dict[str, Any]:
    sample = _clone_tree(samples[destination])
    if condition == "normal":
        return sample
    if condition == "image_shuffle":
        for key in camera_keys:
            if key not in samples[source]:
                raise ValueError(f"Image-shuffle source is missing {key}.")
            sample[key] = _clone_tree(samples[source][key])
        return sample
    if condition == "state_shuffle":
        state = samples[source].get("observation.state")
        if not isinstance(state, torch.Tensor):
            raise ValueError("State-shuffle source has no robot state tensor.")
        sample["observation.state"] = state.clone()
        return sample
    if condition == "image_zero":
        for key in camera_keys:
            image = sample.get(key)
            if not isinstance(image, torch.Tensor):
                raise ValueError(f"Zero-image sample is missing {key}.")
            sample[key] = torch.zeros_like(image)
        return sample
    raise ValueError(f"Unsupported modality condition: {condition}.")


def _verify_requires_grad(policy: Any) -> dict[str, Any]:
    """Fail closed unless the rebuild reproduces the registered freeze pattern."""

    frozen_groups = {"vision_encoder", "language_model"}
    per_group: dict[str, dict[str, int]] = {}
    for group, _ in GRADIENT_GROUPS:
        per_group[group] = {"trainable": 0, "frozen": 0}
    unmatched: list[str] = []
    for name, parameter in policy.named_parameters():
        try:
            group = _group_of(name)
        except ValueError:
            unmatched.append(name)
            continue
        trainable = bool(parameter.requires_grad)
        if group in frozen_groups and trainable:
            raise ValueError(f"Frozen-module parameter is trainable: {name}.")
        if group == "action_expert" and trainable and "lm_head" in name:
            raise ValueError(f"Expert lm_head must stay frozen: {name}.")
        per_group[group]["trainable" if trainable else "frozen"] += 1
    if unmatched:
        raise ValueError(
            "Unmatched policy parameters fail the group contract: "
            f"{sorted(unmatched)}"
        )
    for group, counts in per_group.items():
        if counts["trainable"] + counts["frozen"] == 0:
            raise ValueError(f"Gradient group matched no parameters: {group}.")
        if group in frozen_groups and counts["trainable"] != 0:
            raise ValueError(f"Freeze verification failed for {group}.")
    return per_group


def _group_gradient_norms(policy: Any) -> dict[str, float]:
    squared: dict[str, float] = {group: 0.0 for group, _ in GRADIENT_GROUPS}
    seen: dict[str, int] = {group: 0 for group, _ in GRADIENT_GROUPS}
    for name, parameter in policy.named_parameters():
        group = _group_of(name)
        gradient = parameter.grad
        if gradient is None:
            continue
        value = gradient.detach().to(torch.float64)
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"Non-finite gradient in {name}.")
        squared[group] += float(value.square().sum())
        seen[group] += 1
    return {
        group: (math.sqrt(squared[group]) if seen[group] else 0.0)
        for group, _ in GRADIENT_GROUPS
    }


def _main(args: argparse.Namespace) -> int:
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("HF_DATASETS_OFFLINE") != "1":
        raise RuntimeError("Gradient diagnostics require networking disabled.")
    device = torch.device(str(os.environ["ROSETTA_TORCH_DEVICE"]))
    if device.type != "xpu" or not torch.xpu.is_available():
        raise RuntimeError("The Zen gradient diagnostic requires the registered XPU runtime.")

    artifact_dir = (
        simulator._absolute_root("ROSETTA_ARTIFACT_ROOT")
        / EXPERIMENT_ID
        / args.artifact_id
    )
    expected_manifest_sha = ARTIFACT_MANIFEST_SHA.get(args.artifact_id)
    if expected_manifest_sha is None:
        raise ValueError("Artifact is not one of the two registered Zen deploy artifacts.")
    manifest = _validate_artifact(artifact_dir, expected_manifest_sha)
    config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    normalization = json.loads(
        (artifact_dir / "normalization.json").read_text(encoding="utf-8")
    )

    zen_plan = yaml.safe_load(ZEN_FIRSTACTION_PLAN.read_text(encoding="utf-8"))
    validation = zen_plan["validation"]
    episodes = [int(value) for value in validation["episodes"]]
    offsets = [int(value) for value in validation["frame_offsets"]]
    hidden = {31, 6, 1, 24, 5}
    train_episodes = set(int(value) for value in zen_plan["training"]["episodes"])
    if (
        any(episode in train_episodes for episode in episodes)
        or any(episode in hidden for episode in episodes)
        or validation.get("hidden_test_loaded") is not False
    ):
        raise ValueError("Gradient samples must be the registered validation episodes only.")

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
    policy_cfg.freeze_vision_encoder = REGISTERED_ADAPTATION["freeze_vision_encoder"]
    policy_cfg.train_expert_only = REGISTERED_ADAPTATION["train_expert_only"]
    policy_cfg.train_state_proj = REGISTERED_ADAPTATION["train_state_proj"]

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
        raise ValueError("Zen artifact has no explicit action-space identity.")
    action_space = SmolVLAActionSpace(**raw_action_space)
    contract_path = REPOSITORY_ROOT / "configs/sim/aloha_insertion_smolvla.yaml"
    if file_sha256(contract_path) != CONTRACT_SHA:
        raise ValueError("Registered Action Contract checksum changed.")
    ensure_smolvla_action_boundary(
        preprocessor,
        postprocessor,
        load_action_contract(contract_path),
        action_space,
        action_contract_sha256=str(config["action_contract_sha256"]),
        upstream_revision=str(config["upstream_revision"]),
    )
    freeze_verification = _verify_requires_grad(policy)

    indices = evaluator._validation_indices(dataset, episodes, offsets)
    if len(indices) != int(validation["total_samples"]):
        raise ValueError("Sample count differs from the registered validation protocol.")
    samples = [_clone_tree(dataset[relative]) for _, _, relative in indices]
    episode_tensor = torch.tensor(
        [episode for episode, _, _ in indices], dtype=torch.int64
    )
    frame_tensor = torch.tensor([offset for _, offset, _ in indices], dtype=torch.int64)
    shuffle = cross_episode_shuffle_indices(
        episode_tensor, frame_indices=frame_tensor, seed=args.shuffle_seed
    )
    camera_keys = [str(value) for value in dataset.meta.camera_keys]
    if not camera_keys:
        raise ValueError("Gradient diagnostic dataset has no camera features.")

    autocast_dtype = evaluator._autocast_dtype("bf16")
    policy.eval()
    condition_reports: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for condition in CONDITIONS:
        per_sample_norms: dict[str, list[float]] = {
            group: [] for group, _ in GRADIENT_GROUPS
        }
        losses: list[float] = []
        for destination in range(len(samples)):
            source = int(shuffle[destination])
            sample = _perturb_sample(
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
                raise ValueError("Gradient diagnostic batch has no action tensor.")
            noise = torch.zeros(
                (1, policy.config.chunk_size, policy.config.max_action_dim),
                device=device,
                dtype=action.dtype,
            )
            flow_time = torch.full((1,), args.flow_time, device=device, dtype=action.dtype)
            policy.reset()
            policy.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ):
                loss, _ = policy(_clone_tree(batch), noise=noise, time=flow_time)
            loss.backward()
            torch.xpu.synchronize()
            norms = _group_gradient_norms(policy)
            for group in per_sample_norms:
                per_sample_norms[group].append(norms[group])
            losses.append(float(loss.detach().cpu()))
            if not math.isfinite(losses[-1]):
                raise FloatingPointError("Gradient diagnostic produced a non-finite loss.")
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
    for condition in CONDITIONS[1:]:
        perturbed = condition_reports[condition]["per_group_mean_gradient_l2"]
        condition_reports[condition]["gradient_ratio_vs_normal"] = {
            group: (
                perturbed[group] / normal_groups[group]
                if normal_groups[group] > 0.0
                else None
            )
            for group in normal_groups
        }

    report = {
        "schema_version": 1,
        "status": "complete",
        "stage": "smolvla_zen_module_gradient_diagnostic",
        "gating_evidence": False,
        "experiment_id": EXPERIMENT_ID,
        "artifact_id": args.artifact_id,
        "artifact_manifest_sha256": file_sha256(artifact_dir / "manifest.json"),
        "registered_adaptation": REGISTERED_ADAPTATION,
        "freeze_verification": freeze_verification,
        "gradient_groups": {
            group: [list(alternative) for alternative in alternatives]
            for group, alternatives in GRADIENT_GROUPS
        },
        "protocol": {
            "samples": "registered validation episodes / frame offsets",
            "episodes": episodes,
            "frame_offsets": offsets,
            "sample_count": len(samples),
            "conditions": list(CONDITIONS),
            "shuffle_seed": args.shuffle_seed,
            "shuffle_policy": "cross_episode_same_frame_offset_derangement",
            "noise": "zeros",
            "flow_time": args.flow_time,
            "mixed_precision": "bf16",
            "policy_mode": "eval",
            "optimizer_created": False,
            "weight_updates": 0,
            "teacher_forced_observations": True,
        },
        "dataset_revision": dataset_manifest.resolved_revision,
        "dataset_manifest_sha256": file_sha256(dataset_root / "manifest.json"),
        "action_contract_sha256": CONTRACT_SHA,
        "conditions": condition_reports,
        "elapsed_seconds": time.perf_counter() - started,
        "diagnostic_script_sha256": file_sha256(Path(__file__)),
        "code_identity": workspace_code_identity(REPOSITORY_ROOT),
        "hidden_test_loaded": False,
        "network_disabled": True,
    }
    json.dumps(report, allow_nan=False)
    identity = {
        "artifact": report["artifact_manifest_sha256"],
        "shuffle_seed": args.shuffle_seed,
        "flow_time": args.flow_time,
        "script": report["diagnostic_script_sha256"],
    }
    short = args.artifact_id.split("-")[3]
    destination = (
        simulator._absolute_root("ROSETTA_RUN_ROOT")
        / EXPERIMENT_ID
        / "diagnostics"
        / f"zen-module-gradients-{short}-{stable_hash(identity)[:16]}.json"
    )
    create_json(destination, report)
    print(json.dumps(condition_reports, indent=2, sort_keys=True))
    print(f"Report: {destination.name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-id",
        default="m2-smolvla450m-zen-cuda-b64-firstaction-001-step0316-deploy-001",
    )
    parser.add_argument("--shuffle-seed", type=int, default=SHUFFLE_SEED)
    parser.add_argument("--flow-time", type=float, default=FLOW_TIME)
    args = parser.parse_args()
    if args.shuffle_seed < 0 or not 0.0 < args.flow_time < 1.0:
        raise ValueError("Shuffle seed must be non-negative and flow time in (0, 1).")
    return _main(args)


if __name__ == "__main__":
    raise SystemExit(main())
