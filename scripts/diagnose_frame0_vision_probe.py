"""Non-gating paired correct/mismatched-image probe with fixed noise.

Uses a checksum-verified cache and artifact; train is the default split.
Reports alignment and sensitivity separately, never inferring information
absence from a failed action regression. Optional reports are create-only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import smolvla_sim_gate as simulator  # noqa: E402
import smolvla_vcdropout_protocol as protocol  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: E402

# Registration side effect: the artifact's saved processor pipeline uses
# Rosetta-registered steps (action-contract projection); importing the
# processor module registers them before the pipeline is built.
from rosetta_reality.vla.processor import ensure_smolvla_action_boundary  # noqa: E402,F401
from rosetta_reality.vla.vision_diagnostics import (  # noqa: E402
    load_frame_zero_context,
    paired_alignment,
    verify_deploy_artifact,
)

DATASET_CONFIG = REPOSITORY_ROOT / "configs/data/aloha_sim_insertion_m2.yaml"
HIDDEN = set(protocol.HIDDEN_TEST_EPISODES)
INSTRUCTION = "Insert the peg into the socket."


def _load_artifact(artifact_id: str, device: torch.device) -> tuple[Any, tuple[Any, Any]]:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", artifact_id):
        raise ValueError("Artifact id must be a single path-safe identifier.")
    artifact_dir = (
        Path(os.environ["ROSETTA_ARTIFACT_ROOT"]).resolve() / protocol.EXPERIMENT_ID / artifact_id
    )
    if not artifact_dir.is_dir():
        raise FileNotFoundError(f"artifact directory missing: {artifact_dir.name}")
    manifest_sha = verify_deploy_artifact(artifact_dir, artifact_id, protocol.EXPERIMENT_ID)
    config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    normalization = json.loads((artifact_dir / "normalization.json").read_text(encoding="utf-8"))
    pretrained = artifact_dir / "pretrained_model"
    policy_cfg = SmolVLAConfig.from_pretrained(pretrained, local_files_only=True)
    policy_cfg.device = device.type
    policy_cfg.pretrained_path = pretrained
    policy_cfg.pretrained_revision = None
    policy_cfg.load_vlm_weights = False

    artifact_metadata = simulator._ArtifactMetadata(config, normalization)
    policy = make_policy(
        cfg=policy_cfg,
        ds_meta=artifact_metadata,
        rename_map=config["rename_map"],
    )
    policy.eval()
    policy._rosetta_diagnostic_manifest_sha256 = manifest_sha
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=pretrained,
        pretrained_revision=None,
        dataset_stats=artifact_metadata.stats,
        preprocessor_overrides={
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "features": {
                    **policy.config.input_features,
                    **policy.config.output_features,
                },
                "norm_map": policy.config.normalization_mapping,
                "stats": artifact_metadata.stats,
            },
            "rename_observations_processor": {"rename_map": config["rename_map"]},
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
                "stats": artifact_metadata.stats,
            }
        },
    )
    return policy, (preprocessor, postprocessor)


def _expert_first_actions() -> tuple[np.ndarray, list[int]]:
    context = load_frame_zero_context(REPOSITORY_ROOT, "train")
    return context["actions"], context["episodes"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-id", default="m2-smolvla450m-vcdropout-cuda-b64-001-step0237-deploy-001"
    )
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--output", type=Path, help="Optional create-only JSON report.")
    arguments = parser.parse_args()
    if arguments.output is not None and arguments.output.exists():
        raise FileExistsError("Diagnostic output already exists.")
    context = load_frame_zero_context(REPOSITORY_ROOT, arguments.split)
    expert, episodes = context["actions"], context["episodes"]
    # A cyclic image permutation is a valid fixed-state intervention only when
    # all states are exactly equal. Fail before loading weights otherwise.
    if not np.array_equal(
        context["states"], np.broadcast_to(context["states"][0], context["states"].shape)
    ):
        raise ValueError("Frame-zero state degeneracy is required for this paired probe.")
    cfg = context["config"]
    dataset = LeRobotDataset(
        cfg.repo_id,
        root=context["root"],
        episodes=episodes,
        revision=cfg.revision,
        download_videos=False,
        return_uint8=True,
    )
    device = torch.device(os.environ.get("ROSETTA_TORCH_DEVICE", "cpu"))
    policy, (preprocessor, postprocessor) = _load_artifact(arguments.artifact_id, device)
    starts = dict(
        zip(
            dataset.meta.episodes["episode_index"],
            dataset.meta.episodes["dataset_from_index"],
            strict=True,
        )
    )
    camera_key = cfg.cameras["top"]
    # Use the same noise for every image, then pair outputs with a fixed
    # cyclic image mismatch. State/instruction/noise are unchanged by design.
    noise_seeds = (None, 20260905, 20260906, 20260907)
    results = []
    for noise_seed in noise_seeds:
        outputs = []
        for episode in episodes:
            sample = dataset[dataset.absolute_to_relative_idx[int(starts[episode])]]
            if (
                int(sample[cfg.fields.episode_index]) != episode
                or int(sample[cfg.fields.frame_index]) != 0
            ):
                raise ValueError("Image sample identity differs from the requested label.")
            image, state = sample[camera_key], sample[cfg.fields.state]
            if image.ndim == 4:
                image = image[0]
            if state.ndim == 2:
                state = state[0]
            if not np.array_equal(state.cpu().numpy(), context["states"][episodes.index(episode)]):
                raise ValueError("Image and label readers returned different states.")
            if image.dtype == torch.uint8:
                image = image.float() / 255.0
            batch = preprocessor(
                {camera_key: image, "observation.state": state, "task": INSTRUCTION}
            )
            shape = (1, policy.config.chunk_size, policy.config.max_action_dim)
            if noise_seed is None:
                noise = torch.zeros(shape)
            else:
                noise = torch.randn(
                    shape, generator=torch.Generator(device="cpu").manual_seed(noise_seed)
                )
            noise = noise.to(device=device, dtype=batch["observation.state"].dtype)
            policy.reset()
            with (
                torch.inference_mode(),
                torch.autocast(device_type=device.type, dtype=torch.bfloat16),
            ):
                action = postprocessor(policy.predict_action_chunk(batch, noise=noise))
            outputs.append(action[0, 0].detach().float().cpu().numpy().astype(np.float64))
        predictions = np.asarray(outputs)
        result = paired_alignment(predictions, np.roll(predictions, 1, axis=0), expert)
        result.update({"noise_seed": noise_seed, "predictions": predictions.tolist()})
        results.append(result)
    report = {
        "schema_version": 2,
        "stage": "frame_zero_paired_alignment",
        "gating": False,
        "artifact_id": arguments.artifact_id,
        "dataset_revision": cfg.revision,
        "artifact_manifest_sha256": policy._rosetta_diagnostic_manifest_sha256,
        "split": arguments.split,
        "episodes": episodes,
        "mismatched_image_episodes": episodes[-1:] + episodes[:-1],
        "hidden_rows_materialized": False,
        "results": results,
        "action_space": "postprocessed_standard_aloha_joint_position",
        "note": (
            "Per-dimension MAE retains arm-radian/gripper-unit semantics; "
            "aggregate MAE is only the historical proxy."
        ),
    }
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if arguments.output is not None:
        with arguments.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
