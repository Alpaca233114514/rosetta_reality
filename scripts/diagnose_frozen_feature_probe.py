"""Non-gating mean/spatial visual readouts with train-only alpha selection.

Failure of either readout cannot establish absence of visual information.
The probe samples tower/connector outputs, not the expert's contextualized KV.
Explicit grids must match the pinned model's patch/resampling layout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for root in (REPOSITORY_ROOT / "src", REPOSITORY_ROOT / "scripts"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from diagnose_frame0_vision_probe import _load_artifact  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from rosetta_reality.vla.vision_diagnostics import (  # noqa: E402
    fit_probe,
    load_frame_zero_context,
    spatial_features,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-id", default="m2-smolvla450m-vcdropout-cuda-b64-001-step0237-deploy-001"
    )
    parser.add_argument("--tower-grid", type=int, nargs=2, required=True, metavar=("H", "W"))
    parser.add_argument("--connector-grid", type=int, nargs=2, required=True, metavar=("H", "W"))
    parser.add_argument("--output", type=Path, help="Optional create-only JSON report.")
    arguments = parser.parse_args()
    if arguments.output is not None and arguments.output.exists():
        raise FileExistsError("Diagnostic output already exists.")
    context = load_frame_zero_context(REPOSITORY_ROOT, "non_hidden")
    cfg, episodes = context["config"], context["episodes"]
    dataset = LeRobotDataset(
        cfg.repo_id,
        root=context["root"],
        episodes=episodes,
        revision=cfg.revision,
        download_videos=False,
        return_uint8=True,
    )
    device = torch.device(os.environ.get("ROSETTA_TORCH_DEVICE", "cpu"))
    policy, (preprocessor, _) = _load_artifact(arguments.artifact_id, device)
    core = policy.model.vlm_with_expert.vlm.model
    modules = {"vision_tower": core.vision_model, "connector": core.connector}
    if any(p.requires_grad for module in modules.values() for p in module.parameters()):
        raise ValueError("Frozen-feature probe received trainable visual modules.")
    grids = {
        "vision_tower": tuple(arguments.tower_grid),
        "connector": tuple(arguments.connector_grid),
    }
    storage = {}

    def hook_for(name):
        def hook(_module, _args, output):
            tensor = output if isinstance(output, torch.Tensor) else output.last_hidden_state
            if tensor.ndim != 3 or tensor.shape[0] != 1:
                raise ValueError("Expected one image with a token-by-channel representation.")
            storage[name] = tensor[0].detach().float().cpu().numpy()

        return hook

    handles = [module.register_forward_hook(hook_for(name)) for name, module in modules.items()]
    starts = dict(
        zip(
            dataset.meta.episodes["episode_index"],
            dataset.meta.episodes["dataset_from_index"],
            strict=True,
        )
    )
    features = {name: {"mean": [], "spatial_2x2": []} for name in modules}
    image_hashes = []
    camera = cfg.cameras["top"]
    try:
        for index, episode in enumerate(episodes):
            sample = dataset[dataset.absolute_to_relative_idx[int(starts[episode])]]
            if (
                int(sample[cfg.fields.episode_index]) != episode
                or int(sample[cfg.fields.frame_index]) != 0
            ):
                raise ValueError("Image sample identity differs from the requested label.")
            image, state = sample[camera], sample[cfg.fields.state]
            if image.ndim == 4:
                image = image[0]
            if state.ndim == 2:
                state = state[0]
            if not np.array_equal(state.cpu().numpy(), context["states"][index]):
                raise ValueError("Image and label readers returned different states.")
            image_hashes.append(
                hashlib.sha256(image.contiguous().cpu().numpy().tobytes()).hexdigest()
            )
            if image.dtype == torch.uint8:
                image = image.float() / 255.0
            batch = preprocessor(
                {
                    camera: image,
                    "observation.state": state,
                    "task": "Insert the peg into the socket.",
                }
            )
            images, masks = policy.prepare_images(batch)
            if not bool(masks[0].all()):
                raise ValueError("The first camera must be the real, unmasked camera.")
            patch = int(core.vision_model.config.patch_size)
            height, width = images[0].shape[-2:]
            scale = int(core.connector.scale_factor)
            if patch <= 0 or scale <= 0 or height % patch or width % patch:
                raise ValueError("Unsupported patch/resampling geometry.")
            tower_grid = (height // patch, width // patch)
            if tower_grid[0] != tower_grid[1] or any(side % scale for side in tower_grid):
                raise ValueError("Pinned connector requires a square, scale-divisible token grid.")
            expected_grids = {
                "vision_tower": tower_grid,
                "connector": tuple(side // scale for side in tower_grid),
            }
            if grids != expected_grids:
                raise ValueError(
                    f"Declared grids disagree with runtime patch layout: {expected_grids}"
                )
            storage.clear()
            with (
                torch.inference_mode(),
                torch.autocast(device_type=device.type, dtype=torch.bfloat16),
            ):
                policy.model.vlm_with_expert.embed_image(images[0].to(device))
            for name in modules:
                tokens = storage[name]
                features[name]["mean"].append(tokens.mean(axis=0))
                features[name]["spatial_2x2"].append(spatial_features(tokens, grids[name]))
    finally:
        for handle in handles:
            handle.remove()
    results = {
        name: {
            readout: fit_probe(np.stack(values), context["actions"], len(context["train"]))
            for readout, values in readouts.items()
        }
        for name, readouts in features.items()
    }
    report = {
        "schema_version": 2,
        "stage": "frozen_spatial_readout",
        "gating": False,
        "artifact_id": arguments.artifact_id,
        "dataset_revision": cfg.revision,
        "artifact_manifest_sha256": policy._rosetta_diagnostic_manifest_sha256,
        "train_episodes": context["train"],
        "validation_episodes": context["validation"],
        "hidden_rows_materialized": False,
        "grids": grids,
        "image_sha256": image_hashes,
        "results": results,
        "limitations": [
            "Validation uses five previously opened episodes; it is not a fresh final test.",
            "Report both readouts; do not select them or hyperparameters using validation.",
            "Contextualized expert KV is downstream of these features and remains untested.",
            "Readout failure does not prove information absence or justify unfreezing alone.",
        ],
    }
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if arguments.output is not None:
        with arguments.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
