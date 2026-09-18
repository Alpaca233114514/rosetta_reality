"""Create-only train-slice visual grounding across four trajectory offsets.

This diagnostic never trains, selects a checkpoint, changes Gate protocols or
loads validation/hidden samples. See the dated local diagnostic registration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for root in (REPOSITORY_ROOT / "src", REPOSITORY_ROOT / "scripts"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Temporal diagnostic output is create-only.")
    if any(os.environ.get(name) != "1" for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")):
        raise RuntimeError("Temporal visual diagnostics require offline model/data access.")

    import numpy as np
    import torch
    import yaml
    from diagnose_frame0_vision_probe import _load_artifact
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from rosetta_reality.data import resolve_prepared_cache
    from rosetta_reality.data.config import load_dataset_config
    from rosetta_reality.experiment import file_sha256
    from rosetta_reality.sim import load_action_contract
    from rosetta_reality.vla.action_space import load_smolvla_experiment
    from rosetta_reality.vla.visual_grounding import (
        FRAME_OFFSETS,
        NOISE_SEEDS,
        TRAIN_EPISODES,
        intervention_sample,
        read_temporal_rows,
        summarize_offset,
        validate_image_ingress,
    )

    experiment_path = (
        REPOSITORY_ROOT
        / "configs/vla/smolvla_450m_aloha_insertion_action_repair_bounded_gripper_003.yaml"
    )
    experiment = load_smolvla_experiment(experiment_path, REPOSITORY_ROOT)
    dataset_path = REPOSITORY_ROOT / "configs/data/aloha_sim_insertion_m2.yaml"
    cfg = load_dataset_config(dataset_path)
    if (
        cfg.repo_id != experiment["dataset"]["identifier"]
        or cfg.revision != experiment["dataset"]["revision"]
    ):
        raise ValueError("Experiment and dataset identity disagree.")
    # Check the requested immutable manifest before any weight deserialization.
    if not args.artifact_id or any(
        char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in args.artifact_id
    ):
        raise ValueError("Invalid artifact identifier.")
    manifest = (
        Path(os.environ["ROSETTA_ARTIFACT_ROOT"])
        / experiment["experiment_id"]
        / args.artifact_id
        / "manifest.json"
    )
    if file_sha256(manifest) != args.expected_manifest_sha256:
        raise ValueError("Artifact manifest differs from the preregistered checksum.")
    root, _ = resolve_prepared_cache(cfg, REPOSITORY_ROOT, validate_checksums=True)
    rows = read_temporal_rows(
        root, cfg, experiment["dataset"]["train_episodes"], experiment["dataset"]["test_episodes"]
    )
    dataset = LeRobotDataset(
        cfg.repo_id,
        root=root,
        episodes=list(TRAIN_EPISODES),
        revision=cfg.revision,
        download_videos=False,
        return_uint8=True,
    )
    starts = dict(
        zip(
            dataset.meta.episodes["episode_index"],
            dataset.meta.episodes["dataset_from_index"],
            strict=True,
        )
    )
    contract_path = REPOSITORY_ROOT / "configs/sim/aloha_insertion_smolvla.yaml"
    contract = load_action_contract(contract_path)
    if (
        file_sha256(contract_path)
        != "fc71a0438f0e3af7258e5b52d82fa22fc53c12b47901606cbee715524392ac62"
    ):
        raise ValueError("The registered Action Contract changed.")
    dimensions = yaml.safe_load(contract_path.read_text())["action"]["dimensions"]
    device = torch.device(os.environ["ROSETTA_TORCH_DEVICE"])
    started = time.monotonic()
    policy, (pre, post) = _load_artifact(args.artifact_id, device)
    if policy.config.n_obs_steps != 1 or policy.config.n_action_steps != 1:
        raise ValueError(
            "The diagnostic requires the registered single-observation first-action policy."
        )
    results, image_ids = [], []
    ingress = None
    calls = 0
    for offset in FRAME_OFFSETS:
        samples, targets, states = [], [], []
        for episode in TRAIN_EPISODES:
            sample = dataset[dataset.absolute_to_relative_idx[int(starts[episode]) + offset]]
            if (
                int(sample[cfg.fields.episode_index]) != episode
                or int(sample[cfg.fields.frame_index]) != offset
            ):
                raise ValueError("Decoded observation identity differs from the registered sample.")
            row = rows[(episode, offset)]
            if not np.array_equal(
                sample[cfg.fields.state].numpy(), row[cfg.fields.state]
            ) or not np.array_equal(sample[cfg.fields.action].numpy(), row[cfg.fields.action]):
                raise ValueError("Video reader and label scanner disagree on sample identity.")
            if sample["task"] != cfg.expected_instruction:
                raise ValueError("Dataset instruction differs from the registered task.")
            samples.append(sample)
            image = sample[cfg.cameras["top"]]
            image_ids.append(
                {
                    "episode": episode,
                    "frame": offset,
                    "sha256": hashlib.sha256(image.contiguous().numpy().tobytes()).hexdigest(),
                }
            )
            raw = torch.as_tensor(row[cfg.fields.action], dtype=torch.float32)
            if (
                (raw < contract.lower_bounds - contract.source_overshoot_tolerances)
                | (raw > contract.upper_bounds + contract.source_overshoot_tolerances)
            ).any():
                raise ValueError("Expert action exceeds the registered source tolerance.")
            targets.append(contract.clip(raw)[0].numpy())
            states.append(row[cfg.fields.state])
        for noise_seed in NOISE_SEEDS:
            predictions = {"correct": [], "mismatched": []}
            shape = (1, policy.config.chunk_size, policy.config.max_action_dim)
            noise = (
                torch.zeros(shape)
                if noise_seed is None
                else torch.randn(shape, generator=torch.Generator().manual_seed(noise_seed))
            )
            for condition in predictions:
                for index, destination in enumerate(samples):
                    if time.monotonic() - started > 1200:
                        raise TimeoutError(
                            "The registered 20-minute inference budget was exceeded."
                        )
                    source = (
                        destination
                        if condition == "correct"
                        else samples[(index - 1) % len(samples)]
                    )
                    batch = pre(intervention_sample(destination, source, cfg.cameras["top"]))
                    ingress = validate_image_ingress(policy, batch)
                    policy.reset()
                    with (
                        torch.inference_mode(),
                        torch.autocast(device_type=device.type, dtype=torch.bfloat16),
                    ):
                        output = post(
                            policy.predict_action_chunk(
                                batch,
                                noise=noise.to(
                                    device=device, dtype=batch["observation.state"].dtype
                                ).clone(),
                            )
                        )
                    if output.shape != (1, policy.config.chunk_size, contract.dimension):
                        raise ValueError("Predicted chunk shape differs from the Action Contract.")
                    if not torch.isfinite(output).all():
                        raise FloatingPointError("Non-finite action terminates the diagnostic.")
                    predictions[condition].append(output[0, 0].detach().float().cpu().numpy())
                    calls += 1
            result = summarize_offset(
                predictions["correct"], predictions["mismatched"], targets, states, dimensions
            )
            result.update(
                {
                    "frame_offset": offset,
                    "noise_seed": noise_seed,
                    "correct_predictions": np.asarray(predictions["correct"]).tolist(),
                    "mismatched_predictions": np.asarray(predictions["mismatched"]).tolist(),
                }
            )
            results.append(result)
        print(json.dumps({"completed_offset": offset, "forward_calls": calls}), flush=True)
    report = {
        "schema_version": 1,
        "stage": "train_temporal_visual_grounding",
        "gating": False,
        "artifact_id": args.artifact_id,
        "artifact_manifest_sha256": args.expected_manifest_sha256,
        "dataset_revision": cfg.revision,
        "dataset_manifest_sha256": file_sha256(root / "manifest.json"),
        "action_contract_sha256": file_sha256(contract_path),
        "container_image_id": os.environ.get("ROSETTA_CONTAINER_IMAGE_ID"),
        "device": device.type,
        "torch_version": torch.__version__,
        "episodes": list(TRAIN_EPISODES),
        "mismatched_image_episodes": list(TRAIN_EPISODES[-1:] + TRAIN_EPISODES[:-1]),
        "frame_offsets": list(FRAME_OFFSETS),
        "noise_seeds": list(NOISE_SEEDS),
        "hidden_test_loaded": False,
        "validation_loaded": False,
        "optimizer_created": False,
        "forward_calls": calls,
        "elapsed_seconds": time.monotonic() - started,
        "image_identity": image_ids,
        "ingress": ingress,
        "results": results,
        "implementation_sha256": {
            name: file_sha256(REPOSITORY_ROOT / name)
            for name in (
                "scripts/diagnose_visual_grounding.py",
                "scripts/diagnose_frame0_vision_probe.py",
                "src/rosetta_reality/vla/visual_grounding.py",
                "src/rosetta_reality/vla/vision_diagnostics.py",
            )
        },
        "limitations": [
            "Train-slice evidence only; it cannot establish generalization or Gate 4 success.",
            "Cross-episode images may conflict with destination proprioception away from reset.",
            "Sensitivity and target alignment are separate; neither proves task competence.",
            "Labels use Action Contract projection; aggregate mixed-unit MAE is a proxy.",
        ],
    }
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(text)
    print(json.dumps({"status": "complete", "forward_calls": calls}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
