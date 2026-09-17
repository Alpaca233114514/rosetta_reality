"""Read-only raw-cache versus native train-view video comparison; no policy weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ["HF_HUB_OFFLINE"] == os.environ["HF_DATASETS_OFFLINE"] == "1"
    root = args.reference_workspace.resolve(strict=True)
    assert root.name == "20260914T075851Z-95cf9cf9483b-6512cafb4943"
    sys.path[:0] = [str(root), str(root / "src"), str(root / "scripts")]
    import torch
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from torch.utils.data import default_collate

    from rosetta_reality.data.config import load_dataset_config
    from scripts import run_smolvla_v2 as launcher
    from scripts.smolvla_forward_check import _make_processors
    from scripts.training_chain_gpu_audit import SAMPLES, temporal_data

    torch.set_num_threads(1)
    plan_path = root / "runs/training-chain-gpu-audit-20260914-005/smoke.yaml"
    plan, _, exp = launcher._resolve_plan(plan_path)
    config = SmolVLAConfig.from_pretrained(
        launcher.phase_runner._model_root(exp), local_files_only=True
    )
    for section in ("policy", "adaptation"):
        for key, value in exp["model"][section].items():
            if hasattr(config, key):
                setattr(config, key, value)
    _, raw, _ = temporal_data(
        SimpleNamespace(pre=lambda batch: batch, policy=SimpleNamespace(config=config))
    )
    view = (root / plan["normalization"]["dataset_view_manifest"]).parent
    data_config = load_dataset_config(root / "configs/data/aloha_sim_insertion_m2.yaml")
    meta = LeRobotDatasetMetadata(data_config.repo_id, root=view, revision=data_config.revision)
    deltas = resolve_delta_timestamps(config, meta)
    ds = LeRobotDataset(
        data_config.repo_id, root=view, episodes=[49, 4], revision=data_config.revision,
        download_videos=False, video_backend="pyav", delta_timestamps=deltas,
    )
    starts = dict(zip(meta.episodes["episode_index"], meta.episodes["dataset_from_index"]))
    rows, native_samples = [], []
    for pair, original in zip(SAMPLES, raw, strict=True):
        ep, frame = pair
        actual = ds[ds.absolute_to_relative_idx[int(starts[ep]) + frame]]
        native_samples.append(actual)
        expected = original["observation.images.top"].float() / 255
        value = actual["observation.images.top"]
        cuda_scaled = (original["observation.images.top"].cuda().float() / 255).cpu()
        row = {
            "sample": pair, "expected_shape": list(expected.shape),
            "actual_shape": list(value.shape), "actual_dtype": str(value.dtype),
            "exact": torch.equal(value, expected),
            "max_abs_flat": (
                float((value.flatten() - expected.flatten()).abs().max())
                if value.numel() == expected.numel() else None
            ),
            "decoded_bytes_equal": torch.equal(
                (value * 255).round().to(torch.uint8).flatten(),
                original["observation.images.top"].flatten(),
            ),
            "state_exact": torch.equal(actual["observation.state"], original["observation.state"]),
            "action_exact": torch.equal(actual["action"], original["action"]),
            "cpu_cuda_scaling_max_abs": float((cuda_scaled - expected).abs().max()),
            "cpu_cuda_scaling_different_values": int((cuda_scaled != expected).sum()),
            "cuda_scaled_pixels_roundtrip_exact": torch.equal(
                (cuda_scaled * 255).round().to(torch.uint8), original["observation.images.top"]
            ),
        }
        rows.append(row)
    config.pretrained_path = launcher.phase_runner._model_root(exp)
    config.pretrained_revision = None
    config.device = "cpu"
    pre, _ = _make_processors(
        SimpleNamespace(policy=config, rename_map=exp["dataset"]["rename_map"]),
        SimpleNamespace(config=config), ds, torch.device("cpu"),
    )
    processor_rows = []
    for start in (0, 4):
        batch = default_collate(native_samples[start:start + 4])
        expected = batch["observation.images.top"].clone()
        processed = pre(batch)
        actual = processed["observation.images.camera1"]
        processor_rows.append({
            "start": start, "before_shape": list(expected.shape),
            "after_shape": list(actual.shape), "exact": torch.equal(actual, expected),
            "max_abs_flat": float((actual.flatten() - expected.flatten()).abs().max()),
            "keys_after_processor": sorted(processed),
            "state_shape_after_processor": list(processed["observation.state"].shape),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    values = torch.arange(256, dtype=torch.uint8)
    cpu_values = values.float() / 255
    cuda_values = (values.cuda().float() / 255).cpu()
    with args.output.open("x") as stream:
        json.dump({
            "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            "policy_created": False, "optimizer_steps": 0, "rows": rows,
            "native_delta_timestamps": deltas,
            "processor_rows": processor_rows,
            "all_256_pixel_values": {
                "max_abs": float((cpu_values - cuda_values).abs().max()),
                "different_values": int((cpu_values != cuda_values).sum()),
                "uint8_roundtrip_exact": torch.equal(
                    (cuda_values * 255).round().to(torch.uint8), values
                ),
            },
        }, stream, indent=2)
    print(json.dumps(rows))
    print(json.dumps(processor_rows))


if __name__ == "__main__":
    main()
