"""Local streaming train/dev numeric audit and explicit temporal sampling draft.

No policy, optimizer, remote connection, download or hidden-row materialization.
Images are a separate optional, explicitly measured sample rather than inferred
from numeric integrity. Outputs are create-only, including failure records.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def audit(output: Path, decode_images: bool = False, all_images: bool = False):
    if all_images:
        decode_images = True
    import numpy as np
    import pyarrow.dataset as arrow
    import torch

    from rosetta_reality.data import resolve_prepared_cache
    from rosetta_reality.data.config import load_dataset_config
    from rosetta_reality.experiment import file_sha256
    from rosetta_reality.sim import load_action_contract
    from rosetta_reality.vla.action_space import load_smolvla_action_space, load_smolvla_experiment
    from rosetta_reality.vla.processor import (
        standard_aloha_action_to_model,
        standard_aloha_state_to_pi,
    )
    from rosetta_reality.vla.training.sample_audit import (
        coverage,
        schedule_digest,
        spaced_frames,
        temporal_schedule,
    )
    from rosetta_reality.vla.vision_diagnostics import validate_splits
    from scripts.inspect_hestia_schedule import inspect_schedule

    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("HF_DATASETS_OFFLINE") != "1":
        raise RuntimeError("Dataset audit must run offline in the registered container")
    cfg_path = ROOT / "configs/data/aloha_sim_insertion_m2.yaml"
    exp_path = (
        ROOT / "configs/vla/smolvla_450m_aloha_insertion_action_repair_bounded_gripper_003.yaml"
    )
    cfg = load_dataset_config(cfg_path)
    exp = load_smolvla_experiment(exp_path, ROOT)
    train, dev, hidden = (
        exp["dataset"][key] for key in ("train_episodes", "validation_episodes", "test_episodes")
    )
    validate_splits(train, dev, hidden)
    if cfg.repo_id != exp["dataset"]["identifier"] or cfg.revision != exp["dataset"]["revision"]:
        raise ValueError("Dataset and experiment identity differ")
    root, _ = resolve_prepared_cache(cfg, ROOT, validate_checksums=True)
    info = json.loads((root / "meta/info.json").read_text())
    fps = info["fps"]
    if type(fps) not in (int, float) or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Dataset FPS is invalid")
    contract_path = ROOT / exp["action_contract"]["derived"]
    contract = load_action_contract(contract_path)
    space = load_smolvla_action_space(exp, require_explicit=True)
    dataset = arrow.dataset(root / "data", format="parquet")
    fields = cfg.fields
    results, lengths, sampled_rows, raw_actions = {}, {}, {}, {}
    columns = [
        fields.episode_index,
        fields.frame_index,
        fields.timestamp,
        fields.state,
        fields.action,
    ]
    for ep in train + dev:
        # Predicate is applied before any row or array is materialized.
        scanner = dataset.scanner(
            columns=columns, filter=arrow.field(fields.episode_index) == ep, batch_size=256
        )
        seen, times, states, actions = set(), {}, [], []
        rows = {}
        for batch in scanner.to_batches():
            for row in batch.to_pylist():
                frame = row[fields.frame_index]
                if (
                    row[fields.episode_index] != ep
                    or type(frame) is not int
                    or frame < 0
                    or frame in seen
                ):
                    raise ValueError(f"Duplicate or invalid episode/frame identity in episode {ep}")
                seen.add(frame)
                times[frame] = row[fields.timestamp]
                rows[frame] = row
        if not seen or seen != set(range(len(seen))):
            raise ValueError(f"Missing frame in episode {ep}")
        lengths[ep] = len(seen)
        for frame in range(len(seen)):
            states.append(rows[frame][fields.state])
            actions.append(rows[frame][fields.action])
        state, action = np.asarray(states, dtype=np.float64), np.asarray(actions, dtype=np.float64)
        if decode_images:
            raw_actions[ep] = action
        timestamp = np.asarray([times[t] for t in range(len(seen))], dtype=np.float64)
        if state.shape != action.shape or state.shape != (len(seen), contract.dimension):
            raise ValueError("State/action dimension contract differs")
        if not all(np.isfinite(a).all() for a in (state, action, timestamp)):
            raise ValueError("Nonfinite raw input")
        if not np.allclose(timestamp, np.arange(len(seen)) / fps, atol=1e-5, rtol=0):
            raise ValueError("Timestamp/frame alignment differs")
        lower, upper = contract.lower_bounds.numpy(), contract.upper_bounds.numpy()
        overshoot = np.maximum(np.maximum(lower - action, action - upper), 0)
        if (overshoot > contract.source_overshoot_tolerances.numpy() + 1e-6).any():
            raise ValueError("Source action exceeds registered overshoot tolerance")
        projected = np.clip(action, lower, upper)
        encoded = standard_aloha_action_to_model(
            torch.from_numpy(projected), space.representation_adapter
        ).numpy()
        encoded_state = standard_aloha_state_to_pi(torch.from_numpy(state)).numpy()
        if not np.isfinite(encoded).all() or not np.isfinite(encoded_state).all():
            raise ValueError("Nonfinite encoded input")
        results[str(ep)] = {
            "split": "train" if ep in train else "validation",
            "rows": len(seen),
            "projected_elements": int((overshoot > 0).sum()),
            "projected_per_dimension": (overshoot > 0).sum(axis=0).tolist(),
            "maximum_overshoot": overshoot.max(axis=0).tolist(),
            "state_range": [state.min(axis=0).tolist(), state.max(axis=0).tolist()],
            "encoded_action_range": [encoded.min(axis=0).tolist(), encoded.max(axis=0).tolist()],
        }
        for frame in range(len(seen)) if all_images else spaced_frames(len(seen), 9):
            sampled_rows[(ep, frame)] = rows[frame]
        print(f"Numeric audit episode {ep}: {len(seen)} rows", flush=True)
    control = inspect_schedule()["sample_identities"]
    train_lengths = {ep: lengths[ep] for ep in train}
    treatment = temporal_schedule(control, train_lengths, seed=20260809)
    for name, samples in (("control", control), ("temporal", treatment)):
        write(
            output / f"{name}-schedule.json",
            {
                "status": "draft_not_launchable",
                "training_authorized": False,
                "seed": 20260809,
                "sample_identities": samples,
                "schedule_sha256": schedule_digest(samples),
                "coverage": coverage(samples, train_lengths),
            },
        )
    write(
        output / "evaluation-samples.json",
        [{"episode": ep, "frame": frame} for ep, frame in sampled_rows],
    )
    image_records = []
    if decode_images:
        import hashlib

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset(
            cfg.repo_id,
            root=root,
            episodes=train + dev,
            revision=cfg.revision,
            download_videos=False,
            return_uint8=True,
            video_backend="pyav",
            delta_timestamps={"action": [i / fps for i in range(50)]},
        )
        starts = dict(
            zip(
                ds.meta.episodes["episode_index"],
                ds.meta.episodes["dataset_from_index"],
                strict=True,
            )
        )
        for (ep, frame), row in sampled_rows.items():
            sample = ds[ds.absolute_to_relative_idx[int(starts[ep]) + frame]]
            if int(sample[fields.episode_index]) != ep or int(sample[fields.frame_index]) != frame:
                raise ValueError("Decoded sample identity differs")
            if not np.array_equal(sample[fields.state].numpy(), row[fields.state]):
                raise ValueError("Decoded state disagrees with raw scanner")
            if not np.array_equal(sample[fields.action][0].numpy(), row[fields.action]):
                raise ValueError("Action chunk is not aligned to the current frame")
            target_indices = np.minimum(np.arange(50) + frame, lengths[ep] - 1)
            if not np.array_equal(sample[fields.action].numpy(), raw_actions[ep][target_indices]):
                raise ValueError("Complete action chunk disagrees with raw episode targets")
            expected_pad = np.arange(50) + frame >= lengths[ep]
            if not np.array_equal(sample["action_is_pad"].numpy(), expected_pad):
                raise ValueError("Action padding crosses an episode boundary")
            if sample["task"] != cfg.expected_instruction:
                raise ValueError("Instruction differs")
            image = sample[cfg.cameras["top"]]
            if image.dtype != torch.uint8 or image.ndim != 3 or image.shape[0] != 3:
                raise ValueError("Camera must decode to uint8 CHW RGB")
            image_records.append(
                {
                    "episode": ep,
                    "frame": frame,
                    "sha256": hashlib.sha256(image.numpy().tobytes()).hexdigest(),
                    "valid_action_slots": int((~sample["action_is_pad"]).sum()),
                }
            )
        write(output / "decoded-samples.json", image_records)
    report = {
        "status": "completed_numeric_audit",
        "model_loaded": False,
        "optimizer_steps": 0,
        "hidden_rows_materialized": 0,
        "dataset_revision": cfg.revision,
        "dataset_manifest_sha256": file_sha256(root / "manifest.json"),
        "dataset_config_sha256": file_sha256(cfg_path),
        "experiment_sha256": file_sha256(exp_path),
        "action_contract_sha256": file_sha256(contract_path),
        "train_rows": sum(lengths[ep] for ep in train),
        "validation_rows": sum(lengths[ep] for ep in dev),
        "episodes": results,
        "decoded_image_samples": len(image_records),
        "full_action_chunk_samples_verified": len(image_records),
        "all_nonhidden_image_frames_decoded": len(image_records) == sum(lengths.values()),
        "all_image_frames_verified": False,
        "independent_video_pts_and_physical_semantics": "not measured",
        "saved_processor_normalization_parity": "not measured",
        "control_coverage": coverage(control, train_lengths),
        "temporal_coverage": coverage(treatment, train_lengths),
        "task_success": "not measured",
        "m2_complete": False,
    }
    write(output / "result.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--decode-images", action="store_true")
    parser.add_argument("--all-images", action="store_true", help="Decode every train/dev frame")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        result = audit(args.output, args.decode_images, args.all_images)
    except BaseException as exc:
        write(
            args.output / "failure.json",
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "model_loaded": False,
                "optimizer_steps": 0,
            },
        )
        raise
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "train_rows", "validation_rows", "decoded_image_samples")
            }
        )
    )


if __name__ == "__main__":
    main()
