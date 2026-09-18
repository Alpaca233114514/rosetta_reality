"""Audit real train-only temporal loader images against independent PyAV seeking.

No policy is constructed and no optimizer runs. Decoded video agreement establishes
digital alignment, not correspondence between physical observations and labels.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


def audit(output):
    import av
    import numpy as np
    import pyarrow.dataset as arrow
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.scripts import lerobot_train
    from torch.utils.data import DataLoader

    from rosetta_reality.data import resolve_prepared_cache
    from rosetta_reality.data.config import load_dataset_config
    from rosetta_reality.experiment import file_sha256
    from rosetta_reality.vla.action_space import load_smolvla_experiment
    from rosetta_reality.vla.training.features import FeatureStack

    if any(os.environ.get(k) != "1" for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")):
        raise ValueError("Offline runtime required")
    torch.set_num_threads(1)
    cfg = load_dataset_config(ROOT / "configs/data/aloha_sim_insertion_m2.yaml")
    exp = load_smolvla_experiment(
        ROOT / "configs/vla/smolvla_450m_aloha_insertion_action_repair_bounded_gripper_003.yaml",
        ROOT,
    )
    episodes = exp["dataset"]["train_episodes"][:2]
    root, _ = resolve_prepared_cache(cfg, ROOT, validate_checksums=True)
    ds = LeRobotDataset(
        cfg.repo_id, root=root, revision=cfg.revision, episodes=episodes,
        download_videos=False, return_uint8=True, video_backend="pyav",
        delta_timestamps={"action": [i / 50 for i in range(50)]},
    )
    rows = arrow.dataset(root / "data", format="parquet").to_table(
        filter=arrow.field(cfg.fields.episode_index).isin(episodes),
        columns=[cfg.fields.episode_index, cfg.fields.frame_index,
                 cfg.fields.timestamp, cfg.fields.state, cfg.fields.action],
    ).to_pylist()
    raw = {(r[cfg.fields.episode_index], r[cfg.fields.frame_index]): r for r in rows}
    samples = [[ep, frame] for frame in (0, 249, 450, 499) for ep in episodes]
    schedule = output / "schedule.json"
    schedule.write_text(json.dumps({"status": "preregistered", "training_authorized": True,
                                    "seed": exp["seed"], "sample_identities": samples}))
    context = SimpleNamespace(
        phase="smoke", experiment=exp,
        plan={"scope": "bounded_temporal_sampling",
              "optimizer_smoke": {"batch_size": 4, "steps": 2, "episodes": episodes}},
    )
    stack = FeatureStack.from_plan({"features": [{
        "name": "explicit_sample_schedule", "path": schedule.relative_to(ROOT).as_posix(),
        "sha256": file_sha256(schedule),
    }]})
    original = lerobot_train.EpisodeAwareSampler
    records = []
    try:
        stack.install_all(context)
        sampler = lerobot_train.EpisodeAwareSampler(
            ds.meta.episodes["dataset_from_index"], ds.meta.episodes["dataset_to_index"],
            episodes, shuffle=True, seed=exp["seed"],
            absolute_to_relative_idx=ds.absolute_to_relative_idx,
        )
        for batch in DataLoader(ds, sampler=sampler, batch_size=4, num_workers=0):
            for i in range(len(batch["episode_index"])):
                ep, frame = int(batch["episode_index"][i]), int(batch["frame_index"][i])
                row = raw[(ep, frame)]
                assert [ep, frame] == samples[len(records)]
                length = sum(k[0] == ep for k in raw)
                target = np.asarray([raw[(ep, min(frame + t, length - 1))][cfg.fields.action]
                                     for t in range(50)])
                assert np.array_equal(batch["action"][i].numpy(), target)
                assert np.array_equal(batch["action_is_pad"][i].numpy(),
                                      np.arange(50) + frame >= length)
                assert np.array_equal(batch[cfg.fields.state][i].numpy(), row[cfg.fields.state])
                assert abs(float(batch["timestamp"][i]) - frame / ds.meta.fps) < 1e-5
                key = cfg.cameras["top"]
                video = root / ds.meta.get_video_file_path(ep, key)
                metadata = ds.meta.episodes[ep]
                query = float(metadata[f"videos/{key}/from_timestamp"]) + frame / ds.meta.fps
                candidates = []
                with av.open(str(video)) as container:
                    stream = container.streams.video[0]
                    container.seek(int(query / stream.time_base), stream=stream,
                                   backward=True, any_frame=False)
                    for decoded in container.decode(stream):
                        timestamp = float(decoded.pts * decoded.time_base)
                        # Decoder keyframe preroll is discarded before RGB materialization.
                        if abs(timestamp - query) <= 1 / ds.meta.fps:
                            candidates.append((abs(timestamp - query), timestamp,
                                               decoded.to_ndarray(format="rgb24")))
                        if timestamp >= query + 1 / ds.meta.fps:
                            break
                distance, pts, image = min(candidates, key=lambda x: x[0])
                assert distance < 1e-4, "Video PTS disagrees with registered frame time"
                expected = np.transpose(image, (2, 0, 1))
                actual = batch[key][i].numpy()
                assert np.array_equal(actual, expected), "Dataset image disagrees with direct video"
                records.append({"episode": ep, "frame": frame, "video_pts": pts,
                                "timestamp_error": distance, "image_exact": True,
                                "valid_action_slots": int((~batch["action_is_pad"][i]).sum())})
    finally:
        stack.restore_all(context)
    assert lerobot_train.EpisodeAwareSampler is original
    assert len(records) == len(samples)
    return {"status": "passed", "samples": records, "loader_inputs": len(samples),
            "unique_input_frames": len(samples), "raw_rows_materialized": len(rows),
            "hidden_rows_materialized": 0, "model_weights_loaded": False,
            "optimizer_steps": 0, "actual_training_updates": "not measured",
            "physical_image_state_semantics": "not established by digital alignment",
            "video_decoder": "explicit pyav; production default equivalence not established"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"status": "failed"}
    try:
        result = audit(args.output)
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        with (args.output / "result.json").open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
