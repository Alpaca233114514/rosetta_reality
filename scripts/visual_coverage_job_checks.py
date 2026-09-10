"""Real input and full-chunk smoke checks for the registered unattended job."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def validate_state_action_shapes(batch, *, n_obs_steps, chunk_size, action_dimension):
    expected = {
        "observation.state": (1, n_obs_steps, action_dimension),
        "action": (1, chunk_size, action_dimension),
    }
    for key, shape in expected.items():
        actual = tuple(batch[key].shape)
        if actual != shape:
            raise ValueError(f"Native {key} shape differs: expected {shape}, received {actual}")


def main():
    import evaluate_visual_native_small as n
    import numpy as np
    import torch

    from rosetta_reality.vla.fixed_visual_samples import resolve_visual_sample_indices
    from rosetta_reality.vla.processor import PiAlohaPostprocessorStep

    mode, plan_name, output = sys.argv[1:]
    plan, parent, experiment = n._resolve_plan(Path(plan_name).resolve())
    physical_path = ROOT / experiment["action_contract"]["derived"]
    physical_sha = n.file_sha256(physical_path)
    n._validate_prerequisites(plan, experiment, parent, physical_sha)
    norm_path, _, view = n._validate_normalization(plan, experiment, parent, physical_sha)
    norm = json.loads(norm_path.read_text())
    context = n.load_frame_zero_context(ROOT, "non_hidden")
    episodes = (
        experiment["dataset"]["train_episodes"] + experiment["dataset"]["validation_episodes"]
    )
    assert len(episodes) == 45 and not set(episodes) & set(experiment["dataset"]["test_episodes"])
    smoke = plan["optimizer_smoke"]
    source = (
        Path(os.environ["ROSETTA_CHECKPOINT_ROOT"])
        / experiment["experiment_id"]
        / "smoke"
        / smoke["run_name"]
        / "checkpoints"
        / f"{smoke['steps']:06d}"
        / "pretrained_model"
    )
    cfg = n.SmolVLAConfig.from_pretrained(source, local_files_only=True)
    cfg.device, cfg.pretrained_path, cfg.pretrained_revision = "cuda", source, None
    for section in ("policy", "adaptation"):
        for key, value in experiment["model"][section].items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
    cfg.load_vlm_weights = False
    assert cfg.num_steps == 10 and cfg.chunk_size == 50 and cfg.max_action_dim == 32
    data_cfg = context["config"]
    meta = n.LeRobotDatasetMetadata(
        data_cfg.repo_id, root=context["root"], revision=data_cfg.revision
    )
    dataset = n.LeRobotDataset(
        data_cfg.repo_id,
        root=context["root"],
        episodes=episodes,
        revision=data_cfg.revision,
        delta_timestamps=n.resolve_delta_timestamps(cfg, meta),
        download_videos=False,
        return_uint8=True,
    )
    dataset.meta.stats.update(
        {
            k: {s: np.asarray(v) for s, v in values.items()}
            for k, values in norm["effective_stats"].items()
        }
    )
    torch.manual_seed(experiment["seed"])
    policy = n.make_policy(
        cfg=cfg, ds_meta=dataset.meta, rename_map=experiment["dataset"]["rename_map"]
    )
    pre, post = n._make_processors(
        SimpleNamespace(policy=cfg, rename_map=experiment["dataset"]["rename_map"]),
        policy,
        dataset,
        torch.device("cuda"),
    )
    physical = n.load_action_contract(physical_path)
    n.ensure_smolvla_action_boundary(
        pre,
        post,
        physical,
        n.load_smolvla_action_space(experiment),
        action_contract_sha256=physical_sha,
        upstream_revision=experiment["upstream"]["revision"],
    )
    decoders = [step for step in post.steps if isinstance(step, PiAlohaPostprocessorStep)]
    assert len(decoders) == 1
    decoder = decoders[0]

    def array(tensor):
        return tensor.detach().cpu().double().numpy()

    def digest(tensor):
        return hashlib.sha256(str(tensor.dtype).encode() + array(tensor).tobytes()).hexdigest()

    starts = dict(
        zip(
            dataset.meta.episodes["episode_index"],
            dataset.meta.episodes["dataset_from_index"],
            strict=True,
        )
    )
    batches, records = [], []
    assert len(dataset.meta.camera_keys) == 1
    camera = dataset.meta.camera_keys[0]
    for ep in episodes:
        sample = dataset[dataset.absolute_to_relative_idx[int(starts[ep])]]
        assert int(sample["episode_index"]) == ep and int(sample["frame_index"]) == 0
        assert not bool(sample["action_is_pad"].any())
        raw = n.default_collate([sample])
        assert raw[camera].dtype == torch.uint8 and raw["action"].shape == (1, 50, 14)
        record = {
            "episode": ep,
            "frame": 0,
            "raw_image_sha256": digest(raw[camera]),
            "raw_state_sha256": digest(raw["observation.state"]),
            "target_sha256": digest(raw["action"]),
            "timestamp": float(sample["timestamp"]),
            "task": sample["task"],
        }
        raw[camera] = raw[camera].float() / 255
        batch = pre(raw)
        validate_state_action_shapes(
            batch,
            n_obs_steps=cfg.n_obs_steps,
            chunk_size=cfg.chunk_size,
            action_dimension=physical.dimension,
        )
        images, masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        assert len(images) == len(masks) == 3 and state.shape == (1, 32)
        assert bool(masks[0].all()) and not any(bool(x.any()) for x in masks[1:])
        assert all(bool((image == -1).all()) for image in images[1:])
        assert torch.isfinite(images[0]).all() and images[0].min() >= -1 and images[0].max() <= 1
        fields = [
            state,
            batch["observation.language.tokens"],
            batch["observation.language.attention_mask"],
            *masks,
            *images[1:],
        ]
        record["nonvisual_sha256"] = hashlib.sha256(
            "|".join(digest(x) for x in fields).encode()
        ).hexdigest()
        post(batch["action"].clone())
        physical.validate_tensor(decoder.last_unclipped_action)
        assert not bool(physical.clip(decoder.last_unclipped_action)[1].any())
        batch.pop("action")
        records.append(record)
        batches.append(batch)
    assert len({r["raw_image_sha256"] for r in records}) == 45
    assert len({r["nonvisual_sha256"] for r in records}) == 1
    assert len({r["raw_state_sha256"] for r in records}) == 1
    if mode == "samples":
        active = experiment["dataset"]["train_episodes"]
        train = n.LeRobotDataset(
            data_cfg.repo_id,
            root=view,
            episodes=active,
            revision=data_cfg.revision,
            delta_timestamps=n.resolve_delta_timestamps(cfg, meta),
            download_videos=False,
            return_uint8=True,
        )
        indices = resolve_visual_sample_indices(
            [(ep, 0) for ep in active],
            train.meta.episodes["dataset_from_index"],
            train.meta.episodes["dataset_to_index"],
            active,
            train.absolute_to_relative_idx,
        )
        for ep, index in zip(active, indices, strict=True):
            row = train[index]
            assert int(row["episode_index"]) == ep and int(row["frame_index"]) == 0
        save(
            output,
            {
                "status": "passed",
                "records": records,
                "actual_train_view_indices": indices,
                "actual_train_episodes": active,
                "action_contract_sha256": physical_sha,
                "normalization_sha256": n.file_sha256(norm_path),
                "model_source_sha256": n.file_sha256(source / "model.safetensors"),
                "hidden_test_loaded": False,
                "optimizer_steps": 0,
            },
        )
    elif mode == "predict":
        # A two-step smoke must preserve complete chunks through a separate reload.
        policy.eval()
        predictions, standards = [], []
        for seed in (None, 20260905, 20260906, 20260907):
            noise = (
                torch.zeros((1, 50, 32))
                if seed is None
                else torch.randn((1, 50, 32), generator=torch.Generator().manual_seed(seed))
            )
            for batch in batches:
                policy.reset()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = policy.predict_action_chunk(batch, noise=noise.cuda().clone())
                    post(prediction.clone())
                standard = decoder.last_unclipped_action
                assert torch.isfinite(prediction).all() and torch.isfinite(standard).all()
                physical.validate_tensor(standard)
                assert not bool(physical.clip(standard)[1].any())
                predictions.append(array(prediction))
                standards.append(array(standard))
        with Path(output).open("xb") as stream:
            np.savez(stream, normalized=np.asarray(predictions), standard=np.asarray(standards))
        save(
            str(output) + ".json",
            {
                "status": "passed",
                "pid": os.getpid(),
                "records": records,
                "checkpoint_sha256": n.file_sha256(source / "model.safetensors"),
                "optimizer_steps": 0,
            },
        )
    else:
        raise ValueError("Unknown input check mode")


if __name__ == "__main__":
    main()
