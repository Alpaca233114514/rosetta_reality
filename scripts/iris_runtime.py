"""Shared, native CUDA input and output boundary for the registered Iris run."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
NOISES = (None, 20260905, 20260906, 20260907)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_checkpoint(source, identities, plan):
    """Reject an unsealed checkpoint before constructing a model or reading data."""
    import re

    source = Path(source).resolve(strict=True)
    required = {
        "config.json",
        "train_config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    }
    if not isinstance(identities, dict) or not required <= set(identities):
        raise ValueError("Checkpoint requires a complete registered file inventory")
    if not any(
        name.endswith("_normalizer_processor.safetensors") for name in identities
    ) or not any(name.endswith("_unnormalizer_processor.safetensors") for name in identities):
        raise ValueError("Checkpoint must retain both serialized normalization states")
    actual = {p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file()}
    if actual != set(identities):
        raise ValueError("Checkpoint file set differs from its seal")
    for name, expected in identities.items():
        relative = Path(name)
        path = source / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or path.is_symlink()
            or not path.resolve().is_relative_to(source)
            or not isinstance(expected, str)
            or re.fullmatch("[0-9a-f]{64}", expected) is None
            or sha(path) != expected
        ):
            raise ValueError("Checkpoint path or file hash changed")
    saved = json.loads((source / "train_config.json").read_text())
    active = plan["optimizer_smoke"]
    if (
        any(
            saved.get(key) != value
            for key, value in {
                "job_name": active["run_name"],
                "steps": active["steps"],
                "batch_size": active["batch_size"],
            }.items()
        )
        or saved.get("dataset", {}).get("episodes") != active["episodes"]
    ):
        raise ValueError("Checkpoint belongs to a different run or training stage")
    return source


def save(path, obj):
    with Path(path).open("x") as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)


def budget(deadline):
    import math

    import torch

    if type(deadline) not in (int, float) or not math.isfinite(deadline) or time.time() >= deadline:
        raise TimeoutError("Iris registered deadline exceeded")
    if (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 8 * 1024**3
        or torch.cuda.max_memory_allocated() > 8 * 1024**3
        or torch.cuda.max_memory_reserved() > 10 * 1024**3
    ):
        raise MemoryError("Iris registered memory budget exceeded")


def load_native(plan_path, *, checkpoint=None, checkpoint_files=None, split="train", deadline):
    import evaluate_visual_native_small as n
    import numpy as np
    import torch

    from rosetta_reality.vla.processor import PiAlohaPostprocessorStep

    if os.environ.get("ROSETTA_TORCH_DEVICE") != "cuda" or any(
        os.environ.get(key) != "1" for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")
    ):
        raise ValueError("Registered offline AutoDL runtime required")
    if split not in ("train", "non_hidden"):
        raise ValueError("Iris only permits registered train or non-hidden evaluation splits")
    budget(deadline)
    plan, parent, exp = n._resolve_plan(Path(plan_path))
    source = (
        verify_checkpoint(checkpoint, checkpoint_files, plan)
        if checkpoint is not None
        else n.phase_runner._model_root(exp)
    )
    source_sha = sha(source / "model.safetensors")
    physical_path = ROOT / exp["action_contract"]["derived"]
    physical_sha = sha(physical_path)
    n._validate_prerequisites(plan, exp, parent, physical_sha)
    norm_path, _, _ = n._validate_normalization(plan, exp, parent, physical_sha)
    norm = json.loads(norm_path.read_text())
    context = n.load_frame_zero_context(ROOT, split)
    episodes = context["episodes"]
    if split == "train" and episodes != plan["training"]["episodes"]:
        raise ValueError("Calibration train40 order changed")
    cfg = n.SmolVLAConfig.from_pretrained(source, local_files_only=True)
    cfg.device, cfg.pretrained_path, cfg.pretrained_revision = "cuda", source, None
    for section in ("policy", "adaptation"):
        for key, value in exp["model"][section].items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
    cfg.load_vlm_weights = False
    if cfg.num_steps != 10 or cfg.chunk_size != 50 or cfg.adapt_to_pi_aloha or cfg.use_amp:
        raise ValueError("Native policy contract differs")
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
            key: {stat: np.asarray(value) for stat, value in values.items()}
            for key, values in norm["effective_stats"].items()
        }
    )
    torch.manual_seed(20260809)
    policy = n.make_policy(cfg=cfg, ds_meta=dataset.meta, rename_map=exp["dataset"]["rename_map"])
    if checkpoint is None:
        pre, post = n._make_processors(
            SimpleNamespace(policy=cfg, rename_map=exp["dataset"]["rename_map"]),
            policy,
            dataset,
            torch.device("cuda"),
        )
    else:
        from lerobot.policies.factory import make_pre_post_processors

        pre, post = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=source,
            pretrained_revision=None,
            preprocessor_overrides={"device_processor": {"device": "cuda"}},
        )
    physical = n.load_action_contract(physical_path)
    n.ensure_smolvla_action_boundary(
        pre,
        post,
        physical,
        n.load_smolvla_action_space(exp),
        action_contract_sha256=physical_sha,
        upstream_revision=exp["upstream"]["revision"],
    )
    decoders = [step for step in post.steps if isinstance(step, PiAlohaPostprocessorStep)]
    if len(decoders) != 1:
        raise ValueError("Unique native action decoder required")
    starts = dict(
        zip(
            dataset.meta.episodes["episode_index"],
            dataset.meta.episodes["dataset_from_index"],
            strict=True,
        )
    )
    batches, targets, standards, image_hashes = [], [], [], []
    for ep in episodes:
        budget(deadline)
        sample = dataset[dataset.absolute_to_relative_idx[int(starts[ep])]]
        if (
            int(sample["episode_index"]) != ep
            or int(sample["frame_index"]) != 0
            or bool(sample["action_is_pad"].any())
        ):
            raise ValueError("Frame-zero nonpadded sample contract differs")
        batch = n.default_collate([sample])
        for key in dataset.meta.camera_keys:
            if batch[key].dtype != torch.uint8:
                raise ValueError("Raw image dtype differs")
            image_hashes.append(hashlib.sha256(batch[key].numpy().tobytes()).hexdigest())
            batch[key] = batch[key].float() / 255.0
        batch = pre(batch)
        if batches:
            for key in (
                "observation.state",
                "observation.language.tokens",
                "observation.language.attention_mask",
            ):
                if not torch.equal(batch[key], batches[0][key]):
                    raise ValueError("Nonvisual frame-zero inputs vary by scene")
        targets.append(batch["action"][0].detach().cpu().double().numpy())
        post(batch["action"].clone())
        standards.append(decoders[0].last_unclipped_action[0].detach().cpu().double().numpy())
        batches.append(batch)
    if len(set(image_hashes)) != len(episodes):
        raise ValueError("Distinct registered images required")
    policy.eval()
    budget(deadline)
    return SimpleNamespace(
        policy=policy,
        pre=pre,
        post=post,
        decoder=decoders[0],
        physical=physical,
        plan=plan,
        experiment=exp,
        episodes=episodes,
        batches=batches,
        targets=np.asarray(targets),
        standards=np.asarray(standards),
        source_sha256=source_sha,
        image_sha256=image_hashes,
        source=source,
        checkpoint_files=checkpoint_files,
        processor_verification=(
            "saved_checkpoint_state_no_statistics_override"
            if checkpoint is not None
            else "registered_base_statistics"
        ),
    )


def observations(batch):
    required = {
        "observation.state",
        "observation.language.tokens",
        "observation.language.attention_mask",
    }
    if not required <= set(batch):
        raise ValueError("Missing required policy observations")
    return {
        key: value
        for key, value in batch.items()
        if key.startswith("observation.") or key.endswith("_padding_mask")
    }


def verify_reload(first, second):
    """Require independent processes, matching input identities and all seven arrays."""
    import numpy as np

    first, second = Path(first), Path(second)
    metadata = [json.loads(Path(str(path) + ".json").read_text()) for path in (first, second)]
    keys = {
        "normalized_predictions",
        "standard_predictions",
        "normalized_targets",
        "standard_targets",
        "internal_grippers",
        "noise",
        "valid_mask",
    }
    for path, meta in zip((first, second), metadata, strict=True):
        if (
            meta.get("array_sha256") != sha(path)
            or meta.get("parameters_unchanged") is not True
            or meta.get("hidden_test_loaded") is not False
            or meta.get("processor_verification") != "saved_checkpoint_state_no_statistics_override"
        ):
            raise ValueError("Reload evidence integrity or saved processor state failed")
        if type(meta.get("collector_pid")) is not int or meta["collector_pid"] <= 0:
            raise ValueError("Reload collector process identity missing")
    if metadata[0]["collector_pid"] == metadata[1]["collector_pid"]:
        raise ValueError("Reload requires two independent collector processes")
    for key in ("source_sha256", "episodes", "image_sha256", "noise_seeds"):
        if key not in metadata[0] or metadata[0][key] != metadata[1].get(key):
            raise ValueError("Reload source or input identity differs")
    checked = 0
    with np.load(first, allow_pickle=False) as a, np.load(second, allow_pickle=False) as b:
        if set(a.files) != keys or set(b.files) != keys:
            raise ValueError("Independent reload requires all seven arrays")
        for key in sorted(keys):
            x, y = a[key], b[key]
            if (
                not x.size
                or x.shape != y.shape
                or x.dtype != y.dtype
                or not np.isfinite(x).all()
                or not np.isfinite(y).all()
                or not np.array_equal(x, y)
            ):
                raise ValueError("Independent reload array differs: " + key)
            checked += x.size
    return {
        "status": "passed",
        "exact_arrays": sorted(keys),
        "scalar_checks": checked,
        "independent_processes": True,
        "saved_processor_state_loaded": True,
        "maximum_absolute_difference": 0.0,
        "m2_complete": False,
    }


def collect_native(context, output, deadline):
    import numpy as np
    import torch

    from rosetta_reality.vla.visual_coverage import action_groups
    from scripts.diagnose_zen_noise_transfer import parameter_digests

    output = Path(output)
    if output.exists() or Path(str(output) + ".json").exists():
        raise FileExistsError("Evaluation arrays and metadata are create-only")
    if (
        context.policy.training
        or len(context.batches) != len(context.episodes)
        or len(set(context.episodes)) != len(context.episodes)
        or len(context.targets) != len(context.episodes)
    ):
        raise ValueError("Evaluation mode or complete episode coverage differs")
    before = parameter_digests(context.policy)
    dimensions = [
        {
            "name": d.name,
            "unit": d.unit,
            "encoding": d.encoding,
            "minimum": d.minimum,
            "maximum": d.maximum,
        }
        for d in context.physical.dimensions
    ]
    groups = action_groups(dimensions)
    predictions, standards, internals, noises = [], [], [], []
    for seed in NOISES:
        cfg = context.policy.config
        shape = (1, cfg.chunk_size, cfg.max_action_dim)
        noise = (
            torch.zeros(shape)
            if seed is None
            else torch.randn(shape, generator=torch.Generator().manual_seed(seed))
        ).cuda()
        noises.append(noise.cpu().numpy())
        ps, ss, ins = [], [], []
        for batch in context.batches:
            budget(deadline)
            context.policy.reset()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                pred = context.policy.predict_action_chunk(observations(batch), noise=noise.clone())
                context.post(pred.clone())
            standard = context.decoder.last_unclipped_action
            context.physical.validate_tensor(standard)
            if not all(
                bool(torch.isfinite(t).all())
                for t in (pred, standard, context.decoder.last_model_action)
            ) or bool(context.physical.clip(standard)[1].any()):
                raise ValueError("Nonfinite or clipped native action")
            ps.append(pred[0].cpu().double().numpy())
            ss.append(standard[0].cpu().double().numpy())
            ins.append(
                context.decoder.last_model_action[0, :, groups["gripper_normalized"]]
                .cpu()
                .double()
                .numpy()
            )
        predictions.append(ps)
        standards.append(ss)
        internals.append(ins)
    if parameter_digests(context.policy) != before:
        raise ValueError("Evaluation changed model parameters")
    if sha(context.source / "model.safetensors") != context.source_sha256:
        raise ValueError("Checkpoint weight file changed during evaluation")
    arrays = {
        "normalized_predictions": np.asarray(predictions),
        "standard_predictions": np.asarray(standards),
        "normalized_targets": context.targets,
        "standard_targets": context.standards,
        "internal_grippers": np.asarray(internals),
        "noise": np.asarray(noises),
        "valid_mask": np.ones(context.targets.shape, dtype=bool),
    }
    with Path(output).open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    save(
        str(output) + ".json",
        {
            "source_sha256": context.source_sha256,
            "array_sha256": sha(output),
            "episodes": context.episodes,
            "image_sha256": context.image_sha256,
            "noise_seeds": list(NOISES),
            "parameters_unchanged": True,
            "processor_verification": context.processor_verification,
            "independent_saved_processor_reload": False,
            "collector_pid": os.getpid(),
            "policy_forwards": 4 * len(context.episodes),
            "optimizer_steps": 0,
            "hidden_test_loaded": False,
            "m2_complete": False,
        },
    )
