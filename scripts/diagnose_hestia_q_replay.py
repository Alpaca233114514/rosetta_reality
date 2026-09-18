"""Guarded same-checkpoint real-image K/V scene ablations."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from rosetta_reality.vla.visual_fit import array_hash, file_hash, read_bundle  # noqa: E402
from scripts.diagnose_zen_noise_transfer import parameter_digests  # noqa: E402
from scripts.hestia_q_replay import (  # noqa: E402
    CONDITIONS,
    MODES,
    Replay,
    condition_spec,
    validate_protocol,
)


def difference(actual, expected, atol, rtol):
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError("Prediction shape or finite contract differs")
    delta = np.abs(actual - expected)
    return {
        "passed": bool(np.all(delta <= atol + rtol * np.abs(expected))),
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "atol": atol,
        "rtol": rtol,
    }


def collect(plan, output):
    import evaluate_visual_native_small as native
    import torch

    from rosetta_reality.vla.processor import PiAlohaPostprocessorStep

    started = time.monotonic()
    if not torch.cuda.is_available() or os.environ.get("ROSETTA_TORCH_DEVICE") != "cuda":
        raise ValueError("Registered local CUDA unavailable; no fallback")
    torch.manual_seed(20260809)
    torch.cuda.reset_peak_memory_stats()

    def budget():
        if (
            time.monotonic() - started > plan["maximum_seconds"]
            or time.time() >= plan["deadline_unix"]
        ):
            raise TimeoutError("CUDA diagnostic deadline exceeded")
        if (
            torch.cuda.max_memory_allocated() > 10 * 1024**3
            or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 8 * 1024**3
        ):
            raise MemoryError("CUDA resource budget exceeded")

    old, meta = read_bundle(Path(plan["historical_bundle"]))
    reference = None
    if plan["reference_arrays"] is not None:
        with np.load(plan["reference_arrays"], allow_pickle=False) as saved:
            reference = {k: saved[k] for k in ("normalized_predictions", "standard_predictions")}
    context = native.load_frame_zero_context(ROOT, "non_hidden")
    if context["episodes"] != meta["episodes"] or context["episodes"] != plan["episodes"]:
        raise ValueError("Sample/split identity changed")
    if file_hash(context["root"] / "manifest.json") != plan["dataset_manifest_sha256"]:
        raise ValueError("Dataset identity changed")
    historical, parent, experiment = native._resolve_plan(Path(plan["historical_plan"]))
    physical_path = ROOT / experiment["action_contract"]["derived"]
    physical_sha = file_hash(physical_path)
    if physical_sha != meta["common_identity"]["physical_contract_sha256"]:
        raise ValueError("Action Contract differs from historical C")
    norm_path, _, _ = native._validate_normalization(historical, experiment, parent, physical_sha)
    if file_hash(norm_path) != meta["common_identity"]["normalization_sha256"]:
        raise ValueError("Historical normalization differs")
    norm = json.loads(norm_path.read_text())
    physical = native.load_action_contract(physical_path)
    source = Path(plan["checkpoint"])
    cfg = native.SmolVLAConfig.from_pretrained(source, local_files_only=True)
    cfg.device, cfg.pretrained_path, cfg.pretrained_revision = "cuda", source, None
    for section in ("policy", "adaptation"):
        for key, value in experiment["model"][section].items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
    cfg.load_vlm_weights = False
    if (
        cfg.num_steps != 10
        or cfg.chunk_size != 50
        or cfg.max_action_dim != 32
        or cfg.adapt_to_pi_aloha
        or cfg.use_amp
    ):
        raise ValueError("Native inference contract differs")
    dc = context["config"]
    ds_meta = native.LeRobotDatasetMetadata(dc.repo_id, root=context["root"], revision=dc.revision)
    dataset = native.LeRobotDataset(
        dc.repo_id,
        root=context["root"],
        episodes=plan["episodes"],
        revision=dc.revision,
        delta_timestamps=native.resolve_delta_timestamps(cfg, ds_meta),
        download_videos=False,
        return_uint8=True,
    )
    dataset.meta.stats.update(
        {
            key: {s: np.asarray(v) for s, v in values.items()}
            for key, values in norm["effective_stats"].items()
        }
    )
    policy = native.make_policy(
        cfg=cfg, ds_meta=dataset.meta, rename_map=experiment["dataset"]["rename_map"]
    )
    component = Replay(
        policy,
        plan["base_step"],
        Path(plan["scene_reference"]),
        output,
    )
    intervention = None
    policy.eval()
    if {p.device.type for p in policy.parameters()} != {"cuda"}:
        raise ValueError("Policy not fully on registered CUDA")
    pre, post = native._make_processors(
        SimpleNamespace(policy=cfg, rename_map=experiment["dataset"]["rename_map"]),
        policy,
        dataset,
        torch.device("cuda"),
    )
    native.ensure_smolvla_action_boundary(
        pre,
        post,
        physical,
        native.load_smolvla_action_space(experiment),
        action_contract_sha256=physical_sha,
        upstream_revision=experiment["upstream"]["revision"],
    )
    decoders = [s for s in post.steps if isinstance(s, PiAlohaPostprocessorStep)]
    if len(decoders) != 1:
        raise ValueError("Standard action observer unavailable")
    decoder = decoders[0]
    before = parameter_digests(policy)
    budget()

    def cpu(tensor):
        return tensor.detach().cpu().double().numpy()

    def tensor_hash(tensor):
        return hashlib.sha256(
            str(tensor.dtype).encode() + array_hash(cpu(tensor)).encode()
        ).hexdigest()

    starts = dict(
        zip(
            dataset.meta.episodes["episode_index"],
            dataset.meta.episodes["dataset_from_index"],
            strict=True,
        )
    )
    camera = meta["actual_camera_key"]
    required = {
        "observation.state",
        "observation.language.tokens",
        "observation.language.attention_mask",
        camera,
    }
    batches, ys, standards, image_hashes = [], [], [], []
    for index, episode in enumerate(plan["episodes"]):
        budget()
        sample = dataset[dataset.absolute_to_relative_idx[int(starts[episode])]]
        if (
            int(sample["episode_index"]) != episode
            or int(sample["frame_index"]) != 0
            or bool(sample["action_is_pad"].any())
        ):
            raise ValueError("Frame/chunk identity differs")
        batch = native.default_collate([sample])
        image = batch[dc.cameras["top"]]
        if image.dtype != torch.uint8 or tensor_hash(image) != meta["raw_image_hashes"][index]:
            raise ValueError("Raw image differs from historical C")
        image_hashes.append(tensor_hash(image))
        batch[dc.cameras["top"]] = image.float() / 255
        batch = pre(batch)
        ys.append(cpu(batch["action"][0]))
        post(batch["action"].clone())
        standards.append(cpu(decoder.last_unclipped_action[0]))
        keys = required | {key for key in batch if key.endswith("_padding_mask")}
        if not required <= set(batch) or batch["observation.state"].shape[-1] != 14:
            raise ValueError("Actual policy inputs differ")
        batches.append({key: batch[key] for key in sorted(keys)})
    targets, standard_targets = np.asarray(ys), np.asarray(standards)
    identities = {
        key: difference(value, old[key], 1e-6, 0)
        for key, value in (("normalized_targets", targets), ("standard_targets", standard_targets))
    }
    if not all(v["passed"] for v in identities.values()):
        raise ValueError("Processor targets do not reproduce historical C")
    noises = [torch.from_numpy(value.copy()) for value in old["noise"]]
    if any(
        array_hash(cpu(noise)) != array_hash(old["noise"][i].astype(np.float64))
        for i, noise in enumerate(noises)
    ):
        raise ValueError("Saved CPU noise changed")
    trace = []
    original = policy.model.sample_actions

    def observed(images, masks, tokens, attention, state, noise=None, **kwargs):
        if kwargs or len(images) != 3 or len(masks) != 3 or state.shape != (1, 32):
            raise ValueError("Native inference signature differs")
        if (
            not bool(masks[0].all())
            or any(bool(m.any()) for m in masks[1:])
            or any(not bool((im == -1).all()) for im in images[1:])
        ):
            raise ValueError("Real/empty camera contract differs")
        fields = [state, tokens, attention, *masks, *images[1:]]
        nonvisual = hashlib.sha256("|".join(tensor_hash(t) for t in fields).encode()).hexdigest()
        if trace and trace[0][1] != nonvisual:
            raise ValueError("Nonvisual conditioning varies across images")
        core = policy.model.vlm_with_expert
        vision = core.get_vlm_model().vision_model
        scale = core.get_vlm_model().connector.scale_factor
        patch = vision.config.patch_size
        if (
            cfg.add_image_special_tokens
            or cfg.prefix_length != 0
            or tuple(tokens.shape) != (1, 48)
            or attention.shape != tokens.shape
            or any(
                (im.shape[-2] // patch // scale, im.shape[-1] // patch // scale) != (8, 8)
                for im in images
            )
        ):
            raise ValueError("Registered 3x64 image, 48 language, 1 state prefix differs")
        prefix_mask = torch.cat(
            [
                *(mask[:, None].expand(1, 64) for mask in masks),
                attention.bool(),
                torch.ones((1, 1), dtype=torch.bool, device=state.device),
            ],
            dim=1,
        )
        component.set_layout(prefix_mask)
        trace.append([tensor_hash(images[0]), nonvisual, tensor_hash(noise)])
        return original(images, masks, tokens, attention, state, noise=noise)

    policy.model.sample_actions = observed
    predictions = {m: [] for m in MODES}
    decoded = {m: [] for m in MODES}
    reference_root = Path(plan["scene_reference"])
    with np.load(
        reference_root / f"base{plan['base_step']}_kmean/arrays.npz", allow_pickle=False
    ) as saved:
        k_reference = {k: saved[k] for k in ("normalized_predictions", "standard_predictions")}
    controls = 0
    executed = 0
    try:
        for ni, noise in enumerate(noises):
            ps, ss = {m: [] for m in MODES}, {m: [] for m in MODES}
            for bi, batch in enumerate(batches):
                for mode in MODES:
                    budget()
                    component.begin(bi, ni, mode)
                    policy.reset()
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        pred = policy.predict_action_chunk(batch, noise=noise.to("cuda").clone())
                        post(pred.clone())
                    executed += 1
                    component.end()
                    standard = decoder.last_unclipped_action
                    physical.validate_tensor(standard)
                    if not torch.isfinite(pred).all() or bool(physical.clip(standard)[1].any()):
                        raise ValueError("Nonfinite or clipped prediction")
                    p, s = cpu(pred[0]), cpu(standard[0])
                    if mode != "kmean_qnative":
                        ref = k_reference if mode == "kmean" else reference
                        differences = {
                            key: difference(value, ref[key][ni, bi], 0, 0)
                            for key, value in (
                                ("normalized_predictions", p),
                                ("standard_predictions", s),
                            )
                        }
                        if not all(d["passed"] for d in differences.values()):
                            with (output / "control-failure.npz").open("xb") as stream:
                                np.savez_compressed(
                                    stream,
                                    actual_normalized=p,
                                    actual_standard=s,
                                    expected_normalized=ref["normalized_predictions"][ni, bi],
                                    expected_standard=ref["standard_predictions"][ni, bi],
                                )
                            raise ValueError(
                                f"Exact control failed: {ni}/{bi}/{mode}: {differences}"
                            )
                        controls += 1
                    ps[mode].append(p)
                    ss[mode].append(s)
                if (bi + 1) % 5 == 0:
                    print(
                        json.dumps(
                            {
                                "base": plan["base_step"],
                                "noise": ni,
                                "images": bi + 1,
                                "forwards": executed,
                                "seconds": time.monotonic() - started,
                            }
                        ),
                        flush=True,
                    )
            for mode in MODES:
                predictions[mode].append(ps[mode])
                decoded[mode].append(ss[mode])
        intervention = component.finish()
    finally:
        policy.model.sample_actions = original
        component.close()
        with (output / "execution-count.json").open("x") as stream:
            json.dump(
                {
                    "policy_forwards": executed,
                    "exact_controls": controls,
                    "completed_by_mode": component.completed,
                    "optimizer_steps": 0,
                },
                stream,
                indent=2,
            )
    if before != parameter_digests(policy):
        raise ValueError("Model parameters changed during replay")
    if len(trace) != 1080:
        raise ValueError("Replay input trace count differs")
    for ni, noise in enumerate(noises):
        for row in range(45):
            records = trace[(ni * 45 + row) * 6 : (ni * 45 + row + 1) * 6]
            if any(r != records[0] for r in records):
                raise ValueError("Inputs changed across replay conditions")
            if records[0][2] != tensor_hash(noise):
                raise ValueError("Replay noise identity differs")
    arrays = {
        "normalized_targets": targets,
        "standard_targets": standard_targets,
        "noise": old["noise"],
    }
    for mode in MODES:
        arrays[f"{mode}_normalized_predictions"] = np.asarray(predictions[mode])
        arrays[f"{mode}_standard_predictions"] = np.asarray(decoded[mode])
    with (output / "arrays.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(output / "arrays.npz", allow_pickle=False) as saved:
        if any(not np.array_equal(v, saved[k]) for k, v in arrays.items()):
            raise ValueError("Saved arrays reload differs")
    result = {
        "id": plan["id"],
        "condition": plan["condition"],
        "base_step": plan["base_step"],
        "input_identity": identities,
        "all_parameters_unchanged": True,
        "parameter_tensors": len(before),
        "replay": intervention,
        "model_forwards": executed,
        "optimizer_steps": 0,
        "exact_control_forwards": controls,
        "same_device_control": {"passed": controls == 900},
        "seconds": time.monotonic() - started,
        "peak_cuda_allocated": torch.cuda.max_memory_allocated(),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "array_sha256": file_hash(output / "arrays.npz"),
        "hidden_loaded": False,
    }
    with (output / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result), flush=True)


def validate_permit(template, permit, condition, now):
    condition_spec(condition)
    if (
        permit.get("model_execution_authorized") is not True
        or permit.get("training_authorized") is not False
        or permit.get("shutdown_authorized") is not True
    ):
        raise ValueError("Explicit inference and shutdown authorization required")
    if permit.get("template_sha256") != template or permit.get("watchdog_active") is not True:
        raise ValueError("Template/watchdog differs")
    if permit.get("allowed_conditions") != list(CONDITIONS):
        raise ValueError("Frozen condition order differs")
    start, end = permit.get("started_unix", 0), permit.get("deadline_unix", 0)
    if not 0 < end - start <= 4200 or not start <= now < end:
        raise ValueError("Missing or expired shared deadline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--permit", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    args = parser.parse_args()
    plan = json.loads(args.template.read_text(encoding="utf-8-sig"))
    validate_protocol(plan)
    if plan["id"] != "hestia-q-replay-20260913-001":
        raise ValueError("Unregistered intervention template")
    permit = json.loads(args.permit.read_text())
    validate_permit(file_hash(args.template), permit, args.condition, time.time())
    ticks = Path(f"/proc/{permit['watchdog_pid']}/stat").read_text().rsplit(")", 1)[1].split()[19]
    if ticks != permit["watchdog_ticks"]:
        raise ValueError("Watchdog identity changed")
    job = Path(plan["output"])
    if job.is_absolute() or ".." in job.parts:
        raise ValueError("Job path escaped")
    own = Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]
    active = json.loads((job / "active-child.json").read_text())
    if active != {"pid": os.getpid(), "ticks": own, "condition": args.condition}:
        raise ValueError("Collector PID identity not registered")
    profile = os.environ.get("ROSETTA_AUTODL_RUNTIME_PROFILE")
    if (
        not profile
        or file_hash(Path(profile)) != plan["profile_sha256"]
        or os.environ.get("ROSETTA_TORCH_DEVICE") != "cuda"
    ):
        raise ValueError("Registered CUDA profile required")
    if any(os.environ.get(k) != "1" for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")):
        raise ValueError("Offline execution required")
    for name, sha in plan["sha256"].items():
        if file_hash(Path(name)) != sha:
            raise ValueError(f"Input/source drift: {name}")
    installed = Path(importlib.metadata.distribution("lerobot").locate_file("lerobot"))
    for name, sha in plan["upstream_files"].items():
        if file_hash(installed / name) != sha:
            raise ValueError("Upstream changed")

    checkpoint_root = Path(os.environ["ROSETTA_CHECKPOINT_ROOT"]).resolve(strict=True)
    sources = {}
    for step in (640, 1280):
        source = checkpoint_root / plan["checkpoint_prefix"] / f"{step:06d}" / "pretrained_model"
        if not source.resolve(strict=True).is_relative_to(checkpoint_root):
            raise ValueError("Checkpoint escaped")
        for name, sha in plan["checkpoints"][str(step)].items():
            path = source / name
            if (
                path.is_symlink()
                or not path.resolve(strict=True).is_relative_to(source.resolve())
                or file_hash(path) != sha
            ):
                raise ValueError("Checkpoint file identity differs")
        sources[step] = source
    audit = json.loads(Path(plan["parameter_audit"]).read_text())
    if audit.get("frozen_vlm_exact") is not True or audit.get("tensor_count") != 500:
        raise ValueError("Frozen backbone pair audit required")
    base, mode = condition_spec(args.condition)
    plan.update(
        condition=args.condition,
        base_step=base,
        checkpoint=str(sources[base]),
        reference_arrays=plan["reference_endpoints"][str(base)],
        deadline_unix=permit["deadline_unix"],
        maximum_seconds=2100,
        normalized_atol=0.0,
        standard_atol=0.0,
        rtol=0.0,
        metric_relative_tolerance=0.0,
    )
    import torch

    expected = plan["historical_runtime"]
    if (
        torch.__version__ != expected["torch"]
        or np.__version__ != expected["numpy"]
        or importlib.metadata.version("lerobot") != expected["lerobot"]
    ):
        raise ValueError("Original CUDA runtime required")
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or torch.cuda.get_device_name(0) != expected["device_name"]
    ):
        raise ValueError("Original GPU model required")
    output = job / args.condition
    output.mkdir(parents=True, exist_ok=False)
    try:
        collect(plan, output)
    except Exception as error:
        with (output / "failure.json").open("x") as stream:
            json.dump(
                {"error_type": type(error).__name__, "message": str(error), "optimizer_steps": 0},
                stream,
                indent=2,
            )
        raise


if __name__ == "__main__":
    main()
