"""Bounded local native Hestia inference; no optimizer or remote execution."""

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
from scripts.diagnose_kv_full_chunk import chunk_metrics  # noqa: E402
from scripts.diagnose_zen_noise_transfer import parameter_digests  # noqa: E402


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
    if not torch.xpu.is_available() or os.environ.get("ROSETTA_TORCH_DEVICE") != "xpu":
        raise ValueError("Registered local XPU unavailable; no fallback")
    torch.manual_seed(20260809)
    torch.xpu.reset_peak_memory_stats()

    def budget():
        if time.monotonic() - started > plan["maximum_seconds"]:
            raise TimeoutError("Local diagnostic deadline exceeded")
        if (
            torch.xpu.max_memory_allocated() > 4 * 1024**3
            or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 5 * 1024**3
        ):
            raise MemoryError("Local resource budget exceeded")

    old, meta = read_bundle(Path(plan["historical_bundle"]))
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
    cfg.device, cfg.pretrained_path, cfg.pretrained_revision = "xpu", source, None
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
    policy.eval()
    if {p.device.type for p in policy.parameters()} != {"xpu"}:
        raise ValueError("Policy not fully on registered XPU")
    pre, post = native._make_processors(
        SimpleNamespace(policy=cfg, rename_map=experiment["dataset"]["rename_map"]),
        policy,
        dataset,
        torch.device("xpu"),
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
        trace.append([tensor_hash(images[0]), nonvisual, tensor_hash(noise)])
        return original(images, masks, tokens, attention, state, noise=noise)

    policy.model.sample_actions = observed
    predictions, decoded = [], []
    try:
        for ni, noise in enumerate(noises):
            ps, ss = [], []
            for bi, batch in enumerate(batches):
                budget()
                policy.reset()
                with torch.inference_mode(), torch.autocast("xpu", dtype=torch.bfloat16):
                    pred = policy.predict_action_chunk(batch, noise=noise.to("xpu").clone())
                    post(pred.clone())
                standard = decoder.last_unclipped_action
                physical.validate_tensor(standard)
                if not torch.isfinite(pred).all() or bool(physical.clip(standard)[1].any()):
                    raise ValueError("Nonfinite or clipped native prediction")
                ps.append(cpu(pred[0]))
                ss.append(cpu(standard[0]))
                if plan["step"] == 1280 and ni == 0 and bi == 0:
                    first = difference(
                        ps[-1][..., :14],
                        old["normalized_predictions"][0, 0, ..., :14],
                        plan["normalized_atol"],
                        plan["rtol"],
                    )
                    with (output / "first-control.json").open("x") as f:
                        json.dump(first, f, indent=2)
                    if not first["passed"]:
                        np.savez_compressed(
                            output / "first-control-failure.npz",
                            actual=ps[-1],
                            expected=old["normalized_predictions"][0, 0],
                        )
                        raise ValueError("First cross-device control failed frozen tolerance")
                if (bi + 1) % 15 == 0:
                    print(
                        json.dumps(
                            {
                                "noise_index": ni,
                                "images": bi + 1,
                                "seconds": time.monotonic() - started,
                            }
                        ),
                        flush=True,
                    )
            predictions.append(ps)
            decoded.append(ss)
    finally:
        policy.model.sample_actions = original
    if before != parameter_digests(policy):
        raise ValueError("Model parameters changed during inference")
    if len(trace) != 180 or any(trace[i][:2] != trace[i % 45][:2] for i in range(180)):
        raise ValueError("Native input trace parity failed")
    for ni, noise in enumerate(noises):
        if any(t[2] != tensor_hash(noise) for t in trace[ni * 45 : (ni + 1) * 45]):
            raise ValueError("Native noise differs")
    arrays = {
        "normalized_predictions": np.asarray(predictions),
        "standard_predictions": np.asarray(decoded),
        "normalized_targets": targets,
        "standard_targets": standard_targets,
        "noise": old["noise"],
    }
    with (output / "arrays.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(output / "arrays.npz", allow_pickle=False) as restored:
        if not all(np.array_equal(v, restored[k]) for k, v in arrays.items()):
            raise ValueError("Array reload differs")
    groups = {
        "joint": [i for i, d in enumerate(meta["dimensions"]) if d["unit"] == "radian"],
        "gripper": [i for i, d in enumerate(meta["dimensions"]) if d["unit"] != "radian"],
    }
    metrics = {
        str(seed): {
            w: chunk_metrics(
                arrays["standard_predictions"][i, 40:],
                standard_targets[40:],
                standard_targets[:40],
                groups,
                a,
                b,
            )
            for w, (a, b) in {"full": (0, 50), "first": (0, 1), "last": (49, 50)}.items()
        }
        for i, seed in enumerate(meta["noise_conditions"])
    }
    control = {}
    if plan["step"] == 1280:
        for key, atol in (
            ("normalized_predictions", plan["normalized_atol"]),
            ("standard_predictions", plan["standard_atol"]),
        ):
            control[key] = difference(arrays[key][..., :14], old[key][..., :14], atol, plan["rtol"])
        deltas = []
        for key, target in (
            ("normalized_predictions", targets),
            ("standard_predictions", standard_targets),
        ):
            for dims in groups.values():
                for a, b in ((0, 50), (0, 1)):
                    actual = np.abs(
                        arrays[key][:, 40:, a:b, dims] - target[None, 40:, a:b, dims]
                    ).mean(axis=(1, 2, 3))
                    expected = np.abs(
                        old[key][:, 40:, a:b, dims] - target[None, 40:, a:b, dims]
                    ).mean(axis=(1, 2, 3))
                    deltas.extend((np.abs(actual - expected) / np.maximum(expected, 1e-8)).tolist())
        control["maximum_relative_metric_change"] = max(deltas)
        control["passed"] = (
            all(
                v["passed"]
                for v in (control["normalized_predictions"], control["standard_predictions"])
            )
            and max(deltas) <= plan["metric_relative_tolerance"]
        )
    result = {
        "id": plan["id"],
        "step": plan["step"],
        "input_identity": identities,
        "raw_images_exact": True,
        "native_trace_count": len(trace),
        "all_parameters_unchanged": True,
        "parameter_tensors": len(before),
        "model_forwards": 180,
        "optimizer_steps": 0,
        "cross_device_control": control,
        "metrics": metrics,
        "seconds": time.monotonic() - started,
        "peak_xpu_allocated": torch.xpu.max_memory_allocated(),
        "torch": torch.__version__,
        "array_sha256": file_hash(output / "arrays.npz"),
        "hidden_loaded": False,
    }
    with (output / "result.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    if plan["step"] == 1280 and not control["passed"]:
        raise ValueError("Full cross-device parity failed; quarters remain stopped")
    print(
        json.dumps(
            {
                "status": "passed",
                "step": plan["step"],
                "control": control,
                "seconds": result["seconds"],
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8-sig"))
    if (
        os.environ.get("ROSETTA_CONTAINER_IMAGE_ID") != plan["image"]
        or os.environ.get("HF_HUB_OFFLINE") != "1"
    ):
        raise ValueError("Fixed offline container required")
    for name, expected in plan["sha256"].items():
        if file_hash(Path(name)) != expected:
            raise ValueError(f"Input/source drift: {name}")
    installed = Path(importlib.metadata.distribution("lerobot").locate_file("lerobot"))
    for name, expected in plan["upstream_files"].items():
        if file_hash(installed / name) != expected:
            raise ValueError("Upstream source drift")
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    try:
        collect(plan, output)
    except Exception as error:
        with (output / "failure.json").open("x") as f:
            json.dump(
                {"type": type(error).__name__, "message": str(error), "optimizer_steps": 0},
                f,
                indent=2,
            )
        raise


if __name__ == "__main__":
    main()
