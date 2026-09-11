"""Repeated local BF16/FP32 observations of one unchanged Hestia sample."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
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
from scripts.diagnose_hestia_local_checkpoint import difference  # noqa: E402
from scripts.diagnose_zen_noise_transfer import parameter_digests  # noqa: E402


def run(plan, output):
    import evaluate_visual_native_small as native
    import torch

    from rosetta_reality.vla.processor import PiAlohaPostprocessorStep

    started = time.monotonic()
    if not torch.xpu.is_available() or os.environ.get("ROSETTA_TORCH_DEVICE") != "xpu":
        raise ValueError("Registered XPU unavailable")
    torch.manual_seed(20260809)
    torch.xpu.reset_peak_memory_stats()

    def budget():
        if (
            time.monotonic() - started > 600
            or torch.xpu.max_memory_allocated() > 4 * 1024**3
            or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 5 * 1024**3
        ):
            raise RuntimeError("Registered resource/deadline limit")

    old, meta = read_bundle(Path(plan["historical_bundle"]))
    with np.load(plan["failed_control"], allow_pickle=False) as data:
        failed = data["actual"].copy()
        if not np.array_equal(data["expected"], old["normalized_predictions"][0, 0]):
            raise ValueError("Prior failure reference differs")
    context = native.load_frame_zero_context(ROOT, "non_hidden")
    if context["episodes"] != meta["episodes"] or meta["episodes"][0] != 49:
        raise ValueError("Registered sample identity differs")
    if file_hash(context["root"] / "manifest.json") != plan["dataset_manifest_sha256"]:
        raise ValueError("Cache identity differs")
    prior, parent, experiment = native._resolve_plan(Path(plan["historical_plan"]))
    physical_path = ROOT / experiment["action_contract"]["derived"]
    physical_sha = file_hash(physical_path)
    norm_path, _, _ = native._validate_normalization(prior, experiment, parent, physical_sha)
    if (
        physical_sha != meta["common_identity"]["physical_contract_sha256"]
        or file_hash(norm_path) != meta["common_identity"]["normalization_sha256"]
    ):
        raise ValueError("Contract or normalization differs")
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
    if cfg.use_amp or cfg.num_steps != 10 or cfg.chunk_size != 50 or cfg.max_action_dim != 32:
        raise ValueError("Inference contract differs")
    dc = context["config"]
    dm = native.LeRobotDatasetMetadata(dc.repo_id, root=context["root"], revision=dc.revision)
    dataset = native.LeRobotDataset(
        dc.repo_id,
        root=context["root"],
        episodes=[49],
        revision=dc.revision,
        delta_timestamps=native.resolve_delta_timestamps(cfg, dm),
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
        raise ValueError("Unexpected parameter device")
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
        raise ValueError("Action observer differs")
    decoder = decoders[0]
    before = parameter_digests(policy)
    starts = dict(
        zip(
            dataset.meta.episodes["episode_index"],
            dataset.meta.episodes["dataset_from_index"],
            strict=True,
        )
    )
    sample = dataset[dataset.absolute_to_relative_idx[int(starts[49])]]
    if (
        int(sample["episode_index"]) != 49
        or int(sample["frame_index"]) != 0
        or bool(sample["action_is_pad"].any())
    ):
        raise ValueError("Sample differs")
    batch = native.default_collate([sample])
    image = batch[dc.cameras["top"]]
    raw_sha = hashlib.sha256(
        str(image.dtype).encode() + array_hash(image.double().numpy()).encode()
    ).hexdigest()
    if image.dtype != torch.uint8 or raw_sha != meta["raw_image_hashes"][0]:
        raise ValueError("Raw image differs")
    batch[dc.cameras["top"]] = image.float() / 255
    batch = pre(batch)
    norm_target = batch["action"][0].detach().cpu().double().numpy()
    post(batch["action"].clone())
    std_target = decoder.last_unclipped_action[0].detach().cpu().double().numpy()
    if (
        not difference(norm_target, old["normalized_targets"][0], 1e-6, 0)["passed"]
        or not difference(std_target, old["standard_targets"][0], 1e-6, 0)["passed"]
    ):
        raise ValueError("Target pipeline differs")
    required = {
        meta["actual_camera_key"],
        "observation.state",
        "observation.language.tokens",
        "observation.language.attention_mask",
    }
    keys = required | {k for k in batch if k.endswith("_padding_mask")}
    if not required <= set(batch):
        raise ValueError("Incomplete policy input")
    batch = {k: batch[k] for k in sorted(keys)}
    arrays, records = {}, {}
    count = 0
    for mode in ("bf16", "fp32"):
        values, standards = [], []
        for ni, noise_array in enumerate(old["noise"]):
            reps, stds = [], []
            for repetition in range(2):
                budget()
                policy.reset()
                amp = (
                    torch.autocast("xpu", dtype=torch.bfloat16)
                    if mode == "bf16"
                    else contextlib.nullcontext()
                )
                with torch.inference_mode(), amp:
                    pred = policy.predict_action_chunk(
                        batch, noise=torch.from_numpy(noise_array.copy()).to("xpu")
                    )
                    post(pred.clone())
                count += 1
                standard = decoder.last_unclipped_action
                if not torch.isfinite(pred).all() or bool(physical.clip(standard)[1].any()):
                    raise ValueError("Nonfinite or illegal action")
                value = pred[0].detach().cpu().double().numpy()
                if count == 1 and not np.array_equal(value, failed):
                    np.savez_compressed(
                        output / "prior-control-failure.npz", actual=value, prior=failed
                    )
                    raise ValueError("Prior failed XPU control itself did not repeat exactly")
                reps.append(value)
                stds.append(standard[0].detach().cpu().double().numpy())
            values.append(reps)
            standards.append(stds)
            if not np.array_equal(reps[0], reps[1]) or not np.array_equal(stds[0], stds[1]):
                raise ValueError("Same-process repeatability failed")
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "noise_index": ni,
                        "forwards": count,
                        "seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
        arrays[mode + "_normalized"] = np.asarray(values)
        arrays[mode + "_standard"] = np.asarray(standards)
        records[mode] = {
            str(seed): {
                "normalized_difference": difference(
                    values[i][0][..., :14],
                    old["normalized_predictions"][i, 0, ..., :14],
                    0.01,
                    0.01,
                ),
                "standard_difference": difference(
                    standards[i][0], old["standard_predictions"][i, 0], 0.005, 0.01
                ),
            }
            for i, seed in enumerate(meta["noise_conditions"])
        }
    after = parameter_digests(policy)
    if before != after:
        raise ValueError("Parameters changed")
    with (output / "arrays.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(output / "arrays.npz", allow_pickle=False) as loaded:
        if not all(np.array_equal(value, loaded[key]) for key, value in arrays.items()):
            raise ValueError("Array reload differs")
    result = {
        "status": "completed",
        "prior_xpu_failure_reproduced_exactly": True,
        "all_same_process_repeats_exact": True,
        "all_parameter_tensors_unchanged": len(before),
        "forwards": count,
        "optimizer_steps": 0,
        "episode": 49,
        "development_images_loaded": False,
        "hidden_loaded": False,
        "records": records,
        "seconds": time.monotonic() - started,
        "peak_xpu_allocated": torch.xpu.max_memory_allocated(),
        "array_sha256": file_hash(output / "arrays.npz"),
    }
    with (output / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--invocation", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8-sig"))
    if (
        os.environ.get("ROSETTA_CONTAINER_IMAGE_ID") != plan["image"]
        or os.environ.get("HF_HUB_OFFLINE") != "1"
    ):
        raise ValueError("Registered offline container required")
    for path, sha in plan["sha256"].items():
        if file_hash(Path(path)) != sha:
            raise ValueError(f"Input/source drift: {path}")
    output = Path(plan["output"]) / str(args.invocation)
    output.mkdir(parents=True, exist_ok=False)
    try:
        run(plan, output)
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
