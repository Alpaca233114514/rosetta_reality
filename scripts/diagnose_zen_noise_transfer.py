"""Bounded native Gaussian-noise comparison with an immutable zero-noise control."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from scripts.diagnose_kv_full_chunk import chunk_metrics
from scripts.diagnose_kv_regularization import digest


def make_noise(shape, seed):
    import torch

    if seed is None:
        return torch.zeros(shape)
    return torch.randn(shape, generator=torch.Generator(device="cpu").manual_seed(seed))


def parameter_digests(policy):
    import torch

    return {
        name: hashlib.sha256(
            p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        for name, p in policy.named_parameters()
    }


def collect(plan, output):
    import inspect

    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.smolvla import modeling_smolvla, smolvlm_with_expert

    from rosetta_reality.vla.vision_diagnostics import load_frame_zero_context
    from scripts.diagnose_frame0_vision_probe import INSTRUCTION, _load_artifact

    started = time.monotonic()
    if not torch.xpu.is_available() or os.environ.get("ROSETTA_TORCH_DEVICE") != "xpu":
        raise ValueError("Registered local XPU is unavailable.")
    device = torch.device("xpu")
    torch.manual_seed(plan["seed"])
    torch.xpu.reset_peak_memory_stats()
    for module in (modeling_smolvla, smolvlm_with_expert):
        path = Path(inspect.getfile(module))
        if digest(path) != plan["upstream_files"][path.name]:
            raise ValueError("Pinned native model source differs.")

    def budget():
        if time.monotonic() - started > plan["runtime"]["maximum_seconds"]:
            raise TimeoutError("Registered model diagnostic budget exceeded.")
        if torch.xpu.max_memory_allocated() > plan["runtime"]["maximum_allocated_bytes"]:
            raise MemoryError("Registered XPU memory budget exceeded.")

    context = load_frame_zero_context(Path.cwd(), "non_hidden")
    if context["episodes"] != plan["train_episodes"] + plan["development_episodes"]:
        raise ValueError("Registered episode order differs.")
    if digest(context["root"] / "manifest.json") != plan["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest differs.")
    if not np.array_equal(
        context["states"], np.broadcast_to(context["states"][0], context["states"].shape)
    ):
        raise ValueError("Frame-zero nonvisual inputs must be identical.")
    with np.load(plan["source_arrays"], allow_pickle=False) as saved:
        if not np.array_equal(saved["raw_targets"][:, 0], context["actions"]):
            raise ValueError("Raw target identity differs.")
        zero = saved["native_standard"].copy()
    reference = json.loads(Path(plan["input_preflight"]).read_text())
    cfg = context["config"]
    dataset = LeRobotDataset(
        cfg.repo_id,
        root=context["root"],
        episodes=context["episodes"],
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
    samples = []
    for index, episode in enumerate(context["episodes"]):
        sample = dataset[dataset.absolute_to_relative_idx[int(starts[episode])]]
        image, state = sample[cfg.cameras["top"]], sample[cfg.fields.state]
        if (
            int(sample[cfg.fields.episode_index]) != episode
            or int(sample[cfg.fields.frame_index]) != 0
            or image.dtype != torch.uint8
            or tuple(image.shape) != (3, 480, 640)
            or hashlib.sha256(image.contiguous().numpy().tobytes()).hexdigest()
            != reference["image_sha256"][index]
            or not np.array_equal(state.numpy(), context["states"][index])
            or not np.array_equal(sample[cfg.fields.action].numpy(), context["actions"][index])
            or sample["task"] != INSTRUCTION
        ):
            raise ValueError("Actual image/state/action/task identity differs.")
        samples.append((image.float() / 255, state))
        budget()
    (output / "input-preflight.json").write_text(
        json.dumps(
            {
                "passed": True,
                "samples": 45,
                "image_hashes_match": True,
                "hidden_rows_materialized": False,
            }
        ),
        encoding="utf-8",
    )
    artifact_root = (
        Path(os.environ["ROSETTA_ARTIFACT_ROOT"]) / plan["experiment_id"] / plan["artifact_id"]
    )
    if digest(artifact_root / "manifest.json") != plan["artifact_manifest_sha256"]:
        raise ValueError("Artifact manifest differs before weight load.")
    dependency = json.loads(Path(plan["vlm_dependency_manifest"]).read_text())
    if dependency["revision"] != plan["vlm_revision"] or dependency["status"] != "validated":
        raise ValueError("VLM dependency registration differs.")
    dependency_root = Path(os.environ["HF_HOME"]) / dependency["cache_layout"]
    for name, entry in dependency["files"].items():
        if digest(dependency_root / name) != entry["sha256"]:
            raise ValueError("VLM dependency file differs.")
    if (dependency_root.parents[1] / "refs/main").read_text().strip() != plan["vlm_revision"]:
        raise ValueError("Offline VLM ref differs.")
    policy, (preprocessor, postprocessor) = _load_artifact(plan["artifact_id"], device)
    if policy._rosetta_diagnostic_manifest_sha256 != plan["artifact_manifest_sha256"]:
        raise ValueError("Loaded artifact identity differs.")
    before = parameter_digests(policy)
    shape = (1, policy.config.chunk_size, policy.config.max_action_dim)
    if shape != (1, 50, 32):
        raise ValueError("Registered native noise shape differs.")
    collected = {str(seed): [] for seed in plan["noise_seeds"]}
    calls = 0

    def predict(batch, seed):
        nonlocal calls
        if calls >= plan["runtime"]["maximum_forward_calls"]:
            raise RuntimeError("Maximum forward calls reached.")
        budget()
        noise = make_noise(shape, seed).to(device=device, dtype=batch["observation.state"].dtype)
        policy.reset()
        with torch.inference_mode(), torch.autocast(device_type="xpu", dtype=torch.bfloat16):
            prediction = postprocessor(policy.predict_action_chunk(batch, noise=noise))
        calls += 1
        values = prediction.detach().float().cpu().numpy()[0]
        if values.shape != (50, 14) or not np.isfinite(values).all():
            raise ValueError("Native action shape/finite check failed.")
        budget()
        return values

    parity = False
    for index, (image, state) in enumerate(samples):
        batch = preprocessor(
            {cfg.cameras["top"]: image, "observation.state": state, "task": INSTRUCTION}
        )
        _, masks = policy.prepare_images(batch)
        if [bool(mask.all()) for mask in masks] != [True, False, False]:
            raise ValueError("Native camera masks differ.")
        if index == 0:
            parity = np.array_equal(predict(batch, None), zero[0])
            if not parity:
                raise ValueError(
                    "Historical full-chunk zero-noise control did not reproduce exactly."
                )
            print("Full-chunk zero-noise reference reproduced exactly", flush=True)
        for seed in plan["noise_seeds"]:
            collected[str(seed)].append(predict(batch, seed))
        if (index + 1) % 5 == 0:
            print(
                json.dumps(
                    {
                        "samples_completed": index + 1,
                        "forward_calls": calls,
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    after = parameter_digests(policy)
    if before != after:
        raise ValueError("Model parameters changed during inference.")
    return {key: np.stack(value) for key, value in collected.items()}, {
        "model_forward_calls": calls,
        "first_sample_zero_full_chunk_exact": parity,
        "all_model_parameters_unchanged": True,
        "parameter_tensors": len(before),
        "parameter_digests": before,
        "elapsed_seconds": time.monotonic() - started,
        "peak_xpu_allocated_bytes": torch.xpu.max_memory_allocated(),
        "torch": torch.__version__,
        "device": torch.xpu.get_device_name(0),
    }


def main():
    from rosetta_reality.sim.action_contract import load_action_contract

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8-sig"))
    if (
        os.environ.get("ROSETTA_CONTAINER_IMAGE_ID") != plan["runtime"]["image"]
        or os.environ.get("HF_HUB_OFFLINE") != "1"
    ):
        raise ValueError("Registered offline container required.")
    for path, sha in plan["input_and_code_sha256"].items():
        if digest(path) != sha:
            raise ValueError(f"Identity drift: {path}")
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    predictions, verification = collect(plan, output)
    with np.load(plan["source_arrays"], allow_pickle=False) as saved:
        target, zero = saved["projected_targets"].copy(), saved["native_standard"].copy()
    with np.load(plan["frozen_bias_arrays"], allow_pickle=False) as saved:
        bias = saved["bias"].copy()
    predictions["zero"] = zero
    predictions["gaussian_ensemble"] = np.mean(
        [predictions[str(s)] for s in plan["noise_seeds"]], axis=0
    )
    contract = load_action_contract(Path(plan["action_contract"]))
    lower, upper = contract.lower_bounds.numpy(), contract.upper_bounds.numpy()
    groups = {}
    for d, dimension in enumerate(contract.dimensions):
        groups.setdefault(dimension.unit, []).append(d)
    windows = {
        "full": (0, 50),
        "first": (0, 1),
        "early": (0, 10),
        "middle": (10, 25),
        "late": (25, 50),
        "last": (49, 50),
    }
    metrics = {
        mode: {
            kind: {
                window: chunk_metrics(
                    np.clip(values + shift, lower, upper)[40:],
                    target[40:],
                    target[:40],
                    groups,
                    start,
                    stop,
                )
                for window, (start, stop) in windows.items()
            }
            for kind, shift in (("native", 0), ("frozen_zero_bias", bias))
        }
        for mode, values in predictions.items()
    }
    criteria = {}
    for group in groups:
        baseline = metrics["zero"]["native"]["full"][group]
        modes = [metrics[str(seed)]["native"]["full"][group] for seed in plan["noise_seeds"]]
        criteria[group] = {
            "all_gaussian_mean_bias_below_80_percent_of_zero": all(
                m["squared_mean_bias"] < 0.8 * baseline["squared_mean_bias"] for m in modes
            ),
            "mean_gaussian_mae_below_zero": float(np.mean([m["mae"] for m in modes]))
            < baseline["mae"],
            "mean_gaussian_first_mae_not_regressed": float(
                np.mean(
                    [
                        metrics[str(seed)]["native"]["first"][group]["mae"]
                        for seed in plan["noise_seeds"]
                    ]
                )
            )
            <= metrics["zero"]["native"]["first"][group]["mae"] + 1e-8,
        }
    with (output / "predictions.npz").open("xb") as stream:
        np.savez_compressed(stream, **predictions)
    with np.load(output / "predictions.npz", allow_pickle=False) as saved:
        if not all(np.array_equal(v, saved[k]) for k, v in predictions.items()):
            raise ValueError("Saved predictions differ after reload.")
    report = {
        "id": plan["id"],
        "status": "completed",
        "plan_sha256": digest(args.plan),
        "metrics": metrics,
        "criteria": criteria,
        "zero_specific_bias_supported": all(all(c.values()) for c in criteria.values()),
        "verification": verification,
        "arrays_sha256": digest(output / "predictions.npz"),
        "array_reload_exact": True,
        "hidden_rows_materialized": False,
        "policy_optimizer_created": False,
        "task_success": "not measured",
        "m2_complete": False,
    }
    with (output / "result.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "completed", "criteria": criteria}), flush=True)


if __name__ == "__main__":
    main()
