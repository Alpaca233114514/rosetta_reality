"""Registered local early/late KV-depth experiment on an unchanged SmolVLA artifact.

Reads inputs to the native expert K/V projections with observation-only hooks.
The only experimental axis is layer 1 versus layer 15 (zero based). No policy
optimizer, model download, simulation, checkpoint selection or deployment occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
for source in (REPOSITORY / "src", REPOSITORY / "scripts"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def experiment(plan: dict, output: Path) -> dict:
    import numpy as np
    import torch
    import yaml
    from diagnose_frame0_vision_probe import INSTRUCTION, _load_artifact
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.smolvla import modeling_smolvla, smolvlm_with_expert

    from rosetta_reality.vla.contextual_kv_probe import (
        compare_depths,
        observe_projection_input,
        visual_kv_features,
    )
    from rosetta_reality.vla.vision_diagnostics import load_frame_zero_context

    started = time.monotonic()
    if os.environ.get("ROSETTA_CONTAINER_IMAGE_ID") != plan["runtime"]["image"]:
        raise ValueError("Registered Linux container identity is required.")
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise ValueError("Offline Hub mode is required.")
    for module in (modeling_smolvla, smolvlm_with_expert):
        path = Path(inspect.getfile(module))
        if digest(path) != plan["upstream_files"][path.name]:
            raise ValueError(f"Pinned upstream implementation drift: {path.name}")
    device = torch.device(plan["runtime"]["accelerator"])
    if device.type != "xpu" or not torch.xpu.is_available():
        raise ValueError("Registered XPU is unavailable; no automatic device fallback.")
    if os.environ.get("ROSETTA_TORCH_DEVICE") != device.type:
        raise ValueError("Launcher and registered accelerator disagree.")
    torch.manual_seed(plan["seed"])
    torch.xpu.reset_peak_memory_stats()

    def budget() -> None:
        if time.monotonic() - started > plan["runtime"]["maximum_seconds"]:
            raise TimeoutError("Registered diagnostic time budget exceeded.")
        if torch.xpu.max_memory_allocated() > plan["runtime"]["maximum_allocated_bytes"]:
            raise MemoryError("Registered diagnostic XPU allocation budget exceeded.")

    context = load_frame_zero_context(REPOSITORY, "non_hidden")
    if digest(context["root"] / "manifest.json") != plan["dataset_manifest_sha256"]:
        raise ValueError("Prepared dataset manifest differs.")
    if (
        context["train"] != plan["train_episodes"]
        or context["validation"] != plan["development_episodes"]
    ):
        raise ValueError("Registered train/development episode order differs.")
    if not np.array_equal(
        context["states"], np.broadcast_to(context["states"][0], context["states"].shape)
    ):
        raise ValueError("Image intervention requires identical frame-zero states.")
    cfg = context["config"]
    if cfg.revision != plan["dataset_revision"]:
        raise ValueError("Dataset revision differs.")
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
    # Complete the real input preflight before any weight load.
    samples, image_hashes = [], []
    camera = cfg.cameras["top"]
    for index, episode in enumerate(context["episodes"]):
        sample = dataset[dataset.absolute_to_relative_idx[int(starts[episode])]]
        if (
            int(sample[cfg.fields.episode_index]) != episode
            or int(sample[cfg.fields.frame_index]) != 0
        ):
            raise ValueError("Dataset sample identity mismatch.")
        image, state = sample[camera], sample[cfg.fields.state]
        if image.ndim != 3 or image.shape[0] != 3 or image.dtype != torch.uint8:
            raise ValueError("Expected one uint8 RGB observation.")
        if not image.max() > image.min() or sample["task"] != INSTRUCTION:
            raise ValueError("Image/task input contract mismatch.")
        if not np.array_equal(state.numpy(), context["states"][index]):
            raise ValueError("State readers disagree.")
        if not np.array_equal(sample[cfg.fields.action].numpy(), context["actions"][index]):
            raise ValueError("Action readers disagree.")
        samples.append((image.float() / 255.0, state))
        image_hashes.append(hashlib.sha256(image.contiguous().numpy().tobytes()).hexdigest())
        budget()
    write_json(
        output / "input-preflight.json",
        {
            "passed": True,
            "episodes": context["episodes"],
            "frame_offset": 0,
            "image_sha256": image_hashes,
            "states_identical": True,
            "hidden_rows_materialized": False,
            "dataset_revision": cfg.revision,
        },
    )
    artifact_root = Path(os.environ["ROSETTA_ARTIFACT_ROOT"]) / plan["experiment_id"]
    manifest_path = artifact_root / plan["artifact_id"] / "manifest.json"
    if digest(manifest_path) != plan["artifact_manifest_sha256"]:
        raise ValueError("Registered artifact manifest differs before weight loading.")
    dependency = json.loads((REPOSITORY / plan["vlm_dependency_manifest"]).read_text())
    if dependency["revision"] != plan["vlm_revision"] or dependency["status"] != "validated":
        raise ValueError("VLM dependency identity differs.")
    dependency_root = Path(os.environ["HF_HOME"]) / dependency["cache_layout"]
    for name, entry in dependency["files"].items():
        if digest(dependency_root / name) != entry["sha256"]:
            raise ValueError(f"VLM dependency file differs: {name}")
    if (dependency_root.parents[1] / "refs/main").read_text().strip() != plan["vlm_revision"]:
        raise ValueError("Offline VLM main ref differs from pinned revision.")
    policy, (preprocessor, postprocessor) = _load_artifact(plan["artifact_id"], device)
    if policy._rosetta_diagnostic_manifest_sha256 != plan["artifact_manifest_sha256"]:
        raise ValueError("Artifact identity differs from registration.")
    core = policy.model.vlm_with_expert
    if any(parameter.requires_grad for parameter in core.vlm.parameters()):
        raise ValueError("VLM must be frozen.")
    if policy.config.add_image_special_tokens or policy.config.attention_mode != "cross_attn":
        raise ValueError("Registered native prefix layout requires plain cross-attention.")
    layer_indices = plan["layer_indices"]
    if core.num_vlm_layers != 16 or core.num_expert_layers != core.num_vlm_layers:
        raise ValueError("Registered VLM/expert layer mapping differs.")
    if core.self_attn_every_n_layers != 2 or layer_indices != {"early": 1, "late": 15}:
        raise ValueError("Registered taps must be the first and last cross-attention layer.")
    model_hashes_before = {
        name: hashlib.sha256(
            parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        for name, parameter in core.vlm.named_parameters()
    }
    dimensions = yaml.safe_load((REPOSITORY / plan["action_contract"]).read_text())["action"][
        "dimensions"
    ]
    if len(dimensions) != context["actions"].shape[1]:
        raise ValueError("Action Contract width differs from dataset labels.")
    groups: dict[str, list[int]] = {}
    for index, item in enumerate(dimensions):
        groups.setdefault(item["unit"], []).append(index)
    if set(groups) != {"radian", "normalized"}:
        raise ValueError("Unexpected Action Contract physical units.")
    grid = tuple(plan["image_token_grid"])
    features: dict[str, list] = {"early": [], "late": []}
    standard_outputs, hook_counts = [], []
    no_hook_parity = False
    for index, (image, state) in enumerate(samples):
        budget()
        batch = preprocessor({camera: image, "observation.state": state, "task": INSTRUCTION})
        images, masks = policy.prepare_images(batch)
        if [bool(mask.all()) for mask in masks] != [True, False, False]:
            raise ValueError("Native real/empty camera mask contract differs.")
        vision = core.get_vlm_model().vision_model
        scale = core.get_vlm_model().connector.scale_factor
        patch = vision.config.patch_size
        height, width = images[0].shape[-2:]
        if (height // patch // scale, width // patch // scale) != grid:
            raise ValueError("Native image token grid differs.")
        noise = torch.zeros(
            (1, policy.config.chunk_size, policy.config.max_action_dim),
            device=device,
            dtype=batch["observation.state"].dtype,
        )

        def predict():
            policy.reset()
            with (
                torch.inference_mode(),
                torch.autocast(device_type=device.type, dtype=torch.bfloat16),
            ):
                return (
                    postprocessor(policy.predict_action_chunk(batch, noise=noise))
                    .detach()
                    .float()
                    .cpu()
                )

        reference = predict() if index == 0 else None
        captured, counts, handles = {}, {}, []

        try:
            for arm, layer in layer_indices.items():
                for kind in ("k", "v"):
                    module = getattr(core.lm_expert.layers[layer].self_attn, f"{kind}_proj")
                    hook = observe_projection_input(captured, counts, f"{arm}_{kind}")
                    handles.append(module.register_forward_pre_hook(hook))
            prediction = predict()
        finally:
            for handle in handles:
                handle.remove()
        expected = {
            f"{arm}_{kind}": policy.config.num_steps for arm in layer_indices for kind in ("k", "v")
        }
        if counts != expected:
            raise ValueError("Hooks did not observe every native denoising step.")
        if reference is not None:
            no_hook_parity = torch.equal(reference, prediction)
            if not no_hook_parity:
                raise ValueError("Observation hooks changed the full action chunk.")
        if not torch.isfinite(prediction).all():
            raise ValueError("Nonfinite standard action output.")
        for arm in layer_indices:
            features[arm].append(
                visual_kv_features(
                    captured[f"{arm}_k"],
                    captured[f"{arm}_v"],
                    grid=grid,
                    bins=plan["pooling_bins"],
                )
            )
        standard_outputs.append(prediction.numpy())
        hook_counts.append(counts)
        print(
            json.dumps({"episode": context["episodes"][index], "completed": index + 1}), flush=True
        )
    budget()
    model_hashes_after = {
        name: hashlib.sha256(
            parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        for name, parameter in core.vlm.named_parameters()
    }
    if model_hashes_after != model_hashes_before:
        raise ValueError("Frozen VLM parameters changed during observation.")
    early, late = np.stack(features["early"]), np.stack(features["late"])
    result = compare_depths(
        early,
        late,
        context["actions"],
        train_count=len(context["train"]),
        groups=groups,
        alphas=tuple(plan["alphas"]),
        seed=plan["seed"],
        epsilon=plan["epsilon"],
    )
    arrays = {
        "early": early,
        "late": late,
        "actions": context["actions"],
        "episodes": np.asarray(context["episodes"]),
        "standard_outputs": np.stack(standard_outputs),
    }
    with (output / "features.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(output / "features.npz", allow_pickle=False) as reloaded:
        if any(not np.array_equal(value, reloaded[name]) for name, value in arrays.items()):
            raise ValueError("Persisted diagnostic arrays failed exact reload.")
    return {
        "schema_version": 1,
        "id": plan["id"],
        "status": "completed",
        "single_axis": "native_expert_input_visual_kv_depth_1_vs_15",
        "artifact_id": plan["artifact_id"],
        "artifact_manifest_sha256": policy._rosetta_diagnostic_manifest_sha256,
        "dataset_revision": cfg.revision,
        "train_episodes": context["train"],
        "development_episodes": context["validation"],
        "hidden_rows_materialized": False,
        "layer_indices": layer_indices,
        "image_token_grid": grid,
        "pooling_bins": plan["pooling_bins"],
        "hook_counts": hook_counts,
        "hook_full_chunk_parity": no_hook_parity,
        "array_reload_exact": True,
        "vlm_parameters_unchanged": model_hashes_before == model_hashes_after,
        "vlm_parameter_tensors": len(model_hashes_before),
        "model_forward_calls": len(samples) + 1,
        "optimizer_created": False,
        "features_sha256": digest(output / "features.npz"),
        "results": result,
        "runtime": {
            "image": os.environ["ROSETTA_CONTAINER_IMAGE_ID"],
            "torch": torch.__version__,
            "device": str(device),
            "elapsed_seconds": time.monotonic() - started,
            "peak_xpu_allocated_bytes": torch.xpu.max_memory_allocated(),
        },
        "limitations": [
            "One unchanged Zen artifact; the coverage40 checkpoint is not available locally.",
            "Forty train frames and five previously opened development frames; first actions only.",
            "Shared alpha is selected on early-layer train folds; OOF scores remain tuning scores.",
            "Both taps use 2x2 pooling; readout failure does not prove information absence.",
            "Layer tap comparison does not modify the policy or establish a closed-loop repair.",
            "Exact array reload is not a new independent model-process reload.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    arguments = parser.parse_args()
    plan = json.loads(arguments.plan.read_text(encoding="utf-8"))
    for name, checksum in plan["implementation_files"].items():
        if digest(REPOSITORY / name) != checksum:
            raise ValueError(f"Registered implementation drift: {name}")
    output = REPOSITORY / plan["output_root"]
    if not output.resolve().is_relative_to((REPOSITORY / "runs").resolve()):
        raise ValueError("Diagnostic output must remain under runs/.")
    output.mkdir(parents=True, exist_ok=False)
    try:
        report = experiment(plan, output)
        report["plan_sha256"] = digest(arguments.plan)
        write_json(output / "result.json", report)
    except Exception as exc:
        # Keep the first failure; no alternate device, retry, or threshold change.
        write_json(
            output / "failure.json",
            {
                "status": "not measured",
                "error_type": type(exc).__name__,
                "error": str(exc).replace(str(REPOSITORY), "<repository>"),
                "plan_sha256": digest(arguments.plan),
                "m2_complete": False,
            },
        )
        raise
    print(json.dumps({"status": report["status"], "output": plan["output_root"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
