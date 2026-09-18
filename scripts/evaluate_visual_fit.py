"""Collect native full-chunk B/C evidence under the Hestia fit-strength contract.

Model execution requires a separately authorized, sealed execution contract.
The historical A/B collector and its fixed 256-step guard remain unchanged.
Heavy imports occur only after authorization, identity and deadline checks.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import resource
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from rosetta_reality.vla import visual_fit_contract as contract_checks  # noqa: E402
from rosetta_reality.vla.visual_coverage import action_groups  # noqa: E402
from rosetta_reality.vla.visual_fit import (  # noqa: E402
    FORMULA_PROTOCOL,
    PROTOCOL,
    array_hash,
    compare_arms,
    compare_reload,
    file_hash,
    read_bundle,
    summarize_bundle,
    write_bundle,
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def repository_file(name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Execution identity must use repository-relative paths.")
    path = ROOT / relative
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError("Execution identity escaped the workspace.")
    return path


def collect(contract_path: Path, arm: str, output: Path) -> dict:
    contract = json.loads(contract_path.read_text())
    contract_checks.check_execution(ROOT, contract)
    if arm not in {"B", "C"}:
        raise ValueError("Only the registered B/C arms may be collected")
    if output.exists():
        raise FileExistsError("Prediction output already exists.")
    # These imports are intentionally unreachable from a draft or --help.
    import evaluate_visual_native_small as native
    import numpy as np
    import torch

    from rosetta_reality.vla.processor import PiAlohaPostprocessorStep

    if not torch.cuda.is_available():
        raise RuntimeError("The registered CUDA worker has no visible device.")
    limits = contract["resource_limits"]

    def budget_check():
        if time.time() >= contract["deadline_unix"]:
            raise TimeoutError("Shared execution deadline reached.")
        if (
            torch.cuda.max_memory_allocated() > limits["cuda_allocated_bytes"]
            or torch.cuda.max_memory_reserved() > limits["cuda_reserved_bytes"]
            or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > limits["host_rss_bytes"]
        ):
            raise RuntimeError("Registered memory limit exceeded.")

    review = json.loads(repository_file(contract["review_plan"]["path"]).read_text())
    data = review["data"]
    views = {
        "train8": data["train40"][:8],
        "train40": data["train40"],
        "dev5": data["development_validation5"],
    }
    episodes = views["train40"] + views["dev5"]
    if len(episodes) != 45 or set(episodes) & set(data["sealed_hidden5"]):
        raise ValueError("Registered scene boundary changed.")
    identity = contract["arms"][arm]
    plan_path = repository_file(identity["plan"]["path"])
    if file_hash(plan_path) != identity["plan"]["sha256"]:
        raise ValueError("Arm plan identity drift.")
    plan, parent, experiment = native._resolve_plan(plan_path)
    actual_revisions = contract_checks.validate_experiment(experiment)
    contract_path_physical = repository_file(experiment["action_contract"]["derived"])
    physical_sha = file_hash(contract_path_physical)
    native._validate_prerequisites(plan, experiment, parent, physical_sha)
    norm_path, _, _ = native._validate_normalization(plan, experiment, parent, physical_sha)
    norm = json.loads(norm_path.read_text())
    physical = native.load_action_contract(contract_path_physical)
    # The checkpoint folder and every consumed file must be sealed after training.
    checkpoint_relative = Path(identity["checkpoint_relative_to_root"])
    if checkpoint_relative.is_absolute() or ".." in checkpoint_relative.parts:
        raise ValueError("Unsafe checkpoint path.")
    source = contract_checks.relative_file(
        Path(os.environ["ROSETTA_CHECKPOINT_ROOT"]), checkpoint_relative.as_posix()
    )
    consumed = identity["checkpoint_files"]
    if not isinstance(consumed, dict) or not {
        "model.safetensors",
        "config.json",
        "train_config.json",
    } <= set(consumed):
        raise ValueError("Checkpoint has not been sealed.")
    # Include processor JSON and state files, not only the model tensor file.
    actual = {p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file()}
    if actual != set(consumed):
        raise ValueError("Checkpoint file inventory differs from the sealed identity.")
    for name, digest in consumed.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or file_hash(source / name) != digest:
            raise ValueError("Checkpoint file identity drift.")
    saved = json.loads((source / "train_config.json").read_text())
    recipe = contract_checks.validate_saved_recipe(saved, plan, arm, identity)
    contract_checks.verify_arm_evidence(ROOT, contract, arm, source)
    processor_sha = contract_checks.processor_identity(consumed)
    identity = {**identity, "training_recipe": recipe}
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
        raise ValueError("Native inference shape/precision/representation contract changed.")
    # This helper filters all raw row access to the non-hidden split first.
    context = native.load_frame_zero_context(ROOT, "non_hidden")
    data_cfg = context["config"]
    if data_cfg.revision != actual_revisions["data_revision"]:
        raise ValueError("Actual dataset cache revision differs from the pinned experiment")
    meta = native.LeRobotDatasetMetadata(
        data_cfg.repo_id, root=context["root"], revision=data_cfg.revision
    )
    dataset = native.LeRobotDataset(
        data_cfg.repo_id,
        root=context["root"],
        episodes=episodes,
        revision=data_cfg.revision,
        delta_timestamps=native.resolve_delta_timestamps(cfg, meta),
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
    policy = native.make_policy(
        cfg=cfg, ds_meta=dataset.meta, rename_map=experiment["dataset"]["rename_map"]
    )
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
    decoders = [step for step in post.steps if isinstance(step, PiAlohaPostprocessorStep)]
    if len(decoders) != 1:
        raise ValueError("Cannot observe the standard action before safety projection.")
    decoder = decoders[0]
    dimensions = [
        {
            "name": d.name,
            "unit": d.unit,
            "encoding": d.encoding,
            "minimum": d.minimum,
            "maximum": d.maximum,
        }
        for d in physical.dimensions
    ]
    groups = action_groups(dimensions)
    if len(dimensions) != 14 or groups["gripper_normalized"] != [6, 13]:
        raise ValueError("Physical action group identity changed.")

    def cpu(value):
        return value.detach().cpu().double().numpy()

    def tensor_digest(value):
        return hashlib.sha256(
            str(value.dtype).encode() + array_hash(cpu(value)).encode()
        ).hexdigest()

    starts = dict(
        zip(
            dataset.meta.episodes["episode_index"],
            dataset.meta.episodes["dataset_from_index"],
            strict=True,
        )
    )
    batches, targets, standards, raw_images = [], [], [], []
    if len(dataset.meta.camera_keys) != 1:
        raise ValueError("Registered dataset must supply exactly one real camera.")
    actual_camera = experiment["dataset"]["rename_map"].get(
        dataset.meta.camera_keys[0], dataset.meta.camera_keys[0]
    )
    required_keys = {
        "observation.state",
        "observation.language.tokens",
        "observation.language.attention_mask",
        actual_camera,
    }
    for ep in episodes:
        budget_check()
        sample = dataset[dataset.absolute_to_relative_idx[int(starts[ep])]]
        if (
            int(sample["episode_index"]) != ep
            or int(sample["frame_index"]) != 0
            or bool(sample["action_is_pad"].any())
        ):
            raise ValueError("Frame-zero identity or full target horizon mismatch.")
        batch = native.default_collate([sample])
        image = batch[dataset.meta.camera_keys[0]]
        if image.dtype != torch.uint8 or image.ndim not in (4, 5):
            raise ValueError("Raw camera ingress dtype/shape drift.")
        raw_images.append(tensor_digest(image))
        batch[dataset.meta.camera_keys[0]] = image.float() / 255
        batch = pre(batch)
        targets.append(cpu(batch["action"][0]))
        post(batch["action"].clone())
        standards.append(cpu(decoder.last_unclipped_action[0]))
        keys = required_keys | {k for k in batch if k.endswith("_padding_mask")}
        if not required_keys <= set(batch):
            raise ValueError("Processed policy inputs are incomplete.")
        if [key for key in policy.config.image_features if key in batch] != [actual_camera]:
            raise ValueError("Actual image feature identity differs from the registered camera.")
        if batch["observation.state"].shape[-1] != 14:
            raise ValueError("The actual state must have 14 dimensions; config shape is not proof.")
        batches.append({k: batch[k] for k in sorted(keys)})
    if len(set(raw_images)) != len(episodes):
        raise ValueError("Duplicate selected raw images.")
    noises = [
        torch.zeros((1, 50, 32))
        if seed is None
        else torch.randn((1, 50, 32), generator=torch.Generator().manual_seed(seed))
        for seed in (None, 20260905, 20260906, 20260907)
    ]
    traces = []
    original = policy.model.sample_actions

    def traced_sample(images, masks, tokens, attention, state, noise=None, **kwargs):
        # Observe arguments of the actual native decoder call, after queue/preparation.
        if kwargs or len(images) != 3 or len(masks) != 3:
            raise ValueError("Native inference arguments/camera count changed.")
        if any(mask.dtype != torch.bool for mask in masks):
            raise ValueError("Native camera masks must be Boolean.")
        if state.shape != (1, 32) or not torch.isfinite(state).all():
            raise ValueError("Actual native state padding or values are invalid.")
        if (
            not bool(masks[0].all())
            or any(bool(x.any()) for x in masks[1:])
            or any(not bool((x == -1).all()) for x in images[1:])
        ):
            raise ValueError("Real/empty camera mask or placeholder contract changed.")
        if not torch.isfinite(images[0]).all() or images[0].min() < -1 or images[0].max() > 1:
            raise ValueError("Prepared real image range is invalid.")
        fields = [state, tokens, attention, *masks, *images[1:]]
        traces.append(
            (
                tensor_digest(images[0]),
                hashlib.sha256("|".join(tensor_digest(x) for x in fields).encode()).hexdigest(),
                tensor_digest(noise),
            )
        )
        if traces[-1][1] != traces[0][1]:
            raise ValueError("Actual nonvisual conditioning differs across images.")
        return original(images, masks, tokens, attention, state, noise=noise)

    policy.model.sample_actions = traced_sample
    policy.eval()
    torch.cuda.reset_peak_memory_stats()
    predictions, decoded, internal, value_dtypes = [], [], [], []
    try:
        for noise in noises:
            ps, ss, gs = [], [], []
            for batch in batches:
                budget_check()
                policy.reset()
                with (
                    torch.inference_mode(),
                    torch.autocast("cuda", dtype=torch.bfloat16),
                ):
                    pred = policy.predict_action_chunk(batch, noise=noise.cuda().clone())
                    post(pred.clone())
                standard = decoder.last_unclipped_action
                ps.append(cpu(pred[0]))
                ss.append(cpu(standard[0]))
                gs.append(cpu(decoder.last_model_action[0, :, groups["gripper_normalized"]]))
                value_dtypes.append([str(pred.dtype), str(standard.dtype)])
                if (
                    not torch.isfinite(pred).all()
                    or not torch.isfinite(standard).all()
                    or not torch.isfinite(decoder.last_model_action).all()
                ):
                    raise FloatingPointError("Nonfinite native prediction.")
                physical.validate_tensor(standard)
                _, illegal = physical.clip(standard)
                if bool(illegal.any()):
                    raise ValueError("Predicted standard action required safety clipping.")
            predictions.append(ps)
            decoded.append(ss)
            internal.append(gs)
    except Exception:
        partial = output.parent / f"{output.name}.partial-{uuid.uuid4().hex}"
        partial.mkdir(parents=True, exist_ok=False)
        for name, value in {
            "completed_normalized": predictions,
            "completed_standard": decoded,
            "completed_internal_grippers": internal,
            "current_normalized": ps,
            "current_standard": ss,
            "current_internal_grippers": gs,
            "normalized_targets": targets,
            "standard_targets": standards,
            "noise": [n.numpy() for n in noises],
        }.items():
            with (partial / f"{name}.npy").open("xb") as stream:
                np.save(stream, np.asarray(value), allow_pickle=False)
        write_json(
            partial / "status.json",
            {
                "status": "incomplete_not_scored",
                "completed_noise_conditions": len(predictions),
                "current_predictions": len(ps),
                "native_input_traces": traces,
                "arm": arm,
                "arm_identity": identity,
                "hidden_test_loaded": False,
                "m2_complete": False,
            },
        )
        raise
    finally:
        policy.model.sample_actions = original
    budget_check()
    if len(traces) != 180 or any(traces[i][:2] != traces[i % 45][:2] for i in range(180)):
        raise ValueError("Native input trace count or per-noise conditioning drift.")
    for condition, noise in enumerate(noises):
        if any(t[2] != tensor_digest(noise) for t in traces[condition * 45 : (condition + 1) * 45]):
            raise ValueError("Native decoder did not consume the exact fixed noise.")
    arrays = {
        "normalized_predictions": np.asarray(predictions),
        "standard_predictions": np.asarray(decoded),
        "internal_grippers": np.asarray(internal),
        "normalized_targets": np.asarray(targets),
        "standard_targets": np.asarray(standards),
        "valid_mask": np.ones((45, 50, 14), dtype=bool),
        "noise": np.stack([n.numpy() for n in noises]),
    }
    metadata = {
        "protocol": PROTOCOL,
        "formula_protocol": FORMULA_PROTOCOL,
        "arm": arm,
        "episodes": episodes,
        "views": views,
        "hidden_episodes": data["sealed_hidden5"],
        "hidden_test_loaded": False,
        "target_conditioning": False,
        "native_denoising_steps": 10,
        "model_mode": "eval_inference",
        "chunk_length": 50,
        "max_action_dim": 32,
        "dimensions": dimensions,
        "noise_conditions": [None, 20260905, 20260906, 20260907],
        "noise_hashes": [array_hash(n) for n in arrays["noise"]],
        "nonvisual_hashes": [t[1] for t in traces[:45]],
        "image_hashes": [t[0] for t in traces[:45]],
        "raw_image_hashes": raw_images,
        "actual_camera_key": actual_camera,
        "native_prediction_dtypes": value_dtypes,
        "common_identity": {
            **actual_revisions,
            "processor_identity_sha256": processor_sha,
            "execution_contract_sha256": file_hash(contract_path),
            "physical_contract_sha256": physical_sha,
            "normalization_sha256": file_hash(norm_path),
            "noise_generation": "cpu_float32_then_cuda_no_cast",
            "inference_autocast": "bf16",
            "runtime_versions": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "numpy": np.__version__,
                "lerobot": importlib.metadata.version("lerobot"),
                "cuda": torch.version.cuda,
                "device_name": torch.cuda.get_device_name(),
            },
        },
        "arm_identity": identity,
        "process": {"pid": os.getpid(), "invocation_id": str(uuid.uuid4())},
    }
    write_bundle(output, arrays, metadata)
    result = summarize_bundle(arrays, metadata)
    result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    result["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
    write_json(output / "metrics.json", result)
    return {"status": "evidence_saved", "optimizer_steps": 0, "m2_complete": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--contract", type=Path, required=True)
    collect_parser.add_argument("--arm", choices=("B", "C"), required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    for mode in ("reload", "compare"):
        item = sub.add_parser(mode)
        item.add_argument("--first", type=Path, required=True)
        item.add_argument("--second", type=Path, required=True)
        item.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "collect":
        try:
            result = collect(args.contract, args.arm, args.output)
        except Exception as exc:
            # No host paths/tracebacks or credentials in durable publicable evidence.
            failure = args.output.parent / f"{args.output.name}.failure-{uuid.uuid4().hex}.json"
            write_json(
                failure,
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "optimizer_steps": 0,
                    "task_success": "not measured",
                    "m2_complete": False,
                },
            )
            raise
    else:
        if args.output.exists():
            raise FileExistsError("Comparison output already exists.")
        if args.mode == "reload":
            result = compare_reload(args.first, args.second)
        else:
            a, am = read_bundle(args.first)
            b, bm = read_bundle(args.second)
            result = compare_arms(a, am, b, bm)
        write_json(args.output, result)
        if args.mode == "reload" and not result["passed"]:
            raise SystemExit(1)
        if args.mode == "compare" and not result["offline_metric_criteria_passed"]:
            raise SystemExit(1)
    print(json.dumps(result if args.mode == "collect" else {"status": "comparison_saved"}))


if __name__ == "__main__":
    main()
