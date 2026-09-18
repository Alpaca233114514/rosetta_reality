"""CUDA-only future collection behind an explicit permit and live watchdog.

The current no-card plan cannot call this module. No optimizer or checkpoint
selection is implemented here. Real CUDA validation remains a separate stage.
"""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path

from rosetta_reality.vla.visual_research import donors, require, seal_bundle, sha256, write_json


def collect(plan, output, root):
    # Authorization and lifetime checks precede torch imports or weight access.
    require(plan.get("gpu_authorized") is True, "No CUDA collection authorization")
    episodes = plan["train_episodes"] + plan["development_episodes"]
    require(
        len(plan["train_episodes"]) == 40
        and len(plan["development_episodes"]) == 5
        and len(set(episodes)) == 45
        and not set(episodes) & set(plan["hidden_episodes"]),
        "Collection episode protocol differs",
    )
    require(plan["frame_offsets"] == [0, 125, 250, 375], "Collection sampling grid differs")
    require(plan["checkpoint_step"] in (2500, 5000), "Unregistered checkpoint step")
    registration_path = root / plan["gpu_registration"]
    require(sha256(registration_path) == plan["gpu_registration_sha256"], "GPU registration drift")
    registration = json.loads(registration_path.read_text())
    require(registration["execution_authorized"] is True, "GPU execution permit missing")
    require(
        0 < registration["deadline"] - registration["started"] <= 3600,
        "Shared GPU window exceeds cap",
    )
    from scripts.run_iris_furnace import alive

    watchdog = registration["watchdog"]
    require(alive(watchdog["pid"], watchdog["ticks"]), "Independent watchdog absent")

    def budget():
        require(
            time.time() < registration["deadline"] - 300,
            "GPU collection must leave recovery reserve",
        )

    budget()
    import numpy as np
    import torch
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from torch.utils.data import default_collate

    from rosetta_reality.data.cache_resolver import resolve_prepared_cache
    from rosetta_reality.data.config import load_dataset_config
    from rosetta_reality.vla.image_scaling import canonical_rgb_uint8
    from scripts.diagnose_zen_noise_transfer import parameter_digests
    from scripts.iris_gate import load_gate_context
    from scripts.iris_runtime import budget as memory_budget
    from scripts.iris_runtime import observations

    require(torch.cuda.is_available(), "Actual CUDA runtime absent")
    source_root = Path(os.environ["ROSETTA_AUTODL_ROOT"]).resolve()
    source = (source_root / plan["checkpoint_relative"]).resolve()
    training_plan = (source_root / plan["training_plan_relative"]).resolve()
    require(
        source.is_relative_to(source_root) and training_plan.is_relative_to(source_root),
        "Source escaped durable root",
    )
    require(sha256(training_plan) == plan["training_plan_sha256"], "Training plan drift")
    context = load_gate_context(
        training_plan,
        checkpoint=source,
        checkpoint_files=plan["checkpoint_files"],
        split="non_hidden",
        deadline=registration["deadline"],
    )
    require(
        context.source_sha256 == plan["checkpoint_files"]["model.safetensors"], "Wrong checkpoint"
    )
    config = load_dataset_config(root / plan["dataset_config"])
    cache, _ = resolve_prepared_cache(config, root, validate_checksums=True)
    require(
        sha256(cache / "manifest.json") == plan["dataset_manifest_sha256"], "Dataset manifest drift"
    )
    require(
        sha256(cache / "cache_checksums.json") == plan["cache_checksums_sha256"],
        "Dataset content inventory drift",
    )
    policy = context.policy
    policy.eval()
    before = parameter_digests(policy)
    episodes = plan["train_episodes"] + plan["development_episodes"]
    metadata = LeRobotDatasetMetadata(config.repo_id, root=cache, revision=config.revision)
    dataset = LeRobotDataset(
        config.repo_id,
        root=cache,
        revision=config.revision,
        episodes=episodes,
        delta_timestamps=resolve_delta_timestamps(policy.config, metadata),
        return_uint8=True,
        download_videos=False,
    )
    starts = dict(
        zip(
            metadata.episodes["episode_index"], metadata.episodes["dataset_from_index"], strict=True
        )
    )
    ids = np.array([[e, f] for e in episodes for f in plan["frame_offsets"]], dtype=np.int64)
    schedule_path = source_root / plan["training_schedule_source"]
    require(sha256(schedule_path) == plan["training_schedule_sha256"], "Training schedule drift")
    schedule = json.loads(schedule_path.read_text())["sample_identities"]
    ingress_path = source_root / plan["training_ingress_source"]
    require(sha256(ingress_path) == plan["training_ingress_sha256"], "Training ingress drift")
    ingress = json.loads(ingress_path.read_text())
    require(
        [[r["episode"], r["frame"]] for r in ingress["records"]] == schedule
        and ingress["optimizer_steps"] == 5000, "Consumed training sequence differs"
    )
    seen_inputs = set(map(tuple, schedule[: plan["checkpoint_step"] * 4]))
    reference_inputs = set(map(tuple, schedule[:2500 * 4]))
    seen_at_checkpoint = np.array([tuple(pair) in seen_inputs for pair in ids])
    seen_by_step2500 = np.array([tuple(pair) in reference_inputs for pair in ids])
    batches, targets, raw_targets, normalized_targets, masks, input_records = [], [], [], [], [], []
    lower, upper = context.physical.lower_bounds, context.physical.upper_bounds
    for episode, frame in ids:
        budget()
        sample = dataset[dataset.absolute_to_relative_idx[int(starts[int(episode)]) + int(frame)]]
        require(
            int(sample["episode_index"]) == episode and int(sample["frame_index"]) == frame,
            "Actual sample identity drift",
        )
        raw_targets.append(sample["action"].numpy().copy())
        targets.append(sample["action"].maximum(lower).minimum(upper).numpy().copy())
        masks.append((~sample["action_is_pad"]).numpy().copy())
        batch = default_collate([sample])
        image = batch[config.cameras["top"]]
        require(image.dtype == torch.uint8, "Original RGB must be uint8")
        import hashlib

        input_records.append(
            {
                "episode": int(episode),
                "frame": int(frame),
                "raw_image_sha256": hashlib.sha256(image.numpy().tobytes()).hexdigest(),
            }
        )
        batch[config.cameras["top"]] = canonical_rgb_uint8(image)
        batch = context.pre(batch)
        normalized_targets.append(batch["action"][0].detach().cpu().numpy().copy())
        batches.append(observations(batch))
    require(np.asarray(masks).all(), "Main sampling grid contains padding")
    zero_rows = np.flatnonzero(ids[:, 1] == 0)
    reference = batches[int(zero_rows[0])]
    nonvisual = [key for key in reference if not key.startswith("observation.images")]
    for i in zero_rows:
        for key in nonvisual:
            a, b = batches[int(i)][key], reference[key]
            require(
                torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b,
                "Frame0 nonvisual inputs differ",
            )
    noise_shape = (1, policy.config.chunk_size, policy.config.max_action_dim)
    noises = []
    for seed in plan["noise_seeds"]:
        if seed is None:
            noise = torch.zeros(noise_shape, dtype=torch.float32)
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            noise = torch.randn(noise_shape, generator=generator, dtype=torch.float32)
        noises.append(noise)
    require(
        plan["noise_seeds"] == [None, 20260905, 20260906, 20260907],
        "Inference noise protocol drift",
    )
    shape = (len(noises), len(ids), context.physical.chunk_length, context.physical.dimension)
    correct, wrong, normalized, unprojected, normalized_wrong, unprojected_wrong = (
        np.empty(shape, dtype=np.float32) for _ in range(6)
    )
    donor_rows = donors(ids, plan["train_episodes"], "cyclic")
    calls = 0
    expected_forwards = 1268

    def predict(batch, noise):
        nonlocal calls
        budget()
        memory_budget(registration["deadline"] - 300)
        policy.reset()
        torch.cuda.synchronize()
        began = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            native = policy.predict_action_chunk(batch, noise=noise.to("cuda").clone())
            standard = context.post(native.clone())
        calls += 1
        if calls % 180 == 0:
            print(json.dumps({"completed_forwards": calls}), flush=True)
        result = [
            a[0].detach().float().cpu().numpy().copy()
            for a in (native, standard, context.decoder.last_unclipped_action)
        ]
        require(all(np.isfinite(a).all() for a in result), "Nonfinite policy output")
        if calls == 2:
            elapsed = time.perf_counter() - began
            projected_seconds = (expected_forwards - calls) * elapsed * 1.25
            remaining = registration["deadline"] - 300 - time.time()
            write_json(
                output / "throughput.json",
                {
                    "warmup_forwards": 1,
                    "measured_seconds": elapsed,
                    "remaining_projected_seconds_with_25pct_margin": projected_seconds,
                    "remaining_budget_after_reserve": remaining,
                },
            )
            require(projected_seconds < remaining, "Measured throughput exceeds GPU budget")
        return result

    controls = []
    for n, noise in enumerate(noises):
        for i, batch in enumerate(batches):
            normalized[n, i], correct[n, i], unprojected[n, i] = predict(batch, noise)
            if i in (0, len(ids) - 1):
                copied = {
                    k: v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
                    for k, v in batch.items()
                }
                repeat = predict(copied, noise)
                require(
                    all(
                        np.array_equal(a, b)
                        for a, b in zip(
                            repeat,
                            (normalized[n, i], correct[n, i], unprojected[n, i]),
                            strict=True,
                        )
                    ),
                    "Self-copy control changed output",
                )
                controls.append({"noise_index": n, "sample_index": i, "exact": True})
            if ids[i, 1] == 0:
                wrong[n, i] = correct[
                    n, i
                ]  # Analysis uses all nonself scene outputs at frame zero.
                normalized_wrong[n, i] = normalized[n, i]
                unprojected_wrong[n, i] = unprojected[n, i]
            else:
                swapped = dict(batch)
                donor = batches[int(donor_rows[i][0])]
                for key in batch:
                    if key.startswith("observation.images"):
                        swapped[key] = donor[key].clone()
                normalized_wrong[n, i], wrong[n, i], unprojected_wrong[n, i] = predict(
                    swapped, noise
                )
    require(calls == expected_forwards, "Collection forward count differs")
    require(parameter_digests(policy) == before, "Collection changed parameters")
    arrays = {
        "correct": correct,
        "wrong": wrong,
        "normalized_correct": normalized,
        "unprojected_correct": unprojected,
        "targets": np.asarray(targets),
        "raw_targets": np.asarray(raw_targets),
        "normalized_targets": np.asarray(normalized_targets),
        "valid_mask": np.asarray(masks),
        "identities": ids,
        "noise": np.stack([n.numpy() for n in noises]),
        "input_seen_at_checkpoint": seen_at_checkpoint,
        "input_seen_by_step2500": seen_by_step2500,
        "input_state": np.stack([b["observation.state"][0].cpu().numpy() for b in batches]),
        "input_language_tokens": np.stack([
            b["observation.language.tokens"][0].cpu().numpy() for b in batches
        ]),
        "input_language_attention_mask": np.stack([
            b["observation.language.attention_mask"][0].cpu().numpy() for b in batches
        ]),
        "normalized_wrong": normalized_wrong,
        "unprojected_wrong": unprojected_wrong,
        "frame0_nonvisual_equal": np.array(True),
        "cyclic_donors": np.array([d.item() for d in donor_rows]),
    }
    with (output / "arrays.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    groups = {
        side + "_" + kind: [
            i
            for i, d in enumerate(context.physical.dimensions)
            if d.name.startswith(side + "_") and (("gripper" in d.name) == (kind == "gripper"))
        ]
        for side in ("left", "right")
        for kind in ("joints", "gripper")
    }
    write_json(
        output / "result.json",
        {
            "status": "collected",
            "groups": groups,
            "inputs": input_records,
            "controls": controls,
            "forwards": calls,
            "parameters_unchanged": True,
            "optimizer_steps": 0,
            "model_loaded": True,
            "checkpoint_sha256": context.source_sha256,
            "checkpoint_step": plan["checkpoint_step"],
            "training_schedule_sha256": plan["training_schedule_sha256"],
            "training_ingress_sha256": plan["training_ingress_sha256"],
            "exposure_reference_step": 2500,
            "m2_complete": False,
        },
    )
    seal_bundle(output, {"stage": "collect", "plan_sha256": plan["_sha256"]})


def supervise(plan_path, job, window):
    """Bounded detached collector; the independently registered window owns shutdown."""
    import shutil
    import subprocess
    import sys

    from scripts.diagnose_smolvla_visual_research import ROOT, load_plan
    from scripts.iris_protocol import bind_inputs
    from scripts.run_iris_furnace import alive, process_tree_rss, terminate, ticks

    plan = load_plan(plan_path, "collect")
    registration = json.loads((window / "registration.json").read_text())
    permit = json.loads((ROOT / plan["gpu_registration"]).read_text())
    require(alive(permit["watchdog"]["pid"], permit["watchdog"]["ticks"]), "Watchdog absent")
    require(time.time() < registration["deadline"], "Shared window expired")
    require(os.environ.get("ROSETTA_TORCH_DEVICE") == "cuda", "Offline CUDA profile required")
    job.mkdir(parents=True, exist_ok=False)
    write_json(job / "launch.json", {"plan_sha256": sha256(plan_path), "optimizer_steps": 0})
    bind_inputs()
    commands = [
        ("tests", ["-m", "pytest", "-q", "tests/test_visual_research.py", "-o", "addopts="]),
        ("doctor", ["scripts/autodl_doctor_cuda.py"]),
        ("collect", ["scripts/diagnose_smolvla_visual_research.py", "--plan", str(plan_path),
                     "--stage", "collect", "--output", str(job / "collect")]),
        ("analyze", ["scripts/diagnose_smolvla_visual_research.py", "--plan", str(plan_path),
                     "--stage", "analyze", "--input", str(job / "collect"),
                     "--output", str(job / "analyze")]),
    ]
    stages, error, child, identity = [], None, None, None
    try:
        for name, args in commands:
            require(time.time() < registration["deadline"], "Shared window expired")
            require(shutil.disk_usage(os.environ["ROSETTA_AUTODL_ROOT"]).free > 512 * 1024**2,
                    "Insufficient storage safety reserve")
            began = time.time()
            with (job / (name + ".log")).open("xb") as log:
                child = subprocess.Popen([sys.executable, *args], cwd=ROOT,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
                identity = {"pid": child.pid, "ticks": ticks(child.pid), "stage": name}
                temporary = window / "active-child.tmp"
                temporary.write_text(json.dumps(identity))
                temporary.replace(window / "active-child.json")
                while child.poll() is None:
                    require(time.time() < registration["deadline"], "Shared work deadline")
                    require(process_tree_rss(child.pid) <= 8 * 1024**3, "Worker RSS exceeds 8GiB")
                    require(sum(p.stat().st_size for p in job.rglob("*") if p.is_file())
                            < 256 * 1024**2, "Output exceeds 256MiB")
                    time.sleep(1)
            stages.append({"stage": name, "exit_code": child.returncode,
                           "seconds": time.time() - began})
            print(json.dumps(stages[-1]), flush=True)
            require(child.returncode == 0, "Stage failed: " + name)
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        if child is not None and child.poll() is None:
            terminate(child.pid, identity["ticks"])
    finally:
        result = {"stages": stages, "error": error, "optimizer_steps": 0,
                  "status": "passed" if error is None else "failed"}
        write_json(job / "worker-exited.json", result)
        write_json(window / "worker-exited.json", {"job": job.name, "error": error})
        seal_bundle(job, {"stage": "gpu_research", "plan_sha256": sha256(plan_path)})
