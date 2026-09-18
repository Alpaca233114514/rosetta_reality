"""Registered real-model training-frame, numerical and Gate boundary audit."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]

from scripts.iris_runtime import budget, load_native, save, sha  # noqa: E402

NAME = "training-chain-gpu-audit-20260914-007"
SAMPLES = [[ep, frame] for frame in (0, 249, 450, 499) for ep in (49, 4)]
STAGES = (
    "prepare",
    "doctor",
    "benchmark",
    "preflight",
    "parity",
    "smoke",
    "collect-first",
    "collect-reload",
    "verify-reload",
)


def load(path):
    return json.loads(Path(path).read_text())


def tensor_sha(value):
    return hashlib.sha256(
        value.detach()
        .cpu()
        .contiguous()
        .view(-1)
        .view(__import__("torch").uint8)
        .numpy()
        .tobytes()
    ).hexdigest()


def prepare(job):
    import yaml

    from scripts.iris_protocol import baseline, bind_inputs
    from scripts.run_smolvla_v2 import _resolve_plan

    bind_inputs()
    plan = copy.deepcopy(baseline())
    plan.update(
        plan_id=NAME,
        run_name=NAME,
        status="preregistered",
        scope="bounded_temporal_sampling",
        training_authorized=True,
        hypothesis="Engineering audit of exact temporal ingress and native updates",
    )
    plan["preflight"].update(run_name=NAME + "-preflight", episodes=[49, 4], batch_size=1)
    plan["optimizer_smoke"].update(
        run_name=NAME + "-smoke",
        episodes=[49, 4],
        batch_size=4,
        steps=2,
        save_freq=2,
        log_freq=1,
        num_workers=0,
    )
    plan["features"] = [
        f for f in plan["features"] if f["name"] != "fixed_frame_sampler"
    ]
    plan["features"].insert(2, {"name": "canonical_image_scaling"})
    schedule = job / "schedule.json"
    save(
        schedule,
        {
            "status": "preregistered",
            "training_authorized": True,
            "seed": 20260809,
            "sample_identities": SAMPLES,
        },
    )
    plan["features"].insert(
        2,
        {
            "name": "explicit_sample_schedule",
            "path": schedule.relative_to(ROOT).as_posix(),
            "sha256": sha(schedule),
        },
    )
    plan["implementation_files"] = load(job / "registration.json")["sources"]
    plan["stop_conditions"] = [
        "Exactly two updates; no resume or formal furnace",
        "Stop on drift, nonfinite, 8GiB allocated/RSS, 10GiB reserved",
        "Shared 2700 second deadline; preserve all failures",
    ]
    path = job / "smoke.yaml"
    with path.open("x") as stream:
        yaml.safe_dump(plan, stream, sort_keys=False)
    assert _resolve_plan(path)[0] == plan
    save(
        job / "plan-seal.json", {"sha256": sha(path), "schedule_sha256": sha(schedule)}
    )


def plan_path(job):
    path = job / "smoke.yaml"
    assert sha(path) == load(job / "plan-seal.json")["sha256"]
    return path


def temporal_data(context):
    import numpy as np
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from torch.utils.data import default_collate

    from rosetta_reality.data import resolve_prepared_cache
    from rosetta_reality.data.config import load_dataset_config
    from rosetta_reality.vla.image_scaling import canonical_rgb_uint8

    cfg = load_dataset_config(ROOT / "configs/data/aloha_sim_insertion_m2.yaml")
    root, _ = resolve_prepared_cache(cfg, ROOT, validate_checksums=True)
    meta = LeRobotDatasetMetadata(cfg.repo_id, root=root, revision=cfg.revision)
    deltas = resolve_delta_timestamps(context.policy.config, meta)
    assert context.policy.config.n_obs_steps == 1
    assert context.policy.config.chunk_size == 50
    ds = LeRobotDataset(
        cfg.repo_id,
        root=root,
        revision=cfg.revision,
        episodes=[49, 4],
        download_videos=False,
        return_uint8=True,
        delta_timestamps=deltas,
    )
    starts = dict(
        zip(
            ds.meta.episodes["episode_index"],
            ds.meta.episodes["dataset_from_index"],
            strict=True,
        )
    )
    batches, raws = [], []
    for ep, frame in SAMPLES:
        raw = ds[ds.absolute_to_relative_idx[int(starts[ep]) + frame]]
        assert [int(raw["episode_index"]), int(raw["frame_index"])] == [ep, frame]
        assert np.array_equal(
            raw["action_is_pad"].numpy(), np.arange(50) + frame >= 500
        )
        batch = default_collate([copy.deepcopy(raw)])
        for key in ds.meta.camera_keys:
            batch[key] = canonical_rgb_uint8(batch[key].to("cuda"))
        processed = context.pre(batch)
        assert processed["action"].shape == (1, 50, raw["action"].shape[-1])
        assert processed["action_is_pad"].shape == (1, 50)
        batches.append(processed)
        raws.append(raw)
    return batches, raws, sha(root / "manifest.json")


def parity(job, deadline):
    import torch
    from lerobot.policies.smolvla import modeling_smolvla

    from rosetta_reality.vla.training.masked_camera import (
        install_masked_camera_encoder_skip,
        restore_masked_camera_encoder_skip,
    )
    from scripts.diagnose_zen_noise_transfer import parameter_digests
    from scripts.iris_runtime import observations

    context = load_native(plan_path(job), split="train", deadline=deadline)
    batches, raw_samples, _ = temporal_data(context)
    before = parameter_digests(context.policy)
    records, scaling_records = [], []
    # Three bounded real forwards/backwards per implementation, including tail padding.
    for index in (0, 2, 6):
        reference = None
        reference_action, reference_loss = None, None
        for variant in ("native", "skip", "cpu_scaling"):
            skip = variant == "skip"
            budget(deadline)
            if skip:
                install_masked_camera_encoder_skip(modeling_smolvla)
            try:
                context.policy.train()
                context.policy.zero_grad(set_to_none=True)
                torch.manual_seed(20260914)
                torch.cuda.manual_seed_all(20260914)
                measured_batch = copy.deepcopy(batches[index])
                if variant == "cpu_scaling":
                    measured_batch["observation.images.camera1"] = (
                        raw_samples[index]["observation.images.top"].float() / 255
                    ).unsqueeze(0).cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, _ = context.policy(measured_batch)
                loss.backward()
                grads = {}
                for name, parameter in context.policy.named_parameters():
                    if parameter.grad is not None:
                        assert bool(torch.isfinite(parameter.grad).all())
                        grads[name] = tensor_sha(parameter.grad)
                assert bool(torch.isfinite(loss)) and grads
                context.policy.eval()
                noise = torch.randn(
                    (1, 50, context.policy.config.max_action_dim),
                    generator=torch.Generator().manual_seed(20260905),
                ).cuda()
                with (
                    torch.inference_mode(),
                    torch.autocast("cuda", dtype=torch.bfloat16),
                ):
                    action = context.policy.predict_action_chunk(
                        observations(measured_batch), noise=noise
                    )
                current = {
                    "loss": tensor_sha(loss),
                    "gradients": grads,
                    "full_action": tensor_sha(action),
                }
                if reference is None:
                    reference = current
                    reference_action = action.detach().float().cpu()
                    reference_loss = float(loss.detach())
                elif variant == "skip":
                    assert reference == current, (
                        "Masked-camera skip changes real loss/gradients/action"
                    )
                    records.append(
                        {
                            "sample": SAMPLES[index],
                            "exact": True,
                            "gradient_tensors": len(grads),
                            "loss": float(loss),
                        }
                    )
                else:
                    scaling_records.append({
                        "sample": SAMPLES[index],
                        "loss_exact": current["loss"] == reference["loss"],
                        "loss_abs_difference": abs(float(loss.detach()) - reference_loss),
                        "full_action_exact": current["full_action"] == reference["full_action"],
                        "full_action_max_abs": float(
                            (action.detach().float().cpu() - reference_action).abs().max()
                        ),
                        "gradient_tensors": len(grads),
                        "gradient_tensors_exact": sum(
                            value == reference["gradients"].get(name)
                            for name, value in grads.items()
                        ),
                    })
                    assert reference == current, "Canonical CPU/CUDA model behavior differs"
            finally:
                if skip:
                    restore_masked_camera_encoder_skip(modeling_smolvla)
                context.policy.zero_grad(set_to_none=True)
    assert parameter_digests(context.policy) == before
    save(
        job / "parity.json",
        {
            "status": "passed",
            "records": records,
            "optimizer_steps": 0,
            "parameters_unchanged": True,
            "cpu_cuda_pixel_scaling_effect": scaling_records,
        },
    )


def smoke(job, deadline):
    from types import SimpleNamespace

    import lerobot.scripts.lerobot_train as trainer
    import torch
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    import scripts.run_smolvla_v2 as launcher
    from rosetta_reality.sim import load_action_contract
    from rosetta_reality.vla.action_space import load_smolvla_action_space
    from rosetta_reality.vla.processor import (
        standard_aloha_action_to_model,
        standard_aloha_state_to_pi,
    )
    from rosetta_reality.vla.training.observed_launch import run_observed_launch
    from scripts.diagnose_zen_noise_transfer import parameter_digests
    from scripts.iris_stage import checkpoint_path, inventory

    plan, _, experiment = launcher._resolve_plan(plan_path(job))
    policy_config = SmolVLAConfig.from_pretrained(
        launcher.phase_runner._model_root(experiment), local_files_only=True
    )
    for section in ("policy", "adaptation"):
        for key, value in experiment["model"][section].items():
            if hasattr(policy_config, key):
                setattr(policy_config, key, value)
    _, raw_samples, _ = temporal_data(
        SimpleNamespace(
            pre=lambda value: value, policy=SimpleNamespace(config=policy_config)
        )
    )
    raw_by_pair = {
        tuple(pair): raw for pair, raw in zip(SAMPLES, raw_samples, strict=True)
    }
    physical = load_action_contract(ROOT / experiment["action_contract"]["derived"])
    space = load_smolvla_action_space(experiment)
    stats = load(ROOT / plan["normalization"]["report"])["effective_stats"]

    policies, records = [], []
    original_make, original_step = trainer.make_policy, torch.optim.AdamW.step
    original_cycle = trainer.cycle
    pending_pairs = None
    updates = 0

    def cycle(*args, **kwargs):
        nonlocal pending_pairs
        for batch in original_cycle(*args, **kwargs):
            assert pending_pairs is None
            pending_pairs = list(
                zip(
                    batch["episode_index"].cpu().tolist(),
                    batch["frame_index"].cpu().tolist(),
                    strict=True,
                )
            )
            yield batch
            assert pending_pairs is None, "Delivered batch did not reach policy ingress"

    def make(*args, **kwargs):
        policy = original_make(*args, **kwargs)
        frozen = {
            n: h
            for n, h in parameter_digests(policy).items()
            if not dict(policy.named_parameters())[n].requires_grad
        }
        policies.append((policy, frozen))

        def ingress(module, inputs):
            nonlocal pending_pairs
            budget(deadline)
            batch = inputs[0]
            assert pending_pairs is not None
            pairs = pending_pairs
            assert batch["action"].shape[0] == len(pairs)
            expected = [
                tuple(p) for p in SAMPLES[len(records) : len(records) + len(pairs)]
            ]
            assert pairs == expected, "Actual model forward consumed unexpected frames"
            for i, (ep, frame) in enumerate(pairs):
                expected_pad = (
                    torch.arange(50, device=batch["action_is_pad"].device) + frame
                    >= 500
                )
                assert torch.equal(batch["action_is_pad"][i], expected_pad)
                assert bool(torch.isfinite(batch["action"][i]).all())
                raw = raw_by_pair[(ep, frame)]
                projected = (
                    raw["action"]
                    .maximum(physical.lower_bounds)
                    .minimum(physical.upper_bounds)
                )
                encoded = {
                    "action": standard_aloha_action_to_model(
                        projected, space.representation_adapter
                    ),
                    "observation.state": standard_aloha_state_to_pi(
                        raw["observation.state"]
                    ),
                }
                for key, value in encoded.items():
                    mean = torch.tensor(stats[key]["mean"], dtype=value.dtype)
                    std = torch.tensor(stats[key]["std"], dtype=value.dtype)
                    expected_value = (value - mean) / (std + 1e-8)
                    torch.testing.assert_close(
                        batch[key][i].cpu(), expected_value, atol=1e-6, rtol=1e-6
                    )
                actual_image = batch["observation.images.camera1"][i]
                expected_image = (raw["observation.images.top"].float() / 255).to(
                    actual_image.device
                )
                torch.testing.assert_close(actual_image, expected_image, atol=0, rtol=0)
                assert torch.equal(
                    (actual_image.cpu() * 255).round().to(torch.uint8),
                    raw["observation.images.top"],
                )
                records.append(
                    {
                        "episode": ep,
                        "frame": frame,
                        "valid_slots": int((~expected_pad).sum()),
                        "model_tensor_shapes": {
                            key: list(batch[key][i].shape)
                            for key in (
                                "action", "action_is_pad", "observation.state",
                                "observation.images.camera1",
                            )
                        },
                        "model_action_target_sha256": tensor_sha(batch["action"][i]),
                        "model_state_sha256": tensor_sha(batch["observation.state"][i]),
                        "model_image_sha256": tensor_sha(
                            batch["observation.images.camera1"][i]
                        ),
                    }
                )
            pending_pairs = None

        policy.register_forward_pre_hook(ingress)
        return policy

    def step(optimizer, *args, **kwargs):
        nonlocal updates
        budget(deadline)
        assert updates < 2
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    assert bool(torch.isfinite(parameter.grad).all())
        value = original_step(optimizer, *args, **kwargs)
        updates += 1
        return value

    trainer.make_policy, torch.optim.AdamW.step = make, step
    trainer.cycle = cycle
    sys.argv = ["run_smolvla_v2.py", "smoke", "--plan", str(plan_path(job))]
    try:
        run_observed_launch(
            trainer,
            launcher.main,
            expected_samples=SAMPLES,
            batch_size=4,
            output=job / "observation",
        )
    finally:
        trainer.make_policy, torch.optim.AdamW.step = original_make, original_step
        trainer.cycle = original_cycle
    assert updates == 2 and len(records) == 8 and len(policies) == 1
    policy, frozen = policies[0]
    current = parameter_digests(policy)
    assert all(current[n] == h for n, h in frozen.items())
    plan, _, _ = launcher._resolve_plan(plan_path(job))
    checkpoint = checkpoint_path(plan)
    save(checkpoint / "rosetta_image_preprocessing.json", {
        "schema_version": 1,
        "recipe": "float32_pixel_table_round_float64_integer_divide_255",
        "raw_dtype": "uint8", "output_dtype": "float32", "range": [0.0, 1.0],
        "implementation_sha256": sha(ROOT / "src/rosetta_reality/vla/image_scaling.py"),
    })
    launch_root = (
        Path(os.environ["ROSETTA_RUN_ROOT"]) / experiment["experiment_id"] / "launch"
    )
    for mode, section in (("preflight", "preflight"), ("smoke", "optimizer_smoke")):
        run_name = plan[section]["run_name"]
        save(job / (mode + "-launch.json"), load(launch_root / (run_name + ".json")))
        save(
            job / (mode + "-runtime-experiment.json"),
            load(launch_root / (run_name + "-runtime-experiment.json")),
        )
    save(job / "native-train-config.json", load(checkpoint / "train_config.json"))
    metrics = load(checkpoint.parent / "rosetta_checkpoint_metrics.json")
    assert metrics["step"] == 2
    save(
        job / "checkpoint.json",
        {
            "files": inventory(checkpoint),
            "steps": 2,
            "frozen_tensors": len(frozen),
            "frozen_unchanged": True,
        },
    )
    save(
        job / "model-ingress.json",
        {
            "status": "passed",
            "records": records,
            "optimizer_steps": updates,
            "unique_input_frames": 8,
            "native_temporal_config": {
                "n_obs_steps": policy.config.n_obs_steps,
                "chunk_size": policy.config.chunk_size,
                "observation_delta_indices": policy.config.observation_delta_indices,
                "action_delta_indices": policy.config.action_delta_indices,
            },
        },
    )


def collect(job, name, deadline):
    import numpy as np
    import torch

    from rosetta_reality.vla.reload_evidence import write_bundle
    from scripts.diagnose_zen_noise_transfer import parameter_digests
    from scripts.iris_runtime import observations
    from scripts.iris_stage import checkpoint_path
    from scripts.run_smolvla_v2 import _resolve_plan

    path = plan_path(job)
    plan, _, _ = _resolve_plan(path)
    source = checkpoint_path(plan)
    files = load(job / "checkpoint.json")["files"]
    context = load_native(
        path,
        checkpoint=source,
        checkpoint_files=files,
        split="train",
        deadline=deadline,
    )
    batches, raws, data_sha = temporal_data(context)
    before = parameter_digests(context.policy)
    predictions, standards, noises, samples, masks, inputs = [], [], [], [], [], []
    for seed in (0, 20260905):
        for pair, batch, raw in zip(SAMPLES, batches, raws, strict=True):
            budget(deadline)
            shape = (1, 50, context.policy.config.max_action_dim)
            noise = (
                torch.zeros(shape)
                if seed == 0
                else torch.randn(shape, generator=torch.Generator().manual_seed(seed))
            ).cuda()
            context.policy.reset()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                action = context.policy.predict_action_chunk(
                    observations(batch), noise=noise.clone()
                )
                context.post(action.clone())
            predictions.append(action[0].float().cpu().numpy())
            standards.append(
                context.decoder.last_unclipped_action[0].float().cpu().numpy()
            )
            noises.append(noise[0].cpu().numpy())
            masks.append((~raw["action_is_pad"]).numpy())
            samples.append(pair)
            inputs.append(
                {
                    k: tensor_sha(v)
                    for k, v in observations(batch).items()
                    if isinstance(v, torch.Tensor)
                }
            )
    assert parameter_digests(context.policy) == before

    def object_sha(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    identity = {
        "plan_sha256": sha(path),
        "model_sha256": files["model.safetensors"],
        "processor_sha256": object_sha(
            {k: v for k, v in files.items() if "processor" in k}
        ),
        "action_contract_sha256": sha(
            ROOT / context.experiment["action_contract"]["derived"]
        ),
        "dataset_manifest_sha256": data_sha,
        "input_sha256": object_sha(inputs),
        "inference_recipe_sha256": object_sha(
            {"seeds": [0, 20260905], "steps": 10, "autocast": "bf16", "device": "cuda",
             "image_scaling": "canonical_float32_pixel_table"}
        ),
        "sample_count": len(samples),
        "chunk_size": 50,
        "normalized_action_dim": predictions[0].shape[-1],
        "standard_action_dim": standards[0].shape[-1],
        "noise_action_dim": noises[0].shape[-1],
    }
    write_bundle(
        job / name,
        {
            "normalized_actions": np.asarray(predictions),
            "standard_actions": np.asarray(standards),
            "noise": np.asarray(noises),
            "sample_identities": np.asarray(samples, dtype=np.int64),
            "valid_mask": np.asarray(masks),
        },
        identity,
    )
    save(
        job / (name + "-execution.json"),
        {
            "pid": os.getpid(),
            "model_loaded": True,
            "saved_processors_loaded": True,
            "forwards": len(samples),
            "parameters_unchanged": True,
            "optimizer_steps": 0,
        },
    )


def run(job, stage):
    import subprocess

    registration = load(job / "registration.json")
    deadline = registration["deadline"]
    for name, expected in registration["sources"].items():
        assert sha(ROOT / name) == expected, name
    assert time.time() < deadline
    if stage == "prepare":
        prepare(job)
    elif stage == "doctor":
        from importlib.metadata import distribution

        from scripts.iris_protocol import UPSTREAM

        installed = Path(distribution("lerobot").locate_file("lerobot"))
        for name, expected in UPSTREAM.items():
            assert sha(installed / name) == expected
        subprocess.run([sys.executable, "scripts/check_env.py"], check=True)
        subprocess.run(["bash", "scripts/run_autodl.sh", "doctor"], check=True)
    elif stage == "benchmark":
        subprocess.run(
            [
                sys.executable,
                "scripts/benchmark_smolvla.py",
                "--config",
                "configs/vla/smolvla_450m_aloha_insertion_action_repair_bounded_gripper_003.yaml",
            ],
            check=True,
        )
    elif stage == "preflight":
        subprocess.run(
            [
                sys.executable,
                "scripts/run_smolvla_v2.py",
                "preflight",
                "--plan",
                str(plan_path(job)),
            ],
            check=True,
        )
    elif stage == "parity":
        parity(job, deadline)
    elif stage == "smoke":
        smoke(job, deadline)
    elif stage in ("collect-first", "collect-reload"):
        collect(job, stage, deadline)
    elif stage == "verify-reload":
        from rosetta_reality.vla.reload_evidence import compare_bundles

        proof = compare_bundles(
            job / "collect-first",
            job / "collect-reload",
            expected_identity=load(job / "collect-first/manifest.json")["identity"],
        )
        save(job / "reload-proof.json", proof)
        assert proof["status"] == "passed"
    else:
        from scripts import iris_gate

        iris_gate.NAME = NAME + "-gate"
        iris_gate.ARMS = {"control": "461"}
        gate_job = job.parent / iris_gate.NAME
        if stage == "gate-render":
            gate_job.mkdir(exist_ok=False)
            save(gate_job / "registration.json", registration)
            iris_gate.render(gate_job)
        else:
            result = iris_gate.run_gate(gate_job, "control", stage)
            save(
                job / (stage + "-outcome.json"),
                {
                    "engine_exit_code": result,
                    "meaning": "measured outcome, not audit pass",
                },
            )
            assert result in (0, 1), (
                "Gate execution did not produce a normal measured outcome"
            )
            if stage == "gate3":
                assert result == 0, "Gate 3 failed; Gate 4 remains closed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    run(args.job.resolve(), args.stage)
