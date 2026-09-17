"""Fresh bounded full-frame furnace using the verified native training chain."""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT), str(ROOT / 'scripts')]
from scripts import training_chain_gpu_audit as checks  # noqa: E402
from scripts.iris_runtime import budget, save, sha  # noqa: E402

load, tensor_sha, plan_path = checks.load, checks.tensor_sha, checks.plan_path
NAME = 'canonical-fullframes-20260914-001'
STAGES = (
    'prepare', 'doctor', 'benchmark', 'preflight', 'parity', 'train',
    'collect-first', 'collect-reload', 'verify-reload',
)

def smoke(job, deadline):

    import lerobot.scripts.lerobot_train as trainer
    import torch

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
    SAMPLES = load(job / 'schedule.json')['sample_identities']
    raw_by_pair = {}
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
            raw_by_pair.clear()
            for i, pair in enumerate(pending_pairs):
                raw_by_pair[pair] = {
                    k: v[i].detach().cpu().clone()
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }
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
        assert updates < 5000
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
    assert updates == 5000 and len(records) == 20000 and len(policies) == 1
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
    assert metrics["step"] == 5000
    recovery = {}
    for number in (2500, 5000):
        directory = checkpoint.parent.parent / f"{number:06d}"
        assert (directory / "training_state/optimizer_state.safetensors").is_file()
        assert load(directory / "rosetta_checkpoint_metrics.json")["step"] == number
        recovery[str(number)] = inventory(directory)
    save(
        job / "checkpoint.json",
        {
            "files": inventory(checkpoint),
            "steps": 5000,
            "frozen_tensors": len(frozen),
            "frozen_unchanged": True,
            "recovery_checkpoints": recovery,
        },
    )


    save(
        job / "model-ingress.json",
        {
            "status": "passed",
            "records": records,
            "optimizer_steps": updates,
            "unique_input_frames": 20000,
            "native_temporal_config": {
                "n_obs_steps": policy.config.n_obs_steps,
                "chunk_size": policy.config.chunk_size,
                "observation_delta_indices": policy.config.observation_delta_indices,
                "action_delta_indices": policy.config.action_delta_indices,
            },
        },
    )



def full_schedule(episodes):
    import random

    if len(episodes) != 40 or len(set(episodes)) != 40:
        raise ValueError("Exactly 40 distinct registered train episodes required")
    if set(episodes).intersection({31, 6, 1, 24, 5, 22, 13, 7, 33, 45}):
        raise ValueError("Non-training episode in furnace schedule")
    samples = [[ep, frame] for ep in episodes for frame in range(500)]
    random.Random(20260809).shuffle(samples)
    assert len(samples) == len(set(map(tuple, samples))) == 20000
    return samples


def prepare(job):
    import yaml

    from scripts.iris_protocol import baseline, bind_inputs
    from scripts.run_smolvla_v2 import _resolve_plan

    bind_inputs()
    plan = copy.deepcopy(baseline())
    episodes = plan["training"]["episodes"]
    samples = full_schedule(episodes)
    plan.update(plan_id=NAME, run_name=NAME, status="preregistered",
                scope="bounded_temporal_sampling", training_authorized=True,
                hypothesis="Fresh canonical full-frame baseline; no single-cause claim")
    plan["training"].update(steps=5000, save_freq=2500, log_freq=25,
                            checkpoint_steps=[2500, 5000])
    plan["training"]["scheduler"].update(num_warmup_steps=16, num_decay_steps=5000)
    plan["optimizer_smoke"].update(run_name=NAME, episodes=episodes,
                                    steps=5000, batch_size=4, save_freq=2500,
                                    log_freq=25, num_workers=0)
    plan["preflight"].update(run_name=NAME+"-preflight", episodes=[49, 4], batch_size=1)
    plan["features"] = [f for f in plan["features"] if f["name"] != "fixed_frame_sampler"]
    schedule = job / "schedule.json"
    save(schedule, {"status":"preregistered", "training_authorized":True,
                    "seed":20260809, "sample_identities":samples})
    plan["features"][2:2] = [
        {"name":"explicit_sample_schedule", "path":schedule.relative_to(ROOT).as_posix(),
         "sha256":sha(schedule)}, {"name":"canonical_image_scaling"}]
    plan["implementation_files"] = load(job / "registration.json")["sources"]
    plan["stop_conditions"] = [
        "Exactly 5000 fresh updates and 20000 unique train frames; no resume or retry",
        "7200 second shared deadline; 8GiB allocated/RSS, 10GiB reserved",
        "Preserve both complete recovery checkpoints; stop on identity or finite failure"]
    path = job / "smoke.yaml"
    with path.open("x") as stream:
        yaml.safe_dump(plan, stream, sort_keys=False)
    assert _resolve_plan(path)[0] == plan
    save(job / "plan-seal.json", {"sha256":sha(path), "schedule_sha256":sha(schedule)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    registration = load(args.job / "registration.json")
    assert registration["id"] == NAME
    for name, expected in registration["sources"].items():
        assert sha(ROOT / name) == expected, name
    assert time.time() < registration["deadline"]
    if args.stage == "prepare":
        prepare(args.job)
    elif args.stage == "train":
        smoke(args.job, registration["deadline"])
    else:
        checks.run(args.job, args.stage)


if __name__ == "__main__":
    main()
