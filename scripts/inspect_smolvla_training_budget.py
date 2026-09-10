"""CPU-only native scheduler/sampler checks; never load models or real datasets."""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import os
import platform
import resource
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
CONTROL = "configs/vla/m2-smolvla450m-visual-native-b4-pilot-003.yaml"
REVIEW = "reports/training/m2-smolvla-native-visual-coverage40-plan-2026-09-10.json"
EXPECTED = {
    "scheduler": "05d57770348fbb3f412f52804a8d0f015f8128a37c55d5ce8ed9af6ed8522bc9",
    "sampler": "f715aaaa1118ef8901928f92975f7bc08bca412860f072bbcca84555303cc38e",
    "accelerate": "e08c5f5cd3f7a8b8aaece03ad51f1c37994e390faf47d368281ad49c31bc84d1",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_budget():
    import numpy as np
    import torch
    import yaml
    from accelerate.data_loader import DataLoaderShard
    from lerobot.datasets.sampler import EpisodeAwareSampler
    from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

    from rosetta_reality.vla.fixed_visual_samples import validate_visual_samples
    from rosetta_reality.vla.training import features
    from rosetta_reality.vla.training.launch import build_training_arguments
    from rosetta_reality.vla.training.plan import (
        validate_optimizer_contract,
        validate_plan_structure,
    )

    torch.set_num_threads(1)
    sources = {
        "scheduler": CosineDecayWithWarmupSchedulerConfig,
        "sampler": EpisodeAwareSampler,
        "accelerate": DataLoaderShard,
    }
    for name, obj in sources.items():
        if digest(inspect.getfile(obj)) != EXPECTED[name]:
            raise ValueError("Installed source differs: " + name)
    control = yaml.safe_load((ROOT / CONTROL).read_text())
    review = json.loads((ROOT / REVIEW).read_text())
    for name, sha in control["implementation_files"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Audited project source differs: " + name)
    train = review["data"]["train40"]
    samples = [{"episode": ep, "frame": 0} for ep in train]
    experiment = {
        "dataset": {
            "train_episodes": train,
            "validation_episodes": review["data"]["development_validation5"],
            "test_episodes": review["data"]["sealed_hidden5"],
        }
    }
    bad = copy.deepcopy(control["training"])
    bad["steps"] = 1280
    try:
        validate_optimizer_contract(bad)
    except ValueError as exc:
        rejected = str(exc)
    else:
        raise AssertionError("Old 1280/256 draft must remain rejected")
    candidate = copy.deepcopy(control)
    candidate.update(formal_training_claim=False, formal_training_authorized=False)
    candidate["training"].update(
        episodes=train,
        steps=1280,
        save_freq=320,
        checkpoint_steps=[320, 640, 960, 1280],
    )
    candidate["training"]["scheduler"]["num_decay_steps"] = 1280
    candidate["optimizer_smoke"].update(episodes=train, steps=1280, save_freq=320)
    for entry in candidate["features"]:
        if entry["name"] == "fixed_frame_sampler":
            entry["sample_identities"] = samples
    validate_plan_structure(candidate, known_features=features.FEATURE_FACTORIES)
    validate_visual_samples(samples, candidate, experiment, "smoke")
    try:
        validate_visual_samples(samples, candidate, experiment, "formal")
    except ValueError as exc:
        formal_rejection = str(exc)
    else:
        raise AssertionError("Formal mode must remain rejected")
    # Assemble the actual CLI with synthetic paths, without invoking the launcher.
    from rosetta_reality.vla import load_smolvla_experiment

    parent = load_smolvla_experiment(
        ROOT / control["parent_experiment"]["config"], ROOT
    )
    argv = build_training_arguments(
        candidate,
        parent,
        mode="smoke",
        run_name="athena-cpu-schema-fixture",
        model_root=Path("synthetic-model"),
        dataset_root=Path("synthetic-data"),
        output_dir=Path("synthetic-output"),
        device="cpu",
    )
    assert "--steps=1280" in argv and "--policy.scheduler_decay_steps=1280" in argv
    assert "--save_freq=320" in argv and "--batch_size=4" in argv
    scheduler_results = {}
    for steps in (256, 1280):
        parameter = torch.nn.Parameter(torch.zeros(1, device="cpu"))
        optimizer = torch.optim.AdamW([parameter], lr=1e-4)
        scheduler = CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=16, num_decay_steps=steps, peak_lr=1e-4, decay_lr=2.5e-6
        ).build(optimizer, steps)
        rates = []
        for index in range(steps):
            value = optimizer.param_groups[0]["lr"]
            expected = (
                ((1 / 17 - 1) * (1 - index / 16) + 1)
                if index < 16
                else (0.975 * 0.5 * (1 + np.cos(np.pi * index / steps)) + 0.025)
            ) * 1e-4
            assert np.isclose(value, expected, rtol=1e-12, atol=1e-16)
            rates.append(value)
            parameter.grad = torch.zeros_like(parameter)
            optimizer.step()
            scheduler.step()
        assert scheduler.last_epoch == steps
        assert scheduler.get_last_lr() == [2.5e-6]
        scheduler_results[str(steps)] = {
            "actual_lr": rates,
            "last_epoch": scheduler.last_epoch,
            "last_lr": scheduler.get_last_lr(),
        }
    assert (
        scheduler_results["256"]["actual_lr"][:16]
        == (scheduler_results["1280"]["actual_lr"][:16])
    )
    assert (
        scheduler_results["256"]["actual_lr"][16]
        != (scheduler_results["1280"]["actual_lr"][16])
    )

    def sample_case(steps, record_emissions):
        torch.manual_seed(20260809)
        context = SimpleNamespace(phase="smoke", plan=candidate, experiment=experiment)
        module = SimpleNamespace(EpisodeAwareSampler=EpisodeAwareSampler)
        original_getter = features._lerobot_train_module
        feature = features.FixedFrameSamplerFeature(
            {"phase": "smoke", "sample_identities": samples}
        )
        emitted = []
        features._lerobot_train_module = lambda: module
        try:
            feature.install(context)
            native = module.EpisodeAwareSampler

            class Observed(native):
                def __iter__(self):
                    for index in super().__iter__():
                        emitted.append(int(index))
                        yield index

            sampler = (Observed if record_emissions else native)(
                list(range(0, 5000, 100)),
                list(range(100, 5100, 100)),
                train,
                shuffle=True,
                seed=20260809,
                absolute_to_relative_idx={ep * 100: i for i, ep in enumerate(train)},
            )
            loader = DataLoaderShard(
                list(range(40)), batch_size=4, sampler=sampler, num_workers=0
            )

            def batches():
                while True:
                    yield from loader

            iterator = batches()
            consumed = []
            for _ in range(steps):
                consumed.extend(next(iterator).tolist())
            rng_sha = hashlib.sha256(
                torch.get_rng_state().numpy().tobytes()
            ).hexdigest()
            return consumed, emitted, rng_sha
        finally:
            if module.EpisodeAwareSampler is not EpisodeAwareSampler:
                feature.restore(context)
            features._lerobot_train_module = original_getter

    sampler_results = {}
    for steps in (2, 10, 256, 320, 1280):
        consumed, emitted, rng_sha = sample_case(steps, True)
        baseline, _, baseline_rng = sample_case(steps, False)
        assert consumed == baseline and rng_sha == baseline_rng
        expected = []
        for epoch in range((steps * 4 + 39) // 40):
            seed = int(
                np.random.SeedSequence([20260809, epoch]).generate_state(1, np.uint64)[
                    0
                ]
            )
            expected.extend(
                torch.randperm(
                    40, generator=torch.Generator().manual_seed(seed)
                ).tolist()
            )
        tail = 0 if steps % 10 == 0 else 4
        assert consumed == expected[: steps * 4]
        assert (
            len(emitted) == len(consumed) + tail and emitted == expected[: len(emitted)]
        )
        counts = Counter(train[i] for i in consumed)
        if steps == 1280:
            assert len(counts) == 40 and set(counts.values()) == {128}
        sampler_results[str(steps)] = {
            "consumed_count": len(consumed),
            "emitted_count": len(emitted),
            "unused_prefetch": tail,
            "episode_counts": dict(counts),
            "exact_native_order": True,
            "observation_preserves_torch_rng_and_consumed_batches": True,
            "consumed_schedule_sha256": hashlib.sha256(
                json.dumps(consumed, separators=(",", ":")).encode()
            ).hexdigest(),
        }
    return {
        "status": "passed",
        "sources": EXPECTED,
        "old_draft_rejection": rejected,
        "formal_sample_rejection": formal_rejection,
        "candidate_schema_and_cli_passed": True,
        "candidate_launch_authorized_by_this_probe": False,
        "scheduler": scheduler_results,
        "sampler": sampler_results,
        "dummy_cpu_optimizer_updates": 1536,
        "model_optimizer_updates": 0,
        "model_loaded": False,
        "real_dataset_loaded": False,
        "hidden_test_loaded": False,
        "production_consumed_update_observer": "not implemented or tested",
        "m2_complete": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if platform.system() != "Linux" or os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError("Run only in the registered offline Linux environment")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError("Output must be a new file inside the workspace")
    started = time.time()
    result = inspect_budget()
    result["elapsed_seconds"] = time.time() - started
    result["peak_host_rss_bytes"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    )
    assert (
        result["elapsed_seconds"] < 600 and result["peak_host_rss_bytes"] < 2 * 1024**3
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k != "scheduler"}))


if __name__ == "__main__":
    main()
