"""Inspect native sampler order using synthetic indices; never open real data."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import itertools
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def inspect_schedule() -> dict:
    import numpy as np
    import torch
    from lerobot.datasets.sampler import EpisodeAwareSampler
    from prepare_visual_coverage import REVIEW, UPSTREAM

    import rosetta_reality.vla.training.features as features

    source = Path(inspect.getfile(EpisodeAwareSampler))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != UPSTREAM["datasets/sampler.py"]:
        raise ValueError("Installed sampler differs from the audited source.")
    review = json.loads((ROOT / REVIEW).read_text())
    train = review["data"]["train40"]
    result = {}
    for arm, episodes in (("A", train[:8]), ("B", train)):
        samples = [{"episode": ep, "frame": 0} for ep in episodes]
        context = SimpleNamespace(
            phase="smoke",
            plan={
                "scope": "bounded_visual_overfit",
                "optimizer_smoke": {"episodes": episodes},
            },
            experiment={
                "dataset": {
                    "train_episodes": train,
                    "validation_episodes": review["data"]["development_validation5"],
                    "test_episodes": review["data"]["sealed_hidden5"],
                }
            },
        )
        module = SimpleNamespace(EpisodeAwareSampler=EpisodeAwareSampler)
        original_getter = features._lerobot_train_module
        feature = features.FixedFrameSamplerFeature(
            {"phase": "smoke", "sample_identities": samples}
        )
        features._lerobot_train_module = lambda: module
        try:
            feature.install(context)
            # Deliberately non-contiguous episode numbers and synthetic offsets.
            sampler = module.EpisodeAwareSampler(
                list(range(0, 5000, 100)),
                list(range(100, 5100, 100)),
                episodes,
                shuffle=True,
                seed=20260809,
                absolute_to_relative_idx={
                    ep * 100: index for index, ep in enumerate(episodes)
                },
            )

            def actual_epochs():
                while True:
                    yield from iter(sampler)

            indices = list(itertools.islice(actual_epochs(), 1024))
            independent = []
            for epoch in range((1024 + len(episodes) - 1) // len(episodes)):
                seed = int(
                    np.random.SeedSequence([20260809, epoch]).generate_state(
                        1, dtype=np.uint64
                    )[0]
                )
                independent.extend(
                    torch.randperm(
                        len(episodes), generator=torch.Generator().manual_seed(seed)
                    ).tolist()
                )
            if indices != independent[:1024]:
                raise ValueError(
                    "Native sampler order differs from the registered epoch permutation."
                )
            schedule = [episodes[i] for i in indices]
            counts = Counter(schedule)
            histogram = dict(Counter(counts.values()))
            expected = {128: 8} if arm == "A" else {26: 24, 25: 16}
            if histogram != expected:
                raise ValueError(
                    "Exposure allocation differs from the fixed update budget."
                )
            result[arm] = {
                "episodes": episodes,
                "schedule": schedule,
                "schedule_sha256": hashlib.sha256(
                    json.dumps(schedule, separators=(",", ":")).encode()
                ).hexdigest(),
                "counts": dict(counts),
                "count_histogram": histogram,
                "native_and_independent_orders_equal": True,
            }
        finally:
            if module.EpisodeAwareSampler is not EpisodeAwareSampler:
                feature.restore(context)
            features._lerobot_train_module = original_getter
    return {
        "status": "synthetic_sampler_verified",
        "sampler_source_sha256": digest,
        "seed": 20260809,
        "batch_size": 4,
        "steps": 256,
        "arms": result,
        "real_dataset_index_mapping": "not measured",
        "model_loaded": False,
        "data_loaded": False,
        "optimizer_created": False,
        "optimizer_steps": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Sampler evidence already exists.")
    result = inspect_schedule()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(
        json.dumps(
            {
                "status": result["status"],
                "A": result["arms"]["A"]["count_histogram"],
                "B": result["arms"]["B"]["count_histogram"],
            }
        )
    )


if __name__ == "__main__":
    main()
