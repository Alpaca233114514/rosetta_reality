"""Verify the native 5120-sample Hestia order using only synthetic indices."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def inspect_schedule():
    import inspect

    import numpy as np
    import torch
    from lerobot.datasets.sampler import EpisodeAwareSampler

    from rosetta_reality.vla import visual_fit as fit
    from rosetta_reality.vla.training import features
    from rosetta_reality.vla.visual_fit_contract import UPSTREAM

    if fit.file_hash(Path(inspect.getfile(EpisodeAwareSampler))) != UPSTREAM["datasets/sampler.py"]:
        raise ValueError("Native sampler source changed")
    episodes = fit.TRAIN40
    context = SimpleNamespace(
        phase="smoke",
        plan={"scope": "bounded_visual_overfit", "optimizer_smoke": {"episodes": episodes}},
        experiment={
            "dataset": {
                "train_episodes": episodes,
                "validation_episodes": fit.DEV5,
                "test_episodes": fit.HIDDEN5,
            }
        },
    )
    module = SimpleNamespace(EpisodeAwareSampler=EpisodeAwareSampler)
    original = features._lerobot_train_module
    feature = features.FixedFrameSamplerFeature(
        {
            "phase": "smoke",
            "sample_identities": [{"episode": ep, "frame": 0} for ep in episodes],
        }
    )
    features._lerobot_train_module = lambda: module
    try:
        feature.install(context)
        sampler = module.EpisodeAwareSampler(
            list(range(0, 5000, 100)),
            list(range(100, 5100, 100)),
            episodes,
            shuffle=True,
            seed=20260809,
            absolute_to_relative_idx={ep * 100: i for i, ep in enumerate(episodes)},
        )

        def epochs():
            while True:
                yield from iter(sampler)

        actual = list(itertools.islice(epochs(), 5120))
        expected = []
        for epoch in range(128):
            seed = int(
                np.random.SeedSequence([20260809, epoch]).generate_state(1, dtype=np.uint64)[0]
            )
            expected.extend(
                torch.randperm(40, generator=torch.Generator().manual_seed(seed)).tolist()
            )
        if actual != expected:
            raise ValueError("Native and independent Hestia schedules differ")
        samples = [[episodes[index], 0] for index in actual]
        if any(sum(sample[0] == ep for sample in samples) != 128 for ep in episodes):
            raise ValueError("Native repetition allocation changed")
        return {
            "status": "passed",
            "sample_identities": samples,
            "native_order_verified": True,
            "expected_schedule_sha256": hashlib.sha256(
                json.dumps(samples, separators=(",", ":")).encode()
            ).hexdigest(),
            "model_loaded": False,
            "real_data_loaded": False,
            "optimizer_steps": 0,
        }
    finally:
        if module.EpisodeAwareSampler is not EpisodeAwareSampler:
            feature.restore(context)
        features._lerobot_train_module = original


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Schedule evidence already exists")
    result = inspect_schedule()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({"status": "passed", "samples": 5120}))
