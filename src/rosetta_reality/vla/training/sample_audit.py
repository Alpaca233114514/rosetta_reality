"""Explicit temporal coverage, independent of model execution or global RNG."""

from __future__ import annotations

import hashlib
import json
from collections import Counter


def spaced_frames(length: int, count: int) -> list[int]:
    if type(length) is not int or type(count) is not int or count < 2 or length < count:
        raise ValueError("Temporal sampling requires length >= count >= 2")
    return [index * (length - 1) // (count - 1) for index in range(count)]


def schedule_digest(samples) -> str:
    return hashlib.sha256(json.dumps(samples, separators=(",", ":")).encode()).hexdigest()


def temporal_schedule(control, lengths, *, seed: int, per_episode: int = 128):
    """Preserve episode order; replace repeated frame zero with shuffled temporal frames.

    Only caller-supplied training lengths may be passed. Dedicated per-episode
    NumPy generators leave the model, sampler and global NumPy RNG unchanged.
    """
    import numpy as np

    counts = Counter()
    for pair in control:
        if len(pair) != 2 or any(type(x) is not int for x in pair) or pair[1] != 0:
            raise ValueError("Control must contain explicit integer episode/frame-zero pairs")
        counts[pair[0]] += 1
    if not counts or set(counts) != set(lengths) or any(v != per_episode for v in counts.values()):
        raise ValueError("Every registered train episode must have the same exposure budget")
    frames = {
        ep: iter(
            np.random.default_rng(np.random.SeedSequence([seed, ep]))
            .permutation(spaced_frames(lengths[ep], per_episode))
            .tolist()
        )
        for ep in counts
    }
    return [[ep, next(frames[ep])] for ep, _ in control]


def coverage(samples, lengths, *, chunk_size: int = 50):
    if type(chunk_size) is not int or chunk_size < 1 or not lengths:
        raise ValueError("Coverage requires episode lengths and a positive chunk size")
    if any(
        type(ep) is not int or ep < 0 or type(n) is not int or n < 1 for ep, n in lengths.items()
    ):
        raise ValueError("Episode lengths must be positive integer counts")
    counts, targets = Counter(), set()
    for pair in samples:
        if len(pair) != 2 or any(type(x) is not int for x in pair):
            raise ValueError("Coverage requires integer episode/frame pairs")
        ep, frame = pair
        if ep not in lengths or not 0 <= frame < lengths[ep]:
            raise ValueError("Sample outside registered episode/frame scope")
        counts[(ep, frame)] += 1
        targets.update((ep, t) for t in range(frame, min(frame + chunk_size, lengths[ep])))
    total = sum(lengths.values())
    return {
        "sample_exposures": sum(counts.values()),
        "unique_input_frames": len(counts),
        "unique_target_frames": len(targets),
        "train_rows": total,
        "input_coverage_fraction": len(counts) / total,
        "target_coverage_fraction": len(targets) / total,
        "full_input_traversal": len(counts) == total,
        "per_episode": {
            str(ep): {
                "length": length,
                "exposures": sum(n for (e, _), n in counts.items() if e == ep),
                "unique_frames": sum(e == ep for e, _ in counts),
            }
            for ep, length in lengths.items()
        },
    }
