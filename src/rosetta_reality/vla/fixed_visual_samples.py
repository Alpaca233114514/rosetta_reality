"""Explicit cross-episode samples for bounded native-loss visual overfit checks.

Only the existing smoke entry may use this protocol. Formal training and
historical single-episode sampling retain their original contracts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def validate_visual_samples(
    raw: Any, plan: Mapping[str, Any], experiment: Mapping[str, Any], phase: str
) -> tuple[tuple[int, int], ...]:
    """Reject implicit samples, split leakage and use outside bounded diagnostics."""
    if phase != "smoke" or plan.get("scope") != "bounded_visual_overfit":
        raise ValueError("Explicit visual samples require a bounded visual smoke plan.")
    if not isinstance(raw, list) or not raw:
        raise ValueError("Explicit visual samples must be a nonempty list.")
    samples = []
    for item in raw:
        if (
            not isinstance(item, dict)
            or set(item) != {"episode", "frame"}
            or any(type(item[k]) is not int or item[k] < 0 for k in item)
        ):
            raise ValueError("Visual samples require nonnegative integer episode/frame.")
        samples.append((item["episode"], item["frame"]))
    if len(set(samples)) != len(samples):
        raise ValueError("Duplicate visual sample identity.")
    selected = {ep for ep, _ in samples}
    dataset = experiment["dataset"]
    forbidden = set(dataset["validation_episodes"]) | set(dataset["test_episodes"])
    if not selected <= set(dataset["train_episodes"]) or selected & forbidden:
        raise ValueError("Visual samples cross the registered training boundary.")
    if selected != set(plan["optimizer_smoke"]["episodes"]):
        raise ValueError("Sample episodes differ from the active smoke scope.")
    return tuple(samples)


def resolve_visual_sample_indices(
    samples: Sequence[tuple[int, int]],
    starts: Sequence[int],
    stops: Sequence[int],
    active_episodes: Sequence[int] | None,
    absolute_to_relative: Mapping[int, int] | None,
) -> list[int]:
    """Map absolute episode/frame identities through the native dataset view."""
    if len(starts) != len(stops) or active_episodes is None:
        raise ValueError("Dataset episode boundaries are incomplete.")
    selected = {ep for ep, _ in samples}
    if set(active_episodes) != selected or len(active_episodes) != len(selected):
        raise ValueError("Active dataset episodes differ from the fixed sample scope.")
    indices = []
    for episode, frame in samples:
        if episode < 0 or episode >= len(starts) or frame < 0:
            raise ValueError("Visual sample is outside the dataset metadata.")
        start, stop = int(starts[episode]), int(stops[episode])
        absolute = start + frame
        if start < 0 or stop <= start or absolute >= stop:
            raise ValueError("Visual sample is outside its episode.")
        if absolute_to_relative is None:
            index = absolute
        else:
            if absolute not in absolute_to_relative:
                raise ValueError("Visual sample is absent from the active view.")
            index = int(absolute_to_relative[absolute])
        if index < 0:
            raise ValueError("Visual sample resolved to a negative dataset index.")
        indices.append(index)
    if len(set(indices)) != len(indices):
        raise ValueError("Visual sample indices are not unique.")
    return indices
