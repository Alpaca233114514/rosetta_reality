"""Pure tensor diagnostics for M2 policy failure analysis."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor


def scalar_summary(values: Tensor) -> dict[str, float | int]:
    """Summarize a finite one-dimensional diagnostic vector."""

    values = values.detach().to(torch.float32).flatten()
    if values.numel() == 0:
        return {
            "samples": 0,
            "minimum": 0.0,
            "p10": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "p90": 0.0,
            "maximum": 0.0,
        }
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("Scalar diagnostic contains NaN or Inf.")
    quantiles = torch.quantile(values, torch.tensor([0.1, 0.5, 0.9]))
    return {
        "samples": int(values.numel()),
        "minimum": float(values.min()),
        "p10": float(quantiles[0]),
        "median": float(quantiles[1]),
        "mean": float(values.mean()),
        "p90": float(quantiles[2]),
        "maximum": float(values.max()),
    }


def paired_l2_distances(left: Tensor, right: Tensor) -> Tensor:
    """Return one L2 distance for each aligned pair of vectors."""

    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("Paired L2 diagnostics require matching rank-two tensors.")
    return torch.linalg.vector_norm(
        left.to(torch.float32) - right.to(torch.float32),
        dim=-1,
    )


def paired_cosine_distances(left: Tensor, right: Tensor) -> Tensor:
    """Return finite cosine distances for aligned vector pairs."""

    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("Paired cosine diagnostics require matching rank-two tensors.")
    left_normalized = torch.nn.functional.normalize(
        left.to(torch.float32), dim=-1, eps=1e-12
    )
    right_normalized = torch.nn.functional.normalize(
        right.to(torch.float32), dim=-1, eps=1e-12
    )
    return (1.0 - (left_normalized * right_normalized).sum(dim=-1)).clamp_min(0.0)


def pearson_correlation(left: Tensor, right: Tensor) -> float | None:
    """Return Pearson correlation, or ``None`` for a degenerate vector."""

    left = left.detach().to(torch.float64).flatten()
    right = right.detach().to(torch.float64).flatten()
    if left.shape != right.shape or left.numel() < 2:
        raise ValueError("Correlation diagnostics require equal non-scalar vectors.")
    if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        raise FloatingPointError("Correlation diagnostic contains NaN or Inf.")
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) <= 1e-12:
        return None
    return float((left * right).sum() / denominator)


def cross_episode_shuffle_indices(
    episode_ids: Tensor,
    *,
    frame_indices: Tensor | None = None,
    seed: int = 0,
) -> Tensor:
    """Build a deterministic feature-source permutation with no same-episode match.

    When ``frame_indices`` is supplied, every source has the same frame index as its
    destination.  This preserves the coarse trajectory phase while breaking the
    scene/action association across episodes.
    """

    if episode_ids.ndim != 1:
        raise ValueError("Episode identifiers must be rank one.")
    if frame_indices is not None and (
        frame_indices.ndim != 1 or frame_indices.shape != episode_ids.shape
    ):
        raise ValueError("Frame indices must match the episode identifier vector.")
    count = int(episode_ids.numel())
    result = torch.empty(count, dtype=torch.long)
    if count == 0:
        return result

    if frame_indices is None:
        groups = {0: list(range(count))}
    else:
        groups: dict[int, list[int]] = {}
        for index, frame in enumerate(frame_indices.tolist()):
            groups.setdefault(int(frame), []).append(index)

    episodes = [int(value) for value in episode_ids.tolist()]
    for group_key, raw_group in sorted(groups.items()):
        group = sorted(raw_group, key=lambda index: (episodes[index], index))
        size = len(group)
        if size < 2:
            raise ValueError(
                f"Cannot shuffle frame group {group_key}: fewer than two samples."
            )
        offsets = list(range(1, size))
        start = (int(seed) + int(group_key) * 1_000_003) % len(offsets)
        offsets = offsets[start:] + offsets[:start]
        selected: list[int] | None = None
        for offset in offsets:
            candidate = group[offset:] + group[:offset]
            if all(
                episodes[destination] != episodes[source]
                for destination, source in zip(group, candidate)
            ):
                selected = candidate
                break
        if selected is None:
            raise ValueError(
                f"Cannot construct a cross-episode derangement for frame group {group_key}."
            )
        for destination, source in zip(group, selected):
            result[destination] = source

    if sorted(result.tolist()) != list(range(count)):
        raise AssertionError("Cross-episode shuffle did not produce a permutation.")
    if bool(episode_ids.eq(episode_ids[result]).any()):
        raise AssertionError("Cross-episode shuffle retained a same-episode source.")
    if frame_indices is not None and not bool(
        frame_indices.eq(frame_indices[result]).all()
    ):
        raise AssertionError("Within-frame shuffle changed a frame index.")
    return result


def nearest_cross_episode_indices(
    query_states: Tensor,
    query_episode_ids: Tensor,
    reference_states: Tensor,
    reference_episode_ids: Tensor,
    *,
    batch_size: int = 256,
) -> tuple[Tensor, Tensor]:
    """Match each query to the nearest state from a different episode."""

    if (
        query_states.ndim != 2
        or reference_states.ndim != 2
        or query_states.shape[1] != reference_states.shape[1]
    ):
        raise ValueError("Nearest-state diagnostics require compatible rank-two states.")
    if (
        query_episode_ids.ndim != 1
        or reference_episode_ids.ndim != 1
        or query_episode_ids.shape[0] != query_states.shape[0]
        or reference_episode_ids.shape[0] != reference_states.shape[0]
    ):
        raise ValueError("State and episode tensors must share sample dimensions.")
    if batch_size <= 0:
        raise ValueError("Nearest-state diagnostic batch size must be positive.")
    if query_states.shape[0] == 0 or reference_states.shape[0] == 0:
        raise ValueError("Nearest-state diagnostics require non-empty tensors.")

    reference_states = reference_states.to(torch.float32)
    matched_indices: list[Tensor] = []
    matched_distances: list[Tensor] = []
    for start in range(0, query_states.shape[0], batch_size):
        stop = min(start + batch_size, query_states.shape[0])
        distances = torch.cdist(
            query_states[start:stop].to(torch.float32),
            reference_states,
        )
        same_episode = query_episode_ids[start:stop, None].eq(
            reference_episode_ids[None, :]
        )
        distances = distances.masked_fill(same_episode, torch.inf)
        values, indices = distances.min(dim=1)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("At least one query has no cross-episode reference state.")
        matched_indices.append(indices.cpu())
        matched_distances.append(values.cpu())
    return torch.cat(matched_indices), torch.cat(matched_distances)


def pairwise_l2_summary(values: Tensor) -> dict[str, float | int]:
    """Summarize unique pairwise L2 distances without counting the diagonal."""

    if values.ndim != 2:
        raise ValueError("Pairwise diagnostics require a rank-two tensor.")
    count = int(values.shape[0])
    if count < 2:
        return {"samples": count, "pairs": 0, "mean": 0.0, "maximum": 0.0}
    distances = torch.pdist(values.to(torch.float32), p=2)
    return {
        "samples": count,
        "pairs": int(distances.numel()),
        "mean": float(distances.mean()),
        "maximum": float(distances.max()),
    }


def pairwise_cosine_summary(values: Tensor) -> dict[str, float | int]:
    """Summarize unique pairwise cosine distances with finite zero-vector handling."""

    if values.ndim != 2:
        raise ValueError("Pairwise diagnostics require a rank-two tensor.")
    count = int(values.shape[0])
    if count < 2:
        return {"samples": count, "pairs": 0, "mean": 0.0, "maximum": 0.0}
    normalized = torch.nn.functional.normalize(values.to(torch.float32), dim=-1, eps=1e-12)
    matrix = 1.0 - normalized @ normalized.transpose(0, 1)
    indices = torch.triu_indices(count, count, offset=1)
    distances = matrix[indices[0], indices[1]].clamp_min(0.0)
    return {
        "samples": count,
        "pairs": int(distances.numel()),
        "mean": float(distances.mean()),
        "maximum": float(distances.max()),
    }


def action_error_summary(predicted: Tensor, target: Tensor) -> dict[str, float]:
    """Return whole-chunk and first-action errors for physical action tensors."""

    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError("Action diagnostics require matching [sample, chunk, action] tensors.")
    difference = predicted.to(torch.float32) - target.to(torch.float32)
    absolute = difference.abs()
    return {
        "chunk_mae": float(absolute.mean()),
        "chunk_rmse": float(difference.square().mean().sqrt()),
        "first_action_mae": float(absolute[:, 0].mean()),
        "first_action_rmse": float(difference[:, 0].square().mean().sqrt()),
    }


def action_dimension_diagnostics(
    predicted: Tensor,
    target: Tensor,
    lower_bounds: Tensor,
    upper_bounds: Tensor,
    dimension_names: Sequence[str],
) -> dict[str, object]:
    """Expose per-dimension and arm/gripper errors hidden by aggregate MAE."""

    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError("Action dimension diagnostics require matching rank-three tensors.")
    dimension = predicted.shape[-1]
    if (
        lower_bounds.shape != (dimension,)
        or upper_bounds.shape != (dimension,)
        or len(dimension_names) != dimension
        or len(set(dimension_names)) != dimension
    ):
        raise ValueError("Action dimension diagnostics received an invalid contract.")
    predicted = predicted.detach().to(torch.float64)
    target = target.detach().to(torch.float64)
    lower = lower_bounds.detach().to(torch.float64).view(1, 1, -1)
    upper = upper_bounds.detach().to(torch.float64).view(1, 1, -1)
    if not bool(
        torch.isfinite(predicted).all()
        and torch.isfinite(target).all()
        and torch.isfinite(lower).all()
        and torch.isfinite(upper).all()
    ):
        raise FloatingPointError("Action dimension diagnostics contain NaN or Inf.")
    if not bool((lower < upper).all()):
        raise ValueError("Action dimension diagnostics received invalid limits.")

    projected_target = torch.maximum(torch.minimum(target, upper), lower)
    prediction_violation = (predicted < lower) | (predicted > upper)
    target_projection = projected_target.ne(target)
    raw_error = predicted - target
    projected_error = predicted - projected_target
    midpoint = (lower + upper) / 2
    dimensions: dict[str, dict[str, float | int]] = {}
    for index, raw_name in enumerate(dimension_names):
        name = str(raw_name)
        predicted_dimension = predicted[..., index]
        target_dimension = target[..., index]
        raw_dimension_error = raw_error[..., index]
        projected_dimension_error = projected_error[..., index]
        entry: dict[str, float | int] = {
            "mae": float(raw_dimension_error.abs().mean()),
            "rmse": float(raw_dimension_error.square().mean().sqrt()),
            "contract_projected_target_mae": float(projected_dimension_error.abs().mean()),
            "first_action_mae": float(raw_dimension_error[:, 0].abs().mean()),
            "predicted_minimum": float(predicted_dimension.min()),
            "predicted_maximum": float(predicted_dimension.max()),
            "predicted_mean": float(predicted_dimension.mean()),
            "target_minimum": float(target_dimension.min()),
            "target_maximum": float(target_dimension.max()),
            "target_mean": float(target_dimension.mean()),
            "prediction_strict_violation_rate": float(
                prediction_violation[..., index].to(torch.float64).mean()
            ),
            "target_projection_rate": float(
                target_projection[..., index].to(torch.float64).mean()
            ),
        }
        if "gripper" in name:
            predicted_open = predicted_dimension >= midpoint[..., index]
            target_open = projected_target[..., index] >= midpoint[..., index]
            entry.update(
                {
                    "open_close_accuracy": float(
                        predicted_open.eq(target_open).to(torch.float64).mean()
                    ),
                    "predicted_open_rate": float(predicted_open.to(torch.float64).mean()),
                    "target_open_rate": float(target_open.to(torch.float64).mean()),
                    "predicted_below_minimum_rate": float(
                        predicted_dimension.lt(lower[..., index]).to(torch.float64).mean()
                    ),
                    "predicted_above_maximum_rate": float(
                        predicted_dimension.gt(upper[..., index]).to(torch.float64).mean()
                    ),
                }
            )
        dimensions[name] = entry

    groups: dict[str, list[int]] = {
        "left_arm": [
            index
            for index, name in enumerate(dimension_names)
            if str(name).startswith("left_") and "gripper" not in str(name)
        ],
        "left_gripper": [
            index
            for index, name in enumerate(dimension_names)
            if str(name) == "left_gripper"
        ],
        "right_arm": [
            index
            for index, name in enumerate(dimension_names)
            if str(name).startswith("right_") and "gripper" not in str(name)
        ],
        "right_gripper": [
            index
            for index, name in enumerate(dimension_names)
            if str(name) == "right_gripper"
        ],
    }
    if any(not indices for indices in groups.values()):
        raise ValueError("Action dimension names do not expose both ALOHA arms and grippers.")
    group_diagnostics = {
        name: {
            "dimensions": [str(dimension_names[index]) for index in indices],
            "mae": float(raw_error[..., indices].abs().mean()),
            "contract_projected_target_mae": float(
                projected_error[..., indices].abs().mean()
            ),
            "first_action_mae": float(raw_error[:, 0, indices].abs().mean()),
        }
        for name, indices in groups.items()
    }
    return {
        "sample_count": int(predicted.shape[0]),
        "chunk_length": int(predicted.shape[1]),
        "action_dimension": int(dimension),
        "dimensions": dimensions,
        "groups": group_diagnostics,
    }


def phase_labels(frame_indices: Tensor, boundaries: Sequence[int]) -> list[str]:
    """Assign deterministic half-open phase labels to frame indices."""

    if frame_indices.ndim != 1:
        raise ValueError("Frame indices must be rank one.")
    ordered = [int(value) for value in boundaries]
    if not ordered or ordered[0] <= 0 or any(
        current <= previous for previous, current in zip(ordered, ordered[1:])
    ):
        raise ValueError("Phase boundaries must be strictly increasing positive integers.")
    labels: list[str] = []
    for raw_frame in frame_indices.tolist():
        frame = int(raw_frame)
        lower = 0
        assigned = False
        for upper in ordered:
            if frame < upper:
                labels.append(f"{lower:03d}-{upper - 1:03d}")
                assigned = True
                break
            lower = upper
        if not assigned:
            labels.append(f"{ordered[-1]:03d}+")
    return labels


def finite_number(value: float) -> float:
    """Reject non-finite values before they enter a JSON diagnostic artifact."""

    if not math.isfinite(value):
        raise FloatingPointError("Diagnostic calculation produced NaN or Inf.")
    return value
