"""Bounded, train-only temporal visual-grounding diagnostic primitives.

No import-time models, data, network, optimizer or environment mutations.
Each counterfactual keeps the destination state, instruction and target;
only the same-offset source image changes. Noise is supplied independently.
"""

from __future__ import annotations

from typing import Any

import numpy as np

TRAIN_EPISODES = (49, 4, 23, 43, 21)
FRAME_OFFSETS = (0, 100, 250, 450)
NOISE_SEEDS = (None, 20260905, 20260906, 20260907)


def read_temporal_rows(root, config, train: list[int], hidden: list[int]):
    """Materialize only the registered train episodes and offsets at scan time."""
    import pyarrow.dataset as arrow

    if not set(TRAIN_EPISODES) <= set(train) or set(TRAIN_EPISODES) & set(hidden):
        raise ValueError("Temporal diagnostic episodes violate the train-only split.")
    if not set(TRAIN_EPISODES) <= set(config.episodes):
        raise ValueError("Temporal diagnostic episodes are outside the dataset config.")
    fields = config.fields
    table = arrow.dataset(root / "data", format="parquet").to_table(
        columns=[fields.episode_index, fields.frame_index, fields.state, fields.action],
        filter=arrow.field(fields.episode_index).isin(TRAIN_EPISODES)
        & arrow.field(fields.frame_index).isin(FRAME_OFFSETS),
    )
    rows = table.to_pylist()
    by_key = {}
    for row in rows:
        key = (int(row[fields.episode_index]), int(row[fields.frame_index]))
        if key in by_key:
            raise ValueError("Duplicate temporal sample identity.")
        for name in (fields.state, fields.action):
            vector = np.asarray(row[name], dtype=np.float64)
            if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
                raise ValueError("Temporal state/action must be a finite nonempty vector.")
        by_key[key] = row
    expected = {(episode, offset) for episode in TRAIN_EPISODES for offset in FRAME_OFFSETS}
    if set(by_key) != expected:
        raise ValueError("Temporal samples are missing or outside the registered identity.")
    return by_key


def intervention_sample(destination: dict, source: dict, camera: str) -> dict:
    """Use source RGB with destination proprioception/task, never source labels."""
    import torch

    for sample in (destination, source):
        if camera not in sample or "observation.state" not in sample or "task" not in sample:
            raise ValueError("A diagnostic sample is missing RGB, state or task.")
    image = source[camera]
    state = destination["observation.state"]
    if not isinstance(image, torch.Tensor) or not isinstance(state, torch.Tensor):
        raise ValueError("Diagnostic RGB/state must be tensors.")
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Expected one CHW RGB observation.")
    if state.ndim != 1 or not torch.isfinite(state).all():
        raise ValueError("Expected one finite state vector.")
    if image.dtype == torch.uint8:
        image = image.float() / 255.0
    elif not image.is_floating_point():
        raise ValueError("RGB must be uint8 or floating point in [0, 1].")
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("RGB is outside its finite [0, 1] contract.")
    return {
        camera: image.clone(),
        "observation.state": state.clone(),
        "task": destination["task"],
    }


def validate_image_ingress(policy: Any, batch: dict) -> dict[str, Any]:
    """Check every prepared camera on every sample, before accepting an output."""
    import torch

    images, masks = policy.prepare_images(batch)
    expected = 1 + policy.config.empty_cameras
    if len(images) != expected or len(masks) != expected:
        raise ValueError("Prepared camera count differs from the one-camera contract.")
    for index, (image, mask) in enumerate(zip(images, masks, strict=True)):
        if image.ndim != 4 or image.shape[:2] != (1, 3):
            raise ValueError("Prepared RGB must have shape [1, 3, height, width].")
        if not torch.isfinite(image).all() or image.min() < -1 or image.max() > 1:
            raise ValueError("Prepared SigLIP RGB is outside [-1, 1].")
        if mask.dtype != torch.bool or mask.shape != (1,):
            raise ValueError("Prepared camera masks must be one boolean per sample.")
        if bool(mask.item()) != (index == 0):
            raise ValueError("The real camera is masked or a placeholder is unmasked.")
    return {
        "prepared_image_shapes": [list(image.shape) for image in images],
        "camera_masks": [bool(mask.item()) for mask in masks],
    }


def summarize_offset(correct, mismatched, truth, state, dimensions) -> dict:
    """Keep paired gains by physical unit and compare state persistence explicitly."""
    from rosetta_reality.vla.vision_diagnostics import paired_alignment

    result = paired_alignment(correct, mismatched, truth)
    arrays = [np.asarray(value, dtype=np.float64) for value in (correct, mismatched, truth, state)]
    correct, mismatched, truth, state = arrays
    if state.shape != truth.shape or not np.isfinite(state).all():
        raise ValueError("State-persistence baseline requires the same finite action space.")
    if len(dimensions) != truth.shape[1]:
        raise ValueError("Action dimension metadata is missing.")
    result["state_persistence_mae"] = float(np.abs(state - truth).mean())
    result["state_persistence_per_dimension_mae"] = np.abs(state - truth).mean(axis=0).tolist()
    result["action_minus_state_per_dimension_mae"] = result["state_persistence_per_dimension_mae"]
    groups = {}
    for index, dimension in enumerate(dimensions):
        unit = dimension["unit"]
        groups.setdefault(unit, []).append(index)
    result["by_unit"] = {
        unit: {
            "dimensions": indices,
            "correct_mae": float(np.abs(correct[:, indices] - truth[:, indices]).mean()),
            "mismatched_mae": float(np.abs(mismatched[:, indices] - truth[:, indices]).mean()),
            "paired_mae_gain": float(
                (
                    np.abs(mismatched[:, indices] - truth[:, indices])
                    - np.abs(correct[:, indices] - truth[:, indices])
                ).mean()
            ),
        }
        for unit, indices in groups.items()
    }
    result["state_shuffle_degenerate"] = bool(
        np.array_equal(state, np.broadcast_to(state[0], state.shape))
    )
    return result
