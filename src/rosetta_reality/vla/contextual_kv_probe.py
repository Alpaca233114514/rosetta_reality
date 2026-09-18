"""Paired early/late visual-KV readouts; no model, data or network at import."""

from __future__ import annotations

from typing import Any

import numpy as np

from rosetta_reality.vla.vision_diagnostics import fit_probe, ridge_predict, spatial_features


def observe_projection_input(storage: dict, counts: dict, name: str):
    """Return an observation-only PyTorch pre-hook; reject changing prefix inputs."""

    def hook(_module, inputs):
        values = inputs[0].detach().float().cpu().numpy().copy()
        if name in storage and not np.array_equal(values, storage[name]):
            raise ValueError("Expert prefix KV changed across denoising steps.")
        storage[name] = values
        counts[name] = counts.get(name, 0) + 1
        return None

    return hook


def visual_kv_features(
    keys: np.ndarray, values: np.ndarray, *, grid: tuple[int, int], bins: int
) -> np.ndarray:
    """Pool only the leading real-camera tokens, excluding language/padding/state.

    Inputs are the native expert k_proj/v_proj inputs, [1, prefix, KV width],
    before the expert's own projection. K already includes the VLM's RoPE.
    The caller must prove the camera is first, unmasked, with no special tokens.
    """
    keys, values = np.asarray(keys), np.asarray(values)
    count = grid[0] * grid[1]
    if (
        keys.ndim != 3
        or keys.shape != values.shape
        or keys.shape[0] != 1
        or keys.shape[1] < count
        or keys.shape[2] == 0
        or count <= 0
        or not np.isfinite(keys).all()
        or not np.isfinite(values).all()
    ):
        raise ValueError("Expected matching finite batch-one prefix K/V arrays.")
    return spatial_features(
        np.concatenate((keys[0, :count], values[0, :count]), axis=-1), grid, bins=bins
    )


def _metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    reference: np.ndarray,
    groups: dict[str, list[int]],
) -> dict[str, Any]:
    result = {}
    for name, indices in groups.items():
        pred, target = prediction[:, indices], truth[:, indices]
        error = pred - target
        # Average each nonself image's error, never average predictions first.
        pair_error = np.abs(pred[None, :, :] - target[:, None, :]).mean(axis=-1)
        nonself = ~np.eye(len(target), dtype=bool)
        centered_pred, centered_target = pred - pred.mean(0), target - target.mean(0)
        covariance = float((centered_pred * centered_target).mean())
        pred_variance = float(np.square(centered_pred).mean())
        target_variance = float(np.square(centered_target).mean())
        denominator = np.sqrt(pred_variance * target_variance)
        result[name] = {
            "dimensions": indices,
            "mae": float(np.abs(error).mean()),
            "mse": float(np.square(error).mean()),
            "per_episode_mae": np.abs(error).mean(-1).tolist(),
            "per_dimension_mae": np.abs(error).mean(0).tolist(),
            "all_nonself_image_mae": float(pair_error[nonself].mean()),
            "paired_mae_gain": float(pair_error[nonself].mean() - np.abs(error).mean()),
            "train_mean_baseline_mae": float(np.abs(target - reference[:, indices].mean(0)).mean()),
            "train_median_baseline_mae": float(
                np.abs(target - np.median(reference[:, indices], axis=0)).mean()
            ),
            "prediction_scene_variance": pred_variance,
            "target_scene_variance": target_variance,
            "scene_covariance": covariance,
            "scene_correlation": float(covariance / denominator) if denominator > 1e-15 else None,
            "squared_mean_bias": float(np.square(pred.mean(0) - target.mean(0)).mean()),
        }
    return result


def compare_depths(
    early: np.ndarray,
    late: np.ndarray,
    actions: np.ndarray,
    *,
    train_count: int,
    groups: dict[str, list[int]],
    alphas: tuple[float, ...],
    seed: int,
    epsilon: float,
) -> dict[str, Any]:
    """Change only extraction depth; both arms share early-train-selected alpha.

    The same train rows fit centering/scaling and ridge coefficients separately
    in each representation. Development labels never choose alpha or a layer.
    Alpha selection uses the historical mixed-unit MAE proxy; conclusions require
    separate joint/gripper metrics and are explicitly non-gating.
    """
    early, late, actions = (np.asarray(x, dtype=np.float64) for x in (early, late, actions))
    if (
        early.ndim != 2
        or early.shape != late.shape
        or early.shape[1] == 0
        or actions.ndim != 2
        or len(actions) != len(early)
        or not 5 <= train_count <= len(actions) - 2
        or any(not np.isfinite(x).all() for x in (early, late, actions))
        or not np.isfinite(epsilon)
        or epsilon <= 0
    ):
        raise ValueError(
            "Readouts require equal-width finite features and disjoint evaluation rows."
        )
    indices = [index for group in groups.values() for index in group]
    if (
        not groups
        or any(not group for group in groups.values())
        or len(indices) != len(set(indices))
        or set(indices) != set(range(actions.shape[1]))
    ):
        raise ValueError("Physical-unit groups must partition the action dimensions.")
    selection = fit_probe(early, actions, train_count, alphas=alphas, seed=seed)
    alpha = selection["alpha"]
    train_indices = np.arange(train_count)
    folds = np.array_split(np.random.default_rng(seed).permutation(train_indices), 5)
    arms = {}
    for name, features in (("early", early), ("late", late)):
        prediction = ridge_predict(features[:train_count], actions[:train_count], features, alpha)
        oof = np.empty_like(actions[:train_count])
        for held in folds:
            fitted = np.setdiff1d(train_indices, held)
            oof[held] = ridge_predict(features[fitted], actions[fitted], features[held], alpha)
        arms[name] = {
            "train_fit": _metrics(
                prediction[:train_count], actions[:train_count], actions[:train_count], groups
            ),
            # Each OOF row comes from a different fitted fold model. Swapping
            # these rows would change both image and model, so it is not an
            # image intervention and must not produce a visual-gain claim.
            "train_oof_tuning": {
                group: {
                    "mae": float(np.abs(oof[:, dims] - actions[:train_count, dims]).mean()),
                    "mse": float(np.square(oof[:, dims] - actions[:train_count, dims]).mean()),
                }
                for group, dims in groups.items()
            },
            "development": _metrics(
                prediction[train_count:], actions[train_count:], actions[:train_count], groups
            ),
            "predictions": prediction.tolist(),
            "train_oof_predictions": oof.tolist(),
        }
    comparison = {}
    for group in groups:
        control = arms["early"]["development"][group]
        treatment = arms["late"]["development"][group]
        comparison[group] = {
            "late_minus_early_mae": treatment["mae"] - control["mae"],
            "late_better_than_early": treatment["mae"] < control["mae"] - epsilon,
            "early_beats_both_constants": control["mae"]
            < min(control["train_mean_baseline_mae"], control["train_median_baseline_mae"])
            - epsilon,
            "late_beats_both_constants": treatment["mae"]
            < min(treatment["train_mean_baseline_mae"], treatment["train_median_baseline_mae"])
            - epsilon,
            "late_positive_visual_gain": treatment["paired_mae_gain"] > epsilon,
        }
    return {
        "alpha": alpha,
        "alpha_selected_from": "early_layer_train_folds_only_shared_by_both_arms",
        "alpha_candidates": list(alphas),
        "early_train_cv_mae_by_alpha": selection["train_cv_mae_by_alpha"],
        "train_fold_seed": seed,
        "folds": [fold.tolist() for fold in folds],
        "feature_width": early.shape[1],
        "arms": arms,
        "comparison": comparison,
        "late_readout_criteria_passed": all(
            item["late_better_than_early"]
            and item["late_beats_both_constants"]
            and item["late_positive_visual_gain"]
            for item in comparison.values()
        ),
        "gating": False,
        "policy_improvement": "not measured",
        "task_success": "not measured",
        "m2_complete": False,
    }
