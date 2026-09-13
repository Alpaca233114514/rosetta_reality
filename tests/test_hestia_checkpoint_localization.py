"""Counterexamples for the saved-array localization diagnostic."""

import numpy as np
import pytest

from scripts.diagnose_hestia_checkpoint_localization import (
    compare,
    split_rows,
    transitions,
    validate_layout,
)


def test_opposed_noise_errors_must_not_cancel_before_mae():
    y = np.zeros((3, 4, 2))
    p = np.stack([np.ones_like(y), -np.ones_like(y)])
    result = compare(p, p / 2, y, y)
    assert result["640"]["mae"]["mean"] == 1
    assert result["1280"]["mae"]["mean"] == 0.5
    assert result["change"]["mae"]["mean"] == -0.5
    assert result["640"]["scene_correlation_by_noise"] == [None, None]


def test_larger_response_can_worsen_scene_correspondence():
    y = np.array([-1.0, 0.0, 1.0])[None, :, None, None]
    p, q = 0.5 * y, -y
    result = compare(p, q, y[0], y[0])
    assert result["640"]["scene_correlation_by_noise"] == [1.0]
    assert result["1280"]["scene_correlation_by_noise"] == [-1.0]
    assert result["change"]["mse"]["mean"] > 0
    assert result["change"]["minus_twice_covariance"]["mean"] > 0


def test_paired_delta_identity_for_nonzero_bias_and_noise():
    rng = np.random.default_rng(19)
    p, q = rng.normal(size=(2, 4, 5, 3, 2))
    y = rng.normal(size=(5, 3, 2))
    result = compare(p, q, y, rng.normal(size=(8, 3, 2)))
    c = result["change"]
    np.testing.assert_allclose(
        c["mse"]["by_noise"],
        np.array(c["prediction_variance"]["by_noise"])
        + c["minus_twice_covariance"]["by_noise"]
        + np.array(c["squared_mean_bias"]["by_noise"]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        c["mse"]["by_noise"],
        np.array(c["squared_prediction_movement"]["by_noise"])
        + c["minus_twice_residual_alignment"]["by_noise"],
        atol=1e-12,
    )


def test_nonfinite_or_unaligned_inputs_rejected():
    p = np.zeros((2, 3, 4, 5))
    with pytest.raises(ValueError, match="aligned"):
        compare(p, p, p[0, :-1], p[0])
    p[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="aligned"):
        compare(p, p, np.zeros((3, 4, 5)), np.zeros((3, 4, 5)))


def test_split_lookup_uses_episode_identity_and_rejects_hidden():
    meta = {
        "episodes": list(range(44, -1, -1)),
        "views": {"train40": list(range(40)), "dev5": list(range(40, 45))},
        "hidden_episodes": list(range(45, 50)),
        "hidden_test_loaded": False,
    }
    rows = split_rows(meta)
    assert rows["dev5"] == [4, 3, 2, 1, 0]
    meta["hidden_episodes"].append(44)
    with pytest.raises(ValueError, match="Hidden row"):
        split_rows(meta)


def test_event_censoring_and_recrossings_are_retained():
    result = transitions([0.9, 0.5, 0.2, 0.8, 0.1], 0.5)
    assert result["downcross_slots"] == [2, 4]
    assert result["upcross_slots"] == [3]
    assert not result["starts_below_threshold"]
    result = transitions([0.1, 0.1, 0.1], 0.5)
    assert result["starts_below_threshold"]
    assert result["downcross_slots"] == []


def test_native_noise_width_is_distinct_from_saved_physical_width():
    meta = {
        "episodes": [8, 2, 9],
        "chunk_length": 5,
        "dimensions": [{}, {}],
        "noise_conditions": [None, 1],
        "max_action_dim": 4,
    }
    arrays = {"noise": np.zeros((2, 1, 5, 4))}
    for space in ("standard", "normalized"):
        arrays[space + "_predictions"] = np.zeros((2, 3, 5, 2))
        arrays[space + "_targets"] = np.zeros((3, 5, 2))
    mask = np.ones((3, 5, 2), dtype=bool)
    validate_layout(arrays, meta, mask)
    with pytest.raises(ValueError, match="physical-action boolean mask"):
        validate_layout(arrays, meta, np.ones((3, 5, 4), dtype=bool))
    mask[0, 0, 0] = False
    with pytest.raises(ValueError, match="physical-action boolean mask"):
        validate_layout(arrays, meta, mask)
    mask[:] = True
    arrays["normalized_predictions"] = np.zeros((2, 3, 5, 4))
    with pytest.raises(ValueError, match="physical-action width"):
        validate_layout(arrays, meta, mask)
