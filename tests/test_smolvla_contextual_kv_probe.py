"""Synthetic checks for spatial identity, train-only selection and causal metrics."""

import numpy as np
import pytest

from rosetta_reality.vla.contextual_kv_probe import (
    compare_depths,
    observe_projection_input,
    visual_kv_features,
)


def test_projection_observer_preserves_output_and_rejects_changing_prefix():
    import torch

    module = torch.nn.Linear(3, 2)
    sample = torch.arange(12.0).reshape(1, 4, 3)
    original = sample.clone()
    expected = module(sample)
    storage, counts = {}, {}
    handle = module.register_forward_pre_hook(observe_projection_input(storage, counts, "early_k"))
    try:
        assert torch.equal(module(sample), expected)
        assert torch.equal(module(sample), expected)
        assert torch.equal(sample, original)
        assert counts == {"early_k": 2}
        with pytest.raises(ValueError, match="changed"):
            module(sample + 1)
    finally:
        handle.remove()
    assert torch.equal(module(sample), expected)


def test_visual_slice_excludes_language_padding_and_state():
    keys = np.arange(12.0).reshape(1, 6, 2)
    values = keys + 20
    first = visual_kv_features(keys, values, grid=(2, 2), bins=2)
    keys[:, 4:] = 1000
    values[:, 4:] = -1000
    np.testing.assert_array_equal(first, visual_kv_features(keys, values, grid=(2, 2), bins=2))
    assert first.shape == (16,)
    assert not np.array_equal(
        first, visual_kv_features(keys[:, ::-1], values[:, ::-1], grid=(2, 2), bins=2)
    )


@pytest.mark.parametrize("shape", [(2, 8, 3), (1, 3, 3), (8, 3)])
def test_wrong_batch_or_short_prefix_fails(shape):
    with pytest.raises(ValueError):
        visual_kv_features(np.zeros(shape), np.zeros(shape), grid=(2, 2), bins=2)


def _comparison(early, late, y):
    return compare_depths(
        early,
        late,
        y,
        train_count=30,
        groups={"radian": [0], "normalized": [1]},
        alphas=(0.001, 1.0, 100.0),
        seed=20260911,
        epsilon=1e-8,
    )


def test_development_labels_do_not_choose_alpha_or_change_train_predictions():
    rng = np.random.default_rng(12)
    early, late = rng.normal(size=(35, 8)), rng.normal(size=(35, 8))
    y = early[:, :2] * 0.1
    original = _comparison(early, late, y)
    changed = y.copy()
    changed[30:] += 100
    altered = _comparison(early, late, changed)
    assert original["alpha"] == altered["alpha"]
    for arm in ("early", "late"):
        assert original["arms"][arm]["predictions"] == altered["arms"][arm]["predictions"]
        assert original["arms"][arm]["train_fit"] == altered["arms"][arm]["train_fit"]


def test_identical_representations_cannot_claim_depth_improvement():
    rng = np.random.default_rng(13)
    x = rng.normal(size=(35, 8))
    result = _comparison(x, x.copy(), x[:, :2])
    assert result["arms"]["early"] == result["arms"]["late"]
    assert not result["late_readout_criteria_passed"]
    assert result["gating"] is False


def test_actual_signal_beats_uninformative_early_features():
    rng = np.random.default_rng(14)
    late = rng.normal(size=(35, 2))
    early = np.zeros_like(late)
    result = _comparison(early, late, late * 0.1)
    assert result["late_readout_criteria_passed"]
    for group in ("radian", "normalized"):
        metric = result["arms"]["late"]["development"][group]
        assert metric["paired_mae_gain"] > 0
        expected = (
            metric["target_scene_variance"]
            + metric["prediction_scene_variance"]
            - 2 * metric["scene_covariance"]
            + metric["squared_mean_bias"]
        )
        assert metric["mse"] == pytest.approx(expected, abs=1e-12)


def test_unequal_width_and_missing_physical_group_are_rejected():
    x = np.zeros((35, 8))
    y = np.zeros((35, 2))
    with pytest.raises(ValueError, match="equal-width"):
        _comparison(x, x[:, :4], y)
    with pytest.raises(ValueError, match="partition"):
        compare_depths(
            x,
            x,
            y,
            train_count=30,
            groups={"radian": [0]},
            alphas=(1.0,),
            seed=1,
            epsilon=1e-8,
        )
