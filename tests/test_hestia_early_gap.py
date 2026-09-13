"""Counterexamples for sign, train-only template and per-noise gap decomposition."""

import numpy as np
import pytest

from scripts.diagnose_hestia_early_gap import decompose


@pytest.mark.parametrize("sign,bias,gap", [(-1, 0, 3), (1, 0, -1), (1, 3, 8)])
def test_known_covariance_and_bias(sign, bias, gap):
    y = np.array([-1.0, 1.0])[:, None, None]
    p = (sign * y + bias)[None]
    r = decompose(p, y, y, p)
    assert r["model_minus_train_mean_mse"]["mean"] == gap
    assert r["prediction_scene_variance"]["mean"] == 1
    assert r["minus_twice_scene_covariance"]["mean"] == -2 * sign
    assert r["squared_scene_mean_bias_difference"]["mean"] == bias**2


def test_template_uses_train_outputs_without_noise_cancellation():
    y = np.zeros((2, 3, 1))
    p = np.zeros((2, 2, 3, 1))
    tp = np.stack([np.full((4, 3, 1), 2.0), np.full((4, 3, 1), -2.0)])
    r = decompose(p, y, np.zeros((4, 3, 1)), tp)
    assert r["errors"]["train_prediction_template"]["mae"]["by_noise"] == [2, 2]


def test_unaligned_or_nonfinite_rejected():
    y = np.zeros((2, 3, 1))
    with pytest.raises(ValueError, match="aligned"):
        decompose(y[None], y, y, y[None, :, :-1])
    p = y[None].copy()
    p[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="aligned"):
        decompose(p, y, y, y[None])
