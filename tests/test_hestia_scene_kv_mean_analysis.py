"""Paired-sign and noise-order counterexamples for intervention postprocessing."""

import numpy as np
import pytest

from scripts.analyze_hestia_scene_kv_mean import paired_effect


def test_correct_recovery_and_wrong_direction_have_opposite_signs():
    y = np.ones((2, 3, 1))
    base = np.zeros((2, 2, 3, 1))
    treatment = base.copy()
    treatment[0] = 1
    treatment[1] = -1
    value = paired_effect(base, treatment, y)
    assert value["mse_delta_by_noise"] == [-1, 3]
    assert value["mae_delta_by_noise"] == [-1, 1]
    assert value["movement_squared_by_noise"] == [1, 1]
    assert value["minus_twice_residual_alignment_by_noise"] == [-2, 2]


def test_invalid_pair_rejected():
    p = np.zeros((2, 2, 3, 1))
    with pytest.raises(ValueError, match="shapes"):
        paired_effect(p, p[:, :, :-1], p[0])
    y = p[0].copy()
    y[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        paired_effect(p, p, y)
