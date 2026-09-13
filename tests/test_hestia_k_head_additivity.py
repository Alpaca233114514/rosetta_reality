import numpy as np
import pytest

from scripts.analyze_hestia_k_head_additivity import compare_changes


def test_exact_additivity_and_zero_denominator():
    singles = np.arange(5 * 4 * 3, dtype=np.float64).reshape(5, 4, 3)
    result = compare_changes(singles.sum(0), singles)
    assert result["approximately_additive"]
    assert result["residual_fraction"] == [0.0] * 4
    zero = compare_changes(np.zeros((4, 3)), np.zeros((5, 4, 3)))
    assert not zero["approximately_additive"]
    assert zero["residual_fraction"] == [None] * 4


def test_nonlinear_scaling_and_nonfinite_fail_closed():
    singles = np.ones((5, 4, 3))
    result = compare_changes(singles.sum(0) * 2, singles)
    assert result["residual_fraction"] == [0.25] * 4
    assert not result["approximately_additive"]
    singles[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        compare_changes(np.ones((4, 3)), singles)
