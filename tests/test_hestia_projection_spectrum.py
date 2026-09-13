import numpy as np
import pytest

from scripts.diagnose_hestia_projection_spectrum import spectrum


def test_rank_defect_is_not_a_finite_condition():
    m, s = spectrum(np.diag([2.0, 1.0, 0.0]))
    assert m["rank"] == 2 and m["condition"] is None
    np.testing.assert_array_equal(s, [2, 1, 0])


def test_identity_and_scale_invariance():
    a, _ = spectrum(np.eye(5))
    b, _ = spectrum(17 * np.eye(5))
    assert a["rank"] == b["rank"] == 5
    assert a["stable_rank"] == b["stable_rank"] == 5
    assert a["condition"] == b["condition"] == 1


def test_rotated_known_spectrum():
    q, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(6, 6)))
    expected = np.array([7, 5, 2, 1, 0.5, 0.01])
    m, s = spectrum(q @ np.diag(expected) @ q.T)
    np.testing.assert_allclose(s, expected, atol=1e-12, rtol=0)
    assert m["rank"] == 6 and m["dimensions_below_one_percent_max"] == 1


def test_zero_matrix_remains_rank_zero():
    m, _ = spectrum(np.zeros((4, 4)))
    assert m["rank"] == 0 and m["condition"] is None and m["stable_rank"] == 0


def test_nonfinite_is_rejected():
    with pytest.raises(ValueError, match="Finite"):
        spectrum([[np.nan]])
