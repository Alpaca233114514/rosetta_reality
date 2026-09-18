"""Counterexamples for shared-kernel probes and native spatial pooling."""

import numpy as np
import pytest

from rosetta_reality.vla.vision_diagnostics import ridge_predict
from scripts.diagnose_hestia_prefix_readout import linear_weights, multi_crossfit, pool
from scripts.diagnose_kv_regularization import select_alpha


@pytest.mark.parametrize("width", [3, 70])
def test_shared_weights_match_original_ridge(width):
    rng = np.random.default_rng(52)
    x, q, y = rng.normal(size=(13, width)), rng.normal(size=(4, width)), rng.normal(size=(13, 7))
    x[:, 0] = q[:, 0] = 2
    alphas = [0.001, 1, 10000]
    actual = linear_weights(x, q, alphas) @ y
    for i, a in enumerate(alphas):
        np.testing.assert_allclose(actual[i], ridge_predict(x, y, q, a), atol=1e-9, rtol=1e-9)


def test_nested_matches_independent_old_solver_and_isolates_labels():
    rng = np.random.default_rng(17)
    x, y = rng.normal(size=(13, 17)), rng.normal(size=(13, 5))
    groups = {"a": [0, 3], "b": [1, 2, 4]}
    alphas = [0.001, 1, 10000]
    seed = 29
    p, means, medians, records = multi_crossfit(x, y, groups, alphas, seed, 10)
    for row in (3, 10):
        donors = np.array([j for j in range(10) if j != row])
        for name, cols in groups.items():
            alpha, scores = select_alpha(
                x[donors], y[donors][:, cols], alphas, seed + row if row < 10 else seed
            )
            assert records[row]["groups"][name]["alpha"] == alpha
            np.testing.assert_allclose(records[row]["groups"][name]["inner_mae"], scores, atol=1e-9)
            np.testing.assert_allclose(
                p[row, cols],
                ridge_predict(x[donors], y[donors][:, cols], x[row : row + 1], alpha)[0],
                atol=1e-9,
            )
        np.testing.assert_allclose(means[row], y[donors].mean(0))
        np.testing.assert_allclose(medians[row], np.median(y[donors], axis=0))
    changed = y.copy()
    changed[3] += 1000
    changed[10:] -= 1000
    pp, _, _, rr = multi_crossfit(x, changed, groups, alphas, seed, 10)
    np.testing.assert_array_equal(p[3], pp[3])
    assert records[3] == rr[3]
    changed = y.copy()
    changed[10:] += 1000
    pp, _, _, rr = multi_crossfit(x, changed, groups, alphas, seed, 10)
    np.testing.assert_array_equal(p, pp)
    assert records == rr


def test_constant_features_give_donor_mean():
    y = np.arange(39, dtype=float).reshape(13, 3)
    p, m, _, _ = multi_crossfit(np.ones((13, 4)), y, {"all": [0, 1, 2]}, [1, 10], 3, 10)
    np.testing.assert_allclose(p, m)


def test_ordered_pooling_preserves_quadrants_and_kv_order():
    k = np.broadcast_to(np.arange(64).reshape(1, 64, 1), (1, 64, 320)).copy()
    v = k + 100
    full = pool(k, v, 1)
    np.testing.assert_array_equal(full, np.r_[np.full(320, 31.5), np.full(320, 131.5)])
    q = pool(k, v, 2).reshape(4, 640)
    for row, expected in enumerate((13.5, 17.5, 45.5, 49.5)):
        np.testing.assert_array_equal(q[row, :320], np.full(320, expected))
        np.testing.assert_array_equal(q[row, 320:], np.full(320, expected + 100))


def test_rejects_bad_groups_and_nonfinite():
    x = np.ones((13, 3))
    y = np.ones((13, 2))
    with pytest.raises(ValueError):
        multi_crossfit(x, y, {"bad": [0, 0]}, [1], 1, 10)
    y[2, 1] = np.nan
    with pytest.raises(ValueError):
        multi_crossfit(x, y, {"all": [0, 1]}, [1], 1, 10)
