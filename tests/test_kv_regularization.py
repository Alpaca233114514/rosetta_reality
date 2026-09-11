"""Leakage and arithmetic checks for the saved-feature regularization diagnostic."""

import numpy as np
import pytest

from rosetta_reality.vla.vision_diagnostics import ridge_predict
from scripts.diagnose_kv_regularization import compare, nested_predictions


def test_development_labels_cannot_change_fits_or_alpha():
    rng = np.random.default_rng(24)
    x, y = rng.normal(size=(22, 8)), rng.normal(size=(22, 2))
    kwargs = dict(train_count=20, groups={"a": [0], "b": [1]}, grids={"a": [1, 1000]}, seed=9)
    before, pa = compare(x, y, **kwargs)
    y[20:] += 100
    after, pb = compare(x, y, **kwargs)
    assert before["a"]["alpha"] == after["a"]["alpha"]
    assert before["a"]["outer_folds"] == after["a"]["outer_folds"]
    for key in pa:
        np.testing.assert_array_equal(pa[key], pb[key])


def test_outer_held_labels_do_not_enter_their_fit():
    rng = np.random.default_rng(31)
    x, y = rng.normal(size=(20, 8)), rng.normal(size=(20, 2))
    before, bm, bd, folds = nested_predictions(x, y, [1, 100], 7)
    held = folds[0]["held_rows"]
    y[held] += 1000
    after, am, ad, new_folds = nested_predictions(x, y, [1, 100], 7)
    assert folds[0] == new_folds[0]
    for a, b in ((before, after), (bm, am), (bd, ad)):
        np.testing.assert_array_equal(a[held], b[held])


def test_strong_ridge_tends_to_train_mean():
    rng = np.random.default_rng(5)
    x, y = rng.normal(size=(20, 8)), rng.normal(size=(20, 2))
    pred = ridge_predict(x, y, x[:3] + 2, 1e14)
    np.testing.assert_allclose(pred, np.broadcast_to(y.mean(0), pred.shape), atol=1e-10)


@pytest.mark.parametrize("grid", [[], [0], [float("nan")], [-1]])
def test_invalid_grid_rejected(grid):
    with pytest.raises(ValueError, match="Alpha grids"):
        compare(
            np.ones((12, 3)),
            np.ones((12, 1)),
            train_count=10,
            groups={"a": [0]},
            grids={"a": grid},
            seed=1,
        )
