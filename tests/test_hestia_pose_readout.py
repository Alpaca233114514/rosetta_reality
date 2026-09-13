"""Counterexamples for held-label leakage, indexing, and scene weighting."""

import numpy as np
import pytest

from scripts.diagnose_hestia_pose_readout import crossfit, describe


def fixture():
    rng = np.random.default_rng(71)
    x = rng.normal(size=(13, 4))
    y = rng.normal(size=(13, 2, 3))
    return x, y


def test_development_labels_cannot_change_predictions_or_alphas():
    x, y = fixture()
    a, fa = crossfit(x, y, 11, {"a": [0, 2], "b": [1]}, [0.1, 10], 7)
    y[11:] += 1000
    b, fb = crossfit(x, y, 11, {"a": [0, 2], "b": [1]}, [0.1, 10], 7)
    np.testing.assert_array_equal(a, b)
    assert fa == fb


def test_outer_held_label_does_not_leak_to_own_prediction():
    x, y = fixture()
    a, fa = crossfit(x, y, 11, {"all": [0, 1, 2]}, [0.1, 10], 7)
    y[3] += 500
    b, fb = crossfit(x, y, 11, {"all": [0, 1, 2]}, [0.1, 10], 7)
    np.testing.assert_array_equal(a[3], b[3])
    assert fa[3] == fb[3]
    assert 3 not in fa[3]["donor_rows"]


def test_group_column_order_and_constant_targets():
    x, y = fixture()
    y[:] = [9, -7, 4]
    a, _ = crossfit(x, y, 11, {"a": [2, 0], "b": [1]}, [0.1, 10], 7)
    np.testing.assert_allclose(a, y, atol=1e-12)


def test_duplicate_group_dimension_rejected():
    x, y = fixture()
    with pytest.raises(ValueError, match="partition"):
        crossfit(x, y, 11, {"a": [0, 1], "b": [1, 2]}, [1], 7)


def test_nonfinite_rejected():
    x, y = fixture()
    y[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        crossfit(x, y, 11, {"all": [0, 1, 2]}, [1], 7)


def test_scene_average_keeps_all_development_rows():
    a = np.zeros((45, 8))
    a[40:] = np.arange(5)[:, None]
    r = describe(a, list(range(45)))
    assert r["dev5"] == 2
    assert r["dev_per_episode"]["44"] == 4
    assert len(r["dev_per_episode"]) == 5
