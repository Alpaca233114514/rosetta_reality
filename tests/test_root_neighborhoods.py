"""Counterexamples for saved-array neighborhood analysis, no model/data loading."""

import numpy as np
import pytest

from scripts.analyze_root_neighborhoods import crossing, neighbors


def test_donors_exclude_self_and_development_even_when_closer():
    episodes = [2, 49, 4, 23, 13, 33]
    x = np.array([[0], [1], [2], [3], [0], [0]], dtype=float)
    near, _, _ = neighbors(x, episodes, episodes[:4], standardize=True)
    for i, donors in enumerate(near):
        assert set(donors) <= {0, 1, 2, 3}
        assert i not in donors
    assert near[4].tolist() == [0, 1, 2]


def test_development_values_cannot_change_train_scaling_or_neighbors():
    x = np.array([[0, 1], [1, 1], [2, 1], [3, 1], [4, 1]], dtype=float)
    a, _, scales = neighbors(x, [2, 49, 4, 23, 13], [2, 49, 4, 23], standardize=True)
    x[4] = [1e9, -1e9]
    b, _, new_scales = neighbors(x, [2, 49, 4, 23, 13], [2, 49, 4, 23], standardize=True)
    np.testing.assert_array_equal(a[:4], b[:4])
    assert scales == new_scales


def test_missing_train_donor_is_rejected():
    with pytest.raises(ValueError, match="Missing"):
        neighbors(np.zeros((4, 1)), [2, 49, 4, 13], [2, 49, 4, 23])


def test_events_keep_censoring_separate_from_zero():
    assert crossing([0.2, 0.1], 0.5) == {"status": "already_below_at_start", "slot": None}
    assert crossing([0.8, 0.7], 0.5) == {"status": "no_crossing_in_window", "slot": None}
    assert crossing([0.8, 0.2], 0.5)["slot"] == pytest.approx(0.5)


def test_duplicate_episodes_rejected():
    with pytest.raises(ValueError, match="Unique"):
        neighbors(np.zeros((5, 2)), [2, 2, 4, 49, 23], [2, 4, 49, 23])
