import numpy as np
import pytest

from scripts.analyze_hestia_value_route_events import first_opening


def test_censored_and_initially_open_are_distinct():
    values = np.array([[[0.2, 0.3, 0.7], [0.2, 0.4, 0.3], [0.8, 0.2, 0.1]]])
    times, initial = first_opening(values, 0.5)
    np.testing.assert_array_equal(times, [[2, -1, 0]])
    np.testing.assert_array_equal(initial, [[False, False, True]])


def test_nonfinite_event_series_rejected():
    with pytest.raises(ValueError):
        first_opening(np.array([[[np.nan, 0.7]]]), 0.5)
