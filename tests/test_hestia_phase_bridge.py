"""Counterexamples for phase interpolation and event censoring."""

import numpy as np
import pytest

from scripts.diagnose_hestia_phase_bridge import event_window, phase_locations, resample


def test_linear_roundtrip_and_scene_specific_event():
    y = np.broadcast_to(np.arange(100)[None, :, None], (3, 100, 2)).astype(float).copy()
    events = np.array([40.0, 50.0, 67.0])
    canonical = resample(y, phase_locations(events, inverse=False))
    np.testing.assert_array_equal(canonical[:, 50, 0], events)
    restored = resample(canonical, phase_locations(events, inverse=True))
    # At a phase kink a second interpolation can smooth one fractional-grid point.
    assert np.abs(restored - y[:, :50]).max() < 0.03
    np.testing.assert_array_equal(restored[:, 0], y[:, 0])


def test_estimated_map_does_not_use_target_event():
    estimate = np.array([45.0, 55.0])
    a = phase_locations(estimate, inverse=True)
    np.testing.assert_array_equal(a, phase_locations(estimate.copy(), inverse=True))
    assert np.all(np.diff(a, axis=1) > 0)
    assert a[0, 45] == 50


@pytest.mark.parametrize("event", [0.0, 99.0, -1.0, np.nan])
def test_invalid_onset_rejected(event):
    with pytest.raises(ValueError, match="interior"):
        phase_locations([event], inverse=True)


def test_extrapolation_rejected():
    with pytest.raises(ValueError, match="extrapolation"):
        resample(np.zeros((1, 10, 2)), np.array([[10.0]]))


def test_native_event_exact_window():
    a = np.zeros((2, 50, 3))
    a[:, :, 0] = np.arange(50)
    a[0, 48:, 2] = 1
    a[1, 44:, 2] = 1
    w, e = event_window(a, 2)
    assert e == [48, 44]
    np.testing.assert_array_equal(w[:, 0, 0], [29, 25])
    np.testing.assert_array_equal(w[:, -1, 0], [48, 44])


def test_censored_native_event_rejected():
    with pytest.raises(ValueError, match="Censored"):
        event_window(np.zeros((2, 50, 3)), 2)
