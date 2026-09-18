import numpy as np
import pytest

from scripts.diagnose_hestia_phase_alignment import aligned_targets


def test_shifted_identical_phase_targets_align_exactly():
    y = np.zeros((2, 30, 2))
    y[0, :, 0] = np.arange(30) - 10
    y[1, :, 0] = np.arange(30) - 15
    y[:, :, 1] = 7
    result, indices = aligned_targets(y, [10, 15], np.arange(-5, 5))
    np.testing.assert_array_equal(result[0], result[1])
    np.testing.assert_array_equal(indices[1], np.arange(10, 20))
    clock, _ = aligned_targets(y, [10, 10], np.arange(-5, 5))
    assert not np.array_equal(clock[0], clock[1])


@pytest.mark.parametrize("anchors", [[None, 5], [1, 5], [29, 5]])
def test_censoring_and_bounds_must_not_be_clipped_or_filled(anchors):
    with pytest.raises(ValueError):
        aligned_targets(np.zeros((2, 30, 2)), anchors, np.arange(-5, 5))
