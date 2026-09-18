import numpy as np
import pytest

from scripts.diagnose_hestia_local_checkpoint import difference


def test_elementwise_tolerance_rejects_one_outlier_despite_small_mean():
    expected = np.zeros((50, 14))
    actual = expected.copy()
    actual[49, 13] = 0.011
    result = difference(actual, expected, 0.01, 0.01)
    assert not result["passed"]
    assert result["mean_abs"] < 0.01


@pytest.mark.parametrize("actual", [np.zeros((3, 2)), np.full((2, 2), np.nan)])
def test_shape_or_nonfinite_predictions_stop(actual):
    with pytest.raises(ValueError, match="shape or finite"):
        difference(actual, np.zeros((2, 2)), 0.01, 0.01)
