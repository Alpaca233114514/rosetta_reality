import numpy as np
import pytest

from scripts.diagnose_socket_position import locate_blue


def test_fixed_blue_position_and_red_distractor():
    rgb = np.zeros((100, 150, 3), dtype=np.uint8)
    rgb[20:30, 50:70, 2] = 220
    rgb[60:80, 10:30, 0] = 255
    position, record = locate_blue(rgb)
    np.testing.assert_array_equal(position, [59.5, 24.5])
    assert record["areas"] == [200, 200]
    assert record["threshold_shift_pixels"] == 0


def test_empty_or_dispersed_blue_fails_closed():
    rgb = np.zeros((300, 300, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="area"):
        locate_blue(rgb)
    rgb[10:20, 10:20, 2] = 255
    rgb[250:260, 250:260, 2] = 255
    with pytest.raises(ValueError, match="extent"):
        locate_blue(rgb)


def test_unstable_threshold_fails_closed():
    rgb = np.zeros((100, 150, 3), dtype=np.uint8)
    rgb[20:30, 30:50, 2] = 90
    rgb[20:30, 60:80, 2] = 220
    with pytest.raises(ValueError, match="unstable"):
        locate_blue(rgb)


def test_non_uint8_rejected():
    with pytest.raises(ValueError, match="uint8"):
        locate_blue(np.zeros((100, 100, 3)))
