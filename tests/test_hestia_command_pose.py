import numpy as np
import pytest

from scripts.diagnose_hestia_command_pose import rotation_error, rotation_mean, summarize


def rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@pytest.mark.parametrize("angle", [0, 1e-9, 0.7, np.pi, -np.pi + 0.1])
def test_rotation_geodesic_endpoints_and_units(angle):
    assert rotation_error(np.eye(3), rotation_z(angle)) == pytest.approx(abs(angle), abs=1e-12)


def test_rotation_mean_respects_wraparound_and_proper_rotation():
    mean = rotation_mean(np.stack([rotation_z(np.pi - 0.1), rotation_z(-np.pi + 0.1)]))
    assert rotation_error(mean, rotation_z(np.pi)) < 1e-12
    assert np.linalg.det(mean) == pytest.approx(1)
    np.testing.assert_allclose(mean.T @ mean, np.eye(3), atol=1e-12)


def test_ambiguous_orientation_does_not_get_silently_filled():
    with pytest.raises(ValueError, match="Degenerate"):
        rotation_mean(np.stack([np.eye(3), rotation_z(np.pi)]))


def test_aggregation_preserves_scene_arm_metric_axes():
    error = np.zeros((3, 4, 2, 2))
    error[0, :, 0, 0] = 0.01
    error[2, :, 1, 1] = 0.5
    result = summarize(error, {"train": [0, 1], "dev": [2]}, [20, 10, 7])
    assert result["train"]["left"]["position_l2_m"]["mean"] == pytest.approx(0.005)
    assert result["dev"]["right"]["orientation_geodesic_rad"]["per_episode"] == {"7": 0.5}
