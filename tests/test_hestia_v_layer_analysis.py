import numpy as np

from scripts.analyze_hestia_v_layer import scene_error_terms


def test_correcting_global_offset_changes_only_bias():
    y = np.array([-1.0, 1.0]).reshape(2, 1, 1)
    before = scene_error_terms(np.broadcast_to(y + 3, (4, *y.shape)), y)
    after = scene_error_terms(np.broadcast_to(y, (4, *y.shape)), y)
    np.testing.assert_allclose(after["mse"] - before["mse"], -9)
    np.testing.assert_allclose(
        after["squared_scene_mean_bias"] - before["squared_scene_mean_bias"], -9
    )
    for term in ("prediction_scene_variance", "minus_twice_scene_covariance"):
        np.testing.assert_array_equal(before[term], after[term])


def test_scene_correspondence_repair_is_covariance_not_mean_shift():
    y = np.array([-1.0, 1.0]).reshape(2, 1, 1)
    before = scene_error_terms(np.broadcast_to(-y, (4, *y.shape)), y)
    after = scene_error_terms(np.broadcast_to(y, (4, *y.shape)), y)
    np.testing.assert_allclose(after["mse"] - before["mse"], -4)
    np.testing.assert_allclose(
        after["minus_twice_scene_covariance"] - before["minus_twice_scene_covariance"], -4
    )
    for term in ("squared_scene_mean_bias", "prediction_scene_variance"):
        np.testing.assert_array_equal(before[term], after[term])
