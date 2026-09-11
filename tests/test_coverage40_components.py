import numpy as np

from scripts.diagnose_coverage40_components import decompose


def test_same_mse_does_not_hide_bias_or_reversed_scene_mapping():
    target = np.array([-1.0, 1.0])[:, None, None]
    biased = decompose(target + 2, target, target)
    reversed_mapping = decompose(-target, target, target)
    assert biased["mse"] == reversed_mapping["mse"] == 4
    assert biased["mean_bias_squared"] == 4
    assert biased["scene_correlation"] == 1
    assert reversed_mapping["mean_bias_squared"] == 0
    assert reversed_mapping["scene_correlation"] == -1


def test_constant_prediction_correlation_is_unmeasured():
    target = np.array([-1.0, 1.0])[:, None, None]
    result = decompose(np.zeros_like(target), target, target)
    assert result["scene_correlation"] is None
    assert result["prediction_to_target_variance_ratio"] == 0
    assert result["target_scene_variance"] == result["mse"] == 1
