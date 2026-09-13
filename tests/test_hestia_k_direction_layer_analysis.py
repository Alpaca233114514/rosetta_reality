import numpy as np

from scripts.analyze_hestia_layer_attention import probability_change


def test_image_mass_and_image_token_distribution_are_separated():
    n = np.array([0.1, 0.2, 0.1])
    scaled = probability_change(n, n * 2)
    assert np.isclose(scaled["image_mass_delta"], 0.4)
    assert scaled["within_image_total_variation"] == scaled["within_image_js"] == 0
    changed = probability_change(n, np.array([0.2, 0.1, 0.1]))
    assert np.isclose(changed["image_mass_delta"], 0)
    assert np.isclose(changed["within_image_total_variation"], 0.25)
    assert changed["within_image_js"] > 0


def test_absent_image_attention_stays_finite():
    value = probability_change(np.zeros(64), np.zeros(64))
    assert all(np.isfinite(x) and x == 0 for x in value.values())
