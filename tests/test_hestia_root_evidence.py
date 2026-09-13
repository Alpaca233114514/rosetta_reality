import numpy as np
import pytest

from scripts.analyze_hestia_root_evidence import factorial, positive_ratio, summarize


def test_additive_and_interacting_effects():
    a = np.zeros((4, 5))
    f = factorial(a, a + 2, a + 3, a + 9)
    assert f["interaction"]["mean"] == 4
    assert f["k_order_average"]["mean"] == 4
    assert f["v_order_average"]["mean"] == 5
    assert f["joint_effect"]["mean"] == 9
    assert factorial(a, a + 2, a + 3, a + 5)["interaction"]["mean"] == 0


def test_both_singles_can_help_but_joint_harms():
    a = np.full((4, 5), 10.0)
    f = factorial(a, a - 1, a - 2, a + 3)
    assert f["k_given_old_v"]["mean"] < 0
    assert f["v_given_old_k"]["mean"] < 0
    assert f["joint_effect"]["mean"] > 0


def test_one_scene_can_reverse_conclusion():
    x = np.array([[10, -1, -1, -1, -1]], dtype=float)
    s = summarize(x)
    assert s["mean"] > 0
    assert s["leave_one_episode_out_by_noise"][0][0] == -1


def test_preserves_noise_instead_of_averaging_predictions():
    x = np.array([[1, 2], [-3, -4]], dtype=float)
    assert summarize(x)["by_noise"] == [1.5, -3.5]


@pytest.mark.parametrize("denominator", [0, -1])
def test_nonpositive_denominator_is_not_a_recovery_rate(denominator):
    assert positive_ratio(1, denominator) is None


def test_signed_and_over_100_percent_effects_are_not_clamped():
    assert positive_ratio(-1, 2) == -0.5
    assert positive_ratio(3, 2) == 1.5


def test_nonfinite_rejected():
    with pytest.raises(ValueError, match="Invalid metric"):
        summarize(np.array([[np.nan, 1]]))


def test_factorial_pairing_rejected():
    with pytest.raises(ValueError, match="Unpaired"):
        factorial(np.zeros((4, 5)), np.zeros((4, 4)), np.zeros((4, 5)), np.zeros((4, 5)))
