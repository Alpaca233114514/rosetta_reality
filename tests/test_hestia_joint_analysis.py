import numpy as np
import pytest

from scripts.analyze_hestia_joint_attention import qualify, subset_exact
from scripts.analyze_hestia_k_head_interaction import joint_residual


def error(values):
    return {k: dict(mean=float(np.mean(values)), by_noise=values) for k in ("mae", "mse")}


def test_joint_threshold_and_noise_consistency():
    assert qualify(error([10] * 4), error([5] * 4), error([6] * 4))["pair_dominance_qualified"]
    assert not qualify(error([10] * 4), error([5] * 4), error([6.1] * 4))[
        "pair_dominance_qualified"
    ]
    assert not qualify(error([10] * 4), error([5] * 4), error([11, 4, 4, 4]))[
        "pair_dominance_qualified"
    ]


def test_joint_isolation_checks_complement_and_allows_later_propagation():
    native = (np.zeros((2, 8, 15, 64)), np.zeros((2, 8, 15, 8)))
    changed = tuple(x.copy() for x in native)
    for x in changed:
        x[:, 0, 3:9] = 1
        x[:, 1:] = 2
    assert subset_exact(native, changed, [1, 2]) == [0, 1, 2, 9, 10, 11, 12, 13, 14]
    changed[1][:, 0, 14] = 1
    with pytest.raises(ValueError, match="Unselected"):
        subset_exact(native, changed, [1, 2])


def test_actual_joint_residual_is_not_a_sum_of_mse_benefits():
    native = np.ones((4, 2, 3, 5))
    first, second = native + 2, native + 3
    exact = joint_residual(native, first, second, native + 5)
    assert exact["residual_fraction"] == [0.0] * 4
    changed = joint_residual(native, first, second, native + 10)
    assert changed["residual_fraction"] == [0.25] * 4
