import numpy as np

from scripts.diagnose_chunk_time_shift import analyze, shift_chunk


def test_negative_lag_holds_start_and_preserves_first_action():
    value = np.arange(5).reshape(1, 5, 1)
    np.testing.assert_array_equal(shift_chunk(value, -2).ravel(), [0, 0, 0, 1, 2])
    np.testing.assert_array_equal(shift_chunk(value, 2).ravel(), [2, 3, 4, 4, 4])


def test_dev_labels_do_not_choose_lag():
    rng = np.random.default_rng(11)
    prediction, targets = rng.normal(size=(6, 5, 2)), rng.normal(size=(6, 5, 2))
    a, loo_a, selected_a = analyze(prediction, targets, [-1, 0, 1], 4)
    targets[4:] += 100
    b, loo_b, selected_b = analyze(prediction, targets, [-1, 0, 1], 4)
    assert selected_a == selected_b
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(loo_a, loo_b)


def test_loo_held_target_is_excluded_from_lag_selection():
    rng = np.random.default_rng(12)
    prediction, targets = rng.normal(size=(6, 5, 2)), rng.normal(size=(6, 5, 2))
    _, a, sa = analyze(prediction, targets, [-1, 0, 1], 4)
    targets[0] += 100
    _, b, sb = analyze(prediction, targets, [-1, 0, 1], 4)
    assert sa["loo_lags"][0] == sb["loo_lags"][0]
    np.testing.assert_array_equal(a[0], b[0])
