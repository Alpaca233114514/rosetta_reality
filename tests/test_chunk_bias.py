import numpy as np

from scripts.diagnose_chunk_bias import correct_bias


def test_development_labels_do_not_change_correction():
    rng = np.random.default_rng(47)
    prediction, target = rng.normal(size=(6, 3, 2)), rng.normal(size=(6, 3, 2))
    before = correct_bias(prediction, target, 4)
    target[4:] += 1000
    after = correct_bias(prediction, target, 4)
    for a, b in zip(before, after, strict=True):
        np.testing.assert_array_equal(a, b)


def test_loo_excludes_held_target():
    rng = np.random.default_rng(13)
    prediction, target = rng.normal(size=(6, 3, 2)), rng.normal(size=(6, 3, 2))
    _, before, _ = correct_bias(prediction, target, 4)
    target[0] += 10
    _, after, _ = correct_bias(prediction, target, 4)
    np.testing.assert_allclose(before[0], after[0], atol=1e-14, rtol=0)


def test_shared_offset_is_removed_without_changing_scene_differences():
    rng = np.random.default_rng(3)
    target = rng.normal(size=(6, 3, 2))
    prediction = target + np.arange(6).reshape(3, 2)
    corrected, loo, _ = correct_bias(prediction, target, 4)
    np.testing.assert_allclose(corrected, target, atol=1e-14)
    np.testing.assert_allclose(loo, target[:4], atol=1e-14)
    np.testing.assert_allclose(corrected[5] - corrected[4], prediction[5] - prediction[4])
