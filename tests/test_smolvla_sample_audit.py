"""Temporal coverage arithmetic and RNG/identity counterexamples."""

import numpy as np
import pytest

from rosetta_reality.vla.training.sample_audit import coverage, spaced_frames, temporal_schedule


def test_exposures_are_not_unique_frames_or_targets():
    samples = [[7, 0]] * 128
    result = coverage(samples, {7: 500})
    assert result["sample_exposures"] == 128
    assert result["unique_input_frames"] == 1
    assert result["unique_target_frames"] == 50
    assert result["full_input_traversal"] is False


def test_temporal_schedule_preserves_episode_order_and_global_rng():
    control = [[ep, 0] for _ in range(128) for ep in [7, 3]]
    before = np.random.get_state()
    actual = temporal_schedule(control, {7: 500, 3: 501}, seed=19)
    after = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(before, after))
    assert [ep for ep, _ in actual] == [ep for ep, _ in control]
    assert actual == temporal_schedule(control, {7: 500, 3: 501}, seed=19)
    assert len(set(map(tuple, actual))) == 256
    assert coverage(actual, {7: 500, 3: 501})["unique_target_frames"] == 1001
    assert (7, 499) in map(tuple, actual)


def test_tail_targets_do_not_cross_episode():
    assert coverage([[7, 499]], {7: 500})["unique_target_frames"] == 1


@pytest.mark.parametrize("samples", [[[8, 0]], [[7, 500]], [[7, -1]], [[True, 0]]])
def test_out_of_scope_samples_rejected(samples):
    with pytest.raises(ValueError):
        coverage(samples, {7: 500})


def test_short_episode_rejected_without_silent_duplication():
    with pytest.raises(ValueError):
        spaced_frames(100, 128)


def test_temporal_budget_or_split_mismatch_rejected():
    with pytest.raises(ValueError):
        temporal_schedule([[7, 0]] * 127, {7: 500}, seed=19)
    with pytest.raises(ValueError):
        temporal_schedule([[7, 0]] * 128, {7: 500, 9: 500}, seed=19)
