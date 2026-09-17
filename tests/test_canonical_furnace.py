import random
from collections import Counter

import pytest

from scripts.canonical_furnace import full_schedule


def test_full_schedule_preserves_rng_and_visits_every_training_frame_once():
    episodes = sorted(set(range(50)) - {31, 6, 1, 24, 5, 22, 13, 7, 33, 45})
    before = random.getstate()
    samples = full_schedule(episodes)
    assert random.getstate() == before
    assert samples == full_schedule(episodes)
    assert len(set(map(tuple, samples))) == len(samples) == 20000
    assert Counter(ep for ep, _ in samples) == dict.fromkeys(episodes, 500)
    for ep in episodes:
        assert sorted(frame for scene, frame in samples if scene == ep) == list(range(500))


def test_schedule_rejects_duplicate_and_nontraining_episodes():
    with pytest.raises(ValueError):
        full_schedule([0] * 40)
    with pytest.raises(ValueError):
        full_schedule(list(range(40)))
