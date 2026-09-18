"""Native sampler integration with sealed explicit frame order."""

import pytest
from lerobot.datasets.sampler import EpisodeAwareSampler

from rosetta_reality.vla.training.temporal_sampler import sampler_class


def test_native_sampler_retains_exact_explicit_order_and_repeats():
    samples = [[1, 0], [0, 9], [1, 9], [1, 0]]
    cls = sampler_class(EpisodeAwareSampler, samples, 17)
    sampler = cls(
        [0, 10],
        [10, 20],
        [1, 0],
        shuffle=True,
        seed=17,
        absolute_to_relative_idx={0: 0, 9: 1, 10: 2, 19: 3},
    )
    assert list(sampler) == [2, 1, 3, 2]
    assert list(sampler) == [2, 1, 3, 2]
    assert len(sampler) == 4


@pytest.mark.parametrize("sample", [[[1, 10]], [[2, 0]]])
def test_temporal_sampler_rejects_episode_boundaries(sample):
    cls = sampler_class(EpisodeAwareSampler, sample, 17)
    with pytest.raises(ValueError):
        cls([0, 10], [10, 20], [sample[0][0]], seed=17)


def test_temporal_sampler_rejects_missing_view_index():
    cls = sampler_class(EpisodeAwareSampler, [[1, 9]], 17)
    with pytest.raises(ValueError, match="absent"):
        cls([0, 10], [10, 20], [1], seed=17, absolute_to_relative_idx={10: 0})


def test_temporal_sampler_rejects_changed_seed():
    cls = sampler_class(EpisodeAwareSampler, [[1, 0]], 17)
    with pytest.raises(ValueError, match="seed"):
        cls([0, 10], [10, 20], [1], seed=18)
