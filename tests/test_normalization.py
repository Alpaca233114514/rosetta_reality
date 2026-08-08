"""Online population statistics and persistence tests."""

import pytest
import torch

from rosetta_reality.data.normalization import (
    DatasetStatistics,
    NormalizationStats,
    RunningMoments,
    denormalize,
    load_dataset_statistics,
    normalize,
    save_dataset_statistics,
)


def test_population_statistics_normalization_and_zero_variance() -> None:
    moments = RunningMoments()
    moments.update(torch.tensor([[1.0, 2.0], [3.0, 2.0]]))
    stats = moments.finalize()
    values = torch.tensor([[1.0, 2.0], [3.0, 2.0]])
    normalized = normalize(values, stats)

    assert torch.allclose(stats.mean, torch.tensor([2.0, 2.0]))
    assert torch.allclose(stats.std, torch.tensor([1.0, 1e-6]))
    assert torch.allclose(denormalize(normalized, stats), values)


def test_statistics_json_round_trip_is_idempotent(tmp_path) -> None:
    statistics = DatasetStatistics(
        state=NormalizationStats(torch.tensor([1.0]), torch.tensor([2.0])),
        action=NormalizationStats(torch.tensor([3.0]), torch.tensor([4.0])),
        state_count=5,
        action_count=5,
    )
    path = tmp_path / "statistics.json"

    save_dataset_statistics(path, statistics)
    save_dataset_statistics(path, statistics)

    assert load_dataset_statistics(path).to_dict() == statistics.to_dict()


def test_statistics_refuse_to_overwrite_different_values(tmp_path) -> None:
    path = tmp_path / "statistics.json"
    first = DatasetStatistics(
        state=NormalizationStats(torch.tensor([0.0]), torch.tensor([1.0])),
        action=NormalizationStats(torch.tensor([0.0]), torch.tensor([1.0])),
        state_count=1,
        action_count=1,
    )
    second = DatasetStatistics(
        state=NormalizationStats(torch.tensor([1.0]), torch.tensor([1.0])),
        action=first.action,
        state_count=1,
        action_count=1,
    )
    save_dataset_statistics(path, first)

    with pytest.raises(FileExistsError):
        save_dataset_statistics(path, second)
