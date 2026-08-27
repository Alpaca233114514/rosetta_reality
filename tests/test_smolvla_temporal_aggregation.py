"""Faust receding-horizon temporal aggregation tests."""

import pytest
import torch

from scripts.smolvla_faust_temporal_aggregation_sim import (
    _aggregate_current_action,
)


def test_temporal_aggregation_uses_offsets_and_favors_newer_predictions() -> None:
    old = torch.tensor([[0.0], [10.0], [20.0]])
    new = torch.tensor([[30.0], [40.0], [50.0]])

    raw, processed = _aggregate_current_action(
        [(0, old, old + 1.0), (1, new, new + 1.0)],
        current_step=1,
        decay=0.05,
    )

    older_weight = torch.exp(torch.tensor(-0.05, dtype=torch.float64))
    expected = (older_weight * 10.0 + 30.0) / (older_weight + 1.0)
    assert raw.item() == pytest.approx(float(expected))
    assert processed.item() == pytest.approx(float(expected + 1.0))


def test_temporal_aggregation_rejects_missing_current_prediction() -> None:
    with pytest.raises(ValueError, match="No finite prediction"):
        _aggregate_current_action(
            [(0, torch.zeros(1, 2), torch.zeros(1, 2))],
            current_step=2,
            decay=0.05,
        )

    with pytest.raises(ValueError, match="positive decay"):
        _aggregate_current_action([], current_step=0, decay=0.0)


def test_temporal_aggregation_can_match_original_act_oldest_first_order() -> None:
    old = torch.tensor([[0.0], [10.0], [20.0]])
    new = torch.tensor([[30.0], [40.0], [50.0]])

    raw, _ = _aggregate_current_action(
        [(0, old, old), (1, new, new)],
        current_step=1,
        decay=0.01,
        weighting="older_predictions_original_act_order",
    )

    newer_weight = torch.exp(torch.tensor(-0.01, dtype=torch.float64))
    expected = (10.0 + newer_weight * 30.0) / (1.0 + newer_weight)
    assert raw.item() == pytest.approx(float(expected))

    with pytest.raises(ValueError, match="weighting order"):
        _aggregate_current_action(
            [(0, old, old)],
            current_step=0,
            decay=0.01,
            weighting="unknown",
        )
