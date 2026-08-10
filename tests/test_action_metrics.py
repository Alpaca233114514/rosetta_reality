from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rosetta_reality.data.normalization import NormalizationStats
from rosetta_reality.eval import action_metrics
from rosetta_reality.sim import load_action_contract

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_action_metrics_report_physical_violations() -> None:
    contract = load_action_contract(
        REPOSITORY_ROOT / "configs" / "sim" / "aloha_insertion.yaml"
    )
    target = torch.zeros(2, 3, 14)
    predicted = target.clone()
    predicted[0, 0, 6] = 2.0
    stats = NormalizationStats(mean=torch.zeros(14), std=torch.ones(14))

    metrics = action_metrics(predicted, target, contract, stats)

    assert metrics["action_mae"] == pytest.approx(2 / (2 * 3 * 14))
    assert metrics["first_action_mae"] == pytest.approx(2 / (2 * 14))
    assert metrics["invalid_action_rate"] == 0.5
    assert metrics["limit_violation_rate"] == pytest.approx(1 / (2 * 3 * 14))


def test_action_metrics_distinguish_raw_from_projected_actions() -> None:
    contract = load_action_contract(
        REPOSITORY_ROOT / "configs" / "sim" / "aloha_insertion.yaml"
    )
    target = torch.zeros(1, 1, 14)
    raw = target.clone()
    raw[0, 0, 6] = 2.0
    projected, _ = contract.clip(raw)
    stats = NormalizationStats(mean=torch.zeros(14), std=torch.ones(14))

    metrics = action_metrics(
        projected,
        target,
        contract,
        stats,
        raw_predicted=raw,
    )

    assert metrics["invalid_action_rate"] == 0.0
    assert metrics["raw_invalid_action_rate"] == 1.0
    assert metrics["projection_element_rate"] == pytest.approx(1 / 14)
    assert metrics["maximum_projection_magnitude"] == 1.0
