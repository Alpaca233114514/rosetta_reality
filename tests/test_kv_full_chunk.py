from pathlib import Path

import numpy as np
import pytest

from scripts.diagnose_kv_full_chunk import chunk_metrics, flattened_groups, project_targets


def test_flattened_groups_preserve_time_major_action_order():
    assert flattened_groups({"joint": [0, 2], "grip": [1]}, 2, 3) == {
        "joint": [0, 2, 3, 5],
        "grip": [1, 4],
    }


def test_native_projection_clips_only_allowed_overshoot():
    from rosetta_reality.sim.action_contract import load_action_contract

    contract = load_action_contract(Path("configs/sim/aloha_insertion_smolvla.yaml"))
    raw = np.zeros((2, 50, 14))
    raw[0, 0, 6], raw[1, 0, 13] = -0.1, 1.1
    target, record = project_targets(raw, contract, "0" * 64)
    assert target[0, 0, 6] == 0 and target[1, 0, 13] == 1
    assert sum(record["changed_elements_by_dimension"]) == 2
    assert raw[0, 0, 6] == -0.1
    raw[0, 0, 6] = -0.3
    with pytest.raises(ValueError, match="tolerance"):
        project_targets(raw, contract, "0" * 64)


def test_chunk_groups_windows_and_mismatch_mean_error():
    truth = np.arange(24, dtype=float).reshape(3, 4, 2)
    pred = truth.copy()
    pred[:, 0, 0] += 1
    metrics = chunk_metrics(pred, truth, truth, {"joint": [0], "grip": [1]}, 0, 4)
    assert metrics["joint"]["mae"] == 0.25
    assert metrics["grip"]["mae"] == 0
    assert metrics["joint"]["paired_mae_gain"] > 0
    last = chunk_metrics(pred, truth, truth, {"joint": [0], "grip": [1]}, 3, 4)
    assert last["joint"]["mae"] == 0
