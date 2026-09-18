from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from rosetta_reality.sim import load_action_contract
from scripts import eval as evaluate_script

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPOSITORY_ROOT / "configs" / "sim" / "aloha_insertion.yaml"


def test_aloha_contract_records_complete_physical_semantics() -> None:
    contract = load_action_contract(CONTRACT_PATH)

    assert contract.dimension == 14
    assert contract.frequency_hz == 50
    assert contract.chunk_length == 8
    assert contract.semantics == "absolute_joint_position_targets"
    assert contract.control_mode == "position"
    assert contract.dimension_names[6] == "left_gripper"
    assert contract.dimension_names[13] == "right_gripper"
    assert contract.source_overshoot_tolerances[0].item() == 0.0
    assert contract.source_overshoot_tolerances[6].item() == pytest.approx(0.20)


def test_contract_rejects_equal_width_but_reordered_actions() -> None:
    contract = load_action_contract(CONTRACT_PATH)
    reordered = list(contract.dimension_names)
    reordered[0], reordered[7] = reordered[7], reordered[0]

    with pytest.raises(ValueError, match="ordering is incompatible"):
        contract.validate_order(reordered)


def test_contract_clips_only_out_of_range_fields() -> None:
    contract = load_action_contract(CONTRACT_PATH)
    action = torch.zeros(contract.dimension)
    action[0] = 4.0
    action[6] = -0.1
    clipped, mask = contract.clip(action)

    assert clipped[0].item() == pytest.approx(3.14158)
    assert clipped[6].item() == pytest.approx(0.0)
    assert mask.nonzero().flatten().tolist() == [0, 6]


def test_contract_rejects_nonfinite_actions() -> None:
    contract = load_action_contract(CONTRACT_PATH)
    action = torch.zeros(contract.dimension)
    action[4] = torch.nan

    with pytest.raises(ValueError, match="NaN or Inf"):
        contract.clip(action)


def test_action_contract_defaults_to_receding_single_step_execution() -> None:
    contract = load_action_contract(CONTRACT_PATH)

    assert contract.chunk_execution_steps == 1


def test_action_contract_rejects_execution_longer_than_chunk() -> None:
    contract = load_action_contract(CONTRACT_PATH)

    with pytest.raises(ValueError, match="within the chunk length"):
        replace(contract, chunk_execution_steps=contract.chunk_length + 1)


def test_evaluation_rejects_exported_action_contract_drift(tmp_path: Path) -> None:
    experiment = {"action_contract": "configs/sim/aloha_insertion.yaml"}
    contract = load_action_contract(CONTRACT_PATH)
    payload = json.loads(json.dumps(asdict(contract), allow_nan=False))
    contract_path = tmp_path / "action_contract.json"
    contract_path.write_text(json.dumps(payload), encoding="utf-8")

    matched = evaluate_script._matching_action_contract(experiment, tmp_path)
    assert matched.dimension_names == contract.dimension_names
    assert torch.equal(matched.lower_bounds, contract.lower_bounds)
    payload["dimensions"][0]["maximum"] -= 0.1
    contract_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="differs from the evaluation config"):
        evaluate_script._matching_action_contract(experiment, tmp_path)
