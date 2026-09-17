"""Counterexamples for the reviewed post-training execution boundaries."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rosetta_reality.vla.action_space import SmolVLAActionSpace, load_smolvla_experiment
from scripts.canonical_fullframes_contract import (
    internal_grippers,
    sha,
    train_action_table,
    validate_gate_protocol,
    verify_files,
)
from scripts.canonical_fullframes_endpoint import _build_config

ROOT = Path(__file__).resolve().parents[1]


def test_actual_parent_builds_complete_action_space():
    config = load_smolvla_experiment(
        ROOT / "configs/vla/smolvla_450m_aloha_insertion_action_repair_bounded_gripper_003.yaml",
        ROOT,
    )
    plan = {
        "endpoint": {"model_safetensors_sha256": "a" * 64},
        "data": {"dataset_revision": "d", "model_revision": "m"},
        "gate_protocol_authority": {"action_contract_sha256": "b" * 64},
    }
    result = _build_config(plan, config, {"features": {}, "fps": 50}, "test", "p", "n", "v", "s")
    assert SmolVLAActionSpace(**result["action_space"]).adapt_to_pi_aloha is False
    assert result["rename_map"] == config["dataset"]["rename_map"]


@pytest.mark.parametrize("change", ["weight", "processor", "extra", "missing"])
def test_full_seal_rejects_changed_or_resealed_file_set(tmp_path, change):
    (tmp_path / "model").write_bytes(b"original")
    (tmp_path / "processor").write_bytes(b"saved-state")
    expected = {p.name: sha(p) for p in tmp_path.iterdir()}
    if change == "weight":
        (tmp_path / "model").write_bytes(b"changed")
    elif change == "processor":
        (tmp_path / "processor").write_bytes(b"changed")
    elif change == "extra":
        (tmp_path / "extra").write_bytes(b"unregistered")
    else:
        expected["absent"] = "0" * 64
    with pytest.raises(ValueError):
        verify_files(tmp_path, expected)


def test_baseline_filters_hidden_rows_before_action_materialization(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "rows.parquet"
    pq.write_table(
        pa.table(
            {"episode_index": [49, 31, 22], "action": [[1.0] * 14, [9999.0] * 14, [8888.0] * 14]}
        ),
        path,
    )
    calls = []

    def read(path, **kwargs):
        assert kwargs["filters"] == [("episode_index", "in", [49])]
        calls.append(kwargs)
        return pq.read_table(path, **kwargs)

    table = train_action_table(SimpleNamespace(read_table=read), path, [49])
    assert calls and table["episode_index"].to_pylist() == [49]
    assert table["action"].to_pylist() == [[1.0] * 14]


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("gate4", "minimum_task_success_rate", 0.0),
        ("gate4", "seeds", [1000]),
        ("gate4", "maximum_steps", 1),
        ("gate3", "seed", 1),
        ("inference", "noise", "zeros"),
    ],
)
def test_executed_yaml_protocol_rejects_drift(section, field, value):
    registered = {
        "gate3": {"environment_seed": 20260809},
        "gate4": {
            "minimum_task_success_rate": 0.2,
            "seeds": list(range(1000, 1005)),
            "maximum_steps": 500,
        },
        "executed_inference": {"noise": "seeded_standard_normal"},
        "executed_gate_resources": {"accelerator": "cuda"},
    }
    actual = {
        "gate3": {"seed": 20260809, "report_suffix": "471"},
        "gate4": {**registered["gate4"], "report_suffix": "471"},
        "inference": copy.deepcopy(registered["executed_inference"]),
        "resources": registered["executed_gate_resources"],
    }
    validate_gate_protocol(actual, registered)
    actual[section][field] = value
    with pytest.raises(ValueError):
        validate_gate_protocol(actual, registered)


def test_gripper_statistics_exclude_twelve_arm_joints():
    dims = [SimpleNamespace(encoding="gripper" if i in (6, 13) else None) for i in range(14)]
    actions = torch.full((1, 50, 14), 999.0)
    actions[..., 6], actions[..., 13] = -0.5, 0.25
    values = internal_grippers(actions, dims)
    assert values.numel() == 100
    assert float(values.min()) == -0.5 and float(values.max()) == 0.25
