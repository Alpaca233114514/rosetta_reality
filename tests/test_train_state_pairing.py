from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from rosetta_reality.experiment import stable_hash
from rosetta_reality.sim.action_contract import ActionContract, ActionDimension
from scripts import pair_train_states
from scripts.pair_train_states import (
    _assemble_pairing,
    _exact_train_scope,
    _ordered_episode_reports,
    _replay_episode,
)


def _experiment() -> dict[str, Any]:
    return {
        "experiment_id": "v010-test",
        "dataset": {
            "config": "configs/data/unused.yaml",
            "split": {
                "train": list(range(40)),
                "validation": [40, 41],
                "test": [42, 43],
            },
        },
        "action_contract": "configs/sim/unused.yaml",
    }


def _contract() -> ActionContract:
    return ActionContract(
        name="test",
        schema_version=1,
        embodiment="test",
        environment_id="test/Test-v0",
        action_type="continuous",
        semantics="absolute_joint_position_targets",
        control_mode="position",
        space="joint",
        reference_frame="joint_local",
        frequency_hz=50.0,
        timestamp_alignment="observation_t_predicts_actions_t_through_t_plus_7",
        chunk_length=8,
        chunk_execution="receding_horizon_first_action",
        simulator_expansion="identity",
        dimensions=(ActionDimension("joint", "radian", -0.5, 0.5),),
        chunk_execution_steps=1,
    )


def test_export_rejects_scope_bypass_before_dataset_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _experiment()
    dataset_io_called = False

    monkeypatch.setattr(
        pair_train_states,
        "load_experiment_config",
        lambda _path, _root: experiment,
    )

    def forbidden_dataset_io(_experiment: Any) -> Any:
        nonlocal dataset_io_called
        dataset_io_called = True
        raise AssertionError("dataset I/O occurred before the train-scope gate")

    monkeypatch.setattr(pair_train_states, "_dataset_context", forbidden_dataset_io)

    with pytest.raises(ValueError, match="exact ordered train split"):
        pair_train_states.export_images(
            Path("unused.yaml"),
            requested_episodes=list(reversed(experiment["dataset"]["split"]["train"])),
        )

    assert dataset_io_called is False


def test_exact_train_scope_rejects_validation_and_hidden_test_attempts() -> None:
    experiment = _experiment()
    train = experiment["dataset"]["split"]["train"]

    for bypass in (train + [40], train + [42], train[:-1]):
        with pytest.raises(ValueError, match="exact ordered train split"):
            _exact_train_scope(experiment, bypass)


class _ReplayEnvironment:
    def __init__(self, states: list[float], *, done_after: int | None = None) -> None:
        self.states = states
        self.done_after = done_after
        self.position = 0
        self.executed: list[torch.Tensor] = []

    def reset(self, *, seed: int | None = None) -> dict[str, torch.Tensor]:
        assert seed == 7
        self.position = 0
        return {"robot_state": torch.tensor([self.states[0]])}

    def step(
        self, action: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], float, bool, dict[str, Any]]:
        self.executed.append(action.clone())
        self.position += 1
        done = self.done_after is not None and self.position >= self.done_after
        return (
            {"robot_state": torch.tensor([self.states[self.position]])},
            0.0,
            done,
            {},
        )


def _rows(recorded_states: list[float]) -> list[dict[str, Any]]:
    return [
        {"frame": frame, "state": [state], "action": [2.0]}
        for frame, state in enumerate(recorded_states)
    ]


def test_replay_captures_pre_action_state_and_uses_exclusive_crossing_cutoff() -> None:
    environment = _ReplayEnvironment([0.0, 1.0, 2.0, 3.0])

    result = _replay_episode(
        environment,
        _contract(),
        _rows([0.0, 1.0, 9.0]),
        action_field="action",
        state_field="state",
        frame_field="frame",
        seed=7,
    )

    assert result["exclusive_cutoff"] == 2
    assert result["cutoff_reason"] == "recorded_state_mae"
    assert list(result["states"]) == [0, 1]
    assert torch.equal(result["states"][0], torch.tensor([0.0]))
    assert torch.equal(result["states"][1], torch.tensor([1.0]))
    assert len(environment.executed) == 2
    assert all(torch.equal(action, torch.tensor([0.5])) for action in environment.executed)


def test_done_after_action_t_makes_t_plus_one_the_exclusive_cutoff() -> None:
    environment = _ReplayEnvironment([0.0, 1.0, 2.0], done_after=1)

    result = _replay_episode(
        environment,
        _contract(),
        _rows([0.0, 1.0]),
        action_field="action",
        state_field="state",
        frame_field="frame",
        seed=7,
    )

    assert result["exclusive_cutoff"] == 1
    assert result["cutoff_reason"] == "done"
    assert list(result["states"]) == [0]


def test_eligible_rule_digest_and_recorded_fallback_are_exact() -> None:
    episode_ids = torch.tensor([7, 7, 7, 8, 8])
    frame_indices = torch.tensor([0, 5, 10, 0, 1])
    recorded = torch.tensor([[70.0], [75.0], [80.0], [80.0], [81.0]])
    replays = {
        7: {
            "exclusive_cutoff": 13,
            "states": {frame: torch.tensor([float(frame)]) for frame in range(13)},
        },
        8: {
            "exclusive_cutoff": 8,
            "states": {frame: torch.tensor([100.0 + frame]) for frame in range(8)},
        },
    }

    paired, mask, digest, counts = _assemble_pairing(
        episode_ids=episode_ids,
        frame_indices=frame_indices,
        recorded_states=recorded,
        replays=replays,
        train_episodes=[7, 8],
    )

    assert torch.equal(mask, torch.tensor([True, True, False, True, False]))
    assert torch.equal(paired, torch.tensor([[0.0], [5.0], [80.0], [100.0], [81.0]]))
    assert digest == stable_hash([[7, 0], [7, 5], [8, 0]])
    assert counts == {7: 2, 8: 1}


def test_episode_reports_follow_train_order_and_loader_prefix_schema() -> None:
    episode_ids = torch.tensor([7, 7, 7, 8, 8])
    frame_indices = torch.tensor([0, 5, 10, 0, 5])
    mask = torch.tensor([True, True, False, True, False])
    replays = {
        7: {
            "exclusive_cutoff": 13,
            "cutoff_reason": "recorded_state_mae",
            "executed_steps": 13,
        },
        8: {
            "exclusive_cutoff": 8,
            "cutoff_reason": "done",
            "executed_steps": 8,
        },
    }
    reset_reports = {
        7: {"selected_seed": 3, "top_candidates": [{"seed": 3}]},
        8: {"selected_seed": 4, "top_candidates": [{"seed": 4}]},
    }

    reports = _ordered_episode_reports(
        train_episodes=[8, 7],
        episode_ids=episode_ids,
        frame_indices=frame_indices,
        pairing_mask=mask,
        replays=replays,
        reset_reports=reset_reports,
    )

    assert [report["episode"] for report in reports] == [8, 7]
    assert reports[0]["source_anchor_count"] == 2
    assert reports[0]["paired_anchor_count"] == 1
    assert reports[0]["eligible_frame_indices"] == [0]
    assert reports[0]["exclusive_valid_step_stop"] == 8
    assert reports[1]["source_anchor_count"] == 3
    assert reports[1]["paired_anchor_count"] == 2
    assert reports[1]["eligible_frame_indices"] == [0, 5]
    assert reports[1]["exclusive_valid_step_stop"] == 13
