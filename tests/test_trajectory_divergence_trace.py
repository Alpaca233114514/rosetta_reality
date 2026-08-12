from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from rosetta_reality.data.normalization import DatasetStatistics, NormalizationStats
from rosetta_reality.sim import load_action_contract
from scripts import sim_gate
from scripts.sim_gate import (
    _factorial_action_metrics,
    _first_crossings,
    _require_validation_episode,
    _teacher_forced_decomposition_episode,
    _trace_alignment,
    _trajectory_divergence_episode,
    _validated_trace_smoke_report,
    _write_trajectory_report,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _experiment() -> dict[str, Any]:
    return {
        "dataset": {
            "split": {
                "train": [1],
                "validation": [7, 13],
                "test": [31],
            }
        }
    }


@pytest.mark.parametrize("episode", [1, 31, 99])
def test_trace_rejects_non_validation_episode(episode: int) -> None:
    with pytest.raises(ValueError, match="validation|hidden-test"):
        _require_validation_episode(_experiment(), episode)


def test_first_crossings_are_fixed_inclusive_and_first_only() -> None:
    values = [0.009, 0.01, 0.026, 0.2, 0.0]

    assert _first_crossings(values) == {
        "0.01": 1,
        "0.025": 2,
        "0.05": 3,
        "0.1": 3,
    }


def test_factorial_action_metrics_separate_state_image_and_interaction() -> None:
    result = _factorial_action_metrics(
        torch.tensor([0.0, 0.0]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 2.0]),
        torch.tensor([1.5, 3.0]),
    )

    assert result["state_swap_at_recorded_image"]["l2"] == pytest.approx(1.0)
    assert result["image_swap_at_recorded_state"]["l2"] == pytest.approx(2.0)
    assert result["joint_recorded_to_sim_swap"]["l2"] == pytest.approx(11.25**0.5)
    assert result["image_state_interaction"]["l2"] == pytest.approx(1.25**0.5)


def _alignment_report(path: Path, *, image_episodes: list[int]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "diagnostic": "m2_expert_replay_with_image_aligned_initial_seed",
                "action_mode": "contract_clipped",
                "action_contract_sha256": "c" * 64,
                "frequency_hz": 50.0,
                "validation_scope": {
                    "experiment_id": "experiment",
                    "experiment_config_sha256": "d" * 64,
                    "split": "validation",
                    "episodes": [7, 13],
                    "test_split_opened": False,
                },
                "dataset": {
                    "revision": "a" * 40,
                    "manifest_sha256": "b" * 64,
                },
                "initial_image_artifact": {
                    "identity_hash": "e" * 64,
                    "manifest_sha256": "f" * 64,
                    "dataset_revision": "a" * 40,
                    "dataset_manifest_sha256": "b" * 64,
                    "episodes": image_episodes,
                    "validation_scope": {
                        "experiment_id": "experiment",
                        "experiment_config_sha256": "d" * 64,
                        "split": "validation",
                        "episodes": [7, 13],
                        "test_split_opened": False,
                    },
                },
                "episodes": [
                    {
                        "episode": episode,
                        "selected_seed": episode + 100,
                        "selected_alignment": {
                            "pixel_mae": 0.001,
                            "pixel_rmse": 0.002,
                            "pooled_4x4_mae": 0.001,
                        },
                    }
                    for episode in image_episodes
                ],
            }
        ),
        encoding="utf-8",
    )


def test_trace_alignment_requires_exact_validation_image_and_report_coverage(
    tmp_path: Path,
) -> None:
    path = tmp_path / "alignment.json"
    _alignment_report(path, image_episodes=[7, 13])

    alignment = _trace_alignment(
        path,
        episode=7,
        validation_episodes={7, 13},
        dataset_revision="a" * 40,
        dataset_manifest_sha256="b" * 64,
        action_contract_sha256="c" * 64,
        frequency_hz=50.0,
        experiment_id="experiment",
        experiment_config_sha256="d" * 64,
        maximum_mae=0.005,
    )

    assert alignment["selected_seed"] == 107
    _alignment_report(path, image_episodes=[7])
    with pytest.raises(ValueError, match="exact validation split"):
        _trace_alignment(
            path,
            episode=7,
            validation_episodes={7, 13},
            dataset_revision="a" * 40,
            dataset_manifest_sha256="b" * 64,
            action_contract_sha256="c" * 64,
            frequency_hz=50.0,
            experiment_id="experiment",
            experiment_config_sha256="d" * 64,
            maximum_mae=0.005,
        )


def test_trajectory_report_phase_is_path_safe() -> None:
    with pytest.raises(ValueError, match="phase"):
        _write_trajectory_report(
            {
                "experiment_id": "safe-experiment",
                "identity": {"value": 1},
                "protocol": {"phase": "../escape"},
            }
        )

    with pytest.raises(ValueError, match="compliant|range"):
        _write_trajectory_report(
            {
                "experiment_id": "safe-experiment",
                "identity": {"value": float("nan")},
                "protocol": {"phase": "smoke"},
            }
        )


def _trace_step(index: int) -> dict[str, Any]:
    stream = {
        "pre_state": [float(index)],
        "raw_action": [0.0],
        "clipped_action": [0.0],
        "raw_clip_mask": [False],
        "post_state": [float(index + 1)],
        "reward": 0.0,
        "joint_limits": {"mask": [False], "count": 0},
        "contacts": {"pairs": [], "unexpected_pairs": [], "unexpected_count": 0},
    }
    return {
        "step_index": index,
        "dataset_frame_index": index,
        "dataset_timestamp": index / 50.0,
        "expert": stream,
        "policy": stream,
        "divergence": {
            "post_state_mae": 0.0,
            "post_state_l2": 0.0,
            "post_state_maximum_absolute_difference": 0.0,
            "clipped_action_mae": 0.0,
            "clipped_action_l2": 0.0,
            "reward_delta_policy_minus_expert": 0.0,
            "reward_diverged": False,
        },
    }


def test_full_trace_requires_matching_passed_three_step_smoke(tmp_path: Path) -> None:
    identity = {"artifact_manifest_sha256": "a" * 64, "episode": 7, "seed": 107}
    path = tmp_path / "smoke.json"
    steps = [_trace_step(index) for index in range(3)]
    prefix = sim_gate._trace_prefix_digest(steps)
    path.write_text(
        json.dumps(
            {
                "status": "passed",
                "diagnostic": "m2_validation_trajectory_divergence",
                "identity": identity,
                "protocol": {"phase": "smoke", "steps_requested": 3},
                "reset": {"cross_environment_aligned": True},
                "steps": steps,
                "summary": {
                    "steps_executed": 3,
                    "end_reason": "requested_steps_completed",
                    "canonical_first_three_steps_sha256": prefix,
                },
                "test_split_opened": False,
            }
        ),
        encoding="utf-8",
    )

    result = _validated_trace_smoke_report(path, identity=identity)

    assert result["canonical_first_three_steps_sha256"] == prefix
    with pytest.raises(ValueError, match="matching passed"):
        _validated_trace_smoke_report(path, identity={**identity, "seed": 108})
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["steps"][0]["expert"]["reward"] = 1.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        _validated_trace_smoke_report(path, identity=identity)


class _FakeEnvironment:
    def __init__(
        self,
        contract: Any,
        *,
        role: str,
        image_offset: float = 0.0,
        nonfinite_reward: bool = False,
        backend: Any | None = None,
    ) -> None:
        self.contract = contract
        self.role = role
        self.image_offset = image_offset
        self.nonfinite_reward = nonfinite_reward
        self.state = torch.zeros(contract.dimension)
        self.actions: list[torch.Tensor] = []
        self.seed: int | None = None
        self.closed = False
        self._backend = object() if backend is None else backend

    @property
    def raw_environment(self) -> Any:
        return self._backend

    def _observation(self) -> dict[str, Any]:
        return {
            "robot_state": self.state.clone(),
            "images": {
                "top": torch.full((3, 2, 2), self.image_offset, dtype=torch.float32)
            },
        }

    def reset(self, *, seed: int) -> dict[str, Any]:
        self.seed = seed
        self.state.zero_()
        return self._observation()

    def step(self, action: torch.Tensor):
        assert bool((action >= self.contract.lower_bounds).all())
        assert bool((action <= self.contract.upper_bounds).all())
        self.actions.append(action.clone())
        self.state = action.clone()
        reward = float("nan") if self.nonfinite_reward else float(
            self.role == "expert" and len(self.actions) >= 2
        )
        return self._observation(), reward, False, {
            "is_success": False,
            "terminated": False,
            "truncated": False,
        }

    def contact_pairs(self):
        if self.role == "policy" and self.actions:
            return (("table", "vx300s_left/wrist"),)
        return ()

    def is_unexpected_collision_pair(self, first: str, second: str) -> bool:
        return {first, second} == {"table", "vx300s_left/wrist"}

    def close(self) -> None:
        self.closed = True


class _FakePolicy:
    def __init__(self, contract: Any, *, nonfinite: bool = False) -> None:
        self.contract = contract
        self.nonfinite = nonfinite
        self.calls: list[torch.Tensor] = []

    def __call__(self, _observations: dict[str, Any], state: torch.Tensor) -> torch.Tensor:
        self.calls.append(state.detach().cpu().clone())
        value = torch.zeros(1, self.contract.chunk_length, self.contract.dimension)
        value[0, 0, 0] = self.contract.upper_bounds[0] + 1.0
        if self.nonfinite:
            value[0, 0, 0] = torch.nan
        return value


class _StateRoutedPolicy:
    def __init__(self, contract: Any) -> None:
        self.contract = contract
        self.calls: list[torch.Tensor] = []

    def __call__(self, _observations: dict[str, Any], state: torch.Tensor) -> torch.Tensor:
        self.calls.append(state.detach().cpu().clone())
        value = torch.zeros(1, self.contract.chunk_length, self.contract.dimension)
        value[0, 0, 0] = state[0, 1]
        value[0, 1:, :] = 123.0
        return value


class _NondeterministicPolicy:
    def __init__(self, contract: Any) -> None:
        self.contract = contract
        self.calls = 0

    def __call__(self, _observations: dict[str, Any], _state: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        value = torch.zeros(1, self.contract.chunk_length, self.contract.dimension)
        value[0, 0, 0] = float(self.calls)
        return value


def _statistics(dimension: int) -> DatasetStatistics:
    stats = NormalizationStats(torch.zeros(dimension), torch.ones(dimension))
    return DatasetStatistics(stats, stats, state_count=1, action_count=1)


def _rows(contract: Any) -> tuple[list[dict[str, Any]], SimpleNamespace]:
    fields = SimpleNamespace(
        action="action",
        state="state",
        timestamp="timestamp",
        frame_index="frame_index",
    )
    rows = []
    for index in range(3):
        action = torch.zeros(contract.dimension)
        action[1] = contract.lower_bounds[1] - 1.0
        rows.append(
            {
                "action": action.tolist(),
                "state": torch.zeros(contract.dimension).tolist(),
                "timestamp": index / contract.frequency_hz,
                "frame_index": index,
            }
        )
    return rows, fields


def test_three_step_trace_uses_distinct_envs_and_records_non_oracle_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_action_contract(
        REPOSITORY_ROOT / "configs" / "sim" / "aloha_insertion.yaml"
    )
    environments = [
        _FakeEnvironment(contract, role="expert"),
        _FakeEnvironment(contract, role="policy"),
    ]
    monkeypatch.setattr(
        sim_gate,
        "GymAlohaEnvironment",
        lambda *_args, **_kwargs: environments.pop(0),
    )
    rows, fields = _rows(contract)
    policy = _FakePolicy(contract)

    result = _trajectory_divergence_episode(
        policy,
        _statistics(contract.dimension),
        contract,
        "instruction",
        rows,
        fields,
        seed=107,
        maximum_steps=3,
    )

    assert result["summary"]["steps_executed"] == 3
    assert len(policy.calls) == 3
    assert result["steps"][0]["expert_reference"] == {
        "type": "time_indexed_expert_reference",
        "state_conditioned": False,
        "recovery_oracle": False,
        "warning": (
            "After policy divergence this action is not a recovery oracle for the "
            "policy-visited state."
        ),
    }
    assert result["steps"][0]["expert"]["raw_clipped_elements"] == 1
    assert result["steps"][0]["policy"]["raw_clipped_elements"] == 1
    assert result["summary"]["first_events"]["policy_unexpected_collision"] == 0
    assert result["summary"]["first_events"]["reward_divergence"] == 1
    assert len(result["summary"]["canonical_first_three_steps_sha256"]) == 64
    json.dumps(result, allow_nan=False)


def test_teacher_forced_decomposition_routes_three_streams_without_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_action_contract(
        REPOSITORY_ROOT / "configs" / "sim" / "aloha_insertion.yaml"
    )
    expert = _FakeEnvironment(contract, role="expert")
    closed = _FakeEnvironment(contract, role="policy")
    environments = [expert, closed]
    monkeypatch.setattr(
        sim_gate,
        "GymAlohaEnvironment",
        lambda *_args, **_kwargs: environments.pop(0),
    )
    rows, fields = _rows(contract)
    policy = _StateRoutedPolicy(contract)

    result = _teacher_forced_decomposition_episode(
        policy,
        _statistics(contract.dimension),
        contract,
        "instruction",
        rows,
        fields,
        seed=107,
    )

    assert result["summary"]["steps_executed"] == 3
    assert len(policy.calls) == 6
    assert torch.equal(policy.calls[0], policy.calls[1])
    assert not torch.equal(policy.calls[2], policy.calls[3])
    assert result["steps"][0]["policy_on_expert_stream"]["executed"] is False
    assert result["steps"][0]["policy_closed_loop"]["executed"] is True
    assert result["steps"][0]["dataset_reference"]["recovery_oracle"] is False
    assert result["summary"]["step_zero_same_input_maximum_absolute_difference"] == 0
    assert result["summary"]["closed_vs_policy_on_expert_action_l2"][1] > 0
    for index, action in enumerate(expert.actions):
        expected, _ = contract.clip(torch.as_tensor(rows[index]["action"]))
        assert torch.equal(action, expected)
    teacher_actions = [
        torch.tensor(step["policy_on_expert_stream"]["clipped_action"])
        for step in result["steps"]
    ]
    assert any(
        not torch.equal(executed, teacher)
        for executed, teacher in zip(expert.actions, teacher_actions)
    )
    assert expert.closed and closed.closed
    json.dumps(result, allow_nan=False)


def test_teacher_forced_rejects_nondeterministic_step_zero_and_closes_envs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_action_contract(
        REPOSITORY_ROOT / "configs" / "sim" / "aloha_insertion.yaml"
    )
    expert = _FakeEnvironment(contract, role="expert")
    closed = _FakeEnvironment(contract, role="policy")
    environments = [expert, closed]
    monkeypatch.setattr(
        sim_gate,
        "GymAlohaEnvironment",
        lambda *_args, **_kwargs: environments.pop(0),
    )
    rows, fields = _rows(contract)

    with pytest.raises(RuntimeError, match="identical reset inputs"):
        _teacher_forced_decomposition_episode(
            _NondeterministicPolicy(contract),
            _statistics(contract.dimension),
            contract,
            "instruction",
            rows,
            fields,
            seed=107,
        )

    assert expert.closed and closed.closed


def test_trace_rejects_shared_simulator_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_action_contract(
        REPOSITORY_ROOT / "configs" / "sim" / "aloha_insertion.yaml"
    )
    shared_backend = object()
    environments = [
        _FakeEnvironment(contract, role="expert", backend=shared_backend),
        _FakeEnvironment(contract, role="policy", backend=shared_backend),
    ]
    created = list(environments)
    monkeypatch.setattr(
        sim_gate,
        "GymAlohaEnvironment",
        lambda *_args, **_kwargs: environments.pop(0),
    )
    rows, fields = _rows(contract)

    with pytest.raises(RuntimeError, match="share a simulator backend"):
        _trajectory_divergence_episode(
            _FakePolicy(contract),
            _statistics(contract.dimension),
            contract,
            "instruction",
            rows,
            fields,
            seed=107,
            maximum_steps=3,
        )

    assert all(environment.closed for environment in created)


@pytest.mark.parametrize("failure", ["reset", "policy", "reward"])
def test_trace_failures_close_both_environments(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    contract = load_action_contract(
        REPOSITORY_ROOT / "configs" / "sim" / "aloha_insertion.yaml"
    )
    expert = _FakeEnvironment(
        contract,
        role="expert",
        nonfinite_reward=failure == "reward",
    )
    policy_environment = _FakeEnvironment(
        contract,
        role="policy",
        image_offset=1.0 if failure == "reset" else 0.0,
    )
    environments = [expert, policy_environment]
    monkeypatch.setattr(
        sim_gate,
        "GymAlohaEnvironment",
        lambda *_args, **_kwargs: environments.pop(0),
    )
    rows, fields = _rows(contract)
    policy = _FakePolicy(contract, nonfinite=failure == "policy")

    with pytest.raises(ValueError):
        _trajectory_divergence_episode(
            policy,
            _statistics(contract.dimension),
            contract,
            "instruction",
            rows,
            fields,
            seed=107,
            maximum_steps=3,
        )

    assert expert.closed
    assert policy_environment.closed
