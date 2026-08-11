from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from rosetta_reality.data.normalization import DatasetStatistics, NormalizationStats
from rosetta_reality.experiment import (
    file_sha256,
    frozen_artifact_recipe,
    load_experiment_config,
)
from rosetta_reality.sim import load_action_contract
from scripts import sim_gate
from scripts.sim_gate import (
    _cache_stride_matched_frames,
    _report_experiment_id,
    _requested_torch_device,
    _rollout_episode,
    _small_rollout_acceptance,
    _task_evaluation_acceptance,
    _validated_initial_alignment,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_CONFIG = (
    REPOSITORY_ROOT / "configs" / "experiments" / "m2_qwen08b_frozen_001.yaml"
)


def test_online_artifact_rejects_same_id_with_processor_drift(tmp_path: Path) -> None:
    experiment = load_experiment_config(EXPERIMENT_CONFIG, REPOSITORY_ROOT)
    artifact_config = frozen_artifact_recipe(experiment)
    artifact_config["processor"] = dict(artifact_config["processor"])
    artifact_config["processor"]["prompt_mode"] = "drifted"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(artifact_config), encoding="utf-8")
    manifest = {
        "status": "verified",
        "experiment_id": experiment["experiment_id"],
        "files": {"config.json": file_sha256(config_path)},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="differs at processor"):
        sim_gate._load_online_artifact(EXPERIMENT_CONFIG, tmp_path)


@pytest.mark.parametrize(
    ("frame_stride", "expected"),
    [(5, [0, 5, 10]), (2, [0, 2, 10])],
)
def test_recorded_domain_cache_frames_follow_configured_stride(
    frame_stride: int,
    expected: list[int],
) -> None:
    assert _cache_stride_matched_frames((0, 1, 2, 5, 10), frame_stride) == expected


@pytest.mark.parametrize("frame_stride", (0, -1, True))
def test_recorded_domain_cache_frames_reject_invalid_stride(
    frame_stride: int,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _cache_stride_matched_frames((0, 1, 2, 5, 10), frame_stride)


def test_simulation_device_selector_defaults_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROSETTA_TORCH_DEVICE", raising=False)

    assert _requested_torch_device() == torch.device("cpu")


def test_simulation_device_selector_rejects_unapproved_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROSETTA_TORCH_DEVICE", "cuda")

    with pytest.raises(ValueError, match="either cpu or xpu"):
        _requested_torch_device()


def _aggregate(**overrides: float | int) -> dict[str, float | int]:
    value: dict[str, float | int] = {
        "task_success_rate": 0.2,
        "invalid_action_rate": 0.0,
        "raw_limit_violation_rate": 0.0,
        "executed_limit_violation_rate": 0.0,
        "joint_limit_violations": 0,
        "unexpected_collisions": 0,
    }
    value.update(overrides)
    return value


def _small_rollout_metrics(**overrides: float | int) -> dict[str, float | int]:
    value: dict[str, float | int] = {
        "rollout_length": 20,
        "invalid_action_rate": 0.0,
        "raw_limit_violation_rate": 0.0,
        "executed_limit_violation_rate": 0.0,
        "joint_limit_violations": 0,
        "unexpected_collisions": 0,
    }
    value.update(overrides)
    return value


@pytest.mark.parametrize(
    "override",
    [
        {"raw_limit_violation_rate": 0.01},
        {"unexpected_collisions": 1},
    ],
)
def test_gate3_rejects_raw_limit_violations_and_collisions(
    override: dict[str, float | int],
) -> None:
    criteria = _small_rollout_acceptance(
        _small_rollout_metrics(**override),
        artifact_reload_verified=True,
    )

    assert not all(criteria.values())


def test_gate4_acceptance_separates_task_capability_from_safe_execution() -> None:
    criteria = _task_evaluation_acceptance(
        _aggregate(task_success_rate=0.0),
        total_steps=500,
        minimum_task_success_rate=0.2,
        maximum_unexpected_collisions=0,
    )

    assert criteria["finite_actions"]
    assert criteria["joint_limits_respected"]
    assert not criteria["minimum_task_success_rate"]
    assert not all(criteria.values())


@pytest.mark.parametrize("experiment_id", ("../escape", "nested/name", r"nested\name"))
def test_gate_report_rejects_path_escaping_experiment_id(experiment_id: str) -> None:
    with pytest.raises(ValueError, match="path-safe token"):
        _report_experiment_id({"experiment_id": experiment_id})


def test_gate4_acceptance_rejects_unexpected_collisions() -> None:
    criteria = _task_evaluation_acceptance(
        _aggregate(unexpected_collisions=1),
        total_steps=500,
        minimum_task_success_rate=0.2,
        maximum_unexpected_collisions=0,
    )

    assert not criteria["maximum_unexpected_collisions"]


@pytest.mark.parametrize(
    ("success_rate", "collision_limit"),
    [(-0.1, 0), (1.1, 0), (0.2, -1)],
)
def test_gate4_acceptance_validates_thresholds(
    success_rate: float,
    collision_limit: int,
) -> None:
    with pytest.raises(ValueError):
        _task_evaluation_acceptance(
            _aggregate(),
            total_steps=500,
            minimum_task_success_rate=success_rate,
            maximum_unexpected_collisions=collision_limit,
        )


def _alignment_report(path: Path, *, seed: int = 10, mae: float = 0.001) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "dataset": {"revision": "a" * 40, "manifest_sha256": "b" * 64},
                "episodes": [
                    {
                        "episode": 2,
                        "selected_seed": seed,
                        "selected_alignment": {"pooled_4x4_mae": mae},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_gate2_alignment_binds_episode_seed_and_dataset(tmp_path: Path) -> None:
    path = tmp_path / "alignment.json"
    _alignment_report(path)

    result = _validated_initial_alignment(
        path,
        episode=2,
        seed=10,
        dataset_revision="a" * 40,
        dataset_manifest_sha256="b" * 64,
        maximum_mae=0.005,
    )

    assert result["within_tolerance"]
    assert result["report_sha256"] == file_sha256(path)


def test_gate2_alignment_rejects_seed_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "alignment.json"
    _alignment_report(path, seed=11)

    with pytest.raises(ValueError, match="seed differs"):
        _validated_initial_alignment(
            path,
            episode=2,
            seed=10,
            dataset_revision="a" * 40,
            dataset_manifest_sha256="b" * 64,
            maximum_mae=0.005,
        )


def test_gate2_alignment_can_reuse_a_passed_matching_gate2_report(
    tmp_path: Path,
) -> None:
    path = tmp_path / "gate2.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "gate": "m2_gate_2_dataset_action_replay",
                "status": "passed",
                "dataset_revision": "a" * 40,
                "dataset_manifest_sha256": "b" * 64,
                "episode": 2,
                "seed": 10,
                "acceptance_criteria": {
                    "initial_object_pose_image_alignment": True,
                },
                "initial_object_pose_alignment": {
                    "pooled_4x4_mae": 0.001,
                    "within_tolerance": True,
                },
            }
        ),
        encoding="utf-8",
    )

    result = _validated_initial_alignment(
        path,
        episode=2,
        seed=10,
        dataset_revision="a" * 40,
        dataset_manifest_sha256="b" * 64,
        maximum_mae=0.005,
    )

    assert result["source_type"] == "passed_gate2_replay"
    assert result["within_tolerance"]
    assert result["report_sha256"] == file_sha256(path)


def test_gate2_alignment_rejects_failed_gate2_source(tmp_path: Path) -> None:
    path = tmp_path / "gate2.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "gate": "m2_gate_2_dataset_action_replay",
                "status": "failed",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="passed schema-v2"):
        _validated_initial_alignment(
            path,
            episode=2,
            seed=10,
            dataset_revision="a" * 40,
            dataset_manifest_sha256="b" * 64,
            maximum_mae=0.005,
        )


def test_rollout_executes_declared_chunk_steps_before_reobserving(monkeypatch) -> None:
    contract = replace(
        load_action_contract(
            REPOSITORY_ROOT / "configs" / "sim" / "aloha_insertion.yaml"
        ),
        chunk_execution="open_loop_first_3_then_reobserve",
        chunk_execution_steps=3,
    )
    stats = NormalizationStats(
        mean=torch.zeros(contract.dimension),
        std=torch.ones(contract.dimension),
    )
    statistics = DatasetStatistics(stats, stats, state_count=1, action_count=1)

    class FakeEnvironment:
        def __init__(self, _contract, *, maximum_episode_steps: int) -> None:
            self.maximum_episode_steps = maximum_episode_steps
            self.steps = 0

        def _observation(self):
            return {"robot_state": torch.zeros(contract.dimension), "images": {}}

        def reset(self, *, seed: int):
            assert seed == 7
            return self._observation()

        def step(self, action):
            assert action.shape == (contract.dimension,)
            self.steps += 1
            done = self.steps >= self.maximum_episode_steps
            return self._observation(), 0.0, done, {
                "is_success": False,
                "terminated": False,
                "truncated": done,
            }

        def contact_pairs(self):
            return []

        def is_unexpected_collision_pair(self, *_pair):
            return False

        def close(self) -> None:
            return None

    class FakePolicy:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, _observation, _state):
            self.calls += 1
            return torch.zeros(1, contract.chunk_length, contract.dimension)

    monkeypatch.setattr(sim_gate, "GymAlohaEnvironment", FakeEnvironment)
    policy = FakePolicy()

    metrics = _rollout_episode(
        policy,
        statistics,
        contract,
        "instruction",
        seed=7,
        maximum_steps=5,
    )

    assert metrics["rollout_length"] == 5
    assert metrics["policy_inference_calls"] == 2
    assert metrics["chunk_execution_steps"] == 3
    assert metrics["end_reason"] == "time_limit_truncation"


def test_gate1_horizon_exceeds_requested_control_steps(monkeypatch, tmp_path: Path) -> None:
    created_maximum_steps: list[int] = []

    class FakeEnvironment:
        def __init__(self, contract, *, maximum_episode_steps: int) -> None:
            self.contract = contract
            self.maximum_episode_steps = maximum_episode_steps
            self.steps = 0
            self.state = torch.zeros(contract.dimension)
            created_maximum_steps.append(maximum_episode_steps)

        def reset(self, *, seed: int):
            assert seed == 9
            self.steps = 0
            self.state = torch.zeros(self.contract.dimension)
            return {"robot_state": self.state.clone()}

        def step(self, action):
            self.contract.validate_tensor(action, allow_chunk=False)
            self.steps += 1
            self.state = action.clone()
            return (
                {"robot_state": self.state.clone()},
                0.0,
                self.steps >= self.maximum_episode_steps,
                {},
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr(sim_gate, "GymAlohaEnvironment", FakeEnvironment)
    monkeypatch.setattr(
        sim_gate,
        "_write_report",
        lambda _gate, _payload: tmp_path / "gate1.json",
    )

    result = sim_gate.scripted(
        REPOSITORY_ROOT / "configs" / "sim" / "aloha_insertion.yaml",
        seed=9,
        steps_per_dimension=5,
        experiment_id="test-experiment",
    )

    assert result == 0
    assert created_maximum_steps == [6]
