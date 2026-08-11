from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from rosetta_reality.sim import load_action_contract
from scripts import smolvla_sim_gate
from scripts.evaluate_smolvla_validation import _percentile, _validation_indices
from scripts.run_smolvla_formal import _training_coverage, _validate_plan
from scripts.train_smolvla_trackio import _convert_statistics

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_training_coverage_counts_sample_exposures_as_dataset_passes() -> None:
    coverage = _training_coverage(
        {"batch_size": 1, "steps": 20_000, "minimum_dataset_passes": 1.0},
        20_000,
    )

    assert coverage["sample_exposures"] == 20_000
    assert coverage["dataset_passes"] == 1.0


def test_training_coverage_rejects_a_registered_partial_pass() -> None:
    with pytest.raises(ValueError, match="below the registered minimum"):
        _training_coverage(
            {"batch_size": 1, "steps": 1_000, "minimum_dataset_passes": 1.0},
            20_000,
        )


def test_formal_plan_keeps_train_validation_and_test_disjoint() -> None:
    plan = yaml.safe_load(
        (REPOSITORY_ROOT / "configs/vla/smolvla_450m_aloha_insertion_formal_001.yaml").read_text(
            encoding="utf-8"
        )
    )
    base = yaml.safe_load(
        (REPOSITORY_ROOT / "configs/vla/smolvla_450m_aloha_insertion.yaml").read_text(
            encoding="utf-8"
        )
    )

    train = set(plan["training"]["episodes"])
    validation = set(plan["validation"]["episodes"])
    hidden_test = set(base["dataset"]["test_episodes"])
    assert list(plan["training"]["episodes"]) == base["dataset"]["train_episodes"]
    assert list(plan["validation"]["episodes"]) == base["dataset"]["validation_episodes"]
    assert not train & validation
    assert not train & hidden_test
    assert not validation & hidden_test
    assert plan["initialization"]["overfit_checkpoint_used"] is False


def test_optimized_formal_plan_is_bound_to_measured_xpu_evidence() -> None:
    plan_path = (
        REPOSITORY_ROOT
        / "configs/vla/smolvla_450m_aloha_insertion_formal_optimized_001.yaml"
    )

    plan, _base_path, _experiment = _validate_plan(
        plan_path,
        require_runtime_evidence=False,
    )
    coverage = _training_coverage(plan["training"], 20_000)

    assert plan["resources"]["memory_limit"] == "8g"
    assert plan["resources"]["memory_swap_limit"] == "8g"
    assert plan["training"]["batch_size"] == 12
    assert plan["training"]["policy"] == {
        "empty_cameras": 2,
        "compile_model": True,
        "compile_mode": "reduce-overhead",
        "skip_fully_masked_camera_encoding": True,
    }
    assert coverage["dataset_passes"] == pytest.approx(1.008)
    assert coverage["sample_exposures"] == 20_160


def test_optimized_formal_plan_rejects_more_than_eight_gigabytes(tmp_path: Path) -> None:
    source = (
        REPOSITORY_ROOT
        / "configs/vla/smolvla_450m_aloha_insertion_formal_optimized_001.yaml"
    )
    plan = yaml.safe_load(source.read_text(encoding="utf-8"))
    plan["resources"]["memory_limit"] = "9g"
    plan["resources"]["memory_swap_limit"] = "9g"
    modified = tmp_path / "formal-optimized-over-budget.yaml"
    modified.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="authorized 8 GB"):
        _validate_plan(modified)


def test_simulation_plan_preserves_gate_order_and_sealed_test() -> None:
    plan = yaml.safe_load(
        (REPOSITORY_ROOT / "configs/vla/smolvla_450m_aloha_insertion_sim_001.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert plan["role"] == "vla"
    assert plan["stage"] == "m2_closed_loop_simulation"
    assert plan["status"] == "preregistered"
    assert plan["gate4"]["require_gate3_passed"] is True
    assert plan["inference"]["chunk_execution_steps"] == 1
    assert plan["resources"]["runtime"] == "docker_linux_from_wsl"
    assert plan["resources"]["memory_limit"] == plan["resources"]["memory_swap_limit"]
    assert plan["hidden_test_loaded"] is False

    projected = yaml.safe_load(
        (REPOSITORY_ROOT / "configs/vla/smolvla_450m_aloha_insertion_sim_002.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert projected["prior_failure"]["failed_criterion"] == "raw_actions_within_contract"
    assert projected["inference"]["policy_output_projection"] == "action_contract_clip"
    assert projected["inference"]["unprojected_decoder_action_role"] == (
        "non_gating_diagnostic"
    )
    assert projected["gate4"]["require_gate3_passed"] is True
    assert projected["hidden_test_loaded"] is False


def test_registered_projection_keeps_decoder_violation_as_diagnostic(monkeypatch) -> None:
    contract = load_action_contract(
        REPOSITORY_ROOT / "configs/sim/aloha_insertion_smolvla.yaml"
    )

    class FakeEnvironment:
        def __init__(self, _contract, *, maximum_episode_steps: int) -> None:
            self.last_clip_mask = torch.zeros(contract.dimension, dtype=torch.bool)

        def reset(self, *, seed: int):
            return {"robot_state": torch.zeros(contract.dimension), "images": {}}

        def step(self, action):
            assert torch.equal(action, contract.clip(action)[0])
            return self.reset(seed=0), 0.0, True, {
                "is_success": False,
                "terminated": False,
                "truncated": True,
            }

        def contact_pairs(self):
            return ()

        def is_unexpected_collision_pair(self, *_pair):
            return False

        def close(self) -> None:
            return None

    class FakePolicy:
        def __init__(self) -> None:
            self.noise_configuration = None

        def configure_noise(self, mode, seed) -> None:
            self.noise_configuration = (mode, seed)

        def predict(self, _observation, _instruction):
            action = torch.zeros(contract.chunk_length, contract.dimension)
            action[0, 0] = contract.upper_bounds[0] + 1
            return action

    monkeypatch.setattr(smolvla_sim_gate, "GymAlohaEnvironment", FakeEnvironment)
    policy = FakePolicy()
    metrics = smolvla_sim_gate._rollout(
        policy,
        contract,
        "instruction",
        seed=7,
        maximum_steps=1,
        project_policy_output=True,
        noise_mode="seeded_standard_normal",
        policy_noise_seed=19,
    )

    assert policy.noise_configuration == ("seeded_standard_normal", 19)
    assert metrics["policy_noise_seed"] == 19
    assert metrics["unprojected_limit_violation_rate"] == 1 / contract.dimension
    assert metrics["policy_output_limit_violation_rate"] == 0
    assert metrics["executed_limit_violation_rate"] == 0
    assert metrics["unprojected_dimension_diagnostics"]["left_waist"][
        "strict_violation_count"
    ] == 1


def test_rollout_reuses_predicted_chunk_before_reobserving(monkeypatch) -> None:
    contract = replace(
        load_action_contract(
            REPOSITORY_ROOT / "configs/sim/aloha_insertion_smolvla.yaml"
        ),
        chunk_execution="diagnostic_first_5_then_reobserve",
        chunk_execution_steps=5,
    )

    class FakeEnvironment:
        def __init__(self, _contract, *, maximum_episode_steps: int) -> None:
            self.steps = 0
            self.maximum_episode_steps = maximum_episode_steps
            self.last_clip_mask = torch.zeros(contract.dimension, dtype=torch.bool)

        def reset(self, *, seed: int):
            self.steps = 0
            return {"robot_state": torch.zeros(contract.dimension), "images": {}}

        def step(self, action):
            self.steps += 1
            done = self.steps >= self.maximum_episode_steps
            observation = {
                "robot_state": torch.zeros(contract.dimension),
                "images": {},
            }
            return observation, 0.0, done, {
                "is_success": False,
                "terminated": False,
                "truncated": done,
            }

        def contact_pairs(self):
            return ()

        def is_unexpected_collision_pair(self, *_pair):
            return False

        def close(self) -> None:
            return None

    class FakePolicy:
        def __init__(self) -> None:
            self.predict_calls = 0

        def configure_noise(self, mode, seed) -> None:
            assert (mode, seed) == ("zeros", None)

        def predict(self, _observation, _instruction):
            self.predict_calls += 1
            return torch.zeros(contract.chunk_length, contract.dimension)

    monkeypatch.setattr(smolvla_sim_gate, "GymAlohaEnvironment", FakeEnvironment)
    policy = FakePolicy()
    metrics = smolvla_sim_gate._rollout(
        policy,
        contract,
        "instruction",
        seed=7,
        maximum_steps=10,
        project_policy_output=True,
        noise_mode="zeros",
        policy_noise_seed=None,
    )

    assert metrics["rollout_length"] == 10
    assert metrics["policy_inference_calls"] == 2
    assert policy.predict_calls == 2


def test_train_only_statistics_convert_without_changing_values() -> None:
    converted = _convert_statistics(
        {
            "action": {
                "mean": [0.1, -0.2],
                "std": [0.3, 0.4],
                "min": [-1.0, -2.0],
                "max": [1.0, 2.0],
                "count": [20],
            }
        }
    )

    assert converted["action"]["mean"].dtype == torch.float64
    assert converted["action"]["count"].dtype == torch.int64
    assert converted["action"]["mean"].tolist() == [0.1, -0.2]
    assert converted["action"]["count"].tolist() == [20]


def test_validation_indices_preserve_registered_episode_and_offset_order() -> None:
    class Episodes:
        values = {
            "episode_index": [2, 7],
            "dataset_from_index": [100, 500],
            "length": [20, 30],
        }

        def __getitem__(self, key: str) -> list[int]:
            return self.values[key]

    class Metadata:
        episodes = Episodes()

    class Dataset:
        meta = Metadata()
        absolute_to_relative_idx = {100: 0, 105: 5, 500: 20, 505: 25}

    assert _validation_indices(Dataset(), [7, 2], [0, 5]) == [
        (7, 0, 20),
        (7, 5, 25),
        (2, 0, 0),
        (2, 5, 5),
    ]


def test_validation_latency_percentile_uses_nearest_rank() -> None:
    assert _percentile([0.4, 0.1, 0.3, 0.2], 0.95) == 0.4
