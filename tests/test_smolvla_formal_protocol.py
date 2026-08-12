import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from rosetta_reality.experiment import file_sha256
from rosetta_reality.sim import ActionDimension, load_action_contract
from scripts import smolvla_sim_gate
from scripts.evaluate_smolvla_validation import _percentile, _validation_indices
from scripts.export_smolvla import _validated_artifact_id
from scripts.inspect_smolvla_quarter import _expected_learning_rate
from scripts.run_smolvla_formal import (
    _load_formal_plan,
    _optimizer_arguments,
    _optimizer_contract,
    _training_coverage,
    _validate_monitoring,
    _validate_plan,
    _validate_saved_optimizer_contract,
)
from scripts.run_smolvla_phase import _validate_gate
from scripts.select_smolvla_checkpoint import _validated_checkpoint_hashes
from scripts.train_smolvla_trackio import _convert_statistics

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_training_coverage_counts_sample_exposures_as_dataset_passes() -> None:
    coverage = _training_coverage(
        {"batch_size": 1, "steps": 20_000, "minimum_dataset_passes": 1.0},
        20_000,
    )

    assert coverage["sample_exposures"] == 20_000
    assert coverage["dataset_passes"] == 1.0


def test_formal_monitoring_requires_exact_five_minute_sleep() -> None:
    plan = {
        "training": {"steps": 400, "checkpoint_steps": [100, 200, 300, 400]},
        "monitoring": {
            "policy": "sleep_between_quarter_checkpoints",
            "wake_fractions": [0.25, 0.5, 0.75, 1.0],
            "wake_steps": [100, 200, 300, 400],
            "blocking_command": "sleep",
            "sleep_poll_seconds": 300,
            "estimated_total_minutes": 60,
            "hidden_test_loaded": False,
        },
    }

    assert _validate_monitoring(plan)["sleep_poll_seconds"] == 300
    plan["monitoring"]["sleep_poll_seconds"] = 60
    with pytest.raises(ValueError, match="quarter-only sleep policy"):
        _validate_monitoring(plan)


@pytest.mark.parametrize("value", ["../escape", "/tmp/escape", "a/b", "ab"])
def test_export_artifact_id_rejects_path_escape(value: str) -> None:
    with pytest.raises(ValueError, match="path-safe component"):
        _validated_artifact_id(value)


def test_export_artifact_id_accepts_registered_component() -> None:
    value = "m2-smolvla450m-faust-b8-step1875-001"
    assert _validated_artifact_id(value) == value


def test_checkpoint_selection_binds_all_validated_processor_files(
    tmp_path: Path,
) -> None:
    files = {
        "model_safetensors_sha256": "model.safetensors",
        "policy_config_sha256": "config.json",
        "preprocessor_config_sha256": "policy_preprocessor.json",
        "postprocessor_config_sha256": "policy_postprocessor.json",
        "preprocessor_statistics_sha256": "custom_step_7_normalizer.safetensors",
        "postprocessor_statistics_sha256": (
            "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
        ),
    }
    for name, filename in files.items():
        if name == "preprocessor_config_sha256":
            (tmp_path / filename).write_text(
                json.dumps(
                    {
                        "steps": [
                            {
                                "registry_name": "normalizer_processor",
                                "state_file": files["preprocessor_statistics_sha256"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
        elif name == "postprocessor_config_sha256":
            (tmp_path / filename).write_text(
                json.dumps(
                    {
                        "steps": [
                            {
                                "registry_name": "unnormalizer_processor",
                                "state_file": files["postprocessor_statistics_sha256"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
        elif not (tmp_path / filename).exists():
            (tmp_path / filename).write_bytes(filename.encode())
    report = {
        "model_source": {
            name: file_sha256(tmp_path / filename)
            for name, filename in files.items()
            if name
            in {
                "model_safetensors_sha256",
                "policy_config_sha256",
                "preprocessor_config_sha256",
                "postprocessor_config_sha256",
            }
        },
        "processor_statistics": {
            name: file_sha256(tmp_path / filename)
            for name, filename in files.items()
            if name.endswith("statistics_sha256")
        },
    }

    assert set(_validated_checkpoint_hashes(tmp_path, report)) == set(files)
    (tmp_path / files["preprocessor_statistics_sha256"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint file changed"):
        _validated_checkpoint_hashes(tmp_path, report)


def test_simulation_policy_shape_follows_versioned_action_contract() -> None:
    contract = replace(
        load_action_contract(
            REPOSITORY_ROOT / "configs/sim/aloha_insertion_smolvla.yaml"
        ),
        dimensions=tuple(
            ActionDimension(
                name=f"action_{index}",
                unit="normalized",
                minimum=-1.0,
                maximum=1.0,
            )
            for index in range(7)
        ),
        chunk_length=12,
    )
    policy_config = SimpleNamespace(
        chunk_size=12,
        output_features={"action": SimpleNamespace(shape=(7,))},
        input_features={"observation.state": SimpleNamespace(shape=(5,))},
    )

    assert smolvla_sim_gate._validate_policy_contract_shape(policy_config, contract) == 5
    policy_config.chunk_size = 50
    with pytest.raises(ValueError, match="policy dimensions"):
        smolvla_sim_gate._validate_policy_contract_shape(policy_config, contract)


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

    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
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


def test_three_furnace_program_is_exactly_ordered_and_bounded() -> None:
    registry = yaml.safe_load(
        (
            REPOSITORY_ROOT
            / "configs/vla/smolvla_450m_aloha_insertion_three_furnace_001.yaml"
        ).read_text(encoding="utf-8")
    )

    assert registry["maximum_formal_runs"] == 3
    assert registry["execution"] == "strictly_sequential"
    assert registry["codenames_in_order"] == ["Odyssey", "Don Quixote", "Moby Dick"]
    assert len(registry["runs"]) == 3
    assert [run["ordinal"] for run in registry["runs"]] == [1, 2, 3]
    assert [run["codename"] for run in registry["runs"]] == registry[
        "codenames_in_order"
    ]


def test_historical_formal_plan_rejects_current_security_hardened_runtime() -> None:
    with pytest.raises(ValueError, match="implementation changed"):
        _validate_plan(
            REPOSITORY_ROOT
            / "configs/vla/smolvla_450m_aloha_insertion_formal_optimized_001.yaml",
            require_runtime_evidence=False,
        )


@pytest.mark.parametrize(
    ("filename", "codename", "ordinal", "peak_lr", "decay_lr"),
    [
        ("smolvla_450m_aloha_insertion_odyssey_001.yaml", "Odyssey", 1, 1.0e-4, 2.5e-6),
        (
            "smolvla_450m_aloha_insertion_don_quixote_001.yaml",
            "Don Quixote",
            2,
            7.5e-5,
            1.875e-6,
        ),
        (
            "smolvla_450m_aloha_insertion_moby_dick_001.yaml",
            "Moby Dick",
            3,
            1.25e-4,
            3.125e-6,
        ),
    ],
)
def test_furnace_plans_preregister_explicit_optimizer_and_quarter_wakes(
    filename: str,
    codename: str,
    ordinal: int,
    peak_lr: float,
    decay_lr: float,
) -> None:
    plan = _load_formal_plan(REPOSITORY_ROOT / "configs/vla" / filename)
    contract = _optimizer_contract(plan["training"])

    assert plan["furnace_program"]["codename"] == codename
    assert plan["furnace_program"]["ordinal"] == ordinal
    assert plan["furnace_program"]["maximum_formal_runs"] == 3
    assert plan["monitoring"]["wake_steps"] == [420, 840, 1260, 1680]
    assert plan["monitoring"]["blocking_command"] == "sleep"
    assert contract is not None
    assert contract["optimizer"]["lr"] == peak_lr
    assert contract["scheduler"]["peak_lr"] == peak_lr
    assert contract["scheduler"]["decay_lr"] == decay_lr
    assert contract["scheduler"]["num_warmup_steps"] == 56
    assert contract["scheduler"]["num_decay_steps"] == 1680


def test_optimizer_contract_drives_cli_saved_config_and_expected_lr() -> None:
    training = {
        "steps": 1680,
        "optimizer": {
            "type": "adamw",
            "lr": 1.0e-4,
            "betas": [0.9, 0.95],
            "eps": 1.0e-8,
            "weight_decay": 1.0e-10,
            "grad_clip_norm": 10.0,
        },
        "scheduler": {
            "type": "cosine_decay_with_warmup",
            "num_warmup_steps": 56,
            "num_decay_steps": 1680,
            "peak_lr": 1.0e-4,
            "decay_lr": 2.5e-6,
        },
    }
    contract = _optimizer_contract(training)
    assert contract is not None
    arguments = _optimizer_arguments(training)
    assert "--policy.optimizer_lr=0.0001" in arguments
    assert "--policy.scheduler_warmup_steps=56" in arguments
    assert "--policy.scheduler_decay_steps=1680" in arguments
    assert "--policy.scheduler_decay_lr=2.5e-06" in arguments

    saved = {
        "optimizer": contract["optimizer"],
        "scheduler": contract["scheduler"],
        "policy": {
            "optimizer_lr": 1.0e-4,
            "optimizer_betas": [0.9, 0.95],
            "optimizer_eps": 1.0e-8,
            "optimizer_weight_decay": 1.0e-10,
            "optimizer_grad_clip_norm": 10.0,
            "scheduler_warmup_steps": 56,
            "scheduler_decay_steps": 1680,
            "scheduler_decay_lr": 2.5e-6,
        },
    }
    assert _validate_saved_optimizer_contract(saved, training) == contract
    expected_at_warmup_boundary = 1.0e-4 * (
        (1.0 - 0.025) * 0.5 * (1.0 + math.cos(math.pi * 56 / 1680)) + 0.025
    )
    assert _expected_learning_rate(contract, 0) == pytest.approx(1.0e-4 / 57)
    assert _expected_learning_rate(contract, 56) == pytest.approx(
        expected_at_warmup_boundary
    )
    assert _expected_learning_rate(contract, 1680) == pytest.approx(2.5e-6)


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


def test_postprocessed_execution_keeps_decoder_violation_as_diagnostic(
    monkeypatch,
) -> None:
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
            return action, contract.clip(action)[0]

    monkeypatch.setattr(smolvla_sim_gate, "GymAlohaEnvironment", FakeEnvironment)
    policy = FakePolicy()
    metrics = smolvla_sim_gate._rollout(
        policy,
        contract,
        "instruction",
        seed=7,
        maximum_steps=1,
        project_policy_output=False,
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


def test_gate2_replay_rejects_sealed_episode(tmp_path: Path) -> None:
    report = tmp_path / "gate2.json"
    report.write_text(
        json.dumps(
            {
                "status": "passed",
                "gate": "m2_gate_2_dataset_action_replay",
                "experiment_id": "experiment",
                "action_contract_sha256": "a" * 64,
                "dataset_revision": "b" * 40,
                "episode": 31,
                "acceptance_criteria": {"timestamp_alignment": True},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Gate 2 dataset identity"):
        _validate_gate(
            report,
            expected_gate="m2_gate_2_dataset_action_replay",
            experiment_id="experiment",
            contract_sha256="a" * 64,
            dataset_revision="b" * 40,
            allowed_replay_episodes=[2, 7, 22],
        )
