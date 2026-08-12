from pathlib import Path

import yaml

from rosetta_reality.vla import load_smolvla_experiment

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = (
    REPOSITORY_ROOT
    / "configs/vla/smolvla_450m_aloha_insertion_faust_001.yaml"
)
EXPERIMENT_PATH = (
    REPOSITORY_ROOT
    / "configs/vla/smolvla_450m_aloha_insertion_action_repair_bounded_gripper_003.yaml"
)
BATCH8_PLAN_PATH = (
    REPOSITORY_ROOT
    / "configs/vla/smolvla_450m_aloha_insertion_faust_batch8_002.yaml"
)


def test_faust_is_fresh_base_one_pass_and_quarter_only() -> None:
    plan = yaml.safe_load(PLAN_PATH.read_text(encoding="utf-8"))
    experiment = yaml.safe_load(EXPERIMENT_PATH.read_text(encoding="utf-8"))

    assert plan["plan_id"] == "m2-smolvla450m-faust-001"
    assert plan["furnace_program"]["codename"] == "Faust"
    assert plan["furnace_program"]["maximum_formal_runs"] == 1
    assert plan["initialization"]["source"] == "revision_pinned_base"
    assert plan["initialization"]["overfit_checkpoint_used"] is False
    assert plan["training"]["episodes"] == [
        49,
        4,
        23,
        43,
        21,
        37,
        18,
        34,
        0,
        47,
        38,
        29,
        3,
        26,
        14,
        17,
        44,
        30,
        15,
        42,
        10,
        35,
        25,
        32,
        19,
        36,
        41,
        28,
        8,
        27,
        16,
        11,
        2,
        20,
        9,
        39,
        46,
        48,
        12,
        40,
    ]
    assert plan["training"]["batch_size"] == 1
    assert plan["training"]["steps"] == 20_000
    assert plan["training"]["minimum_dataset_passes"] == 1.0
    assert plan["training"]["checkpoint_steps"] == [5_000, 10_000, 15_000, 20_000]
    assert plan["training"]["save_freq"] == 5_000
    assert plan["monitoring"] == {
        "policy": "sleep_between_quarter_checkpoints",
        "estimated_total_minutes": 360,
        "wake_fractions": [0.25, 0.5, 0.75, 1.0],
        "wake_steps": [5_000, 10_000, 15_000, 20_000],
        "blocking_command": "sleep",
        "sleep_poll_seconds": 60,
        "hidden_test_loaded": False,
    }
    assert plan["training"]["policy"] == {
        "empty_cameras": 2,
        "compile_model": False,
        "skip_fully_masked_camera_encoding": False,
    }
    assert experiment["phases"]["overfit"]["steps"] == 200


def test_faust_preserves_repaired_action_boundary_and_sealed_split() -> None:
    plan = yaml.safe_load(PLAN_PATH.read_text(encoding="utf-8"))

    train = set(plan["training"]["episodes"])
    validation = set(plan["validation"]["episodes"])
    hidden = {31, 6, 1, 24, 5}
    assert not train & validation
    assert not train & hidden
    assert not validation & hidden
    assert plan["action_space"] == {
        "adapt_to_pi_aloha": False,
        "dataset_space": "standard_aloha_joint_position",
        "explicit": True,
        "model_internal_space": "normalized_pi_aloha_arms_bounded_sine_grippers",
        "normalization": "train_only_mean_std_after_representation_adapter",
        "reject_source_beyond_contract_tolerance": True,
        "representation_adapter": "rosetta_pi_aloha_arms_bounded_sine_grippers",
        "representation_adapter_stage": (
            "after_target_projection_before_normalization"
        ),
        "schema_version": 1,
        "target_projection": "action_contract_clip",
        "target_projection_stage": "before_normalization",
    }
    assert plan["hidden_test_loaded"] is False
    assert plan["validation"]["hidden_test_loaded"] is False


def test_faust_hash_inventory_covers_runtime_boundary() -> None:
    plan = yaml.safe_load(PLAN_PATH.read_text(encoding="utf-8"))
    files = set(plan["implementation_files"])

    assert {
        "scripts/run_smolvla_action_repair_formal.py",
        "scripts/train_smolvla_action_repair_formal.py",
        "scripts/evaluate_smolvla_action_repair_validation.py",
        "scripts/inspect_smolvla_faust_quarter.py",
        "src/rosetta_reality/vla/action_space.py",
        "src/rosetta_reality/vla/checkpoint_memory.py",
        "src/rosetta_reality/vla/processor.py",
        "src/rosetta_reality/tracking/trackio_lerobot.py",
    } <= files
    assert all(len(value) == 64 for value in plan["implementation_files"].values())


def test_faust_batch8_restart_is_one_pass_and_quarter_only() -> None:
    plan = yaml.safe_load(BATCH8_PLAN_PATH.read_text(encoding="utf-8"))
    experiment = load_smolvla_experiment(EXPERIMENT_PATH, REPOSITORY_ROOT)

    assert plan["furnace_program"]["codename"] == "Faust"
    assert plan["furnace_program"]["attempt"] == 2
    assert plan["supersedes"]["reason"] == "user_requested_batch8_restart"
    assert plan["initialization"]["source"] == "revision_pinned_base"
    assert plan["initialization"]["overfit_checkpoint_used"] is False
    assert plan["initialization"]["interrupted_checkpoint_used"] is False
    assert plan["training"]["episodes"] == experiment["dataset"]["train_episodes"]
    assert plan["training"]["batch_size"] == 8
    assert plan["training"]["steps"] == 2_500
    assert plan["training"]["batch_size"] * plan["training"]["steps"] == 20_000
    assert plan["training"]["checkpoint_steps"] == [625, 1_250, 1_875, 2_500]
    assert plan["monitoring"]["wake_steps"] == [625, 1_250, 1_875, 2_500]
    assert plan["monitoring"]["blocking_command"] == "sleep"
    assert plan["optimizer_smoke"]["batch_size"] == 8
    assert plan["optimizer_smoke"]["steps"] == 2
    assert plan["training"]["policy"] == {
        "empty_cameras": 2,
        "compile_model": True,
        "compile_mode": "reduce-overhead",
        "skip_fully_masked_camera_encoding": True,
    }
    assert plan["resources"]["memory_limit"] == "8g"
    assert plan["resources"]["memory_swap_limit"] == "8g"
