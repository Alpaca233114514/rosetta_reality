from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from rosetta_reality.experiment import (
    _branch_from_git_reference,
    frozen_artifact_recipe,
    load_experiment_config,
    stable_hash,
    validate_frozen_artifact_recipe,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "experiments" / "m2_qwen08b_frozen_001.yaml"
SPATIAL_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "experiments"
    / "m2_qwen08b_frozen_002_spatial.yaml"
)
COMBINED_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "experiments"
    / "m2_qwen08b_frozen_003_global_spatial.yaml"
)
XPU_CONTROL_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "experiments"
    / "m2_qwen08b_frozen_004_spatial_xpu_control.yaml"
)
RESIDUAL_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "experiments"
    / "m2_qwen08b_frozen_005_spatial_residual_xpu.yaml"
)
INSTRUCT_CONTROL_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "experiments"
    / "m2_qwen08b_frozen_006_instruct_spatial_xpu_control.yaml"
)
STATE_JITTER_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "experiments"
    / "m2_qwen08b_frozen_008_state_jitter_xpu.yaml"
)
FIRST_ACTION_LOSS_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "experiments"
    / "m2_qwen08b_frozen_009_first_action_loss_xpu.yaml"
)
TRAIN_STATE_PAIRING_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "experiments"
    / "m2_qwen08b_frozen_010_train_state_pairing_xpu.yaml"
)
EXTENDED_HORIZON_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "experiments"
    / "m2_qwen08b_frozen_011_extended_horizon_xpu.yaml"
)
STRIDE2_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "experiments"
    / "m2_qwen08b_frozen_012_stride2_xpu.yaml"
)
STRIDE2_FIRST_ACTION_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "experiments"
    / "m2_qwen08b_frozen_013_stride2_first_action_xpu.yaml"
)
EARLY_PHASE_FIRST_ACTION_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "experiments"
    / "m2_qwen08b_frozen_014_early_phase_first_action_xpu.yaml"
)
FUSION512_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "experiments"
    / "m2_qwen08b_frozen_015_fusion512_xpu.yaml"
)


def test_m2_experiment_has_disjoint_complete_episode_split() -> None:
    config = load_experiment_config(CONFIG_PATH, REPOSITORY_ROOT)

    split = config["dataset"]["split"]
    assert len(split["train"]) == 40
    assert len(split["validation"]) == 5
    assert len(split["test"]) == 5
    assert len(set(split["train"]) | set(split["validation"]) | set(split["test"])) == 50
    assert config["action_expert"]["output_projection"] == "clip_to_action_contract"


def test_spatial_candidate_changes_representation_but_preserves_training_axes() -> None:
    reference = load_experiment_config(CONFIG_PATH, REPOSITORY_ROOT)
    candidate = load_experiment_config(SPATIAL_CONFIG_PATH, REPOSITORY_ROOT)

    assert candidate["backbone"]["pooling"] == "image_spatial_2x2"
    assert candidate["backbone"]["identifier"] == reference["backbone"]["identifier"]
    assert candidate["backbone"]["adaptation"] == reference["backbone"]["adaptation"]
    assert candidate["dataset"] == reference["dataset"]
    assert candidate["action_contract"] == reference["action_contract"]
    assert candidate["action_expert"] == reference["action_expert"]
    assert candidate["training"] == reference["training"]


def test_combined_candidate_preserves_semantic_axes_and_declares_xpu_runtime() -> None:
    reference = load_experiment_config(CONFIG_PATH, REPOSITORY_ROOT)
    candidate = load_experiment_config(COMBINED_CONFIG_PATH, REPOSITORY_ROOT)

    assert candidate["backbone"]["pooling"] == (
        "attention_masked_mean_plus_image_spatial_2x2"
    )
    assert candidate["backbone"]["identifier"] == reference["backbone"]["identifier"]
    assert candidate["backbone"]["adaptation"] == reference["backbone"]["adaptation"]
    assert candidate["dataset"] == reference["dataset"]
    assert candidate["action_contract"] == reference["action_contract"]
    assert candidate["action_expert"] == reference["action_expert"]
    assert candidate["training"] == reference["training"]
    assert candidate["resources"]["training_device"] == "xpu"


def test_residual_candidate_changes_only_action_parameterization_from_xpu_control() -> None:
    control = load_experiment_config(XPU_CONTROL_CONFIG_PATH, REPOSITORY_ROOT)
    candidate = load_experiment_config(RESIDUAL_CONFIG_PATH, REPOSITORY_ROOT)

    control_expert = copy.deepcopy(control["action_expert"])
    candidate_expert = copy.deepcopy(candidate["action_expert"])
    assert control_expert.pop("prediction_parameterization") == "absolute"
    assert candidate_expert.pop("prediction_parameterization") == (
        "residual_from_current_state"
    )
    assert candidate_expert == control_expert
    assert candidate["backbone"] == control["backbone"]
    assert candidate["dataset"] == control["dataset"]
    assert candidate["action_contract"] == control["action_contract"]
    assert candidate["training"] == control["training"]
    assert candidate["resources"] == control["resources"]


def test_instruct_is_explicitly_m2_ineligible_and_preserves_downstream_axes() -> None:
    control = load_experiment_config(XPU_CONTROL_CONFIG_PATH, REPOSITORY_ROOT)
    candidate = load_experiment_config(INSTRUCT_CONTROL_CONFIG_PATH, REPOSITORY_ROOT)

    assert candidate["experiment_role"] == "auxiliary_backbone_control"
    assert candidate["m2_completion_eligible"] is False
    assert candidate["backbone"]["identifier"] == "Qwen/Qwen3.5-0.8B"
    assert candidate["backbone"]["processor"]["prompt_mode"] == "chat_template"
    assert candidate["backbone"]["pooling"] == control["backbone"]["pooling"]
    assert candidate["dataset"] == control["dataset"]
    assert candidate["action_contract"] == control["action_contract"]
    assert candidate["action_expert"] == control["action_expert"]
    assert candidate["training"] == control["training"]
    assert candidate["resources"] == control["resources"]


def test_state_jitter_candidate_changes_only_training_input_perturbation() -> None:
    control = load_experiment_config(XPU_CONTROL_CONFIG_PATH, REPOSITORY_ROOT)
    candidate = load_experiment_config(STATE_JITTER_CONFIG_PATH, REPOSITORY_ROOT)

    control_training = copy.deepcopy(control["training"])
    candidate_training = copy.deepcopy(candidate["training"])
    assert "state_noise_std_normalized" not in control_training
    assert candidate_training.pop("state_noise_std_normalized") == 0.05
    assert candidate_training == control_training
    assert candidate["backbone"] == control["backbone"]
    assert candidate["dataset"] == control["dataset"]
    assert candidate["action_contract"] == control["action_contract"]
    assert candidate["action_expert"] == control["action_expert"]
    assert candidate["resources"] == control["resources"]


def test_first_action_loss_candidate_changes_only_execution_aligned_loss() -> None:
    control = load_experiment_config(XPU_CONTROL_CONFIG_PATH, REPOSITORY_ROOT)
    candidate = load_experiment_config(FIRST_ACTION_LOSS_CONFIG_PATH, REPOSITORY_ROOT)

    control_training = copy.deepcopy(control["training"])
    candidate_training = copy.deepcopy(candidate["training"])
    assert "first_action_loss_weight" not in control_training
    assert candidate_training.pop("first_action_loss_weight") == 1.0
    assert candidate_training == control_training
    assert candidate["backbone"] == control["backbone"]
    assert candidate["dataset"] == control["dataset"]
    assert candidate["action_contract"] == control["action_contract"]
    assert candidate["action_expert"] == control["action_expert"]
    assert candidate["simulation"] == control["simulation"]
    assert candidate["resources"] == control["resources"]


def test_train_state_pairing_candidate_changes_only_training_input_pairing() -> None:
    control = load_experiment_config(XPU_CONTROL_CONFIG_PATH, REPOSITORY_ROOT)
    candidate = load_experiment_config(TRAIN_STATE_PAIRING_CONFIG_PATH, REPOSITORY_ROOT)

    control_training = copy.deepcopy(control["training"])
    candidate_training = copy.deepcopy(candidate["training"])
    pairing = candidate_training.pop("aligned_expert_replay_state_pairing")
    assert pairing == {
        "enabled": True,
        "weight": 1.0,
        "manifest": "state-pairing/manifest.json",
    }
    assert candidate_training == control_training
    assert candidate["backbone"] == control["backbone"]
    assert candidate["dataset"] == control["dataset"]
    assert candidate["action_contract"] == control["action_contract"]
    assert candidate["action_expert"] == control["action_expert"]
    assert candidate["simulation"] == control["simulation"]
    assert candidate["resources"] == control["resources"]


def test_extended_horizon_candidate_changes_only_maximum_epochs() -> None:
    control = load_experiment_config(XPU_CONTROL_CONFIG_PATH, REPOSITORY_ROOT)
    candidate = load_experiment_config(EXTENDED_HORIZON_CONFIG_PATH, REPOSITORY_ROOT)

    control_training = copy.deepcopy(control["training"])
    candidate_training = copy.deepcopy(candidate["training"])
    assert control_training.pop("maximum_epochs") == 40
    assert candidate_training.pop("maximum_epochs") == 80
    assert candidate_training == control_training
    for key in ("backbone", "dataset", "action_contract", "action_expert", "simulation"):
        assert candidate[key] == control[key]
    assert candidate["resources"] == control["resources"]


def test_stride2_candidate_changes_only_visible_anchor_density() -> None:
    control = load_experiment_config(XPU_CONTROL_CONFIG_PATH, REPOSITORY_ROOT)
    candidate = load_experiment_config(STRIDE2_CONFIG_PATH, REPOSITORY_ROOT)

    control_dataset = copy.deepcopy(control["dataset"])
    candidate_dataset = copy.deepcopy(candidate["dataset"])
    assert control_dataset.pop("frame_stride") == 5
    assert candidate_dataset.pop("frame_stride") == 2
    assert candidate_dataset == control_dataset
    for key in ("backbone", "action_contract", "action_expert", "training", "simulation"):
        assert candidate[key] == control[key]
    assert candidate["resources"] == control["resources"]


def test_stride2_first_action_candidate_changes_only_execution_aligned_loss() -> None:
    control = load_experiment_config(STRIDE2_CONFIG_PATH, REPOSITORY_ROOT)
    candidate = load_experiment_config(STRIDE2_FIRST_ACTION_CONFIG_PATH, REPOSITORY_ROOT)

    control_training = copy.deepcopy(control["training"])
    candidate_training = copy.deepcopy(candidate["training"])
    assert "first_action_loss_weight" not in control_training
    assert candidate_training.pop("first_action_loss_weight") == 1.0
    assert candidate_training == control_training
    for key in ("backbone", "dataset", "action_contract", "action_expert", "simulation"):
        assert candidate[key] == control[key]
    assert candidate["resources"] == control["resources"]


def test_early_phase_candidate_changes_only_scoped_execution_aligned_loss() -> None:
    control = load_experiment_config(STRIDE2_CONFIG_PATH, REPOSITORY_ROOT)
    candidate = load_experiment_config(EARLY_PHASE_FIRST_ACTION_CONFIG_PATH, REPOSITORY_ROOT)

    control_training = copy.deepcopy(control["training"])
    candidate_training = copy.deepcopy(candidate["training"])
    assert "early_phase_first_action_loss" not in control_training
    assert candidate_training.pop("early_phase_first_action_loss") == {
        "weight": 1.0,
        "maximum_frame_index_exclusive": 50,
        "expected_selected_train_samples": 1000,
    }
    assert candidate_training == control_training
    for key in ("backbone", "dataset", "action_contract", "action_expert", "simulation"):
        assert candidate[key] == control[key]
    assert candidate["resources"] == control["resources"]


def test_fusion512_candidate_changes_only_action_expert_fusion_width() -> None:
    control = load_experiment_config(STRIDE2_CONFIG_PATH, REPOSITORY_ROOT)
    candidate = load_experiment_config(FUSION512_CONFIG_PATH, REPOSITORY_ROOT)

    control_expert = copy.deepcopy(control["action_expert"])
    candidate_expert = copy.deepcopy(candidate["action_expert"])
    assert control_expert.pop("fusion_dim") == 256
    assert candidate_expert.pop("fusion_dim") == 512
    assert candidate_expert == control_expert
    for key in ("backbone", "dataset", "action_contract", "training", "simulation"):
        assert candidate[key] == control[key]
    assert candidate["resources"] == control["resources"]


def test_experiment_rejects_episode_leakage(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw = copy.deepcopy(raw)
    raw["dataset"]["split"]["validation"][0] = raw["dataset"]["split"]["train"][0]
    path = tmp_path / "leaky.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="Episode leakage"):
        load_experiment_config(path, REPOSITORY_ROOT)


def test_experiment_rejects_non_base_model_identity(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw = copy.deepcopy(raw)
    raw["backbone"]["identifier"] = "Qwen/Qwen3.5-0.8B"
    path = tmp_path / "non-base.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="explicit Qwen3.5-0.8B-Base"):
        load_experiment_config(path, REPOSITORY_ROOT)


def test_experiment_rejects_instruct_control_claiming_m2_eligibility(tmp_path: Path) -> None:
    raw = yaml.safe_load(INSTRUCT_CONTROL_CONFIG_PATH.read_text(encoding="utf-8"))
    raw = copy.deepcopy(raw)
    raw["m2_completion_eligible"] = True
    path = tmp_path / "instruct-m2-bypass.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="M2-ineligible control"):
        load_experiment_config(path, REPOSITORY_ROOT)


def test_stable_hash_ignores_mapping_order() -> None:
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


def test_frozen_artifact_recipe_rejects_processor_drift() -> None:
    experiment = load_experiment_config(CONFIG_PATH, REPOSITORY_ROOT)
    artifact_config = frozen_artifact_recipe(experiment)

    validate_frozen_artifact_recipe(experiment, artifact_config)
    artifact_config = copy.deepcopy(artifact_config)
    artifact_config["processor"]["prompt"] = "A different feature recipe"

    with pytest.raises(ValueError, match="differs at processor"):
        validate_frozen_artifact_recipe(experiment, artifact_config)


def test_git_reference_preserves_feature_branch_prefix() -> None:
    assert (
        _branch_from_git_reference("refs/heads/codex/m2-qwen08b-frozen-001")
        == "codex/m2-qwen08b-frozen-001"
    )


def test_experiment_rejects_nonpositive_checkpoint_interval(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw = copy.deepcopy(raw)
    raw["training"]["checkpoint_every_epochs"] = 0
    path = tmp_path / "invalid-checkpoint-interval.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint_every_epochs must be positive"):
        load_experiment_config(path, REPOSITORY_ROOT)


@pytest.mark.parametrize("weight", (-0.1, float("inf"), float("nan")))
def test_experiment_rejects_unsafe_first_action_loss_weight(
    tmp_path: Path, weight: float
) -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw = copy.deepcopy(raw)
    raw["training"]["first_action_loss_weight"] = weight
    path = tmp_path / "invalid-first-action-weight.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="first_action_loss_weight"):
        load_experiment_config(path, REPOSITORY_ROOT)


@pytest.mark.parametrize("experiment_id", ("../escape", "nested/name", r"nested\name"))
def test_experiment_rejects_path_escaping_identifier(
    tmp_path: Path,
    experiment_id: str,
) -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw = copy.deepcopy(raw)
    raw["experiment_id"] = experiment_id
    path = tmp_path / "unsafe-experiment-id.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="path-safe token"):
        load_experiment_config(path, REPOSITORY_ROOT)
