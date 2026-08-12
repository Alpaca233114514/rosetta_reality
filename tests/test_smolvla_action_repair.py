from pathlib import Path

import pytest
import torch

from rosetta_reality.eval.diagnostics import action_dimension_diagnostics
from rosetta_reality.experiment import file_sha256
from rosetta_reality.sim import load_action_contract
from rosetta_reality.vla import (
    load_smolvla_action_space,
    load_smolvla_experiment,
)
from rosetta_reality.vla.checkpoint_memory import install_checkpoint_memory_trim
from rosetta_reality.vla.fixed_samples import (
    load_fixed_frame_protocol,
    resolve_fixed_dataset_indices,
)
from rosetta_reality.vla.processor import (
    BOUNDED_SINE_ACTION_ADAPTER,
    PI_ALOHA_POSTPROCESSOR_REGISTRY_NAME,
    PI_ALOHA_PREPROCESSOR_REGISTRY_NAME,
    REGISTRY_NAME,
    ActionContractProjectionProcessorStep,
    bounded_sine_action_to_standard,
    ensure_action_contract_projection,
    ensure_smolvla_action_boundary,
    pi_aloha_action_to_standard,
    standard_aloha_action_to_bounded_sine,
    standard_aloha_action_to_pi,
    standard_aloha_state_to_pi,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REPAIR_CONFIG = (
    REPOSITORY_ROOT
    / "configs/vla/smolvla_450m_aloha_insertion_action_repair_001.yaml"
)
REPAIR_SMOKE_CONFIG = (
    REPOSITORY_ROOT
    / "configs/vla/smolvla_450m_aloha_insertion_action_repair_smoke_001.yaml"
)
FIXED_OVERFIT_CONFIG = (
    REPOSITORY_ROOT
    / "configs/vla/smolvla_450m_aloha_insertion_action_repair_fixed_overfit_002.yaml"
)
BOUNDED_GRIPPER_CONFIG = (
    REPOSITORY_ROOT
    / "configs/vla/smolvla_450m_aloha_insertion_action_repair_bounded_gripper_003.yaml"
)
CONTRACT_PATH = REPOSITORY_ROOT / "configs/sim/aloha_insertion_smolvla.yaml"


def test_repair_overlay_pins_history_and_explicit_action_space() -> None:
    experiment = load_smolvla_experiment(REPAIR_CONFIG, REPOSITORY_ROOT)
    action_space = load_smolvla_action_space(experiment, require_explicit=True)

    assert experiment["experiment_inheritance"] == {
        "config": "configs/vla/smolvla_450m_aloha_insertion.yaml",
        "sha256": file_sha256(
            REPOSITORY_ROOT / "configs/vla/smolvla_450m_aloha_insertion.yaml"
        ),
    }
    assert experiment["status"] == "preregistered_diagnostics_only"
    assert experiment["repair_protocol"]["optimizer_authorized"] is False
    assert action_space.adapt_to_pi_aloha is False
    assert action_space.target_projection == "action_contract_clip"
    assert action_space.target_projection_stage == "before_normalization"
    assert action_space.representation_adapter == "rosetta_pi_aloha"
    assert action_space.model_internal_space == "normalized_pi_aloha"


def test_repair_smoke_overlay_authorizes_only_bounded_fresh_phases() -> None:
    experiment = load_smolvla_experiment(REPAIR_SMOKE_CONFIG, REPOSITORY_ROOT)
    action_space = load_smolvla_action_space(experiment, require_explicit=True)
    protocol = experiment["repair_protocol"]

    assert experiment["status"] == "preregistered_action_repair_smoke_and_overfit"
    assert protocol["optimizer_authorized"] is True
    assert protocol["authorized_phases"] == ["smoke", "overfit"]
    assert protocol["historical_checkpoints_are_initialization"] is False
    assert protocol["hidden_test_loaded"] is False
    assert experiment["dataset"]["test_episodes"] == [31, 6, 1, 24, 5]
    assert action_space.adapt_to_pi_aloha is False
    assert action_space.target_projection == "action_contract_clip"


def test_fixed_overfit_registers_exact_train_only_frames() -> None:
    experiment = load_smolvla_experiment(FIXED_OVERFIT_CONFIG, REPOSITORY_ROOT)
    protocol = load_fixed_frame_protocol(experiment, "overfit_resume")

    assert experiment["experiment_id"].endswith("fixed-overfit-002")
    assert protocol.episode == 49
    assert protocol.frame_indices == (0, 50, 100, 190, 220, 250, 400, 450)
    assert protocol.sampler == "deterministic_epoch_permutation"
    assert protocol.hidden_test_loaded is False
    assert protocol.episode not in experiment["dataset"]["test_episodes"]


def test_fixed_frames_resolve_through_episode_filtered_view() -> None:
    experiment = load_smolvla_experiment(FIXED_OVERFIT_CONFIG, REPOSITORY_ROOT)
    protocol = load_fixed_frame_protocol(experiment, "smoke")
    starts = [episode * 500 for episode in range(50)]
    stops = [(episode + 1) * 500 for episode in range(50)]
    mapping = {24500 + offset: offset for offset in range(500)}

    resolved = resolve_fixed_dataset_indices(
        protocol,
        starts,
        stops,
        [49],
        mapping,
    )

    assert resolved == list(protocol.frame_indices)
    with pytest.raises(ValueError, match="exactly the fixed episode"):
        resolve_fixed_dataset_indices(protocol, starts, stops, [48, 49], mapping)


def test_bounded_gripper_action_space_is_explicit_and_fresh() -> None:
    experiment = load_smolvla_experiment(BOUNDED_GRIPPER_CONFIG, REPOSITORY_ROOT)
    action_space = load_smolvla_action_space(experiment, require_explicit=True)

    assert action_space.representation_adapter == BOUNDED_SINE_ACTION_ADAPTER
    assert (
        action_space.model_internal_space
        == "normalized_pi_aloha_arms_bounded_sine_grippers"
    )
    assert experiment["phases"]["overfit"]["steps"] == 200
    assert experiment["phases"]["overfit"]["save_freq"] == 50
    assert [
        experiment["phases"]["overfit"]["save_freq"] * quarter
        for quarter in range(1, 5)
    ] == [50, 100, 150, 200]
    assert experiment["repair_protocol"]["historical_checkpoints_are_initialization"] is False


def test_bounded_sine_gripper_round_trip_and_arbitrary_output_bounds() -> None:
    action = torch.linspace(-1.0, 1.0, 4 * 3 * 14, dtype=torch.float64).reshape(
        4, 3, 14
    )
    action[..., 6] = torch.linspace(0.0, 1.0, 12, dtype=torch.float64).reshape(4, 3)
    action[..., 13] = action[..., 6].flip(0)

    internal = standard_aloha_action_to_bounded_sine(action)
    decoded = bounded_sine_action_to_standard(internal)

    assert torch.allclose(decoded, action, atol=1e-12, rtol=0.0)
    arbitrary = internal.clone()
    arbitrary[..., 6] = torch.linspace(-20.0, 20.0, 12).reshape(4, 3)
    arbitrary[..., 13] = arbitrary[..., 6].flip(0)
    bounded = bounded_sine_action_to_standard(arbitrary)
    assert bool(((bounded[..., [6, 13]] >= 0) & (bounded[..., [6, 13]] <= 1)).all())


def test_checkpoint_memory_trim_wraps_resume_and_save_once(monkeypatch) -> None:
    events: list[str] = []

    class FakeTrainModule:
        @staticmethod
        def resume_after_prepare() -> str:
            events.append("resume")
            return "resumed"

        @staticmethod
        def save_checkpoint() -> str:
            events.append("save")
            return "saved"

    monkeypatch.setattr(
        "rosetta_reality.vla.checkpoint_memory.release_checkpoint_headroom",
        lambda: events.append("trim"),
    )
    module = FakeTrainModule()
    install_checkpoint_memory_trim(module)
    install_checkpoint_memory_trim(module)

    assert module.resume_after_prepare() == "resumed"
    assert module.save_checkpoint() == "saved"
    assert events == ["resume", "trim", "trim", "save"]


def test_projection_clips_tolerated_gripper_overshoot_and_rejects_more() -> None:
    contract = load_action_contract(CONTRACT_PATH)
    step = ActionContractProjectionProcessorStep.from_contract(
        contract,
        action_contract_sha256=file_sha256(CONTRACT_PATH),
    )
    action = torch.zeros(2, contract.chunk_length, contract.dimension)
    action[..., 6] = 1.1
    action[..., 13] = -0.05

    projected = step({"action": action})["action"]

    assert torch.all(projected[..., 6] == 1.0)
    assert torch.all(projected[..., 13] == 0.0)
    invalid = action.clone()
    invalid[..., 13] = -0.21
    with pytest.raises(ValueError, match="overshoot tolerance"):
        step({"action": invalid})


def test_projection_is_serializable_and_inserted_before_normalization() -> None:
    contract = load_action_contract(CONTRACT_PATH)

    class FakeNormalizer:
        _registry_name = "normalizer_processor"

    class FakePreprocessor:
        steps = [object(), FakeNormalizer()]

    preprocessor = FakePreprocessor()
    ensure_action_contract_projection(
        preprocessor,
        contract,
        action_contract_sha256=file_sha256(CONTRACT_PATH),
    )
    assert getattr(preprocessor.steps[1].__class__, "_registry_name") == REGISTRY_NAME
    assert getattr(preprocessor.steps[2].__class__, "_registry_name") == (
        "normalizer_processor"
    )
    reloaded = ActionContractProjectionProcessorStep(**preprocessor.steps[1].get_config())
    assert reloaded.get_config() == preprocessor.steps[1].get_config()
    ensure_action_contract_projection(
        preprocessor,
        contract,
        action_contract_sha256=file_sha256(CONTRACT_PATH),
    )
    assert len(preprocessor.steps) == 3


def test_pi_aloha_action_adapter_round_trip_uses_independent_tensors() -> None:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    shell = object.__new__(SmolVLAPolicy)
    action = torch.linspace(-2.0, 2.0, 2 * 3 * 14, dtype=torch.float64).reshape(2, 3, 14)
    original = action.clone()

    internal = standard_aloha_action_to_pi(action)
    decoded = pi_aloha_action_to_standard(internal)
    upstream = SmolVLAPolicy._pi_aloha_encode_actions_inv(shell, action.clone())

    assert torch.equal(action, original)
    assert torch.allclose(internal, upstream, atol=1e-12, rtol=0.0)
    assert torch.allclose(decoded, original, atol=1e-12, rtol=0.0)


def test_pi_aloha_state_adapter_supports_observation_history() -> None:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    shell = object.__new__(SmolVLAPolicy)
    state = torch.linspace(0.05, 0.95, 2 * 3 * 14, dtype=torch.float64).reshape(
        2, 3, 14
    )

    internal = standard_aloha_state_to_pi(state)
    upstream = SmolVLAPolicy._pi_aloha_decode_state(
        shell, state.reshape(-1, 14).clone()
    ).reshape_as(state)

    assert internal.shape == (2, 3, 14)
    assert torch.allclose(internal, upstream, atol=1e-12, rtol=0.0)


def test_complete_action_boundary_orders_pre_and_post_processors() -> None:
    contract = load_action_contract(CONTRACT_PATH)
    experiment = load_smolvla_experiment(REPAIR_CONFIG, REPOSITORY_ROOT)
    action_space = load_smolvla_action_space(experiment, require_explicit=True)

    class FakeNormalizer:
        _registry_name = "normalizer_processor"

    class FakeUnnormalizer:
        _registry_name = "unnormalizer_processor"

    class FakePipeline:
        def __init__(self, steps: list[object]) -> None:
            self.steps = steps

    preprocessor = FakePipeline([object(), FakeNormalizer()])
    postprocessor = FakePipeline([FakeUnnormalizer(), object()])
    ensure_smolvla_action_boundary(
        preprocessor,
        postprocessor,
        contract,
        action_space,
        action_contract_sha256=file_sha256(CONTRACT_PATH),
        upstream_revision=experiment["upstream"]["revision"],
    )
    pre_names = [getattr(step.__class__, "_registry_name", None) for step in preprocessor.steps]
    post_names = [
        getattr(step.__class__, "_registry_name", None) for step in postprocessor.steps
    ]

    assert pre_names[1:4] == [
        REGISTRY_NAME,
        PI_ALOHA_PREPROCESSOR_REGISTRY_NAME,
        "normalizer_processor",
    ]
    assert post_names[:2] == [
        "unnormalizer_processor",
        PI_ALOHA_POSTPROCESSOR_REGISTRY_NAME,
    ]
    ensure_smolvla_action_boundary(
        preprocessor,
        postprocessor,
        contract,
        action_space,
        action_contract_sha256=file_sha256(CONTRACT_PATH),
        upstream_revision=experiment["upstream"]["revision"],
    )
    assert len(preprocessor.steps) == 4
    assert len(postprocessor.steps) == 3


def test_per_dimension_diagnostics_expose_right_gripper_failure() -> None:
    contract = load_action_contract(CONTRACT_PATH)
    target = torch.zeros(2, 3, contract.dimension)
    predicted = target.clone()
    predicted[..., 13] = -0.2

    result = action_dimension_diagnostics(
        predicted,
        target,
        contract.lower_bounds,
        contract.upper_bounds,
        contract.dimension_names,
    )

    right = result["dimensions"]["right_gripper"]
    assert right["mae"] == pytest.approx(0.2)
    assert right["prediction_strict_violation_rate"] == 1.0
    assert right["predicted_below_minimum_rate"] == 1.0
    assert result["groups"]["right_gripper"]["mae"] == pytest.approx(0.2)
    assert result["groups"]["left_gripper"]["mae"] == 0.0
