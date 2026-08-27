"""Version-2 training feature registry tests (no model weights, no data)."""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import pytest
import torch

import rosetta_reality.vla.training.features as features_module
from rosetta_reality.experiment import file_sha256
from rosetta_reality.vla import load_smolvla_action_space, load_smolvla_experiment
from rosetta_reality.vla.training import TrainingContext
from rosetta_reality.vla.training.features import (
    FEATURE_FACTORIES,
    FeatureStack,
    TrackioLoggingFeature,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BOUNDED_GRIPPER_CONFIG = (
    REPOSITORY_ROOT
    / "configs/vla/smolvla_450m_aloha_insertion_action_repair_bounded_gripper_003.yaml"
)
CONTRACT_PATH = REPOSITORY_ROOT / "configs/sim/aloha_insertion_smolvla.yaml"


class FakeTrainModule:
    """Minimal stand-in for ``lerobot.scripts.lerobot_train``."""

    def __init__(self) -> None:
        self.make_train_eval_datasets = staticmethod(self._make_datasets)
        self.make_pre_post_processors = staticmethod(self._make_processors)
        self.resume_after_prepare = staticmethod(self._resume)
        self.save_checkpoint = staticmethod(self._save)
        self.WandBLogger = object

    @staticmethod
    def _make_datasets(cfg: Any) -> tuple[Any, Any]:
        dataset = types.SimpleNamespace(meta=types.SimpleNamespace(stats={}))
        eval_dataset = types.SimpleNamespace(meta=types.SimpleNamespace(stats={}))
        return dataset, eval_dataset

    @staticmethod
    def _make_processors(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        class FakeNormalizer:
            _registry_name = "normalizer_processor"

        class FakeUnnormalizer:
            _registry_name = "unnormalizer_processor"

        class FakePipeline:
            def __init__(self, steps: list[object]) -> None:
                self.steps = steps

        return (
            FakePipeline([object(), FakeNormalizer()]),
            FakePipeline([FakeUnnormalizer(), object()]),
        )

    @staticmethod
    def _resume() -> str:
        return "resumed"

    @staticmethod
    def _save() -> str:
        return "saved"


def _context(
    tmp_path: Path,
    *,
    plan: dict[str, Any] | None = None,
    experiment: dict[str, Any] | None = None,
    normalization_report: Path | None = None,
    device: str = "xpu",
) -> TrainingContext:
    if experiment is None:
        experiment = load_smolvla_experiment(BOUNDED_GRIPPER_CONFIG, REPOSITORY_ROOT)
    action_space = load_smolvla_action_space(experiment, require_explicit=True)
    plan_path = tmp_path / "plan.yaml"
    if not plan_path.exists():
        plan_path.write_text(json.dumps(plan or {}), encoding="utf-8")
    return TrainingContext(
        plan=plan or {},
        experiment=experiment,
        action_space=action_space,
        plan_path=plan_path,
        experiment_path=BOUNDED_GRIPPER_CONFIG,
        contract_path=CONTRACT_PATH,
        normalization_report=normalization_report or tmp_path / "normalization.json",
        phase="formal",
        device=device,
        run_name="unit-run",
    )


def test_registry_covers_the_declared_feature_set() -> None:
    assert set(FEATURE_FACTORIES) == {
        "trackio_logging",
        "train_only_statistics",
        "masked_camera_skip",
        "action_boundary_projection",
        "fixed_frame_sampler",
        "horizon_weight_profile",
        "state_robustness_jitter",
        "checkpoint_memory_trim",
    }


def test_feature_stack_respects_declaration_order_and_rejects_abuse() -> None:
    plan = {
        "features": [
            {"name": "checkpoint_memory_trim"},
            {"name": "train_only_statistics"},
        ]
    }
    stack = FeatureStack.from_plan(plan)
    assert [feature.name for feature in stack.features] == [
        "checkpoint_memory_trim",
        "train_only_statistics",
    ]
    duplicated = {
        "features": [
            {"name": "checkpoint_memory_trim"},
            {"name": "checkpoint_memory_trim"},
        ]
    }
    with pytest.raises(ValueError, match="declared twice"):
        FeatureStack.from_plan(duplicated)
    with pytest.raises(ValueError, match="Unknown training feature"):
        FeatureStack.from_plan({"features": [{"name": "nope"}]})
    with pytest.raises(ValueError, match="at least one feature"):
        FeatureStack.from_plan({"features": []})


def test_train_only_statistics_wraps_and_restores_the_dataset_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_path = tmp_path / "normalization.json"
    report_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "stage": "smolvla_train_only_normalization",
                "source_split": "train",
                "validation_episodes_loaded": False,
                "hidden_test_loaded": False,
                "train_episodes": [49],
                "effective_stats": {
                    "action": {"min": [0.0], "max": [1.0], "count": [10]},
                    "observation.state": {"mean": [0.5], "std": [0.1]},
                },
            }
        ),
        encoding="utf-8"
    )
    module = FakeTrainModule()
    monkeypatch.setattr(features_module, "_lerobot_train_module", lambda: module)
    context = _context(tmp_path, normalization_report=report_path)
    feature = FEATURE_FACTORIES["train_only_statistics"]({})

    feature.install(context)
    with pytest.raises(RuntimeError, match="already installed"):
        feature.install(context)
    cfg = types.SimpleNamespace(
        dataset=types.SimpleNamespace(episodes=[49])
    )
    dataset, eval_dataset = module.make_train_eval_datasets(cfg)
    expected = torch.tensor([10], dtype=torch.int64)
    assert torch.equal(dataset.meta.stats["action"]["count"], expected)
    assert "observation.state" in eval_dataset.meta.stats
    out_of_scope = types.SimpleNamespace(
        dataset=types.SimpleNamespace(episodes=[49, 4])
    )
    with pytest.raises(ValueError, match="outside the train-only normalization scope"):
        module.make_train_eval_datasets(out_of_scope)
    feature.restore(context)
    dataset, _ = module.make_train_eval_datasets(cfg)
    assert dataset.meta.stats == {}


def test_checkpoint_memory_trim_fails_closed_on_double_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    module = FakeTrainModule()
    monkeypatch.setattr(features_module, "_lerobot_train_module", lambda: module)
    monkeypatch.setattr(
        features_module,
        "release_checkpoint_headroom",
        lambda device=None: events.append(device),
    )
    context = _context(tmp_path, device="cuda")
    feature = FEATURE_FACTORIES["checkpoint_memory_trim"]({})

    feature.install(context)
    with pytest.raises(RuntimeError, match="already installed"):
        feature.install(context)
    assert module.resume_after_prepare() == "resumed"
    assert module.save_checkpoint() == "saved"
    assert events == ["cuda", "cuda"]
    feature.restore(context)
    assert module.resume_after_prepare() == "resumed"
    assert events == ["cuda", "cuda"]


def test_action_boundary_projection_installs_the_registered_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rosetta_reality.vla.processor import REGISTRY_NAME

    module = FakeTrainModule()
    monkeypatch.setattr(features_module, "_lerobot_train_module", lambda: module)
    context = _context(tmp_path)
    feature = FEATURE_FACTORIES["action_boundary_projection"]({})

    feature.install(context)
    preprocessor, _ = module.make_pre_post_processors()
    installed = [
        getattr(step.__class__, "_registry_name", None) for step in preprocessor.steps
    ]
    assert REGISTRY_NAME in installed
    with pytest.raises(RuntimeError, match="already installed"):
        feature.install(context)
    feature.restore(context)
    preprocessor, _ = module.make_pre_post_processors()
    installed = [
        getattr(step.__class__, "_registry_name", None) for step in preprocessor.steps
    ]
    assert REGISTRY_NAME not in installed


def _fake_modeling_module(tmp_path: Path, filename: str) -> types.ModuleType:
    source = tmp_path / filename
    source.write_text("# fake pinned upstream implementation\n", encoding="utf-8")

    class VLAFlowMatching:
        def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise, time):
            del images, img_masks, lang_tokens, lang_masks, state, noise, time
            return actions.square()

    class SmolVLAPolicy:
        training = True

        def __init__(self) -> None:
            self.model = VLAFlowMatching()

        def forward(self, batch, noise=None, time=None, reduction="mean"):
            losses = self.model.forward(
                None, None, None, None, None, batch["action"], noise, time
            )
            if reduction == "none":
                return losses.mean(dim=(1, 2)), {"loss": 0.0}
            return losses.mean(), {"loss": 0.0}

    module = types.ModuleType(f"fake_{filename}")
    module.__file__ = str(source)
    module.VLAFlowMatching = VLAFlowMatching
    module.SmolVLAPolicy = SmolVLAPolicy
    return module


def test_horizon_feature_delegates_to_the_pinned_loss_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rosetta_reality.vla.horizon_loss as horizon_loss_module
    from rosetta_reality.vla.horizon_loss import (
        NORMALIZATION_SELECTED_VALID_MEAN,
        PROFILE_FIRST_ACTION_ONLY,
    )

    module = _fake_modeling_module(tmp_path, "horizon_modeling.py")
    monkeypatch.setattr(features_module, "_load_modeling_module", lambda: module)
    fake_sha = file_sha256(Path(module.__file__))
    monkeypatch.setattr(horizon_loss_module, "UPSTREAM_IMPLEMENTATION_SHA256", fake_sha)
    original_install = horizon_loss_module.install_horizon_weight_profile
    monkeypatch.setattr(
        horizon_loss_module,
        "install_horizon_weight_profile",
        lambda modeling_module, profile, **kwargs: original_install(
            modeling_module, profile, upstream_sha256=fake_sha, **kwargs
        ),
    )
    plan = {
        "loss_contract": {
            "profile": PROFILE_FIRST_ACTION_ONLY,
            "chunk_size": 3,
            "normalization": NORMALIZATION_SELECTED_VALID_MEAN,
            "upstream_implementation_sha256": fake_sha,
        }
    }
    experiment = load_smolvla_experiment(BOUNDED_GRIPPER_CONFIG, REPOSITORY_ROOT)
    experiment["model"]["policy"]["chunk_size"] = 3
    context = _context(tmp_path, plan=plan, experiment=experiment)
    feature = FEATURE_FACTORIES["horizon_weight_profile"]({})

    feature.install(context)
    actions = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    losses = module.VLAFlowMatching().forward(None, None, None, None, None, actions, None, None)
    assert torch.equal(losses[0, 0], torch.tensor([1.0, 4.0]))
    assert bool((losses[0, 1:] == 0.0).all())
    feature.restore(context)
    losses = module.VLAFlowMatching().forward(None, None, None, None, None, actions, None, None)
    assert torch.equal(losses[0, 2], torch.tensor([25.0, 36.0]))


def test_state_robustness_feature_installs_and_restores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rosetta_reality.vla.state_robustness as state_robustness_module

    module = _fake_modeling_module(tmp_path, "state_modeling.py")
    monkeypatch.setattr(features_module, "_load_modeling_module", lambda: module)
    fake_sha = file_sha256(Path(module.__file__))
    monkeypatch.setattr(
        state_robustness_module, "UPSTREAM_IMPLEMENTATION_SHA256", fake_sha
    )
    original_install = state_robustness_module.install_state_robustness_profile
    monkeypatch.setattr(
        state_robustness_module,
        "install_state_robustness_profile",
        lambda modeling_module, profile, **kwargs: original_install(
            modeling_module, profile, upstream_sha256=fake_sha, **kwargs
        ),
    )
    plan = {
        "state_robustness_contract": {
            "profile": "normalized_gaussian_state_jitter",
            "upstream_implementation_sha256": fake_sha,
            "input_space": "train_normalized_observation_state",
            "normalized_standard_deviation": 0.05,
            "training_only": True,
            "target_semantics": "unchanged_absolute_expert_action",
        }
    }
    context = _context(tmp_path, plan=plan)
    feature = FEATURE_FACTORIES["state_robustness_jitter"]({})

    feature.install(context)
    assert getattr(
        module.SmolVLAPolicy.forward, "_rosetta_state_robustness_profile", None
    ) == "normalized_gaussian_state_jitter"
    feature.restore(context)
    assert getattr(
        module.SmolVLAPolicy.forward, "_rosetta_state_robustness_profile", None
    ) is None


def test_fixed_frame_sampler_restricts_to_registered_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rosetta_reality.vla.fixed_samples import load_fixed_frame_protocol

    module = FakeTrainModule()

    class FakeEpisodeAwareSampler:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("The fixed-frame sampler must replace this constructor.")

    module.EpisodeAwareSampler = FakeEpisodeAwareSampler
    monkeypatch.setattr(features_module, "_lerobot_train_module", lambda: module)
    experiment = load_smolvla_experiment(
        REPOSITORY_ROOT
        / "configs/vla/smolvla_450m_aloha_insertion_action_repair_fixed_overfit_002.yaml",
        REPOSITORY_ROOT,
    )
    context = _context(tmp_path, experiment=experiment)
    with pytest.raises(ValueError, match="phase"):
        FEATURE_FACTORIES["fixed_frame_sampler"]({"phase": "formal"})
    feature = FEATURE_FACTORIES["fixed_frame_sampler"]({"phase": "smoke"})

    feature.install(context)
    starts = [episode * 500 for episode in range(50)]
    stops = [(episode + 1) * 500 for episode in range(50)]
    mapping = {24500 + offset: offset for offset in range(500)}
    sampler = module.EpisodeAwareSampler(starts, stops, [49], 0, 0, False, 0, mapping)
    protocol = load_fixed_frame_protocol(experiment, "smoke")
    assert sampler.indices == list(protocol.frame_indices)
    feature.restore(context)
    assert module.EpisodeAwareSampler is FakeEpisodeAwareSampler


def test_trackio_logging_swaps_the_logger_class_without_module_patching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = FakeTrainModule()
    monkeypatch.setattr(features_module, "_lerobot_train_module", lambda: module)
    plan = {
        "loss_contract": {
            "profile": "first_action_only",
            "normalization": "mean_over_selected_valid_entries",
        }
    }
    context = _context(tmp_path, plan=plan)
    original_logger = module.WandBLogger
    with pytest.raises(ValueError, match="logger must be standard or accelerator"):
        TrackioLoggingFeature({"logger": "wandb"})
    feature = TrackioLoggingFeature({"logger": "standard"})

    feature.install(context)
    assert module.WandBLogger.__name__ == "PlanBoundTrackioLogger"
    assert module.WandBLogger is not original_logger
    extension = feature._public_extension(context)
    assert extension["temporal_loss_profile"] == "first_action_only"
    assert extension["bounded_gripper_decoder"] is True
    assert extension["training_harness"] == "v2"
    feature.restore(context)
    assert module.WandBLogger is original_logger


def test_install_failure_rolls_back_already_installed_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = FakeTrainModule()
    monkeypatch.setattr(features_module, "_lerobot_train_module", lambda: module)
    context = _context(
        tmp_path, normalization_report=tmp_path / "missing-normalization.json"
    )
    stack = FeatureStack.from_plan(
        {
            "features": [
                {"name": "checkpoint_memory_trim"},
                {"name": "train_only_statistics"},
            ]
        }
    )

    with pytest.raises(FileNotFoundError):
        stack.install_all(context)
    assert stack.installed == []
    assert getattr(module, "_rosetta_v2_feature_checkpoint_memory_trim_installed", False) is (
        False
    )
    assert module.save_checkpoint() == "saved"
