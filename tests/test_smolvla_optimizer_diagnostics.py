"""Version-2 optimizer/checkpoint instrumentation feature tests (no weights).

Covers the two audit-driven instrumentation features added to the v2 registry:
``gradient_clip_diagnostics`` (finding O2: pre/post-clip global and per-module
gradient norms) and ``checkpoint_metric_snapshot`` (finding T9: the exact
same-step metric row inside every checkpoint directory).  Everything runs on
tiny CPU tensors and fake trainer modules; no model weights or datasets.
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import pytest
import torch

import rosetta_reality.vla.training.features as features_module
from rosetta_reality.vla.training import TrainingContext
from rosetta_reality.vla.training.features import (
    FEATURE_FACTORIES,
    FeatureStack,
)
from rosetta_reality.vla.training.plan import validate_plan_structure


class FakeTrainModule:
    """Minimal stand-in exposing the surfaces the instrumentation wraps."""

    def __init__(self) -> None:
        self.update_policy = self._update
        self.resume_after_prepare = self._resume
        self.save_checkpoint = self._save
        self.save_calls: list[dict[str, Any]] = []

    @staticmethod
    def _update(tracker: Any, *args: Any, **kwargs: Any) -> Any:
        return tracker

    @staticmethod
    def _resume() -> str:
        return "resumed"

    def _save(self, *args: Any, **kwargs: Any) -> str:
        self.save_calls.append(dict(kwargs))
        return "saved"


class FakeTracker:
    def __init__(self, loss: float, grad_norm: float, lr: float, update_s: float):
        self.loss = loss
        self.grad_norm = grad_norm
        self.lr = lr
        self.update_s = update_s


class FakeAverageMeter:
    """Mimic the pinned ``lerobot.utils.logging_utils.AverageMeter`` surface.

    The pinned trainer assigns ``train_metrics.<name>`` and
    ``MetricsTracker.__setattr__`` routes that into ``AverageMeter.update``, so
    the value a step leaves behind lives in ``.val`` — not the meter object and
    not a plain float.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count


class PinnedStyleTracker:
    """Tracker whose metrics are ``AverageMeter`` instances, as upstream."""

    def __init__(self, loss: float, grad_norm: float, lr: float, update_s: float):
        self.loss = FakeAverageMeter("loss")
        self.grad_norm = FakeAverageMeter("grdn")
        self.lr = FakeAverageMeter("lr")
        self.update_s = FakeAverageMeter("updt_s")
        for name, value in (
            ("loss", loss),
            ("grad_norm", grad_norm),
            ("lr", lr),
            ("update_s", update_s),
        ):
            getattr(self, name).update(value)


def _context(tmp_path: Path, plan: dict[str, Any]) -> TrainingContext:
    return TrainingContext(
        plan=plan,
        experiment={"experiment_id": "test-experiment"},
        action_space=types.SimpleNamespace(
            representation_adapter="pi_aloha",
            target_projection="action_contract_clip",
        ),
        plan_path=tmp_path / "plan.yaml",
        experiment_path=tmp_path / "experiment.yaml",
        contract_path=tmp_path / "contract.yaml",
        normalization_report=tmp_path / "normalization.json",
        phase="formal",
        device="cpu",
        run_name="test-run",
    )


def _plan(*feature_names: str) -> dict[str, Any]:
    return {
        "features": [{"name": name} for name in feature_names],
        "training": {
            "optimizer": {"type": "adamw", "grad_clip_norm": 10.0},
        },
    }


def _full_plan() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "role": "vla",
        "status": "preregistered",
        "plan_id": "test-plan-1",
        "run_name": "test-run-1",
        "parent_experiment": {
            "config": "c",
            "sha256": "0" * 64,
            "experiment_id": "e",
        },
        "training": {
            "episodes": [1],
            "batch_size": 1,
            "steps": 4,
            "save_freq": 1,
            "log_freq": 1,
            "checkpoint_steps": [1, 2, 3, 4],
            "eval_split": 0.0,
            "validation_gradients": False,
            "hidden_test_loaded": False,
            "policy": {
                "empty_cameras": 0,
                "compile_model": False,
                "skip_fully_masked_camera_encoding": False,
            },
            "optimizer": {
                "type": "adamw",
                "lr": 1.0e-4,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 1e-10,
                "grad_clip_norm": 0.0,
            },
            "scheduler": {
                "type": "cosine_decay_with_warmup",
                "num_warmup_steps": 1,
                "num_decay_steps": 4,
                "peak_lr": 1.0e-4,
                "decay_lr": 2.5e-6,
            },
        },
        "validation": {
            "episodes": [2],
            "frame_offsets": [0],
            "total_samples": 1,
            "hidden_test_loaded": False,
        },
        "resources": {
            "memory_limit": "m",
            "memory_swap_limit": "m",
            "mixed_precision": "bf16",
            "cpu_limit": 1,
            "checkpoint_memory_trim": False,
        },
        "features": [{"name": "gradient_clip_diagnostics"}],
        "prerequisites": {},
        "normalization": {
            "source_split": "train",
            "report": "a",
            "report_sha256": "0" * 64,
            "dataset_view_manifest": "b",
            "dataset_view_manifest_sha256": "0" * 64,
            "validation_episodes_loaded": False,
            "hidden_test_loaded": False,
        },
        "implementation_files": {"scripts/x.py": "0" * 64},
        "stop_conditions": ["nonfinite_loss_gradient_or_action"],
        "hidden_test_loaded": False,
    }


def test_new_features_are_registered() -> None:
    assert FEATURE_FACTORIES["gradient_clip_diagnostics"] is (
        features_module.GradientClipDiagnosticsFeature
    )
    assert FEATURE_FACTORIES["checkpoint_metric_snapshot"] is (
        features_module.CheckpointMetricSnapshotFeature
    )


def test_clip_diagnostics_require_positive_clip_norm_in_plan() -> None:
    plan = _full_plan()
    del plan["training"]["optimizer"]
    del plan["training"]["scheduler"]
    with pytest.raises(ValueError, match="positive"):
        validate_plan_structure(plan, known_features=FEATURE_FACTORIES)
    optimizer = {
        "type": "adamw",
        "lr": 1.0e-4,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 1e-10,
        "grad_clip_norm": 10.0,
    }
    scheduler = {
        "type": "cosine_decay_with_warmup",
        "num_warmup_steps": 1,
        "num_decay_steps": 4,
        "peak_lr": 1.0e-4,
        "decay_lr": 2.5e-6,
    }
    plan["training"]["optimizer"] = optimizer
    plan["training"]["scheduler"] = scheduler
    assert validate_plan_structure(plan, known_features=FEATURE_FACTORIES)


def test_clip_diagnostics_records_pre_and_post_norms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import accelerate

    fake = FakeTrainModule()
    monkeypatch.setattr(
        features_module, "_lerobot_train_module", lambda: fake, raising=True
    )

    def fake_clip(self: Any, parameters: Any, max_norm: float, **kwargs: Any) -> Any:
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    monkeypatch.setattr(
        accelerate.Accelerator, "clip_grad_norm_", fake_clip, raising=True
    )
    monkeypatch.setenv("ROSETTA_RUN_ROOT", str(tmp_path))
    feature = features_module.GradientClipDiagnosticsFeature({})
    context = _context(tmp_path, _plan("gradient_clip_diagnostics"))
    feature.install(context)
    try:
        module = torch.nn.Sequential(
            torch.nn.Linear(3, 2),
            torch.nn.Linear(2, 1),
        )
        module[0].weight.grad = torch.full_like(module[0].weight, 1.0)
        module[1].weight.grad = torch.full_like(module[1].weight, 2.0)
        fake.update_policy(FakeTracker(0.5, 1.0, 1e-4, 0.1), module)
        # The trainer passes a fresh parameters() generator per update; the
        # wrapper must not consume it before the original clip runs.
        result = accelerate.Accelerator.clip_grad_norm_(
            types.SimpleNamespace(), module.parameters(), 100.0
        )
        pre_global = float(
            sum(
                parameter.grad.detach().double().square().sum()
                for parameter in module.parameters()
                if parameter.grad is not None
            )
            ** 0.5
        )
        assert float(result) == pytest.approx(pre_global)
        # A second call below the current norm must actually clip: this
        # fails closed if the wrapper starves the original clip of inputs.
        clipped = accelerate.Accelerator.clip_grad_norm_(
            types.SimpleNamespace(), module.parameters(), 1.0
        )
        assert float(clipped) == pytest.approx(pre_global)
        clipped_global = float(
            sum(
                parameter.grad.detach().double().square().sum()
                for parameter in module.parameters()
                if parameter.grad is not None
            )
            ** 0.5
        )
        assert clipped_global == pytest.approx(1.0, rel=1e-4)
        record_path = (
            tmp_path / "test-experiment" / "diagnostics" / "gradient-clip-test-run.jsonl"
        )
        lines = record_path.read_text(encoding="utf-8").splitlines()
        records = json.loads(lines[0])
        assert records["update"] == 1
        assert records["pre_clip_l2"]["global"] == pytest.approx(pre_global)
        assert set(records["pre_clip_l2"]) == {"0", "1", "global"}
        assert records["post_clip_l2"]["global"] == pytest.approx(pre_global)
        clipped_record = json.loads(lines[1])
        assert clipped_record["update"] == 2
        assert clipped_record["pre_clip_l2"]["global"] == pytest.approx(pre_global)
        assert clipped_record["post_clip_l2"]["global"] == pytest.approx(
            1.0, rel=1e-4
        )
    finally:
        feature.restore(context)
    assert getattr(accelerate.Accelerator, feature._CLIP_MARKER, False) is False
    with pytest.raises(FileExistsError):
        feature.install(context)


def test_clip_diagnostics_fail_closed_on_non_finite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import accelerate

    fake = FakeTrainModule()
    monkeypatch.setattr(
        features_module, "_lerobot_train_module", lambda: fake, raising=True
    )
    monkeypatch.setattr(
        accelerate.Accelerator,
        "clip_grad_norm_",
        lambda self, parameters, max_norm, **kwargs: torch.tensor(0.0),
        raising=True,
    )
    monkeypatch.setenv("ROSETTA_RUN_ROOT", str(tmp_path))
    feature = features_module.GradientClipDiagnosticsFeature({})
    context = _context(tmp_path, _plan("gradient_clip_diagnostics"))
    feature.install(context)
    try:
        parameter = torch.nn.Parameter(torch.zeros(1))
        parameter.grad = torch.tensor([float("nan")])
        with pytest.raises(FloatingPointError, match="Non-finite"):
            accelerate.Accelerator.clip_grad_norm_(
                types.SimpleNamespace(), [parameter], 5.0
            )
    finally:
        feature.restore(context)


def test_checkpoint_snapshot_writes_exact_step_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeTrainModule()
    monkeypatch.setattr(
        features_module, "_lerobot_train_module", lambda: fake, raising=True
    )
    feature = features_module.CheckpointMetricSnapshotFeature({})
    context = _context(tmp_path, _plan("checkpoint_metric_snapshot"))
    feature.install(context)
    try:
        # Pinned-tracker shape: AverageMeter fields whose step value is .val.
        fake.update_policy(PinnedStyleTracker(0.5, 1.0, 1e-4, 0.1))
        fake.update_policy(PinnedStyleTracker(0.4, 0.9, 9e-5, 0.2))
        checkpoint_dir = tmp_path / "checkpoints" / "000002"
        checkpoint_dir.mkdir(parents=True)
        fake.save_checkpoint(checkpoint_dir=checkpoint_dir, step=2)
        payload = json.loads(
            (checkpoint_dir / "rosetta_checkpoint_metrics.json").read_text(
                encoding="utf-8"
            )
        )
        assert payload["step"] == 2
        assert payload["metrics"] == {
            "loss": 0.4,
            "grad_norm": 0.9,
            "lr": 9e-5,
            "update_s": 0.2,
        }
        # Plain-float trackers stay supported (defensive scalar branch).
        fake.update_policy(FakeTracker(0.3, 0.8, 8e-5, 0.3))
        checkpoint_dir3 = tmp_path / "checkpoints" / "000003"
        checkpoint_dir3.mkdir(parents=True)
        fake.save_checkpoint(checkpoint_dir=checkpoint_dir3, step=3)
        payload3 = json.loads(
            (checkpoint_dir3 / "rosetta_checkpoint_metrics.json").read_text(
                encoding="utf-8"
            )
        )
        assert payload3["metrics"] == {
            "loss": 0.3,
            "grad_norm": 0.8,
            "lr": 8e-5,
            "update_s": 0.3,
        }
        assert payload["memory"]["api"] in {"torch.cuda", "torch.xpu", "unavailable"}
        with pytest.raises(ValueError, match="finding T9"):
            fake.save_checkpoint(checkpoint_dir=checkpoint_dir, step=4)
    finally:
        feature.restore(context)
    assert fake.save_calls == [
        {"checkpoint_dir": checkpoint_dir, "step": 2},
        {"checkpoint_dir": checkpoint_dir3, "step": 3},
    ]


def test_snapshot_composes_with_checkpoint_memory_trim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeTrainModule()
    monkeypatch.setattr(
        features_module, "_lerobot_train_module", lambda: fake, raising=True
    )
    monkeypatch.setattr(
        features_module, "release_checkpoint_headroom", lambda device=None: None
    )
    plan = _plan("checkpoint_memory_trim", "checkpoint_metric_snapshot")
    plan["resources"] = {"checkpoint_memory_trim": True}
    stack = FeatureStack.from_plan(plan)
    context = _context(tmp_path, plan)
    installed = stack.install_all(context)
    try:
        assert installed == ["checkpoint_memory_trim", "checkpoint_metric_snapshot"]
        fake.update_policy(FakeTracker(0.5, 1.0, 1e-4, 0.1))
        checkpoint_dir = tmp_path / "checkpoints" / "000001"
        checkpoint_dir.mkdir(parents=True)
        fake.save_checkpoint(checkpoint_dir=checkpoint_dir, step=1)
        assert (checkpoint_dir / "rosetta_checkpoint_metrics.json").is_file()
    finally:
        stack.restore_all(context)
    assert fake.save_calls == [{"checkpoint_dir": checkpoint_dir, "step": 1}]
    assert not getattr(fake, "_rosetta_v2_feature_checkpoint_memory_trim_installed")
    assert not getattr(fake, "_rosetta_v2_feature_checkpoint_metric_snapshot_installed")
