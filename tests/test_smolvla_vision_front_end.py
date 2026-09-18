"""Vision-front-end scope tests: application exactness and feature install.

These are synthetic no-weight tests: the pinned SmolVLA layout is mirrored by
a small module tree with the frozen-baseline ``requires_grad`` pattern, and
the feature is exercised against a stubbed trainer module.  They never load
model weights, datasets or accelerators.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from torch import nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for candidate in (str(REPOSITORY_ROOT / "src"), str(REPOSITORY_ROOT / "scripts")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from rosetta_reality.vla.action_space import (  # noqa: E402
    load_smolvla_action_space,
    load_smolvla_experiment,
)
from rosetta_reality.vla.training import TrainingContext  # noqa: E402
from rosetta_reality.vla.training import features as features_module  # noqa: E402
from rosetta_reality.vla.vision_front_end import (  # noqa: E402
    FRONT_END_SCOPE,
    apply_front_end_scope,
    install_front_end_checkpointing,
    restore_front_end_checkpointing,
)

BOUNDED_GRIPPER_CONFIG = (
    REPOSITORY_ROOT
    / "configs/vla/smolvla_450m_aloha_insertion_action_repair_bounded_gripper_003.yaml"
)
CONTRACT_PATH = REPOSITORY_ROOT / "configs/sim/aloha_insertion_smolvla.yaml"


class _FakeVlmModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_model = nn.Linear(4, 4)
        self.connector = nn.Linear(4, 4)
        self.text_model = nn.Linear(4, 4)
        self.embed_tokens = nn.Embedding(8, 4)


class _FakeVlm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeVlmModel()
        self.lm_head = nn.Linear(4, 8)


class _FakeVlmWithExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vlm = _FakeVlm()
        self.lm_expert = nn.Linear(4, 4)


class _FakePolicyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vlm_with_expert = _FakeVlmWithExpert()
        self.state_proj = nn.Linear(4, 4)
        self.action_in_proj = nn.Linear(4, 4)
        self.action_out_proj = nn.Linear(4, 4)


def _baseline(model: _FakePolicyModel) -> None:
    """Mirror the frozen-baseline flags: the whole VLM is frozen."""
    for parameter in model.vlm_with_expert.vlm.parameters():
        parameter.requires_grad = False
    for parameter in model.state_proj.parameters():
        parameter.requires_grad = True
    for parameter in model.action_in_proj.parameters():
        parameter.requires_grad = True
    for parameter in model.action_out_proj.parameters():
        parameter.requires_grad = True
    for parameter in model.vlm_with_expert.lm_expert.parameters():
        parameter.requires_grad = True


def _scope_names(model: _FakePolicyModel) -> set[str]:
    names: set[str] = set()
    for dotted in FRONT_END_SCOPE:
        module = model.vlm_with_expert.get_submodule(dotted)
        names.update(f"vlm_with_expert.{dotted}.{name}" for name, _ in module.named_parameters())
    return names


def test_scope_flips_exactly_the_declared_front_end() -> None:
    model = _FakePolicyModel()
    _baseline(model)
    report = apply_front_end_scope(model)

    scope = _scope_names(model)
    assert scope, "the synthetic tree must expose scope parameters"
    for name, parameter in model.named_parameters():
        if name in scope:
            assert parameter.requires_grad, name
        elif name.startswith("vlm_with_expert.vlm."):
            assert not parameter.requires_grad, name
        else:
            assert parameter.requires_grad, name

    assert report["scope_parameter_count"] == len(scope)
    assert report["newly_trainable_parameter_count"] == len(scope)
    assert report["changed_parameter_count"] == len(scope)
    assert report["language_model_stays_frozen"] is True
    assert report["profile"] == "visual_front_end_unfreeze"
    assert report["scope_modules"] == ["vlm.model.vision_model", "vlm.model.connector"]


def test_scope_is_idempotent_on_reapplication() -> None:
    model = _FakePolicyModel()
    _baseline(model)
    first = apply_front_end_scope(model)
    second = apply_front_end_scope(model)
    assert second["newly_trainable_parameter_count"] == 0
    assert second["scope_parameter_count"] == first["scope_parameter_count"]
    assert second["scope_parameter_names_sha256"] == first["scope_parameter_names_sha256"]


def test_scope_fails_closed_when_a_declared_module_is_missing() -> None:
    model = _FakePolicyModel()
    _baseline(model)
    del model.vlm_with_expert.vlm.model.connector
    with pytest.raises(ValueError, match="declared scope module"):
        apply_front_end_scope(model)


def test_scope_fails_closed_on_parameters_aliased_outside_the_scope() -> None:
    model = _FakePolicyModel()
    _baseline(model)
    shared = model.vlm_with_expert.vlm.model.vision_model.weight
    model.vlm_with_expert.vlm.model.text_model.weight = shared
    before = {name: p.requires_grad for name, p in model.named_parameters()}
    with pytest.raises(ValueError, match="aliased outside the scope"):
        apply_front_end_scope(model)
    assert {name: p.requires_grad for name, p in model.named_parameters()} == before


@pytest.mark.parametrize("path", ["model.text_model", "model.embed_tokens", "lm_head"])
def test_scope_rejects_trainable_nonvisual_vlm_before_changing_flags(path: str) -> None:
    model = _FakePolicyModel()
    _baseline(model)
    for parameter in model.vlm_with_expert.vlm.get_submodule(path).parameters():
        parameter.requires_grad = True
    before = {name: p.requires_grad for name, p in model.named_parameters()}
    with pytest.raises(ValueError, match="Non-visual VLM parameter must stay frozen"):
        apply_front_end_scope(model)
    assert {name: p.requires_grad for name, p in model.named_parameters()} == before


def test_scope_checks_language_model_namespace_without_vacuous_success() -> None:
    model = _FakePolicyModel()
    _baseline(model)
    core = model.vlm_with_expert.vlm.model
    core.language_model = core.text_model
    del core.text_model
    report = apply_front_end_scope(model)
    assert report["frozen_non_visual_vlm_parameter_count"] == 5
    assert report["language_model_stays_frozen"] is True
    core.language_model.weight.requires_grad = True
    with pytest.raises(ValueError, match="Non-visual VLM parameter must stay frozen"):
        apply_front_end_scope(model)


@pytest.mark.parametrize("missing", ["vision_model", "connector", "nonvisual"])
def test_scope_rejects_empty_parameter_groups_without_mutation(missing: str) -> None:
    model = _FakePolicyModel()
    _baseline(model)
    core = model.vlm_with_expert.vlm.model
    if missing == "nonvisual":
        core.text_model = nn.Identity()
        core.embed_tokens = nn.Identity()
        model.vlm_with_expert.vlm.lm_head = nn.Identity()
    else:
        setattr(core, missing, nn.Identity())
    before = {name: p.requires_grad for name, p in model.named_parameters()}
    with pytest.raises(ValueError, match="no parameters|No non-visual"):
        apply_front_end_scope(model)
    assert {name: p.requires_grad for name, p in model.named_parameters()} == before


def _context(tmp_path: Path, plan: dict[str, Any]) -> TrainingContext:
    experiment = load_smolvla_experiment(BOUNDED_GRIPPER_CONFIG, REPOSITORY_ROOT)
    action_space = load_smolvla_action_space(experiment, require_explicit=True)
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return TrainingContext(
        plan=plan,
        experiment=experiment,
        action_space=action_space,
        plan_path=plan_path,
        experiment_path=BOUNDED_GRIPPER_CONFIG,
        contract_path=CONTRACT_PATH,
        normalization_report=tmp_path / "normalization.json",
        phase="formal",
        device="cuda",
        run_name="unit-vfunfreeze-run",
    )


def _valid_plan() -> dict[str, Any]:
    return {
        "features": [{"name": "vision_front_end_unfreeze"}],
        "vision_front_end_contract": {
            "profile": "visual_front_end_unfreeze",
            "scope": ["vlm.model.vision_model", "vlm.model.connector"],
            "language_model": "frozen",
            "activation_checkpointing": "per_module_nonreentrant_vision_front_end",
        },
    }


def test_feature_requires_the_frozen_baseline_adaptation_and_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feature = features_module.VisionFrontEndUnfreezeFeature({})
    module = types.SimpleNamespace(make_policy=lambda *a, **k: None)
    monkeypatch.setattr(features_module, "_lerobot_train_module", lambda: module)
    context = _context(tmp_path, _valid_plan())

    broken_plan = _valid_plan()
    del broken_plan["vision_front_end_contract"]
    with pytest.raises(ValueError, match="treatment contract is missing"):
        feature.install(_context(tmp_path, broken_plan))

    experiment = dict(context.experiment)
    model_section = dict(experiment["model"])
    adaptation = dict(model_section["adaptation"])
    adaptation["train_expert_only"] = False
    model_section["adaptation"] = adaptation
    experiment["model"] = model_section
    with pytest.raises(ValueError, match="frozen-baseline parent adaptation"):
        feature.install(
            TrainingContext(
                plan=_valid_plan(),
                experiment=experiment,
                action_space=context.action_space,
                plan_path=context.plan_path,
                experiment_path=context.experiment_path,
                contract_path=context.contract_path,
                normalization_report=context.normalization_report,
                phase="formal",
                device="cuda",
                run_name="unit-vfunfreeze-run",
            )
        )


def test_feature_wraps_make_policy_writes_scope_report_and_restores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feature = features_module.VisionFrontEndUnfreezeFeature({})
    model = _FakePolicyModel()
    _baseline(model)
    policy = types.SimpleNamespace(model=model)
    first_call = [True]

    def _make_policy(*args: Any, **kwargs: Any) -> types.SimpleNamespace:
        if first_call:
            first_call.clear()
            return policy
        fresh_model = _FakePolicyModel()
        _baseline(fresh_model)
        return types.SimpleNamespace(model=fresh_model)

    module = types.SimpleNamespace(make_policy=_make_policy)
    monkeypatch.setattr(features_module, "_lerobot_train_module", lambda: module)
    monkeypatch.setenv("ROSETTA_RUN_ROOT", str(tmp_path))
    context = _context(tmp_path, _valid_plan())

    feature.install(context)
    try:
        assert module.make_policy() is policy
        scope = _scope_names(model)
        assert all(
            parameter.requires_grad for name, parameter in model.named_parameters() if name in scope
        )
        report_path = (
            tmp_path
            / str(context.experiment["experiment_id"])
            / "diagnostics"
            / "vision-front-end-scope-unit-vfunfreeze-run.json"
        )
        assert report_path.is_file()
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["scope"]["profile"] == "visual_front_end_unfreeze"
        assert payload["run_name"] == "unit-vfunfreeze-run"
        # The create-only guard refuses a second report for the same run;
        # the second call receives a fresh policy, mirroring one make_policy
        # invocation per trainer process.
        with pytest.raises(FileExistsError, match="create-only"):
            module.make_policy()
    finally:
        feature.restore(context)
    assert module.make_policy is _make_policy


class _CheckpointHost(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        tower = nn.Sequential(nn.Linear(5, 8), nn.GELU(), nn.Dropout(0.3), nn.Linear(8, 5)).double()
        connector = nn.Linear(5, 5).double()
        vlm_model = types.SimpleNamespace(vision_model=tower, connector=connector)
        vlm = types.SimpleNamespace(model=vlm_model)
        self.vlm_with_expert = types.SimpleNamespace(vlm=vlm)


def test_front_end_checkpointing_preserves_outputs_and_grads_exactly() -> None:
    import torch

    host = _CheckpointHost()
    tower = host.vlm_with_expert.vlm.model.vision_model
    inputs = torch.randn(6, 5, dtype=torch.float64)

    torch.manual_seed(11)
    plain_input = inputs.clone().requires_grad_(True)
    plain_output = tower(plain_input)
    plain_output.sum().backward()
    plain_grad = plain_input.grad.clone()
    plain_param_grads = [p.grad.clone() for p in tower.parameters()]

    for parameter in tower.parameters():
        parameter.grad = None
    record = install_front_end_checkpointing(host)
    assert record["profile"] == "per_module_nonreentrant_activation_checkpointing"
    assert record["wrapped_modules"] == [
        "vlm.model.vision_model",
        "vlm.model.connector",
    ]

    torch.manual_seed(11)
    checkpointed_input = inputs.clone().requires_grad_(True)
    checkpointed_output = tower(checkpointed_input)
    checkpointed_output.sum().backward()

    assert torch.equal(checkpointed_output, plain_output)
    assert torch.equal(checkpointed_input.grad, plain_grad)
    for parameter, reference in zip(tower.parameters(), plain_param_grads, strict=True):
        assert torch.equal(parameter.grad, reference)

    restore_front_end_checkpointing(host)
    assert not getattr(tower.forward, "_rosetta_front_end_checkpoint", False)


def test_front_end_checkpointing_installs_only_once() -> None:
    host = _CheckpointHost()
    install_front_end_checkpointing(host)
    with pytest.raises(RuntimeError, match="already installed"):
        install_front_end_checkpointing(host)


def test_checkpoint_conflict_in_connector_does_not_wrap_tower() -> None:
    host = _CheckpointHost()
    core = host.vlm_with_expert.vlm.model
    original_tower = core.vision_model.forward
    original_connector = core.connector.forward

    def conflicting_forward(*args, **kwargs):
        return original_connector(*args, **kwargs)

    conflicting_forward._rosetta_front_end_checkpoint = True
    core.connector.forward = conflicting_forward
    with pytest.raises(RuntimeError, match="already installed"):
        install_front_end_checkpointing(host)
    assert core.vision_model.forward == original_tower
    assert core.connector.forward is conflicting_forward
