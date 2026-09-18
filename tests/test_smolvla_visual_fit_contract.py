"""Counterexamples for actual Hestia recipe, recovery and observation evidence."""

from __future__ import annotations

import builtins
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest
import yaml

from rosetta_reality.vla import visual_fit as fit
from rosetta_reality.vla import visual_fit_contract as checks

ROOT = Path(__file__).resolve().parents[1]


def training(arm="C"):
    plan = yaml.safe_load(
        (ROOT / "configs/vla/m2-smolvla450m-visual-native-b4-pilot-003.yaml").read_text()
    )
    count = fit.UPDATES[arm]
    plan["run_name"] = fit.RUN_NAMES[arm]
    for name in ("training", "optimizer_smoke"):
        plan[name].update(
            episodes=fit.TRAIN40[:], steps=count, batch_size=4, save_freq=256 if arm == "B" else 320
        )
    plan["training"]["checkpoint_steps"] = [256] if arm == "B" else [320, 640, 960, 1280]
    plan["training"]["scheduler"] = checks.expected_scheduler(arm)
    plan["features"][2]["sample_identities"] = [{"episode": ep, "frame": 0} for ep in fit.TRAIN40]
    saved = {
        "job_name": fit.RUN_NAMES[arm],
        "steps": count,
        "batch_size": 4,
        "seed": 20260809,
        "num_workers": 0,
        "resume": False,
        "dataset": {"episodes": fit.TRAIN40[:], "revision": fit.REVISIONS["data_revision"]},
        "optimizer": copy.deepcopy(fit.OPTIMIZER),
        "scheduler": checks.expected_scheduler(arm),
        "accelerator": {"gradient_accumulation": {"steps": 1}},
        "policy": {
            "freeze_vision_encoder": True,
            "train_expert_only": True,
            "train_state_proj": True,
            "use_amp": False,
        },
    }
    identity = {"run_name": fit.RUN_NAMES[arm], "checkpoint_files": {"train_config.json": "a" * 64}}
    return plan, saved, identity


@pytest.mark.parametrize("arm", ["B", "C"])
def test_distinct_actual_saved_update_and_decay_budgets(arm):
    plan, saved, identity = training(arm)
    recipe = checks.validate_saved_recipe(saved, plan, arm, identity)
    assert recipe["optimizer_steps"] == recipe["scheduler_decay_steps"] == fit.UPDATES[arm]
    assert recipe["completed_sample_exposures"] == 4 * fit.UPDATES[arm]
    assert recipe["optimizer_resumed"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p, s: s.update(steps=256),
        lambda p, s: s.update(resume=True),
        lambda p, s: s.update(batch_size=True),
        lambda p, s: s["dataset"].update(episodes=fit.TRAIN40[::-1]),
        lambda p, s: s["dataset"].update(revision="0" * 40),
        lambda p, s: s["accelerator"]["gradient_accumulation"].update(steps=2),
        lambda p, s: s["scheduler"].update(num_decay_steps=256),
        lambda p, s: s["policy"].update(train_expert_only=False),
        lambda p, s: p["optimizer_smoke"].update(steps=256),
        lambda p, s: p["optimizer_smoke"].update(episodes=fit.TRAIN40[:8]),
        lambda p, s: p["features"].append({"name": "visual_paired_alignment"}),
        lambda p, s: p["initialization"].update(optimizer_state_reused=True),
        lambda p, s: p["parent_experiment"].update(sha256="f" * 64),
        lambda p, s: p["training"].update(checkpoint_steps=[1280]),
    ],
)
def test_saved_or_active_recipe_drift_is_rejected(mutation):
    plan, saved, identity = training()
    mutation(plan, saved)
    with pytest.raises(ValueError):
        checks.validate_saved_recipe(saved, plan, "C", identity)


def observation():
    samples = [[ep, 0] for _ in range(128) for ep in fit.TRAIN40]
    digest = hashlib.sha256(json.dumps(samples, separators=(",", ":")).encode()).hexdigest()
    rates = [
        [
            1e-4
            * (
                ((1 / 17 - 1) * (1 - i / 16) + 1)
                if i < 16
                else (0.975 * 0.5 * (1 + math.cos(math.pi * i / 1280)) + 0.025)
            )
        ]
        for i in range(1280)
    ]
    report = {
        "status": "passed",
        "expected_schedule_sha256": digest,
        "observation": {
            "status": "complete",
            "expected_samples": 5120,
            "delivered_samples": 5120,
            "completed_samples": 5120,
            "completed_updates": 1280,
            "successful_optimizer_step_calls": 1280,
            "learning_rates": rates,
            "errors": [],
            "pending_samples": [],
        },
    }
    schedule = {"status": "passed", "native_order_verified": True, "sample_identities": samples}
    resources = {
        "status": "passed",
        "stop_reasons": [],
        "peak_cuda_allocated_bytes": 3 * 1024**3,
        "peak_cuda_reserved_bytes": 4 * 1024**3,
        "peak_host_rss_bytes": 4 * 1024**3,
    }
    return report, schedule, resources


def test_complete_observation_requires_actual_success_not_prefetch():
    report, schedule, resources = observation()
    checks.validate_observation(report, schedule, resources, "C")
    report["observation"]["delivered_samples"] += 4
    with pytest.raises(ValueError, match="count"):
        checks.validate_observation(report, schedule, resources, "C")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r, s, m: r["observation"].update(completed_updates=1279),
        lambda r, s, m: r["observation"].update(pending_samples=[[49, 0]]),
        lambda r, s, m: r["observation"].update(errors=[{"type": "SkippedStep"}]),
        lambda r, s, m: r["observation"]["learning_rates"].__setitem__(16, [1e-4]),
        lambda r, s, m: r["observation"]["learning_rates"].__setitem__(900, [float("nan")]),
        lambda r, s, m: s["sample_identities"].__setitem__(1, [fit.HIDDEN5[0], 0]),
        lambda r, s, m: s["sample_identities"].reverse(),
        lambda r, s, m: m.update(peak_cuda_allocated_bytes=9 * 1024**3),
        lambda r, s, m: m.update(stop_reasons=["deadline"]),
    ],
)
def test_incomplete_wrong_order_old_scheduler_and_resource_failures(mutation):
    report, schedule, resources = observation()
    mutation(report, schedule, resources)
    with pytest.raises(ValueError):
        checks.validate_observation(report, schedule, resources, "C")


def test_processor_identity_excludes_weights_but_includes_tokenizer():
    files = {name: "a" * 64 for name in checks.PROCESSOR_FILES}
    files.update(
        {"model.safetensors": "b" * 64, "config.json": "c" * 64, "train_config.json": "d" * 64}
    )
    original = checks.processor_identity(files)
    files["model.safetensors"] = "e" * 64
    assert checks.processor_identity(files) == original
    files["tokenizer/tokenizer.json"] = "f" * 64
    assert checks.processor_identity(files) != original
    del files["policy_postprocessor.json"]
    with pytest.raises(ValueError, match="incomplete"):
        checks.processor_identity(files)


def test_complete_inventory_rejects_modified_and_unregistered_files(tmp_path):
    (tmp_path / "state.json").write_text("{}")
    inventory = {"state.json": fit.file_hash(tmp_path / "state.json")}
    checks.check_inventory(tmp_path, inventory)
    (tmp_path / "extra.json").write_text("{}")
    with pytest.raises(ValueError, match="inventory"):
        checks.check_inventory(tmp_path, inventory)
    inventory["extra.json"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        checks.check_inventory(tmp_path, inventory)


@pytest.mark.parametrize("name", ["../secret", "/absolute", "C:/absolute", "a\\b", "a/../b"])
def test_evidence_paths_cannot_escape_root(tmp_path, name):
    with pytest.raises(ValueError):
        checks.relative_file(tmp_path, name)


def test_draft_cannot_import_torch_or_model_runtime(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "hestia_collector_test", ROOT / "scripts/evaluate_visual_fit.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    draft = tmp_path / "draft.json"
    draft.write_text(json.dumps({"protocol": fit.PROTOCOL, "status": "draft"}))
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name in {"torch", "evaluate_visual_native_small"} or name.startswith("lerobot"):
            raise AssertionError("A draft reached model imports")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(ValueError, match="sealed"):
        module.collect(draft, "C", tmp_path / "predictions")
    assert not (tmp_path / "predictions").exists()
