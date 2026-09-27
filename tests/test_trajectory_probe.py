"""Synthetic contracts and saved numeric arrays only; no model/data execution."""

from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import yaml

from rosetta_reality.vla.training import trajectory_probe
from rosetta_reality.vla.training.trajectory_probe import (
    NOISE_CONDITIONS,
    effective_runtime_contract,
    observe_effective_runtime,
    read_installed_features,
    schedule,
    score_saved_predictions,
)
from scripts import prepare_trajectory_probe as prepare


def test_four_round_schedule_counts_exposures_separately_from_unique():
    for episodes, expected_steps in (((2,), 500), ((2, 49, 4, 23), 2000)):
        samples = schedule(episodes)
        assert len(samples) == expected_steps * 4
        assert len({tuple(sample) for sample in samples}) == len(episodes) * 500
        for epoch in range(4):
            one_round = samples[epoch * len(episodes) * 500 : (epoch + 1) * len(episodes) * 500]
            assert len(set(map(tuple, one_round))) == len(episodes) * 500
        assert samples == schedule(episodes)
        assert samples[:500] != samples[500:1000]


@pytest.mark.parametrize("episode", [31, 6, 1, 24, 5, 22, 13, 7, 33, 45])
def test_hidden_and_development_episodes_rejected(episode):
    with pytest.raises(ValueError, match="training episodes"):
        schedule((episode,))


def _runtime():
    parameter = SimpleNamespace(requires_grad=True)
    policy = SimpleNamespace(named_parameters=lambda: [("expert.weight", parameter)])
    optimizer = SimpleNamespace(
        param_groups=[
            {
                "params": [parameter],
                "lr": 0.001,
                "betas": (0.9, 0.95),
                "eps": 1e-8,
                "weight_decay": 0.0,
            }
        ]
    )
    scheduler = SimpleNamespace(get_last_lr=lambda: [0.001])
    features = {
        "explicit_sample_schedule": {
            "installed": True,
            "evidence": "native sampler identity ledger",
        }
    }
    loss = {
        "reduction": "global_valid_action_entries",
        "upstream_source_sha256": "a" * 64,
        "valid_steps_per_sample": [2],
        "denominator": 2,
        "action_dimension": 1,
    }
    expected = {
        "trainable_parameters": ["expert.weight"],
        "optimizer_groups": [
            {
                "parameters": ["expert.weight"],
                "lr": 0.001,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.0,
            }
        ],
        "features": ["explicit_sample_schedule"],
        "actual_lr": [0.001],
        "loss_mask": loss,
        "save_grid": [1],
    }
    ledger = {"status": "complete", "completed_updates": 1}
    return policy, optimizer, features, scheduler, loss, expected, ledger


def test_live_runtime_contract_rejects_parameter_feature_and_lr_drift():
    values = _runtime()
    assert effective_runtime_contract(*values)["status"] == "observed_contract_match"
    mismatches = []
    changed = list(_runtime())
    changed[5] = copy.deepcopy(changed[5])
    changed[5]["trainable_parameters"] = []
    mismatches.append(changed)
    changed = list(_runtime())
    changed[2] = {"explicit_sample_schedule": {"installed": False, "evidence": "claimed"}}
    mismatches.append(changed)
    changed = list(_runtime())
    changed[3] = SimpleNamespace(get_last_lr=lambda: [0.0005])
    mismatches.append(changed)
    changed = list(_runtime())
    changed[4] = {**changed[4], "denominator": 5}
    mismatches.append(changed)
    for mismatch in mismatches:
        with pytest.raises(ValueError):
            effective_runtime_contract(*mismatch)


def test_default_loss_uses_global_valid_denominator_with_unequal_padding(tmp_path, monkeypatch):
    source = tmp_path / "loss.py"
    source.write_text("synthetic default global reduction fixture")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(trajectory_probe, "PINNED_NATIVE_LOSS_SHA256", digest)
    batch = {
        "action": SimpleNamespace(shape=(2, 3, 2)),
        "action_is_pad": [[False, False, False], [False, True, True]],
    }
    result = trajectory_probe.native_loss_descriptor(
        batch, source_path=source, source_sha256=digest
    )
    assert result["denominator"] == 8
    assert result["reduction"] == "global_valid_action_entries"
    assert result["valid_steps_per_sample"] == [3, 1]


def test_native_first_batch_observer_writes_only_measured_contract(tmp_path, monkeypatch):
    policy, optimizer, features, scheduler, _, expected, ledger = _runtime()
    source = tmp_path / "upstream-loss.py"
    source.write_text("pinned synthetic reduction fixture\n")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(trajectory_probe, "PINNED_NATIVE_LOSS_SHA256", source_sha)
    expected["loss_source_path"] = str(source)
    expected["loss_source_sha256"] = source_sha
    expected["loss_mask"] = {
        "reduction": "global_valid_action_entries",
        "upstream_source_sha256": source_sha,
        "valid_steps_per_sample": [2],
        "denominator": 2,
        "action_dimension": 1,
    }
    batch = {"action": SimpleNamespace(shape=(1, 2, 1)), "action_is_pad": [[False, False]]}
    calls = []
    module = SimpleNamespace(
        make_optimizer_and_scheduler=lambda cfg: (optimizer, scheduler),
        update_policy=lambda metrics, active, data, opt: calls.append((active, data, opt)),
    )
    originals = module.make_optimizer_and_scheduler, module.update_policy
    output = tmp_path / "effective-runtime-contract.json"
    with observe_effective_runtime(
        module,
        expected=expected,
        feature_reader=lambda *_: features,
        ledger=ledger,
        output=output,
    ):
        actual_optimizer, _ = module.make_optimizer_and_scheduler(
            SimpleNamespace(steps=1, save_freq=1)
        )
        accelerated = SimpleNamespace(
            optimizer=actual_optimizer, param_groups=actual_optimizer.param_groups
        )
        module.update_policy(None, policy, batch, accelerated)
    assert (module.make_optimizer_and_scheduler, module.update_policy) == originals
    assert len(calls) == 1
    observed = json.loads(output.read_text())
    assert observed["loss_mask"]["denominator"] == 2
    assert observed["expected_save_grid"] == [1]
    assert observed["feature_evidence_scope"] == "synthetic_callback"
    with pytest.raises(FileExistsError):
        with observe_effective_runtime(
            module,
            expected=expected,
            feature_reader=lambda *_: features,
            ledger=ledger,
            output=output,
        ):
            pass


def test_live_feature_stack_reads_marker_and_rejects_drift():
    stack = SimpleNamespace(installed=["explicit_sample_schedule"])
    marker = "_rosetta_v2_feature_explicit_sample_schedule_installed"
    module = SimpleNamespace(**{marker: True})
    assert read_installed_features(stack, module)["explicit_sample_schedule"]["installed"]
    setattr(module, marker, False)
    with pytest.raises(ValueError, match="marker"):
        read_installed_features(stack, module)


def test_saved_array_scoring_excludes_tail_and_separates_sides(tmp_path):
    threshold_source = tmp_path / "thresholds.json"
    threshold_source.write_text(
        json.dumps(
            {
                "left": {"close": 0.2, "open": 0.8},
                "right": {"close": 0.2, "open": 0.8},
            }
        )
    )
    thresholds = {
        "source_path": str(threshold_source),
        "source_sha256": hashlib.sha256(threshold_source.read_bytes()).hexdigest(),
        "left": {"close": 0.2, "open": 0.8},
        "right": {"close": 0.2, "open": 0.8},
    }
    records = []
    for noise in NOISE_CONDITIONS:
        for frame in range(500):
            event = frame == 0
            target = [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0 if event else 0.0, 0.0, 0.0]]
            records.append(
                {
                    "noise": noise,
                    "episode": 2,
                    "frame": frame,
                    "predicted": [[1.0, 0.25, 0.0, 0.5], [0.0, 0.0, 0.0, 0.0]],
                    "target": target,
                    "valid": [True, frame != 499],
                }
            )
    result = score_saved_predictions(
        records,
        expected_episodes=[2],
        thresholds=thresholds,
        gripper_indices=(1, 3),
        horizon=2,
        action_dimension=4,
        action_names=("left_joint", "left_gripper", "right_joint", "right_gripper"),
    )
    assert result["cohort_frames"] == 500
    assert result["tail_padding_excluded_from_metrics"] is True
    assert result["noise_curves"]["zero"]["first"]["left"]["joint"] == 1.0
    assert result["noise_curves"]["zero"]["full"]["left"]["joint"] == pytest.approx(500 / 999)
    assert result["noise_curves"]["zero"]["gripper_bias"]["left"]["event"]["count"] == 1
    assert result["noise_curves"]["zero"]["gripper_bias"]["right"]["hold"]["count"] == 499
    assert result["noise_curves"]["zero"]["gripper_bias"]["right"]["censored"]["count"] == 1
    assert result["noise_curves"]["zero"]["crossings"]["left"]["target_crossings"] == 1
    assert result["crossing_rows"][0]["target_step"] == 1
    broken = copy.deepcopy(records)
    broken[0]["valid"] = [False, True]
    with pytest.raises(ValueError, match="tail-padding"):
        score_saved_predictions(
            broken,
            expected_episodes=[2],
            thresholds=thresholds,
            gripper_indices=(1, 3),
            horizon=2,
            action_dimension=4,
            action_names=("left_joint", "left_gripper", "right_joint", "right_gripper"),
        )
    with pytest.raises(ValueError, match="cohorts"):
        score_saved_predictions(
            records[:-1],
            expected_episodes=[2],
            thresholds=thresholds,
            gripper_indices=(1, 3),
            horizon=2,
            action_dimension=4,
            action_names=("left_joint", "left_gripper", "right_joint", "right_gripper"),
        )


def test_prepared_package_is_sealed_draft_and_launcher_rejects_it(tmp_path, monkeypatch):
    for member in prepare.implementation_members():
        copy_path = tmp_path / member
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        copy_path.write_bytes((prepare.ROOT / member).read_bytes())
    source_bytes = (prepare.ROOT / prepare.SOURCE).read_bytes()
    threshold_bytes = (prepare.ROOT / prepare.THRESHOLDS).read_bytes()
    threshold_path = tmp_path / prepare.THRESHOLDS
    threshold_path.parent.mkdir(parents=True)
    threshold_path.write_bytes(threshold_bytes)
    plan_path = tmp_path / prepare.SOURCE
    plan_path.parent.mkdir(parents=True)
    plan_path.write_bytes(source_bytes)
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    output = tmp_path / "prepared"
    manifest = prepare.prepare(plan_path, output)
    assert manifest["source_plan"]["sha256"] == prepare.SOURCE_SHA256
    assert manifest["execution"] is None and manifest["training_authorized"] is False
    assert manifest["optimizer_steps"] == 0
    for name, steps, exposures, unique in (
        ("smoke", 2, 2, 2),
        ("a", 500, 2000, 500),
        ("b", 2000, 8000, 2000),
    ):
        plan = yaml.safe_load((output / f"{name}-draft.yaml").read_text())
        assert plan["status"] == "draft" and plan["execution"] is None
        assert plan["training_authorized"] is False
        assert plan["rollout_authorized"] is False
        assert plan["optimizer_smoke"]["steps"] == steps
        assert plan["training"]["steps"] == (500 if name == "smoke" else steps)
        structural_copy = copy.deepcopy(plan)
        structural_copy["status"] = "preregistered"
        prepare.validate_plan_structure(structural_copy, known_features=prepare.FEATURE_FACTORIES)
        prepare.validate_local_implementation(structural_copy, tmp_path)
        assert plan["probe"]["sample_exposures"] == exposures
        assert plan["probe"]["unique_episode_frames"] == unique
        assert plan["training"]["scheduler"]["num_decay_steps"] == (
            500 if name == "smoke" else steps
        )
        assert plan["optimizer_smoke"]["save_checkpoint"] is True
        assert "5000" not in " ".join(plan["stop_conditions"])
        if name != "smoke":
            assert plan["training"]["checkpoint_steps"] == [steps * i // 4 for i in range(1, 5)]
        samples = json.loads((output / f"{name}-schedule.json").read_text())["sample_identities"]
        assert len(samples) == exposures
        if name == "smoke":
            assert samples == [[2, 0], [2, 499]]
    # This is the launcher's first structural gate, independent of model/data.
    from rosetta_reality.vla.training.plan import validate_plan_structure

    with pytest.raises(ValueError, match="preregistered"):
        validate_plan_structure(
            yaml.safe_load((output / "a-draft.yaml").read_text()), known_features=[]
        )
    with pytest.raises(FileExistsError):
        prepare.prepare(plan_path, output)
