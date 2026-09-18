"""Counterexamples for the 2026-09-14 local training-chain audit."""

from types import SimpleNamespace

import pytest
from test_smolvla_training_plan_schema import _base_plan

from rosetta_reality.vla.training.features import FEATURE_FACTORIES, FeatureStack
from rosetta_reality.vla.training.plan import validate_plan_structure


def test_restore_attempts_all_features_after_one_failure():
    calls = []

    def restore_bad(_):
        calls.append("bad")
        raise RuntimeError("restore failed")

    good = SimpleNamespace(name="good", restore=lambda _: calls.append("good"))
    bad = SimpleNamespace(name="bad", restore=restore_bad)
    stack = FeatureStack((good, bad), installed=["good", "bad"])
    with pytest.raises(BaseException, match="restore"):
        stack.restore_all(None)
    assert calls == ["bad", "good"]


def test_install_failure_survives_rollback_failure():
    calls = []

    def restore_bad(_):
        calls.append("restore")
        raise RuntimeError("secondary restore")

    def install_bad(_):
        raise ValueError("primary install")

    first = SimpleNamespace(name="first", install=lambda _: None, restore=restore_bad)
    second = SimpleNamespace(name="second", install=install_bad)
    stack = FeatureStack((first, second))
    with pytest.raises(ValueError, match="primary install"):
        stack.install_all(None)
    assert calls == ["restore"]


@pytest.mark.parametrize("section", [None, "training", "optimizer_smoke"])
def test_unimplemented_resume_is_rejected(section):
    plan = _base_plan()
    target = plan if section is None else plan.setdefault(section, {})
    target["resume"] = True
    with pytest.raises(ValueError, match="resume"):
        validate_plan_structure(plan, known_features=FEATURE_FACTORIES)


def test_duplicate_validation_frames_are_rejected():
    plan = _base_plan()
    plan["validation"]["frame_offsets"] = [0, 0]
    with pytest.raises(ValueError, match="repeat|duplicate"):
        validate_plan_structure(plan, known_features=FEATURE_FACTORIES)


def test_validation_cannot_invent_an_episode_outside_parent_split():
    from scripts import run_smolvla_v2 as launcher

    plan = _base_plan()
    plan["validation"]["episodes"] = [999]
    experiment = {
        "dataset": {
            "train_episodes": plan["training"]["episodes"],
            "validation_episodes": [22, 13],
            "test_episodes": [31],
        }
    }
    with pytest.raises(ValueError, match="validation|Validation"):
        launcher._validate_split(plan, experiment, "train")


def test_checkpoint_grid_must_include_final_step():
    plan = _base_plan()
    plan["training"].update(save_freq=1000, checkpoint_steps=[1000, 2000])
    with pytest.raises(ValueError, match="final|save frequency"):
        validate_plan_structure(plan, known_features=FEATURE_FACTORIES)


def test_ignored_accumulation_cannot_be_registered():
    plan = _base_plan()
    plan["training"]["gradient_accumulation_steps"] = 4
    with pytest.raises(ValueError, match="accumulation"):
        validate_plan_structure(plan, known_features=FEATURE_FACTORIES)


def test_launch_rejects_stale_implementation_before_resource_or_data_access(tmp_path, monkeypatch):
    from scripts import run_smolvla_v2 as launcher

    plan = _base_plan()
    source = tmp_path / "entry.py"
    source.write_text("changed implementation")
    plan["implementation_files"] = {"entry.py": "0" * 64}
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(launcher, "_resolve_plan", lambda _: (plan, source, {}))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("ROSETTA_DOCKER_MEMORY_LIMIT", "8g")
    monkeypatch.setenv("ROSETTA_DOCKER_MEMORY_SWAP_LIMIT", "8g")
    monkeypatch.setattr(launcher.sys, "argv", ["launcher", "smoke", "--plan", str(source)])

    def data_access(*args):
        raise AssertionError("Implementation identity was not checked before downstream work")

    monkeypatch.setattr(launcher, "_validate_split", data_access)
    with pytest.raises(ValueError, match="[Ii]mplementation"):
        launcher.main()
