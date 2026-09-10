"""Regression coverage for real launch failures found by the visual smoke."""

import itertools
import types

import pytest

import rosetta_reality.vla.training.features as features


def test_logger_uses_resolved_parent_and_plan(monkeypatch):
    import trackio

    import rosetta_reality.tracking.trackio_lerobot as bridge

    observed = {}
    context = types.SimpleNamespace(
        experiment={
            "experiment_id": "visual-small",
            "dataset": {"revision": "pinned"},
            "resources": {"accelerator": "xpu"},
        },
        plan={
            "tracking": {"project": "visual-small", "space_id": ""},
            "resources": {"accelerator": "cuda"},
        },
    )
    monkeypatch.setenv("ROSETTA_VLA_PHASE", "smoke")

    def raw_config_forbidden():
        raise AssertionError("Raw inherited YAML must not be consumed by the v2 logger.")

    def public_config(cfg, experiment, phase):
        observed["experiment"] = experiment
        return {}

    def initialize(**kwargs):
        observed["init"] = kwargs
        return types.SimpleNamespace(id="local-run")

    monkeypatch.setattr(bridge, "_experiment_config", raw_config_forbidden)
    monkeypatch.setattr(bridge, "_public_config", public_config)
    monkeypatch.setattr(trackio, "init", initialize)
    feature = features.TrackioLoggingFeature({})
    monkeypatch.setattr(feature, "_public_extension", lambda _: {})
    cfg = types.SimpleNamespace(
        job_name="visual-smoke", resume=False, wandb=types.SimpleNamespace(run_id=None)
    )
    feature._build_logger_class(context)(cfg)
    assert observed["experiment"]["dataset"]["revision"] == "pinned"
    assert observed["experiment"]["resources"]["accelerator"] == "cuda"
    assert observed["init"]["project"] == "visual-small"
    assert cfg.wandb.run_id == "local-run"


@pytest.mark.parametrize("training_fails", [True, False])
def test_cleanup_preserves_primary_failure(tmp_path, monkeypatch, training_fails):
    import lerobot.scripts.lerobot_train as upstream
    import train_smolvla_v2 as entry

    import rosetta_reality.tracking.trackio_lerobot as bridge

    plan = tmp_path / "plan.yaml"
    plan.write_text("test: true")
    monkeypatch.setenv(entry.LAUNCHER_VALIDATED_ENV, "1")
    monkeypatch.setenv(entry.PLAN_PATH_ENV, str(plan))
    monkeypatch.setattr(entry, "load_v2_plan", lambda *args: {})
    monkeypatch.setattr(entry, "validate_plan_structure", lambda *args, **kwargs: None)
    monkeypatch.setattr(entry, "_training_context", lambda *args: None)
    restored = []
    stack = types.SimpleNamespace(
        install_all=lambda _: ["trackio_logging"], restore_all=lambda _: restored.append(True)
    )
    monkeypatch.setattr(entry, "feature_stack_from_plan", lambda _: stack)

    def train():
        if training_fails:
            raise KeyError("primary-config-error")

    def finish():
        raise RuntimeError("secondary-cleanup-error")

    monkeypatch.setattr(upstream, "main", train)
    monkeypatch.setattr(bridge, "finish_trackio", finish)
    expected = KeyError if training_fails else RuntimeError
    message = "primary-config-error" if training_fails else "secondary-cleanup-error"
    with pytest.raises(expected, match=message):
        entry.main()
    assert restored == [True]


def test_native_sampler_yields_only_registered_cross_episode_samples(monkeypatch):
    import lerobot.scripts.lerobot_train as upstream

    original = upstream.EpisodeAwareSampler
    module = types.SimpleNamespace(EpisodeAwareSampler=original)
    monkeypatch.setattr(features, "_lerobot_train_module", lambda: module)
    episodes = [4, 1]
    samples = [{"episode": 4, "frame": 0}, {"episode": 1, "frame": 2}]
    context = types.SimpleNamespace(
        phase="smoke",
        plan={"scope": "bounded_visual_overfit", "optimizer_smoke": {"episodes": episodes}},
        experiment={
            "dataset": {
                "train_episodes": episodes,
                "validation_episodes": [2],
                "test_episodes": [3],
            }
        },
    )
    feature = features.FixedFrameSamplerFeature({"phase": "smoke", "sample_identities": samples})
    feature.install(context)
    try:
        sampler = module.EpisodeAwareSampler(
            [0, 10, 20, 30, 40],
            [10, 20, 30, 40, 50],
            episodes,
            shuffle=True,
            seed=42,
            absolute_to_relative_idx={12: 2, 40: 10},
        )
        actual = list(itertools.islice(iter(sampler), 3))
        assert sorted(actual) == [2, 10]
        assert len(sampler) == 2
    finally:
        feature.restore(context)
    assert module.EpisodeAwareSampler is original
