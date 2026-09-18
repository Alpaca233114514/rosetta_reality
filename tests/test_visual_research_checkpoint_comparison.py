"""Synthetic arithmetic and control failures; these are not checkpoint evidence."""

import json
import sys

import numpy as np
import pytest

from rosetta_reality.vla.visual_research import analyze_predictions, seal_bundle
from scripts import compare_visual_research_checkpoints as compare


@pytest.fixture
def pair(tmp_path):
    groups = {
        "left_joints": list(range(6)),
        "right_joints": list(range(7, 13)),
        "left_gripper": [6],
        "right_gripper": [13],
    }
    ids = np.array([[e, f] for e in range(45) for f in (0, 125, 250, 375)])
    reference = np.arange(180) < 83
    models = (
        "a3494600183d0a24c71ab7db0976afbc57567ac714ac678fd572c607e06a2f0c",
        "d4c0d87cd8e66c723b07ec84f7f5e47875268bfd4e2057ec5683fdf56adcabef",
    )
    for step, value, model in zip((2500, 5000), (1.0, 2.0), models, strict=True):
        root = tmp_path / str(step)
        collect = root / "collect"
        collect.mkdir(parents=True)
        (root / "analyze").mkdir()
        arrays = {
            "identities": ids,
            "correct": np.full((4, 180, 50, 14), value),
            "wrong": np.full((4, 180, 50, 14), value),
            "targets": np.zeros((180, 50, 14)),
            "raw_targets": np.zeros((180, 50, 14)),
            "normalized_targets": np.zeros((180, 50, 14)),
            "valid_mask": np.ones((180, 50), dtype=bool),
            "noise": np.zeros((4, 1, 50, 32)),
            "input_state": np.zeros((180, 14)),
            "input_language_tokens": np.zeros((180, 48), dtype=int),
            "input_language_attention_mask": np.ones((180, 48), dtype=bool),
            "input_seen_by_step2500": reference,
            "input_seen_at_checkpoint": reference if step == 2500 else np.arange(180) < 160,
            "cyclic_donors": np.zeros(180, dtype=int),
            "frame0_nonvisual_equal": np.array(True),
        }
        np.savez_compressed(collect / "arrays.npz", **arrays)
        meta = dict(
            checkpoint_step=step,
            checkpoint_sha256=model,
            groups=groups,
            inputs=[dict(episode=int(e), frame=int(f)) for e, f in ids],
            parameters_unchanged=True,
            optimizer_steps=0,
            forwards=1268,
            controls=[dict(exact=True) for _ in range(8)],
            synthetic_fixture=True,
        )
        (collect / "result.json").write_text(json.dumps(meta))
        seal_bundle(collect, {"synthetic_fixture": True})
        result = analyze_predictions(arrays, list(range(40)), groups)
        (root / "analyze/result.json").write_text(json.dumps(result))
    return tmp_path


def invoke(root, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare",
            "--early",
            str(root / "2500"),
            "--late",
            str(root / "5000"),
            "--output",
            str(root / "out"),
        ],
    )
    compare.main()


def test_known_degradation_and_fixed_cohorts(pair, monkeypatch):
    invoke(pair, monkeypatch)
    result = json.loads((pair / "out/result.json").read_text())
    assert result["records"] == 512
    assert result["independent_metric_checks"] == 103680
    rows = [json.loads(x) for x in (pair / "out/comparisons.jsonl").read_text().splitlines()]
    for r in rows:
        assert r["metrics"]["correct_mse"]["late_minus_early"]["mean"] == pytest.approx(3.0)
        assert r["metrics"]["correct_mae"]["late_minus_early"]["mean"] == pytest.approx(1.0)
        if r["cohort"] == "unseen_by_step2500":
            assert r["early_seen"] == 0 and r["late_seen"] == r["input_count"]


def test_changed_noise_fails_even_with_resealed_bundle(pair, monkeypatch):
    root = pair / "2500/collect"
    with np.load(root / "arrays.npz", allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    arrays["noise"][1, 0, 0, 0] = 1
    np.savez_compressed(root / "arrays.npz", **arrays)
    manifest = json.loads((root / "manifest.json").read_text())
    path = root / "arrays.npz"
    manifest["files"]["arrays.npz"] = dict(bytes=path.stat().st_size, sha256=compare.digest(path))
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(AssertionError):
        invoke(pair, monkeypatch)
    assert not (pair / "out").exists()


def test_episode_unit_avoids_pseudoreplication_and_empty_is_null():
    result = compare.summarize(np.array([0.0, 2.0, 10.0]), np.array([1, 1, 2]))
    assert result["mean"] == 5.5
    assert result["episode_count"] == 2
    assert compare.summarize(np.array([]), np.array([])) is None
