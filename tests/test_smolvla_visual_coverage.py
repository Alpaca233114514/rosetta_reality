"""Synthetic counterexamples for the fixed frame-zero development protocol."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from rosetta_reality.vla.visual_coverage import (
    PROTOCOL,
    array_hash,
    compare_arms,
    compare_reload,
    read_bundle,
    score_view,
    summarize_bundle,
    validate_bundle,
    write_bundle,
)

ROOT = Path(__file__).resolve().parents[1]
DIMS = [
    {"name": "joint", "unit": "radian", "minimum": -10, "maximum": 10},
    {
        "name": "gripper",
        "unit": "normalized",
        "encoding": "0_closed_1_open",
        "minimum": 0,
        "maximum": 1,
    },
]


def evidence(arm="A"):
    target = np.repeat(
        np.array(
            [[-1, 0], [1, 1], [0, 0.5], [-0.5, 0.25], [0.5, 0.75]], dtype=np.float64
        )[:, None],
        3,
        axis=1,
    )
    noise = np.zeros((4, 1, 3, 4), dtype=np.float32)
    for i in range(1, 4):
        noise[i] = np.random.default_rng(i).standard_normal((1, 3, 4))
    arrays = {
        "normalized_predictions": np.repeat(target[None], 4, axis=0),
        "standard_predictions": np.repeat(target[None], 4, axis=0),
        "internal_grippers": np.zeros((4, 5, 3, 1)),
        "normalized_targets": target.copy(),
        "standard_targets": target.copy(),
        "valid_mask": np.ones_like(target, dtype=bool),
        "noise": noise,
    }
    meta = {
        "protocol": PROTOCOL,
        "arm": arm,
        "episodes": [10, 11, 12, 20, 21],
        "views": {"train8": [10, 11], "train40": [10, 11, 12], "dev5": [20, 21]},
        "hidden_episodes": [30],
        "hidden_test_loaded": False,
        "target_conditioning": False,
        "noise_conditions": [None, 20260905, 20260906, 20260907],
        "noise_hashes": [array_hash(n) for n in noise],
        "chunk_length": 3,
        "max_action_dim": 4,
        "native_denoising_steps": 10,
        "model_mode": "eval_inference",
        "dimensions": copy.deepcopy(DIMS),
        "nonvisual_hashes": ["a" * 64] * 5,
        "image_hashes": [f"{i:064x}" for i in range(5)],
        "common_identity": {"registration": "synthetic"},
        "arm_identity": {"arm": arm},
        "process": {"pid": 1, "invocation_id": "first"},
    }
    return arrays, meta


def test_all_nonself_averages_errors_not_predictions():
    target = np.repeat(np.array([[-2.0, 0], [0, 0.5], [2, 1]])[:, None], 2, axis=1)
    score = score_view(target, target, np.ones_like(target, bool), DIMS)["all_valid"]
    # Six ordered mismatches: joint squared errors 4,16,4,4,16,4;
    # gripper squared errors .25,1,.25,.25,1,.25, averaged over both dimensions.
    assert score["mismatch"] == pytest.approx(4.25)
    assert score["mean_floor"] == pytest.approx(17 / 12)
    assert score["correct"] == 0 and score["passed"]
    averaged_donor_error = []
    for i in range(3):
        other_mean = target[[j for j in range(3) if j != i]].mean(axis=0)
        averaged_donor_error.append(np.square(other_mean - target[i]).mean())
    assert score["mismatch"] > np.mean(averaged_donor_error)


def test_image_invariant_oracle_cannot_pass():
    arrays, meta = evidence()
    y = arrays["normalized_targets"]
    p = np.broadcast_to(y.mean(axis=0), y.shape)
    for group in score_view(p, y, arrays["valid_mask"], meta["dimensions"]).values():
        assert group["correct"] == pytest.approx(group["mean_floor"])
        assert group["visual_gain"] == pytest.approx(0)
        assert not group["passed"]


def test_reversed_visual_mapping_is_negative_evidence():
    arrays, meta = evidence()
    y = arrays["normalized_targets"][:2]
    result = score_view(y[::-1], y, arrays["valid_mask"][:2], meta["dimensions"])
    assert result["all_valid"]["visual_gain"] < 0
    assert not result["all_valid"]["passed"]


def test_gripper_failure_not_hidden_by_joint_fit():
    arrays, meta = evidence()
    arrays["normalized_predictions"][:, 3:, :, 1] = 0.5
    result = summarize_bundle(arrays, meta)
    assert result["views"]["dev5"][0]["normalized"]["all_valid"]["passed"]
    assert not result["development_passed"]


def test_constant_gripper_labels_are_not_measurable():
    arrays, meta = evidence()
    arrays["normalized_targets"][..., 1] = 0.5
    result = summarize_bundle(arrays, meta)
    g = result["views"]["dev5"][0]["normalized"]["gripper_normalized"]
    assert g["status"] == "not measurable" and not result["development_passed"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda a, m: m.update(arm="C"),
        lambda a, m: m.update(chunk_length=0),
        lambda a, m: m.update(max_action_dim=1),
        lambda a, m: m.update(hidden_test_loaded=True),
        lambda a, m: m.update(target_conditioning=True),
        lambda a, m: m.update(hidden_episodes=[10]),
        lambda a, m: m.update(noise_conditions=[None, 1, 2, 3]),
        lambda a, m: m.update(native_denoising_steps=9),
        lambda a, m: m.update(model_mode="train"),
        lambda a, m: m["nonvisual_hashes"].__setitem__(1, "b" * 64),
        lambda a, m: m["image_hashes"].__setitem__(1, m["image_hashes"][0]),
        lambda a, m: m["views"]["dev5"].__setitem__(0, 10),
        lambda a, m: a["valid_mask"].__setitem__((0, 2, 1), False),
        lambda a, m: a["noise"].__setitem__((0, 0, 0, 0), 1),
        lambda a, m: a["noise"].__setitem__((1, 0, 0, 0), 1),
        lambda a, m: a["normalized_predictions"].__setitem__((0, 0, 0, 0), np.nan),
        lambda a, m: a.update(standard_predictions=a["standard_predictions"][:, :, :1]),
        lambda a, m: m.update(process={"pid": 1}),
    ],
)
def test_corrupt_or_incomplete_evidence_is_rejected(mutation):
    arrays, meta = evidence()
    mutation(arrays, meta)
    with pytest.raises(ValueError):
        validate_bundle(arrays, meta)


def test_bundle_is_create_only_and_detects_file_tampering(tmp_path):
    arrays, meta = evidence()
    path = tmp_path / "bundle"
    write_bundle(path, arrays, meta)
    loaded, actual = read_bundle(path)
    assert actual == meta and np.array_equal(loaded["noise"], arrays["noise"])
    with pytest.raises(FileExistsError):
        write_bundle(path, arrays, meta)
    with (path / "noise.npy").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="file identity"):
        read_bundle(path)


@pytest.mark.parametrize("changed", [False, True])
def test_reload_compares_later_chunk_values_even_if_first_action_equal(
    tmp_path, changed
):
    arrays, meta = evidence()
    write_bundle(tmp_path / "first", arrays, meta)
    other = copy.deepcopy(meta)
    other["process"] = {"pid": 2, "invocation_id": "second"}
    if changed:
        arrays["normalized_predictions"][0, 0, 2, 0] += 0.01
    write_bundle(tmp_path / "second", arrays, other)
    result = compare_reload(tmp_path / "first", tmp_path / "second")
    assert result["passed"] is not changed
    assert result["exact_arrays"]["standard_predictions"]
    assert result["compared_entire_chunk"] and not result["metric_only_comparison"]


@pytest.mark.parametrize("same", ["pid", "invocation_id"])
def test_relabelled_same_process_is_not_independent_reload(tmp_path, same):
    arrays, meta = evidence()
    write_bundle(tmp_path / "first", arrays, meta)
    second = {"pid": 2, "invocation_id": "second"}
    second[same] = meta["process"][same]
    meta["process"] = second
    write_bundle(tmp_path / "second", arrays, meta)
    with pytest.raises(ValueError, match="independent"):
        compare_reload(tmp_path / "first", tmp_path / "second")


def test_perfect_both_arms_does_not_prove_coverage_gain():
    a, am = evidence("A")
    b, bm = evidence("B")
    result = compare_arms(a, am, b, bm)
    assert result["B"]["development_passed"]
    assert (
        not result["coverage_gain_passed"]
        and not result["offline_metric_criteria_passed"]
    )
    assert result["m2_complete"] is False


def test_coverage_gain_needs_fit_and_physical_nonregression():
    a, am = evidence("A")
    b, bm = evidence("B")
    a["normalized_predictions"][:, 3:] += 0.3
    a["standard_predictions"][:, 3:] += 0.3
    # A constant bias changes C but not visual gain, so it is insufficient.
    assert not compare_arms(a, am, b, bm)["coverage_gain_passed"]
    a["normalized_predictions"][:, 3:] = 0
    result = compare_arms(a, am, b, bm)
    assert result["offline_metric_criteria_passed"]
    b["standard_predictions"][:, 3:, 2, 0] += 3
    result = compare_arms(a, am, b, bm)
    assert not result["physical_nonregression"]["chunk/joint_radian"]
    assert not result["offline_metric_criteria_passed"]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_draft_refuses_execution_before_model_import(monkeypatch):
    module = load_script("evaluate_visual_coverage")
    with pytest.raises(ValueError, match="draft cannot run"):
        module.check_execution(
            {"protocol": PROTOCOL, "status": "draft_no_compute_authorization"}
        )


def test_negative_comparison_cli_saves_evidence_and_stops(tmp_path, monkeypatch):
    import sys

    module = load_script("evaluate_visual_coverage")
    for name in ("A", "B"):
        arrays, meta = evidence(name)
        write_bundle(tmp_path / name, arrays, meta)
    output = tmp_path / "comparison.json"
    monkeypatch.setattr(sys, "argv", ["evaluate_visual_coverage.py", "compare",
        "--first", str(tmp_path / "A"), "--second", str(tmp_path / "B"),
        "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 1
    result = json.loads(output.read_text())
    assert not result["coverage_gain_passed"] and not result["m2_complete"]


def test_plan_changes_active_scope_and_preserves_native_recipe():
    module = load_script("prepare_visual_coverage")
    control = yaml.safe_load((ROOT / module.CONTROL).read_text())
    original = copy.deepcopy(control)
    review = json.loads((ROOT / module.REVIEW).read_text())
    plans = module.build_plans(control, review)
    main = plans["main256"]
    assert control == original
    assert main["training"] == control["training"]
    assert main["initialization"] == control["initialization"]
    assert main["resources"] == control["resources"]
    assert main["tracking"] == control["tracking"]
    assert main["optimizer_smoke"]["episodes"] == review["data"]["train40"]
    sampler = next(f for f in main["features"] if f["name"] == "fixed_frame_sampler")
    assert sampler["sample_identities"] == [
        {"episode": e, "frame": 0} for e in review["data"]["train40"]
    ]
    assert plans["smoke2"]["optimizer_smoke"]["steps"] == 2
    assert main["optimizer_smoke"]["steps"] == 256
    assert main["status"] == "pending_compute_authorization"
    assert not main["formal_training_claim"] and not main["formal_training_authorized"]


def test_real_loader_rejects_pending_candidate_authorization():
    from rosetta_reality.vla.training.plan import validate_plan_structure

    module = load_script("prepare_visual_coverage")
    control = yaml.safe_load((ROOT / module.CONTROL).read_text())
    review = json.loads((ROOT / module.REVIEW).read_text())
    candidate = module.build_plans(control, review)["main256"]
    with pytest.raises(ValueError, match="preregistered"):
        validate_plan_structure(
            candidate, known_features={f["name"] for f in candidate["features"]}
        )
