"""Boundary counterexamples for data-first paired research, no policy weights."""

import json

import numpy as np
import pytest

from rosetta_reality.vla.visual_research import (
    analyze_predictions,
    decomposition,
    donors,
    episode_summary,
    seal_bundle,
    verify_bundle,
)
from scripts.audit_smolvla_inventory import inventory, inventory_v2


def test_inventory_counterexample_and_complete_roots(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "src/rosetta_reality/vla").mkdir(parents=True)
    path = tmp_path / "scripts/canonical_entry.py"
    path.write_text(
        "from . import helper\nimport importlib\nimportlib.import_module(module_name)\n"
    )
    shell = tmp_path / "scripts/launch_canonical.sh"
    shell.write_text("python scripts/canonical_entry.py\n")
    nested = tmp_path / "reports/training/nested/closure.json"
    nested.parent.mkdir(parents=True)
    nested.write_text("{}")
    assert "scripts/canonical_entry.py" not in inventory(tmp_path)
    current = inventory_v2(tmp_path)
    assert current["scripts/canonical_entry.py"]["imports"] == [".", "importlib"]
    assert current["scripts/canonical_entry.py"]["dynamic_calls"]
    assert "reports/training/nested/closure.json" in current
    assert current["scripts/launch_canonical.sh"]["declared_file_references"] == [
        "scripts/canonical_entry.py"
    ]


def test_donors_exclude_query_and_development_labels():
    ids = np.array([[1, 0], [2, 0], [3, 0], [4, 0], [1, 5], [2, 5], [3, 5], [4, 5]])
    references = donors(ids, [1, 2])
    assert references[0].tolist() == [1]
    assert references[2].tolist() == [0, 1]
    assert references[6].tolist() == [4, 5]
    cyclic = donors(ids, [1, 2], "cyclic")
    assert [r.item() for r in cyclic] == [1, 0, 3, 2, 5, 4, 7, 6]


def test_duplicate_sample_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        donors(np.array([[1, 0], [1, 0]]), [1])


def test_decomposition_separates_reaction_from_mapping():
    y = np.array([[-1.0], [1.0]])
    wrong_direction = decomposition(-y, y)
    assert wrong_direction["prediction_variance"] == 1
    assert wrong_direction["covariance"] == -1
    assert wrong_direction["mse"] == 4


def test_episode_means_do_not_weight_repeated_frames():
    result = episode_summary([1, 1, 1, 9], [10, 10, 10, 20])
    assert result["mean"] == 5
    assert result["episode_count"] == 2
    assert result["leave_one_out_range"] == [1, 9]


def test_full_chunk_change_not_hidden_by_equal_first_action():
    target = np.arange(16, dtype=float).reshape(4, 2, 2) / 10
    prediction = target[None].copy()
    prediction[:, :, 1] += 2
    arrays = {
        "correct": prediction,
        "wrong": prediction.copy(),
        "targets": target,
        "identities": np.array([[1, 0], [2, 0], [3, 0], [4, 0]]),
        "valid_mask": np.ones((4, 2), dtype=bool),
        "frame0_nonvisual_equal": np.array(True),
    }
    result = analyze_predictions(arrays, [1, 2], {"joints": [0, 1]})
    assert result["records"][0]["correct_mse"]["mean"] == 0
    assert result["records"][1]["correct_mse"]["mean"] == pytest.approx(2)


def test_frame0_requires_equal_nonvisual_controls():
    arrays = {
        "correct": np.zeros((1, 4, 1, 1)),
        "wrong": np.zeros((1, 4, 1, 1)),
        "targets": np.zeros((4, 1, 1)),
        "identities": np.array([[1, 0], [2, 0], [3, 0], [4, 0]]),
        "valid_mask": np.ones((4, 1), dtype=bool),
        "frame0_nonvisual_equal": np.array(False),
    }
    with pytest.raises(ValueError, match="controls absent"):
        analyze_predictions(arrays, [1, 2], {"joint": [0]})


def test_unseen_input_is_not_reported_as_training_fit():
    arrays = {
        "correct": np.zeros((1, 4, 1, 1)),
        "wrong": np.zeros((1, 4, 1, 1)),
        "targets": np.zeros((4, 1, 1)),
        "identities": np.array([[1, 0], [2, 0], [3, 0], [4, 0]]),
        "valid_mask": np.ones((4, 1), dtype=bool),
        "frame0_nonvisual_equal": np.array(True),
        "input_seen_at_checkpoint": np.array([True, False, False, False]),
    }
    row = analyze_predictions(arrays, [1, 2], {"joint": [0]})["records"][0]
    assert row["cohort_interpretation"] == "training_split_contains_unseen_inputs"
    assert row["input_seen_at_checkpoint"] == {"1": True, "2": False}
    arrays["input_seen_at_checkpoint"][2] = True
    with pytest.raises(ValueError, match="Development input"):
        analyze_predictions(arrays, [1, 2], {"joint": [0]})


def test_collector_rejects_absent_permit_before_model_import(tmp_path):
    from scripts.visual_research_collect import collect

    with pytest.raises(ValueError, match="authorization"):
        collect({"gpu_authorized": False}, tmp_path, tmp_path)


def exposure_fixture():
    prediction = np.array([0, 0, 10, 10, 0, 0], dtype=float).reshape(1, 6, 1, 1)
    return {
        "correct": prediction,
        "wrong": prediction.copy(),
        "targets": np.zeros((6, 1, 1)),
        "identities": np.array([[e, 0] for e in range(1, 7)]),
        "valid_mask": np.ones((6, 1), dtype=bool),
        "frame0_nonvisual_equal": np.array(True),
        "input_seen_at_checkpoint": np.array([True, True, False, False, False, False]),
        "input_seen_by_step2500": np.array([True, True, False, False, False, False]),
    }


def test_mixed_fit_counterexample_is_separated():
    row = analyze_predictions(exposure_fixture(), [1, 2, 3, 4], {"joint": [0]})["records"][0]
    assert row["correct_mse"]["mean"] == 50  # Historical full-cohort metric stays intact.
    groups = row["exposure_strata"]
    assert groups["current_seen"]["metrics"]["correct_mse"]["mean"] == 0
    assert groups["current_unseen"]["metrics"]["correct_mse"]["mean"] == 100
    assert groups["current_seen"]["training_fit_eligible"]
    assert not groups["current_unseen"]["training_fit_eligible"]


def test_exposure_slicing_keeps_original_image_donors():
    row = analyze_predictions(exposure_fixture(), [1, 2, 3, 4], {"joint": [0]})["records"][0]
    # Query 1 still uses image outputs from episodes 2, 3, 4, not just seen episode 2.
    actual = row["exposure_strata"]["current_seen"]["metrics"]["wrong_mse"]
    assert actual["per_episode"]["1"] == pytest.approx(200 / 3)
    assert actual["per_episode"]["1"] == row["wrong_mse"]["per_episode"]["1"]


def test_fixed_cohort_survives_later_checkpoint_and_empty_is_not_zero():
    arrays = exposure_fixture()
    early = analyze_predictions(arrays, [1, 2, 3, 4], {"joint": [0]})["records"][0]
    arrays["input_seen_at_checkpoint"][:4] = True
    late = analyze_predictions(arrays, [1, 2, 3, 4], {"joint": [0]})["records"][0]
    for name in ("seen_by_step2500", "unseen_by_step2500"):
        assert early["exposure_strata"][name]["identities"] == (
            late["exposure_strata"][name]["identities"]
        )
    assert late["exposure_strata"]["current_unseen"]["metrics"] is None
    assert late["exposure_strata"]["current_unseen"]["status"] == "empty_cohort_not_measured"
    assert late["exposure_strata"]["unseen_by_step2500"]["training_fit_eligible"]


def test_invalid_fixed_cohort_is_rejected():
    arrays = exposure_fixture()
    arrays["input_seen_by_step2500"][2] = True
    with pytest.raises(ValueError, match="exceeds current exposure"):
        analyze_predictions(arrays, [1, 2, 3, 4], {"joint": [0]})


def test_bundle_tampering_rejected(tmp_path):
    (tmp_path / "data.json").write_text("{}")
    seal_bundle(tmp_path, {"stage": "unit"})
    assert verify_bundle(tmp_path)["stage"] == "unit"
    (tmp_path / "data.json").write_text("[]")
    with pytest.raises(ValueError, match="checksum"):
        verify_bundle(tmp_path)


def test_bundle_path_escape_rejected(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"files": {"../outside": {"sha256": "x"}}}))
    with pytest.raises(ValueError, match="Unsafe"):
        verify_bundle(tmp_path)


def test_current_plan_denies_gpu_before_import_or_output(tmp_path):
    from scripts.diagnose_smolvla_visual_research import load_plan

    plan = {
        "schema_version": 1,
        "status": "preregistered",
        "optimizer_updates": 0,
        "hidden_test_loaded": False,
        "authorized_stages": ["evidence"],
        "sources": {},
        "evidence_sha256": {},
    }
    file = tmp_path / "plan.json"
    file.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="authorization"):
        load_plan(file, "collect")
