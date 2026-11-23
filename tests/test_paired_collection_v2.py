"""File-only refusal tests; ML/simulation imports are deliberately absent."""
import importlib.util
import copy
from pathlib import Path
import pytest

_spec = importlib.util.spec_from_file_location(
    "_paired_rules_test", Path(__file__).resolve().parents[1] /
    "src/rosetta_reality/data/paired_collection_v2.py")
_rules = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rules)
match_pair, valid_starts = _rules.match_pair, _rules.valid_starts


def rows(n=100):
    return [{"step": i, "positive_target": True,
             "action_source": "state_conditioned_teacher"} for i in range(n)]


def test_full_native_chunks_no_tail_padding():
    assert valid_starts(rows(49)) == []
    assert valid_starts(rows(50)) == [0]
    assert valid_starts(rows(100)) == list(range(51))


def test_deviation_rejects_every_crossing_chunk():
    data = rows()
    data[49]["positive_target"] = False
    data[49]["action_source"] = "registered_deliberate_deviation"
    assert valid_starts(data) == [50]


def test_failed_unknown_and_missing_step_rejected():
    for key, value in (("positive_target", False), ("positive_target", None),
                       ("action_source", "unknown"), ("step", 100)):
        data = rows(50)
        data[25][key] = value
        assert valid_starts(data) == []


def pair():
    shared = dict(provider_id="test", pair_id="pair", parent_id="parent",
                  scene_fingerprint="scene", reset_state_sha256="state",
                  contract_sha256="contract", model_sha256="model",
                  task_success=True, safety_stop=False, restore_exact=True,
                  effective_input_frames=100, valid_target_slots=5000)
    return (dict(shared, kind="S", attempt_id="s", deviation=None),
            dict(shared, kind="R", attempt_id="r", verified_recovery_chain=True))


def test_matching_tolerance_and_real_success_required():
    s, r = pair()
    assert match_pair(s, r)["matched"]
    r["effective_input_frames"] = 94
    assert not match_pair(s, r)["matched"]
    s, r = pair()
    r["task_success"] = False
    assert not match_pair(s, r)["matched"]


def test_missing_recovery_or_identity_rejected():
    s, r = pair()
    r["verified_recovery_chain"] = False
    assert not match_pair(s, r)["matched"]
    s, r = pair()
    r["scene_fingerprint"] = "other"
    assert not match_pair(s, r)["matched"]


def test_complete_protection_boolean_without_actual_evidence_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="sealed evidence reference"):
        _rules.protected_inventory(tmp_path, {"phase": "Q2", "protected_scene_inventory_complete": True})


def test_unavailable_geometry_inventory_cannot_be_misreported_complete(tmp_path):
    result = _rules.protected_inventory(tmp_path, {"phase": "Q2", "protected_scene_inventory_complete": False})
    assert result["complete"] is False and result["fingerprints"] == []


def paired_rows(n=200):
    s = [{"step": i, "positive_target": True, "action_source": "state_conditioned_teacher",
          "state": [float(i)], "action": [float(i)], "before": {"state": i},
          "after": {"integration": [i]}, "phase": "grasp", "images_sha256": str(i)}
         for i in range(n)]
    return s, copy.deepcopy(s)


def test_003b_symmetric_sampler_preserves_ordinary_positive_targets():
    s, r = paired_rows()
    r[70]["positive_target"] = False
    r[70]["action_source"] = "registered_deliberate_deviation"
    original_s = copy.deepcopy(s)
    result = _rules.symmetric_sampler(s, r, 70)
    assert s == original_s
    assert result["S_symmetric_excluded_starts"] == list(range(21, 71))
    assert result["common_valid_chunk_starts"] == list(range(21)) + list(range(71, 151))
    assert result["raw_S_input_frames"] == 151 and result["raw_R_input_frames"] == 101
    assert result["sampler_effective_input_frames_per_role"] == 101
    assert result["sufficient_common_starts"] and result["pre_deviation_prefix_exact"]


def test_003b_short_common_grid_is_explicitly_ineligible():
    s, r = paired_rows(100)
    r[50]["positive_target"] = False
    r[50]["action_source"] = "registered_deliberate_deviation"
    result = _rules.symmetric_sampler(s, r, 50)
    assert result["common_valid_chunk_starts"] == [0]
    assert result["sufficient_common_starts"] is False


def test_003b_unknown_or_different_prefix_is_not_exact():
    s, r = paired_rows()
    r[70]["positive_target"] = False
    r[70]["action_source"] = "registered_deliberate_deviation"
    del r[10]["before"]
    assert _rules.symmetric_sampler(s, r, 70)["pre_deviation_prefix_exact"] is False
    s, r = paired_rows()
    r[10]["action"] = [-5.0]
    assert _rules.symmetric_sampler(s, r, 70)["common_valid_chunk_starts"] == []
