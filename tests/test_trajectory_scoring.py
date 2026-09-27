"""Counterexamples for saved numeric scoring; no policy, data or simulator."""

import copy
import hashlib
import json

import pytest

from rosetta_reality.vla.training.trajectory_scoring import (
    NOISE_CONDITIONS,
    score_saved_predictions,
)

SYNTHETIC_ACTION_NAMES = ("left_joint", "left_gripper", "right_joint", "right_gripper")


def fixture(tmp_path, frames=(0,)):
    path = tmp_path / "thresholds.json"
    values = {side: {"close": 0.2, "open": 0.8} for side in ("left", "right")}
    path.write_text(json.dumps(values))
    thresholds = {
        "source_path": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        **values,
    }
    records = []
    for noise in NOISE_CONDITIONS:
        for frame in frames:
            records.append(
                {
                    "noise": noise,
                    "episode": 2,
                    "frame": frame,
                    "target": [[0.0, 0.9, 0.0, 0.0], [0.0, 0.1, 0.0, 0.0]],
                    "predicted": [[0.0, 0.9, 0.0, 0.0], [0.0, 0.1, 0.0, 0.0]],
                    "valid": [True, frame != 499],
                }
            )
    return records, thresholds


def score(records, thresholds, *, frames=(0,), **kwargs):
    return score_saved_predictions(
        records,
        expected_episodes=[2],
        expected_frames=[(2, frame) for frame in frames],
        full_frames=False,
        thresholds=thresholds,
        horizon=2,
        action_dimension=4,
        gripper_indices=(1, 3),
        action_names=SYNTHETIC_ACTION_NAMES,
        **kwargs,
    )


def test_full_default_cohort_and_numeric_only_scope(tmp_path):
    records, thresholds = fixture(tmp_path, range(500))
    result = score_saved_predictions(
        records,
        expected_episodes=[2],
        thresholds=thresholds,
        horizon=2,
        action_dimension=4,
        gripper_indices=(1, 3),
        action_names=SYNTHETIC_ACTION_NAMES,
    )
    assert result["cohort_frames"] == 500
    assert result["schema_version"] == 2
    assert result["scope"] == "saved_numeric_cohort_only"
    assert result["identity_verified"] is False
    assert result["task_success"] == "not_measured"


def test_same_frames_with_different_noise_targets_or_shapes_rejected(tmp_path):
    records, thresholds = fixture(tmp_path)
    altered = copy.deepcopy(records)
    altered[1]["target"][0][1] = 0.8
    with pytest.raises(ValueError, match="target, mask"):
        score(altered, thresholds)
    altered = copy.deepcopy(records)
    altered[1]["valid"] = [True, False]
    with pytest.raises(ValueError):
        score(altered, thresholds)
    altered = copy.deepcopy(records)
    altered[1]["target"].append([0.0] * 4)
    altered[1]["predicted"].append([0.0] * 4)
    altered[1]["valid"].append(True)
    with pytest.raises(ValueError, match="shape"):
        score(altered, thresholds)


def test_partial_requires_nonempty_explicit_integer_cohort(tmp_path):
    _, thresholds = fixture(tmp_path)
    with pytest.raises(ValueError, match="nonempty"):
        score_saved_predictions(
            [],
            expected_episodes=[2],
            expected_frames=[],
            full_frames=False,
            thresholds=thresholds,
            horizon=2,
            action_dimension=4,
            gripper_indices=(1, 3),
            action_names=SYNTHETIC_ACTION_NAMES,
        )
    with pytest.raises(ValueError, match="integers"):
        score_saved_predictions(
            [],
            expected_episodes=[True],
            expected_frames=[(True, 0)],
            full_frames=False,
            thresholds=thresholds,
            horizon=2,
            action_dimension=4,
            gripper_indices=(1, 3),
            action_names=SYNTHETIC_ACTION_NAMES,
        )


def test_padded_nan_is_rejected_before_scoring(tmp_path):
    records, thresholds = fixture(tmp_path, (499,))
    records[0]["predicted"][1][0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        score(records, thresholds, frames=(499,))


def test_directional_crossing_miss_and_tail_censor_are_distinct(tmp_path):
    records, thresholds = fixture(tmp_path)
    opposite = copy.deepcopy(records)
    for row in opposite:
        row["predicted"][0][1], row["predicted"][1][1] = 0.1, 0.9
    result = score(opposite, thresholds)
    left = next(
        row for row in result["crossing_rows"] if row["noise"] == "zero" and row["side"] == "left"
    )
    assert left["status"] == "direction_mismatch"
    assert left["target_direction"] == "closing" and left["predicted_direction"] == "opening"
    assert result["noise_curves"]["zero"]["crossings"]["left"]["both_crossed"] == 0
    missing = copy.deepcopy(records)
    for row in missing:
        row["predicted"][1][1] = 0.9
    result = score(missing, thresholds)
    assert result["crossing_rows"][0]["status"] == "miss_within_window"
    tail, thresholds = fixture(tmp_path, (499,))
    result = score(tail, thresholds, frames=(499,))
    assert result["noise_curves"]["zero"]["gripper_bias"]["right"]["censored"]["count"] == 1
    assert result["noise_curves"]["zero"]["gripper_bias"]["right"]["hold"]["count"] == 0
