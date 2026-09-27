"""Score sealed numeric trajectory arrays without importing a policy or dataset."""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import struct
from pathlib import Path

NOISE_CONDITIONS = ("zero", "seed_20260905", "seed_20260906", "seed_20260907")
TRAIN_EPISODES = frozenset((2, 49, 4, 23))
CANONICAL_ACTION_NAMES = (
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
)


def _integer(value):
    return type(value) is int


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError("Saved actions must contain numeric scalars")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Nonfinite saved action, including padded slots")
    return number


def _crossing(values, bounds):
    for step, (before, after) in enumerate(zip(values, values[1:]), 1):
        if before > bounds["close"] >= after:
            return {"step": step, "direction": "closing"}
        if before < bounds["open"] <= after:
            return {"step": step, "direction": "opening"}
    return None


def _mean(pair):
    total, count = pair
    return total / count if count else None


def _thresholds(declaration):
    source = Path(declaration["source_path"])
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != declaration["source_sha256"]:
        raise ValueError("Gripper threshold source SHA changed")
    raw = json.loads(
        payload,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError("Nonfinite threshold source: " + value)
        ),
    )
    result = {}
    for side in ("left", "right"):
        if declaration.get(side) != raw.get(side):
            raise ValueError("Gripper thresholds differ from sealed source")
        bounds = raw[side]
        close, opened = _finite(bounds["close"]), _finite(bounds["open"])
        if not 0 <= close < opened <= 1:
            raise ValueError("Invalid sealed gripper thresholds")
        result[side] = {"close": close, "open": opened}
    return result


def score_saved_predictions(
    records,
    *,
    expected_episodes,
    thresholds,
    gripper_indices=(6, 13),
    full_frames=True,
    expected_frames=None,
    horizon=50,
    action_dimension=14,
    action_names=CANONICAL_ACTION_NAMES,
):
    """Compare the same episode/frame targets and masks under four noise conditions.

    This checks supplied numeric rows only. The caller must separately bind their
    files, model, processor and policy identity to an immutable manifest.
    """
    episodes = list(expected_episodes)
    if (
        not episodes
        or any(not _integer(ep) for ep in episodes)
        or len(set(episodes)) != len(episodes)
        or not set(episodes) <= TRAIN_EPISODES
    ):
        raise ValueError("Expected episodes must be distinct registered training integers")
    if (
        not _integer(horizon)
        or horizon <= 0
        or not _integer(action_dimension)
        or action_dimension <= 0
    ):
        raise ValueError("Horizon and action dimension must be positive integers")
    if (
        not isinstance(gripper_indices, (tuple, list))
        or len(gripper_indices) != 2
        or any(not _integer(i) for i in gripper_indices)
        or not 0 < gripper_indices[0] < gripper_indices[1] == action_dimension - 1
    ):
        raise ValueError("Two ordered gripper indices must match the action dimension")
    if (
        not isinstance(action_names, (tuple, list))
        or len(action_names) != action_dimension
        or len(set(action_names)) != action_dimension
        or any(not isinstance(name, str) or not name for name in action_names)
        or action_names[gripper_indices[0]] != "left_gripper"
        or action_names[gripper_indices[1]] != "right_gripper"
    ):
        raise ValueError("Declared action names and gripper indices differ")
    if full_frames:
        if expected_frames is not None:
            raise ValueError("Full-frame mode derives its cohort from expected episodes")
        expected = {(ep, frame) for ep in episodes for frame in range(500)}
    else:
        if expected_frames is None:
            raise ValueError("Partial mode requires explicit nonempty expected_frames")
        frames = list(expected_frames)
        if (
            not frames
            or any(
                not isinstance(pair, (list, tuple))
                or len(pair) != 2
                or any(not _integer(v) for v in pair)
                for pair in frames
            )
            or len({tuple(pair) for pair in frames}) != len(frames)
        ):
            raise ValueError("Partial expected_frames must be nonempty unique integer pairs")
        expected = {tuple(pair) for pair in frames}
        if any(ep not in episodes or not 0 <= frame < 500 for ep, frame in expected):
            raise ValueError("Partial expected frame outside registered cohort")
    bounds = _thresholds(thresholds)
    cohorts = {noise: set() for noise in NOISE_CONDITIONS}
    reference = {}
    totals = {
        noise: {
            phase: {
                side: {group: [0.0, 0] for group in ("joint", "gripper")}
                for side in ("left", "right")
            }
            for phase in ("first", "full")
        }
        for noise in NOISE_CONDITIONS
    }
    bias = {
        noise: {
            side: {phase: [0.0, 0] for phase in ("event", "hold", "nonhold", "censored")}
            for side in ("left", "right")
        }
        for noise in NOISE_CONDITIONS
    }
    crossings = {
        noise: {
            side: {
                key: 0
                for key in (
                    "target_crossings",
                    "predicted_crossings",
                    "both_crossed",
                    "direction_mismatch",
                    "miss_within_window",
                    "prediction_censored",
                    "timing_delta_sum",
                )
            }
            for side in ("left", "right")
        }
        for noise in NOISE_CONDITIONS
    }
    crossing_rows = []
    for row in records:
        noise = row["noise"]
        if noise not in cohorts:
            raise ValueError("Unregistered noise condition")
        ep, frame = row["episode"], row["frame"]
        if not _integer(ep) or not _integer(frame):
            raise ValueError("Episode and frame must be integers, not booleans")
        pair = (ep, frame)
        if pair not in expected or pair in cohorts[noise]:
            raise ValueError("Unexpected or duplicate saved frame")
        cohorts[noise].add(pair)
        predicted, target, valid = row["predicted"], row["target"], row["valid"]
        if (
            len(predicted) != horizon
            or len(target) != horizon
            or len(valid) != horizon
            or any(type(flag) is not bool for flag in valid)
            or not valid[0]
            or valid != sorted(valid, reverse=True)
        ):
            raise ValueError("Saved chunk shape or tail-padding mask differs")
        predicted_values, target_values = [], []
        for prediction, truth in zip(predicted, target, strict=True):
            if len(prediction) != action_dimension or len(truth) != action_dimension:
                raise ValueError("Action dimension differs from declared contract")
            predicted_values.append([_finite(value) for value in prediction])
            target_values.append([_finite(value) for value in truth])
        valid_count = sum(valid)
        if valid_count != min(horizon, 500 - frame):
            raise ValueError("Tail padding differs from 500-frame episode")
        identity = (
            tuple(valid),
            tuple(struct.pack("<d", value) for slot in target_values for value in slot),
        )
        if pair in reference and reference[pair] != identity:
            raise ValueError("Four-noise target, mask or chunk shape differs")
        reference[pair] = identity
        for slot in range(valid_count):
            for index, (value, truth) in enumerate(
                zip(predicted_values[slot], target_values[slot], strict=True)
            ):
                side = "left" if index <= gripper_indices[0] else "right"
                group = "gripper" if index in gripper_indices else "joint"
                for phase in ("full", "first") if slot == 0 else ("full",):
                    item = totals[noise][phase][side][group]
                    item[0] += abs(value - truth)
                    item[1] += 1
        for side, index in zip(("left", "right"), gripper_indices, strict=True):
            truth = [target_values[slot][index] for slot in range(valid_count)]
            predicted_side = [predicted_values[slot][index] for slot in range(valid_count)]
            target_event = _crossing(truth, bounds[side])
            predicted_event = _crossing(predicted_side, bounds[side])
            item = crossings[noise][side]
            item["target_crossings"] += target_event is not None
            item["predicted_crossings"] += predicted_event is not None
            if (
                target_event
                and predicted_event
                and target_event["direction"] == predicted_event["direction"]
            ):
                status = "both_crossed"
                item["both_crossed"] += 1
                item["timing_delta_sum"] += predicted_event["step"] - target_event["step"]
            elif target_event and predicted_event:
                status = "direction_mismatch"
                item["direction_mismatch"] += 1
            elif target_event:
                status = "target_only_censored" if valid_count < horizon else "miss_within_window"
                item["prediction_censored" if valid_count < horizon else "miss_within_window"] += 1
            elif predicted_event:
                status = "prediction_only"
            else:
                status = "neither_crossed"
            crossing_rows.append(
                {
                    "noise": noise,
                    "episode": ep,
                    "frame": frame,
                    "side": side,
                    "target_step": target_event["step"] if target_event else None,
                    "target_direction": target_event["direction"] if target_event else None,
                    "predicted_step": predicted_event["step"] if predicted_event else None,
                    "predicted_direction": predicted_event["direction"]
                    if predicted_event
                    else None,
                    "status": status,
                }
            )
            phase = (
                "event"
                if target_event
                else "censored"
                if valid_count < horizon
                else "hold"
                if all(value <= bounds[side]["close"] for value in truth)
                else "nonhold"
            )
            item = bias[noise][side][phase]
            item[0] += predicted_values[0][index] - target_values[0][index]
            item[1] += 1
    if any(values != expected for values in cohorts.values()):
        raise ValueError("Four-noise saved frame cohorts differ or lack expected coverage")
    curves = {}
    for noise in NOISE_CONDITIONS:
        curves[noise] = {
            phase: {
                side: {group: _mean(item) for group, item in groups.items()}
                for side, groups in totals[noise][phase].items()
            }
            for phase in ("first", "full")
        }
        curves[noise]["gripper_bias"] = {
            side: {phase: {"bias": _mean(item), "count": item[1]} for phase, item in phases.items()}
            for side, phases in bias[noise].items()
        }
        curves[noise]["crossings"] = {
            side: {
                **item,
                "mean_prediction_minus_target_step": item["timing_delta_sum"] / item["both_crossed"]
                if item["both_crossed"]
                else None,
            }
            for side, item in crossings[noise].items()
        }
    return {
        "schema_version": 2,
        "status": "saved_arrays_scored",
        "scope": "saved_numeric_cohort_only",
        "identity_verified": False,
        "noise_curves": curves,
        "cohort_frames": len(expected),
        "tail_padding_excluded_from_metrics": True,
        "all_saved_values_finite": True,
        "crossing_rows": crossing_rows,
        "task_success": "not_measured",
    }
