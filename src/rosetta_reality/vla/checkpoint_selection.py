"""Validation-only selection with explicit identities and deterministic early ties."""

from __future__ import annotations

import math

from rosetta_reality.vla.training.plan import is_sha256

METRICS = (
    "first_action_mae",
    "fixed_flow_loss",
    "action_mae",
    "invalid_action_rate",
    "joint_limit_violation_rate",
)


def select_reports(records, *, identity, validation, checkpoint_steps, fixed_input):
    """Every record contains a sealed report and independently declared source identity.

    The base is a reference, not a checkpoint candidate. A negative improvement
    is retained; zero base MAE has an undefined fractional improvement.
    """
    if not checkpoint_steps or any(type(s) is not int or s <= 0 for s in checkpoint_steps):
        raise ValueError("Checkpoint steps must be positive integers")
    if len(checkpoint_steps) != len(set(checkpoint_steps)):
        raise ValueError("Repeated checkpoint step")
    episodes, frames = validation["episodes"], validation["frame_offsets"]
    for values in (episodes, frames):
        if not values or any(type(x) is not int or x < 0 for x in values):
            raise ValueError("Validation requires integer episode/frame identities")
        if len(values) != len(set(values)):
            raise ValueError("Duplicate validation identity")
    expected = {("base", 0), *(("checkpoint", s) for s in checkpoint_steps)}
    rows, seen = [], set()
    for record in records:
        report, source = record["report"], record["source"]
        kind, step = source.get("kind"), source.get("step")
        if type(step) is not int or (kind, step) not in expected or (kind, step) in seen:
            raise ValueError("Unexpected, missing or duplicate checkpoint identity")
        seen.add((kind, step))
        if not is_sha256(record.get("report_sha256")) or not is_sha256(source.get("model_sha256")):
            raise ValueError("Report and model identities require SHA-256")
        if report.get("status") != "complete" or report.get("stage") != "smolvla_fixed_validation":
            raise ValueError("Selection requires complete native validation reports")
        if any(report.get(key) != value for key, value in identity.items()):
            raise ValueError("Validation report identity differs from the registered plan")
        if (
            report.get("hidden_test_loaded") is not False
            or report.get("gradients_enabled") is not False
            or report.get("optimizer_created") is not False
            or report.get("network_disabled") is not True
            or report.get("validation_episodes") != episodes
            or report.get("materialized_episodes") != sorted(episodes)
            or report.get("frame_offsets") != frames
            or type(report.get("sample_count")) is not int
            or report["sample_count"] != len(episodes) * len(frames)
            or report.get("fixed_input") != fixed_input
        ):
            raise ValueError("Validation cohort, noise or isolation contract differs")
        actual_source = report.get("model_source", {})
        actual_step = actual_source.get("checkpoint_step", actual_source.get("step", 0))
        if (
            actual_source.get("kind") != kind
            or type(actual_step) is not int
            or actual_step != step
            or ("step" in actual_source and actual_source["step"] != step)
            or actual_source.get("model_safetensors_sha256") != source["model_sha256"]
        ):
            raise ValueError("Validation checkpoint source differs from the sealed input")
        values = {}
        for name in METRICS:
            value = report.get("metrics", {}).get(name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid finite nonnegative metric: {name}")
            if name.endswith("_rate") and value > 1:
                raise ValueError(f"Invalid metric rate: {name}")
            values[name] = float(value)
        rows.append(
            {
                "kind": kind,
                "checkpoint_step": step,
                "model_sha256": source["model_sha256"],
                "report_sha256": record["report_sha256"],
                **values,
            }
        )
    if seen != expected:
        raise ValueError("The complete registered checkpoint grid and base are required")
    checkpoints = [row for row in rows if row["kind"] == "checkpoint"]
    best = min(checkpoints, key=lambda r: (r["first_action_mae"], r["checkpoint_step"]))
    base = next(row for row in rows if row["kind"] == "base")["first_action_mae"]
    return {
        "schema_version": 2,
        "status": "selected",
        "stage": "smolvla_v2_validation_only_selection",
        "primary_metric": "first_action_mae",
        "tie_break": "earlier_checkpoint",
        "base_role": "reference_only",
        "selected_checkpoint_step": best["checkpoint_step"],
        "selected_model_sha256": best["model_sha256"],
        "selected_report_sha256": best["report_sha256"],
        "selected_primary_value": best["first_action_mae"],
        "base_primary_value": base,
        "improves_over_base": best["first_action_mae"] < base,
        "improvement_over_base_fraction": (base - best["first_action_mae"]) / base
        if base
        else None,
        "fraction_undefined_reason": "zero_base_metric" if base == 0 else None,
        "candidates": sorted(rows, key=lambda r: (r["first_action_mae"], r["checkpoint_step"])),
        "identity": identity,
        "validation": validation,
        "fixed_input": fixed_input,
        "hidden_test_loaded": False,
        "m2_complete": False,
        "task_success": "not measured",
    }
