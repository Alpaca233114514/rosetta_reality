"""Select a Way CUDA checkpoint with device-neutral memory provenance."""

from __future__ import annotations

import copy
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import run_smolvla_state_robustness_cuda_formal as formal_runner  # noqa: E402
import select_smolvla_checkpoint as selector  # noqa: E402

from rosetta_reality.experiment import file_sha256  # noqa: E402
from rosetta_reality.vla import load_smolvla_action_space  # noqa: E402


def _training_metrics(
    database: Path,
    run_name: str,
    training: dict[str, Any],
    plan_sha256: str,
) -> dict[str, Any]:
    """Read the historical Trackio schema using generic accelerator metrics."""

    if not database.is_file():
        raise FileNotFoundError("The durable Trackio database is missing.")
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT run_id, step, metrics FROM metrics WHERE run_name=? ORDER BY id",
            (run_name,),
        ).fetchall()
        configs = connection.execute(
            "SELECT run_id, config FROM configs WHERE run_name=? ORDER BY id",
            (run_name,),
        ).fetchall()
    finally:
        connection.close()
    if not rows or len({str(row[0]) for row in rows}) != 1 or len(configs) != 1:
        raise ValueError("Trackio Way run identity is missing or ambiguous.")
    config = json.loads(configs[0][1])
    optimizer_contract = formal_runner._optimizer_contract(training)
    optimizer = optimizer_contract["optimizer"]
    scheduler = optimizer_contract["scheduler"]
    expected = {
        "optimizer_type": optimizer["type"],
        "optimizer_lr": optimizer["lr"],
        "optimizer_betas": optimizer["betas"],
        "optimizer_eps": optimizer["eps"],
        "optimizer_weight_decay": optimizer["weight_decay"],
        "optimizer_grad_clip_norm": optimizer["grad_clip_norm"],
        "scheduler_type": scheduler["type"],
        "scheduler_warmup_steps": scheduler["num_warmup_steps"],
        "scheduler_decay_steps": scheduler["num_decay_steps"],
        "scheduler_peak_lr": scheduler["peak_lr"],
        "scheduler_decay_lr": scheduler["decay_lr"],
    }
    steps = int(training["steps"])
    log_freq = int(training["log_freq"])
    if (
        config.get("phase") != "formal"
        or config.get("steps") != steps
        or config.get("test_split_loaded") is not False
        or config.get("normalization_source_split") != "train"
        or config.get("formal_plan_sha256") != plan_sha256
        or any(config.get(key) != value for key, value in expected.items())
    ):
        raise ValueError("Trackio Way configuration differs from its plan.")
    train_rows: dict[int, dict[str, Any]] = {}
    checkpoint_steps: list[int] = []
    for _, step, raw_metrics in rows:
        metrics = json.loads(raw_metrics)
        numeric = [
            value
            for value in metrics.values()
            if isinstance(value, int | float) and not isinstance(value, bool)
        ]
        if len(numeric) != len(metrics) or not all(
            math.isfinite(float(value)) for value in numeric
        ):
            raise FloatingPointError("Trackio Way metrics are not finite numeric values.")
        if "train/loss" in metrics:
            if step in train_rows:
                raise ValueError("Trackio contains duplicate Way training steps.")
            required = {
                "train/grad_norm",
                "train/lr",
                "train/accelerator_max_allocated_bytes",
            }
            if not required <= set(metrics):
                raise ValueError("Trackio Way step lacks optimizer or memory evidence.")
            train_rows[int(step)] = metrics
        if metrics.get("system/checkpoint_saved") == 1:
            checkpoint_steps.append(int(step))
    expected_steps = list(range(log_freq, steps + 1, log_freq))
    if sorted(train_rows) != expected_steps:
        raise ValueError("Trackio Way training steps are incomplete.")
    ordered = [train_rows[step] for step in expected_steps]
    losses = [float(row["train/loss"]) for row in ordered]
    gradients = [float(row["train/grad_norm"]) for row in ordered]
    learning_rates = [float(row["train/lr"]) for row in ordered]
    allocations = [
        int(row["train/accelerator_max_allocated_bytes"]) for row in ordered
    ]
    maximum_allocation = max(allocations)
    return {
        "run_id": str(rows[0][0]),
        "logged_training_steps": len(train_rows),
        "checkpoint_steps": sorted(checkpoint_steps),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": min(losses),
        "maximum_loss": max(losses),
        "maximum_gradient_norm": max(gradients),
        "initial_learning_rate": learning_rates[0],
        "final_learning_rate": learning_rates[-1],
        "minimum_learning_rate": min(learning_rates),
        "maximum_learning_rate": max(learning_rates),
        "maximum_xpu_allocated_bytes": maximum_allocation,
        "maximum_accelerator_allocated_bytes": maximum_allocation,
        "optimizer_contract": optimizer_contract,
        "all_losses_and_gradients_finite": True,
    }


def _record_control_comparison(
    payload: dict[str, Any],
    plan: dict[str, Any],
    control_plan: dict[str, Any],
    control_report: dict[str, Any],
) -> None:
    control = plan["control_reference"]
    primary = str(plan["validation"]["primary_selection_metric"])
    candidate = payload.get("selected", {}).get("metrics", {}).get(primary)
    baseline = control_report.get("selected", {}).get("metrics", {}).get(primary)
    if (
        primary != "first_action_mae"
        or control_plan.get("run_name") != control.get("control_run")
        or control_report.get("status") != "passed"
        or control_report.get("stage") != "smolvla_formal_checkpoint_selection"
        or control_report.get("experiment_id") != payload.get("experiment_id")
        or control_report.get("formal_plan_sha256") != control.get("plan_sha256")
        or isinstance(candidate, bool)
        or not isinstance(candidate, int | float)
        or not math.isfinite(float(candidate))
        or isinstance(baseline, bool)
        or not isinstance(baseline, int | float)
        or not math.isfinite(float(baseline))
    ):
        raise ValueError("The registered Aster control comparison is invalid.")
    payload["control_comparison"] = {
        "control_run": control["control_run"],
        "selection_report_sha256": control["selection_report_sha256"],
        "metric": primary,
        "control_value": float(baseline),
        "candidate_value": float(candidate),
        "absolute_improvement": float(baseline) - float(candidate),
        "relative_improvement": (float(baseline) - float(candidate)) / float(baseline),
        "acceptance_role": "diagnostic_only_closed_loop_gate_remains_authoritative",
    }


def main() -> int:
    try:
        plan_path = Path(sys.argv[sys.argv.index("--plan") + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise ValueError("Way selection requires an explicit --plan path.") from error
    original_validate_plan = formal_runner._validate_plan
    plan, _base_path, experiment = original_validate_plan(plan_path)
    control = plan.get("control_reference", {})
    control_plan_path = formal_runner._repository_path(str(control.get("plan", "")))
    control_report_path = formal_runner._repository_path(
        str(control.get("selection_report", ""))
    )
    if (
        file_sha256(control_plan_path) != control.get("plan_sha256")
        or file_sha256(control_report_path) != control.get("selection_report_sha256")
    ):
        raise ValueError("The registered Aster control files changed.")
    control_plan = formal_runner._load_yaml(control_plan_path)
    control_report = formal_runner._load_json(control_report_path)

    def validate_plan_with_legacy_memory_alias(
        path: Path,
    ) -> tuple[dict[str, Any], Path, dict[str, Any]]:
        validated, base_path, active_experiment = original_validate_plan(path)
        delegated = copy.deepcopy(validated)
        delegated["resources"]["maximum_peak_xpu_allocated_bytes"] = delegated[
            "resources"
        ]["maximum_peak_accelerator_allocated_bytes"]
        return delegated, base_path, active_experiment

    formal_runner._validate_plan = validate_plan_with_legacy_memory_alias
    selector.formal_runner = formal_runner
    selector._training_metrics = _training_metrics
    original_validation = selector._validation_report
    original_create_json = selector.create_json

    def validation_report(path: Path, **kwargs: Any) -> dict[str, Any]:
        report = original_validation(path, **kwargs)
        if (
            report.get("action_space")
            != load_smolvla_action_space(experiment, require_explicit=True).as_dict()
            or report.get("bounded_gripper_decoder") is not True
            or report.get("state_jitter_active") is not False
            or report.get("state_jitter_training_only") is not True
            or report.get("state_robustness_profile")
            != plan["state_robustness_contract"]["profile"]
        ):
            raise ValueError("A Way validation report lost its clean-input boundary.")
        return report

    def create_json(path: Path, payload: dict[str, Any]) -> None:
        _record_control_comparison(payload, plan, control_plan, control_report)
        acceptance = payload.get("acceptance", {})
        legacy_key = "measured_peak_xpu_allocation_is_within_registered_guard"
        acceptance[
            "measured_peak_accelerator_allocation_is_within_registered_guard"
        ] = acceptance.pop(legacy_key)
        payload["training_metrics"].pop("maximum_xpu_allocated_bytes", None)
        payload.update(
            {
                "trackio_delivery_status": "public_checkpoint_sync_complete",
                "public_sync_performed": True,
                "bounded_gripper_decoder": True,
                "temporal_loss_profile": plan["loss_contract"]["profile"],
                "temporal_loss_normalization": plan["loss_contract"]["normalization"],
                "state_robustness_profile": plan["state_robustness_contract"][
                    "profile"
                ],
                "state_jitter_training_only": True,
                "selection_script_sha256": file_sha256(Path(__file__)),
            }
        )
        original_create_json(path, payload)

    selector._validation_report = validation_report
    selector.create_json = create_json
    return selector.main()


if __name__ == "__main__":
    raise SystemExit(main())
