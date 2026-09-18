"""Inspect a registered Faust quarter using its preceding log-frequency window."""

from __future__ import annotations

import argparse
import json
import math
import os
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

import inspect_smolvla_quarter as inspector  # noqa: E402
import run_smolvla_action_repair_formal as formal_runner  # noqa: E402
import run_smolvla_phase as phase_runner  # noqa: E402

from rosetta_reality.experiment import file_sha256  # noqa: E402
from rosetta_reality.features import create_json  # noqa: E402


def _load_window(
    database: Path,
    run_name: str,
    step: int,
    log_freq: int,
    plan_sha256: str,
    optimizer_contract: dict[str, Any],
) -> tuple[str, int, dict[str, Any]]:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=30.0)
    try:
        configs = connection.execute(
            "SELECT run_id, config FROM configs WHERE run_name=? ORDER BY id",
            (run_name,),
        ).fetchall()
        rows = connection.execute(
            "SELECT run_id, step, metrics FROM metrics "
            "WHERE run_name=? AND step BETWEEN ? AND ? ORDER BY id",
            (run_name, step - log_freq + 1, step),
        ).fetchall()
    finally:
        connection.close()
    if len(configs) != 1 or not rows:
        raise ValueError("The Faust quarter window is not durable yet.")
    run_id = str(configs[0][0])
    config = json.loads(configs[0][1])
    optimizer = optimizer_contract["optimizer"]
    scheduler = optimizer_contract["scheduler"]
    expected_config = {
        "phase": "formal",
        "formal_plan_sha256": plan_sha256,
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
        "test_split_loaded": False,
    }
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise ValueError("The Faust Trackio config differs from its formal plan.")
    train_rows: list[tuple[int, dict[str, Any]]] = []
    checkpoint_saved = False
    for row_run_id, row_step, raw_metrics in rows:
        if str(row_run_id) != run_id:
            raise ValueError("The Faust quarter window has an ambiguous run identity.")
        metrics = json.loads(raw_metrics)
        if "train/loss" in metrics:
            train_rows.append((int(row_step), metrics))
        if int(row_step) == step and metrics.get("system/checkpoint_saved") == 1:
            checkpoint_saved = True
    if not checkpoint_saved or not train_rows:
        raise ValueError("The Faust quarter lacks its checkpoint or preceding train row.")
    metric_step, metrics = max(train_rows, key=lambda item: item[0])
    required = {
        "train/loss",
        "train/grad_norm",
        "train/lr",
        "train/xpu_allocated_bytes",
        "train/xpu_reserved_bytes",
        "train/xpu_max_allocated_bytes",
    }
    if not required <= set(metrics) or any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        for value in metrics.values()
    ):
        raise FloatingPointError("The Faust quarter metrics are incomplete or non-finite.")
    return run_id, metric_step, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    plan, _base, experiment = formal_runner._validate_plan(plan_path)
    monitoring = formal_runner._validate_monitoring(plan)
    contract = formal_runner._optimizer_contract(plan["training"])
    if monitoring is None or contract is None or args.step not in monitoring["wake_steps"]:
        raise ValueError("The requested step is not a registered Faust quarter.")
    checkpoint_root = phase_runner._absolute_root("ROSETTA_CHECKPOINT_ROOT")
    run_root = phase_runner._absolute_root("ROSETTA_RUN_ROOT")
    step_dir = (
        checkpoint_root
        / str(experiment["experiment_id"])
        / "formal"
        / str(plan["run_name"])
        / "checkpoints"
        / f"{args.step:06d}"
    )
    optimizer_state = inspector._validate_optimizer_state(
        step_dir, plan["training"], args.step
    )
    trackio_root = Path(os.environ.get("TRACKIO_DIR", ""))
    if not trackio_root.is_absolute():
        raise ValueError("TRACKIO_DIR must identify the durable Trackio root.")
    log_freq = int(plan["training"]["log_freq"])
    run_id, metric_step, metrics = _load_window(
        trackio_root / f"{experiment['tracking']['project']}.db",
        str(plan["run_name"]),
        args.step,
        log_freq,
        file_sha256(plan_path),
        contract,
    )
    expected_lrs = [
        inspector._expected_learning_rate(contract, step)
        for step in range(metric_step - log_freq + 1, metric_step + 1)
    ]
    tracked_lr = float(metrics["train/lr"])
    if not min(expected_lrs) - 1e-12 <= tracked_lr <= max(expected_lrs) + 1e-12:
        raise ValueError("The tracked LR is outside its actual logging window.")
    maximum_xpu = int(plan["resources"]["maximum_peak_xpu_allocated_bytes"])
    if int(metrics["train/xpu_max_allocated_bytes"]) > maximum_xpu:
        raise MemoryError("The Faust quarter exceeded its XPU guardrail.")
    report = {
        "schema_version": 1,
        "status": "passed",
        "stage": "smolvla_formal_quarter_window_monitor",
        "experiment_id": experiment["experiment_id"],
        "run_name": plan["run_name"],
        "codename": plan["furnace_program"]["codename"],
        "formal_plan_sha256": file_sha256(plan_path),
        "monitor_script_sha256": file_sha256(Path(__file__)),
        "step": args.step,
        "fraction_complete": args.step / int(plan["training"]["steps"]),
        "trackio_run_id": run_id,
        "metric_step": metric_step,
        "checkpoint_event_step": args.step,
        "metrics": metrics,
        "tracked_lr_window": {
            "minimum_expected": min(expected_lrs),
            "maximum_expected": max(expected_lrs),
            "logged_average": tracked_lr,
        },
        "optimizer_state": optimizer_state,
        "xpu_guardrail_bytes": maximum_xpu,
        "hidden_test_loaded": False,
    }
    destination = (
        run_root
        / str(experiment["experiment_id"])
        / "monitoring"
        / f"{plan['run_name']}-step-{args.step:06d}-window.json"
    )
    if destination.exists():
        raise FileExistsError("The Faust quarter window report is create-only.")
    create_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Monitor report: {destination.relative_to(run_root).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
