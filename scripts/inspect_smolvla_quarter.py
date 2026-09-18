"""Inspect one registered formal-training quarter without loading model weights."""

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

import run_smolvla_formal as formal_runner  # noqa: E402
import run_smolvla_phase as phase_runner  # noqa: E402

from rosetta_reality.experiment import file_sha256  # noqa: E402
from rosetta_reality.features import create_json  # noqa: E402


def _expected_learning_rate(contract: dict[str, Any], step: int) -> float:
    scheduler = contract["scheduler"]
    warmup = int(scheduler["num_warmup_steps"])
    decay = int(scheduler["num_decay_steps"])
    peak = float(scheduler["peak_lr"])
    minimum = float(scheduler["decay_lr"])
    if step < warmup:
        if step <= 0:
            return peak / (warmup + 1)
        fraction = 1.0 - step / warmup
        multiplier = (1.0 / (warmup + 1) - 1.0) * fraction + 1.0
        return peak * multiplier
    bounded_step = min(step, decay)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * bounded_step / decay))
    alpha = minimum / peak
    return peak * ((1.0 - alpha) * cosine_decay + alpha)


def _load_trackio_quarter(
    database: Path,
    run_name: str,
    step: int,
    *,
    plan_sha256: str,
    optimizer_contract: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    if not database.is_file():
        raise FileNotFoundError("The durable Trackio database is missing.")
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=30.0)
    try:
        configs = connection.execute(
            "SELECT run_id, config FROM configs WHERE run_name=? ORDER BY id",
            (run_name,),
        ).fetchall()
        rows = connection.execute(
            "SELECT run_id, metrics FROM metrics WHERE run_name=? AND step=? ORDER BY id",
            (run_name, step),
        ).fetchall()
    finally:
        connection.close()
    if len(configs) != 1 or not rows:
        raise ValueError("The registered Trackio run or quarter is not durable yet.")
    run_id = str(configs[0][0])
    if {str(row[0]) for row in rows} != {run_id}:
        raise ValueError("Trackio quarter rows have an ambiguous run identity.")
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
        raise ValueError("Trackio optimizer identity differs from the formal plan.")
    train_metrics: dict[str, Any] | None = None
    checkpoint_saved = False
    for _, raw_metrics in rows:
        metrics = json.loads(raw_metrics)
        if "train/loss" in metrics:
            if train_metrics is not None:
                raise ValueError("Trackio contains duplicate training metrics at the quarter.")
            train_metrics = metrics
        if metrics.get("system/checkpoint_saved") == 1:
            checkpoint_saved = True
    required = {
        "train/loss",
        "train/grad_norm",
        "train/lr",
        "train/xpu_allocated_bytes",
        "train/xpu_reserved_bytes",
        "train/xpu_max_allocated_bytes",
    }
    if train_metrics is None or not checkpoint_saved or not required <= set(train_metrics):
        raise ValueError("The quarter lacks complete training, XPU, or checkpoint evidence.")
    if any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        for value in train_metrics.values()
    ):
        raise FloatingPointError("Trackio quarter metrics are non-finite or non-numeric.")
    return run_id, config, train_metrics


def _validate_optimizer_state(
    step_dir: Path,
    training: dict[str, Any],
    step: int,
) -> dict[str, Any]:
    pretrained = step_dir / "pretrained_model"
    state = step_dir / "training_state"
    required = [
        pretrained / "config.json",
        pretrained / "model.safetensors",
        pretrained / "train_config.json",
        state / "optimizer_param_groups.json",
        state / "optimizer_state.safetensors",
        state / "rng_state.safetensors",
        state / "scheduler_state.json",
        state / "training_step.json",
    ]
    if any(not path.is_file() or path.stat().st_size <= 0 for path in required):
        raise FileNotFoundError("The quarter checkpoint is incomplete.")
    train_config = formal_runner._load_json(pretrained / "train_config.json")
    contract = formal_runner._validate_saved_optimizer_contract(train_config, training)
    if contract is None:
        raise ValueError("The formal optimizer contract is missing.")
    optimizer_groups = json.loads(
        (state / "optimizer_param_groups.json").read_text(encoding="utf-8")
    )
    scheduler_state = formal_runner._load_json(state / "scheduler_state.json")
    training_step = formal_runner._load_json(state / "training_step.json")
    if not isinstance(optimizer_groups, list) or len(optimizer_groups) != 1:
        raise ValueError("The formal optimizer must have exactly one parameter group.")
    group = optimizer_groups[0]
    optimizer = contract["optimizer"]
    expected_lr = _expected_learning_rate(contract, step)
    last_lrs = scheduler_state.get("_last_lr")
    if (
        training_step.get("step") != step
        or training_step.get("batch_size") != training["batch_size"]
        or scheduler_state.get("last_epoch") != step
        or not isinstance(last_lrs, list)
        or len(last_lrs) != 1
        or group.get("betas") != optimizer["betas"]
        or group.get("eps") != optimizer["eps"]
        or group.get("weight_decay") != optimizer["weight_decay"]
        or group.get("initial_lr") != optimizer["lr"]
        or not math.isclose(
            float(group.get("lr", math.nan)), expected_lr, rel_tol=1e-9, abs_tol=1e-12
        )
        or not math.isclose(float(last_lrs[0]), expected_lr, rel_tol=1e-9, abs_tol=1e-12)
    ):
        raise ValueError(
            "The saved optimizer or scheduler state diverged from the expected LR curve."
        )
    return {
        "expected_learning_rate": expected_lr,
        "optimizer_learning_rate": float(group["lr"]),
        "scheduler_learning_rate": float(last_lrs[0]),
        "scheduler_last_epoch": int(scheduler_state["last_epoch"]),
        "train_config_sha256": file_sha256(pretrained / "train_config.json"),
        "optimizer_state_bytes": (state / "optimizer_state.safetensors").stat().st_size,
        "model_safetensors_bytes": (pretrained / "model.safetensors").stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    plan, _base_path, experiment = formal_runner._validate_plan(plan_path)
    monitoring = formal_runner._validate_monitoring(plan)
    contract = formal_runner._optimizer_contract(plan["training"])
    if monitoring is None or contract is None or args.step not in monitoring["wake_steps"]:
        raise ValueError("The requested step is not a registered wake checkpoint.")
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
    optimizer_state = _validate_optimizer_state(step_dir, plan["training"], args.step)
    trackio_root = Path(os.environ.get("TRACKIO_DIR", ""))
    if not trackio_root.is_absolute():
        raise ValueError("TRACKIO_DIR must identify the durable Trackio root.")
    run_id, _config, metrics = _load_trackio_quarter(
        trackio_root / f"{experiment['tracking']['project']}.db",
        str(plan["run_name"]),
        args.step,
        plan_sha256=file_sha256(plan_path),
        optimizer_contract=contract,
    )
    log_freq = int(plan["training"]["log_freq"])
    window_lrs = [
        _expected_learning_rate(contract, current)
        for current in range(args.step - log_freq + 1, args.step + 1)
    ]
    tracked_lr = float(metrics["train/lr"])
    tolerance = 1e-12
    if not min(window_lrs) - tolerance <= tracked_lr <= max(window_lrs) + tolerance:
        raise ValueError("The logged LR is outside its registered averaging window.")
    peak_xpu = int(metrics["train/xpu_max_allocated_bytes"])
    maximum_xpu = int(plan["resources"]["maximum_peak_xpu_allocated_bytes"])
    if peak_xpu > maximum_xpu:
        raise MemoryError("The formal run exceeded its registered XPU allocation guardrail.")
    fraction = args.step / int(plan["training"]["steps"])
    report = {
        "schema_version": 1,
        "status": "passed",
        "stage": "smolvla_formal_quarter_monitor",
        "experiment_id": experiment["experiment_id"],
        "run_name": plan["run_name"],
        "codename": plan["furnace_program"]["codename"],
        "formal_plan_sha256": file_sha256(plan_path),
        "step": args.step,
        "fraction_complete": fraction,
        "trackio_run_id": run_id,
        "metrics": metrics,
        "tracked_lr_window": {
            "minimum_expected": min(window_lrs),
            "maximum_expected": max(window_lrs),
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
        / f"{plan['run_name']}-step-{args.step:06d}.json"
    )
    create_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Monitor report: {destination.relative_to(run_root).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
