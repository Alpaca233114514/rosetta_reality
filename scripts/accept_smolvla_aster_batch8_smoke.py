"""Accept the plan-bound Aster batch-8 optimizer smoke without loading weights."""

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

import run_smolvla_horizon_loss_formal as formal_runner  # noqa: E402
import run_smolvla_phase as phase_runner  # noqa: E402

from rosetta_reality.experiment import file_sha256  # noqa: E402
from rosetta_reality.features import create_json  # noqa: E402


def _finite(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        raise FloatingPointError(f"{label} is not finite.")
    return float(value)


def _trackio_evidence(
    database: Path,
    run_name: str,
    plan_sha256: str,
    smoke: dict[str, Any],
    maximum_xpu: int,
) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=30.0)
    try:
        configs = connection.execute(
            "SELECT run_id, config FROM configs WHERE run_name=? ORDER BY id",
            (run_name,),
        ).fetchall()
        rows = connection.execute(
            "SELECT run_id, step, metrics FROM metrics "
            "WHERE run_name=? ORDER BY id",
            (run_name,),
        ).fetchall()
    finally:
        connection.close()
    if len(configs) != 1:
        raise ValueError("Aster smoke must have exactly one durable Trackio config.")
    run_id = str(configs[0][0])
    config = json.loads(configs[0][1])
    expected_config = {
        "phase": "performance_benchmark",
        "formal_plan_sha256": plan_sha256,
        "batch_size": smoke["batch_size"],
        "steps": smoke["steps"],
        "save_freq": smoke["save_freq"],
        "compile_model": True,
        "compile_mode": "reduce-overhead",
        "skip_fully_masked_camera_encoding": True,
        "test_split_loaded": False,
        "bounded_gripper_decoder": True,
        "temporal_loss_profile": "first_action_only",
        "temporal_loss_normalization": "mean_over_selected_valid_entries",
    }
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise ValueError("Aster smoke Trackio config differs from its plan.")
    if {str(row[0]) for row in rows} != {run_id}:
        raise ValueError("Aster smoke Trackio rows have an ambiguous identity.")
    train: dict[int, dict[str, Any]] = {}
    checkpoint_steps: set[int] = set()
    peak_xpu = 0
    for _, step, raw in rows:
        metrics = json.loads(raw)
        if "train/loss" in metrics:
            step_number = int(step)
            if step_number in train:
                raise ValueError("Aster smoke contains duplicate train metrics.")
            for key in (
                "train/loss",
                "train/grad_norm",
                "train/lr",
                "train/xpu_allocated_bytes",
                "train/xpu_reserved_bytes",
                "train/xpu_max_allocated_bytes",
            ):
                _finite(metrics.get(key), f"step {step_number} {key}")
            peak_xpu = max(peak_xpu, int(metrics["train/xpu_max_allocated_bytes"]))
            train[step_number] = metrics
        if metrics.get("system/checkpoint_saved") == 1:
            checkpoint_steps.add(int(step))
    expected_steps = set(range(1, int(smoke["steps"]) + 1))
    if set(train) != expected_steps or checkpoint_steps != expected_steps:
        raise ValueError("Aster smoke lacks a registered metric or checkpoint step.")
    return {
        "database": database.name,
        "run_id": run_id,
        "metric_steps": sorted(train),
        "checkpoint_steps": sorted(checkpoint_steps),
        "peak_xpu_allocated_bytes": peak_xpu,
        "maximum_peak_xpu_allocated_bytes": maximum_xpu,
        "peak_xpu_within_guardrail": peak_xpu <= maximum_xpu,
    }


def _checkpoint_evidence(
    checkpoint_root: Path,
    experiment_id: str,
    run_name: str,
    smoke: dict[str, Any],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for step in range(1, int(smoke["steps"]) + 1):
        root = (
            checkpoint_root
            / experiment_id
            / "smoke"
            / run_name
            / "checkpoints"
            / f"{step:06d}"
        )
        required = [
            root / "pretrained_model/model.safetensors",
            root / "pretrained_model/train_config.json",
            root / "training_state/optimizer_state.safetensors",
            root / "training_state/scheduler_state.json",
            root / "training_state/training_step.json",
        ]
        if any(not path.is_file() or path.stat().st_size <= 0 for path in required):
            raise FileNotFoundError("Aster smoke checkpoint is incomplete.")
        training_step = formal_runner._load_json(required[-1])
        if (
            training_step.get("step") != step
            or training_step.get("batch_size") != smoke["batch_size"]
        ):
            raise ValueError("Aster smoke checkpoint identity differs from the plan.")
        evidence.append(
            {
                "step": step,
                "model_sha256": file_sha256(required[0]),
                "train_config_sha256": file_sha256(required[1]),
                "optimizer_state_bytes": required[2].stat().st_size,
            }
        )
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    plan, base_path, experiment = formal_runner._validate_plan(plan_path)
    smoke = plan["optimizer_smoke"]
    run_name = str(smoke["run_name"])
    maximum_xpu = int(plan["resources"]["maximum_peak_xpu_allocated_bytes"])
    checkpoint_root = phase_runner._absolute_root("ROSETTA_CHECKPOINT_ROOT")
    run_root = phase_runner._absolute_root("ROSETTA_RUN_ROOT")
    trackio_root = Path(os.environ.get("TRACKIO_DIR", ""))
    if not trackio_root.is_absolute():
        raise ValueError("TRACKIO_DIR must identify the durable Trackio root.")
    trackio = _trackio_evidence(
        trackio_root / f"{experiment['tracking']['project']}.db",
        run_name,
        file_sha256(plan_path),
        smoke,
        maximum_xpu,
    )
    checkpoints = _checkpoint_evidence(
        checkpoint_root,
        str(experiment["experiment_id"]),
        run_name,
        smoke,
    )
    if trackio["peak_xpu_within_guardrail"] is not True:
        raise MemoryError("Aster batch-8 smoke exceeded its XPU guardrail.")
    contract_path = REPOSITORY_ROOT / str(experiment["action_contract"]["derived"])
    report = {
        "schema_version": 1,
        "status": "passed",
        "stage": "smolvla_horizon_loss_batch8_optimizer_smoke_acceptance",
        "experiment_id": experiment["experiment_id"],
        "run_name": run_name,
        "formal_plan_sha256": file_sha256(plan_path),
        "experiment_config_sha256": file_sha256(base_path),
        "action_contract_sha256": file_sha256(contract_path),
        "temporal_loss_profile": plan["loss_contract"]["profile"],
        "temporal_loss_normalization": plan["loss_contract"]["normalization"],
        "batch_size": smoke["batch_size"],
        "steps": smoke["steps"],
        "checkpoint_steps": trackio["checkpoint_steps"],
        "all_metrics_finite": True,
        "peak_xpu_within_guardrail": trackio["peak_xpu_within_guardrail"],
        "trackio": trackio,
        "checkpoints": checkpoints,
        "hidden_test_loaded": False,
    }
    destination = (
        run_root
        / str(experiment["experiment_id"])
        / "acceptance"
        / f"{run_name}.json"
    )
    if destination.exists():
        raise FileExistsError("Aster batch-8 smoke acceptance is create-only.")
    create_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Acceptance: {destination.relative_to(run_root).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
