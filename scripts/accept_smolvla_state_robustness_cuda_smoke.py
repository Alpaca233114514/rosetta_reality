"""Accept a plan-bound Way CUDA smoke after independent checkpoint reload."""

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

import run_smolvla_action_repair_formal as optimizer_contract  # noqa: E402
import run_smolvla_phase as phase_runner  # noqa: E402
import run_smolvla_state_robustness_cuda_smoke as smoke_runner  # noqa: E402

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
    plan: dict[str, Any],
) -> dict[str, Any]:
    connection = sqlite3.connect(
        f"file:{database}?mode=ro&immutable=1", uri=True, timeout=30.0
    )
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
        raise ValueError("Way CUDA smoke must have exactly one Trackio config.")
    run_id = str(configs[0][0])
    config = json.loads(configs[0][1])
    smoke = plan["optimizer_smoke"]
    state_contract = plan["state_robustness_contract"]
    expected_config = {
        "phase": "performance_benchmark",
        "formal_plan_sha256": plan_sha256,
        "batch_size": smoke["batch_size"],
        "steps": smoke["steps"],
        "save_freq": smoke["save_freq"],
        "train_episode_count": 1,
        "compile_model": smoke["policy"]["compile_model"],
        "compile_mode": smoke["policy"]["compile_mode"],
        "skip_fully_masked_camera_encoding": True,
        "test_split_loaded": False,
        "resume": False,
        "accelerator": "cuda",
        "bounded_gripper_decoder": True,
        "temporal_loss_profile": plan["loss_contract"]["profile"],
        "temporal_loss_normalization": plan["loss_contract"]["normalization"],
        "state_robustness_profile": state_contract["profile"],
        "state_noise_std_normalized": state_contract[
            "normalized_standard_deviation"
        ],
        "state_jitter_training_only": True,
        "state_jitter_target_semantics": state_contract["target_semantics"],
        "autodl_runtime_profile_sha256": plan["runtime_profile"]["sha256"],
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected_config.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Way CUDA Trackio config differs from its plan: {mismatches}")
    if {str(row[0]) for row in rows} != {run_id}:
        raise ValueError("Way CUDA Trackio rows have an ambiguous identity.")
    train: dict[int, dict[str, Any]] = {}
    checkpoint_steps: set[int] = set()
    peak = 0
    for _, step, raw in rows:
        metrics = json.loads(raw)
        if "train/loss" in metrics:
            step_number = int(step)
            if step_number in train:
                raise ValueError("Way CUDA smoke contains duplicate train metrics.")
            for key in (
                "train/loss",
                "train/grad_norm",
                "train/lr",
                "train/accelerator_allocated_bytes",
                "train/accelerator_reserved_bytes",
                "train/accelerator_max_allocated_bytes",
                "train/cuda_allocated_bytes",
                "train/cuda_reserved_bytes",
                "train/cuda_max_allocated_bytes",
            ):
                _finite(metrics.get(key), f"step {step_number} {key}")
            if (
                metrics["train/accelerator_max_allocated_bytes"]
                != metrics["train/cuda_max_allocated_bytes"]
            ):
                raise ValueError("Generic and CUDA memory evidence disagree.")
            peak = max(
                peak, int(metrics["train/accelerator_max_allocated_bytes"])
            )
            train[step_number] = metrics
        if metrics.get("system/checkpoint_saved") == 1:
            checkpoint_steps.add(int(step))
    expected_steps = set(range(1, int(smoke["steps"]) + 1))
    if set(train) != expected_steps or checkpoint_steps != expected_steps:
        raise ValueError("Way CUDA smoke lacks a registered metric or checkpoint step.")
    return {
        "database": database.name,
        "run_id": run_id,
        "formal_plan_sha256": config["formal_plan_sha256"],
        "metric_steps": sorted(train),
        "losses": [train[step]["train/loss"] for step in sorted(train)],
        "gradient_norms": [
            train[step]["train/grad_norm"] for step in sorted(train)
        ],
        "checkpoint_steps": sorted(checkpoint_steps),
        "peak_accelerator_allocated_bytes": peak,
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
            raise FileNotFoundError("Way CUDA smoke checkpoint is incomplete.")
        training_step = json.loads(required[-1].read_text(encoding="utf-8"))
        if (
            training_step.get("step") != step
            or training_step.get("batch_size") != smoke["batch_size"]
        ):
            raise ValueError("Way CUDA checkpoint identity differs from the plan.")
        evidence.append(
            {
                "step": step,
                "model_sha256": file_sha256(required[0]),
                "train_config_sha256": file_sha256(required[1]),
                "optimizer_state_bytes": required[2].stat().st_size,
            }
        )
    return evidence


def _write_memory_failure(
    plan_path: Path,
    plan: dict[str, Any],
    experiment: dict[str, Any],
) -> None:
    if plan["activation"]["mode"] != "primary":
        return
    run_root = phase_runner._absolute_root("ROSETTA_RUN_ROOT")
    destination = (
        run_root
        / str(experiment["experiment_id"])
        / "failures"
        / f"{plan['run_name']}.json"
    )
    if destination.exists():
        return
    create_json(
        destination,
        {
            "schema_version": 1,
            "status": "failed",
            "stage": "smolvla_state_robustness_cuda_smoke_failure",
            "experiment_id": experiment["experiment_id"],
            "run_name": plan["run_name"],
            "formal_plan_sha256": file_sha256(plan_path),
            "batch_size": plan["optimizer_smoke"]["batch_size"],
            "failure_class": "peak_memory_guard_exceeded",
            "exception_type": "MemoryGuardExceeded",
            "automatic_retry_performed": False,
            "checkpoint_or_optimizer_state_reused_by_fallback": False,
            "hidden_test_loaded": False,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--reload-report", type=Path, required=True)
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    plan, base_path, experiment = smoke_runner._validate_plan(plan_path)
    smoke = plan["optimizer_smoke"]
    run_name = str(plan["run_name"])
    run_root = phase_runner._absolute_root("ROSETTA_RUN_ROOT")
    checkpoint_root = phase_runner._absolute_root("ROSETTA_CHECKPOINT_ROOT")
    trackio_root = Path(os.environ.get("TRACKIO_DIR", ""))
    if not trackio_root.is_absolute():
        raise ValueError("TRACKIO_DIR must identify the durable Trackio root.")
    launch_path = (
        run_root
        / str(experiment["experiment_id"])
        / "launch"
        / f"{run_name}.json"
    )
    completion_path = (
        run_root
        / str(experiment["experiment_id"])
        / "completion"
        / f"{run_name}.json"
    )
    launch = smoke_runner._load_json(launch_path)
    completion = smoke_runner._load_json(completion_path)
    reload_report = smoke_runner._load_json(args.reload_report.resolve())
    if (
        launch.get("status") != "preregistered"
        or launch.get("formal_plan_sha256") != file_sha256(plan_path)
        or launch.get("formal_training_authorized") is not False
        or completion.get("status") != "complete"
        or completion.get("formal_plan_sha256") != file_sha256(plan_path)
        or completion.get("batch_size") != smoke["batch_size"]
        or completion.get("steps") != smoke["steps"]
        or reload_report.get("status") != "passed"
        or reload_report.get("stage")
        != "smolvla_state_robustness_cuda_smoke_independent_reload"
        or reload_report.get("formal_plan_sha256") != file_sha256(plan_path)
        or reload_report.get("checkpoint_step") != smoke["steps"]
        or reload_report.get("hidden_test_loaded") is not False
    ):
        raise ValueError("Way CUDA launch, completion or reload evidence is invalid.")
    trackio = _trackio_evidence(
        trackio_root / f"{experiment['tracking']['project']}.db",
        run_name,
        file_sha256(plan_path),
        plan,
    )
    checkpoints = _checkpoint_evidence(
        checkpoint_root,
        str(experiment["experiment_id"]),
        run_name,
        smoke,
    )
    peak = max(
        int(trackio["peak_accelerator_allocated_bytes"]),
        int(completion["maximum_allocated_bytes"]),
        int(reload_report["accelerator_memory"]["maximum_allocated_bytes"]),
    )
    guard = int(plan["resources"]["maximum_peak_accelerator_allocated_bytes"])
    elapsed = _finite(completion.get("elapsed_seconds"), "completion elapsed time")
    acceptance = {
        "all_metrics_finite": True,
        "checkpoints_at_every_registered_step": trackio["checkpoint_steps"]
        == list(range(1, int(smoke["steps"]) + 1)),
        "independent_final_checkpoint_reload_passed": True,
        "optimizer_contract_matches_aster": optimizer_contract._optimizer_contract(
            smoke_runner._control_training(plan)
        )
        == launch.get("optimizer_contract"),
        "peak_accelerator_allocation_within_guard": peak <= guard,
        "wall_time_within_guard": elapsed
        <= int(plan["resources"]["maximum_wall_time_minutes"]) * 60,
        "hidden_test_not_loaded": True,
    }
    passed = all(acceptance.values())
    if not acceptance["peak_accelerator_allocation_within_guard"]:
        _write_memory_failure(plan_path, plan, experiment)
    contract_path = REPOSITORY_ROOT / str(experiment["action_contract"]["derived"])
    report = {
        "schema_version": 1,
        "status": "passed" if passed else "rejected",
        "stage": "smolvla_state_robustness_cuda_optimizer_smoke_acceptance",
        "experiment_id": experiment["experiment_id"],
        "run_name": run_name,
        "formal_plan_sha256": file_sha256(plan_path),
        "experiment_config_sha256": file_sha256(base_path),
        "action_contract_sha256": file_sha256(contract_path),
        "runtime_profile_sha256": plan["runtime_profile"]["sha256"],
        "launch_manifest_sha256": file_sha256(launch_path),
        "completion_report_sha256": file_sha256(completion_path),
        "loss_contract": plan["loss_contract"],
        "state_robustness_contract": plan["state_robustness_contract"],
        "batch_size": smoke["batch_size"],
        "steps": smoke["steps"],
        "elapsed_seconds": elapsed,
        "peak_accelerator_allocated_bytes": peak,
        "maximum_peak_accelerator_allocated_bytes": guard,
        "trackio": trackio,
        "checkpoints": checkpoints,
        "reload_report_sha256": file_sha256(args.reload_report.resolve()),
        "acceptance": acceptance,
        "hidden_test_loaded": False,
        "formal_training_authorized": False,
        "closed_loop_claim": False,
    }
    destination = (
        run_root
        / str(experiment["experiment_id"])
        / "acceptance"
        / f"{run_name}.json"
    )
    if destination.exists():
        raise FileExistsError("Way CUDA smoke acceptance is create-only.")
    create_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Acceptance: {destination.relative_to(run_root).as_posix()}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
