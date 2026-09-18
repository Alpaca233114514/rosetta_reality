"""Accept a SmolVLA smoke run from immutable gates, Trackio, and reload evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs/vla/smolvla_450m_aloha_insertion.yaml"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rosetta_reality.experiment import file_sha256  # noqa: E402
from rosetta_reality.features import create_json  # noqa: E402
from rosetta_reality.tracking import validate_public_payload  # noqa: E402
from rosetta_reality.vla import (  # noqa: E402
    load_smolvla_action_space,
    load_smolvla_experiment,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping: {path.name}.")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object: {path.name}.")
    json.dumps(value, allow_nan=False)
    return value


def _absolute_root(environment: str) -> Path:
    raw = os.environ.get(environment)
    if not raw:
        raise ValueError(f"{environment} must be set by the Docker runner.")
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError(f"{environment} must be absolute.")
    return path.resolve()


def _relative_evidence(path: Path, run_root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(run_root):
        raise ValueError("Acceptance evidence must remain inside the mounted run root.")
    return resolved.relative_to(run_root).as_posix()


def _decode_json(value: str | bytes) -> dict[str, Any]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise ValueError("Trackio JSON payload must be an object.")
    validate_public_payload(decoded, context="smoke_acceptance_trackio")
    return decoded


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric.")
    number = float(value)
    if not math.isfinite(number):
        raise FloatingPointError(f"{name} must be finite.")
    return number


def _validate_prerequisites(
    experiment: dict[str, Any],
    config_path: Path,
    contract_path: Path,
    benchmark_path: Path,
    gate1_path: Path,
    gate2_path: Path,
    trackio_sync_path: Path,
    preflight_path: Path,
) -> dict[str, dict[str, Any]]:
    config_sha256 = file_sha256(config_path)
    contract_sha256 = file_sha256(contract_path)
    benchmark = _load_json(benchmark_path)
    gate1 = _load_json(gate1_path)
    gate2 = _load_json(gate2_path)
    trackio_sync = _load_json(trackio_sync_path)
    preflight = _load_json(preflight_path)
    dataset = experiment["dataset"]
    tracking = experiment["tracking"]
    action_space = load_smolvla_action_space(experiment)
    if (
        benchmark.get("status") != "complete"
        or benchmark.get("stage") != "pre_training"
        or benchmark.get("experiment_config_sha256") != config_sha256
        or benchmark.get("action_contract_sha256") != contract_sha256
        or benchmark.get("dataset_revision") != dataset["revision"]
        or benchmark.get("evaluated_split") != "validation"
        or benchmark.get("normalization_source_split") != "train"
        or benchmark.get("hidden_test_loaded") is not False
    ):
        raise ValueError("The pre-training benchmark is not valid for this smoke run.")
    if (
        gate1.get("status") != "passed"
        or gate1.get("gate") != "m2_gate_1_scripted_action"
        or gate1.get("experiment_id") != experiment["experiment_id"]
        or gate1.get("action_contract_sha256") != contract_sha256
    ):
        raise ValueError("Gate 1 evidence is invalid.")
    if (
        gate2.get("status") != "passed"
        or gate2.get("gate") != "m2_gate_2_dataset_action_replay"
        or gate2.get("experiment_id") != experiment["experiment_id"]
        or gate2.get("action_contract_sha256") != contract_sha256
        or gate2.get("dataset_revision") != dataset["revision"]
        or gate2.get("acceptance_criteria", {}).get("timestamp_alignment") is not True
    ):
        raise ValueError("Gate 2 evidence is invalid.")
    if (
        trackio_sync.get("status") != "complete"
        or trackio_sync.get("space_id") != tracking["space_id"]
        or trackio_sync.get("space_sdk") != "static"
        or trackio_sync.get("visibility") != "public"
        or trackio_sync.get("contains_sensitive_data") is not False
        or trackio_sync.get("media_uploaded") is not False
        or trackio_sync.get("test_split_loaded") is not False
    ):
        raise ValueError("The pre-run Trackio Space evidence is invalid.")
    action_spec = _load_yaml(contract_path)["action"]
    if (
        preflight.get("status") != "passed"
        or preflight.get("stage") != "real_smolvla_no_optimizer_forward"
        or preflight.get("experiment_config_sha256") != config_sha256
        or preflight.get("action_contract_sha256") != contract_sha256
        or preflight.get("model_revision") != experiment["model"]["revision"]
        or preflight.get("dataset_revision") != dataset["revision"]
        or preflight.get("episodes_loaded") != experiment["phases"]["smoke"]["episodes"]
        or preflight.get("action_dimension") != action_spec["dimension"]
        or preflight.get("chunk_length") != action_spec["chunk_length"]
        or preflight.get("hidden_test_loaded") is not False
        or preflight.get("network_disabled") is not True
        or preflight.get("optimizer_created") is not False
        or preflight.get("gradients_enabled") is not False
        or preflight.get("mixed_precision") != experiment["resources"]["mixed_precision"]
        or preflight.get("action_space") != action_space.as_dict()
        or str(preflight.get("device")) not in str(experiment["resources"]["accelerator"])
    ):
        raise ValueError("The real SmolVLA preflight evidence is invalid.")
    _finite_number(preflight.get("loss"), "preflight loss")
    return {
        "benchmark": benchmark,
        "gate1": gate1,
        "gate2": gate2,
        "trackio_sync": trackio_sync,
        "preflight": preflight,
    }


def _validate_trackio_run(
    experiment: dict[str, Any],
    run_name: str,
) -> dict[str, Any]:
    trackio_root = _absolute_root("TRACKIO_DIR")
    project = str(experiment["tracking"]["project"])
    database = trackio_root / f"{project}.db"
    if not database.is_file():
        raise FileNotFoundError("The local Trackio database is missing.")
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        config_rows = connection.execute(
            "SELECT run_id, config, created_at FROM configs WHERE run_name = ?",
            (run_name,),
        ).fetchall()
        if len(config_rows) != 1:
            raise ValueError("Trackio must contain exactly one config for the smoke run name.")
        run_id, raw_config, created_at = config_rows[0]
        public_config = _decode_json(raw_config)
        metric_rows = connection.execute(
            "SELECT step, metrics, timestamp FROM metrics "
            "WHERE run_id = ? AND run_name = ? ORDER BY id",
            (run_id, run_name),
        ).fetchall()
    finally:
        connection.close()
    smoke = experiment["phases"]["smoke"]
    if (
        public_config.get("experiment_id") != experiment["experiment_id"]
        or public_config.get("phase") != "smoke"
        or public_config.get("model_revision") != experiment["model"]["revision"]
        or public_config.get("dataset_revision") != experiment["dataset"]["revision"]
        or public_config.get("seed") != experiment["seed"]
        or public_config.get("batch_size") != smoke["batch_size"]
        or public_config.get("steps") != smoke["steps"]
        or public_config.get("test_split_loaded") is not False
    ):
        raise ValueError("Trackio smoke config identity is invalid.")
    parsed_metrics = [
        {"step": int(step), "metrics": _decode_json(payload), "timestamp": timestamp}
        for step, payload, timestamp in metric_rows
    ]
    train_by_step: dict[int, dict[str, Any]] = {}
    checkpoint_steps: set[int] = set()
    for row in parsed_metrics:
        metrics = row["metrics"]
        if "train/loss" in metrics:
            train_by_step[row["step"]] = metrics
        if metrics.get("system/checkpoint_saved") == 1:
            checkpoint_steps.add(row["step"])
    expected_steps = set(range(1, int(smoke["steps"]) + 1))
    if set(train_by_step) != expected_steps or checkpoint_steps != expected_steps:
        raise ValueError("Trackio does not contain every registered train/checkpoint smoke step.")
    train_metrics: list[dict[str, float | int]] = []
    for step in sorted(expected_steps):
        values = train_by_step[step]
        train_metrics.append(
            {
                "step": step,
                "loss": _finite_number(values.get("train/loss"), f"step {step} loss"),
                "grad_norm": _finite_number(
                    values.get("train/grad_norm"), f"step {step} gradient norm"
                ),
                "learning_rate": _finite_number(values.get("train/lr"), f"step {step} lr"),
            }
        )
    return {
        "database": database.name,
        "run_id": str(run_id),
        "run_name": run_name,
        "created_at": str(created_at),
        "metric_rows": len(parsed_metrics),
        "train_metrics": train_metrics,
        "checkpoint_steps": sorted(checkpoint_steps),
    }


def _validate_reloads(
    experiment: dict[str, Any],
    config_path: Path,
    contract_path: Path,
    reload_paths: list[Path],
) -> dict[str, Any]:
    if len(reload_paths) < 2:
        raise ValueError("Smoke acceptance requires at least two independent reload reports.")
    expected_script_sha256 = file_sha256(REPOSITORY_ROOT / "scripts/verify_smolvla_checkpoint.py")
    expected_config_sha256 = file_sha256(config_path)
    expected_contract_sha256 = file_sha256(contract_path)
    action_spec = _load_yaml(contract_path)["action"]
    smoke = experiment["phases"]["smoke"]
    action_space = load_smolvla_action_space(experiment)
    reports = [_load_json(path) for path in reload_paths]
    for report in reports:
        if (
            report.get("status") != "passed"
            or report.get("stage") != "smolvla_checkpoint_independent_reload"
            or report.get("experiment_id") != experiment["experiment_id"]
            or report.get("experiment_config_sha256") != expected_config_sha256
            or report.get("verification_script_sha256") != expected_script_sha256
            or report.get("action_contract_sha256") != expected_contract_sha256
            or report.get("model_revision") != experiment["model"]["revision"]
            or report.get("dataset_revision") != experiment["dataset"]["revision"]
            or report.get("episodes_loaded") != smoke["episodes"]
            or report.get("checkpoint_step") != smoke["steps"]
            or report.get("action_dimension") != action_spec["dimension"]
            or report.get("chunk_length") != action_spec["chunk_length"]
            or report.get("action_space") != action_space.as_dict()
            or report.get("hidden_test_loaded") is not False
            or report.get("network_disabled") is not True
        ):
            raise ValueError("An independent checkpoint reload report is invalid.")
        boundary = report.get("serialized_action_boundary", {})
        diagnostics = report.get("raw_standard_action_diagnostics")
        if action_space.explicit and (
            boundary.get("explicit") is not True
            or boundary.get("projection_before_representation_before_normalization")
            is not True
            or boundary.get("unnormalization_before_inverse_and_clamp") is not True
            or not isinstance(diagnostics, dict)
            or not isinstance(
                diagnostics.get("dimensions", {}).get("right_gripper", {}).get("mae"),
                int | float,
            )
        ):
            raise ValueError("Reloaded repair checkpoint did not preserve diagnostics boundary.")
        _finite_number(report.get("fixed_input_loss"), "fixed-input reload loss")
    reference = reports[0]
    comparable_fields = (
        "checkpoint",
        "checkpoint_step",
        "checkpoint_hashes",
        "parameters",
        "fixed_input",
        "fixed_input_loss",
        "loss_details",
        "action_chunk",
        "serialized_action_boundary",
        "raw_standard_action_diagnostics",
    )
    for report in reports[1:]:
        for field in comparable_fields:
            if report.get(field) != reference.get(field):
                raise ValueError(f"Independent reloads differ in deterministic field: {field}.")
    return {
        "count": len(reports),
        "checkpoint": reference["checkpoint"],
        "checkpoint_step": reference["checkpoint_step"],
        "checkpoint_hashes": reference["checkpoint_hashes"],
        "action_chunk": reference["action_chunk"],
        "fixed_input_loss": reference["fixed_input_loss"],
        "verification_script_sha256": expected_script_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke-run-name", required=True)
    parser.add_argument("--benchmark-report", type=Path, required=True)
    parser.add_argument("--gate1-report", type=Path, required=True)
    parser.add_argument("--gate2-report", type=Path, required=True)
    parser.add_argument("--trackio-sync-report", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--reload-report", type=Path, action="append", required=True)
    parser.add_argument("--container-exit-code", type=int, required=True)
    parser.add_argument("--oom-event-count", type=int, required=True)
    args = parser.parse_args()
    if args.container_exit_code != 0 or args.oom_event_count != 0:
        raise RuntimeError("The observed smoke container did not exit cleanly within budget.")

    config_path = args.config.resolve()
    experiment = load_smolvla_experiment(config_path, REPOSITORY_ROOT)
    contract_path = REPOSITORY_ROOT / str(experiment["action_contract"]["derived"])
    run_root = _absolute_root("ROSETTA_RUN_ROOT")
    evidence_paths = [
        args.benchmark_report,
        args.gate1_report,
        args.gate2_report,
        args.trackio_sync_report,
        args.preflight_report,
        *args.reload_report,
    ]
    evidence = [_relative_evidence(path, run_root) for path in evidence_paths]
    _validate_prerequisites(
        experiment,
        config_path,
        contract_path,
        args.benchmark_report,
        args.gate1_report,
        args.gate2_report,
        args.trackio_sync_report,
        args.preflight_report,
    )
    tracking = _validate_trackio_run(experiment, args.smoke_run_name)
    reloads = _validate_reloads(
        experiment,
        config_path,
        contract_path,
        args.reload_report,
    )
    report = {
        "schema_version": 1,
        "status": "passed",
        "stage": "smolvla_tiny_smoke_acceptance",
        "experiment_id": experiment["experiment_id"],
        "smoke_run_name": args.smoke_run_name,
        "experiment_config_sha256": file_sha256(config_path),
        "action_contract_sha256": file_sha256(contract_path),
        "evidence": evidence,
        "tracking": {
            **tracking,
            "space_id": experiment["tracking"]["space_id"],
            "space_url": f"https://huggingface.co/spaces/{experiment['tracking']['space_id']}",
            "public_payload_only": True,
            "test_split_loaded": False,
        },
        "reloads": reloads,
        "runtime_observation": {
            "source": "orchestrated_container_exit_and_docker_events_before_host_restart",
            "container_exit_code": args.container_exit_code,
            "oom_event_count": args.oom_event_count,
            "memory_limit": experiment["resources"]["memory_limit"],
            "memory_swap_limit": experiment["resources"]["memory_swap_limit"],
        },
        "acceptance": {
            "finite_loss_and_gradients": True,
            "checkpoint_written": True,
            "checkpoint_reload_contract_matches": True,
            "trackio_run_readable": True,
            "no_resource_limit_violation": True,
            "hidden_test_loaded": False,
        },
    }
    json.dumps(report, allow_nan=False)
    acceptance_root = run_root / str(experiment["experiment_id"]) / "acceptance"
    destination = acceptance_root / f"{args.smoke_run_name}.json"
    create_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report: {destination.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
