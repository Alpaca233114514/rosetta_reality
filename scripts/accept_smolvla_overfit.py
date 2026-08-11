"""Accept the fixed-sample SmolVLA overfit and explicit-resume gate."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs/vla/smolvla_450m_aloha_insertion.yaml"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import resume_smolvla_overfit as resume_runner  # noqa: E402
import run_smolvla_phase as phase_runner  # noqa: E402

from rosetta_reality.experiment import file_sha256  # noqa: E402
from rosetta_reality.features import create_json  # noqa: E402


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


def _relative_evidence(path: Path, run_root: Path) -> str:
    candidate = path.resolve()
    if not candidate.is_relative_to(run_root):
        raise ValueError("Acceptance evidence must remain inside the mounted run root.")
    if not candidate.is_file():
        raise FileNotFoundError(candidate.name)
    return candidate.relative_to(run_root).as_posix()


def _validate_smoke_acceptance(
    report_path: Path,
    experiment: dict[str, Any],
    config_path: Path,
    contract_sha256: str,
) -> dict[str, Any]:
    report = _load_json(report_path.resolve())
    acceptance = report.get("acceptance", {})
    reloads = report.get("reloads", {})
    if (
        report.get("status") != "passed"
        or report.get("stage") != "smolvla_tiny_smoke_acceptance"
        or report.get("experiment_id") != experiment["experiment_id"]
        or report.get("experiment_config_sha256") != file_sha256(config_path)
        or report.get("action_contract_sha256") != contract_sha256
        or not isinstance(acceptance, dict)
        or any(
            acceptance.get(criterion) is not True
            for criterion in experiment["phases"]["smoke"]["acceptance"]
        )
        or acceptance.get("hidden_test_loaded") is not False
        or not isinstance(reloads, dict)
        or reloads.get("checkpoint_step") != experiment["phases"]["smoke"]["steps"]
    ):
        raise ValueError("The prerequisite tiny-smoke acceptance is invalid.")
    fixed_input_loss = reloads.get("fixed_input_loss")
    if isinstance(fixed_input_loss, bool) or not isinstance(fixed_input_loss, int | float):
        raise ValueError("Tiny-smoke acceptance has no numeric fixed-input reload loss.")
    return {
        "run_name": report["smoke_run_name"],
        "checkpoint": reloads["checkpoint"],
        "fixed_input_loss": float(fixed_input_loss),
    }


def _validate_overfit_reloads(
    report_paths: list[Path],
    experiment: dict[str, Any],
    config_path: Path,
    contract_sha256: str,
    run_name: str,
) -> dict[str, Any]:
    checkpoint_root = phase_runner._absolute_root("ROSETTA_CHECKPOINT_ROOT")
    overfit = experiment["phases"]["overfit"]
    step_dir = (
        checkpoint_root
        / str(experiment["experiment_id"])
        / "overfit"
        / run_name
        / "checkpoints"
        / f"{int(overfit['steps']):06d}"
    )
    resume_runner._validate_reloads(
        report_paths,
        experiment,
        config_path,
        contract_sha256,
        step_dir,
        int(overfit["steps"]),
    )
    reports = [_load_json(path.resolve()) for path in report_paths]
    reference = reports[0]
    return {
        "count": len(reports),
        "run_name": run_name,
        "checkpoint": reference["checkpoint"],
        "checkpoint_step": reference["checkpoint_step"],
        "checkpoint_hashes": reference["checkpoint_hashes"],
        "fixed_input": reference["fixed_input"],
        "fixed_input_loss": float(reference["fixed_input_loss"]),
        "action_chunk": reference["action_chunk"],
        "verification_script_sha256": reference["verification_script_sha256"],
    }


def _validate_resume(
    report_path: Path,
    experiment: dict[str, Any],
    config_path: Path,
    contract_sha256: str,
    overfit_reload: dict[str, Any],
) -> dict[str, Any]:
    report = _load_json(report_path.resolve())
    acceptance = report.get("acceptance", {})
    runtime = report.get("runtime_observation", {})
    tracking = report.get("tracking", {})
    if (
        report.get("status") != "passed"
        or report.get("stage") != "smolvla_explicit_resume_verification"
        or report.get("experiment_id") != experiment["experiment_id"]
        or report.get("experiment_config_sha256") != file_sha256(config_path)
        or report.get("action_contract_sha256") != contract_sha256
        or report.get("verification_script_sha256")
        != file_sha256(REPOSITORY_ROOT / "scripts/verify_smolvla_resume.py")
        or report.get("resume_runner_sha256")
        != file_sha256(REPOSITORY_ROOT / "scripts/resume_smolvla_overfit.py")
        or report.get("source_checkpoint") != overfit_reload["checkpoint"]
        or report.get("source_step") != overfit_reload["checkpoint_step"]
        or report.get("episodes_loaded") != experiment["phases"]["overfit"]["episodes"]
        or report.get("hidden_test_loaded") is not False
        or report.get("network_disabled") is not True
        or not isinstance(acceptance, dict)
        or any(
            acceptance.get(criterion) is not True
            for criterion in (
                "source_checkpoint_complete",
                "optimizer_scheduler_rng_restored",
                "exactly_one_optimizer_step_completed",
                "checkpoint_written",
                "trackio_run_readable",
                "no_resource_limit_violation",
            )
        )
        or acceptance.get("hidden_test_loaded") is not False
        or runtime.get("container_exit_code") != 0
        or runtime.get("oom_event_count") != 0
        or tracking.get("test_split_loaded") is not False
    ):
        raise ValueError("The explicit-resume verification report is invalid.")
    return {
        "run_name": report["run_name"],
        "source_checkpoint": report["source_checkpoint"],
        "source_step": report["source_step"],
        "resumed_checkpoint": report["resumed_checkpoint"],
        "target_step": report["saved_resume"]["target_step"],
        "train_metrics": tracking["train_metrics"],
        "state_progression": report["state_progression"],
        "runtime_observation": runtime,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overfit-run-name", required=True)
    parser.add_argument("--smoke-acceptance-report", type=Path, required=True)
    parser.add_argument("--overfit-reload-report", type=Path, action="append", required=True)
    parser.add_argument("--resume-verification-report", type=Path, required=True)
    parser.add_argument("--overfit-container-exit-code", type=int, required=True)
    parser.add_argument("--overfit-oom-event-count", type=int, required=True)
    args = parser.parse_args()
    if not resume_runner.RUN_NAME_PATTERN.fullmatch(args.overfit_run_name):
        raise ValueError("--overfit-run-name must be a lower-case path-safe identifier.")
    if args.overfit_container_exit_code != 0 or args.overfit_oom_event_count != 0:
        raise RuntimeError("The observed overfit container did not exit cleanly within budget.")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("HF_DATASETS_OFFLINE") != "1":
        raise RuntimeError("Overfit acceptance must run with networking disabled.")

    config_path = args.config.resolve()
    experiment = _load_yaml(config_path)
    contract_path = REPOSITORY_ROOT / str(experiment["action_contract"]["derived"])
    contract_sha256 = file_sha256(contract_path)
    smoke = _validate_smoke_acceptance(
        args.smoke_acceptance_report, experiment, config_path, contract_sha256
    )
    overfit = _validate_overfit_reloads(
        args.overfit_reload_report,
        experiment,
        config_path,
        contract_sha256,
        args.overfit_run_name,
    )
    tracking = resume_runner._validate_overfit_trackio(experiment, overfit["run_name"])
    resume = _validate_resume(
        args.resume_verification_report,
        experiment,
        config_path,
        contract_sha256,
        overfit,
    )
    if overfit["fixed_input"] != {"noise": "zeros", "flow_time": 0.5}:
        raise ValueError("Overfit reload did not use the registered fixed input.")
    if overfit["fixed_input_loss"] >= smoke["fixed_input_loss"]:
        raise ValueError("Overfit checkpoint did not improve the comparable fixed-input loss.")

    run_root = phase_runner._absolute_root("ROSETTA_RUN_ROOT")
    evidence_paths = [
        args.smoke_acceptance_report,
        *args.overfit_reload_report,
        args.resume_verification_report,
    ]
    report = {
        "schema_version": 1,
        "status": "passed",
        "stage": "smolvla_fixed_sample_overfit_acceptance",
        "experiment_id": experiment["experiment_id"],
        "experiment_config_sha256": file_sha256(config_path),
        "action_contract_sha256": contract_sha256,
        "acceptance_script_sha256": file_sha256(Path(__file__)),
        "evidence": [_relative_evidence(path, run_root) for path in evidence_paths],
        "fixed_sample": {
            "episodes": experiment["phases"]["overfit"]["episodes"],
            "hidden_test_loaded": False,
        },
        "tracking": {
            **tracking,
            "space_id": experiment["tracking"]["space_id"],
            "public_payload_only": True,
            "test_split_loaded": False,
        },
        "comparable_reload": {
            "smoke": smoke,
            "overfit": overfit,
            "loss_ratio_overfit_to_smoke": overfit["fixed_input_loss"] / smoke["fixed_input_loss"],
        },
        "resume": resume,
        "runtime_observation": {
            "source": "orchestrated_container_exit_and_docker_events",
            "container_exit_code": args.overfit_container_exit_code,
            "oom_event_count": args.overfit_oom_event_count,
            "memory_limit": experiment["resources"]["memory_limit"],
            "memory_swap_limit": experiment["resources"]["memory_swap_limit"],
        },
        "acceptance": {
            "final_loss_below_initial_loss": tracking["final_loss"] < tracking["initial_loss"],
            "fixed_input_loss_below_smoke": overfit["fixed_input_loss"] < smoke["fixed_input_loss"],
            "checkpoint_reload_contract_matches": True,
            "explicit_resume_completes": True,
            "fixed_samples_only": True,
            "trackio_run_readable": True,
            "no_resource_limit_violation": True,
            "hidden_test_loaded": False,
        },
    }
    required = experiment["phases"]["overfit"]["acceptance"]
    if any(report["acceptance"].get(criterion) is not True for criterion in required):
        raise ValueError("Registered overfit acceptance criteria are not all satisfied.")
    json.dumps(report, allow_nan=False)
    acceptance_root = run_root / str(experiment["experiment_id"]) / "acceptance"
    destination = acceptance_root / f"{overfit['run_name']}.json"
    create_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report: {destination.relative_to(REPOSITORY_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
