"""Accept repair overfit, deterministic reloads, aggregate gripper quality, and resume."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import accept_smolvla_overfit as legacy_accept  # noqa: E402
import resume_smolvla_overfit as legacy_resume  # noqa: E402
import run_smolvla_phase as phase_runner  # noqa: E402

from rosetta_reality.experiment import file_sha256  # noqa: E402
from rosetta_reality.features import create_json  # noqa: E402
from rosetta_reality.vla import load_smolvla_experiment  # noqa: E402

DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "configs/vla/smolvla_450m_aloha_insertion_action_repair_fixed_overfit_002.yaml"
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}.")
    json.dumps(value, allow_nan=False)
    return value


def _gripper_diagnostics(report_path: Path) -> dict[str, Any]:
    report = _load_json(report_path.resolve())
    context = report.get("fixed_sample_context", {})
    raw = report.get("raw_standard_action_diagnostics", {})
    dimensions = raw.get("dimensions", {})
    internal_support = raw.get("model_internal_gripper_support", {})
    if context.get("scope") != "all" or report.get("evaluation_batch_size") != 8:
        raise ValueError("Comparable gripper diagnostics must cover all eight fixed anchors.")
    sides: dict[str, dict[str, float]] = {}
    for name in ("left_gripper", "right_gripper"):
        values = dimensions.get(name, {})
        support = internal_support.get(name, {})
        parsed: dict[str, float] = {}
        for key in (
            "mae",
            "first_action_mae",
            "prediction_strict_violation_rate",
            "predicted_below_minimum_rate",
            "predicted_above_maximum_rate",
            "open_close_accuracy",
        ):
            value = values.get(key)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"Reload report has no numeric {name} {key}.")
            parsed[key] = float(value)
        support_rate = support.get("outside_training_support_rate")
        if isinstance(support_rate, bool) or not isinstance(support_rate, int | float):
            raise ValueError(f"Reload report has no numeric {name} internal support rate.")
        parsed["internal_outside_training_support_rate"] = float(support_rate)
        sides[name] = parsed
    return {
        "sides": sides,
        "aggregate_mae": sum(side["mae"] for side in sides.values()) / len(sides),
        "aggregate_open_close_accuracy": sum(
            side["open_close_accuracy"] for side in sides.values()
        )
        / len(sides),
    }


def _validate_resume(
    report_path: Path,
    experiment: dict[str, Any],
    config_path: Path,
    contract_sha256: str,
    overfit: dict[str, Any],
) -> dict[str, Any]:
    report = _load_json(report_path.resolve())
    acceptance = report.get("acceptance", {})
    runtime = report.get("runtime_observation", {})
    tracking = report.get("tracking", {})
    if (
        report.get("status") != "passed"
        or report.get("stage")
        != "smolvla_action_repair_explicit_resume_verification"
        or report.get("experiment_id") != experiment["experiment_id"]
        or report.get("experiment_config_sha256") != file_sha256(config_path)
        or report.get("action_contract_sha256") != contract_sha256
        or report.get("verification_script_sha256")
        != file_sha256(
            REPOSITORY_ROOT / "scripts/verify_smolvla_action_repair_resume.py"
        )
        or report.get("resume_runner_sha256")
        != file_sha256(REPOSITORY_ROOT / "scripts/resume_smolvla_action_repair.py")
        or report.get("source_checkpoint") != overfit["checkpoint"]
        or report.get("source_step") != overfit["checkpoint_step"]
        or report.get("episodes_loaded") != experiment["phases"]["overfit"]["episodes"]
        or report.get("serialized_action_boundary", {}).get("preserved") is not True
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
                "action_boundary_preserved",
                "trackio_run_readable",
                "no_resource_limit_violation",
            )
        )
        or acceptance.get("hidden_test_loaded") is not False
        or runtime.get("container_exit_code") != 0
        or runtime.get("oom_event_count") != 0
        or tracking.get("test_split_loaded") is not False
    ):
        raise ValueError("The repair explicit-resume verification is invalid.")
    return {
        "run_name": report["run_name"],
        "source_checkpoint": report["source_checkpoint"],
        "source_step": report["source_step"],
        "resumed_checkpoint": report["resumed_checkpoint"],
        "target_step": report["saved_resume"]["target_step"],
        "train_metrics": tracking["train_metrics"],
        "state_progression": report["state_progression"],
        "serialized_action_boundary": report["serialized_action_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overfit-run-name", required=True)
    parser.add_argument("--smoke-acceptance-report", type=Path, required=True)
    parser.add_argument("--smoke-reload-report", type=Path, required=True)
    parser.add_argument("--overfit-reload-report", type=Path, action="append", required=True)
    parser.add_argument("--resume-verification-report", type=Path, required=True)
    parser.add_argument("--overfit-container-exit-code", type=int, required=True)
    parser.add_argument("--overfit-oom-event-count", type=int, required=True)
    args = parser.parse_args()
    if not legacy_resume.RUN_NAME_PATTERN.fullmatch(args.overfit_run_name):
        raise ValueError("--overfit-run-name must be lower-case and path safe.")
    if args.overfit_container_exit_code != 0 or args.overfit_oom_event_count != 0:
        raise RuntimeError("The repair overfit container did not exit cleanly within budget.")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("HF_DATASETS_OFFLINE") != "1":
        raise RuntimeError("Repair overfit acceptance requires networking disabled.")

    config_path = args.config.resolve()
    experiment = load_smolvla_experiment(config_path, REPOSITORY_ROOT)
    contract_path = REPOSITORY_ROOT / str(experiment["action_contract"]["derived"])
    contract_sha256 = file_sha256(contract_path)
    smoke = legacy_accept._validate_smoke_acceptance(
        args.smoke_acceptance_report,
        experiment,
        config_path,
        contract_sha256,
    )
    overfit = legacy_accept._validate_overfit_reloads(
        args.overfit_reload_report,
        experiment,
        config_path,
        contract_sha256,
        args.overfit_run_name,
    )
    tracking = legacy_resume._validate_overfit_trackio(experiment, overfit["run_name"])
    resume = _validate_resume(
        args.resume_verification_report,
        experiment,
        config_path,
        contract_sha256,
        overfit,
    )
    smoke_grippers = _gripper_diagnostics(args.smoke_reload_report)
    overfit_grippers = _gripper_diagnostics(args.overfit_reload_report[0])
    if overfit["fixed_input"] != {"noise": "zeros", "flow_time": 0.5}:
        raise ValueError("Overfit reload did not use the registered fixed input.")
    if overfit["fixed_input_loss"] >= smoke["fixed_input_loss"]:
        raise ValueError("Repair overfit did not improve the comparable fixed-input loss.")
    if (
        overfit_grippers["aggregate_mae"] > smoke_grippers["aggregate_mae"]
        or overfit_grippers["aggregate_open_close_accuracy"]
        < smoke_grippers["aggregate_open_close_accuracy"]
    ):
        raise ValueError("Repair overfit regressed aggregate fixed-anchor gripper quality.")
    if any(
        side[criterion] != 0.0
        for side in overfit_grippers["sides"].values()
        for criterion in (
            "prediction_strict_violation_rate",
            "predicted_below_minimum_rate",
            "predicted_above_maximum_rate",
        )
    ):
        raise ValueError("Repair overfit emitted an illegal standard-space gripper value.")

    run_root = phase_runner._absolute_root("ROSETTA_RUN_ROOT")
    evidence_paths = [
        args.smoke_acceptance_report,
        args.smoke_reload_report,
        *args.overfit_reload_report,
        args.resume_verification_report,
    ]
    report = {
        "schema_version": 1,
        "status": "passed",
        "stage": "smolvla_action_repair_fixed_episode_overfit_acceptance",
        "experiment_id": experiment["experiment_id"],
        "experiment_config_sha256": file_sha256(config_path),
        "action_contract_sha256": contract_sha256,
        "acceptance_script_sha256": file_sha256(Path(__file__)),
        "evidence": [
            legacy_accept._relative_evidence(path, run_root) for path in evidence_paths
        ],
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
            "loss_ratio_overfit_to_smoke": (
                overfit["fixed_input_loss"] / smoke["fixed_input_loss"]
            ),
        },
        "grippers": {
            "smoke": smoke_grippers,
            "overfit": overfit_grippers,
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
            "final_loss_below_initial_loss": tracking["final_loss"]
            < tracking["initial_loss"],
            "fixed_input_loss_below_smoke": overfit["fixed_input_loss"]
            < smoke["fixed_input_loss"],
            "aggregate_gripper_not_regressed": True,
            "standard_gripper_output_legal": True,
            "checkpoint_reload_contract_matches": True,
            "explicit_resume_completes": True,
            "fixed_samples_only": True,
            "trackio_run_readable": True,
            "no_resource_limit_violation": True,
            "hidden_test_loaded": False,
        },
    }
    required = [
        *experiment["phases"]["overfit"]["acceptance"],
        "fixed_input_loss_below_smoke",
        "aggregate_gripper_not_regressed",
        "standard_gripper_output_legal",
    ]
    if any(report["acceptance"].get(criterion) is not True for criterion in required):
        raise ValueError("Repair overfit acceptance criteria are not all satisfied.")
    json.dumps(report, allow_nan=False)
    destination = (
        run_root
        / str(experiment["experiment_id"])
        / "acceptance"
        / f"{overfit['run_name']}.json"
    )
    create_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report: {destination.relative_to(REPOSITORY_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
