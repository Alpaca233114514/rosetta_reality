"""Launch a gated SmolVLA action-repair preflight, smoke, or overfit phase."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import run_smolvla_phase as phase_runner  # noqa: E402

from rosetta_reality.experiment import file_sha256, stable_hash  # noqa: E402
from rosetta_reality.vla import (  # noqa: E402
    load_smolvla_action_space,
    load_smolvla_experiment,
)
from rosetta_reality.vla.fixed_samples import load_fixed_frame_protocol  # noqa: E402
from rosetta_reality.vla.processor import BOUNDED_SINE_ACTION_ADAPTER  # noqa: E402

DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "configs/vla/smolvla_450m_aloha_insertion_action_repair_fixed_overfit_002.yaml"
)
RUN_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{2,79}")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}.")
    json.dumps(value, allow_nan=False)
    return value


def _required_path(value: Path | None, flag: str) -> Path:
    if value is None:
        raise ValueError(f"{flag} is required for an optimizer phase.")
    path = value.resolve()
    if not path.is_file():
        raise FileNotFoundError(path.name)
    return path


def _validate_repair_evidence(
    experiment: dict[str, Any],
    config_path: Path,
    normalization_path: Path,
    diagnostic_path: Path,
) -> Path:
    action_space = load_smolvla_action_space(experiment, require_explicit=True)
    normalization = _load_json(normalization_path)
    diagnostic = _load_json(diagnostic_path)
    contract_path = REPOSITORY_ROOT / str(experiment["action_contract"]["derived"])
    if (
        normalization.get("status") != "complete"
        or normalization.get("stage") != "smolvla_train_only_normalization"
        or normalization.get("experiment_id") != experiment["experiment_id"]
        or normalization.get("experiment_config_sha256") != file_sha256(config_path)
        or normalization.get("action_contract_sha256") != file_sha256(contract_path)
        or normalization.get("action_space") != action_space.as_dict()
        or normalization.get("target_projection", {}).get("mode")
        != "action_contract_clip"
        or normalization.get("target_projection", {}).get("stage")
        != "before_normalization"
        or normalization.get("validation_episodes_loaded") is not False
        or normalization.get("hidden_test_loaded") is not False
        or diagnostic.get("status") != "passed"
        or diagnostic.get("stage") != "smolvla_action_space_no_weights_diagnostic"
        or diagnostic.get("experiment_id") != experiment["experiment_id"]
        or diagnostic.get("experiment_config_sha256") != file_sha256(config_path)
        or diagnostic.get("normalization_report_sha256")
        != file_sha256(normalization_path)
        or diagnostic.get("action_contract_sha256") != file_sha256(contract_path)
        or diagnostic.get("action_space") != action_space.as_dict()
        or diagnostic.get("round_trip", {}).get("passed") is not True
        or (
            action_space.representation_adapter == BOUNDED_SINE_ACTION_ADAPTER
            and diagnostic.get("bounded_gripper_output", {}).get("passed") is not True
        )
        or diagnostic.get("optimizer_created") is not False
        or diagnostic.get("hidden_test_loaded") is not False
    ):
        raise ValueError("Action-repair statistics or adapter evidence has a different identity.")
    run_root = phase_runner._absolute_root("ROSETTA_RUN_ROOT").resolve()
    relative_view = Path(str(normalization.get("dataset_view", "")))
    if relative_view.is_absolute() or ".." in relative_view.parts:
        raise ValueError("Repair dataset view path is unsafe.")
    dataset_root = (run_root / relative_view).resolve()
    if not dataset_root.is_relative_to(run_root) or not dataset_root.is_dir():
        raise ValueError("Repair dataset view is missing or outside the run root.")
    return dataset_root


def _validate_historical_diagnostic(
    experiment: dict[str, Any], report_path: Path, contract_sha256: str
) -> None:
    protocol = experiment["repair_protocol"]
    registered = protocol.get("historical_diagnostic", {})
    registered_path = REPOSITORY_ROOT / str(registered.get("report", ""))
    report = _load_json(report_path)
    normal = report.get("conditions", {}).get("normal", {})
    per_dimension = report.get("normal_action_diagnostics", {})
    right_gripper = per_dimension.get("dimensions", {}).get("right_gripper", {})
    if (
        report_path != registered_path.resolve()
        or file_sha256(report_path) != registered.get("sha256")
        or report.get("status") != "complete"
        or report.get("stage") != "smolvla_teacher_forced_modality_diagnostic"
        or report.get("experiment_id") != registered.get("experiment_id")
        or report.get("checkpoint_step") != registered.get("checkpoint_step")
        or report.get("action_contract_sha256") != contract_sha256
        or report.get("dataset_revision") != experiment["dataset"]["revision"]
        or report.get("optimizer_created") is not False
        or report.get("gradients_enabled") is not False
        or report.get("hidden_test_loaded") is not False
        or not isinstance(normal.get("chunk_mae"), int | float)
        or not isinstance(right_gripper.get("mae"), int | float)
        or set(report.get("conditions", {}))
        != {"normal", "image_shuffle", "image_zero", "state_shuffle"}
    ):
        raise ValueError("The pinned historical modality diagnostic is incomplete or changed.")


def _validate_fixed_sample_evidence(
    path: Path,
    experiment: dict[str, Any],
    config_path: Path,
    phase: str,
) -> None:
    report = _load_json(path)
    protocol = load_fixed_frame_protocol(experiment, phase)
    payload = protocol.as_dict()
    if (
        report.get("status") != "passed"
        or report.get("stage") != "smolvla_fixed_sample_no_weights_diagnostic"
        or report.get("experiment_id") != experiment["experiment_id"]
        or report.get("experiment_config_sha256") != file_sha256(config_path)
        or report.get("dataset_revision") != experiment["dataset"]["revision"]
        or report.get("fixed_sample_protocol") != payload
        or report.get("fixed_sample_protocol_sha256") != stable_hash(payload)
        or report.get("fixed_sample_count") != len(protocol.frame_indices)
        or report.get("episodes_loaded") != [protocol.episode]
        or report.get("model_weights_loaded") is not False
        or report.get("optimizer_created") is not False
        or report.get("validation_episodes_loaded") is not False
        or report.get("hidden_test_loaded") is not False
    ):
        raise ValueError("The fixed-sample diagnostic has a different identity.")


def _validate_tracking_reuse(path: Path, experiment: dict[str, Any]) -> None:
    report = _load_json(path)
    tracking = experiment["tracking"]
    if (
        report.get("status") != "complete"
        or report.get("project") != tracking["project"]
        or report.get("space_id") != tracking["space_id"]
        or report.get("space_sdk") != "static"
        or report.get("visibility") != "public"
        or report.get("contains_sensitive_data") is not False
        or report.get("media_uploaded") is not False
        or report.get("test_split_loaded") is not False
    ):
        raise ValueError("The reusable Trackio Space evidence is incomplete or unsafe.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "smoke", "overfit"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--normalization-report", type=Path, required=True)
    parser.add_argument("--action-space-report", type=Path, required=True)
    parser.add_argument("--benchmark-report", type=Path)
    parser.add_argument("--gate1-report", type=Path)
    parser.add_argument("--gate2-report", type=Path)
    parser.add_argument("--trackio-report", type=Path)
    parser.add_argument("--historical-modality-report", type=Path)
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--smoke-acceptance-report", type=Path)
    parser.add_argument("--fixed-sample-report", type=Path)
    args = parser.parse_args()
    if not RUN_NAME_PATTERN.fullmatch(args.run_name):
        raise ValueError("--run-name must be lower-case and path safe.")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("HF_DATASETS_OFFLINE") != "1":
        raise RuntimeError("SmolVLA action-repair phases require networking disabled.")

    config_path = args.config.resolve()
    experiment = load_smolvla_experiment(config_path, REPOSITORY_ROOT)
    action_space = load_smolvla_action_space(experiment, require_explicit=True)
    protocol = experiment.get("repair_protocol", {})
    if (
        experiment.get("status") != "preregistered_action_repair_smoke_and_overfit"
        or protocol.get("hidden_test_loaded") is not False
        or protocol.get("historical_checkpoints_are_initialization") is not False
    ):
        raise ValueError("Only the preregistered fresh action-repair experiment may run.")
    if args.phase != "preflight" and (
        protocol.get("optimizer_authorized") is not True
        or args.phase not in protocol.get("authorized_phases", [])
    ):
        raise PermissionError("This action-repair optimizer phase is not registered.")
    resources = experiment["resources"]
    if (
        os.environ.get("ROSETTA_DOCKER_MEMORY_LIMIT") != resources["memory_limit"]
        or os.environ.get("ROSETTA_DOCKER_MEMORY_SWAP_LIMIT")
        != resources["memory_swap_limit"]
    ):
        raise ValueError("Active Docker limits differ from the repair experiment.")

    normalization_path = args.normalization_report.resolve()
    diagnostic_path = args.action_space_report.resolve()
    dataset_root = _validate_repair_evidence(
        experiment,
        config_path,
        normalization_path,
        diagnostic_path,
    )
    contract_path = REPOSITORY_ROOT / str(experiment["action_contract"]["derived"])
    contract_sha256 = file_sha256(contract_path)
    if args.phase != "preflight":
        fixed_sample_path = _required_path(
            args.fixed_sample_report, "--fixed-sample-report"
        )
        _validate_fixed_sample_evidence(
            fixed_sample_path,
            experiment,
            config_path,
            args.phase,
        )
        benchmark_path = _required_path(args.benchmark_report, "--benchmark-report")
        gate1_path = _required_path(args.gate1_report, "--gate1-report")
        gate2_path = _required_path(args.gate2_report, "--gate2-report")
        tracking_path = _required_path(args.trackio_report, "--trackio-report")
        modality_path = _required_path(
            args.historical_modality_report, "--historical-modality-report"
        )
        preflight_path = _required_path(args.preflight_report, "--preflight-report")
        phase_runner._validate_benchmark(
            benchmark_path, experiment, config_path, contract_sha256
        )
        phase_runner._validate_gate(
            gate1_path,
            expected_gate="m2_gate_1_scripted_action",
            experiment_id=str(experiment["experiment_id"]),
            contract_sha256=contract_sha256,
            dataset_revision=str(experiment["dataset"]["revision"]),
        )
        phase_runner._validate_gate(
            gate2_path,
            expected_gate="m2_gate_2_dataset_action_replay",
            experiment_id=str(experiment["experiment_id"]),
            contract_sha256=contract_sha256,
            dataset_revision=str(experiment["dataset"]["revision"]),
        )
        _validate_tracking_reuse(tracking_path, experiment)
        _validate_historical_diagnostic(experiment, modality_path, contract_sha256)
        phase_runner._validate_preflight(
            preflight_path, experiment, config_path, contract_sha256
        )
        if args.phase == "overfit":
            smoke_path = _required_path(
                args.smoke_acceptance_report, "--smoke-acceptance-report"
            )
            phase_runner._validate_smoke_acceptance(
                smoke_path, experiment, config_path, contract_sha256
            )

    model_root = phase_runner._model_root(experiment)
    checkpoint_root = phase_runner._absolute_root("ROSETTA_CHECKPOINT_ROOT")
    output_dir = (
        checkpoint_root
        / str(experiment["experiment_id"])
        / args.phase
        / str(args.run_name)
    )
    if output_dir.exists():
        raise FileExistsError("The repair phase output exists; choose a new run name.")
    arguments = phase_runner._phase_arguments(
        experiment,
        args.phase,
        str(args.run_name),
        model_root,
        dataset_root,
        output_dir,
    )
    arguments.append(
        f"--policy.adapt_to_pi_aloha={str(action_space.adapt_to_pi_aloha).lower()}"
    )
    os.environ["ROSETTA_VLA_PHASE"] = (
        "action_repair_preflight" if args.phase == "preflight" else args.phase
    )
    os.environ["ROSETTA_VLA_EXPERIMENT_CONFIG"] = str(config_path)
    os.environ["ROSETTA_VLA_RUN_NAME"] = str(args.run_name)
    os.environ["ROSETTA_VLA_TRAIN_STATS_REPORT"] = str(normalization_path)
    os.environ["ROSETTA_VLA_NORMALIZATION_SHA256"] = file_sha256(normalization_path)
    os.environ["ROSETTA_VLA_REPAIR_PROTOCOL_SHA256"] = file_sha256(diagnostic_path)
    if args.phase != "preflight":
        os.environ["ROSETTA_VLA_FIXED_SAMPLE_REPORT"] = str(fixed_sample_path)
        os.environ["ROSETTA_VLA_FIXED_SAMPLE_SHA256"] = file_sha256(
            fixed_sample_path
        )
    os.environ.pop("ROSETTA_VLA_FORMAL_PLAN_SHA256", None)
    sys.argv = ["lerobot-train", *arguments]
    if args.phase == "preflight":
        from smolvla_forward_check import main as preflight_main

        return preflight_main()
    os.environ["ROSETTA_VLA_ACTION_REPAIR_OPTIMIZER_AUTHORIZED"] = "1"
    from train_smolvla_action_repair import main as train_main

    train_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
