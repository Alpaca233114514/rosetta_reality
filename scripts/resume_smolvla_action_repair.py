"""Resume an accepted repair overfit checkpoint for one controlled optimizer step."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import resume_smolvla_overfit as legacy_resume  # noqa: E402
import run_smolvla_action_repair_phase as repair_runner  # noqa: E402
import run_smolvla_phase as phase_runner  # noqa: E402

from rosetta_reality.experiment import file_sha256  # noqa: E402
from rosetta_reality.vla import load_smolvla_experiment  # noqa: E402

DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "configs/vla/smolvla_450m_aloha_insertion_action_repair_fixed_overfit_002.yaml"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--normalization-report", type=Path, required=True)
    parser.add_argument("--action-space-report", type=Path, required=True)
    parser.add_argument("--fixed-sample-report", type=Path, required=True)
    parser.add_argument("--benchmark-report", type=Path, required=True)
    parser.add_argument("--gate1-report", type=Path, required=True)
    parser.add_argument("--gate2-report", type=Path, required=True)
    parser.add_argument("--trackio-report", type=Path, required=True)
    parser.add_argument("--historical-modality-report", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--smoke-acceptance-report", type=Path, required=True)
    parser.add_argument("--reload-report", type=Path, action="append", required=True)
    args = parser.parse_args()
    if not legacy_resume.RUN_NAME_PATTERN.fullmatch(args.run_name):
        raise ValueError("--run-name must be lower-case and path safe.")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("HF_DATASETS_OFFLINE") != "1":
        raise RuntimeError("Repair resume requires networking disabled.")

    config_path = args.config.resolve()
    experiment = load_smolvla_experiment(config_path, REPOSITORY_ROOT)
    resources = experiment["resources"]
    if (
        os.environ.get("ROSETTA_DOCKER_MEMORY_LIMIT") != resources["memory_limit"]
        or os.environ.get("ROSETTA_DOCKER_MEMORY_SWAP_LIMIT")
        != resources["memory_swap_limit"]
    ):
        raise ValueError("Active Docker limits differ from the repair experiment.")
    normalization_path = args.normalization_report.resolve()
    action_space_path = args.action_space_report.resolve()
    repair_runner._validate_repair_evidence(
        experiment,
        config_path,
        normalization_path,
        action_space_path,
    )
    fixed_sample_path = args.fixed_sample_report.resolve()
    repair_runner._validate_fixed_sample_evidence(
        fixed_sample_path,
        experiment,
        config_path,
        "overfit_resume",
    )
    contract_path = REPOSITORY_ROOT / str(experiment["action_contract"]["derived"])
    contract_sha256 = file_sha256(contract_path)
    phase_runner._validate_benchmark(
        args.benchmark_report.resolve(), experiment, config_path, contract_sha256
    )
    phase_runner._validate_gate(
        args.gate1_report.resolve(),
        expected_gate="m2_gate_1_scripted_action",
        experiment_id=str(experiment["experiment_id"]),
        contract_sha256=contract_sha256,
        dataset_revision=str(experiment["dataset"]["revision"]),
    )
    phase_runner._validate_gate(
        args.gate2_report.resolve(),
        expected_gate="m2_gate_2_dataset_action_replay",
        experiment_id=str(experiment["experiment_id"]),
        contract_sha256=contract_sha256,
        dataset_revision=str(experiment["dataset"]["revision"]),
    )
    repair_runner._validate_tracking_reuse(args.trackio_report.resolve(), experiment)
    repair_runner._validate_historical_diagnostic(
        experiment,
        args.historical_modality_report.resolve(),
        contract_sha256,
    )
    phase_runner._validate_preflight(
        args.preflight_report.resolve(), experiment, config_path, contract_sha256
    )
    phase_runner._validate_smoke_acceptance(
        args.smoke_acceptance_report.resolve(), experiment, config_path, contract_sha256
    )
    step_dir, pretrained_dir, source_step, source_run_name = legacy_resume._source_checkpoint(
        args.source_checkpoint, experiment
    )
    legacy_resume._validate_overfit_trackio(experiment, source_run_name)
    legacy_resume._validate_reloads(
        args.reload_report,
        experiment,
        config_path,
        contract_sha256,
        step_dir,
        source_step,
    )

    checkpoint_root = phase_runner._absolute_root("ROSETTA_CHECKPOINT_ROOT")
    output_dir = (
        checkpoint_root
        / str(experiment["experiment_id"])
        / "overfit_resume"
        / args.run_name
    )
    if output_dir.exists():
        raise FileExistsError("The repair resume output exists; choose a new run name.")
    target_step = source_step + 1
    os.environ["ROSETTA_VLA_PHASE"] = "overfit_resume"
    os.environ["ROSETTA_VLA_EXPERIMENT_CONFIG"] = str(config_path)
    os.environ["ROSETTA_VLA_RUN_NAME"] = args.run_name
    os.environ["ROSETTA_VLA_RESUME_SOURCE_RUN"] = source_run_name
    os.environ["ROSETTA_VLA_RESUME_FROM_STEP"] = str(source_step)
    os.environ["ROSETTA_VLA_TRAIN_STATS_REPORT"] = str(normalization_path)
    os.environ["ROSETTA_VLA_NORMALIZATION_SHA256"] = file_sha256(normalization_path)
    os.environ["ROSETTA_VLA_REPAIR_PROTOCOL_SHA256"] = file_sha256(action_space_path)
    os.environ["ROSETTA_VLA_FIXED_SAMPLE_REPORT"] = str(fixed_sample_path)
    os.environ["ROSETTA_VLA_FIXED_SAMPLE_SHA256"] = file_sha256(fixed_sample_path)
    os.environ["ROSETTA_VLA_ACTION_REPAIR_OPTIMIZER_AUTHORIZED"] = "1"
    sys.argv = [
        "lerobot-train",
        f"--config_path={pretrained_dir / 'train_config.json'}",
        "--resume=true",
        f"--output_dir={output_dir}",
        f"--job_name={args.run_name}",
        f"--steps={target_step}",
        f"--save_freq={target_step}",
        "--log_freq=1",
        "--eval_steps=0",
        "--env_eval_freq=0",
        "--policy.push_to_hub=false",
        "--save_checkpoint_to_hub=false",
        "--wandb.enable=true",
        "--wandb.disable_artifact=true",
        f"--wandb.project={experiment['tracking']['project']}",
    ]
    from train_smolvla_action_repair import main as train_main

    train_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
