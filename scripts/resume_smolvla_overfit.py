"""Resume the accepted fixed-sample SmolVLA overfit checkpoint for one controlled step."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs/vla/smolvla_450m_aloha_insertion.yaml"
RUN_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{2,79}")
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import run_smolvla_phase as phase_runner  # noqa: E402

from rosetta_reality.experiment import file_sha256  # noqa: E402


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


def _decode_json(value: Any) -> dict[str, Any]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("Trackio stored a non-object payload.")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} is not numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite.")
    return result


def _source_checkpoint(
    checkpoint: Path,
    experiment: dict[str, Any],
) -> tuple[Path, Path, int, str]:
    checkpoint_root = phase_runner._absolute_root("ROSETTA_CHECKPOINT_ROOT")
    step_dir = checkpoint.resolve()
    if step_dir.name == "pretrained_model":
        step_dir = step_dir.parent
    if not step_dir.is_relative_to(checkpoint_root):
        raise ValueError("Resume source must remain inside the mounted checkpoint root.")
    relative = step_dir.relative_to(checkpoint_root)
    if (
        len(relative.parts) != 5
        or relative.parts[0] != experiment["experiment_id"]
        or relative.parts[1] != "overfit"
        or relative.parts[3] != "checkpoints"
        or not relative.parts[4].isdigit()
    ):
        raise ValueError("Resume source does not match the registered overfit checkpoint layout.")
    run_name = relative.parts[2]
    if not RUN_NAME_PATTERN.fullmatch(run_name):
        raise ValueError("Resume source run name is not path-safe.")
    last = step_dir.parent / "last"
    if not last.is_symlink() or last.resolve() != step_dir:
        raise ValueError("Resume source must be the immutable last overfit checkpoint.")
    pretrained_dir = step_dir / "pretrained_model"
    training_state_dir = step_dir / "training_state"
    train_config = _load_json(pretrained_dir / "train_config.json")
    training_step = _load_json(training_state_dir / "training_step.json")
    overfit = experiment["phases"]["overfit"]
    dataset = train_config.get("dataset", {})
    policy = train_config.get("policy", {})
    step = training_step.get("step")
    required_files = (
        pretrained_dir / "model.safetensors",
        training_state_dir / "optimizer_state.safetensors",
        training_state_dir / "rng_state.safetensors",
        training_state_dir / "scheduler_state.json",
    )
    if any(not path.is_file() or path.stat().st_size <= 0 for path in required_files):
        raise FileNotFoundError(
            "Resume source is missing model, optimizer, scheduler, or RNG state."
        )
    if (
        dataset.get("repo_id") != experiment["dataset"]["identifier"]
        or dataset.get("revision") != experiment["dataset"]["revision"]
        or dataset.get("episodes") != overfit["episodes"]
        or train_config.get("rename_map") != experiment["dataset"]["rename_map"]
        or train_config.get("seed") != experiment["seed"]
        or train_config.get("batch_size") != overfit["batch_size"]
        or train_config.get("steps") != overfit["steps"]
        or train_config.get("save_freq") != overfit["save_freq"]
        or train_config.get("resume") is not False
        or policy.get("type") != "smolvla"
        or policy.get("pretrained_revision") != experiment["model"]["revision"]
        or step != overfit["steps"]
        or training_step.get("batch_size") != overfit["batch_size"]
        or training_step.get("dp_world_size") != 1
        or Path(str(train_config.get("output_dir"))).resolve() != step_dir.parents[1]
    ):
        raise ValueError("Resume source identity differs from the registered overfit run.")
    return step_dir, pretrained_dir, int(step), run_name


def _validate_overfit_trackio(
    experiment: dict[str, Any],
    run_name: str,
) -> dict[str, Any]:
    trackio_root = phase_runner._absolute_root("TRACKIO_DIR")
    database = trackio_root / f"{experiment['tracking']['project']}.db"
    if not database.is_file():
        raise FileNotFoundError("The durable local Trackio database is missing.")
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        config_rows = connection.execute(
            "SELECT run_id, config FROM configs WHERE run_name = ?",
            (run_name,),
        ).fetchall()
        if len(config_rows) != 1:
            raise ValueError("Trackio must contain exactly one config for the overfit run.")
        run_id, raw_config = config_rows[0]
        public_config = _decode_json(raw_config)
        metric_rows = connection.execute(
            "SELECT step, metrics FROM metrics WHERE run_id = ? AND run_name = ? ORDER BY id",
            (run_id, run_name),
        ).fetchall()
    finally:
        connection.close()
    overfit = experiment["phases"]["overfit"]
    if (
        public_config.get("experiment_id") != experiment["experiment_id"]
        or public_config.get("phase") != "overfit"
        or public_config.get("model_revision") != experiment["model"]["revision"]
        or public_config.get("dataset_revision") != experiment["dataset"]["revision"]
        or public_config.get("seed") != experiment["seed"]
        or public_config.get("batch_size") != overfit["batch_size"]
        or public_config.get("steps") != overfit["steps"]
        or public_config.get("train_episode_count") != len(overfit["episodes"])
        or public_config.get("test_split_loaded") is not False
    ):
        raise ValueError("Trackio overfit config identity is invalid.")
    train_by_step: dict[int, float] = {}
    checkpoint_steps: set[int] = set()
    for raw_step, raw_metrics in metric_rows:
        metrics = _decode_json(raw_metrics)
        step = int(raw_step)
        if "train/loss" in metrics:
            train_by_step[step] = _finite_number(metrics["train/loss"], f"step {step} loss")
        if metrics.get("system/checkpoint_saved") == 1:
            checkpoint_steps.add(step)
    expected_steps = set(range(1, int(overfit["steps"]) + 1))
    expected_checkpoints = {int(overfit["save_freq"]), int(overfit["steps"])}
    if set(train_by_step) != expected_steps or checkpoint_steps != expected_checkpoints:
        raise ValueError("Trackio does not contain the complete registered overfit trajectory.")
    initial_loss = train_by_step[1]
    final_loss = train_by_step[int(overfit["steps"])]
    if final_loss >= initial_loss:
        raise ValueError("The fixed-sample overfit run did not lower its logged loss.")
    return {
        "run_id": str(run_id),
        "steps": len(train_by_step),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "checkpoint_steps": sorted(checkpoint_steps),
    }


def _validate_reloads(
    reports: list[Path],
    experiment: dict[str, Any],
    config_path: Path,
    contract_sha256: str,
    step_dir: Path,
    step: int,
) -> None:
    if len(reports) < 2:
        raise ValueError("Resume requires two independent overfit reload reports.")
    checkpoint_root = phase_runner._absolute_root("ROSETTA_CHECKPOINT_ROOT")
    expected_checkpoint = step_dir.relative_to(checkpoint_root).as_posix()
    expected_script_sha256 = file_sha256(REPOSITORY_ROOT / "scripts/verify_smolvla_checkpoint.py")
    values = [_load_json(path.resolve()) for path in reports]
    for report in values:
        if (
            report.get("status") != "passed"
            or report.get("stage") != "smolvla_checkpoint_independent_reload"
            or report.get("phase") != "overfit"
            or report.get("experiment_id") != experiment["experiment_id"]
            or report.get("experiment_config_sha256") != file_sha256(config_path)
            or report.get("verification_script_sha256") != expected_script_sha256
            or report.get("action_contract_sha256") != contract_sha256
            or report.get("checkpoint") != expected_checkpoint
            or report.get("checkpoint_step") != step
            or report.get("episodes_loaded") != experiment["phases"]["overfit"]["episodes"]
            or report.get("hidden_test_loaded") is not False
            or report.get("network_disabled") is not True
        ):
            raise ValueError("An overfit reload report is not bound to the resume source.")
    comparable = (
        "checkpoint_hashes",
        "parameters",
        "fixed_input",
        "fixed_input_loss",
        "loss_details",
        "action_chunk",
    )
    reference = values[0]
    for report in values[1:]:
        for field in comparable:
            if report.get(field) != reference.get(field):
                raise ValueError(f"Independent overfit reloads differ in field: {field}.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--benchmark-report", type=Path, required=True)
    parser.add_argument("--gate1-report", type=Path, required=True)
    parser.add_argument("--gate2-report", type=Path, required=True)
    parser.add_argument("--trackio-report", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--smoke-acceptance-report", type=Path, required=True)
    parser.add_argument("--reload-report", type=Path, action="append", required=True)
    args = parser.parse_args()
    if not RUN_NAME_PATTERN.fullmatch(args.run_name):
        raise ValueError("--run-name must be a lower-case path-safe identifier.")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("HF_DATASETS_OFFLINE") != "1":
        raise RuntimeError("Resume must run with networking disabled.")

    config_path = args.config.resolve()
    experiment = _load_yaml(config_path)
    resources = experiment["resources"]
    if (
        os.environ.get("ROSETTA_DOCKER_MEMORY_LIMIT") != resources["memory_limit"]
        or os.environ.get("ROSETTA_DOCKER_MEMORY_SWAP_LIMIT") != resources["memory_swap_limit"]
    ):
        raise ValueError("The active Docker memory limits differ from the preregistered budget.")
    contract_path = REPOSITORY_ROOT / str(experiment["action_contract"]["derived"])
    contract_sha256 = file_sha256(contract_path)
    phase_runner._validate_benchmark(
        args.benchmark_report.resolve(), experiment, config_path, contract_sha256
    )
    phase_runner._validate_gate(
        args.gate1_report.resolve(),
        expected_gate="m2_gate_1_scripted_action",
        experiment_id=experiment["experiment_id"],
        contract_sha256=contract_sha256,
        dataset_revision=experiment["dataset"]["revision"],
    )
    phase_runner._validate_gate(
        args.gate2_report.resolve(),
        expected_gate="m2_gate_2_dataset_action_replay",
        experiment_id=experiment["experiment_id"],
        contract_sha256=contract_sha256,
        dataset_revision=experiment["dataset"]["revision"],
    )
    phase_runner._validate_tracking(args.trackio_report.resolve(), experiment)
    phase_runner._validate_preflight(
        args.preflight_report.resolve(), experiment, config_path, contract_sha256
    )
    phase_runner._validate_smoke_acceptance(
        args.smoke_acceptance_report.resolve(), experiment, config_path, contract_sha256
    )
    step_dir, pretrained_dir, source_step, source_run_name = _source_checkpoint(
        args.source_checkpoint, experiment
    )
    _validate_overfit_trackio(experiment, source_run_name)
    _validate_reloads(
        args.reload_report,
        experiment,
        config_path,
        contract_sha256,
        step_dir,
        source_step,
    )

    checkpoint_root = phase_runner._absolute_root("ROSETTA_CHECKPOINT_ROOT")
    output_dir = (
        checkpoint_root / str(experiment["experiment_id"]) / "overfit_resume" / args.run_name
    )
    if output_dir.exists():
        raise FileExistsError("The requested resume output already exists; choose a new run name.")
    target_step = source_step + 1
    os.environ["ROSETTA_VLA_PHASE"] = "overfit_resume"
    os.environ["ROSETTA_VLA_EXPERIMENT_CONFIG"] = str(config_path)
    os.environ["ROSETTA_VLA_RUN_NAME"] = args.run_name
    os.environ["ROSETTA_VLA_RESUME_SOURCE_RUN"] = source_run_name
    os.environ["ROSETTA_VLA_RESUME_FROM_STEP"] = str(source_step)
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
    from train_smolvla_trackio import main as train_main

    train_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
