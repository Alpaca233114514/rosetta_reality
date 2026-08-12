"""Verify one-step SmolVLA resume provenance without loading hidden-test data."""

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


def _processor_state_paths(pretrained_dir: Path) -> dict[str, Path]:
    states: dict[str, Path] = {}
    for config_name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        config = _load_json(pretrained_dir / config_name)
        steps = config.get("steps")
        if not isinstance(steps, list):
            raise ValueError(f"{config_name} has no processor step list.")
        for step in steps:
            if not isinstance(step, dict):
                raise ValueError(f"{config_name} contains an invalid processor step.")
            state_file = step.get("state_file")
            if state_file is None:
                continue
            if not isinstance(state_file, str):
                raise ValueError(f"{config_name} contains an invalid state_file.")
            relative = Path(state_file)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or relative.as_posix() != state_file
                or state_file in states
            ):
                raise ValueError(f"{config_name} contains an unsafe state_file.")
            path = pretrained_dir / relative
            if not path.is_file() or path.stat().st_size <= 0:
                raise FileNotFoundError(f"Referenced processor state is missing: {state_file}.")
            states[state_file] = path
    return states


def _resumed_checkpoint(
    checkpoint: Path,
    experiment: dict[str, Any],
    run_name: str,
    target_step: int,
) -> tuple[Path, Path, Path, list[str]]:
    checkpoint_root = phase_runner._absolute_root("ROSETTA_CHECKPOINT_ROOT")
    step_dir = checkpoint.resolve()
    if step_dir.name == "pretrained_model":
        step_dir = step_dir.parent
    if not step_dir.is_relative_to(checkpoint_root):
        raise ValueError("Resumed checkpoint must remain inside the mounted checkpoint root.")
    relative = step_dir.relative_to(checkpoint_root)
    if (
        len(relative.parts) != 5
        or relative.parts[0] != experiment["experiment_id"]
        or relative.parts[1] != "overfit_resume"
        or relative.parts[2] != run_name
        or relative.parts[3] != "checkpoints"
        or int(relative.parts[4]) != target_step
    ):
        raise ValueError("Resumed checkpoint does not match the registered one-step layout.")
    last = step_dir.parent / "last"
    if not last.is_symlink() or last.resolve() != step_dir:
        raise ValueError("Resumed checkpoint is not the run's immutable last checkpoint.")
    pretrained_dir = step_dir / "pretrained_model"
    training_state_dir = step_dir / "training_state"
    required = [
        pretrained_dir / "config.json",
        pretrained_dir / "model.safetensors",
        pretrained_dir / "policy_preprocessor.json",
        pretrained_dir / "policy_postprocessor.json",
        pretrained_dir / "train_config.json",
        training_state_dir / "optimizer_param_groups.json",
        training_state_dir / "optimizer_state.safetensors",
        training_state_dir / "rng_state.safetensors",
        training_state_dir / "scheduler_state.json",
        training_state_dir / "training_step.json",
    ]
    if any(not path.is_file() or path.stat().st_size <= 0 for path in required):
        raise FileNotFoundError("The resumed checkpoint is incomplete.")
    required.extend(_processor_state_paths(pretrained_dir).values())
    files = [path.relative_to(step_dir).as_posix() for path in required]
    return step_dir, pretrained_dir, training_state_dir, files


def _validate_saved_resume(
    experiment: dict[str, Any],
    run_name: str,
    source_step: int,
    step_dir: Path,
    pretrained_dir: Path,
    training_state_dir: Path,
) -> dict[str, Any]:
    target_step = source_step + 1
    overfit = experiment["phases"]["overfit"]
    train_config = _load_json(pretrained_dir / "train_config.json")
    training_step = _load_json(training_state_dir / "training_step.json")
    scheduler = _load_json(training_state_dir / "scheduler_state.json")
    dataset = train_config.get("dataset", {})
    policy = train_config.get("policy", {})
    if (
        train_config.get("resume") is not True
        or train_config.get("job_name") != run_name
        or train_config.get("steps") != target_step
        or train_config.get("save_freq") != target_step
        or train_config.get("batch_size") != overfit["batch_size"]
        or train_config.get("seed") != experiment["seed"]
        or train_config.get("rename_map") != experiment["dataset"]["rename_map"]
        or Path(str(train_config.get("output_dir"))).resolve() != step_dir.parents[1]
        or dataset.get("repo_id") != experiment["dataset"]["identifier"]
        or dataset.get("revision") != experiment["dataset"]["revision"]
        or dataset.get("episodes") != overfit["episodes"]
        or set(dataset.get("episodes", [])) & set(experiment["dataset"]["test_episodes"])
        or policy.get("type") != "smolvla"
        or policy.get("pretrained_revision") != experiment["model"]["revision"]
        or policy.get("push_to_hub") is not False
        or train_config.get("save_checkpoint_to_hub") is not False
        or training_step.get("step") != target_step
        or training_step.get("batch_size") != overfit["batch_size"]
        or training_step.get("dp_world_size") != 1
        or scheduler.get("last_epoch") != target_step
        or scheduler.get("_step_count") != target_step + 1
    ):
        raise ValueError("Saved one-step resume identity is invalid.")
    return {
        "target_step": target_step,
        "scheduler_last_epoch": int(scheduler["last_epoch"]),
        "scheduler_step_count": int(scheduler["_step_count"]),
        "learning_rate": _finite_number(scheduler["_last_lr"][0], "resumed learning rate"),
    }


def _validate_state_progression(
    source_pretrained_dir: Path,
    source_training_state_dir: Path,
    resumed_pretrained_dir: Path,
    resumed_training_state_dir: Path,
    source_step: int,
) -> dict[str, Any]:
    source_scheduler = _load_json(source_training_state_dir / "scheduler_state.json")
    resumed_scheduler = _load_json(resumed_training_state_dir / "scheduler_state.json")
    if (
        source_scheduler.get("last_epoch") != source_step
        or resumed_scheduler.get("last_epoch") != source_step + 1
        or source_scheduler.get("_step_count") != source_step + 1
        or resumed_scheduler.get("_step_count") != source_step + 2
    ):
        raise ValueError("Scheduler state did not advance by exactly one resumed optimizer step.")
    pairs = {
        "model_safetensors": (
            source_pretrained_dir / "model.safetensors",
            resumed_pretrained_dir / "model.safetensors",
        ),
        "optimizer_state": (
            source_training_state_dir / "optimizer_state.safetensors",
            resumed_training_state_dir / "optimizer_state.safetensors",
        ),
        "rng_state": (
            source_training_state_dir / "rng_state.safetensors",
            resumed_training_state_dir / "rng_state.safetensors",
        ),
    }
    hashes: dict[str, dict[str, Any]] = {}
    for name, (source, resumed) in pairs.items():
        source_hash = file_sha256(source)
        resumed_hash = file_sha256(resumed)
        if source_hash == resumed_hash:
            raise ValueError(f"{name} did not change after the resumed optimizer step.")
        hashes[name] = {"source_sha256": source_hash, "resumed_sha256": resumed_hash}
    source_policy_config_path = source_pretrained_dir / "config.json"
    resumed_policy_config_path = resumed_pretrained_dir / "config.json"
    source_policy_config = _load_json(source_policy_config_path)
    resumed_policy_config = _load_json(resumed_policy_config_path)
    source_policy_config.pop("pretrained_path", None)
    resumed_policy_config.pop("pretrained_path", None)
    if source_policy_config != resumed_policy_config:
        raise ValueError("Policy contract changed across a same-contract resume.")
    hashes["policy_config"] = {
        "source_sha256": file_sha256(source_policy_config_path),
        "resumed_sha256": file_sha256(resumed_policy_config_path),
        "contract_equal_ignoring_pretrained_path": True,
    }
    unchanged_pairs = {
        "preprocessor_config": (
            source_pretrained_dir / "policy_preprocessor.json",
            resumed_pretrained_dir / "policy_preprocessor.json",
        ),
        "postprocessor_config": (
            source_pretrained_dir / "policy_postprocessor.json",
            resumed_pretrained_dir / "policy_postprocessor.json",
        ),
    }
    for name, (source, resumed) in unchanged_pairs.items():
        source_hash = file_sha256(source)
        resumed_hash = file_sha256(resumed)
        if source_hash != resumed_hash:
            raise ValueError(f"{name} changed across a same-contract resume.")
        hashes[name] = {"source_sha256": source_hash, "resumed_sha256": resumed_hash}
    source_states = _processor_state_paths(source_pretrained_dir)
    resumed_states = _processor_state_paths(resumed_pretrained_dir)
    if set(source_states) != set(resumed_states):
        raise ValueError("Processor state inventory changed across a same-contract resume.")
    for relative in sorted(source_states):
        source_hash = file_sha256(source_states[relative])
        resumed_hash = file_sha256(resumed_states[relative])
        if source_hash != resumed_hash:
            raise ValueError(f"Processor state changed across resume: {relative}.")
        hashes[f"processor_state:{relative}"] = {
            "source_sha256": source_hash,
            "resumed_sha256": resumed_hash,
        }
    return hashes


def _validate_trackio_resume(
    experiment: dict[str, Any],
    run_name: str,
    source_run_name: str,
    source_step: int,
) -> dict[str, Any]:
    trackio_root = phase_runner._absolute_root("TRACKIO_DIR")
    database = trackio_root / f"{experiment['tracking']['project']}.db"
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        config_rows = connection.execute(
            "SELECT run_id, config, created_at FROM configs WHERE run_name = ?",
            (run_name,),
        ).fetchall()
        if len(config_rows) != 1:
            raise ValueError("Trackio must contain exactly one config for the resume run.")
        run_id, raw_config, created_at = config_rows[0]
        public_config = _decode_json(raw_config)
        metric_rows = connection.execute(
            "SELECT step, metrics, timestamp FROM metrics "
            "WHERE run_id = ? AND run_name = ? ORDER BY id",
            (run_id, run_name),
        ).fetchall()
    finally:
        connection.close()
    target_step = source_step + 1
    if (
        public_config.get("experiment_id") != experiment["experiment_id"]
        or public_config.get("phase") != "overfit_resume"
        or public_config.get("model_revision") != experiment["model"]["revision"]
        or public_config.get("dataset_revision") != experiment["dataset"]["revision"]
        or public_config.get("steps") != target_step
        or public_config.get("batch_size") != experiment["phases"]["overfit"]["batch_size"]
        or public_config.get("train_episode_count")
        != len(experiment["phases"]["overfit"]["episodes"])
        or public_config.get("resume") is not True
        or public_config.get("resume_from_step") != source_step
        or public_config.get("resume_source_run") != source_run_name
        or public_config.get("test_split_loaded") is not False
    ):
        raise ValueError("Trackio resume config identity is invalid.")
    train_metrics: dict[str, float] | None = None
    checkpoint_steps: set[int] = set()
    for raw_step, raw_metrics, _ in metric_rows:
        metrics = _decode_json(raw_metrics)
        step = int(raw_step)
        if "train/loss" in metrics:
            if step != target_step or train_metrics is not None:
                raise ValueError("Trackio resume run contains an unexpected train step.")
            train_metrics = {
                "loss": _finite_number(metrics.get("train/loss"), "resume loss"),
                "grad_norm": _finite_number(metrics.get("train/grad_norm"), "resume gradient norm"),
                "learning_rate": _finite_number(metrics.get("train/lr"), "resume learning rate"),
            }
        if metrics.get("system/checkpoint_saved") == 1:
            checkpoint_steps.add(step)
    if train_metrics is None or checkpoint_steps != {target_step}:
        raise ValueError("Trackio resume metrics or checkpoint marker are incomplete.")
    return {
        "database": database.name,
        "run_id": str(run_id),
        "run_name": run_name,
        "created_at": str(created_at),
        "metric_rows": len(metric_rows),
        "train_step": target_step,
        "train_metrics": train_metrics,
        "checkpoint_steps": sorted(checkpoint_steps),
        "test_split_loaded": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--resumed-checkpoint", type=Path, required=True)
    parser.add_argument("--container-exit-code", type=int, required=True)
    parser.add_argument("--oom-event-count", type=int, required=True)
    args = parser.parse_args()
    if not RUN_NAME_PATTERN.fullmatch(args.run_name):
        raise ValueError("--run-name must be a lower-case path-safe identifier.")
    if args.container_exit_code != 0 or args.oom_event_count != 0:
        raise RuntimeError("The observed resume container did not exit cleanly within budget.")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("HF_DATASETS_OFFLINE") != "1":
        raise RuntimeError("Resume verification must run with networking disabled.")

    config_path = args.config.resolve()
    experiment = _load_yaml(config_path)
    contract_path = REPOSITORY_ROOT / str(experiment["action_contract"]["derived"])
    source_step_dir, source_pretrained_dir, source_step, source_run_name = (
        resume_runner._source_checkpoint(args.source_checkpoint, experiment)
    )
    source_training_state_dir = source_step_dir / "training_state"
    target_step = source_step + 1
    resumed_step_dir, resumed_pretrained_dir, resumed_training_state_dir, files = (
        _resumed_checkpoint(args.resumed_checkpoint, experiment, args.run_name, target_step)
    )
    saved_resume = _validate_saved_resume(
        experiment,
        args.run_name,
        source_step,
        resumed_step_dir,
        resumed_pretrained_dir,
        resumed_training_state_dir,
    )
    state_progression = _validate_state_progression(
        source_pretrained_dir,
        source_training_state_dir,
        resumed_pretrained_dir,
        resumed_training_state_dir,
        source_step,
    )
    tracking = _validate_trackio_resume(experiment, args.run_name, source_run_name, source_step)
    checkpoint_root = phase_runner._absolute_root("ROSETTA_CHECKPOINT_ROOT")
    report = {
        "schema_version": 1,
        "status": "passed",
        "stage": "smolvla_explicit_resume_verification",
        "experiment_id": experiment["experiment_id"],
        "experiment_config_sha256": file_sha256(config_path),
        "action_contract_sha256": file_sha256(contract_path),
        "verification_script_sha256": file_sha256(Path(__file__)),
        "resume_runner_sha256": file_sha256(REPOSITORY_ROOT / "scripts/resume_smolvla_overfit.py"),
        "resume_mechanism": "lerobot_config_path_with_resume_true",
        "source_run_name": source_run_name,
        "source_checkpoint": source_step_dir.relative_to(checkpoint_root).as_posix(),
        "source_step": source_step,
        "run_name": args.run_name,
        "resumed_checkpoint": resumed_step_dir.relative_to(checkpoint_root).as_posix(),
        "resumed_checkpoint_files": files,
        "saved_resume": saved_resume,
        "state_progression": state_progression,
        "tracking": tracking,
        "model_revision": experiment["model"]["revision"],
        "dataset_revision": experiment["dataset"]["revision"],
        "episodes_loaded": experiment["phases"]["overfit"]["episodes"],
        "hidden_test_loaded": False,
        "network_disabled": True,
        "runtime_observation": {
            "source": "orchestrated_container_exit_and_docker_state",
            "container_exit_code": args.container_exit_code,
            "oom_event_count": args.oom_event_count,
            "memory_limit": experiment["resources"]["memory_limit"],
            "memory_swap_limit": experiment["resources"]["memory_swap_limit"],
        },
        "acceptance": {
            "source_checkpoint_complete": True,
            "optimizer_scheduler_rng_restored": True,
            "exactly_one_optimizer_step_completed": True,
            "checkpoint_written": True,
            "trackio_run_readable": True,
            "no_resource_limit_violation": True,
            "hidden_test_loaded": False,
        },
    }
    json.dumps(report, allow_nan=False)
    run_root = phase_runner._absolute_root("ROSETTA_RUN_ROOT")
    verification_root = run_root / str(experiment["experiment_id"]) / "verification"
    destination = verification_root / f"{args.run_name}.json"
    create_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report: {destination.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
