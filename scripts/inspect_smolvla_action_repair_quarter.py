"""Inspect one registered action-repair overfit quarter without loading weights."""

from __future__ import annotations

import argparse
import json
import math
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

import run_smolvla_phase as phase_runner  # noqa: E402

from rosetta_reality.experiment import file_sha256  # noqa: E402
from rosetta_reality.features import create_json  # noqa: E402
from rosetta_reality.vla import load_smolvla_experiment  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}.")
    json.dumps(value, allow_nan=False)
    return value


def _registered_quarters(overfit: dict[str, Any]) -> list[int]:
    steps = int(overfit["steps"])
    save_freq = int(overfit["save_freq"])
    if steps <= 0 or save_freq <= 0 or steps % 4 != 0 or save_freq != steps // 4:
        raise ValueError("Repair monitoring requires four equal checkpoint quarters.")
    return [save_freq * quarter for quarter in range(1, 5)]


def _validate_checkpoint(
    step_dir: Path,
    experiment: dict[str, Any],
    run_name: str,
    step: int,
) -> dict[str, Any]:
    pretrained = step_dir / "pretrained_model"
    state = step_dir / "training_state"
    required = [
        pretrained / "config.json",
        pretrained / "model.safetensors",
        pretrained / "policy_preprocessor.json",
        pretrained / "policy_postprocessor.json",
        pretrained / "train_config.json",
        state / "optimizer_param_groups.json",
        state / "optimizer_state.safetensors",
        state / "rng_state.safetensors",
        state / "scheduler_state.json",
        state / "training_step.json",
    ]
    if any(not path.is_file() or path.stat().st_size <= 0 for path in required):
        raise FileNotFoundError("The repair quarter checkpoint is incomplete.")
    train_config = _load_json(pretrained / "train_config.json")
    training_step = _load_json(state / "training_step.json")
    overfit = experiment["phases"]["overfit"]
    dataset = train_config.get("dataset", {})
    policy = train_config.get("policy", {})
    expected_output = step_dir.parents[1]
    if (
        training_step.get("step") != step
        or training_step.get("batch_size") != overfit["batch_size"]
        or training_step.get("dp_world_size") != 1
        or train_config.get("job_name") != run_name
        or train_config.get("steps") != overfit["steps"]
        or train_config.get("save_freq") != overfit["save_freq"]
        or train_config.get("batch_size") != overfit["batch_size"]
        or train_config.get("seed") != experiment["seed"]
        or train_config.get("resume") is not False
        or Path(str(train_config.get("output_dir"))).resolve() != expected_output.resolve()
        or dataset.get("repo_id") != experiment["dataset"]["identifier"]
        or dataset.get("revision") != experiment["dataset"]["revision"]
        or dataset.get("episodes") != overfit["episodes"]
        or policy.get("pretrained_revision") != experiment["model"]["revision"]
    ):
        raise ValueError("The quarter checkpoint differs from the registered repair run.")
    return {
        "training_step": training_step,
        "train_config_sha256": file_sha256(pretrained / "train_config.json"),
        "model_safetensors_bytes": (pretrained / "model.safetensors").stat().st_size,
        "optimizer_state_bytes": (state / "optimizer_state.safetensors").stat().st_size,
    }


def _load_trackio_quarter(
    database: Path,
    experiment: dict[str, Any],
    run_name: str,
    step: int,
) -> tuple[str, dict[str, float]]:
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
        raise ValueError("The Trackio run or requested quarter is not durable yet.")
    run_id = str(configs[0][0])
    public_config = json.loads(configs[0][1])
    overfit = experiment["phases"]["overfit"]
    if (
        public_config.get("experiment_id") != experiment["experiment_id"]
        or public_config.get("phase") != "overfit"
        or public_config.get("model_revision") != experiment["model"]["revision"]
        or public_config.get("dataset_revision") != experiment["dataset"]["revision"]
        or public_config.get("batch_size") != overfit["batch_size"]
        or public_config.get("steps") != overfit["steps"]
        or public_config.get("save_freq") != overfit["save_freq"]
        or public_config.get("test_split_loaded") is not False
        or {str(row[0]) for row in rows} != {run_id}
    ):
        raise ValueError("Trackio quarter identity differs from the registered repair run.")
    metrics: dict[str, float] | None = None
    checkpoint_saved = False
    for _, raw in rows:
        values = json.loads(raw)
        if "train/loss" in values:
            metrics = {
                key: float(values[key])
                for key in ("train/loss", "train/grad_norm", "train/lr")
            }
        if values.get("system/checkpoint_saved") == 1:
            checkpoint_saved = True
    if metrics is None or not checkpoint_saved or not all(
        math.isfinite(value) for value in metrics.values()
    ):
        raise ValueError("The quarter lacks finite train metrics or checkpoint evidence.")
    return run_id, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    experiment = load_smolvla_experiment(config_path, REPOSITORY_ROOT)
    if experiment.get("status") != "preregistered_action_repair_smoke_and_overfit":
        raise ValueError("Only the registered action-repair experiment may be monitored.")
    quarters = _registered_quarters(experiment["phases"]["overfit"])
    if args.step not in quarters:
        raise ValueError("The requested step is not a registered quarter checkpoint.")
    checkpoint_root = phase_runner._absolute_root("ROSETTA_CHECKPOINT_ROOT")
    run_root = phase_runner._absolute_root("ROSETTA_RUN_ROOT")
    step_dir = (
        checkpoint_root
        / str(experiment["experiment_id"])
        / "overfit"
        / args.run_name
        / "checkpoints"
        / f"{args.step:06d}"
    )
    checkpoint = _validate_checkpoint(step_dir, experiment, args.run_name, args.step)
    trackio_root = phase_runner._absolute_root("TRACKIO_DIR")
    run_id, metrics = _load_trackio_quarter(
        trackio_root / f"{experiment['tracking']['project']}.db",
        experiment,
        args.run_name,
        args.step,
    )
    report = {
        "schema_version": 1,
        "status": "passed",
        "stage": "smolvla_action_repair_quarter_monitor",
        "experiment_id": experiment["experiment_id"],
        "experiment_config_sha256": file_sha256(config_path),
        "run_name": args.run_name,
        "step": args.step,
        "fraction_complete": args.step / int(experiment["phases"]["overfit"]["steps"]),
        "registered_quarters": quarters,
        "trackio_run_id": run_id,
        "metrics": metrics,
        "checkpoint": checkpoint,
        "model_weights_loaded": False,
        "optimizer_created": False,
        "hidden_test_loaded": False,
    }
    destination = (
        run_root
        / str(experiment["experiment_id"])
        / "monitoring"
        / f"{args.run_name}-step-{args.step:06d}.json"
    )
    create_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Monitor report: {destination.relative_to(run_root).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
