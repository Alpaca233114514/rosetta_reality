"""Select the preregistered SmolVLA formal checkpoint from fixed validation only."""

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
DEFAULT_PLAN = REPOSITORY_ROOT / "configs/vla/smolvla_450m_aloha_insertion_formal_001.yaml"
for root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import run_smolvla_formal as formal_runner  # noqa: E402
import run_smolvla_phase as phase_runner  # noqa: E402

from rosetta_reality.experiment import file_sha256  # noqa: E402
from rosetta_reality.features import create_json  # noqa: E402


def _processor_state_file(
    pretrained_dir: Path, config_name: str, registry_name: str
) -> Path:
    config = formal_runner._load_json(pretrained_dir / config_name)
    steps = config.get("steps")
    if not isinstance(steps, list):
        raise ValueError(f"{config_name} has no processor steps.")
    matches = [
        step.get("state_file")
        for step in steps
        if isinstance(step, dict) and step.get("registry_name") == registry_name
    ]
    if len(matches) != 1 or not isinstance(matches[0], str):
        raise ValueError(f"{config_name} has no unique {registry_name} state file.")
    relative = Path(matches[0])
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != matches[0]
    ):
        raise ValueError(f"{config_name} contains an unsafe processor state path.")
    return pretrained_dir / relative


def _tokenizer_hashes(pretrained_dir: Path) -> dict[str, str]:
    tokenizer = pretrained_dir / "tokenizer"
    files = [path for path in sorted(tokenizer.rglob("*")) if path.is_file()]
    if not files:
        raise FileNotFoundError("Validated checkpoint tokenizer is missing or empty.")
    return {
        path.relative_to(tokenizer).as_posix(): file_sha256(path) for path in files
    }


def _validated_checkpoint_hashes(
    pretrained_dir: Path, report: dict[str, Any]
) -> dict[str, Any]:
    recorded = {
        "model_safetensors_sha256": (
            pretrained_dir / "model.safetensors",
            report.get("model_source", {}).get("model_safetensors_sha256"),
        ),
        "policy_config_sha256": (
            pretrained_dir / "config.json",
            report.get("model_source", {}).get("policy_config_sha256"),
        ),
        "preprocessor_config_sha256": (
            pretrained_dir / "policy_preprocessor.json",
            report.get("model_source", {}).get("preprocessor_config_sha256"),
        ),
        "postprocessor_config_sha256": (
            pretrained_dir / "policy_postprocessor.json",
            report.get("model_source", {}).get("postprocessor_config_sha256"),
        ),
        "preprocessor_statistics_sha256": (
            _processor_state_file(
                pretrained_dir,
                "policy_preprocessor.json",
                "normalizer_processor",
            ),
            report.get("processor_statistics", {}).get(
                "preprocessor_statistics_sha256"
            ),
        ),
        "postprocessor_statistics_sha256": (
            _processor_state_file(
                pretrained_dir,
                "policy_postprocessor.json",
                "unnormalizer_processor",
            ),
            report.get("processor_statistics", {}).get(
                "postprocessor_statistics_sha256"
            ),
        ),
    }
    hashes: dict[str, str] = {}
    for name, (path, expected) in recorded.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Validated checkpoint file is missing: {path.name}.")
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"Validated checkpoint file changed: {path.name}.")
        hashes[name] = actual
    tokenizer_hashes = _tokenizer_hashes(pretrained_dir)
    if tokenizer_hashes != report.get("model_source", {}).get(
        "tokenizer_files_sha256"
    ):
        raise ValueError("Validated checkpoint tokenizer changed.")
    hashes["tokenizer_files_sha256"] = tokenizer_hashes
    return hashes


def _validation_report(
    path: Path,
    *,
    plan: dict[str, Any],
    experiment: dict[str, Any],
    base_path: Path,
    contract_sha256: str,
    normalization_sha256: str,
    plan_sha256: str,
    expected_source: str | int,
) -> dict[str, Any]:
    report = formal_runner._load_json(path)
    validation = plan["validation"]
    source = report.get("model_source", {})
    expected_kind = "base" if expected_source == "base" else "checkpoint"
    expected_step = None if expected_source == "base" else int(expected_source)
    metrics = report.get("metrics", {})
    metric_names = {
        "action_mae",
        "action_rmse",
        "first_action_mae",
        "fixed_flow_loss",
        "invalid_action_rate",
        "joint_limit_violation_rate",
        "action_smoothness_mean_abs_delta",
    }
    if (
        report.get("status") != "complete"
        or report.get("stage") != "smolvla_fixed_validation"
        or report.get("experiment_id") != experiment["experiment_id"]
        or report.get("formal_plan_sha256") != plan_sha256
        or report.get("experiment_config_sha256") != file_sha256(base_path)
        or report.get("action_contract_sha256") != contract_sha256
        or report.get("normalization_report_sha256") != normalization_sha256
        or report.get("validation_episodes") != validation["episodes"]
        or report.get("frame_offsets") != validation["frame_offsets"]
        or report.get("materialized_episodes") != sorted(validation["episodes"])
        or report.get("sample_count") != validation["total_samples"]
        or report.get("hidden_test_loaded") is not False
        or report.get("gradients_enabled") is not False
        or report.get("optimizer_created") is not False
        or source.get("kind") != expected_kind
        or source.get("step") != expected_step
        or not isinstance(metrics, dict)
        or not metric_names <= set(metrics)
        or any(
            not isinstance(metrics[name], int | float)
            or isinstance(metrics[name], bool)
            or not math.isfinite(float(metrics[name]))
            for name in metric_names
        )
    ):
        raise ValueError(f"Invalid formal validation report: {path.name}.")
    if expected_kind == "checkpoint" and not report.get("processor_statistics"):
        raise ValueError("A checkpoint validation did not verify saved processor statistics.")
    return report


def _validate_workspace_identities(
    launch: dict[str, Any], reports: dict[str | int, tuple[Path, dict[str, Any]]]
) -> dict[str, Any]:
    expected = launch.get("code_identity")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("The formal launch has no workspace code identity.")
    for path, report in reports.values():
        if report.get("code_identity") != expected:
            raise ValueError(
                f"Validation report workspace differs from the launch: {path.name}."
            )
    return expected


def _validate_sync_run_snapshot(
    sync: dict[str, Any], plan: dict[str, Any], experiment: dict[str, Any]
) -> dict[str, Any]:
    snapshots = sync.get("run_snapshots")
    matches = [
        snapshot
        for snapshot in snapshots if isinstance(snapshot, dict)
    ] if isinstance(snapshots, list) else []
    matches = [
        snapshot
        for snapshot in matches
        if snapshot.get("run_name") == plan["run_name"]
        and snapshot.get("experiment_id") == experiment["experiment_id"]
        and snapshot.get("phase") == "formal"
        and snapshot.get("formal_plan_sha256") == plan["formal_plan_sha256"]
        and snapshot.get("maximum_logged_step") == plan["training"]["steps"]
    ]
    if (
        sync.get("project") != plan["tracking"]["project"]
        or not isinstance(sync.get("project_snapshot_sha256"), str)
        or len(sync["project_snapshot_sha256"]) != 64
        or len(matches) != 1
    ):
        raise ValueError("The Trackio sync snapshot does not contain the selected formal run.")
    return matches[0]


def _training_metrics(
    database: Path,
    run_name: str,
    training: dict[str, Any],
    plan_sha256: str,
) -> dict[str, Any]:
    if not database.is_file():
        raise FileNotFoundError("The durable Trackio database is missing.")
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT run_id, step, metrics FROM metrics WHERE run_name=? ORDER BY id",
            (run_name,),
        ).fetchall()
        configs = connection.execute(
            "SELECT run_id, config FROM configs WHERE run_name=? ORDER BY id",
            (run_name,),
        ).fetchall()
    finally:
        connection.close()
    if not rows or len({str(row[0]) for row in rows}) != 1 or len(configs) != 1:
        raise ValueError("Trackio formal run identity is missing or ambiguous.")
    config = json.loads(configs[0][1])
    optimizer_contract = formal_runner._optimizer_contract(training)
    expected_optimizer_config: dict[str, Any] = {}
    if optimizer_contract is not None:
        optimizer = optimizer_contract["optimizer"]
        scheduler = optimizer_contract["scheduler"]
        expected_optimizer_config = {
            "optimizer_type": optimizer["type"],
            "optimizer_lr": optimizer["lr"],
            "optimizer_betas": optimizer["betas"],
            "optimizer_eps": optimizer["eps"],
            "optimizer_weight_decay": optimizer["weight_decay"],
            "optimizer_grad_clip_norm": optimizer["grad_clip_norm"],
            "scheduler_type": scheduler["type"],
            "scheduler_warmup_steps": scheduler["num_warmup_steps"],
            "scheduler_decay_steps": scheduler["num_decay_steps"],
            "scheduler_peak_lr": scheduler["peak_lr"],
            "scheduler_decay_lr": scheduler["decay_lr"],
        }
    steps = int(training["steps"])
    log_freq = int(training["log_freq"])
    if (
        config.get("phase") != "formal"
        or config.get("steps") != steps
        or config.get("test_split_loaded") is not False
        or config.get("normalization_source_split") != "train"
        or config.get("formal_plan_sha256") != plan_sha256
        or any(config.get(key) != value for key, value in expected_optimizer_config.items())
    ):
        raise ValueError("Trackio formal configuration differs from the registered plan.")
    train_rows: dict[int, dict[str, Any]] = {}
    checkpoint_steps: list[int] = []
    for _, step, raw_metrics in rows:
        metrics = json.loads(raw_metrics)
        numeric = [
            value
            for value in metrics.values()
            if isinstance(value, int | float) and not isinstance(value, bool)
        ]
        if len(numeric) != len(metrics) or not all(
            math.isfinite(float(value)) for value in numeric
        ):
            raise FloatingPointError("Trackio contains a non-finite or non-numeric metric.")
        if "train/loss" in metrics:
            if step in train_rows:
                raise ValueError("Trackio contains duplicate formal training steps.")
            if "train/grad_norm" not in metrics or "train/lr" not in metrics:
                raise ValueError("Trackio formal training step lacks gradient or LR evidence.")
            train_rows[int(step)] = metrics
        if metrics.get("system/checkpoint_saved") == 1:
            checkpoint_steps.append(int(step))
    expected_logged_steps = list(range(log_freq, steps + 1, log_freq))
    if sorted(train_rows) != expected_logged_steps:
        raise ValueError("Trackio formal training steps are incomplete.")
    losses = [float(train_rows[step]["train/loss"]) for step in sorted(train_rows)]
    gradients = [float(train_rows[step]["train/grad_norm"]) for step in sorted(train_rows)]
    learning_rates = [float(train_rows[step]["train/lr"]) for step in sorted(train_rows)]
    return {
        "run_id": str(rows[0][0]),
        "logged_training_steps": len(train_rows),
        "checkpoint_steps": sorted(checkpoint_steps),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": min(losses),
        "maximum_loss": max(losses),
        "maximum_gradient_norm": max(gradients),
        "initial_learning_rate": learning_rates[0],
        "final_learning_rate": learning_rates[-1],
        "minimum_learning_rate": min(learning_rates),
        "maximum_learning_rate": max(learning_rates),
        "optimizer_contract": optimizer_contract,
        "all_losses_and_gradients_finite": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--trackio-sync-report", type=Path, required=True)
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    plan, base_path, experiment = formal_runner._validate_plan(plan_path)
    contract_path = REPOSITORY_ROOT / str(experiment["action_contract"]["derived"])
    contract_sha256 = file_sha256(contract_path)
    normalization_path, _, _ = formal_runner._validate_normalization(
        plan, experiment, base_path, contract_sha256
    )
    normalization_sha256 = file_sha256(normalization_path)
    launch_path = (
        phase_runner._absolute_root("ROSETTA_RUN_ROOT")
        / str(experiment["experiment_id"])
        / "launch"
        / f"{plan['run_name']}.json"
    )
    launch = formal_runner._load_json(launch_path)
    if (
        launch.get("status") != "preregistered"
        or launch.get("mode") != "train"
        or launch.get("formal_plan_sha256") != file_sha256(plan_path)
        or launch.get("hidden_test_loaded") is not False
    ):
        raise ValueError("The formal training launch manifest is invalid.")
    sync = formal_runner._load_json(args.trackio_sync_report.resolve())
    if (
        sync.get("status") != "complete"
        or sync.get("contains_sensitive_data") is not False
        or sync.get("media_uploaded") is not False
        or sync.get("test_split_loaded") is not False
        or sync.get("space_id") != experiment["tracking"]["space_id"]
    ):
        raise ValueError("The final public Trackio sync report is invalid.")
    plan_with_identity = {**plan, "formal_plan_sha256": file_sha256(plan_path)}
    synced_run = _validate_sync_run_snapshot(sync, plan_with_identity, experiment)

    validation_root = (
        phase_runner._absolute_root("ROSETTA_RUN_ROOT")
        / str(experiment["experiment_id"])
        / "validation"
    )
    expected_sources: list[str | int] = list(plan["validation"]["checkpoints"])
    reports: dict[str | int, tuple[Path, dict[str, Any]]] = {}
    for source in expected_sources:
        suffix = "base" if source == "base" else f"step-{int(source):06d}"
        path = validation_root / f"{plan['validation']['run_name_prefix']}-{suffix}.json"
        reports[source] = (
            path,
            _validation_report(
                path,
                plan=plan,
                experiment=experiment,
                base_path=base_path,
                contract_sha256=contract_sha256,
                normalization_sha256=normalization_sha256,
                plan_sha256=file_sha256(plan_path),
                expected_source=source,
            ),
        )
    selection_code_identity = _validate_workspace_identities(launch, reports)

    checkpoint_root = phase_runner._absolute_root("ROSETTA_CHECKPOINT_ROOT")
    checkpoint_summaries: list[dict[str, Any]] = []
    for source in expected_sources[1:]:
        step = int(source)
        path, report = reports[source]
        step_dir = (
            checkpoint_root
            / str(experiment["experiment_id"])
            / "formal"
            / str(plan["run_name"])
            / "checkpoints"
            / f"{step:06d}"
        )
        training_step = formal_runner._load_json(step_dir / "training_state/training_step.json")
        pretrained_dir = step_dir / "pretrained_model"
        checkpoint_hashes = _validated_checkpoint_hashes(pretrained_dir, report)
        if training_step.get("step") != step:
            raise ValueError("A formal checkpoint differs from its validation report.")
        checkpoint_summaries.append(
            {
                "step": step,
                "validation_report_sha256": file_sha256(path),
                **checkpoint_hashes,
                "metrics": report["metrics"],
                "processor_statistics": report["processor_statistics"],
            }
        )

    primary = str(plan["validation"]["primary_selection_metric"])
    secondary = str(plan["validation"]["secondary_selection_metric"])
    selected = min(
        checkpoint_summaries,
        key=lambda item: (
            float(item["metrics"][primary]),
            float(item["metrics"][secondary]),
            int(item["step"]),
        ),
    )
    base_path_report, base_report = reports["base"]
    base_value = float(base_report["metrics"][primary])
    selected_value = float(selected["metrics"][primary])
    expected_checkpoint_steps = [int(value) for value in expected_sources[1:]]
    trackio_root = Path(os.environ.get("TRACKIO_DIR", ""))
    if not trackio_root.is_absolute():
        raise ValueError("TRACKIO_DIR must identify the durable Trackio root.")
    training_metrics = _training_metrics(
        trackio_root / f"{experiment['tracking']['project']}.db",
        str(plan["run_name"]),
        plan["training"],
        file_sha256(plan_path),
    )
    acceptance = {
        "all_logged_losses_and_gradients_are_finite": training_metrics[
            "all_losses_and_gradients_finite"
        ],
        "all_registered_checkpoints_saved": training_metrics["checkpoint_steps"]
        == expected_checkpoint_steps,
        "checkpoint_reload_and_processor_statistics_match": all(
            bool(item["processor_statistics"]) for item in checkpoint_summaries
        ),
        "validation_action_mae_improves_over_base": selected_value < base_value,
        "hidden_test_not_loaded": True,
    }
    passed = all(acceptance.values())
    report = {
        "schema_version": 1,
        "status": "passed" if passed else "rejected",
        "stage": "smolvla_formal_checkpoint_selection",
        "experiment_id": experiment["experiment_id"],
        "formal_plan_sha256": file_sha256(plan_path),
        "experiment_config_sha256": file_sha256(base_path),
        "action_contract_sha256": contract_sha256,
        "normalization_report_sha256": normalization_sha256,
        "launch_manifest_sha256": file_sha256(launch_path),
        "trackio_sync_report_sha256": file_sha256(args.trackio_sync_report.resolve()),
        "trackio_project_snapshot_sha256": sync["project_snapshot_sha256"],
        "trackio_synced_run": synced_run,
        "code_identity": selection_code_identity,
        "training_metrics": training_metrics,
        "selection_protocol": {
            "split": "validation",
            "primary_metric": primary,
            "secondary_metric": secondary,
            "tie_breaker": "earlier_checkpoint_step",
            "hidden_test_loaded": False,
        },
        "base": {
            "validation_report_sha256": file_sha256(base_path_report),
            "metrics": base_report["metrics"],
        },
        "checkpoints": checkpoint_summaries,
        "selected": selected,
        "primary_improvement": base_value - selected_value,
        "primary_relative_improvement": (base_value - selected_value) / base_value,
        "acceptance": acceptance,
        "hidden_test_loaded": False,
    }
    selection_name = str(
        plan.get("selection", {}).get(
            "report_name", f"{plan['run_name']}-selection.json"
        )
    )
    if Path(selection_name).name != selection_name or not selection_name.endswith(".json"):
        raise ValueError("Formal selection report name must be a safe JSON filename.")
    destination = (
        phase_runner._absolute_root("ROSETTA_RUN_ROOT")
        / str(experiment["experiment_id"])
        / "selection"
        / selection_name
    )
    create_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report: {destination.name}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
