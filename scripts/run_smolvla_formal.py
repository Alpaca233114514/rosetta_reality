"""Launch the preregistered SmolVLA formal preflight or development training run."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
DEFAULT_PLAN = REPOSITORY_ROOT / "configs/vla/smolvla_450m_aloha_insertion_formal_001.yaml"
for root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import run_smolvla_phase as phase_runner  # noqa: E402

from rosetta_reality.experiment import file_sha256, workspace_code_identity  # noqa: E402
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


def _repository_path(raw: str, *, require_file: bool = True) -> Path:
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Formal plan paths must be safe repository-relative paths.")
    path = (REPOSITORY_ROOT / relative).resolve()
    if not path.is_relative_to(REPOSITORY_ROOT):
        raise ValueError("Formal plan path escaped the repository root.")
    if require_file and not path.is_file():
        raise FileNotFoundError(relative.as_posix())
    return path


def _training_coverage(training: dict[str, Any], train_rows: int) -> dict[str, int | float]:
    """Validate and summarize the planned observation exposure.

    Optimizer steps are not an epoch when batch size is one.  Formal plans may
    therefore require a minimum number of dataset passes so a development run
    cannot accidentally stop after seeing only a small prefix of a shuffled
    epoch, as the original 1,000-step run did.
    """

    batch_size = training.get("batch_size")
    steps = training.get("steps")
    minimum_passes = training.get("minimum_dataset_passes")
    if (
        not isinstance(train_rows, int)
        or isinstance(train_rows, bool)
        or train_rows <= 0
        or not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
        or not isinstance(steps, int)
        or isinstance(steps, bool)
        or steps <= 0
    ):
        raise ValueError("Training coverage requires positive integer rows, batch size, and steps.")
    sample_exposures = batch_size * steps
    dataset_passes = sample_exposures / train_rows
    if minimum_passes is not None:
        if (
            isinstance(minimum_passes, bool)
            or not isinstance(minimum_passes, int | float)
            or not math.isfinite(float(minimum_passes))
            or float(minimum_passes) <= 0
        ):
            raise ValueError("minimum_dataset_passes must be a positive finite number.")
        if dataset_passes + 1e-12 < float(minimum_passes):
            raise ValueError(
                "Formal training exposure is below the registered minimum dataset passes: "
                f"planned={dataset_passes:.6f}, required={float(minimum_passes):.6f}."
            )
    return {
        "train_rows": train_rows,
        "batch_size": batch_size,
        "optimizer_steps": steps,
        "sample_exposures": sample_exposures,
        "dataset_passes": dataset_passes,
        "minimum_dataset_passes": (
            0.0 if minimum_passes is None else float(minimum_passes)
        ),
    }


def _validate_performance_optimization(
    plan: dict[str, Any], experiment: dict[str, Any], base_path: Path
) -> None:
    optimization = plan.get("performance_optimization")
    if not isinstance(optimization, dict):
        raise ValueError("The optimized formal plan has no performance evidence.")

    performance_plan_path = _repository_path(str(optimization.get("plan", "")))
    parity_report_path = _repository_path(str(optimization.get("parity_report", "")))
    benchmark_report_path = _repository_path(str(optimization.get("benchmark_report", "")))
    if (
        file_sha256(performance_plan_path) != optimization.get("plan_sha256")
        or file_sha256(parity_report_path) != optimization.get("parity_report_sha256")
        or file_sha256(benchmark_report_path) != optimization.get("benchmark_report_sha256")
    ):
        raise ValueError("Optimized formal performance evidence checksum changed.")

    performance_plan = _load_yaml(performance_plan_path)
    parity_report = _load_json(parity_report_path)
    benchmark_report = _load_json(benchmark_report_path)
    candidate_name = str(optimization.get("selected_candidate", ""))
    candidate = performance_plan.get("candidates", {}).get(candidate_name)
    training = plan.get("training", {})
    policy = training.get("policy", {})
    resources = plan.get("resources", {})
    metrics = benchmark_report.get("metrics", {})
    parity_acceptance = parity_report.get("acceptance", {})
    source_formal = performance_plan.get("formal_plan", {})
    maximum_wall_seconds = optimization.get("maximum_projected_wall_seconds")
    expected_cache_key = f"xpu-{file_sha256(performance_plan_path)[:12]}"
    action_contract = REPOSITORY_ROOT / str(experiment["action_contract"]["derived"])
    if (
        not isinstance(candidate, dict)
        or performance_plan.get("parent_experiment", {}).get("experiment_id")
        != experiment["experiment_id"]
        or source_formal.get("sha256") != benchmark_report.get("formal_plan_sha256")
        or benchmark_report.get("status") != "complete"
        or benchmark_report.get("stage") != "smolvla_xpu_training_performance_benchmark"
        or benchmark_report.get("candidate_name") != candidate_name
        or benchmark_report.get("candidate") != candidate
        or benchmark_report.get("performance_plan_sha256") != file_sha256(performance_plan_path)
        or benchmark_report.get("parity_report_sha256") != file_sha256(parity_report_path)
        or benchmark_report.get("experiment_config_sha256") != file_sha256(base_path)
        or benchmark_report.get("action_contract_sha256") != file_sha256(action_contract)
        or benchmark_report.get("normalization_report_sha256")
        != plan.get("normalization", {}).get("report_sha256")
        or benchmark_report.get("model_revision") != experiment["model"]["revision"]
        or benchmark_report.get("dataset_revision") != experiment["dataset"]["revision"]
        or benchmark_report.get("network_disabled") is not True
        or benchmark_report.get("hidden_test_loaded") is not False
        or benchmark_report.get("checkpoint_written") is not False
        or metrics.get("target_met") is not True
        or not isinstance(maximum_wall_seconds, int | float)
        or isinstance(maximum_wall_seconds, bool)
        or float(metrics.get("projected_one_pass_wall_seconds", math.inf))
        > float(maximum_wall_seconds)
        or int(metrics.get("peak_xpu_allocated_bytes", 2**63))
        > int(benchmark_report.get("target", {}).get("maximum_peak_xpu_allocated_bytes", 0))
        or training.get("batch_size") != candidate.get("batch_size")
        or policy.get("empty_cameras") != candidate.get("empty_cameras")
        or policy.get("compile_model") != candidate.get("compile_model")
        or policy.get("compile_mode") != candidate.get("compile_mode")
        or policy.get("skip_fully_masked_camera_encoding")
        != candidate.get("skip_fully_masked_camera_encoding")
        or optimization.get("compiler_cache_key") != expected_cache_key
        or resources.get("memory_limit")
        != performance_plan.get("resources", {}).get("memory_limit")
        or resources.get("memory_swap_limit")
        != performance_plan.get("resources", {}).get("memory_swap_limit")
        or parity_report.get("status") != "passed"
        or parity_report.get("stage") != "smolvla_masked_camera_encoder_fixed_forward_parity"
        or parity_report.get("performance_plan_sha256") != file_sha256(performance_plan_path)
        or parity_report.get("experiment_config_sha256") != file_sha256(base_path)
        or parity_report.get("action_contract_sha256") != file_sha256(action_contract)
        or parity_report.get("normalization_report_sha256")
        != plan.get("normalization", {}).get("report_sha256")
        or parity_report.get("maximum_absolute_loss_tensor_difference") != 0.0
        or parity_report.get("mean_absolute_loss_tensor_difference") != 0.0
        or parity_acceptance.get("camera_slot_count_unchanged") is not True
        or parity_acceptance.get("vision_encoder_calls_reduced_from_three_to_one") is not True
        or parity_acceptance.get("hidden_test_loaded") is not False
        or parity_acceptance.get("optimizer_created") is not False
        or parity_acceptance.get("gradients_enabled") is not False
    ):
        raise ValueError("Optimized formal performance evidence is invalid or incompatible.")

    implementation = optimization.get("implementation_files")
    if not isinstance(implementation, dict) or not implementation:
        raise ValueError("Optimized formal plan has no implementation checksum inventory.")
    for raw_path, expected_sha256 in implementation.items():
        path = _repository_path(str(raw_path))
        if file_sha256(path) != expected_sha256:
            raise ValueError(f"Optimized trainer implementation changed: {raw_path}.")


def _validate_plan(plan_path: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    plan = _load_yaml(plan_path)
    parent = plan.get("parent_experiment", {})
    base_path = _repository_path(str(parent.get("config", "")))
    if file_sha256(base_path) != parent.get("sha256"):
        raise ValueError("Formal plan parent experiment checksum is stale.")
    experiment = _load_yaml(base_path)
    training = plan.get("training", {})
    validation = plan.get("validation", {})
    preflight = plan.get("preflight", {})
    resources = plan.get("resources", {})
    initialization = plan.get("initialization", {})
    train_episodes = [int(value) for value in training.get("episodes", [])]
    validation_episodes = [int(value) for value in validation.get("episodes", [])]
    test_episodes = {int(value) for value in experiment["dataset"]["test_episodes"]}
    expected_checkpoints = (
        list(
            range(
                int(training.get("save_freq", 0)),
                int(training.get("steps", 0)) + 1,
                int(training.get("save_freq", 0)),
            )
        )
        if int(training.get("save_freq", 0)) > 0
        else []
    )
    if (
        plan.get("schema_version") != 1
        or plan.get("role") != "vla"
        or plan.get("status") != "preregistered"
        or parent.get("experiment_id") != experiment["experiment_id"]
        or train_episodes != experiment["dataset"]["train_episodes"]
        or validation_episodes != experiment["dataset"]["validation_episodes"]
        or set(train_episodes) & set(validation_episodes)
        or (set(train_episodes) | set(validation_episodes)) & test_episodes
        or preflight.get("episodes") != experiment["phases"]["smoke"]["episodes"]
        or preflight.get("batch_size") != 1
        or preflight.get("optimizer_created") is not False
        or not isinstance(training.get("batch_size"), int)
        or isinstance(training.get("batch_size"), bool)
        or training.get("batch_size") <= 0
        or not isinstance(training.get("steps"), int)
        or training.get("steps") <= 0
        or not isinstance(training.get("save_freq"), int)
        or training.get("save_freq") <= 0
        or not isinstance(training.get("log_freq"), int)
        or isinstance(training.get("log_freq"), bool)
        or training.get("log_freq") <= 0
        or training.get("checkpoint_steps") != expected_checkpoints
        or training.get("eval_split") != 0.0
        or training.get("validation_gradients") is not False
        or training.get("hidden_test_loaded") is not False
        or validation.get("hidden_test_loaded") is not False
        or validation.get("total_samples")
        != len(validation_episodes) * len(validation.get("frame_offsets", []))
        or initialization.get("source") != "revision_pinned_base"
        or initialization.get("model_id") != experiment["model"]["identifier"]
        or initialization.get("model_revision") != experiment["model"]["revision"]
        or initialization.get("overfit_checkpoint_used") is not False
        or resources.get("memory_limit") != resources.get("memory_swap_limit")
        or resources.get("mixed_precision") != experiment["resources"]["mixed_precision"]
        or resources.get("cpu_limit") != experiment["resources"]["cpu_limit"]
        or plan.get("tracking", {}).get("space_id") != experiment["tracking"]["space_id"]
    ):
        raise ValueError(
            "Formal plan differs from the registered split, model, or resource contract."
        )
    if plan.get("performance_optimization") is None:
        if (
            training.get("batch_size") != 1
            or training.get("log_freq") != 1
            or resources.get("memory_limit") != experiment["resources"]["memory_limit"]
        ):
            raise ValueError(
                "Legacy formal plans must preserve the registered batch and resources."
            )
    else:
        if resources.get("memory_limit") != "8g":
            raise ValueError(
                "The optimized formal plan must stay within the authorized 8 GB limit."
            )
        _validate_performance_optimization(plan, experiment, base_path)
    return plan, base_path, experiment


def _validate_prerequisites(
    plan: dict[str, Any],
    experiment: dict[str, Any],
    base_path: Path,
    contract_sha256: str,
) -> dict[str, Path]:
    prerequisites = plan.get("prerequisites", {})
    paths: dict[str, Path] = {}
    for name in (
        "benchmark",
        "gate1",
        "gate2",
        "smoke_acceptance",
        "overfit_acceptance",
        "trackio_sync",
    ):
        value = prerequisites.get(name, {})
        path = _repository_path(str(value.get("path", "")))
        if file_sha256(path) != value.get("sha256"):
            raise ValueError(f"Formal prerequisite checksum changed: {name}.")
        paths[name] = path
    phase_runner._validate_benchmark(paths["benchmark"], experiment, base_path, contract_sha256)
    phase_runner._validate_gate(
        paths["gate1"],
        expected_gate="m2_gate_1_scripted_action",
        experiment_id=experiment["experiment_id"],
        contract_sha256=contract_sha256,
        dataset_revision=experiment["dataset"]["revision"],
    )
    phase_runner._validate_gate(
        paths["gate2"],
        expected_gate="m2_gate_2_dataset_action_replay",
        experiment_id=experiment["experiment_id"],
        contract_sha256=contract_sha256,
        dataset_revision=experiment["dataset"]["revision"],
    )
    phase_runner._validate_tracking(paths["trackio_sync"], experiment)
    phase_runner._validate_smoke_acceptance(
        paths["smoke_acceptance"], experiment, base_path, contract_sha256
    )
    overfit = _load_json(paths["overfit_acceptance"])
    acceptance = overfit.get("acceptance", {})
    required = experiment["phases"]["overfit"]["acceptance"]
    if (
        overfit.get("status") != "passed"
        or overfit.get("stage") != "smolvla_fixed_sample_overfit_acceptance"
        or overfit.get("experiment_id") != experiment["experiment_id"]
        or overfit.get("experiment_config_sha256") != file_sha256(base_path)
        or overfit.get("action_contract_sha256") != contract_sha256
        or not isinstance(acceptance, dict)
        or any(acceptance.get(criterion) is not True for criterion in required)
        or acceptance.get("hidden_test_loaded") is not False
    ):
        raise ValueError("Formal plan overfit prerequisite is invalid.")
    return paths


def _validate_normalization(
    plan: dict[str, Any],
    experiment: dict[str, Any],
    base_path: Path,
    contract_sha256: str,
) -> tuple[Path, Path, Path]:
    normalization = plan.get("normalization", {})
    report_path = _repository_path(str(normalization.get("report", "")))
    manifest_path = _repository_path(str(normalization.get("dataset_view_manifest", "")))
    if file_sha256(report_path) != normalization.get("report_sha256") or file_sha256(
        manifest_path
    ) != normalization.get("dataset_view_manifest_sha256"):
        raise ValueError("Formal train-only normalization checksum changed.")
    report = _load_json(report_path)
    manifest = _load_json(manifest_path)
    run_root = phase_runner._absolute_root("ROSETTA_RUN_ROOT")
    relative_view = Path(str(report.get("dataset_view", "")))
    if relative_view.is_absolute() or ".." in relative_view.parts:
        raise ValueError("Train-only dataset view path is unsafe.")
    view_root = (run_root / relative_view).resolve()
    if not view_root.is_relative_to(run_root) or manifest_path.parent.resolve() != view_root:
        raise ValueError("Train-only dataset view identity differs from the normalization report.")
    train_episodes = experiment["dataset"]["train_episodes"]
    if (
        report.get("status") != "complete"
        or report.get("stage") != "smolvla_train_only_normalization"
        or report.get("experiment_id") != experiment["experiment_id"]
        or report.get("experiment_config_sha256") != file_sha256(base_path)
        or report.get("dataset_revision") != experiment["dataset"]["revision"]
        or report.get("action_contract_sha256") != contract_sha256
        or report.get("source_split") != "train"
        or report.get("train_episodes") != train_episodes
        or report.get("train_rows")
        != report.get("effective_stats", {}).get("action", {}).get("count", [None])[0]
        or report.get("train_rows")
        != report.get("effective_stats", {}).get("observation.state", {}).get("count", [None])[0]
        or report.get("validation_episodes_loaded") is not False
        or report.get("hidden_test_loaded") is not False
        or manifest.get("status") != "complete"
        or manifest.get("stage") != "smolvla_train_only_dataset_view"
        or manifest.get("normalization_report_sha256") != file_sha256(report_path)
        or manifest.get("validation_episodes_loaded") is not False
        or manifest.get("hidden_test_loaded") is not False
    ):
        raise ValueError("Formal train-only normalization identity is invalid.")
    _training_coverage(plan["training"], int(report["train_rows"]))
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Train-only dataset view has no checksum inventory.")
    for raw_relative, expected_sha256 in files.items():
        relative = Path(str(raw_relative))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Train-only dataset view contains an unsafe file path.")
        path = view_root / relative
        if not path.is_file() or file_sha256(path) != expected_sha256:
            raise ValueError("Train-only dataset view file checksum changed.")
    view_stats = _load_json(view_root / "meta/stats.json")
    effective_stats = report.get("effective_stats", {})
    visual_features = report.get("visual_features", [])
    visual_statistics = report.get("visual_statistics", {})
    if (
        report.get("visual_statistics_policy") != "imagenet_constants"
        or report.get("visual_statistics_source") != "fixed_constants_not_dataset_rows"
        or not isinstance(effective_stats, dict)
        or not isinstance(visual_features, list)
        or not isinstance(visual_statistics, dict)
        or set(view_stats) != set(effective_stats) | set(visual_features)
        or any(view_stats.get(feature) != visual_statistics for feature in visual_features)
        or any(view_stats.get(feature) != value for feature, value in effective_stats.items())
    ):
        raise ValueError("Train-only dataset view stats differ from the normalization report.")
    return report_path, manifest_path, view_root


def _validate_preflight(
    report_path: Path,
    plan: dict[str, Any],
    experiment: dict[str, Any],
    base_path: Path,
    contract_sha256: str,
    normalization_sha256: str,
    plan_sha256: str,
) -> None:
    report = _load_json(report_path.resolve())
    if (
        report.get("status") != "passed"
        or report.get("stage") != "real_smolvla_no_optimizer_forward"
        or report.get("experiment_id") != experiment["experiment_id"]
        or report.get("experiment_config_sha256") != file_sha256(base_path)
        or report.get("action_contract_sha256") != contract_sha256
        or report.get("formal_plan_sha256") != plan_sha256
        or report.get("normalization_report_sha256") != normalization_sha256
        or report.get("episodes_loaded") != plan["preflight"]["episodes"]
        or report.get("hidden_test_loaded") is not False
        or report.get("network_disabled") is not True
        or report.get("optimizer_created") is not False
        or report.get("gradients_enabled") is not False
    ):
        raise ValueError("Formal train-only normalization preflight is invalid.")


def _validate_base_validation(
    report_path: Path,
    plan: dict[str, Any],
    experiment: dict[str, Any],
    base_path: Path,
    contract_sha256: str,
    normalization_sha256: str,
    plan_sha256: str,
) -> None:
    report = _load_json(report_path.resolve())
    metrics = report.get("metrics", {})
    validation = plan["validation"]
    required_metrics = {
        "action_mae",
        "action_rmse",
        "first_action_mae",
        "fixed_flow_loss",
        "invalid_action_rate",
        "joint_limit_violation_rate",
        "action_smoothness_mean_abs_delta",
        "inference_latency_mean_seconds",
        "inference_latency_p95_seconds",
    }
    if (
        report.get("status") != "complete"
        or report.get("stage") != "smolvla_fixed_validation"
        or report.get("experiment_id") != experiment["experiment_id"]
        or report.get("formal_plan_sha256") != plan_sha256
        or report.get("experiment_config_sha256") != file_sha256(base_path)
        or report.get("action_contract_sha256") != contract_sha256
        or report.get("normalization_report_sha256") != normalization_sha256
        or report.get("model_source", {}).get("kind") != "base"
        or report.get("model_source", {}).get("model_revision")
        != experiment["model"]["revision"]
        or report.get("validation_episodes") != validation["episodes"]
        or report.get("frame_offsets") != validation["frame_offsets"]
        or report.get("materialized_episodes") != sorted(validation["episodes"])
        or report.get("sample_count") != validation["total_samples"]
        or report.get("hidden_test_loaded") is not False
        or report.get("network_disabled") is not True
        or report.get("gradients_enabled") is not False
        or report.get("optimizer_created") is not False
        or report.get("trackio_local_logged") is not True
        or not isinstance(metrics, dict)
        or not required_metrics <= set(metrics)
        or any(
            not isinstance(metrics[name], int | float)
            or isinstance(metrics[name], bool)
            or not math.isfinite(float(metrics[name]))
            for name in required_metrics
        )
    ):
        raise ValueError("Formal base validation prerequisite is invalid.")


def _write_launch_manifest(
    mode: str,
    run_name: str,
    plan_path: Path,
    base_path: Path,
    experiment: dict[str, Any],
    contract_path: Path,
    normalization_report: Path,
    view_manifest: Path,
    prerequisites: dict[str, Path],
    code_identity: dict[str, Any],
    base_validation: Path | None = None,
    performance_optimization: dict[str, Any] | None = None,
) -> Path:
    run_root = phase_runner._absolute_root("ROSETTA_RUN_ROOT")
    report = {
        "schema_version": 1,
        "status": "preregistered",
        "stage": "smolvla_formal_launch",
        "mode": mode,
        "experiment_id": experiment["experiment_id"],
        "run_name": run_name,
        "formal_plan_sha256": file_sha256(plan_path),
        "experiment_config_sha256": file_sha256(base_path),
        "action_contract_sha256": file_sha256(contract_path),
        "normalization_report_sha256": file_sha256(normalization_report),
        "dataset_view_manifest_sha256": file_sha256(view_manifest),
        "model_revision": experiment["model"]["revision"],
        "dataset_revision": experiment["dataset"]["revision"],
        "prerequisites": {name: file_sha256(path) for name, path in sorted(prerequisites.items())},
        "base_validation_sha256": (
            file_sha256(base_validation) if base_validation is not None else None
        ),
        "code_identity": code_identity,
        "hidden_test_loaded": False,
        "performance_optimization": performance_optimization,
    }
    destination = run_root / str(experiment["experiment_id"]) / "launch" / f"{run_name}.json"
    create_json(destination, report)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "train"))
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--base-validation-report", type=Path)
    args = parser.parse_args()
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("HF_DATASETS_OFFLINE") != "1":
        raise RuntimeError("Formal SmolVLA work must run with networking disabled.")

    plan_path = args.plan.resolve()
    plan, base_path, experiment = _validate_plan(plan_path)
    resources = plan["resources"]
    if (
        os.environ.get("ROSETTA_DOCKER_MEMORY_LIMIT") != resources["memory_limit"]
        or os.environ.get("ROSETTA_DOCKER_MEMORY_SWAP_LIMIT") != resources["memory_swap_limit"]
    ):
        raise ValueError("The active Docker memory limits differ from the formal plan.")
    contract_path = REPOSITORY_ROOT / str(experiment["action_contract"]["derived"])
    contract_sha256 = file_sha256(contract_path)
    prerequisites = _validate_prerequisites(plan, experiment, base_path, contract_sha256)
    normalization_report, view_manifest, dataset_root = _validate_normalization(
        plan, experiment, base_path, contract_sha256
    )
    plan_sha256 = file_sha256(plan_path)
    normalization_sha256 = file_sha256(normalization_report)
    if args.mode == "train":
        if args.preflight_report is None:
            raise ValueError("Formal training requires --preflight-report.")
        if args.base_validation_report is None:
            raise ValueError("Formal training requires --base-validation-report.")
        _validate_preflight(
            args.preflight_report,
            plan,
            experiment,
            base_path,
            contract_sha256,
            normalization_sha256,
            plan_sha256,
        )
        _validate_base_validation(
            args.base_validation_report,
            plan,
            experiment,
            base_path,
            contract_sha256,
            normalization_sha256,
            plan_sha256,
        )

    model_root = phase_runner._model_root(experiment)
    checkpoint_root = phase_runner._absolute_root("ROSETTA_CHECKPOINT_ROOT")
    run_name = plan["preflight"]["run_name"] if args.mode == "preflight" else plan["run_name"]
    phase = "preflight" if args.mode == "preflight" else "formal"
    output_dir = checkpoint_root / str(experiment["experiment_id"]) / phase / str(run_name)
    if args.mode == "train" and output_dir.exists():
        raise FileExistsError(
            "The formal output already exists; the preregistered run is create-only."
        )
    code_identity = workspace_code_identity(REPOSITORY_ROOT)
    launch_manifest = _write_launch_manifest(
        args.mode,
        str(run_name),
        plan_path,
        base_path,
        experiment,
        contract_path,
        normalization_report,
        view_manifest,
        prerequisites,
        code_identity,
        args.base_validation_report,
        plan.get("performance_optimization"),
    )
    os.environ["ROSETTA_VLA_PHASE"] = "formal_preflight" if args.mode == "preflight" else "formal"
    os.environ["ROSETTA_VLA_EXPERIMENT_CONFIG"] = str(base_path)
    os.environ["ROSETTA_VLA_RUN_NAME"] = str(run_name)
    os.environ["ROSETTA_VLA_TRAIN_STATS_REPORT"] = str(normalization_report)
    os.environ["ROSETTA_VLA_FORMAL_PLAN_SHA256"] = plan_sha256
    os.environ["ROSETTA_VLA_NORMALIZATION_SHA256"] = normalization_sha256
    os.environ["ROSETTA_VLA_CODE_REVISION"] = str(code_identity["revision"])
    os.environ["ROSETTA_VLA_WORKSPACE_TREE_SHA256"] = str(code_identity["workspace_tree_sha256"])
    os.environ["ROSETTA_VLA_WORKSPACE_DIRTY"] = str(bool(code_identity["dirty"])).lower()
    os.environ["ROSETTA_VLA_WORKSPACE_FILE_COUNT"] = str(code_identity["workspace_file_count"])
    runtime_experiment = copy.deepcopy(experiment)
    runtime_experiment["resources"].update(resources)
    runtime_experiment["phases"]["formal"] = dict(plan["training"])
    if args.mode == "train" and plan.get("performance_optimization") is not None:
        policy = runtime_experiment["phases"]["formal"].pop("policy")
        skip_masked = bool(policy.pop("skip_fully_masked_camera_encoding"))
        runtime_experiment["model"]["policy"].update(policy)
        optimization = plan["performance_optimization"]
        cache_root = (
            phase_runner._absolute_root("ROSETTA_RUN_ROOT")
            / "compiler_cache"
            / str(optimization["compiler_cache_key"])
        )
        triton_cache = cache_root / "triton"
        inductor_cache = cache_root / "inductor"
        triton_cache.mkdir(parents=True, exist_ok=True)
        (inductor_cache / "cache").mkdir(parents=True, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = str(triton_cache)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_cache)
        os.environ["ROSETTA_VLA_SKIP_FULLY_MASKED_CAMERA_ENCODING"] = (
            "1" if skip_masked else "0"
        )
        os.environ["ROSETTA_VLA_PERFORMANCE_PLAN_SHA256"] = str(
            optimization["plan_sha256"]
        )
    else:
        os.environ["ROSETTA_VLA_SKIP_FULLY_MASKED_CAMERA_ENCODING"] = "0"
    sys.argv = [
        "lerobot-train",
        *phase_runner._phase_arguments(
            runtime_experiment,
            phase,
            str(run_name),
            model_root,
            dataset_root,
            output_dir,
        ),
    ]
    print(f"Launch manifest: {launch_manifest.relative_to(REPOSITORY_ROOT).as_posix()}")
    if args.mode == "preflight":
        from smolvla_forward_check import main as preflight_main

        return preflight_main()
    from train_smolvla_trackio import main as train_main

    train_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
