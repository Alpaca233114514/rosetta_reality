"""Launch a validated version-2 SmolVLA preflight, optimizer smoke or training run.

This is the single version-2 launcher.  It consolidates the validation chain
that the historical ``run_smolvla_formal`` / ``run_smolvla_action_repair_*`` /
``run_smolvla_horizon_loss_formal`` / ``run_smolvla_state_robustness_*``
launchers each duplicated, then hands the assembled process to the single
``train_smolvla_v2`` trainer entry:

    offline guard -> plan schema -> parent binding -> split guards
        -> resource guard -> prerequisite evidence -> normalization identity
        -> coverage -> mode reports -> runtime experiment file
        -> launch manifest -> environment -> lerobot-train CLI -> trainer

Deep prerequisite-report semantics reuse the frozen validators in
``run_smolvla_phase`` read-only instead of duplicating them.  No step of this
launcher authorizes a new furnace by itself: the plan, its prerequisites and
its stop conditions are the authorization record.
"""

from __future__ import annotations

import argparse
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
for root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import run_smolvla_phase as phase_runner  # noqa: E402

from rosetta_reality.experiment import file_sha256, workspace_code_identity  # noqa: E402
from rosetta_reality.features import create_json  # noqa: E402
from rosetta_reality.vla.action_space import load_smolvla_experiment  # noqa: E402
from rosetta_reality.vla.training.context import PHASE_FORMAL, PHASE_SMOKE  # noqa: E402
from rosetta_reality.vla.training.features import FEATURE_FACTORIES  # noqa: E402
from rosetta_reality.vla.training.launch import (  # noqa: E402
    MODE_PREFLIGHT,
    MODE_SMOKE,
    MODE_TRAIN,
    build_training_arguments,
    compose_runtime_experiment,
)
from rosetta_reality.vla.training.plan import (  # noqa: E402
    FEATURE_MASKED_CAMERA_SKIP,
    load_v2_plan,
    repository_relative_path,
    training_coverage,
    validate_optimizer_contract,
    validate_plan_structure,
)

PLAN_PATH_ENV = "ROSETTA_VLA_V2_PLAN_PATH"
LAUNCHER_VALIDATED_ENV = "ROSETTA_VLA_V2_LAUNCHER_VALIDATED"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object: {path.name}.")
    json.dumps(value, allow_nan=False)
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping: {path.name}.")
    return value


def _declared_feature_names(plan: dict[str, Any]) -> list[str]:
    return [
        declaration["name"]
        for declaration in plan["features"]
        if isinstance(declaration, dict)
    ]


def _resolve_plan(plan_path: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    plan = load_v2_plan(plan_path, REPOSITORY_ROOT)
    validate_plan_structure(plan, known_features=FEATURE_FACTORIES)
    parent = plan["parent_experiment"]
    base_relative = repository_relative_path(
        parent["config"], context="Parent experiment config"
    )
    base_path = (REPOSITORY_ROOT / base_relative).resolve()
    if not base_path.is_relative_to(REPOSITORY_ROOT):
        raise ValueError("Parent experiment path escaped the repository root.")
    if file_sha256(base_path) != parent["sha256"]:
        raise ValueError("Version-2 plan parent experiment checksum is stale.")
    experiment = load_smolvla_experiment(base_path, REPOSITORY_ROOT)
    if parent["experiment_id"] != experiment["experiment_id"]:
        raise ValueError("Version-2 plan parent experiment identity differs.")
    return plan, base_path, experiment


def _validate_split(plan: dict[str, Any], experiment: dict[str, Any], mode: str) -> None:
    dataset = experiment["dataset"]
    train_episodes = [int(value) for value in dataset["train_episodes"]]
    validation_episodes = {int(value) for value in dataset["validation_episodes"]}
    test_episodes = {int(value) for value in dataset["test_episodes"]}
    phase_key = {
        MODE_PREFLIGHT: "preflight",
        MODE_SMOKE: "optimizer_smoke",
        MODE_TRAIN: "training",
    }[mode]
    episodes = [int(value) for value in plan[phase_key]["episodes"]]
    episode_set = set(episodes)
    if episode_set & test_episodes:
        raise ValueError("A version-2 phase attempted to load hidden-test episodes.")
    if episode_set & validation_episodes:
        raise ValueError("A version-2 phase attempted to load validation episodes.")
    if mode == MODE_TRAIN:
        if episodes != train_episodes:
            raise ValueError(
                "Version-2 formal training must use the registered train split exactly."
            )
    elif not episode_set.issubset(set(train_episodes)):
        raise ValueError("Version-2 smoke/preflight episodes must come from the train split.")
    validation = plan["validation"]
    if set(int(value) for value in validation["episodes"]) & (
        set(train_episodes) | episode_set
    ):
        raise ValueError("Version-2 validation episodes must stay disjoint from training.")
    if set(int(value) for value in validation["episodes"]) & test_episodes:
        raise ValueError("Version-2 validation episodes must stay disjoint from hidden test.")


def _validate_prerequisites(
    plan: dict[str, Any], experiment: dict[str, Any], base_path: Path, contract_sha256: str
) -> dict[str, Path]:
    prerequisites = plan.get("prerequisites", {})
    paths: dict[str, Path] = {}
    for name, declaration in sorted(prerequisites.items()):
        relative = repository_relative_path(
            declaration["path"], context=f"Prerequisite '{name}'"
        )
        path = (REPOSITORY_ROOT / relative).resolve()
        if not path.is_relative_to(REPOSITORY_ROOT) or not path.is_file():
            raise FileNotFoundError(relative.as_posix())
        if file_sha256(path) != declaration["sha256"]:
            raise ValueError(f"Version-2 prerequisite checksum changed: {name}.")
        paths[name] = path
    if "benchmark" in paths:
        phase_runner._validate_benchmark(
            paths["benchmark"], experiment, base_path, contract_sha256
        )
    allowed_replay = [
        *experiment["dataset"]["train_episodes"],
        *experiment["dataset"]["validation_episodes"],
    ]
    for gate_key, expected_gate in (
        ("gate1", "m2_gate_1_scripted_action"),
        ("gate2", "m2_gate_2_dataset_action_replay"),
    ):
        if gate_key in paths:
            phase_runner._validate_gate(
                paths[gate_key],
                expected_gate=expected_gate,
                experiment_id=experiment["experiment_id"],
                contract_sha256=contract_sha256,
                dataset_revision=experiment["dataset"]["revision"],
                allowed_replay_episodes=allowed_replay,
            )
    if "trackio_sync" in paths:
        phase_runner._validate_tracking(paths["trackio_sync"], experiment)
    if "smoke_acceptance" in paths:
        phase_runner._validate_smoke_acceptance(
            paths["smoke_acceptance"], experiment, base_path, contract_sha256
        )
    return paths


def _validate_normalization(
    plan: dict[str, Any], experiment: dict[str, Any], base_path: Path, contract_sha256: str
) -> tuple[Path, Path]:
    normalization = plan["normalization"]
    report_path = (
        REPOSITORY_ROOT / repository_relative_path(
            normalization["report"], context="Normalization report"
        )
    ).resolve()
    manifest_path = (
        REPOSITORY_ROOT / repository_relative_path(
            normalization["dataset_view_manifest"], context="Normalization view manifest"
        )
    ).resolve()
    for path, expected, label in (
        (report_path, normalization["report_sha256"], "normalization report"),
        (manifest_path, normalization["dataset_view_manifest_sha256"], "view manifest"),
    ):
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"Version-2 {label} checksum changed or is missing.")
    report = _load_json(report_path)
    manifest = _load_json(manifest_path)
    run_root = phase_runner._absolute_root("ROSETTA_RUN_ROOT")
    relative_view = Path(str(report.get("dataset_view", "")))
    if relative_view.is_absolute() or ".." in relative_view.parts:
        raise ValueError("Train-only dataset view path is unsafe.")
    view_root = (run_root / relative_view).resolve()
    if not view_root.is_relative_to(run_root) or manifest_path.parent.resolve() != view_root:
        raise ValueError(
            "Train-only dataset view identity differs from the normalization report."
        )
    train_episodes = experiment["dataset"]["train_episodes"]
    effective_stats = report.get("effective_stats", {})
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
        != effective_stats.get("action", {}).get("count", [None])[0]
        or report.get("train_rows")
        != effective_stats.get("observation.state", {}).get("count", [None])[0]
        or report.get("validation_episodes_loaded") is not False
        or report.get("hidden_test_loaded") is not False
        or manifest.get("status") != "complete"
        or manifest.get("stage") != "smolvla_train_only_dataset_view"
        or manifest.get("normalization_report_sha256") != file_sha256(report_path)
        or manifest.get("validation_episodes_loaded") is not False
        or manifest.get("hidden_test_loaded") is not False
    ):
        raise ValueError("Version-2 train-only normalization identity is invalid.")
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


def _validate_preflight_report(
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
        raise ValueError("Version-2 no-optimizer preflight report is invalid.")


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
        raise ValueError("Version-2 base validation prerequisite is invalid.")


def _write_launch_manifest(
    mode: str,
    run_name: str,
    plan: dict[str, Any],
    plan_path: Path,
    base_path: Path,
    experiment: dict[str, Any],
    contract_path: Path,
    normalization_report: Path,
    view_manifest: Path,
    runtime_experiment_path: Path,
    prerequisites: dict[str, Path],
    code_identity: dict[str, Any],
    base_validation: Path | None,
) -> Path:
    run_root = phase_runner._absolute_root("ROSETTA_RUN_ROOT")
    report = {
        "schema_version": 1,
        "status": "preregistered",
        "stage": "smolvla_v2_formal_launch",
        "mode": mode,
        "experiment_id": experiment["experiment_id"],
        "run_name": run_name,
        "formal_plan_sha256": file_sha256(plan_path),
        "experiment_config_sha256": file_sha256(base_path),
        "runtime_experiment_sha256": file_sha256(runtime_experiment_path),
        "action_contract_sha256": file_sha256(contract_path),
        "normalization_report_sha256": file_sha256(normalization_report),
        "dataset_view_manifest_sha256": file_sha256(view_manifest),
        "model_revision": experiment["model"]["revision"],
        "dataset_revision": experiment["dataset"]["revision"],
        "features": _declared_feature_names(plan),
        "prerequisites": {
            name: file_sha256(path) for name, path in sorted(prerequisites.items())
        },
        "base_validation_sha256": (
            file_sha256(base_validation) if base_validation is not None else None
        ),
        "code_identity": code_identity,
        "optimizer_contract": validate_optimizer_contract(plan["training"]),
        "plan_inheritance": plan.get("plan_inheritance"),
        "hidden_test_loaded": False,
    }
    destination = (
        run_root / str(experiment["experiment_id"]) / "launch" / f"{run_name}.json"
    )
    create_json(destination, report)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=(MODE_PREFLIGHT, MODE_SMOKE, MODE_TRAIN))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--base-validation-report", type=Path)
    args = parser.parse_args()
    if (
        os.environ.get("HF_HUB_OFFLINE") != "1"
        or os.environ.get("HF_DATASETS_OFFLINE") != "1"
    ):
        raise RuntimeError("Version-2 SmolVLA work must run with networking disabled.")

    plan_path = args.plan.resolve()
    plan, base_path, experiment = _resolve_plan(plan_path)
    resources = plan["resources"]
    if (
        os.environ.get("ROSETTA_DOCKER_MEMORY_LIMIT") != resources["memory_limit"]
        or os.environ.get("ROSETTA_DOCKER_MEMORY_SWAP_LIMIT")
        != resources["memory_swap_limit"]
    ):
        raise ValueError("The active Docker memory limits differ from the version-2 plan.")
    _validate_split(plan, experiment, args.mode)

    contract_path = REPOSITORY_ROOT / str(experiment["action_contract"]["derived"])
    contract_sha256 = file_sha256(contract_path)
    prerequisites = _validate_prerequisites(plan, experiment, base_path, contract_sha256)
    normalization_report, view_manifest, dataset_view_root = _validate_normalization(
        plan, experiment, base_path, contract_sha256
    )
    plan_sha256 = file_sha256(plan_path)
    normalization_sha256 = file_sha256(normalization_report)
    if args.mode == MODE_TRAIN:
        training_coverage(plan["training"], int(_load_json(normalization_report)["train_rows"]))
        if args.preflight_report is None or args.base_validation_report is None:
            raise ValueError("Version-2 training requires preflight and base-validation reports.")
        _validate_preflight_report(
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
    # Validates the immutable data-cache identity; every phase consumes the
    # validated train-only view resolved from the normalization report, the
    # same dataset surface the registered no-optimizer preflight checks.
    phase_runner._dataset_root(experiment)
    checkpoint_root = phase_runner._absolute_root("ROSETTA_CHECKPOINT_ROOT")
    run_root = phase_runner._absolute_root("ROSETTA_RUN_ROOT")
    run_name = {
        MODE_PREFLIGHT: str(plan["preflight"]["run_name"]),
        MODE_SMOKE: str(plan["optimizer_smoke"]["run_name"]),
        MODE_TRAIN: str(plan["run_name"]),
    }[args.mode]
    phase = {MODE_PREFLIGHT: "formal_preflight", MODE_SMOKE: PHASE_SMOKE, MODE_TRAIN: PHASE_FORMAL}[
        args.mode
    ]
    output_dir = checkpoint_root / str(experiment["experiment_id"]) / phase / run_name
    if args.mode in {MODE_SMOKE, MODE_TRAIN} and output_dir.exists():
        raise FileExistsError(
            "The requested version-2 output already exists; runs are create-only."
        )

    code_identity = workspace_code_identity(REPOSITORY_ROOT)
    runtime_experiment = compose_runtime_experiment(experiment, plan)
    runtime_experiment_path = (
        run_root
        / str(experiment["experiment_id"])
        / "launch"
        / f"{run_name}-runtime-experiment.json"
    )
    create_json(runtime_experiment_path, runtime_experiment)
    launch_manifest = _write_launch_manifest(
        args.mode,
        run_name,
        plan,
        plan_path,
        base_path,
        experiment,
        contract_path,
        normalization_report,
        view_manifest,
        runtime_experiment_path,
        prerequisites,
        code_identity,
        args.base_validation_report,
    )

    device = os.environ.get("ROSETTA_TORCH_DEVICE")
    if not device:
        raise ValueError("ROSETTA_TORCH_DEVICE must be set by the Docker runner.")
    os.environ["ROSETTA_VLA_PHASE"] = phase
    # The composed runtime experiment stays under the durable run root as
    # evidence; consumers of this environment variable require the in-repo
    # parent config, matching the registered historical launcher boundary.
    os.environ["ROSETTA_VLA_EXPERIMENT_CONFIG"] = str(base_path)
    os.environ["ROSETTA_VLA_RUN_NAME"] = run_name
    os.environ["ROSETTA_VLA_TRAIN_STATS_REPORT"] = str(normalization_report)
    os.environ["ROSETTA_VLA_FORMAL_PLAN_SHA256"] = plan_sha256
    os.environ["ROSETTA_VLA_NORMALIZATION_SHA256"] = normalization_sha256
    os.environ["ROSETTA_VLA_CODE_REVISION"] = str(code_identity["revision"])
    os.environ["ROSETTA_VLA_WORKSPACE_TREE_SHA256"] = str(
        code_identity["workspace_tree_sha256"]
    )
    os.environ["ROSETTA_VLA_WORKSPACE_DIRTY"] = str(bool(code_identity["dirty"])).lower()
    os.environ["ROSETTA_VLA_WORKSPACE_FILE_COUNT"] = str(
        code_identity["workspace_file_count"]
    )
    os.environ["ROSETTA_VLA_SKIP_FULLY_MASKED_CAMERA_ENCODING"] = (
        "1" if FEATURE_MASKED_CAMERA_SKIP in _declared_feature_names(plan) else "0"
    )
    os.environ[PLAN_PATH_ENV] = str(plan_path)
    os.environ[LAUNCHER_VALIDATED_ENV] = "1"
    training_policy = plan["training"].get("policy", {})
    if training_policy.get("compile_model"):
        cache_root = run_root / "compiler_cache" / f"v2-{plan_sha256[:12]}"
        triton_cache = cache_root / "triton"
        inductor_cache = cache_root / "inductor"
        triton_cache.mkdir(parents=True, exist_ok=True)
        (inductor_cache / "cache").mkdir(parents=True, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = str(triton_cache)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_cache)

    training_arguments = build_training_arguments(
        plan,
        experiment,
        mode=args.mode,
        run_name=run_name,
        model_root=model_root,
        dataset_root=dataset_view_root,
        output_dir=output_dir,
        device=device,
    )
    sys.argv = ["lerobot-train", *training_arguments]
    print(f"Launch manifest: {launch_manifest.name}")
    if args.mode == MODE_PREFLIGHT:
        from smolvla_forward_check import main as preflight_main

        return preflight_main()
    from train_smolvla_v2 import main as train_main

    train_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
