"""Launch the fresh-base Aster run: Faust protocol plus first-action temporal weighting."""

from __future__ import annotations

import argparse
import copy
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

import run_smolvla_action_repair_formal as faust_runner  # noqa: E402
import run_smolvla_formal as legacy_formal  # noqa: E402
import run_smolvla_phase as phase_runner  # noqa: E402

from rosetta_reality.experiment import file_sha256, workspace_code_identity  # noqa: E402
from rosetta_reality.features import create_json  # noqa: E402
from rosetta_reality.vla import (  # noqa: E402
    load_smolvla_action_space,
    load_smolvla_experiment,
)
from rosetta_reality.vla.horizon_loss import profile_from_plan  # noqa: E402

DEFAULT_PLAN = (
    REPOSITORY_ROOT / "configs/vla/smolvla_450m_aloha_insertion_aster_batch8_003.yaml"
)

_load_yaml = faust_runner._load_yaml
_load_json = faust_runner._load_json
_repository_path = faust_runner._repository_path
_optimizer_contract = faust_runner._optimizer_contract
_optimizer_arguments = faust_runner._optimizer_arguments
_validate_saved_optimizer_contract = faust_runner._validate_saved_optimizer_contract
_validate_monitoring = faust_runner._validate_monitoring
_validate_prerequisites = faust_runner._validate_prerequisites
_validate_normalization = faust_runner._validate_normalization
_validate_preflight = faust_runner._validate_preflight
_validate_base_validation = faust_runner._validate_base_validation


def _validate_optimizer_smoke(
    report_path: Path,
    plan: dict[str, Any],
    experiment: dict[str, Any],
    base_path: Path,
    contract_sha256: str,
    plan_sha256: str,
) -> None:
    report = _load_json(report_path.resolve())
    smoke = plan["optimizer_smoke"]
    if (
        report.get("status") != "passed"
        or report.get("stage")
        != "smolvla_horizon_loss_batch8_optimizer_smoke_acceptance"
        or report.get("experiment_id") != experiment["experiment_id"]
        or report.get("run_name") != smoke["run_name"]
        or report.get("formal_plan_sha256") != plan_sha256
        or report.get("experiment_config_sha256") != file_sha256(base_path)
        or report.get("action_contract_sha256") != contract_sha256
        or report.get("temporal_loss_profile")
        != plan["loss_contract"]["profile"]
        or report.get("temporal_loss_normalization")
        != plan["loss_contract"]["normalization"]
        or report.get("batch_size") != smoke["batch_size"]
        or report.get("steps") != smoke["steps"]
        or report.get("checkpoint_steps") != [1, 2]
        or report.get("all_metrics_finite") is not True
        or report.get("peak_xpu_within_guardrail") is not True
        or report.get("hidden_test_loaded") is not False
    ):
        raise ValueError("Aster batch-8 optimizer smoke evidence is invalid.")


def _prepare_compiler_cache(plan_path: Path) -> dict[str, str]:
    run_root = phase_runner._absolute_root("ROSETTA_RUN_ROOT")
    cache_root = run_root / "compiler_cache" / f"aster-b8-{file_sha256(plan_path)[:12]}"
    triton_cache = cache_root / "triton"
    inductor_cache = cache_root / "inductor"
    triton_cache.mkdir(parents=True, exist_ok=True)
    inductor_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(triton_cache)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_cache)
    return {
        "cache_root": cache_root.relative_to(run_root).as_posix(),
        "triton_cache": triton_cache.relative_to(run_root).as_posix(),
        "inductor_cache": inductor_cache.relative_to(run_root).as_posix(),
    }


def _validate_plan(plan_path: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    plan = _load_yaml(plan_path)
    parent = plan.get("parent_experiment", {})
    base_path = _repository_path(str(parent.get("config", "")))
    if file_sha256(base_path) != parent.get("sha256"):
        raise ValueError("Aster parent experiment checksum is stale.")
    experiment = load_smolvla_experiment(base_path, REPOSITORY_ROOT)
    action_space = load_smolvla_action_space(experiment, require_explicit=True)
    control = plan.get("control_reference", {})
    control_path = _repository_path(str(control.get("plan", "")))
    control_selection = _repository_path(str(control.get("selection_report", "")))
    training = plan.get("training", {})
    validation = plan.get("validation", {})
    preflight = plan.get("preflight", {})
    optimizer_smoke = plan.get("optimizer_smoke", {})
    resources = plan.get("resources", {})
    initialization = plan.get("initialization", {})
    monitoring = _validate_monitoring(plan)
    optimizer_contract = _optimizer_contract(training)
    train_episodes = [int(value) for value in training.get("episodes", [])]
    validation_episodes = [int(value) for value in validation.get("episodes", [])]
    hidden = {int(value) for value in experiment["dataset"]["test_episodes"]}
    expected_quarters = [int(training.get("steps", 0)) * value // 4 for value in range(1, 5)]
    policy = training.get("policy", {})
    implementation = plan.get("implementation_files", {})
    loss_contract = plan.get("loss_contract", {})
    supersedes = plan.get("supersedes", {})
    prior = plan.get("prior_invalid_axis_evidence", {})
    if (
        plan.get("schema_version") != 1
        or plan.get("role") != "vla"
        or plan.get("stage") != "m2_action_repair_development_training"
        or plan.get("status") != "preregistered"
        or plan.get("plan_id") != "m2-smolvla450m-aster-b8-003"
        or plan.get("run_name") != "m2-smolvla450m-aster-b8-003"
        or plan.get("furnace_program", {}).get("codename") != "Aster"
        or control.get("reason") != "single_variable_temporal_horizon_weighting"
        or control.get("control_run") != "m2-smolvla450m-faust-b8-002"
        or control.get("relationship") != "read_only_control"
        or file_sha256(control_path) != control.get("plan_sha256")
        or file_sha256(control_selection) != control.get("selection_report_sha256")
        or _load_json(control_selection).get("status") != "passed"
        or parent.get("experiment_id") != experiment["experiment_id"]
        or train_episodes != experiment["dataset"]["train_episodes"]
        or validation_episodes != experiment["dataset"]["validation_episodes"]
        or set(train_episodes) & set(validation_episodes)
        or (set(train_episodes) | set(validation_episodes)) & hidden
        or supersedes.get("plan")
        != "configs/vla/smolvla_450m_aloha_insertion_aster_batch8_002.yaml"
        or supersedes.get("plan_sha256")
        != "f80456c15225e5a3ff6cb4491434033a67a59349e3a58252770a82d37fcb54ae"
        or supersedes.get("preserve_all_prior_artifacts") is not True
        or prior.get("run_name") != "m2-smolvla450m-aster-b8-002"
        or prior.get("launch_workspace_tree_sha256")
        != "bac032b34c30946620a216d1146d0ab13c4b68f030767aafdcbb7d0dbb2a51f9"
        or prior.get("interpretation")
        != "exploratory_result_only_not_single_axis_acceptance"
        or preflight.get("run_name") != "m2-smolvla450m-aster-b8-preflight-005"
        or preflight.get("episodes") != experiment["phases"]["smoke"]["episodes"]
        or preflight.get("batch_size") != 1
        or preflight.get("optimizer_created") is not False
        or optimizer_smoke.get("run_name") != "m2-smolvla450m-aster-b8-smoke-004"
        or optimizer_smoke.get("episodes") != experiment["phases"]["smoke"]["episodes"]
        or optimizer_smoke.get("batch_size") != 8
        or optimizer_smoke.get("steps") != 2
        or optimizer_smoke.get("save_freq") != 1
        or optimizer_smoke.get("save_checkpoint") is not True
        or optimizer_smoke.get("num_workers") != 0
        or optimizer_smoke.get("persistent_workers") is not False
        or optimizer_smoke.get("hidden_test_loaded") is not False
        or training.get("batch_size") != 8
        or training.get("steps") != 2_500
        or training.get("minimum_dataset_passes") != 1.0
        or training.get("save_freq") != 625
        or training.get("checkpoint_steps") != expected_quarters
        or training.get("log_freq") != 10
        or training.get("num_workers") != 0
        or training.get("persistent_workers") is not False
        or training.get("eval_split") != 0.0
        or training.get("validation_gradients") is not False
        or training.get("hidden_test_loaded") is not False
        or policy
        != {
            "empty_cameras": 2,
            "compile_model": True,
            "compile_mode": "reduce-overhead",
            "skip_fully_masked_camera_encoding": True,
        }
        or optimizer_contract is None
        or monitoring is None
        or initialization.get("source") != "revision_pinned_base"
        or initialization.get("model_id") != experiment["model"]["identifier"]
        or initialization.get("model_revision") != experiment["model"]["revision"]
        or initialization.get("overfit_checkpoint_used") is not False
        or resources.get("runtime") != "docker_linux_from_wsl"
        or resources.get("accelerator") != "xpu"
        or resources.get("memory_limit") != "8g"
        or resources.get("memory_swap_limit") != "8g"
        or resources.get("mixed_precision") != experiment["resources"]["mixed_precision"]
        or resources.get("cpu_limit") != experiment["resources"]["cpu_limit"]
        or not isinstance(resources.get("maximum_peak_xpu_allocated_bytes"), int)
        or resources["maximum_peak_xpu_allocated_bytes"] > 7 * 1024**3
        or validation.get("total_samples")
        != len(validation_episodes) * len(validation.get("frame_offsets", []))
        or validation.get("hidden_test_loaded") is not False
        or validation.get("primary_selection_metric") != "first_action_mae"
        or validation.get("secondary_selection_metric") != "action_mae"
        or validation.get("run_name_prefix")
        != "m2-smolvla450m-aster-b8-validation-004"
        or plan.get("tracking", {}).get("space_id") != experiment["tracking"]["space_id"]
        or plan.get("action_space") != action_space.as_dict()
        or not isinstance(implementation, dict)
        or not implementation
        or not isinstance(loss_contract, dict)
        or loss_contract.get("profile") != "first_action_only"
        or loss_contract.get("chunk_size") != 50
        or loss_contract.get("normalization")
        != "mean_over_selected_valid_entries"
    ):
        raise ValueError("Aster plan differs from its single-variable loss contract.")
    profile_from_plan(plan, int(experiment["model"]["policy"]["chunk_size"]))
    if legacy_formal._validate_furnace_program(plan, plan_path) is None:
        raise ValueError("Aster plan has no bound three-furnace registry entry.")
    for raw_path, expected_sha256 in implementation.items():
        path = _repository_path(str(raw_path))
        if file_sha256(path) != expected_sha256:
            raise ValueError(f"Aster implementation checksum changed: {raw_path}.")
    return plan, base_path, experiment


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
    prerequisites: dict[str, Path],
    base_validation: Path | None,
    optimizer_smoke: Path | None,
    compiler_cache: dict[str, str],
) -> Path:
    report = {
        "schema_version": 1,
        "status": "preregistered",
        "stage": "smolvla_horizon_loss_formal_launch",
        "mode": mode,
        "experiment_id": experiment["experiment_id"],
        "run_name": run_name,
        "codename": "Aster",
        "loss_contract": plan["loss_contract"],
        "control_reference": plan["control_reference"],
        "furnace_program": legacy_formal._validate_furnace_program(plan, plan_path),
        "formal_plan_sha256": file_sha256(plan_path),
        "experiment_config_sha256": file_sha256(base_path),
        "action_contract_sha256": file_sha256(contract_path),
        "normalization_report_sha256": file_sha256(normalization_report),
        "dataset_view_manifest_sha256": file_sha256(view_manifest),
        "prerequisites": {
            name: file_sha256(path) for name, path in sorted(prerequisites.items())
        },
        "base_validation_sha256": (
            file_sha256(base_validation) if base_validation is not None else None
        ),
        "optimizer_smoke_sha256": (
            file_sha256(optimizer_smoke) if optimizer_smoke is not None else None
        ),
        "compiler_cache": compiler_cache,
        "initialization": plan["initialization"],
        "optimizer_contract": _optimizer_contract(plan["training"]),
        "monitoring": _validate_monitoring(plan),
        "action_space": load_smolvla_action_space(
            experiment, require_explicit=True
        ).as_dict(),
        "code_identity": workspace_code_identity(REPOSITORY_ROOT),
        "hidden_test_loaded": False,
    }
    destination = (
        phase_runner._absolute_root("ROSETTA_RUN_ROOT")
        / str(experiment["experiment_id"])
        / "launch"
        / f"{run_name}.json"
    )
    create_json(destination, report)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "smoke", "train"))
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--base-validation-report", type=Path)
    parser.add_argument("--optimizer-smoke-report", type=Path)
    args = parser.parse_args()
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("HF_DATASETS_OFFLINE") != "1":
        raise RuntimeError("Aster must run offline inside the registered Docker boundary.")
    plan_path = args.plan.resolve()
    plan, base_path, experiment = _validate_plan(plan_path)
    resources = plan["resources"]
    if (
        os.environ.get("ROSETTA_DOCKER_MEMORY_LIMIT") != resources["memory_limit"]
        or os.environ.get("ROSETTA_DOCKER_MEMORY_SWAP_LIMIT")
        != resources["memory_swap_limit"]
    ):
        raise ValueError("The active memory limit differs from Aster.")
    contract_path = REPOSITORY_ROOT / str(experiment["action_contract"]["derived"])
    contract_sha256 = file_sha256(contract_path)
    prerequisites = _validate_prerequisites(
        plan, experiment, base_path, contract_sha256
    )
    normalization, view_manifest, dataset_root = _validate_normalization(
        plan, experiment, base_path, contract_sha256
    )
    plan_sha256 = file_sha256(plan_path)
    if args.mode in {"smoke", "train"}:
        if args.preflight_report is None or args.base_validation_report is None:
            raise ValueError(
                "Aster optimizer work requires preflight and base validation reports."
            )
        _validate_preflight(
            args.preflight_report.resolve(),
            plan,
            experiment,
            base_path,
            contract_sha256,
            file_sha256(normalization),
            plan_sha256,
        )
        _validate_base_validation(
            args.base_validation_report.resolve(),
            plan,
            experiment,
            base_path,
            contract_sha256,
            file_sha256(normalization),
            plan_sha256,
        )
    if args.mode == "train":
        if args.optimizer_smoke_report is None:
            raise ValueError("Aster batch-8 training requires optimizer smoke evidence.")
        _validate_optimizer_smoke(
            args.optimizer_smoke_report.resolve(),
            plan,
            experiment,
            base_path,
            contract_sha256,
            plan_sha256,
        )
    run_name = (
        plan["preflight"]["run_name"]
        if args.mode == "preflight"
        else (
            plan["optimizer_smoke"]["run_name"]
            if args.mode == "smoke"
            else plan["run_name"]
        )
    )
    phase = (
        "preflight"
        if args.mode == "preflight"
        else ("smoke" if args.mode == "smoke" else "formal")
    )
    output_dir = (
        phase_runner._absolute_root("ROSETTA_CHECKPOINT_ROOT")
        / str(experiment["experiment_id"])
        / phase
        / str(run_name)
    )
    if output_dir.exists():
        raise FileExistsError("The Aster output is create-only.")
    compiler_cache = _prepare_compiler_cache(plan_path)
    launch = _write_launch_manifest(
        args.mode,
        str(run_name),
        plan,
        plan_path,
        base_path,
        experiment,
        contract_path,
        normalization,
        view_manifest,
        prerequisites,
        args.base_validation_report,
        args.optimizer_smoke_report,
        compiler_cache,
    )
    identity = workspace_code_identity(REPOSITORY_ROOT)
    os.environ.update(
        {
            "ROSETTA_VLA_PHASE": (
                "formal_preflight"
                if args.mode == "preflight"
                else ("performance_benchmark" if args.mode == "smoke" else "formal")
            ),
            "ROSETTA_VLA_EXPERIMENT_CONFIG": str(base_path),
            "ROSETTA_VLA_RUN_NAME": str(run_name),
            "ROSETTA_VLA_TRAIN_STATS_REPORT": str(normalization),
            "ROSETTA_VLA_NORMALIZATION_SHA256": file_sha256(normalization),
            "ROSETTA_VLA_FORMAL_PLAN_PATH": str(plan_path),
            "ROSETTA_VLA_FORMAL_PLAN_SHA256": plan_sha256,
            "ROSETTA_VLA_HORIZON_LOSS_AUTHORIZED": "1",
            "ROSETTA_VLA_SKIP_FULLY_MASKED_CAMERA_ENCODING": "0",
            "ROSETTA_VLA_CODE_REVISION": str(identity["revision"]),
            "ROSETTA_VLA_WORKSPACE_TREE_SHA256": str(identity["workspace_tree_sha256"]),
            "ROSETTA_VLA_WORKSPACE_DIRTY": str(bool(identity["dirty"])).lower(),
            "ROSETTA_VLA_WORKSPACE_FILE_COUNT": str(identity["workspace_file_count"]),
        }
    )
    print(f"Compiler cache: {compiler_cache['cache_root']}")
    runtime = copy.deepcopy(experiment)
    runtime["resources"].update(resources)
    runtime_phase = "formal" if args.mode != "smoke" else "performance_benchmark"
    runtime["phases"][runtime_phase] = dict(
        plan["training"] if args.mode != "smoke" else plan["optimizer_smoke"]
    )
    policy = runtime["phases"][runtime_phase].pop("policy", plan["training"]["policy"])
    runtime["model"]["policy"].update(policy)
    os.environ["ROSETTA_VLA_SKIP_FULLY_MASKED_CAMERA_ENCODING"] = str(
        int(bool(policy.get("skip_fully_masked_camera_encoding", False)))
    )
    arguments = phase_runner._phase_arguments(
        runtime,
        "preflight" if args.mode == "preflight" else runtime_phase,
        str(run_name),
        phase_runner._model_root(experiment),
        dataset_root,
        output_dir,
    )
    arguments.extend(_optimizer_arguments(plan["training"]))
    sys.argv = ["lerobot-train", *arguments]
    print(f"Launch manifest: {launch.name}")
    if args.mode == "preflight":
        from smolvla_forward_check import main as preflight_main

        return preflight_main()
    from train_smolvla_horizon_loss_formal import main as train_main

    train_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
