"""Launch a smoke-bound, fresh-base Way formal run on AutoDL CUDA."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import run_smolvla_action_repair_formal as aster_contract  # noqa: E402
import run_smolvla_action_repair_phase as repair_phase  # noqa: E402
import run_smolvla_phase as phase_runner  # noqa: E402

from rosetta_reality.experiment import (  # noqa: E402
    file_sha256,
    workspace_code_identity,
)
from rosetta_reality.features import create_json  # noqa: E402
from rosetta_reality.vla import load_smolvla_experiment  # noqa: E402
from rosetta_reality.vla.horizon_loss import profile_from_plan as horizon_profile  # noqa: E402
from rosetta_reality.vla.state_robustness import profile_from_plan as state_profile  # noqa: E402

DEFAULT_PLAN = (
    REPOSITORY_ROOT
    / "configs/vla/smolvla_450m_aloha_insertion_way_cuda_batch128_formal_001.yaml"
)
RUN_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{2,79}")
_FORMAL_SPECS = {
    "m2-smolvla450m-way-cuda-b128-formal-001": {
        "run_name": "m2-smolvla450m-way-cuda-b128-formal-001",
        "batch_size": 128,
        "steps": 160,
        "save_freq": 40,
        "checkpoint_steps": [40, 80, 120, 160],
        "warmup_steps": 8,
        "sample_exposures": 20_480,
        "smoke_plan_id": "m2-smolvla450m-way-cuda-b128-smoke-001",
        "smoke_run_name": "m2-smolvla450m-way-cuda-b128-smoke-001",
        "validation_prefix": "m2-smolvla450m-way-cuda-b128-validation-001",
        "compile_mode": "reduce-overhead",
    },
    "m2-smolvla450m-way-cuda-b64-formal-001": {
        "run_name": "m2-smolvla450m-way-cuda-b64-formal-001",
        "batch_size": 64,
        "steps": 316,
        "save_freq": 79,
        "checkpoint_steps": [79, 158, 237, 316],
        "warmup_steps": 16,
        "sample_exposures": 20_224,
        "smoke_plan_id": "m2-smolvla450m-way-cuda-b64-smoke-001",
        "smoke_run_name": "m2-smolvla450m-way-cuda-b64-smoke-001",
        "validation_prefix": "m2-smolvla450m-way-cuda-b64-validation-001",
        "compile_mode": "reduce-overhead",
    },
    "m2-smolvla450m-way-cuda-b64-default-formal-002": {
        "run_name": "m2-smolvla450m-way-cuda-b64-default-formal-002",
        "batch_size": 64,
        "steps": 316,
        "save_freq": 79,
        "checkpoint_steps": [79, 158, 237, 316],
        "warmup_steps": 16,
        "sample_exposures": 20_224,
        "smoke_plan_id": "m2-smolvla450m-way-cuda-b64-default-smoke-002",
        "smoke_run_name": "m2-smolvla450m-way-cuda-b64-default-smoke-002",
        "validation_prefix": "m2-smolvla450m-way-cuda-b64-default-validation-002",
        "compile_mode": "default",
    },
}

_load_yaml = aster_contract._load_yaml
_load_json = aster_contract._load_json
_optimizer_contract = aster_contract._optimizer_contract
_optimizer_arguments = aster_contract._optimizer_arguments
_validate_saved_optimizer_contract = aster_contract._validate_saved_optimizer_contract


def _repository_path(raw: str) -> Path:
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Way formal paths must be safe repository-relative paths.")
    if relative.parts and relative.parts[0] == "runs":
        run_root_raw = os.environ.get("ROSETTA_RUN_ROOT")
        if run_root_raw:
            run_root = Path(run_root_raw).resolve()
            path = (run_root / Path(*relative.parts[1:])).resolve()
            if path.is_relative_to(run_root) and path.is_file():
                return path
    path = (REPOSITORY_ROOT / relative).resolve()
    if not path.is_relative_to(REPOSITORY_ROOT) or not path.is_file():
        raise FileNotFoundError(f"Way formal file is missing: {relative.as_posix()}.")
    return path


def _validate_prerequisites(
    plan: dict[str, Any],
    experiment: dict[str, Any],
    base_path: Path,
    contract_sha256: str,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, record in plan.get("prerequisites", {}).items():
        if not isinstance(record, dict):
            raise ValueError("Way prerequisite records must be mappings.")
        path = _repository_path(str(record.get("path", "")))
        if file_sha256(path) != record.get("sha256"):
            raise ValueError(f"Way prerequisite checksum changed: {name}.")
        paths[str(name)] = path
    required = {"normalization", "dataset_view_manifest", "action_space", "gate1", "gate2"}
    if set(paths) != required:
        raise ValueError("Way formal prerequisite set is incomplete.")
    action_space_report = _load_json(paths["action_space"])
    if (
        action_space_report.get("experiment_id") != experiment["experiment_id"]
        or action_space_report.get("experiment_config_sha256") != file_sha256(base_path)
        or action_space_report.get("action_contract_sha256") != contract_sha256
        or action_space_report.get("hidden_test_loaded") is not False
    ):
        raise ValueError("Way action-space prerequisite is invalid.")
    for gate_name, expected_gate in (
        ("gate1", "m2_gate_1_scripted_action"),
        ("gate2", "m2_gate_2_dataset_action_replay"),
    ):
        phase_runner._validate_gate(
            paths[gate_name],
            expected_gate=expected_gate,
            experiment_id=experiment["experiment_id"],
            contract_sha256=contract_sha256,
            dataset_revision=experiment["dataset"]["revision"],
            allowed_replay_episodes=[
                *experiment["dataset"]["train_episodes"],
                *experiment["dataset"]["validation_episodes"],
            ],
        )
    return paths


def _validate_normalization(
    plan: dict[str, Any],
    experiment: dict[str, Any],
    base_path: Path,
    contract_sha256: str,
) -> tuple[Path, Path, Path]:
    paths = _validate_prerequisites(plan, experiment, base_path, contract_sha256)
    normalization = paths["normalization"]
    view_manifest = paths["dataset_view_manifest"]
    dataset_root = repair_phase._validate_repair_evidence(
        experiment,
        base_path,
        normalization,
        paths["action_space"],
    )
    report = _load_json(normalization)
    view = _load_json(view_manifest)
    if (
        report.get("status") != "complete"
        or report.get("validation_episodes_loaded") is not False
        or report.get("hidden_test_loaded") is not False
        or view.get("status") != "complete"
        or view.get("hidden_test_loaded") is not False
    ):
        raise ValueError("Way train-only normalization or view manifest is invalid.")
    return normalization, view_manifest, dataset_root


def _validate_preflight(
    report_path: Path,
    plan: dict[str, Any],
    experiment: dict[str, Any],
    base_path: Path,
    contract_sha256: str,
    normalization_sha256: str,
    plan_sha256: str,
) -> None:
    del plan_sha256
    report = _load_json(report_path.resolve())
    if (
        report.get("status") != "passed"
        or report.get("stage") != "real_smolvla_no_optimizer_forward"
        or report.get("experiment_id") != experiment["experiment_id"]
        or report.get("experiment_config_sha256") != file_sha256(base_path)
        or report.get("action_contract_sha256") != contract_sha256
        or report.get("normalization_report_sha256") != normalization_sha256
        or report.get("formal_plan_sha256") is not None
        or report.get("device") != "cuda"
        or report.get("optimizer_created") is not False
        or report.get("gradients_enabled") is not False
        or report.get("hidden_test_loaded") is not False
    ):
        raise ValueError("Way AutoDL no-optimizer forward evidence is invalid.")


def _validate_plan(plan_path: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    plan = _load_yaml(plan_path)
    spec = _FORMAL_SPECS.get(str(plan.get("plan_id", "")))
    if spec is None:
        raise ValueError("Way formal plan identity is not registered.")
    parent = plan.get("parent_experiment", {})
    base_path = _repository_path(str(parent.get("config", "")))
    experiment = load_smolvla_experiment(base_path, REPOSITORY_ROOT)
    runtime = plan.get("runtime_profile", {})
    runtime_path = _repository_path(str(runtime.get("path", "")))
    smoke = plan.get("cuda_smoke", {})
    training = plan.get("training", {})
    validation = plan.get("validation", {})
    resources = plan.get("resources", {})
    monitoring = plan.get("monitoring", {})
    implementation = plan.get("implementation_files", {})
    hidden = {int(value) for value in experiment["dataset"]["test_episodes"]}
    optimizer = training.get("optimizer", {})
    scheduler = training.get("scheduler", {})
    if (
        plan.get("schema_version") != 1
        or plan.get("role") != "vla"
        or plan.get("stage") != "m2_state_robustness_development_training"
        or plan.get("status") != "preregistered"
        or plan.get("run_name") != spec["run_name"]
        or RUN_NAME_PATTERN.fullmatch(str(plan.get("run_name", ""))) is None
        or plan.get("furnace_program", {}).get("codename") != "Way"
        or parent.get("sha256") != file_sha256(base_path)
        or parent.get("experiment_id") != experiment["experiment_id"]
        or file_sha256(runtime_path) != runtime.get("sha256")
        or runtime.get("profile_id") != "autodl-rtx4090-cuda-001"
        or plan.get("initialization")
        != {
            "source": "revision_pinned_base_model",
            "aster_checkpoint_used": False,
            "faust_checkpoint_used": False,
            "optimizer_state_reused": False,
        }
        or smoke.get("plan_id") != spec["smoke_plan_id"]
        or smoke.get("run_name") != spec["smoke_run_name"]
        or not isinstance(smoke.get("plan_sha256"), str)
        or len(smoke["plan_sha256"]) != 64
        or not isinstance(smoke.get("acceptance_sha256"), str)
        or len(smoke["acceptance_sha256"]) != 64
        or training.get("episodes") != experiment["dataset"]["train_episodes"]
        or set(training.get("episodes", [])) & hidden
        or training.get("batch_size") != spec["batch_size"]
        or training.get("steps") != spec["steps"]
        or training.get("sample_exposures") != spec["sample_exposures"]
        or training.get("aster_control_sample_exposures") != 20_000
        or training.get("save_freq") != spec["save_freq"]
        or training.get("checkpoint_steps") != spec["checkpoint_steps"]
        or training.get("save_checkpoint") is not True
        or training.get("log_freq") != 1
        or training.get("num_workers") != 0
        or training.get("persistent_workers") is not False
        or training.get("eval_split") != 0.0
        or training.get("validation_gradients") is not False
        or training.get("hidden_test_loaded") is not False
        or training.get("policy")
        != {
            "empty_cameras": 2,
            "compile_model": True,
            "compile_mode": spec["compile_mode"],
            "skip_fully_masked_camera_encoding": True,
        }
        or optimizer
        != {
            "type": "adamw",
            "lr": 1.0e-4,
            "betas": [0.9, 0.95],
            "eps": 1.0e-8,
            "weight_decay": 1.0e-10,
            "grad_clip_norm": 10.0,
        }
        or scheduler
        != {
            "type": "cosine_decay_with_warmup",
            "num_warmup_steps": spec["warmup_steps"],
            "num_decay_steps": spec["steps"],
            "peak_lr": 1.0e-4,
            "decay_lr": 2.5e-6,
        }
        or resources.get("runtime") != "autodl_container_instance"
        or resources.get("accelerator") != "cuda"
        or resources.get("mixed_precision") != "bf16"
        or resources.get("memory_limit") != "autodl_platform_container"
        or resources.get("memory_swap_limit") != "autodl_platform_container"
        or resources.get("nested_docker_used") is not False
        or isinstance(resources.get("measured_peak_accelerator_allocated_bytes"), bool)
        or not isinstance(resources.get("measured_peak_accelerator_allocated_bytes"), int)
        or resources["measured_peak_accelerator_allocated_bytes"] <= 0
        or resources["measured_peak_accelerator_allocated_bytes"]
        > int(resources.get("maximum_peak_accelerator_allocated_bytes", 0))
        or int(resources.get("maximum_peak_accelerator_allocated_bytes", 0))
        > 24_696_061_952
        or not 1 <= int(resources.get("expected_wall_time_minutes", 0)) <= 60
        or not int(resources["expected_wall_time_minutes"])
        <= int(resources.get("maximum_wall_time_minutes", 0))
        <= 90
        or monitoring.get("blocking_command") != "sleep"
        or monitoring.get("sleep_poll_seconds") != 300
        or monitoring.get("wake_steps") != spec["checkpoint_steps"]
        or monitoring.get("hard_shutdown_budget_minutes") != 150
        or validation.get("run_name_prefix") != spec["validation_prefix"]
        or validation.get("episodes") != experiment["dataset"]["validation_episodes"]
        or set(validation.get("episodes", [])) & hidden
        or validation.get("frame_offsets") != [0, 125, 250, 375]
        or validation.get("total_samples") != 20
        or validation.get("checkpoints") != ["base", *spec["checkpoint_steps"]]
        or validation.get("primary_selection_metric") != "first_action_mae"
        or validation.get("secondary_selection_metric") != "action_mae"
        or validation.get("hidden_test_loaded") is not False
        or not isinstance(implementation, dict)
        or not implementation
        or plan.get("hidden_test_loaded") is not False
    ):
        raise ValueError("Way formal plan differs from its registered CUDA contract.")
    if _optimizer_contract(training) is None:
        raise ValueError("Way formal optimizer contract is absent.")
    horizon_profile(plan, int(experiment["model"]["policy"]["chunk_size"]))
    state_profile(plan)
    contract_path = REPOSITORY_ROOT / str(experiment["action_contract"]["derived"])
    _validate_prerequisites(plan, experiment, base_path, file_sha256(contract_path))
    for raw_path, expected in implementation.items():
        path = _repository_path(str(raw_path))
        if file_sha256(path) != expected:
            raise ValueError(f"Way formal implementation checksum changed: {raw_path}.")
    return plan, base_path, experiment


def _validate_cuda_smoke(
    report_path: Path,
    plan: dict[str, Any],
    experiment: dict[str, Any],
    base_path: Path,
) -> dict[str, Any]:
    report = _load_json(report_path.resolve())
    smoke = plan["cuda_smoke"]
    if (
        report.get("status") != "passed"
        or report.get("stage")
        != "smolvla_state_robustness_cuda_optimizer_smoke_acceptance"
        or report.get("experiment_id") != experiment["experiment_id"]
        or report.get("run_name") != smoke["run_name"]
        or report.get("formal_plan_sha256") != smoke["plan_sha256"]
        or file_sha256(report_path.resolve()) != smoke["acceptance_sha256"]
        or report.get("experiment_config_sha256") != file_sha256(base_path)
        or report.get("batch_size") != plan["training"]["batch_size"]
        or report.get("steps") != 2
        or report.get("acceptance", {}).get("all_metrics_finite") is not True
        or report.get("acceptance", {}).get(
            "independent_final_checkpoint_reload_passed"
        )
        is not True
        or report.get("acceptance", {}).get(
            "peak_accelerator_allocation_within_guard"
        )
        is not True
        or report.get("hidden_test_loaded") is not False
        or report.get("formal_training_authorized") is not False
    ):
        raise ValueError("Way formal run lacks valid CUDA smoke acceptance.")
    if report["peak_accelerator_allocated_bytes"] != plan["resources"][
        "measured_peak_accelerator_allocated_bytes"
    ]:
        raise ValueError("Way formal memory registration differs from CUDA smoke.")
    return report


def _prepare_compiler_cache(plan_path: Path) -> dict[str, str]:
    run_root = phase_runner._absolute_root("ROSETTA_RUN_ROOT")
    root = run_root / "compiler_cache" / f"way-formal-{file_sha256(plan_path)[:12]}"
    triton = root / "triton"
    inductor = root / "inductor"
    triton.mkdir(parents=True, exist_ok=True)
    inductor.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(triton)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor)
    return {
        "cache_root": root.relative_to(run_root).as_posix(),
        "triton_cache": triton.relative_to(run_root).as_posix(),
        "inductor_cache": inductor.relative_to(run_root).as_posix(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--optimizer-smoke-report", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    args = parser.parse_args()
    if (
        os.environ.get("HF_HUB_OFFLINE") != "1"
        or os.environ.get("HF_DATASETS_OFFLINE") != "1"
        or os.environ.get("ROSETTA_AUTODL_FORMAL_AUTHORIZED") != "1"
        or os.environ.get("ROSETTA_TORCH_DEVICE") != "cuda"
        or not torch.cuda.is_available()
    ):
        raise RuntimeError("Way formal requires its separately registered AutoDL CUDA plan.")
    plan_path = args.plan.resolve()
    plan, base_path, experiment = _validate_plan(plan_path)
    smoke = _validate_cuda_smoke(
        args.optimizer_smoke_report.resolve(), plan, experiment, base_path
    )
    contract_path = REPOSITORY_ROOT / str(experiment["action_contract"]["derived"])
    contract_sha256 = file_sha256(contract_path)
    prerequisites = _validate_prerequisites(
        plan, experiment, base_path, contract_sha256
    )
    normalization, view_manifest, dataset_root = _validate_normalization(
        plan, experiment, base_path, contract_sha256
    )
    _validate_preflight(
        args.preflight_report.resolve(),
        plan,
        experiment,
        base_path,
        contract_sha256,
        file_sha256(normalization),
        file_sha256(plan_path),
    )
    run_name = str(plan["run_name"])
    output_dir = (
        phase_runner._absolute_root("ROSETTA_CHECKPOINT_ROOT")
        / str(experiment["experiment_id"])
        / "formal"
        / run_name
    )
    if output_dir.exists():
        raise FileExistsError("Way formal output is create-only.")
    compiler_cache = _prepare_compiler_cache(plan_path)
    identity = workspace_code_identity(REPOSITORY_ROOT)
    launch = {
        "schema_version": 1,
        "status": "preregistered",
        "stage": "smolvla_state_robustness_cuda_formal_launch",
        "mode": "train",
        "experiment_id": experiment["experiment_id"],
        "run_name": run_name,
        "codename": "Way",
        "formal_plan_sha256": file_sha256(plan_path),
        "experiment_config_sha256": file_sha256(base_path),
        "action_contract_sha256": contract_sha256,
        "normalization_report_sha256": file_sha256(normalization),
        "dataset_view_manifest_sha256": file_sha256(view_manifest),
        "prerequisites": {
            name: file_sha256(path) for name, path in sorted(prerequisites.items())
        },
        "cuda_smoke_acceptance_sha256": file_sha256(
            args.optimizer_smoke_report.resolve()
        ),
        "cuda_smoke_plan_sha256": smoke["formal_plan_sha256"],
        "loss_contract": plan["loss_contract"],
        "state_robustness_contract": plan["state_robustness_contract"],
        "initialization": plan["initialization"],
        "optimizer_contract": _optimizer_contract(plan["training"]),
        "monitoring": plan["monitoring"],
        "compiler_cache": compiler_cache,
        "code_identity": identity,
        "nested_docker_used": False,
        "hidden_test_loaded": False,
    }
    launch_path = (
        phase_runner._absolute_root("ROSETTA_RUN_ROOT")
        / str(experiment["experiment_id"])
        / "launch"
        / f"{run_name}.json"
    )
    create_json(launch_path, launch)
    os.environ.update(
        {
            "ROSETTA_VLA_PHASE": "formal",
            "ROSETTA_VLA_EXPERIMENT_CONFIG": str(base_path),
            "ROSETTA_VLA_RUN_NAME": run_name,
            "ROSETTA_VLA_TRAIN_STATS_REPORT": str(normalization),
            "ROSETTA_VLA_NORMALIZATION_SHA256": file_sha256(normalization),
            "ROSETTA_VLA_FORMAL_PLAN_PATH": str(plan_path),
            "ROSETTA_VLA_FORMAL_PLAN_SHA256": file_sha256(plan_path),
            "ROSETTA_VLA_STATE_ROBUSTNESS_CUDA_FORMAL_AUTHORIZED": "1",
            "ROSETTA_VLA_CODE_REVISION": str(identity["revision"]),
            "ROSETTA_VLA_WORKSPACE_TREE_SHA256": str(identity["workspace_tree_sha256"]),
            "ROSETTA_VLA_WORKSPACE_DIRTY": str(bool(identity["dirty"])).lower(),
            "ROSETTA_VLA_WORKSPACE_FILE_COUNT": str(identity["workspace_file_count"]),
        }
    )
    runtime = copy.deepcopy(experiment)
    runtime["resources"].update(plan["resources"])
    runtime["phases"]["formal"] = dict(plan["training"])
    policy = runtime["phases"]["formal"].pop("policy")
    runtime["model"]["policy"].update(policy)
    os.environ["ROSETTA_VLA_SKIP_FULLY_MASKED_CAMERA_ENCODING"] = str(
        int(bool(policy["skip_fully_masked_camera_encoding"]))
    )
    arguments = phase_runner._phase_arguments(
        runtime,
        "formal",
        run_name,
        phase_runner._model_root(experiment),
        dataset_root,
        output_dir,
    )
    arguments.extend(_optimizer_arguments(plan["training"]))
    sys.argv = ["lerobot-train", *arguments]
    from train_smolvla_state_robustness_cuda_formal import main as train_main

    started = time.perf_counter()
    train_main()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if not math.isfinite(elapsed):
        raise FloatingPointError("Way formal elapsed time is non-finite.")
    completion = {
        "schema_version": 1,
        "status": "complete",
        "stage": "smolvla_state_robustness_cuda_formal_completion",
        "experiment_id": experiment["experiment_id"],
        "run_name": run_name,
        "formal_plan_sha256": file_sha256(plan_path),
        "steps": plan["training"]["steps"],
        "batch_size": plan["training"]["batch_size"],
        "sample_exposures": plan["training"]["sample_exposures"],
        "elapsed_seconds": elapsed,
        "maximum_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "hidden_test_loaded": False,
    }
    completion_path = (
        phase_runner._absolute_root("ROSETTA_RUN_ROOT")
        / str(experiment["experiment_id"])
        / "completion"
        / f"{run_name}.json"
    )
    create_json(completion_path, completion)
    print(json.dumps(completion, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
