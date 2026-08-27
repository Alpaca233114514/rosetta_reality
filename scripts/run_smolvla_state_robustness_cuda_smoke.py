"""Launch the preregistered Way batch-64 CUDA optimizer smoke on AutoDL."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import run_smolvla_action_repair_formal as optimizer_contract  # noqa: E402
import run_smolvla_action_repair_phase as repair_phase  # noqa: E402
import run_smolvla_phase as phase_runner  # noqa: E402

from rosetta_reality.experiment import (  # noqa: E402
    file_sha256,
    workspace_code_identity,
)
from rosetta_reality.features import create_json  # noqa: E402
from rosetta_reality.vla import (  # noqa: E402
    load_smolvla_action_space,
    load_smolvla_experiment,
)
from rosetta_reality.vla.horizon_loss import profile_from_plan as horizon_profile  # noqa: E402
from rosetta_reality.vla.state_robustness import profile_from_plan as state_profile  # noqa: E402

DEFAULT_PLAN = (
    REPOSITORY_ROOT
    / "configs/vla/smolvla_450m_aloha_insertion_way_cuda_batch128_smoke_001.yaml"
)
RUN_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{2,79}")
_PLAN_SPECS = {
    "m2-smolvla450m-way-cuda-b128-smoke-001": {
        "run_name": "m2-smolvla450m-way-cuda-b128-smoke-001",
        "batch_size": 128,
        "expected_wall_time_minutes": 12,
        "maximum_wall_time_minutes": 25,
        "activation_mode": "primary",
        "compile_mode": "reduce-overhead",
    },
    "m2-smolvla450m-way-cuda-b64-smoke-001": {
        "run_name": "m2-smolvla450m-way-cuda-b64-smoke-001",
        "batch_size": 64,
        "expected_wall_time_minutes": 10,
        "maximum_wall_time_minutes": 20,
        "activation_mode": "fallback_after_primary_cuda_memory_failure",
        "compile_mode": "reduce-overhead",
    },
    "m2-smolvla450m-way-cuda-b64-default-smoke-002": {
        "run_name": "m2-smolvla450m-way-cuda-b64-default-smoke-002",
        "batch_size": 64,
        "expected_wall_time_minutes": 10,
        "maximum_wall_time_minutes": 20,
        "activation_mode": (
            "fallback_after_primary_cuda_memory_failure_and_cudagraph_runtime_repair"
        ),
        "compile_mode": "default",
    },
}


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path.name}.")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}.")
    json.dumps(value, allow_nan=False)
    return value


def _repository_path(raw: str) -> Path:
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Way CUDA plan paths must be safe repository-relative paths.")
    if relative.parts and relative.parts[0] == "runs":
        run_root_raw = os.environ.get("ROSETTA_RUN_ROOT")
        if run_root_raw:
            run_root = Path(run_root_raw).resolve()
            path = (run_root / Path(*relative.parts[1:])).resolve()
            if path.is_relative_to(run_root) and path.is_file():
                return path
    path = (REPOSITORY_ROOT / relative).resolve()
    if not path.is_relative_to(REPOSITORY_ROOT) or not path.is_file():
        raise FileNotFoundError(f"Way CUDA prerequisite is missing: {relative.as_posix()}.")
    return path


def _control_training(plan: dict[str, Any]) -> dict[str, Any]:
    control = plan["control_reference"]
    aster = _load_yaml(_repository_path(str(control["aster_plan"])))
    training = aster.get("training")
    if not isinstance(training, dict):
        raise ValueError("The read-only Aster control has no training contract.")
    return training


def _validate_plan(plan_path: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    plan = _load_yaml(plan_path)
    parent = plan.get("parent_experiment", {})
    base_path = _repository_path(str(parent.get("config", "")))
    experiment = load_smolvla_experiment(base_path, REPOSITORY_ROOT)
    control = plan.get("control_reference", {})
    runtime = plan.get("runtime_profile", {})
    runtime_path = _repository_path(str(runtime.get("path", "")))
    runtime_profile = _load_yaml(runtime_path)
    aster_path = _repository_path(str(control.get("aster_plan", "")))
    local_plan_path = _repository_path(str(control.get("local_way_smoke_plan", "")))
    local_acceptance_path = _repository_path(
        str(control.get("local_way_smoke_acceptance", ""))
    )
    aster = _load_yaml(aster_path)
    local_plan = _load_yaml(local_plan_path)
    local_acceptance = _load_json(local_acceptance_path)
    smoke = plan.get("optimizer_smoke", {})
    resources = plan.get("resources", {})
    implementation = plan.get("implementation_files", {})
    spec = _PLAN_SPECS.get(str(plan.get("plan_id", "")))
    if spec is None:
        raise ValueError("Way CUDA smoke has an unregistered plan identity.")
    expected_resources = {
        "runtime": "autodl_container_instance",
        "accelerator": "cuda",
        "device_name": "RTX 4090",
        "mixed_precision": "bf16",
        "minimum_total_accelerator_memory_bytes": 24696061952,
        "maximum_peak_accelerator_allocated_bytes": 24696061952,
        "expected_wall_time_minutes": spec["expected_wall_time_minutes"],
        "maximum_wall_time_minutes": spec["maximum_wall_time_minutes"],
        "checkpoint_memory_trim": True,
        "nested_docker_used": False,
    }
    expected_policy = copy.deepcopy(aster.get("training", {}).get("policy"))
    if not isinstance(expected_policy, dict):
        raise ValueError("The read-only Aster control has no policy contract.")
    expected_policy["compile_mode"] = spec["compile_mode"]
    expected_smoke = {
        "episodes": [49],
        "batch_size": spec["batch_size"],
        "steps": 2,
        "save_freq": 1,
        "save_checkpoint": True,
        "log_freq": 1,
        "num_workers": 0,
        "persistent_workers": False,
        "hidden_test_loaded": False,
        "policy": expected_policy,
        "optimizer": aster.get("training", {}).get("optimizer"),
        "scheduler": aster.get("training", {}).get("scheduler"),
    }
    if (
        plan.get("schema_version") != 1
        or plan.get("role") != "vla"
        or plan.get("stage") != "m2_state_robustness_cuda_optimizer_smoke"
        or plan.get("status") != "preregistered"
        or plan.get("run_name") != spec["run_name"]
        or RUN_NAME_PATTERN.fullmatch(str(plan.get("run_name", ""))) is None
        or plan.get("furnace_program", {}).get("codename") != "Way"
        or plan.get("activation", {}).get("mode") != spec["activation_mode"]
        or parent.get("sha256") != file_sha256(base_path)
        or parent.get("experiment_id") != experiment["experiment_id"]
        or file_sha256(runtime_path) != runtime.get("sha256")
        or runtime.get("profile_id") != runtime_profile.get("profile_id")
        or runtime_profile.get("formal_training", {}).get("enabled_by_profile") is not False
        or runtime_profile.get("nested_docker_supported") is not False
        or file_sha256(aster_path) != control.get("aster_plan_sha256")
        or file_sha256(local_plan_path) != control.get("local_way_smoke_plan_sha256")
        or file_sha256(local_acceptance_path)
        != control.get("local_way_smoke_acceptance_sha256")
        or local_acceptance.get("status") != "passed"
        or local_acceptance.get("formal_plan_sha256")
        != control.get("local_way_smoke_plan_sha256")
        or plan.get("initialization", {}).get("source")
        != "revision_pinned_base_model"
        or plan.get("initialization", {}).get("aster_checkpoint_used") is not False
        or plan.get("initialization", {}).get("faust_checkpoint_used") is not False
        or plan.get("initialization", {}).get("optimizer_state_reused") is not False
        or plan.get("loss_contract") != aster.get("loss_contract")
        or plan.get("state_robustness_contract")
        != local_plan.get("state_robustness_contract")
        or smoke != expected_smoke
        or resources != expected_resources
        or not isinstance(implementation, dict)
        or not implementation
        or plan.get("hidden_test_loaded") is not False
        or plan.get("closed_loop_claim") is not False
    ):
        raise ValueError("Way CUDA smoke differs from its preregistered single-axis contract.")
    activation = plan["activation"]
    if spec["activation_mode"] == "primary":
        fallback_path = _repository_path(str(activation.get("fallback_plan", "")))
        if (
            fallback_path.name
            != "smolvla_450m_aloha_insertion_way_cuda_batch64_smoke_001.yaml"
            or activation.get("fallback_only_after_cuda_memory_failure") is not True
            or activation.get("automatic_in_run_retry") is not False
        ):
            raise ValueError("Way batch-128 primary activation contract is invalid.")
    elif spec["activation_mode"] == "fallback_after_primary_cuda_memory_failure":
        primary_path = _repository_path(str(activation.get("primary_plan", "")))
        if (
            primary_path.name
            != "smolvla_450m_aloha_insertion_way_cuda_batch128_smoke_001.yaml"
            or file_sha256(primary_path) != activation.get("primary_plan_sha256")
            or activation.get("eligible_failure_classes")
            != ["cuda_out_of_memory", "peak_memory_guard_exceeded"]
            or activation.get("failed_primary_checkpoint_or_optimizer_reuse") is not False
            or activation.get("automatic_in_run_retry") is not False
        ):
            raise ValueError("Way batch-64 fallback activation contract is invalid.")
    else:
        primary_path = _repository_path(str(activation.get("primary_plan", "")))
        repair = plan.get("runtime_repair", {})
        failed_plan_path = _repository_path(str(repair.get("failed_plan", "")))
        failure_path = _repository_path(str(repair.get("failure_report", "")))
        failure = _load_json(failure_path)
        expected_repair = {
            "failed_plan": (
                "configs/vla/"
                "smolvla_450m_aloha_insertion_way_cuda_batch64_smoke_001.yaml"
            ),
            "failed_plan_sha256": (
                "36d869600bfed0f5eb20931adbd235043a8122925964024d21e1c0e82e5657e1"
            ),
            "failure_report": (
                "runs/m2-smolvla450m-aloha-insertion-action-repair-bounded-"
                "gripper-003/failures/"
                "m2-smolvla450m-way-cuda-b64-smoke-001.json"
            ),
            "failure_report_sha256": (
                "2abc22fa4a076755da9d7ee3e943cf86ee575cfce2cbc9ac13772feb655c59e8"
            ),
            "failure_class": "non_memory_runtime_failure",
            "exception_type": "AcceleratorError",
            "observed_compile_mode": "reduce-overhead",
            "replacement_compile_mode": "default",
            "torch_version": "2.8.0+cu128",
            "cause": "cuda_graph_capture_rejected_mutating_index_put",
            "research_contract_unchanged": True,
            "checkpoint_or_optimizer_state_reused": False,
        }
        if (
            primary_path.name
            != "smolvla_450m_aloha_insertion_way_cuda_batch128_smoke_001.yaml"
            or file_sha256(primary_path) != activation.get("primary_plan_sha256")
            or activation.get("eligible_failure_classes")
            != ["cuda_out_of_memory", "peak_memory_guard_exceeded"]
            or activation.get("failed_primary_checkpoint_or_optimizer_reuse") is not False
            or activation.get("automatic_in_run_retry") is not False
            or repair != expected_repair
            or file_sha256(failed_plan_path) != repair["failed_plan_sha256"]
            or file_sha256(failure_path) != repair["failure_report_sha256"]
            or failure.get("status") != "failed"
            or failure.get("stage")
            != "smolvla_state_robustness_cuda_smoke_failure"
            or failure.get("run_name") != "m2-smolvla450m-way-cuda-b64-smoke-001"
            or failure.get("formal_plan_sha256") != repair["failed_plan_sha256"]
            or failure.get("failure_class") != repair["failure_class"]
            or failure.get("exception_type") != repair["exception_type"]
            or failure.get("checkpoint_or_optimizer_state_reused_by_fallback")
            is not False
            or failure.get("hidden_test_loaded") is not False
        ):
            raise ValueError("Way CUDA-graph-safe runtime repair is invalid.")
    hidden = {int(value) for value in experiment["dataset"]["test_episodes"]}
    if set(smoke["episodes"]) & hidden:
        raise ValueError("Way CUDA smoke intersects the hidden-test split.")
    action_space = load_smolvla_action_space(experiment, require_explicit=True)
    if not action_space.explicit:
        raise ValueError("Way CUDA smoke requires the bounded-gripper action boundary.")
    horizon_profile(plan, int(experiment["model"]["policy"]["chunk_size"]))
    state_profile(plan)
    optimizer_contract._optimizer_contract(_control_training(plan))
    for evidence in plan.get("prerequisites", {}).values():
        path = _repository_path(str(evidence.get("path", "")))
        if file_sha256(path) != evidence.get("sha256"):
            raise ValueError(f"Way CUDA prerequisite changed: {path.name}.")
    for raw_path, expected in implementation.items():
        path = _repository_path(str(raw_path))
        if file_sha256(path) != expected:
            raise ValueError(f"Way CUDA implementation checksum changed: {raw_path}.")
    return plan, base_path, experiment


def _validate_autodl_evidence(
    *,
    plan: dict[str, Any],
    base_path: Path,
    experiment: dict[str, Any],
    doctor_path: Path,
    benchmark_path: Path,
    preflight_path: Path,
    supplement_path: Path,
) -> dict[str, Any]:
    profile_path = _repository_path(str(plan["runtime_profile"]["path"]))
    profile_sha256 = file_sha256(profile_path)
    contract_path = REPOSITORY_ROOT / str(experiment["action_contract"]["derived"])
    doctor = _load_json(doctor_path)
    benchmark = _load_json(benchmark_path)
    preflight = _load_json(preflight_path)
    supplement = _load_json(supplement_path)
    identity = workspace_code_identity(REPOSITORY_ROOT)
    if (
        doctor.get("status") != "passed"
        or doctor.get("stage") != "autodl_environment_doctor"
        or doctor.get("profile_sha256") != profile_sha256
        or doctor.get("experiment_id") != experiment["experiment_id"]
        or doctor.get("experiment_config_sha256") != file_sha256(base_path)
        or doctor.get("runtime_boundary") != "platform_linux_container"
        or doctor.get("nested_docker_used") is not False
        or doctor.get("accelerator", {}).get("device") != "cuda"
        or "4090" not in str(doctor.get("accelerator", {}).get("name", ""))
        or int(doctor.get("accelerator", {}).get("total_memory_bytes", 0))
        < int(plan["resources"]["minimum_total_accelerator_memory_bytes"])
        or doctor.get("workspace") != identity
        or doctor.get("optimizer_created") is not False
        or doctor.get("hidden_test_loaded") is not False
    ):
        raise ValueError("AutoDL doctor evidence does not bind this Way workspace.")
    if (
        benchmark.get("status") != "complete"
        or benchmark.get("stage") != "pre_training"
        or benchmark.get("experiment_id") != experiment["experiment_id"]
        or benchmark.get("experiment_config_sha256") != file_sha256(base_path)
        or benchmark.get("action_contract_sha256") != file_sha256(contract_path)
        or benchmark.get("hidden_test_loaded") is not False
        or benchmark.get("evaluated_split") != "validation"
    ):
        raise ValueError("AutoDL benchmark evidence is invalid.")
    if (
        preflight.get("status") != "passed"
        or preflight.get("stage") != "real_smolvla_no_optimizer_forward"
        or preflight.get("experiment_id") != experiment["experiment_id"]
        or preflight.get("experiment_config_sha256") != file_sha256(base_path)
        or preflight.get("device") != "cuda"
        or preflight.get("optimizer_created") is not False
        or preflight.get("gradients_enabled") is not False
        or preflight.get("hidden_test_loaded") is not False
        or preflight.get("formal_plan_sha256") is not None
    ):
        raise ValueError("AutoDL no-optimizer forward evidence is invalid.")
    if (
        supplement.get("status") != "passed"
        or supplement.get("stage") != "autodl_cuda_no_optimizer_forward_supplement"
        or supplement.get("profile_sha256") != profile_sha256
        or supplement.get("experiment_id") != experiment["experiment_id"]
        or supplement.get("run_name") != preflight.get("run_name")
        or supplement.get("preflight_report_sha256") != file_sha256(preflight_path)
        or supplement.get("device") != "cuda"
        or "4090" not in str(supplement.get("device_name", ""))
        or supplement.get("optimizer_created") is not False
        or supplement.get("hidden_test_loaded") is not False
        or supplement.get("formal_training_authorized") is not False
    ):
        raise ValueError("AutoDL CUDA forward supplement is invalid.")
    return {
        "doctor_report_sha256": file_sha256(doctor_path),
        "benchmark_report_sha256": file_sha256(benchmark_path),
        "preflight_report_sha256": file_sha256(preflight_path),
        "preflight_supplement_sha256": file_sha256(supplement_path),
        "preflight_maximum_allocated_bytes": int(
            supplement["maximum_allocated_bytes"]
        ),
    }


def _prepare_compiler_cache(plan_path: Path) -> dict[str, str]:
    run_root = phase_runner._absolute_root("ROSETTA_RUN_ROOT")
    root = run_root / "compiler_cache" / f"way-cuda-{file_sha256(plan_path)[:12]}"
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


def _validate_primary_failure(plan: dict[str, Any], path: Path | None) -> str | None:
    if plan["activation"]["mode"] == "primary":
        if path is not None:
            raise ValueError("The primary Way smoke cannot consume fallback evidence.")
        return None
    if path is None:
        raise ValueError("The batch-64 fallback requires a primary failure report.")
    report = _load_json(path)
    primary_path = _repository_path(str(plan["activation"]["primary_plan"]))
    if (
        report.get("status") != "failed"
        or report.get("stage") != "smolvla_state_robustness_cuda_smoke_failure"
        or report.get("formal_plan_sha256") != file_sha256(primary_path)
        or report.get("failure_class")
        not in plan["activation"]["eligible_failure_classes"]
        or report.get("checkpoint_or_optimizer_state_reused_by_fallback") is not False
        or report.get("hidden_test_loaded") is not False
    ):
        raise ValueError("The primary CUDA smoke failure does not activate batch 64.")
    return file_sha256(path)


def _write_failure_report(
    plan_path: Path,
    plan: dict[str, Any],
    experiment: dict[str, Any],
    error: Exception,
) -> None:
    message = str(error).lower()
    failure_class = (
        "cuda_out_of_memory"
        if "out of memory" in message and ("cuda" in message or "gpu" in message)
        else "non_memory_runtime_failure"
    )
    report = {
        "schema_version": 1,
        "status": "failed",
        "stage": "smolvla_state_robustness_cuda_smoke_failure",
        "experiment_id": experiment["experiment_id"],
        "run_name": plan["run_name"],
        "formal_plan_sha256": file_sha256(plan_path),
        "batch_size": plan["optimizer_smoke"]["batch_size"],
        "failure_class": failure_class,
        "exception_type": type(error).__name__,
        "automatic_retry_performed": False,
        "checkpoint_or_optimizer_state_reused_by_fallback": False,
        "hidden_test_loaded": False,
    }
    run_root = phase_runner._absolute_root("ROSETTA_RUN_ROOT")
    destination = (
        run_root
        / str(experiment["experiment_id"])
        / "failures"
        / f"{plan['run_name']}.json"
    )
    if not destination.exists():
        create_json(destination, report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--doctor-report", type=Path, required=True)
    parser.add_argument("--benchmark-report", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--preflight-supplement", type=Path, required=True)
    parser.add_argument("--primary-failure-report", type=Path)
    args = parser.parse_args()
    if (
        os.environ.get("HF_HUB_OFFLINE") != "1"
        or os.environ.get("HF_DATASETS_OFFLINE") != "1"
        or os.environ.get("ROSETTA_AUTODL_TWO_STEP_SMOKE_AUTHORIZED") != "1"
        or os.environ.get("ROSETTA_TORCH_DEVICE") != "cuda"
        or not torch.cuda.is_available()
    ):
        raise RuntimeError("Way CUDA smoke requires the offline AutoDL CUDA boundary.")
    plan_path = args.plan.resolve()
    plan, base_path, experiment = _validate_plan(plan_path)
    primary_failure_sha256 = _validate_primary_failure(
        plan,
        args.primary_failure_report.resolve()
        if args.primary_failure_report is not None
        else None,
    )
    evidence = _validate_autodl_evidence(
        plan=plan,
        base_path=base_path,
        experiment=experiment,
        doctor_path=args.doctor_report.resolve(),
        benchmark_path=args.benchmark_report.resolve(),
        preflight_path=args.preflight_report.resolve(),
        supplement_path=args.preflight_supplement.resolve(),
    )
    prerequisites = plan["prerequisites"]
    normalization = _repository_path(prerequisites["normalization"]["path"])
    action_space_report = _repository_path(prerequisites["action_space"]["path"])
    dataset_root = repair_phase._validate_repair_evidence(
        experiment,
        base_path,
        normalization,
        action_space_report,
    )
    contract_path = REPOSITORY_ROOT / str(experiment["action_contract"]["derived"])
    contract_sha256 = file_sha256(contract_path)
    for gate_name, expected_gate in (
        ("gate1", "m2_gate_1_scripted_action"),
        ("gate2", "m2_gate_2_dataset_action_replay"),
    ):
        phase_runner._validate_gate(
            _repository_path(prerequisites[gate_name]["path"]),
            expected_gate=expected_gate,
            experiment_id=experiment["experiment_id"],
            contract_sha256=contract_sha256,
            dataset_revision=experiment["dataset"]["revision"],
            allowed_replay_episodes=[
                *experiment["dataset"]["train_episodes"],
                *experiment["dataset"]["validation_episodes"],
            ],
        )
    run_name = str(plan["run_name"])
    output_dir = (
        phase_runner._absolute_root("ROSETTA_CHECKPOINT_ROOT")
        / str(experiment["experiment_id"])
        / "smoke"
        / run_name
    )
    if output_dir.exists():
        raise FileExistsError("Way CUDA optimizer-smoke output is create-only.")
    compiler_cache = _prepare_compiler_cache(plan_path)
    identity = workspace_code_identity(REPOSITORY_ROOT)
    launch = {
        "schema_version": 1,
        "status": "preregistered",
        "stage": "smolvla_state_robustness_cuda_optimizer_smoke_launch",
        "experiment_id": experiment["experiment_id"],
        "run_name": run_name,
        "codename": "Way",
        "formal_plan_sha256": file_sha256(plan_path),
        "experiment_config_sha256": file_sha256(base_path),
        "action_contract_sha256": contract_sha256,
        "runtime_profile_sha256": plan["runtime_profile"]["sha256"],
        "normalization_report_sha256": file_sha256(normalization),
        "autodl_evidence": evidence,
        "primary_failure_report_sha256": primary_failure_sha256,
        "loss_contract": plan["loss_contract"],
        "state_robustness_contract": plan["state_robustness_contract"],
        "optimizer_contract": optimizer_contract._optimizer_contract(
            _control_training(plan)
        ),
        "compiler_cache": compiler_cache,
        "code_identity": identity,
        "nested_docker_used": False,
        "hidden_test_loaded": False,
        "formal_training_authorized": False,
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
            "ROSETTA_VLA_PHASE": "performance_benchmark",
            "ROSETTA_VLA_EXPERIMENT_CONFIG": str(base_path),
            "ROSETTA_VLA_RUN_NAME": run_name,
            "ROSETTA_VLA_TRAIN_STATS_REPORT": str(normalization),
            "ROSETTA_VLA_NORMALIZATION_SHA256": file_sha256(normalization),
            "ROSETTA_VLA_FORMAL_PLAN_PATH": str(plan_path),
            "ROSETTA_VLA_FORMAL_PLAN_SHA256": file_sha256(plan_path),
            "ROSETTA_VLA_STATE_ROBUSTNESS_CUDA_PLAN_PATH": str(plan_path),
            "ROSETTA_VLA_STATE_ROBUSTNESS_CUDA_SMOKE_AUTHORIZED": "1",
            "ROSETTA_VLA_CODE_REVISION": str(identity["revision"]),
            "ROSETTA_VLA_WORKSPACE_TREE_SHA256": str(identity["workspace_tree_sha256"]),
            "ROSETTA_VLA_WORKSPACE_DIRTY": str(bool(identity["dirty"])).lower(),
            "ROSETTA_VLA_WORKSPACE_FILE_COUNT": str(identity["workspace_file_count"]),
        }
    )
    runtime = copy.deepcopy(experiment)
    runtime["resources"].update(plan["resources"])
    runtime["phases"]["performance_benchmark"] = dict(plan["optimizer_smoke"])
    policy = runtime["phases"]["performance_benchmark"].pop("policy")
    runtime["model"]["policy"].update(policy)
    os.environ["ROSETTA_VLA_SKIP_FULLY_MASKED_CAMERA_ENCODING"] = str(
        int(bool(policy["skip_fully_masked_camera_encoding"]))
    )
    arguments = phase_runner._phase_arguments(
        runtime,
        "performance_benchmark",
        run_name,
        phase_runner._model_root(experiment),
        dataset_root,
        output_dir,
    )
    arguments.extend(
        optimizer_contract._optimizer_arguments(_control_training(plan))
    )
    sys.argv = ["lerobot-train", *arguments]
    print(json.dumps({"launch": launch_path.name, "run_name": run_name}, sort_keys=True))
    from train_smolvla_state_robustness_cuda_smoke import main as train_main

    training_started = time.perf_counter()
    try:
        train_main()
    except Exception as error:
        _write_failure_report(plan_path, plan, experiment, error)
        raise
    torch.cuda.synchronize()
    completion = {
        "schema_version": 1,
        "status": "complete",
        "stage": "smolvla_state_robustness_cuda_optimizer_smoke_completion",
        "experiment_id": experiment["experiment_id"],
        "run_name": run_name,
        "formal_plan_sha256": file_sha256(plan_path),
        "batch_size": plan["optimizer_smoke"]["batch_size"],
        "steps": plan["optimizer_smoke"]["steps"],
        "elapsed_seconds": time.perf_counter() - training_started,
        "maximum_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "hidden_test_loaded": False,
        "formal_training_authorized": False,
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
