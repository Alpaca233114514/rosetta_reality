"""Run the plan-bound Way state-jitter two-step optimizer smoke."""

from __future__ import annotations

import copy
import json
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

import run_smolvla_horizon_loss_formal as aster_runner  # noqa: E402
import run_smolvla_phase as phase_runner  # noqa: E402

from rosetta_reality.experiment import (  # noqa: E402
    file_sha256,
    workspace_code_identity,
)
from rosetta_reality.features import create_json  # noqa: E402
from rosetta_reality.vla.state_robustness import profile_from_plan  # noqa: E402

DEFAULT_PLAN = (
    REPOSITORY_ROOT
    / "configs/vla/smolvla_450m_aloha_insertion_way_state_jitter_smoke_002.yaml"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Way state-jitter plan must be a YAML mapping.")
    return value


def _repository_path(raw: str) -> Path:
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Way plan paths must be safe repository-relative paths.")
    result = (REPOSITORY_ROOT / relative).resolve()
    if not result.is_relative_to(REPOSITORY_ROOT) or not result.is_file():
        raise ValueError("Way plan path is missing or outside the repository.")
    return result


def _validate_plan(
    plan_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any]]:
    plan = _load_yaml(plan_path)
    control_registration = plan.get("control_reference", {})
    control_path = _repository_path(str(control_registration.get("plan", "")))
    control, base_path, experiment = aster_runner._validate_plan(control_path)
    parent = plan.get("parent_experiment", {})
    smoke = plan.get("optimizer_smoke", {})
    resources = plan.get("resources", {})
    implementation = plan.get("implementation_files", {})
    preflight = plan.get("reused_no_optimizer_preflight", {})
    base_validation = plan.get("reused_clean_base_validation", {})
    preflight_path = _repository_path(str(preflight.get("path", "")))
    base_validation_path = _repository_path(str(base_validation.get("path", "")))
    if (
        plan.get("schema_version") != 1
        or plan.get("role") != "vla"
        or plan.get("stage") != "m2_state_robustness_optimizer_smoke"
        or plan.get("status") != "preregistered"
        or plan.get("plan_id") != "m2-smolvla450m-way-state-jitter-smoke-002"
        or plan.get("run_name") != "m2-smolvla450m-way-state-jitter-smoke-002"
        or plan.get("supersedes", {}).get("plan")
        != "configs/vla/smolvla_450m_aloha_insertion_way_state_jitter_smoke_001.yaml"
        or plan.get("supersedes", {}).get("plan_sha256")
        != "c1dd048eb3264565408050de714297f68a2139d01273bcf053528eccd18b5a5c"
        or plan.get("supersedes", {}).get("failed_evidence_reason")
        != "trackio_formal_plan_sha256_was_null"
        or plan.get("supersedes", {}).get("reuse_checkpoint_or_optimizer_state")
        is not False
        or control_registration.get("relationship") != "read_only_aster_control"
        or file_sha256(control_path) != control_registration.get("plan_sha256")
        or parent.get("config") != control["parent_experiment"]["config"]
        or parent.get("sha256") != file_sha256(base_path)
        or parent.get("experiment_id") != experiment["experiment_id"]
        or plan.get("loss_contract") != control["loss_contract"]
        or smoke.get("episodes") != experiment["phases"]["smoke"]["episodes"]
        or smoke.get("batch_size") != 8
        or smoke.get("steps") != 2
        or smoke.get("save_freq") != 1
        or smoke.get("save_checkpoint") is not True
        or smoke.get("log_freq") != 1
        or smoke.get("num_workers") != 0
        or smoke.get("persistent_workers") is not False
        or smoke.get("hidden_test_loaded") is not False
        or smoke.get("policy") != control["training"]["policy"]
        or plan.get("training_data")
        != {
            "split": "train",
            "episodes": experiment["phases"]["smoke"]["episodes"],
            "validation_episodes_loaded": False,
            "hidden_test_loaded": False,
        }
        or resources != control["resources"]
        or preflight.get("training_axis_active") is not False
        or file_sha256(preflight_path) != preflight.get("sha256")
        or base_validation.get("state_jitter_active") is not False
        or file_sha256(base_validation_path) != base_validation.get("sha256")
        or not isinstance(implementation, dict)
        or not implementation
        or plan.get("hidden_test_loaded") is not False
    ):
        raise ValueError("Way smoke plan differs from its single state-jitter axis.")
    profile_from_plan(plan)
    for raw_path, expected in implementation.items():
        if file_sha256(_repository_path(str(raw_path))) != expected:
            raise ValueError(f"Way implementation checksum changed: {raw_path}.")
    return plan, control, base_path, experiment


def _prepare_compiler_cache(plan_path: Path, plan: dict[str, Any]) -> dict[str, str]:
    run_root = phase_runner._absolute_root("ROSETTA_RUN_ROOT")
    reuse = plan.get("reused_compiler_cache", {})
    if reuse:
        if (
            reuse.get("source_plan_sha256")
            != "c1dd048eb3264565408050de714297f68a2139d01273bcf053528eccd18b5a5c"
            or reuse.get("cache_root") != "compiler_cache/way-jitter-c1dd048eb326"
            or reuse.get("weights_or_optimizer_state") is not False
        ):
            raise ValueError("Way compiler-cache reuse registration is invalid.")
        root = (run_root / str(reuse["cache_root"])).resolve()
        if not root.is_relative_to(run_root):
            raise ValueError("Way compiler cache escapes the durable run root.")
    else:
        root = run_root / "compiler_cache" / f"way-jitter-{file_sha256(plan_path)[:12]}"
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
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get(
        "HF_DATASETS_OFFLINE"
    ) != "1":
        raise RuntimeError("Way state-jitter smoke must run offline in Docker.")
    plan_path = DEFAULT_PLAN.resolve()
    if "--plan" in sys.argv:
        plan_path = Path(sys.argv[sys.argv.index("--plan") + 1]).resolve()
    plan, control, base_path, experiment = _validate_plan(plan_path)
    resources = plan["resources"]
    if (
        os.environ.get("ROSETTA_DOCKER_MEMORY_LIMIT") != resources["memory_limit"]
        or os.environ.get("ROSETTA_DOCKER_MEMORY_SWAP_LIMIT")
        != resources["memory_swap_limit"]
    ):
        raise ValueError("The active memory boundary differs from Way smoke.")
    contract_path = REPOSITORY_ROOT / experiment["action_contract"]["derived"]
    contract_sha256 = file_sha256(contract_path)
    aster_runner._validate_prerequisites(
        control, experiment, base_path, contract_sha256
    )
    normalization, _view_manifest, dataset_root = aster_runner._validate_normalization(
        control, experiment, base_path, contract_sha256
    )
    preflight_path = _repository_path(plan["reused_no_optimizer_preflight"]["path"])
    base_validation_path = _repository_path(
        plan["reused_clean_base_validation"]["path"]
    )
    control_path = _repository_path(plan["control_reference"]["plan"])
    control_sha256 = file_sha256(control_path)
    aster_runner._validate_preflight(
        preflight_path,
        control,
        experiment,
        base_path,
        contract_sha256,
        file_sha256(normalization),
        control_sha256,
    )
    aster_runner._validate_base_validation(
        base_validation_path,
        control,
        experiment,
        base_path,
        contract_sha256,
        file_sha256(normalization),
        control_sha256,
    )
    run_name = str(plan["run_name"])
    output_dir = (
        phase_runner._absolute_root("ROSETTA_CHECKPOINT_ROOT")
        / experiment["experiment_id"]
        / "smoke"
        / run_name
    )
    if output_dir.exists():
        raise FileExistsError("Way optimizer-smoke output is create-only.")
    compiler_cache = _prepare_compiler_cache(plan_path, plan)
    identity = workspace_code_identity(REPOSITORY_ROOT)
    launch = {
        "schema_version": 1,
        "status": "preregistered",
        "stage": "smolvla_state_robustness_optimizer_smoke_launch",
        "experiment_id": experiment["experiment_id"],
        "run_name": run_name,
        "codename": "Way",
        "formal_plan_sha256": file_sha256(plan_path),
        "control_plan_sha256": control_sha256,
        "experiment_config_sha256": file_sha256(base_path),
        "action_contract_sha256": contract_sha256,
        "normalization_report_sha256": file_sha256(normalization),
        "preflight_report_sha256": file_sha256(preflight_path),
        "base_validation_report_sha256": file_sha256(base_validation_path),
        "loss_contract": plan["loss_contract"],
        "state_robustness_contract": plan["state_robustness_contract"],
        "compiler_cache": compiler_cache,
        "code_identity": identity,
        "hidden_test_loaded": False,
    }
    launch_path = (
        phase_runner._absolute_root("ROSETTA_RUN_ROOT")
        / experiment["experiment_id"]
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
            "ROSETTA_VLA_STATE_ROBUSTNESS_PLAN_PATH": str(plan_path),
            "ROSETTA_VLA_STATE_ROBUSTNESS_PLAN_SHA256": file_sha256(plan_path),
            "ROSETTA_VLA_STATE_ROBUSTNESS_AUTHORIZED": "1",
            "ROSETTA_VLA_FORMAL_PLAN_PATH": str(plan_path),
            "ROSETTA_VLA_FORMAL_PLAN_SHA256": file_sha256(plan_path),
            "ROSETTA_VLA_CODE_REVISION": str(identity["revision"]),
            "ROSETTA_VLA_WORKSPACE_TREE_SHA256": str(identity["workspace_tree_sha256"]),
            "ROSETTA_VLA_WORKSPACE_DIRTY": str(bool(identity["dirty"])).lower(),
            "ROSETTA_VLA_WORKSPACE_FILE_COUNT": str(identity["workspace_file_count"]),
        }
    )
    runtime = copy.deepcopy(experiment)
    runtime["resources"].update(resources)
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
    arguments.extend(aster_runner._optimizer_arguments(control["training"]))
    sys.argv = ["lerobot-train", *arguments]
    print(json.dumps({"launch": launch_path.name, "run_name": run_name}, sort_keys=True))
    from train_smolvla_state_robustness_smoke import main as train_main

    train_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
