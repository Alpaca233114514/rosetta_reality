"""Run future AutoDL Way gates with fail-closed roots and backup evidence."""

from __future__ import annotations

import os
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import smolvla_autodl_way_sim_gate as way_gate  # noqa: E402
import smolvla_sim_gate as simulator  # noqa: E402

from rosetta_reality.experiment import file_sha256  # noqa: E402
from rosetta_reality.vla.runtime_compatibility import (  # noqa: E402
    require_absolute_environment_directory,
    resolve_runtime_evidence_path,
)


def _plan_path() -> Path:
    try:
        return Path(sys.argv[sys.argv.index("--plan") + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise ValueError("Way v2 gates require an explicit simulation plan.") from error


def _cuda_runtime() -> dict[str, Any]:
    if os.environ.get("ROSETTA_TORCH_DEVICE") != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Way v2 AutoDL gates require CUDA.")
    return {
        "torch_version": torch.__version__,
        "lerobot_version": version("lerobot"),
        "gym_aloha_version": version("gym-aloha"),
        "trackio_version": version("trackio"),
        "device": "cuda",
        "cuda_name": torch.cuda.get_device_name(0),
        "runtime_boundary": "autodl_container_instance",
        "nested_docker_used": False,
        "network_disabled": True,
    }


def _validate_backup(plan_path: Path, *, run_root: Path, durable_root: Path) -> None:
    plan = simulator._load_yaml(plan_path)
    record = plan.get("artifact_backup")
    if not isinstance(record, dict):
        raise ValueError("Way simulation plan has no artifact-backup record.")
    report_path = resolve_runtime_evidence_path(
        str(record.get("report", "")),
        repository_root=REPOSITORY_ROOT,
        run_root=run_root,
    )
    if file_sha256(report_path) != record.get("report_sha256"):
        raise ValueError("Way artifact-backup report changed.")
    report = simulator._load_json(report_path)
    archive_relative = Path(str(report.get("backup_archive", "")))
    if archive_relative.is_absolute() or ".." in archive_relative.parts:
        raise ValueError("Way backup archive path is unsafe.")
    archive = (durable_root / archive_relative).resolve()
    if (
        not archive.is_relative_to(durable_root)
        or not archive.is_file()
        or file_sha256(archive) != report.get("backup_archive_sha256")
        or archive.stat().st_size != report.get("backup_archive_bytes")
        or report.get("status") != "verified"
        or report.get("stage") != "smolvla_remote_durable_artifact_backup"
        or report.get("artifact_id") != plan.get("artifact_id")
        or report.get("artifact_manifest_sha256")
        != plan.get("artifact_manifest_sha256")
        or report.get("archive_file_set_matches_manifest") is not True
        or report.get("artifact_reload_verified") is not True
        or report.get("same_durable_data_disk") is not True
        or report.get("gate_unlock_scope") != "autodl_gate3_gate4_only"
        or report.get("hidden_test_loaded") is not False
    ):
        raise ValueError("Way remote artifact-backup evidence is invalid.")


def main() -> int:
    plan_path = _plan_path()
    run_root = require_absolute_environment_directory("ROSETTA_RUN_ROOT")
    durable_root = require_absolute_environment_directory("ROSETTA_AUTODL_ROOT")
    _validate_backup(plan_path, run_root=run_root, durable_root=durable_root)

    original_runtime_path = way_gate._runtime_plan_path
    original_runtime = simulator._runtime
    original_create_json = simulator.create_json
    original_repository_path = simulator._repository_path
    original_online_policy = simulator._OnlineSmolVLA
    original_triton_cache = os.environ.get("TRITON_CACHE_DIR")
    original_inductor_cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR")

    def runtime_path(raw: str) -> Path:
        return resolve_runtime_evidence_path(
            raw,
            repository_root=REPOSITORY_ROOT,
            run_root=run_root,
        )

    def create_json(path: Path, payload: dict[str, Any]) -> None:
        if payload.get("gate") in {
            "m2_gate_3_small_policy_rollout",
            "m2_gate_4_episode",
            "m2_gate_4_development_task_evaluation",
        }:
            payload["posttrain_compatibility"] = {
                "schema_version": 2,
                "runtime_roots_fail_closed": True,
                "artifact_backup_verified": True,
                "wrapper_sha256": file_sha256(Path(__file__)),
            }
        original_create_json(path, payload)

    way_gate._runtime_plan_path = runtime_path
    simulator._runtime = _cuda_runtime
    simulator.create_json = create_json
    try:
        return way_gate.main()
    finally:
        way_gate._runtime_plan_path = original_runtime_path
        simulator._runtime = original_runtime
        simulator.create_json = original_create_json
        simulator._repository_path = original_repository_path
        simulator._OnlineSmolVLA = original_online_policy
        if original_triton_cache is None:
            os.environ.pop("TRITON_CACHE_DIR", None)
        else:
            os.environ["TRITON_CACHE_DIR"] = original_triton_cache
        if original_inductor_cache is None:
            os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
        else:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = original_inductor_cache


if __name__ == "__main__":
    raise SystemExit(main())
