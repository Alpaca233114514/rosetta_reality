"""Supply CUDA runtime provenance to the preregistered AutoDL Way gates."""

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


def _plan_path() -> Path:
    try:
        return Path(sys.argv[sys.argv.index("--plan") + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise ValueError("Way AutoDL gates require an explicit simulation plan.") from error


def _validate_remote_backup(plan_path: Path) -> None:
    plan = simulator._load_yaml(plan_path)
    record = plan.get("artifact_backup", {})
    if not isinstance(record, dict):
        raise ValueError("Way simulation plan has no artifact-backup record.")
    report_path = way_gate._runtime_plan_path(str(record.get("report", "")))
    if file_sha256(report_path) != record.get("report_sha256"):
        raise ValueError("Way artifact-backup report changed.")
    report = simulator._load_json(report_path)
    durable_root = Path(os.environ.get("ROSETTA_AUTODL_ROOT", "")).resolve()
    archive_relative = Path(str(report.get("backup_archive", "")))
    archive = (durable_root / archive_relative).resolve()
    if (
        not durable_root.is_absolute()
        or archive_relative.is_absolute()
        or ".." in archive_relative.parts
        or not archive.is_relative_to(durable_root)
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


def _cuda_runtime() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Way AutoDL Gate runtime repair requires CUDA.")
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


def main() -> int:
    _validate_remote_backup(_plan_path())
    simulator._runtime = _cuda_runtime
    original_create_json = simulator.create_json

    def create_json(path: Path, payload: dict[str, Any]) -> None:
        if payload.get("gate") in {
            "m2_gate_3_small_policy_rollout",
            "m2_gate_4_episode",
            "m2_gate_4_development_task_evaluation",
        }:
            payload["autodl_way_sim_runtime_repair_sha256"] = file_sha256(
                Path(__file__)
            )
        original_create_json(path, payload)

    simulator.create_json = create_json
    return way_gate.main()


if __name__ == "__main__":
    raise SystemExit(main())
