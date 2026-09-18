"""Validate an AutoDL RTX 4090 worker without loading dataset rows or model weights."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from importlib.metadata import PackageNotFoundError, distribution, version
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

import run_smolvla_phase as phase_runner  # noqa: E402

from rosetta_reality.data.manifest import validate_cache_checksums  # noqa: E402
from rosetta_reality.experiment import (  # noqa: E402
    file_sha256,
    stable_hash,
    workspace_code_identity,
)
from rosetta_reality.features import create_json  # noqa: E402
from rosetta_reality.vla import load_smolvla_experiment  # noqa: E402

DEFAULT_PROFILE = REPOSITORY_ROOT / "configs/runtime/autodl_rtx4090.yaml"
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "configs/vla/smolvla_450m_aloha_insertion_action_repair_bounded_gripper_003.yaml"
)


def _load_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping: {path.name}.")
    return value


def _load_json_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}.")
    return value


def _validate_recorded_files(root: Path, manifest_path: Path, *, label: str) -> int:
    """Recompute every immutable file record without loading model or dataset content."""

    root = root.resolve()
    manifest = _load_json_mapping(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"{label} manifest contains no file inventory.")
    for raw_relative, raw_record in files.items():
        relative = Path(str(raw_relative))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{label} manifest contains an unsafe file path.")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise FileNotFoundError(f"{label} cache file is missing: {relative.as_posix()}.")
        if not isinstance(raw_record, dict):
            raise ValueError(f"{label} manifest file record is invalid.")
        expected_bytes = raw_record.get("bytes")
        expected_sha256 = raw_record.get("sha256")
        if (
            not isinstance(expected_bytes, int)
            or isinstance(expected_bytes, bool)
            or expected_bytes < 0
            or not isinstance(expected_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
            or path.stat().st_size != expected_bytes
            or file_sha256(path) != expected_sha256
        ):
            raise ValueError(f"{label} cache file identity changed: {relative.as_posix()}.")
    return len(files)


def _absolute_environment_path(name: str) -> Path:
    raw = os.environ.get(name)
    if not raw:
        raise ValueError(f"{name} must be set by scripts/run_autodl.sh.")
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute.")
    return path.resolve()


def _validate_roots(profile: dict[str, Any]) -> dict[str, Path]:
    platform_root = _absolute_environment_path("ROSETTA_AUTODL_PLATFORM_ROOT")
    durable_root = _absolute_environment_path("ROSETTA_AUTODL_ROOT")
    roots = {
        "data": _absolute_environment_path("ROSETTA_DATA_ROOT"),
        "models": _absolute_environment_path("ROSETTA_MODELS_ROOT"),
        "checkpoints": _absolute_environment_path("ROSETTA_CHECKPOINT_ROOT"),
        "artifacts": _absolute_environment_path("ROSETTA_ARTIFACT_ROOT"),
        "runs": _absolute_environment_path("ROSETTA_RUN_ROOT"),
        "trackio": _absolute_environment_path("TRACKIO_DIR"),
        "hf_home": _absolute_environment_path("HF_HOME"),
    }
    if profile.get("storage", {}).get("require_data_disk") is not True:
        raise ValueError("AutoDL profile must require its durable data disk.")
    if not platform_root.is_dir() or not durable_root.is_relative_to(platform_root):
        raise ValueError("Rosetta durable root is not under the AutoDL data disk.")
    for name, path in roots.items():
        if not path.is_dir() or not path.is_relative_to(durable_root):
            raise ValueError(f"Durable {name} root is missing or outside the data disk.")
    if roots["trackio"] != roots["runs"] / "trackio":
        raise ValueError("TRACKIO_DIR must remain under the durable run root.")
    if roots["hf_home"] != roots["models"] / "hf_home":
        raise ValueError("HF_HOME must remain under the durable model root.")
    return {"platform": platform_root, "durable": durable_root, **roots}


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError as error:
        raise RuntimeError(f"Required package is missing: {name}.") from error


def _lerobot_source_revision(expected_revision: str) -> str:
    direct_url_text = distribution("lerobot").read_text("direct_url.json")
    if not direct_url_text:
        raise RuntimeError("LeRobot installation has no immutable source metadata.")
    direct_url = json.loads(direct_url_text)
    url = str(direct_url.get("url", ""))
    expected_suffix = f"/archive/{expected_revision}.zip"
    if not url.startswith("https://github.com/huggingface/lerobot/") or not url.endswith(
        expected_suffix
    ):
        raise RuntimeError("LeRobot source revision differs from the AutoDL profile.")
    return expected_revision


def _validate_accelerator(profile: dict[str, Any]) -> dict[str, Any]:
    accelerator = profile.get("accelerator", {})
    if os.environ.get("ROSETTA_TORCH_DEVICE") != accelerator.get("torch_device"):
        raise ValueError("ROSETTA_TORCH_DEVICE differs from the AutoDL profile.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("AutoDL worker must expose exactly one CUDA GPU.")
    name = torch.cuda.get_device_name(0)
    if re.search(str(accelerator.get("required_name_pattern", "")), name) is None:
        raise RuntimeError(f"CUDA device is not the registered RTX 4090: {name}.")
    properties = torch.cuda.get_device_properties(0)
    total_memory = int(properties.total_memory)
    if total_memory < int(accelerator.get("minimum_total_memory_bytes", 0)):
        raise MemoryError("CUDA device memory is below the registered 4090 floor.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The registered RTX 4090 BF16 path is unavailable.")
    torch.cuda.synchronize(0)
    return {
        "device": "cuda",
        "name": name,
        "device_count": torch.cuda.device_count(),
        "total_memory_bytes": total_memory,
        "capability": list(torch.cuda.get_device_capability(0)),
    }


def doctor(profile_path: Path, config_path: Path) -> Path:
    profile = _load_mapping(profile_path)
    if (
        profile.get("schema_version") != 1
        or profile.get("platform") != "autodl_container_instance"
        or profile.get("runtime_boundary") != "platform_linux_container"
        or profile.get("nested_docker_supported") is not False
        or profile.get("hidden_test_loaded") is not False
        or profile.get("formal_training", {}).get("enabled_by_profile") is not False
        or profile.get("agent_monitoring", {}).get("blocking_command") != "sleep 300"
        or profile.get("agent_monitoring", {}).get("blocking_seconds") != 300
        or profile.get("agent_monitoring", {}).get("fixed_interval") is not True
        or profile.get("agent_monitoring", {}).get("short_polling_allowed") is not False
    ):
        raise ValueError("AutoDL runtime profile is incomplete or unsafe.")
    if os.name != "posix" or not Path("/proc/1/cgroup").is_file():
        raise RuntimeError("AutoDL doctor must run inside the Linux container instance.")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("HF_DATASETS_OFFLINE") != "1":
        raise RuntimeError("AutoDL doctor requires offline model and dataset access.")
    minimum_python = tuple(
        int(value)
        for value in str(profile["python"]["minimum_version"]).split(".")
    )
    if sys.version_info[: len(minimum_python)] < minimum_python:
        raise RuntimeError("AutoDL Python is older than the registered minimum.")

    roots = _validate_roots(profile)
    packages = {
        "torch": torch.__version__,
        "lerobot": _package_version("lerobot"),
        "trackio": _package_version("trackio"),
    }
    for package in ("lerobot", "trackio"):
        expected = str(profile["packages"][package])
        if packages[package] != expected:
            raise RuntimeError(f"{package} version differs: {packages[package]} != {expected}.")
    packages["lerobot_revision"] = _lerobot_source_revision(
        str(profile["packages"]["lerobot_revision"])
    )

    experiment = load_smolvla_experiment(config_path, REPOSITORY_ROOT)
    model_root = phase_runner._model_root(experiment)
    dataset_root = phase_runner._dataset_root(experiment)
    model_manifest_path = model_root / "model_manifest.json"
    dependency_manifest_path = (
        model_root / experiment["model"]["vlm_dependency"]["manifest"]
    )
    dependency_manifest = _load_json_mapping(dependency_manifest_path)
    dependency_relative = Path(str(dependency_manifest.get("cache_layout", "")))
    if (
        not dependency_relative.parts
        or dependency_relative.is_absolute()
        or ".." in dependency_relative.parts
    ):
        raise ValueError("VLM dependency manifest cache layout is unsafe.")
    dependency_root = (roots["hf_home"] / dependency_relative).resolve()
    if not dependency_root.is_relative_to(roots["hf_home"]):
        raise ValueError("VLM dependency cache escaped HF_HOME.")
    model_file_count = _validate_recorded_files(
        model_root, model_manifest_path, label="SmolVLA model"
    )
    dependency_file_count = _validate_recorded_files(
        dependency_root, dependency_manifest_path, label="VLM dependency"
    )
    dataset_file_count = validate_cache_checksums(dataset_root)
    dataset_content_manifest = dataset_root / "cache_checksums.json"
    usage = shutil.disk_usage(roots["durable"])
    report = {
        "schema_version": 1,
        "status": "passed",
        "stage": "autodl_environment_doctor",
        "profile_id": profile["profile_id"],
        "profile_sha256": file_sha256(profile_path),
        "experiment_id": experiment["experiment_id"],
        "experiment_config_sha256": file_sha256(config_path),
        "runtime_boundary": profile["runtime_boundary"],
        "nested_docker_used": False,
        "accelerator": _validate_accelerator(profile),
        "packages": packages,
        "workspace": workspace_code_identity(REPOSITORY_ROOT),
        "storage": {
            "durable_root_free_bytes": int(usage.free),
            "durable_root_total_bytes": int(usage.total),
        },
        "cache_identity": {
            "model_manifest_sha256": file_sha256(model_manifest_path),
            "model_verified_file_count": model_file_count,
            "vlm_dependency_manifest_sha256": file_sha256(dependency_manifest_path),
            "vlm_dependency_verified_file_count": dependency_file_count,
            "dataset_manifest_sha256": file_sha256(dataset_root / "manifest.json"),
            "dataset_content_manifest_sha256": file_sha256(dataset_content_manifest),
            "dataset_verified_file_count": dataset_file_count,
            "model_revision": experiment["model"]["revision"],
            "dataset_revision": experiment["dataset"]["revision"],
        },
        "network_enabled": False,
        "model_weights_loaded": False,
        "dataset_rows_loaded": False,
        "optimizer_created": False,
        "hidden_test_loaded": False,
        "formal_training_authorized": False,
    }
    destination = (
        roots["runs"]
        / experiment["experiment_id"]
        / "hardware"
        / f"autodl-doctor-{stable_hash(report)[:16]}.json"
    )
    create_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"AutoDL doctor report: {destination.name}")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    doctor(args.profile.resolve(), args.config.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
