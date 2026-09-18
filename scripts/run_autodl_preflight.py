"""Run the bounded-gripper no-optimizer forward in the registered AutoDL container."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import run_smolvla_action_repair_phase as repair_phase  # noqa: E402
import run_smolvla_phase as phase_runner  # noqa: E402

from rosetta_reality.experiment import file_sha256  # noqa: E402
from rosetta_reality.features import create_json  # noqa: E402
from rosetta_reality.vla import (  # noqa: E402
    load_smolvla_action_space,
    load_smolvla_experiment,
)

DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "configs/vla/smolvla_450m_aloha_insertion_action_repair_bounded_gripper_003.yaml"
)
RUN_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{2,79}")


def _validate_profile() -> tuple[Path, dict[str, object]]:
    raw = os.environ.get("ROSETTA_AUTODL_RUNTIME_PROFILE")
    if not raw:
        raise ValueError("AutoDL runtime profile is not set.")
    path = Path(raw)
    profile = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        not path.is_absolute()
        or not isinstance(profile, dict)
        or profile.get("platform") != "autodl_container_instance"
        or profile.get("runtime_boundary") != "platform_linux_container"
        or profile.get("nested_docker_supported") is not False
        or profile.get("formal_training", {}).get("enabled_by_profile") is not False
        or os.environ.get("ROSETTA_AUTODL_NO_OPTIMIZER_AUTHORIZED") != "1"
        or os.environ.get("ROSETTA_TORCH_DEVICE") != "cuda"
        or not torch.cuda.is_available()
    ):
        raise RuntimeError("AutoDL no-optimizer CUDA boundary is invalid.")
    return path, profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--normalization-report", type=Path, required=True)
    parser.add_argument("--action-space-report", type=Path, required=True)
    args = parser.parse_args()
    if not RUN_NAME_PATTERN.fullmatch(args.run_name):
        raise ValueError("AutoDL preflight run name must be lower-case and path safe.")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("HF_DATASETS_OFFLINE") != "1":
        raise RuntimeError("AutoDL preflight requires offline caches.")
    profile_path, profile = _validate_profile()

    config_path = args.config.resolve()
    normalization_path = args.normalization_report.resolve()
    diagnostic_path = args.action_space_report.resolve()
    experiment = load_smolvla_experiment(config_path, REPOSITORY_ROOT)
    protocol = experiment.get("repair_protocol", {})
    if (
        experiment.get("status") != "preregistered_action_repair_smoke_and_overfit"
        or protocol.get("hidden_test_loaded") is not False
        or protocol.get("historical_checkpoints_are_initialization") is not False
    ):
        raise ValueError("AutoDL preflight requires the fresh bounded-gripper experiment.")
    dataset_root = repair_phase._validate_repair_evidence(
        experiment,
        config_path,
        normalization_path,
        diagnostic_path,
    )
    action_space = load_smolvla_action_space(experiment, require_explicit=True)
    model_root = phase_runner._model_root(experiment)
    checkpoint_root = phase_runner._absolute_root("ROSETTA_CHECKPOINT_ROOT")
    output_dir = (
        checkpoint_root
        / str(experiment["experiment_id"])
        / "preflight"
        / str(args.run_name)
    )
    arguments = phase_runner._phase_arguments(
        experiment,
        "preflight",
        str(args.run_name),
        model_root,
        dataset_root,
        output_dir,
    )
    arguments.append(
        f"--policy.adapt_to_pi_aloha={str(action_space.adapt_to_pi_aloha).lower()}"
    )
    os.environ["ROSETTA_VLA_PHASE"] = "autodl_action_repair_preflight"
    os.environ["ROSETTA_VLA_EXPERIMENT_CONFIG"] = str(config_path)
    os.environ["ROSETTA_VLA_RUN_NAME"] = str(args.run_name)
    os.environ["ROSETTA_VLA_TRAIN_STATS_REPORT"] = str(normalization_path)
    os.environ["ROSETTA_VLA_NORMALIZATION_SHA256"] = file_sha256(normalization_path)
    os.environ["ROSETTA_VLA_REPAIR_PROTOCOL_SHA256"] = file_sha256(diagnostic_path)
    os.environ.pop("ROSETTA_VLA_FORMAL_PLAN_SHA256", None)
    sys.argv = ["lerobot-train", *arguments]

    torch.cuda.reset_peak_memory_stats()
    from smolvla_forward_check import main as forward_main

    exit_code = forward_main()
    torch.cuda.synchronize()
    run_root = phase_runner._absolute_root("ROSETTA_RUN_ROOT")
    preflight_path = (
        run_root / str(experiment["experiment_id"]) / "preflight" / f"{args.run_name}.json"
    )
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (
        exit_code != 0
        or preflight.get("status") != "passed"
        or preflight.get("optimizer_created") is not False
        or preflight.get("hidden_test_loaded") is not False
    ):
        raise RuntimeError("AutoDL CUDA forward did not produce valid preflight evidence.")
    supplemental = {
        "schema_version": 1,
        "status": "passed",
        "stage": "autodl_cuda_no_optimizer_forward_supplement",
        "profile_id": profile["profile_id"],
        "profile_sha256": file_sha256(profile_path),
        "experiment_id": experiment["experiment_id"],
        "run_name": args.run_name,
        "preflight_report_sha256": file_sha256(preflight_path),
        "device": "cuda",
        "device_name": torch.cuda.get_device_name(0),
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "maximum_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "optimizer_created": False,
        "hidden_test_loaded": False,
        "formal_training_authorized": False,
    }
    destination = (
        run_root
        / str(experiment["experiment_id"])
        / "hardware"
        / f"{args.run_name}-cuda-forward.json"
    )
    create_json(destination, supplemental)
    print(json.dumps(supplemental, indent=2, sort_keys=True))
    print(f"AutoDL CUDA forward supplement: {destination.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
