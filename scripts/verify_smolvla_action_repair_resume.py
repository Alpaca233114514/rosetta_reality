"""Verify one-step repair resume provenance and serialized action-boundary identity."""

from __future__ import annotations

import argparse
import json
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

import resume_smolvla_overfit as legacy_resume  # noqa: E402
import run_smolvla_phase as phase_runner  # noqa: E402
import verify_smolvla_resume as legacy_verify  # noqa: E402

from rosetta_reality.experiment import file_sha256  # noqa: E402
from rosetta_reality.features import create_json  # noqa: E402
from rosetta_reality.vla import load_smolvla_experiment  # noqa: E402
from rosetta_reality.vla.processor import (  # noqa: E402
    PI_ALOHA_POSTPROCESSOR_REGISTRY_NAME,
    PI_ALOHA_PREPROCESSOR_REGISTRY_NAME,
    REGISTRY_NAME,
)

DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "configs/vla/smolvla_450m_aloha_insertion_action_repair_smoke_001.yaml"
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}.")
    return value


def _processor_registries(path: Path) -> list[str]:
    config = _load_json(path)
    steps = config.get("steps")
    if not isinstance(steps, list):
        raise ValueError("Saved processor config has no step list.")
    names = [
        step.get("registry_name")
        for step in steps
        if isinstance(step, dict) and isinstance(step.get("registry_name"), str)
    ]
    if len(names) != len(steps):
        raise ValueError("Saved processor config contains an unnamed step.")
    return names


def _validate_boundary(
    source_pretrained: Path, resumed_pretrained: Path
) -> dict[str, Any]:
    source_pre = source_pretrained / "policy_preprocessor.json"
    source_post = source_pretrained / "policy_postprocessor.json"
    resumed_pre = resumed_pretrained / "policy_preprocessor.json"
    resumed_post = resumed_pretrained / "policy_postprocessor.json"
    source_pre_names = _processor_registries(source_pre)
    source_post_names = _processor_registries(source_post)
    resumed_pre_names = _processor_registries(resumed_pre)
    resumed_post_names = _processor_registries(resumed_post)
    if (
        source_pre_names != resumed_pre_names
        or source_post_names != resumed_post_names
        or source_pre_names.index(REGISTRY_NAME) + 1
        != source_pre_names.index(PI_ALOHA_PREPROCESSOR_REGISTRY_NAME)
        or source_pre_names.index(PI_ALOHA_PREPROCESSOR_REGISTRY_NAME) + 1
        != source_pre_names.index("normalizer_processor")
        or source_post_names.index("unnormalizer_processor") + 1
        != source_post_names.index(PI_ALOHA_POSTPROCESSOR_REGISTRY_NAME)
        or file_sha256(source_pre) != file_sha256(resumed_pre)
        or file_sha256(source_post) != file_sha256(resumed_post)
    ):
        raise ValueError("Explicit resume changed or reordered the repair action boundary.")
    return {
        "preserved": True,
        "preprocessor_steps": source_pre_names,
        "postprocessor_steps": source_post_names,
        "preprocessor_sha256": file_sha256(source_pre),
        "postprocessor_sha256": file_sha256(source_post),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--resumed-checkpoint", type=Path, required=True)
    parser.add_argument("--container-exit-code", type=int, required=True)
    parser.add_argument("--oom-event-count", type=int, required=True)
    args = parser.parse_args()
    if not legacy_verify.RUN_NAME_PATTERN.fullmatch(args.run_name):
        raise ValueError("--run-name must be lower-case and path safe.")
    if args.container_exit_code != 0 or args.oom_event_count != 0:
        raise RuntimeError("The repair resume container did not exit cleanly within budget.")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("HF_DATASETS_OFFLINE") != "1":
        raise RuntimeError("Repair resume verification requires networking disabled.")

    config_path = args.config.resolve()
    experiment = load_smolvla_experiment(config_path, REPOSITORY_ROOT)
    contract_path = REPOSITORY_ROOT / str(experiment["action_contract"]["derived"])
    source_step_dir, source_pretrained, source_step, source_run_name = (
        legacy_resume._source_checkpoint(args.source_checkpoint, experiment)
    )
    source_training_state = source_step_dir / "training_state"
    target_step = source_step + 1
    resumed_step_dir, resumed_pretrained, resumed_training_state, files = (
        legacy_verify._resumed_checkpoint(
            args.resumed_checkpoint,
            experiment,
            args.run_name,
            target_step,
        )
    )
    saved_resume = legacy_verify._validate_saved_resume(
        experiment,
        args.run_name,
        source_step,
        resumed_step_dir,
        resumed_pretrained,
        resumed_training_state,
    )
    state_progression = legacy_verify._validate_state_progression(
        source_pretrained,
        source_training_state,
        resumed_pretrained,
        resumed_training_state,
        source_step,
    )
    tracking = legacy_verify._validate_trackio_resume(
        experiment,
        args.run_name,
        source_run_name,
        source_step,
    )
    boundary = _validate_boundary(source_pretrained, resumed_pretrained)
    checkpoint_root = phase_runner._absolute_root("ROSETTA_CHECKPOINT_ROOT")
    report = {
        "schema_version": 1,
        "status": "passed",
        "stage": "smolvla_action_repair_explicit_resume_verification",
        "experiment_id": experiment["experiment_id"],
        "experiment_config_sha256": file_sha256(config_path),
        "action_contract_sha256": file_sha256(contract_path),
        "verification_script_sha256": file_sha256(Path(__file__)),
        "resume_runner_sha256": file_sha256(
            REPOSITORY_ROOT / "scripts/resume_smolvla_action_repair.py"
        ),
        "resume_mechanism": "lerobot_config_path_with_resume_true",
        "source_run_name": source_run_name,
        "source_checkpoint": source_step_dir.relative_to(checkpoint_root).as_posix(),
        "source_step": source_step,
        "run_name": args.run_name,
        "resumed_checkpoint": resumed_step_dir.relative_to(checkpoint_root).as_posix(),
        "resumed_checkpoint_files": files,
        "saved_resume": saved_resume,
        "state_progression": state_progression,
        "serialized_action_boundary": boundary,
        "tracking": tracking,
        "model_revision": experiment["model"]["revision"],
        "dataset_revision": experiment["dataset"]["revision"],
        "episodes_loaded": experiment["phases"]["overfit"]["episodes"],
        "hidden_test_loaded": False,
        "network_disabled": True,
        "runtime_observation": {
            "source": "orchestrated_container_exit_and_docker_events",
            "container_exit_code": args.container_exit_code,
            "oom_event_count": args.oom_event_count,
            "memory_limit": experiment["resources"]["memory_limit"],
            "memory_swap_limit": experiment["resources"]["memory_swap_limit"],
        },
        "acceptance": {
            "source_checkpoint_complete": True,
            "optimizer_scheduler_rng_restored": True,
            "exactly_one_optimizer_step_completed": True,
            "checkpoint_written": True,
            "action_boundary_preserved": True,
            "trackio_run_readable": True,
            "no_resource_limit_violation": True,
            "hidden_test_loaded": False,
        },
    }
    json.dumps(report, allow_nan=False)
    run_root = phase_runner._absolute_root("ROSETTA_RUN_ROOT")
    destination = (
        run_root
        / str(experiment["experiment_id"])
        / "verification"
        / f"{args.run_name}.json"
    )
    create_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report: {destination.relative_to(REPOSITORY_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
