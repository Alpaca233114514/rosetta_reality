"""Launch only the preregistered no-optimizer SmolVLA action-repair preflight."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import run_smolvla_phase as phase_runner  # noqa: E402

from rosetta_reality.experiment import file_sha256  # noqa: E402
from rosetta_reality.vla import (  # noqa: E402
    load_smolvla_action_space,
    load_smolvla_experiment,
)

DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "configs/vla/smolvla_450m_aloha_insertion_action_repair_001.yaml"
)
RUN_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{2,79}")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("SmolVLA repair evidence must contain a JSON object.")
    return value


def _validate_evidence(
    experiment: dict[str, Any],
    config_path: Path,
    normalization_path: Path,
    diagnostic_path: Path,
) -> tuple[dict[str, Any], Path]:
    action_space = load_smolvla_action_space(experiment, require_explicit=True)
    normalization = _load_json(normalization_path)
    diagnostic = _load_json(diagnostic_path)
    contract_path = REPOSITORY_ROOT / str(experiment["action_contract"]["derived"])
    if (
        normalization.get("status") != "complete"
        or normalization.get("stage") != "smolvla_train_only_normalization"
        or normalization.get("experiment_id") != experiment["experiment_id"]
        or normalization.get("experiment_config_sha256") != file_sha256(config_path)
        or normalization.get("action_space") != action_space.as_dict()
        or normalization.get("target_projection", {}).get("mode")
        != "action_contract_clip"
        or normalization.get("validation_episodes_loaded") is not False
        or normalization.get("hidden_test_loaded") is not False
        or diagnostic.get("status") != "passed"
        or diagnostic.get("stage") != "smolvla_action_space_no_weights_diagnostic"
        or diagnostic.get("experiment_id") != experiment["experiment_id"]
        or diagnostic.get("experiment_config_sha256") != file_sha256(config_path)
        or diagnostic.get("normalization_report_sha256") != file_sha256(normalization_path)
        or diagnostic.get("action_contract_sha256") != file_sha256(contract_path)
        or diagnostic.get("action_space") != action_space.as_dict()
        or diagnostic.get("round_trip", {}).get("passed") is not True
        or diagnostic.get("optimizer_created") is not False
        or diagnostic.get("hidden_test_loaded") is not False
    ):
        raise ValueError("SmolVLA repair evidence is incomplete or has a different identity.")
    run_root = phase_runner._absolute_root("ROSETTA_RUN_ROOT")
    relative_view = Path(str(normalization.get("dataset_view", "")))
    if relative_view.is_absolute() or ".." in relative_view.parts:
        raise ValueError("Repair dataset view path is unsafe.")
    dataset_root = (run_root / relative_view).resolve()
    if not dataset_root.is_relative_to(run_root) or not dataset_root.is_dir():
        raise ValueError("Repair dataset view is missing or outside the run root.")
    return normalization, dataset_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight",))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--normalization-report", type=Path, required=True)
    parser.add_argument("--action-space-report", type=Path, required=True)
    args = parser.parse_args()
    if not RUN_NAME_PATTERN.fullmatch(args.run_name):
        raise ValueError("Repair run name must be lower-case and path safe.")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("HF_DATASETS_OFFLINE") != "1":
        raise RuntimeError("SmolVLA repair preflight requires networking disabled.")
    config_path = args.config.resolve()
    experiment = load_smolvla_experiment(config_path, REPOSITORY_ROOT)
    protocol = experiment.get("repair_protocol", {})
    if (
        experiment.get("status") != "preregistered_diagnostics_only"
        or protocol.get("optimizer_authorized") is not False
        or protocol.get("hidden_test_loaded") is not False
    ):
        raise ValueError("Only the locked diagnostic repair experiment may use this launcher.")
    resources = experiment["resources"]
    if (
        os.environ.get("ROSETTA_DOCKER_MEMORY_LIMIT") != resources["memory_limit"]
        or os.environ.get("ROSETTA_DOCKER_MEMORY_SWAP_LIMIT")
        != resources["memory_swap_limit"]
    ):
        raise ValueError("Active Docker limits differ from the repair experiment.")
    normalization_path = args.normalization_report.resolve()
    diagnostic_path = args.action_space_report.resolve()
    _normalization, dataset_root = _validate_evidence(
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
    os.environ["ROSETTA_VLA_PHASE"] = "action_repair_preflight"
    os.environ["ROSETTA_VLA_EXPERIMENT_CONFIG"] = str(config_path)
    os.environ["ROSETTA_VLA_RUN_NAME"] = str(args.run_name)
    os.environ["ROSETTA_VLA_TRAIN_STATS_REPORT"] = str(normalization_path)
    os.environ["ROSETTA_VLA_NORMALIZATION_SHA256"] = file_sha256(normalization_path)
    os.environ.pop("ROSETTA_VLA_FORMAL_PLAN_SHA256", None)
    os.environ["ROSETTA_VLA_REPAIR_PROTOCOL_SHA256"] = file_sha256(diagnostic_path)
    sys.argv = ["lerobot-train", *arguments]
    from smolvla_forward_check import main as preflight_main

    return preflight_main()


if __name__ == "__main__":
    raise SystemExit(main())
