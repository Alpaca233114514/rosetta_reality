"""Run a plan-bound full-dataset SmolVLA training process with the repaired boundary."""

from __future__ import annotations

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

import train_smolvla_action_repair as repair_train  # noqa: E402
import train_smolvla_trackio as historical_train  # noqa: E402

from rosetta_reality.experiment import file_sha256  # noqa: E402
from rosetta_reality.vla.checkpoint_memory import (  # noqa: E402
    install_checkpoint_memory_trim,
)

_install_masked_camera_encoder_skip = historical_train._install_masked_camera_encoder_skip


def main() -> None:
    if (
        os.environ.get("ROSETTA_VLA_ACTION_REPAIR_FORMAL_AUTHORIZED") != "1"
        or os.environ.get("ROSETTA_VLA_PHASE")
        not in {"formal", "performance_benchmark"}
    ):
        raise PermissionError("The repaired formal optimizer is not plan-authorized.")
    import lerobot.scripts.lerobot_train as lerobot_train
    import run_smolvla_action_repair_formal as formal_runner

    import rosetta_reality.tracking.trackio_lerobot as trackio_bridge

    experiment, action_space, config_path, contract_path = repair_train._repair_context()
    plan_raw = os.environ.get("ROSETTA_VLA_FORMAL_PLAN_PATH")
    if not plan_raw:
        raise ValueError("The repaired formal trainer requires a plan path.")
    plan_path = Path(plan_raw)
    plan, plan_base, plan_experiment = formal_runner._validate_plan(plan_path)
    if (
        plan_base != config_path.resolve()
        or plan_experiment["experiment_id"] != experiment["experiment_id"]
        or os.environ.get("ROSETTA_VLA_FORMAL_PLAN_SHA256") != file_sha256(plan_path)
        or os.environ.get("ROSETTA_VLA_RUN_NAME")
        not in {plan["run_name"], plan["optimizer_smoke"]["run_name"]}
        or experiment.get("repair_protocol", {}).get("hidden_test_loaded") is not False
    ):
        raise ValueError("The active formal process differs from Faust.")
    repair_train._validate_projected_statistics(
        experiment, action_space, config_path, contract_path
    )
    if os.environ.get("ROSETTA_VLA_SKIP_FULLY_MASKED_CAMERA_ENCODING") == "1":
        historical_train._install_masked_camera_encoder_skip()
    historical_train._install_train_only_statistics(lerobot_train)
    repair_train._install_projection(
        lerobot_train, experiment, action_space, contract_path
    )
    install_checkpoint_memory_trim(lerobot_train)
    original_public_config = trackio_bridge._public_config

    def public_config(cfg: Any, active_experiment: dict[str, Any], phase: str) -> dict[str, Any]:
        payload = original_public_config(cfg, active_experiment, phase)
        payload.update(
            {
                "action_representation_adapter": action_space.representation_adapter,
                "action_target_projection": action_space.target_projection,
                "bounded_gripper_decoder": True,
            }
        )
        return payload

    trackio_bridge._experiment_config = lambda: experiment
    trackio_bridge._public_config = public_config
    lerobot_train.WandBLogger = trackio_bridge.TrackioLogger
    try:
        lerobot_train.main()
    finally:
        trackio_bridge.finish_trackio()


if __name__ == "__main__":
    main()
