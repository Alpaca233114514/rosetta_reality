"""Train the plan-bound Way formal run on the AutoDL CUDA worker."""

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
from rosetta_reality.tracking.trackio_accelerator import (  # noqa: E402
    AcceleratorTrackioLogger,
)
from rosetta_reality.vla.checkpoint_accelerator_memory import (  # noqa: E402
    install_checkpoint_memory_trim,
)
from rosetta_reality.vla.horizon_loss import (  # noqa: E402
    install_horizon_weight_profile,
)
from rosetta_reality.vla.horizon_loss import (  # noqa: E402
    profile_from_plan as horizon_profile_from_plan,
)
from rosetta_reality.vla.state_robustness import (  # noqa: E402
    install_state_robustness_profile,
)
from rosetta_reality.vla.state_robustness import (  # noqa: E402
    profile_from_plan as state_profile_from_plan,
)


def main() -> None:
    if (
        os.environ.get("ROSETTA_VLA_STATE_ROBUSTNESS_CUDA_FORMAL_AUTHORIZED") != "1"
        or os.environ.get("ROSETTA_VLA_PHASE") != "formal"
        or os.environ.get("ROSETTA_TORCH_DEVICE") != "cuda"
    ):
        raise PermissionError("Way CUDA formal training is not plan-authorized.")
    import lerobot.scripts.lerobot_train as lerobot_train
    import run_smolvla_state_robustness_cuda_formal as runner

    import rosetta_reality.tracking.trackio_lerobot as trackio_bridge

    experiment, action_space, config_path, contract_path = repair_train._repair_context()
    plan_raw = os.environ.get("ROSETTA_VLA_FORMAL_PLAN_PATH")
    if not plan_raw:
        raise ValueError("Way CUDA formal training requires its plan path.")
    plan_path = Path(plan_raw)
    plan, base_path, validated_experiment = runner._validate_plan(plan_path)
    if (
        base_path != config_path.resolve()
        or validated_experiment["experiment_id"] != experiment["experiment_id"]
        or os.environ.get("ROSETTA_VLA_FORMAL_PLAN_SHA256") != file_sha256(plan_path)
        or os.environ.get("ROSETTA_VLA_RUN_NAME") != plan["run_name"]
        or plan.get("furnace_program", {}).get("codename") != "Way"
        or plan.get("hidden_test_loaded") is not False
    ):
        raise ValueError("The active formal process differs from Way.")
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
    import lerobot.policies.smolvla.modeling_smolvla as modeling_smolvla

    horizon_profile = horizon_profile_from_plan(
        plan, int(experiment["model"]["policy"]["chunk_size"])
    )
    state_profile = state_profile_from_plan(plan)
    install_horizon_weight_profile(modeling_smolvla, horizon_profile)
    install_state_robustness_profile(modeling_smolvla, state_profile)
    original_public_config = trackio_bridge._public_config

    def public_config(
        cfg: Any, active_experiment: dict[str, Any], phase: str
    ) -> dict[str, Any]:
        payload = original_public_config(cfg, active_experiment, phase)
        payload.update(
            {
                "accelerator": "cuda",
                "memory_limit": "autodl_platform_container",
                "memory_swap_limit": "autodl_platform_container",
                "action_representation_adapter": action_space.representation_adapter,
                "action_target_projection": action_space.target_projection,
                "bounded_gripper_decoder": True,
                "temporal_loss_profile": horizon_profile.name,
                "temporal_loss_normalization": horizon_profile.normalization,
                "state_robustness_profile": state_profile.name,
                "state_noise_std_normalized": (
                    state_profile.normalized_standard_deviation
                ),
                "state_jitter_training_only": True,
                "state_jitter_target_semantics": state_profile.target_semantics,
                "autodl_runtime_profile_sha256": plan["runtime_profile"]["sha256"],
                "cuda_smoke_acceptance_sha256": plan["cuda_smoke"][
                    "acceptance_sha256"
                ],
            }
        )
        return payload

    trackio_bridge._experiment_config = lambda: experiment
    trackio_bridge._public_config = public_config
    lerobot_train.WandBLogger = AcceleratorTrackioLogger
    try:
        lerobot_train.main()
    finally:
        trackio_bridge.finish_trackio()


if __name__ == "__main__":
    main()
