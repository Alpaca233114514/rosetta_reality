"""Run the pinned LeRobot trainer with the plan-declared version-2 features.

This is the single version-2 trainer entry point.  It replaces the historical
per-experiment ``train_smolvla_*`` stack: instead of one script per furnace
that layered private monkeypatches over each other, every local extension is
declared in the hash-bound version-2 plan, resolved against the feature
registry and installed in declaration order with rollback and reverse-order
restore.  The pinned upstream trainer remains the only training loop.

Fail-closed boundaries:

- the process must have been validated and assembled by ``run_smolvla_v2``;
- the plan must pass the version-2 schema before anything is installed;
- feature installation order is exactly the plan's declaration order and any
  failure rolls the already-installed features back.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from rosetta_reality.vla import (  # noqa: E402
    load_smolvla_action_space,
    load_smolvla_experiment,
)
from rosetta_reality.vla.training import (  # noqa: E402
    TrainingContext,
    feature_stack_from_plan,
    load_v2_plan,
    validate_plan_structure,
)
from rosetta_reality.vla.training.features import FEATURE_FACTORIES  # noqa: E402

LAUNCHER_VALIDATED_ENV = "ROSETTA_VLA_V2_LAUNCHER_VALIDATED"
PLAN_PATH_ENV = "ROSETTA_VLA_V2_PLAN_PATH"


def _training_context(plan_path: Path) -> TrainingContext:
    experiment_path_raw = os.environ.get("ROSETTA_VLA_EXPERIMENT_CONFIG")
    if not experiment_path_raw:
        raise ValueError("The v2 trainer requires the launcher's runtime experiment file.")
    experiment_path = Path(experiment_path_raw)
    if not experiment_path.is_absolute() or not experiment_path.is_file():
        raise ValueError("The v2 runtime experiment file must be an absolute existing path.")
    experiment = load_smolvla_experiment(experiment_path, REPOSITORY_ROOT)
    action_space = load_smolvla_action_space(experiment, require_explicit=True)
    contract_path = REPOSITORY_ROOT / str(experiment["action_contract"]["derived"])
    if not contract_path.is_file():
        raise FileNotFoundError("The runtime Action Contract is missing.")
    plan = load_v2_plan(plan_path, REPOSITORY_ROOT)
    normalization_relative = Path(str(plan["normalization"]["report"]))
    if normalization_relative.is_absolute() or ".." in normalization_relative.parts:
        raise ValueError("The v2 normalization report path is unsafe.")
    phase = os.environ.get("ROSETTA_VLA_PHASE", "")
    device = os.environ.get("ROSETTA_TORCH_DEVICE", "")
    run_name = os.environ.get("ROSETTA_VLA_RUN_NAME", "")
    return TrainingContext(
        plan=plan,
        experiment=experiment,
        action_space=action_space,
        plan_path=plan_path,
        experiment_path=experiment_path,
        contract_path=contract_path,
        normalization_report=REPOSITORY_ROOT / normalization_relative,
        phase=phase,
        device=device,
        run_name=run_name,
    )


def main() -> None:
    if os.environ.get(LAUNCHER_VALIDATED_ENV) != "1":
        raise PermissionError(
            "The v2 trainer only runs behind the validated run_smolvla_v2 launcher."
        )
    plan_raw = os.environ.get(PLAN_PATH_ENV)
    if not plan_raw:
        raise ValueError("The v2 trainer requires its plan path environment variable.")
    plan_path = Path(plan_raw)
    if not plan_path.is_absolute() or not plan_path.is_file():
        raise ValueError("The v2 plan path must identify an absolute existing file.")

    plan = load_v2_plan(plan_path, REPOSITORY_ROOT)
    validate_plan_structure(plan, known_features=FEATURE_FACTORIES)
    context = _training_context(plan_path)

    import lerobot.scripts.lerobot_train as lerobot_train

    from rosetta_reality.tracking.trackio_lerobot import finish_trackio

    stack = feature_stack_from_plan(plan)
    installed = stack.install_all(context)
    try:
        print(f"Installed v2 training features: {', '.join(installed)}")
        lerobot_train.main()
    finally:
        stack.restore_all(context)
        finish_trackio()


if __name__ == "__main__":
    main()
