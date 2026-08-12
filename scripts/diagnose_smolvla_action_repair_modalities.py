"""Run the fixed Faust modality ablations through the repaired action boundary."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import diagnose_smolvla_modalities as diagnostic  # noqa: E402
import evaluate_smolvla_action_repair_validation as repair_evaluator  # noqa: E402
import evaluate_smolvla_validation as evaluator  # noqa: E402
import run_smolvla_action_repair_formal as formal_runner  # noqa: E402

from rosetta_reality.experiment import file_sha256  # noqa: E402
from rosetta_reality.sim import load_action_contract  # noqa: E402
from rosetta_reality.vla import load_smolvla_action_space  # noqa: E402
from rosetta_reality.vla.processor import ensure_smolvla_action_boundary  # noqa: E402


def main() -> int:
    diagnostic.formal_runner = formal_runner
    diagnostic._load_historical_plan = formal_runner._validate_plan
    evaluator.formal_runner = formal_runner
    evaluator._checkpoint_source = repair_evaluator._checkpoint_source
    evaluator._validate_checkpoint_statistics = repair_evaluator._validate_checkpoint_statistics
    original_loader = evaluator._load_policy_and_dataset

    def load_policy_and_dataset(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        result = original_loader(*args, **kwargs)
        _policy, preprocessor, postprocessor, _dataset, _source, _hashes = result
        experiment = args[1]
        contract_path = REPOSITORY_ROOT / str(experiment["action_contract"]["derived"])
        ensure_smolvla_action_boundary(
            preprocessor,
            postprocessor,
            load_action_contract(contract_path),
            load_smolvla_action_space(experiment, require_explicit=True),
            action_contract_sha256=file_sha256(contract_path),
            upstream_revision=str(experiment["upstream"]["revision"]),
        )
        return result

    evaluator._load_policy_and_dataset = load_policy_and_dataset
    diagnostic.evaluator = evaluator
    original_create_json = diagnostic.create_json

    def create_json(path: Path, payload: dict[str, Any]) -> None:
        payload["bounded_gripper_decoder"] = True
        payload["action_repair_diagnostic_script_sha256"] = file_sha256(Path(__file__))
        original_create_json(path, payload)

    diagnostic.create_json = create_json
    return diagnostic.main()


if __name__ == "__main__":
    raise SystemExit(main())
