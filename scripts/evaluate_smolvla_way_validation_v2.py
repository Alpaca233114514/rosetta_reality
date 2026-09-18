"""Run Way validation with the version-2 post-training compatibility boundary."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import evaluate_smolvla_aster_validation as aster_evaluator  # noqa: E402
import evaluate_smolvla_validation as evaluator  # noqa: E402
import evaluate_smolvla_way_validation as way_evaluator  # noqa: E402
import run_smolvla_phase as phase_runner  # noqa: E402
import run_smolvla_state_robustness_cuda_formal as formal_runner  # noqa: E402

from rosetta_reality.experiment import file_sha256, workspace_code_identity  # noqa: E402
from rosetta_reality.vla.runtime_compatibility import (  # noqa: E402
    plan_with_normalization_alias,
    require_absolute_environment_directory,
    resolve_tokenizer_identity,
)


def _plan_path() -> Path:
    try:
        return Path(sys.argv[sys.argv.index("--plan") + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise ValueError("Way v2 validation requires an explicit --plan path.") from error


def _checkpoint_step(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int | None:
    if len(args) >= 4:
        value = args[3]
    else:
        value = kwargs.get("checkpoint_step")
    return None if value is None else int(value)


def main() -> int:
    plan_path = _plan_path()
    formal_plan, _base_path, experiment = formal_runner._validate_plan(plan_path)
    compatible_plan = plan_with_normalization_alias(formal_plan)
    base_root = phase_runner._model_root(experiment).resolve()
    expected_dependency = compatible_plan.get("base_tokenizer_source")
    if expected_dependency is not None and not isinstance(expected_dependency, dict):
        raise ValueError("base_tokenizer_source must be a mapping when registered.")

    original_validate_plan = formal_runner._validate_plan
    original_tokenizer_hashes = evaluator._tokenizer_hashes
    original_loader = evaluator._load_policy_and_dataset
    original_create_json = evaluator.create_json
    original_sync = evaluator._sync
    original_evaluator_formal_runner = evaluator.formal_runner
    original_checkpoint_source = evaluator._checkpoint_source
    original_statistics_validator = evaluator._validate_checkpoint_statistics
    original_aster_formal_runner = aster_evaluator.formal_runner
    active_tokenizer_identity: dict[str, Any] | None = None

    def validate_plan(
        active_path: Path,
    ) -> tuple[dict[str, Any], Path, dict[str, Any]]:
        validated, base_path, active_experiment = original_validate_plan(active_path)
        if active_path.resolve() != plan_path or validated["plan_id"] != formal_plan["plan_id"]:
            raise ValueError("The delegated Way validation plan identity changed.")
        return copy.deepcopy(compatible_plan), base_path, active_experiment

    def tokenizer_hashes(source_dir: Path) -> dict[str, str]:
        nonlocal active_tokenizer_identity
        tokenizer = source_dir / "tokenizer"
        if tokenizer.is_dir() and any(path.is_file() for path in tokenizer.rglob("*")):
            hf_home = REPOSITORY_ROOT
        else:
            hf_home = require_absolute_environment_directory("HF_HOME")
        hashes, identity = resolve_tokenizer_identity(
            source_dir,
            base_model_root=base_root,
            experiment=experiment,
            hf_home=hf_home,
            expected_tokenizer_identity=expected_dependency,
        )
        active_tokenizer_identity = identity
        return hashes

    def load_policy_and_dataset(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        nonlocal active_tokenizer_identity
        active_tokenizer_identity = None
        result = original_loader(*args, **kwargs)
        if len(result) < 5 or not isinstance(result[4], dict):
            raise ValueError("The delegated evaluator returned an invalid source identity.")
        if active_tokenizer_identity is None:
            raise ValueError("The delegated evaluator did not resolve a tokenizer identity.")
        result[4]["tokenizer_source"] = copy.deepcopy(active_tokenizer_identity)
        result[4]["checkpoint_step"] = _checkpoint_step(args, kwargs)
        return result

    def create_json(path: Path, payload: dict[str, Any]) -> None:
        payload["posttrain_compatibility"] = {
            "schema_version": 2,
            "normalization_alias_resolved": True,
            "tokenizer_source_resolved": True,
            "wrapper_sha256": file_sha256(Path(__file__)),
            "code_identity": workspace_code_identity(REPOSITORY_ROOT),
            "training_identity_changed": False,
        }
        original_create_json(path, payload)

    formal_runner._validate_plan = validate_plan
    evaluator._tokenizer_hashes = tokenizer_hashes
    evaluator._load_policy_and_dataset = load_policy_and_dataset
    evaluator.create_json = create_json
    try:
        return way_evaluator.main()
    finally:
        formal_runner._validate_plan = original_validate_plan
        evaluator._tokenizer_hashes = original_tokenizer_hashes
        evaluator._load_policy_and_dataset = original_loader
        evaluator.create_json = original_create_json
        evaluator._sync = original_sync
        evaluator.formal_runner = original_evaluator_formal_runner
        evaluator._checkpoint_source = original_checkpoint_source
        evaluator._validate_checkpoint_statistics = original_statistics_validator
        aster_evaluator.formal_runner = original_aster_formal_runner


if __name__ == "__main__":
    raise SystemExit(main())
