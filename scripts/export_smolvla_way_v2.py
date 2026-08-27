"""Export Way with the version-2 normalization compatibility boundary."""

from __future__ import annotations

import copy
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

import evaluate_smolvla_validation as evaluator  # noqa: E402
import export_smolvla as exporter  # noqa: E402
import export_smolvla_way as way_exporter  # noqa: E402
import run_smolvla_state_robustness_cuda_formal as formal_runner  # noqa: E402

from rosetta_reality.experiment import file_sha256, workspace_code_identity  # noqa: E402
from rosetta_reality.vla.runtime_compatibility import (  # noqa: E402
    plan_with_normalization_alias,
)


def _plan_path() -> Path:
    try:
        return Path(sys.argv[sys.argv.index("--plan") + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise ValueError("Way v2 export requires an explicit --plan path.") from error


def main() -> int:
    plan_path = _plan_path()
    formal_plan, _base_path, _experiment = formal_runner._validate_plan(plan_path)
    compatible_plan = plan_with_normalization_alias(formal_plan)
    original_validate_plan = formal_runner._validate_plan
    original_create_json = exporter.create_json
    original_evaluator_formal_runner = evaluator.formal_runner
    original_checkpoint_source = evaluator._checkpoint_source
    original_statistics_validator = evaluator._validate_checkpoint_statistics
    original_dataset_loader = evaluator._load_policy_and_dataset
    original_exporter_evaluator = exporter.evaluator
    original_exporter_formal_runner = exporter.formal_runner
    original_copy_policy = exporter._copy_policy
    original_artifact_loader = exporter._load_artifact_policy
    original_triton_cache = os.environ.get("TRITON_CACHE_DIR")
    original_inductor_cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR")

    def validate_plan(
        active_path: Path,
    ) -> tuple[dict[str, Any], Path, dict[str, Any]]:
        validated, base_path, experiment = original_validate_plan(active_path)
        if active_path.resolve() != plan_path or validated["plan_id"] != formal_plan["plan_id"]:
            raise ValueError("The delegated Way export plan identity changed.")
        return copy.deepcopy(compatible_plan), base_path, experiment

    def create_json(path: Path, payload: dict[str, Any]) -> None:
        if path.name in {"config.json", "manifest.json"}:
            payload["posttrain_compatibility"] = {
                "schema_version": 2,
                "normalization_alias_resolved": True,
                "wrapper_sha256": file_sha256(Path(__file__)),
                "code_identity": workspace_code_identity(REPOSITORY_ROOT),
                "training_identity_changed": False,
            }
        original_create_json(path, payload)

    formal_runner._validate_plan = validate_plan
    exporter.create_json = create_json
    try:
        return way_exporter.main()
    finally:
        formal_runner._validate_plan = original_validate_plan
        exporter.create_json = original_create_json
        evaluator.formal_runner = original_evaluator_formal_runner
        evaluator._checkpoint_source = original_checkpoint_source
        evaluator._validate_checkpoint_statistics = original_statistics_validator
        evaluator._load_policy_and_dataset = original_dataset_loader
        exporter.evaluator = original_exporter_evaluator
        exporter.formal_runner = original_exporter_formal_runner
        exporter._copy_policy = original_copy_policy
        exporter._load_artifact_policy = original_artifact_loader
        if original_triton_cache is None:
            os.environ.pop("TRITON_CACHE_DIR", None)
        else:
            os.environ["TRITON_CACHE_DIR"] = original_triton_cache
        if original_inductor_cache is None:
            os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
        else:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = original_inductor_cache


if __name__ == "__main__":
    raise SystemExit(main())
