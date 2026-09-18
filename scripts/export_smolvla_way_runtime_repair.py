"""Export Way through a preregistered validation-schema compatibility layer."""

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

import export_smolvla as exporter  # noqa: E402
import export_smolvla_way as way_exporter  # noqa: E402
import run_smolvla_state_robustness_cuda_formal as formal_runner  # noqa: E402

from rosetta_reality.experiment import (  # noqa: E402
    file_sha256,
    workspace_code_identity,
)


def _argument_path(name: str) -> Path:
    try:
        index = sys.argv.index(name)
        value = Path(sys.argv[index + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise ValueError(f"Way export runtime repair requires {name}.") from error
    del sys.argv[index : index + 2]
    return value


def _argument_value(name: str) -> str:
    try:
        return str(sys.argv[sys.argv.index(name) + 1])
    except (ValueError, IndexError) as error:
        raise ValueError(f"Way export requires {name}.") from error


def _validated_record(record: Any, *, label: str) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"The {label} record must be a mapping.")
    path = formal_runner._repository_path(str(record.get("path", "")))
    if file_sha256(path) != record.get("sha256"):
        raise ValueError(f"The registered {label} changed.")
    return path


def _load_repair_plan(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any]]:
    repair = formal_runner._load_yaml(path)
    formal_path = _validated_record(repair.get("formal_plan"), label="formal plan")
    plan, _base_path, experiment = formal_runner._validate_plan(formal_path)
    selection_path = _validated_record(
        repair.get("selection_report"), label="selection report"
    )
    selection = formal_runner._load_json(selection_path)
    implementation = repair.get("implementation_files", {})
    if not isinstance(implementation, dict) or not implementation:
        raise ValueError("The Way export repair implementation set is empty.")
    for raw_path, expected_hash in implementation.items():
        implementation_path = formal_runner._repository_path(str(raw_path))
        if file_sha256(implementation_path) != expected_hash:
            raise ValueError(f"Way export repair implementation changed: {raw_path}.")
    selected = selection.get("selected", {})
    if (
        repair.get("schema_version") != 1
        or repair.get("role") != "vla"
        or repair.get("stage") != "smolvla_way_export_runtime_repair"
        or repair.get("status") != "preregistered"
        or repair.get("repair_id")
        != "m2-smolvla450m-way-cuda-b64-default-export-runtime-repair-001"
        or repair.get("formal_plan", {}).get("plan_id") != plan["plan_id"]
        or repair.get("formal_plan", {}).get("sha256") != file_sha256(formal_path)
        or repair.get("artifact_id") != _argument_value("--artifact-id")
        or repair.get("selected_checkpoint")
        != {
            "step": selected.get("step"),
            "model_safetensors_sha256": selected.get("model_safetensors_sha256"),
        }
        or repair.get("scope")
        != {
            "export_and_independent_reload_only": True,
            "optimizer_created": False,
            "gradients_enabled": False,
            "checkpoint_or_optimizer_state_modified": False,
            "training_identity_changed": False,
            "hidden_test_loaded": False,
        }
        or repair.get("compatibility_repairs")
        != {"legacy_normalization_alias_from_formal_prerequisites": True}
        or selection.get("status") != "passed"
        or selection.get("formal_plan_sha256") != file_sha256(formal_path)
        or selection.get("hidden_test_loaded") is not False
        or selected.get("step") not in plan["training"]["checkpoint_steps"]
    ):
        raise ValueError("The Way export runtime repair plan is invalid.")
    return repair, plan, formal_path, experiment


def _legacy_normalization_alias(plan: dict[str, Any]) -> dict[str, Any]:
    delegated = copy.deepcopy(plan)
    normalization = plan["prerequisites"]["normalization"]
    view = plan["prerequisites"]["dataset_view_manifest"]
    delegated["normalization"] = {
        "source_split": "train",
        "report": normalization["path"],
        "report_sha256": normalization["sha256"],
        "dataset_view_manifest": view["path"],
        "dataset_view_manifest_sha256": view["sha256"],
        "validation_episodes_loaded": False,
        "hidden_test_loaded": False,
    }
    return delegated


def main() -> int:
    repair_plan_path = _argument_path("--export-repair-plan")
    repair, formal_plan, formal_path, _experiment = _load_repair_plan(
        repair_plan_path
    )
    try:
        active_formal_path = Path(sys.argv[sys.argv.index("--plan") + 1]).resolve()
        active_selection = Path(
            sys.argv[sys.argv.index("--selection-report") + 1]
        ).resolve()
    except (ValueError, IndexError) as error:
        raise ValueError("Way export requires its formal plan and selection report.") from error
    registered_selection = formal_runner._repository_path(
        str(repair["selection_report"]["path"])
    )
    if active_formal_path != formal_path or active_selection != registered_selection:
        raise ValueError("The active Way export inputs differ from the repair plan.")

    original_validate_plan = formal_runner._validate_plan
    delegated_plan = _legacy_normalization_alias(formal_plan)

    def validate_plan_with_legacy_normalization(
        path: Path,
    ) -> tuple[dict[str, Any], Path, dict[str, Any]]:
        validated, base_path, experiment = original_validate_plan(path)
        if validated["plan_id"] != formal_plan["plan_id"]:
            raise ValueError("The delegated Way formal plan identity changed.")
        return copy.deepcopy(delegated_plan), base_path, experiment

    original_create_json = exporter.create_json

    def create_json(path: Path, payload: dict[str, Any]) -> None:
        if path.name in {"config.json", "manifest.json"}:
            payload["export_runtime"] = {
                "repair_id": repair["repair_id"],
                "repair_plan_sha256": file_sha256(repair_plan_path),
                "wrapper_sha256": file_sha256(Path(__file__)),
                "code_identity": workspace_code_identity(REPOSITORY_ROOT),
                "training_identity_changed": False,
            }
        original_create_json(path, payload)

    formal_runner._validate_plan = validate_plan_with_legacy_normalization
    exporter.create_json = create_json
    return way_exporter.main()


if __name__ == "__main__":
    raise SystemExit(main())
