"""Run the preregistered Way validation through an identity-bound compatibility layer."""

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
import evaluate_smolvla_way_validation as way_evaluator  # noqa: E402
import run_smolvla_phase as phase_runner  # noqa: E402
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
        raise ValueError(f"Way validation runtime repair requires {name}.") from error
    del sys.argv[index : index + 2]
    return value


def _validate_file_record(record: Any, *, label: str) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"The {label} repair-plan record must be a mapping.")
    path = formal_runner._repository_path(str(record.get("path", "")))
    if file_sha256(path) != record.get("sha256"):
        raise ValueError(f"The registered {label} file changed.")
    return path


def _load_repair_plan(
    path: Path,
) -> tuple[
    dict[str, Any], dict[str, Any], Path, dict[str, Any], dict[str, Any]
]:
    plan = formal_runner._load_yaml(path)
    formal_record = plan.get("formal_plan", {})
    formal_path = _validate_file_record(formal_record, label="formal plan")
    formal_plan, _base_path, experiment = formal_runner._validate_plan(formal_path)
    launch_path = _validate_file_record(plan.get("formal_launch"), label="formal launch")
    completion_path = _validate_file_record(
        plan.get("formal_completion"), label="formal completion"
    )
    failure_path = _validate_file_record(
        plan.get("failed_validation_attempt"), label="failed validation attempt"
    )
    launch = formal_runner._load_json(launch_path)
    completion = formal_runner._load_json(completion_path)
    implementation = plan.get("implementation_files", {})
    if not isinstance(implementation, dict) or not implementation:
        raise ValueError("The validation repair implementation set is empty.")
    for raw_path, expected_hash in implementation.items():
        implementation_path = formal_runner._repository_path(str(raw_path))
        if file_sha256(implementation_path) != expected_hash:
            raise ValueError(f"Validation repair implementation changed: {raw_path}.")
    if (
        plan.get("schema_version") != 1
        or plan.get("role") != "vla"
        or plan.get("stage") != "smolvla_way_validation_runtime_repair"
        or plan.get("status") != "preregistered"
        or plan.get("repair_id")
        != "m2-smolvla450m-way-cuda-b64-default-validation-runtime-repair-003"
        or formal_record.get("plan_id") != formal_plan["plan_id"]
        or formal_record.get("sha256") != file_sha256(formal_path)
        or plan.get("scope")
        != {
            "validation_only": True,
            "optimizer_created": False,
            "gradients_enabled": False,
            "checkpoint_or_optimizer_state_modified": False,
            "training_identity_changed": False,
            "hidden_test_loaded": False,
        }
        or plan.get("compatibility_repairs")
        != {
            "base_tokenizer_identity_from_pinned_vlm_dependency": True,
            "legacy_normalization_alias_from_formal_prerequisites": True,
            "validation_reports_preserve_formal_launch_code_identity": True,
            "validation_runtime_identity_recorded_separately": True,
        }
        or plan.get("validation_run_name_prefix")
        != formal_plan["validation"]["run_name_prefix"]
        or plan.get("base_tokenizer_source")
        != {
            "source": "pinned_vlm_dependency_snapshot",
            "repo_id": experiment["model"]["vlm_dependency"]["identifier"],
            "revision": experiment["model"]["vlm_dependency"]["revision"],
            "manifest_sha256": (
                "f6d10e2a3b8ba46baefcb8ec8bc5af1421a0738be476c0f71e1f66bc7a42d843"
            ),
            "cache_layout": (
                "hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/"
                "snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467"
            ),
        }
        or launch.get("status") != "preregistered"
        or launch.get("formal_plan_sha256") != file_sha256(formal_path)
        or completion.get("status") != "complete"
        or completion.get("formal_plan_sha256") != file_sha256(formal_path)
        or completion.get("run_name") != formal_plan["run_name"]
        or failure_path.stat().st_size <= 0
    ):
        raise ValueError("The Way validation runtime repair plan is invalid.")
    return plan, formal_plan, formal_path, experiment, launch


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


def _dependency_tokenizer_identity(
    source_dir: Path, experiment: dict[str, Any], expected_identity: dict[str, Any]
) -> tuple[dict[str, str], dict[str, Any]]:
    base_root = phase_runner._model_root(experiment).resolve()
    if source_dir.resolve() != base_root:
        raise FileNotFoundError("A non-base policy tokenizer is missing or empty.")
    dependency = experiment["model"]["vlm_dependency"]
    manifest_path = base_root / str(dependency["manifest"])
    manifest = formal_runner._load_json(manifest_path)
    hf_home = Path(os.environ.get("HF_HOME", "")).resolve()
    if not hf_home.is_absolute():
        raise ValueError("HF_HOME must identify the durable offline model cache.")
    snapshot = (hf_home / str(manifest.get("cache_layout", ""))).resolve()
    if not snapshot.is_relative_to(hf_home) or not snapshot.is_dir():
        raise ValueError("The pinned VLM dependency snapshot path is invalid.")
    recorded = manifest.get("files", {})
    if (
        manifest.get("status") != "validated"
        or manifest.get("repo_id") != dependency["identifier"]
        or manifest.get("revision") != dependency["revision"]
        or not isinstance(recorded, dict)
        or not recorded
    ):
        raise ValueError("The pinned VLM dependency manifest is invalid.")
    actual: dict[str, str] = {}
    for relative, record in sorted(recorded.items()):
        candidate = (snapshot / relative).resolve()
        if (
            not candidate.is_relative_to(snapshot)
            or not candidate.is_file()
            or not isinstance(record, dict)
            or file_sha256(candidate) != record.get("sha256")
            or candidate.stat().st_size != record.get("bytes")
        ):
            raise ValueError(f"Pinned VLM dependency file changed: {relative}.")
        if relative in {
            "added_tokens.json",
            "chat_template.json",
            "merges.txt",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        }:
            actual[relative] = str(record["sha256"])
    if "tokenizer.json" not in actual or "tokenizer_config.json" not in actual:
        raise ValueError("The pinned VLM dependency tokenizer set is incomplete.")
    identity = {
        "source": "pinned_vlm_dependency_snapshot",
        "repo_id": manifest["repo_id"],
        "revision": manifest["revision"],
        "manifest_sha256": file_sha256(manifest_path),
        "cache_layout": manifest["cache_layout"],
    }
    if identity != expected_identity:
        raise ValueError("The pinned VLM tokenizer identity differs from the repair plan.")
    return actual, identity


def main() -> int:
    repair_plan_path = _argument_path("--validation-repair-plan")
    repair_plan, formal_plan, formal_path, experiment, launch = _load_repair_plan(
        repair_plan_path
    )
    try:
        active_formal_path = Path(sys.argv[sys.argv.index("--plan") + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise ValueError("Way validation requires the registered formal plan.") from error
    if active_formal_path != formal_path:
        raise ValueError("The active formal plan differs from the validation repair plan.")

    original_validate_plan = formal_runner._validate_plan
    original_tokenizer_hashes = evaluator._tokenizer_hashes
    original_loader = evaluator._load_policy_and_dataset
    original_create_json = evaluator.create_json
    delegated_plan = _legacy_normalization_alias(formal_plan)
    dependency_identity: dict[str, Any] = {}

    def validate_plan_with_legacy_normalization(
        path: Path,
    ) -> tuple[dict[str, Any], Path, dict[str, Any]]:
        validated, base_path, active_experiment = original_validate_plan(path)
        if validated["plan_id"] != formal_plan["plan_id"]:
            raise ValueError("The delegated formal plan identity changed.")
        return copy.deepcopy(delegated_plan), base_path, active_experiment

    def tokenizer_hashes(source_dir: Path) -> dict[str, str]:
        nonlocal dependency_identity
        tokenizer = source_dir / "tokenizer"
        if tokenizer.is_dir():
            return original_tokenizer_hashes(source_dir)
        hashes, dependency_identity = _dependency_tokenizer_identity(
            source_dir, experiment, repair_plan["base_tokenizer_source"]
        )
        return hashes

    def load_policy_and_dataset(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        result = original_loader(*args, **kwargs)
        if args[3] is None:
            result[4]["tokenizer_source"] = copy.deepcopy(dependency_identity)
        return result

    def create_json(path: Path, payload: dict[str, Any]) -> None:
        payload["code_identity"] = copy.deepcopy(launch["code_identity"])
        payload["validation_runtime"] = {
            "repair_id": "m2-smolvla450m-way-cuda-b64-default-validation-runtime-repair-003",
            "repair_plan_sha256": file_sha256(repair_plan_path),
            "wrapper_sha256": file_sha256(Path(__file__)),
            "code_identity": workspace_code_identity(REPOSITORY_ROOT),
            "formal_launch_code_identity_preserved": True,
            "training_identity_changed": False,
        }
        original_create_json(path, payload)

    formal_runner._validate_plan = validate_plan_with_legacy_normalization
    evaluator._tokenizer_hashes = tokenizer_hashes
    evaluator._load_policy_and_dataset = load_policy_and_dataset
    evaluator.create_json = create_json
    return way_evaluator.main()


if __name__ == "__main__":
    raise SystemExit(main())
