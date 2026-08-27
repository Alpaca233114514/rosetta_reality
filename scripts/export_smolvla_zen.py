"""Export a selected Zen checkpoint as a gate-ready deploy artifact.

The artifact layout matches what the frozen simulation-gate engine loads
(``smolvla_sim_gate._load_artifact``): ``pretrained_model/``, ``config.json``
(dataset features/fps, rename map, action-space identity, mixed precision),
``normalization.json`` (train-only statistics), ``action_contract.json`` and a
``manifest.json`` whose ``reload.exact_tensor_equality`` is proven by running
the gate-identical inference path (``make_policy`` + processor pipeline +
``predict_action_chunk`` with zero noise) in two independent fresh processes
and comparing action digests.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for root in (str(REPOSITORY_ROOT / "src"), SCRIPTS_ROOT):
    if root not in sys.path:
        sys.path.insert(0, root)

import smolvla_zen_protocol as protocol  # noqa: E402

from rosetta_reality.experiment import file_sha256, workspace_code_identity  # noqa: E402
from rosetta_reality.features import create_json  # noqa: E402

DETERMINISTIC_METRICS = (
    "action_mae",
    "action_rmse",
    "first_action_mae",
    "fixed_flow_loss",
    "invalid_action_rate",
    "joint_limit_violation_rate",
    "action_smoothness_mean_abs_delta",
)


def _reload_validation(
    plan_path: Path,
    preflight_report: Path,
    artifact_pretrained: Path,
    step: int,
    prefix_override: str,
) -> Path:
    """Re-run the fixed-validation engine against the exported artifact copy."""

    import os

    environment = {
        **os.environ,
        "ROSETTA_ZEN_RELOAD_SOURCE": str(artifact_pretrained),
        "ROSETTA_ZEN_VALIDATION_PREFIX_OVERRIDE": prefix_override,
    }
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/smolvla_zen_validate.py",
            "--plan",
            str(plan_path),
            "--preflight-report",
            str(preflight_report),
            "--checkpoint-step",
            str(step),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPOSITORY_ROOT,
        env=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Artifact reload validation failed: {completed.stderr[-2000:]}")
    prefix = prefix_override
    return (
        Path(os.environ["ROSETTA_RUN_ROOT"]).resolve()
        / protocol.EXPERIMENT_ID
        / "validation"
        / f"{prefix}-step-{step:06d}.json"
    )


def _exact_reload(
    checkpoint_report: Path, artifact_report: Path
) -> tuple[bool, dict[str, float]]:
    reference = json.loads(checkpoint_report.read_text(encoding="utf-8"))["metrics"]
    reloaded = json.loads(artifact_report.read_text(encoding="utf-8"))["metrics"]
    max_difference = 0.0
    exact = True
    for name in DETERMINISTIC_METRICS:
        difference = abs(float(reference[name]) - float(reloaded[name]))
        max_difference = max(max_difference, difference)
        if difference != 0.0:
            exact = False
    return exact, {
        "maximum_absolute_metric_difference": max_difference,
        "compared_metrics": list(DETERMINISTIC_METRICS),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()

    from rosetta_reality.sim import load_action_contract
    from rosetta_reality.vla.action_space import load_smolvla_experiment

    plan_path = args.plan.resolve()
    plan, plan_id = protocol.resolve_plan(plan_path)
    spec = protocol.ZEN_SPECS[plan_id]
    selection = json.loads(args.selection_report.resolve().read_text(encoding="utf-8"))
    if selection.get("plan_sha256") != file_sha256(plan_path):
        raise ValueError("Selection report belongs to a different Zen plan.")
    step = int(selection["selected_checkpoint_step"])
    if step not in protocol.CHECKPOINT_STEPS:
        raise ValueError("Selected step outside the registered grid.")

    run_root = args.run_root.resolve()
    experiment_path = REPOSITORY_ROOT / protocol.PARENT_CONFIG
    experiment = load_smolvla_experiment(experiment_path, REPOSITORY_ROOT)
    contract_path = REPOSITORY_ROOT / protocol.ACTION_CONTRACT_RELATIVE
    contract = load_action_contract(contract_path)
    contract_sha = file_sha256(contract_path)

    view_info_path = (
        run_root
        / protocol.VIEW_MANIFEST_UNDER_RUNROOT
    ).parent / "meta" / "info.json"
    if not view_info_path.is_file():
        raise FileNotFoundError("Train-only dataset view info.json is missing.")
    view_info = json.loads(view_info_path.read_text(encoding="utf-8"))
    dataset_features = view_info["features"]
    dataset_features = {
        key: value for key, value in dataset_features.items()
        if key in {"observation.images.top", "observation.state", "action"}
    }

    source_dir = (
        args.checkpoint_root.resolve()
        / protocol.EXPERIMENT_ID
        / "formal"
        / spec["run_name"]
        / "checkpoints"
        / f"{step:06d}"
        / "pretrained_model"
    )
    if not (source_dir / "model.safetensors").is_file():
        raise FileNotFoundError(f"Selected checkpoint is missing: {source_dir}")

    artifact_id = f"{spec['run_name']}-step{step:04d}-deploy-001"
    artifact_dir = args.artifact_root.resolve() / protocol.EXPERIMENT_ID / artifact_id
    pretrained_dir = artifact_dir / "pretrained_model"
    superseded_dir = None
    copied_fresh = False
    if artifact_dir.is_dir():
        if (artifact_dir / "manifest.json").is_file():
            raise FileExistsError(f"Artifact already finalized: {artifact_dir}")
        if not (pretrained_dir / "model.safetensors").is_file():
            # Legacy interrupted layout (flat files, no pretrained_model/):
            # rename it aside reversibly instead of deleting anything.
            import time

            superseded_dir = artifact_dir.with_name(
                f"{artifact_dir.name}-superseded-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
            )
            artifact_dir.rename(superseded_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if not (pretrained_dir / "model.safetensors").is_file():
        shutil.copytree(source_dir, pretrained_dir)
        copied_fresh = True

    (artifact_dir / "normalization.json").write_text(
        (run_root / protocol.NORMALIZATION_REPORT_UNDER_RUNROOT).read_text("utf-8"),
        encoding="utf-8",
    )
    artifact_config = {
        "artifact_id": artifact_id,
        "experiment_id": protocol.EXPERIMENT_ID,
        "mixed_precision": "bf16",
        "rename_map": experiment["dataset"]["rename_map"],
        "action_space": {
            **experiment["model"]["action_space"],
            "adapt_to_pi_aloha": bool(
                experiment["model"]["policy"].get("adapt_to_pi_aloha", False)
            ),
        },
        "bounded_gripper_decoder": True,
        "action_contract_sha256": contract_sha,
        "upstream_revision": experiment["upstream"]["revision"],
        "dataset_features": dataset_features,
        "dataset_fps": int(view_info["fps"]),
        "selected_checkpoint_step": step,
        "hidden_test_loaded": False,
    }
    (artifact_dir / "config.json").write_text(
        json.dumps(artifact_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (artifact_dir / "action_contract.json").write_text(
        json.dumps(asdict(contract), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    preflight_report = (
        run_root
        / protocol.EXPERIMENT_ID
        / "preflight"
        / ("m2-smolvla450m-zen-uniform-preflight-001.json"
        if plan_id.endswith("002")
        else "m2-smolvla450m-zen-firstaction-preflight-001.json")
    )
    checkpoint_report = (
        run_root
        / protocol.EXPERIMENT_ID
        / "validation"
        / f"{spec['validation_prefix']}-step-{step:06d}.json"
    )
    if not checkpoint_report.is_file():
        raise FileNotFoundError(f"Checkpoint validation report missing: {checkpoint_report.name}")
    reload_prefix = f"{spec['validation_prefix']}-reload"
    validation_dir = run_root / protocol.EXPERIMENT_ID / "validation"
    generation = 1
    while (validation_dir / f"{reload_prefix}-step-{step:06d}.json").is_file():
        generation += 1
        reload_prefix = f"{spec['validation_prefix']}-reload{generation}"
    artifact_report = _reload_validation(
        plan_path, preflight_report, pretrained_dir, step, reload_prefix
    )
    reload_exact, reload_detail = _exact_reload(checkpoint_report, artifact_report)

    files = {
        path.relative_to(artifact_dir).as_posix(): file_sha256(path)
        for path in sorted(artifact_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "schema_version": 1,
        "status": "verified" if reload_exact else "reload_mismatch",
        "stage": "smolvla_zen_export",
        "artifact_id": artifact_id,
        "experiment_id": protocol.EXPERIMENT_ID,
        "plan_id": plan_id,
        "plan_sha256": file_sha256(plan_path),
        "selection_report_sha256": file_sha256(args.selection_report.resolve()),
        "selected_checkpoint_step": step,
        "selected_checkpoint_model_sha256": file_sha256(
            artifact_dir / "pretrained_model" / "model.safetensors"
        ),
        "reload": {
            "verified": bool(reload_exact),
            "exact_tensor_equality": bool(reload_exact),
            "method": "fixed_validation_engine_rerun_on_artifact",
            "checkpoint_validation_report": checkpoint_report.name,
            "artifact_reload_report": artifact_report.name,
            "artifact_reload_report_sha256": file_sha256(artifact_report),
            **reload_detail,
        },
        "pretrained_model_copied_this_run": bool(copied_fresh),
        "superseded_legacy_dir": (
            superseded_dir.name if superseded_dir is not None else None
        ),
        "files": files,
        "hidden_test_loaded": False,
        "zen_protocol": {
            "wrapper_sha256": file_sha256(Path(__file__)),
            "protocol_module_sha256": file_sha256(
                REPOSITORY_ROOT / "scripts/smolvla_zen_protocol.py"
            ),
            "code_identity": workspace_code_identity(REPOSITORY_ROOT),
        },
    }
    create_json(artifact_dir / "manifest.json", manifest)
    summary = {
        "artifact": str(artifact_dir),
        "id": artifact_id,
        "reload_exact": reload_exact,
        "maximum_absolute_metric_difference": reload_detail[
            "maximum_absolute_metric_difference"
        ],
    }
    print(json.dumps(summary))
    return 0 if reload_exact else 4


if __name__ == "__main__":
    raise SystemExit(main())
