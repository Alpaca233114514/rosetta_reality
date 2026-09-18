"""Bounded post-training preparation and sealed Gate rendering."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from scripts.canonical_fullframes_runtime import (
    PLAN,
    ROOT,
    active,
    artifact_path,
    load,
    source_job,
)
from scripts.iris_runtime import save, sha

EXPERIMENT = "m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003"


def prepare():
    from scripts.canonical_fullframes_endpoint import prepare as endpoint

    plan, job, _ = active()
    source = source_job(plan)
    alias = ROOT / "runs" / plan["source_run"]
    if not alias.exists() and not alias.is_symlink():
        alias.symlink_to(source, target_is_directory=True)
    if alias.resolve() != source.resolve():
        raise ValueError("Original training schedule alias differs")
    for name, expected in plan["source_evidence_files"].items():
        if sha(source / name) != expected:
            raise ValueError("Original training evidence drift: " + name)
    save(job / "checkpoint.json", load(source / "checkpoint.json"))
    durable = Path(os.environ["ROSETTA_AUTODL_ROOT"]) / "runs" / EXPERIMENT
    endpoint(
        argparse.Namespace(
            plan=PLAN,
            run_root=os.environ["ROSETTA_RUN_ROOT"],
            artifact_root=os.environ["ROSETTA_ARTIFACT_ROOT"],
            pretrained_dir=str(
                Path(os.environ["ROSETTA_CHECKPOINT_ROOT"])
                / EXPERIMENT
                / "smoke"
                / plan["source_run"]
                / "checkpoints/005000/pretrained_model"
            ),
            dataset_root=str(durable / "dataset_views/train-only-3e3c6b9d347e5e71"),
            normalization_report=str(durable / "normalization/train-only-3e3c6b9d347e5e71.json"),
            dataset_view_manifest=str(
                durable / "dataset_views/train-only-3e3c6b9d347e5e71/view_manifest.json"
            ),
            recovery_preflight=str(ROOT / plan["recovery_preflight_path"]),
        )
    )


def render():
    import yaml

    from scripts import smolvla_autodl_vfunfreeze_sim_gate as reference
    from scripts.canonical_fullframes_contract import validate_gate_protocol

    plan, job, reg = active()
    artifact = artifact_path(plan)
    outputs = job / "results" / EXPERIMENT
    selection = outputs / "selection/canonical-fullframes-20260914-001-step5000-selection.json"
    offline = job / "offline.json"
    report = load(offline)
    if (
        report["status"] != "complete"
        or report["plan_sha256"] != sha(PLAN)
        or report["artifact_manifest_sha256"] != sha(artifact / "manifest.json")
        or report["frame_offsets"] != plan["offline"]["frame_offsets"]
        or report["hidden_test_loaded"] is not False
    ):
        raise ValueError("Offline report differs from admitted endpoint")
    backup = job / "backup.json"
    save(
        backup,
        {
            "status": "verified",
            "off_host_copy_created": True,
            "receiver_receipt_sha256": plan["receiver_receipt_sha256"],
            "recovery_preflight_sha256": plan["recovery_preflight_sha256"],
        },
    )
    values = dict(
        sim_plan_id=plan["plan_id"] + "-gate",
        experiment_id=EXPERIMENT,
        artifact_id=artifact.name,
        artifact_manifest_sha256=sha(artifact / "manifest.json"),
        alignment_report="not_applicable",
        alignment_sha="0" * 64,
        prior_report=reference.PRIOR_FAILURE["report"],
        prior_sha=reference.PRIOR_FAILURE["report_sha256"],
        prior_task_report=reference.PRIOR_TASK_FAILURE["report"],
        prior_task_sha=reference.PRIOR_TASK_FAILURE["report_sha256"],
        selection_report=selection.relative_to(ROOT).as_posix(),
        selection_sha=sha(selection),
        selected_step=5000,
        model_sha=plan["endpoint"]["model_safetensors_sha256"],
        backup_sha=sha(backup),
        contract_sha=plan["gate_protocol_authority"]["action_contract_sha256"],
        suffix="471",
        sim_code_blocks="  scripts/canonical_fullframes_gate_runtime.py: "
        + sha(ROOT / "scripts/canonical_fullframes_gate_runtime.py"),
    )
    gate = yaml.safe_load(reference.SIM_PLAN_TEMPLATE.format(**values))
    gate.pop("entry_permit")  # This is the older candidate-specific alignment permit.
    gate["hypothesis"] = (
        "Evaluate the fixed canonical full-frame endpoint without threshold changes"
    )
    gate["artifact_backup"]["report"] = backup.relative_to(ROOT).as_posix()
    gate["inference"] = plan["executed_inference"]
    gate["resources"] = plan["executed_gate_resources"]
    for name in ("gate3", "gate4"):
        gate[name] = dict(plan[name])
        if name == "gate3":
            gate[name]["seed"] = gate[name].pop("environment_seed")
        gate[name]["report_suffix"] = "471"
    gate["simulation_code_sha256"] = reg["sources"]
    validate_gate_protocol(gate, plan)
    with (job / "gate.yaml").open("x") as stream:
        yaml.safe_dump(gate, stream, sort_keys=False)
    evidence = [
        selection,
        offline,
        backup,
        artifact / "manifest.json",
        job / "artifact-reload-proof.json",
    ]
    save(
        job / "gate-seal.json",
        {
            "plan_sha256": sha(job / "gate.yaml"),
            "evidence": {p.relative_to(ROOT).as_posix(): sha(p) for p in evidence},
        },
    )
    metadata = job / "artifact-metadata"
    metadata.mkdir()
    for path in artifact.iterdir():
        if path.is_file() and path.suffix == ".json":
            save(metadata / path.name, load(path))


def main():
    import sys

    {"prepare": prepare, "render": render}[sys.argv[1]]()


if __name__ == "__main__":
    main()
