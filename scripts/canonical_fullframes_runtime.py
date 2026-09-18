"""One saved-processor loader and actual artifact reload for offline/Gate use."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
from scripts.canonical_fullframes_contract import sha, verify_files  # noqa: E402
from scripts.iris_runtime import save  # noqa: E402

PLAN = (
    ROOT
    / "configs/vla/canonical_posttrain_gate_004.json"
)
NAME = "canonical-fullframes-posttrain-20260915-004"


def load(path):
    return json.loads(Path(path).read_text())


def job_path():
    return ROOT / "runs" / NAME


def active():
    import time

    from scripts.canonical_fullframes_posttrain import validate_plan
    from scripts.run_iris_furnace import alive

    plan = validate_plan(PLAN)
    job = job_path()
    reg = load(job / "registration.json")
    watchdog = load(job / "watchdog.json")
    if (
        reg["id"] != NAME
        or not reg["execution_authorized"]
        or not reg["started"] <= time.time() < reg["deadline"]
        or not alive(watchdog["pid"], watchdog["ticks"])
    ):
        raise ValueError("Active sealed supervisor/watchdog required")
    for name, expected in reg["sources"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Active source changed: " + name)
    return plan, job, reg


def source_job(plan):
    return (
        Path(os.environ["ROSETTA_AUTODL_ROOT"])
        / "workspaces"
        / plan["source_workspace"]
        / "runs"
        / plan["source_run"]
    )


def artifact_path(plan):
    return (
        Path(os.environ["ROSETTA_ARTIFACT_ROOT"])
        / "m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003"
        / plan["endpoint"]["artifact_id"]
    )


def load_context(artifact=None):
    from scripts.iris_gate import load_gate_context

    plan, _, reg = active()
    artifact = artifact_path(plan) if artifact is None else Path(artifact)
    verify_files(artifact / "pretrained_model", plan["endpoint"]["files"])
    source = source_job(plan)
    path = source / "smoke.yaml"
    if sha(path) != plan["source_training_plan_sha256"]:
        raise ValueError("Training plan identity changed")
    return load_gate_context(
        path,
        checkpoint=artifact / "pretrained_model",
        checkpoint_files=plan["endpoint"]["files"],
        split="train",
        deadline=reg["deadline"],
    )


def collect(label):
    from scripts import training_chain_gpu_audit as collector

    plan, job, reg = active()
    # Reuse the proven full-array collector with the new artifact layout/loader.
    collector.plan_path = lambda _job: source_job(plan) / "smoke.yaml"
    collector.load_native = lambda *args, **kwargs: load_context()
    collector.collect(job, label, reg["deadline"])


def verify_reload():
    from rosetta_reality.vla.reload_evidence import compare_bundles

    plan, job, _ = active()
    identity = load(job / "collect-first/manifest.json")["identity"]
    proof = compare_bundles(
        job / "collect-first", job / "collect-reload", expected_identity=identity
    )
    if proof["status"] != "passed":
        raise ValueError("New artifact independent reload differs")
    native = compare_bundles(
        source_job(plan) / "collect-first", job / "collect-first", expected_identity=identity
    )
    if native["status"] != "passed":
        raise ValueError("New artifact differs from recovered native inference")
    executions = [
        load(job / f"{label}-execution.json") for label in ("collect-first", "collect-reload")
    ]
    if executions[0]["pid"] == executions[1]["pid"] or any(
        not e["model_loaded"]
        or not e["saved_processors_loaded"]
        or e["forwards"] != 16
        or e["optimizer_steps"] != 0
        or not e["parameters_unchanged"]
        for e in executions
    ):
        raise ValueError("Real independent model execution evidence differs")
    save(
        job / "artifact-reload-proof.json",
        {"status": "passed", "proof": proof, "native_comparison": native, "executions": executions},
    )
    artifact = artifact_path(plan)
    manifest = load(artifact / "candidate-manifest.json")
    verify_files(artifact / "pretrained_model", plan["endpoint"]["files"])
    manifest.update(
        status="verified",
        reload={
            "verified": True,
            "exact_tensor_equality": True,
            "independent_processes": 2,
            "proof_sha256": sha(job / "artifact-reload-proof.json"),
        },
    )
    save(artifact / "manifest.json", manifest)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("collect-first", "collect-reload", "verify-reload"))
    args = p.parse_args()
    if args.mode == "verify-reload":
        verify_reload()
    else:
        collect(args.mode)


if __name__ == "__main__":
    main()
