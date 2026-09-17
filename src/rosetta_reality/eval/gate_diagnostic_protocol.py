"""Seed 3 registration validation; importing this module never imports ML libraries."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

from .gate_diagnostic_io import load_json, relative, sha

AUTHORITY = (
    "reports/training/m2-smolvla-canonical-fullframes-posttrain-gate-plan-004-2026-09-15.json"
)
AUTHORITY_SHA = "35511af8c7d0d9d93c827fe72ddec0205fa8932f7660735ddd38e8b7524be75e"
WEIGHT_SHA = "d4c0d87cd8e66c723b07ec84f7f5e47875268bfd4e2057ec5683fdf56adcabef"
CONFIG_SHA = "978ec7b1b9b648749b96cf0967ae4b37e0133d65e491613e6803884d50c0984a"
# Actual call-path identities recorded in canonical-posttrain-template-20260916-005.json.
# A new diagnostic registration cannot silently reseal a modified historical Gate engine.
REFERENCE_IMPLEMENTATIONS = {
    "scripts/smolvla_sim_gate.py": (
        "5b76127a2e2d0e0049181a1d0ab122974" "74cbc8eb433c4fcdb6466c35c53c5ae"
    ),
    "scripts/smolvla_autodl_vfunfreeze_sim_gate.py": (
        "f91ae1f5a61c2eb365bbb673caf865719" "54426b1e03fb27ffeb38bd1280cf777"
    ),
    "src/rosetta_reality/sim/gym_aloha.py": (
        "e9c1005d0ae085e82e0c96e9d18527dce" "7d4749268a71756116cdffbb98d6e7d"
    ),
    "src/rosetta_reality/vla/processor.py": (
        "6751d4dd901da27e0a299bd9426fa4845" "40e85dc12f1f1a62694e063d07e2384"
    ),
}
REQUIRED_SOURCES = {
    "src/rosetta_reality/eval/__init__.py",
    "scripts/diagnose_smolvla_gate.py",
    "scripts/smolvla_sim_gate.py",
    "scripts/smolvla_autodl_vfunfreeze_sim_gate.py",
    "scripts/iris_gate.py",
    "scripts/iris_runtime.py",
    "scripts/evaluate_visual_native_small.py",
    "scripts/canonical_fullframes_contract.py",
    "scripts/run_iris_furnace.py",
    "scripts/diagnose_zen_noise_transfer.py",
    "src/rosetta_reality/eval/rollout_trace.py",
    "src/rosetta_reality/eval/rollout_trace_verify.py",
    "src/rosetta_reality/vla/processor.py",
    "src/rosetta_reality/vla/image_scaling.py",
    "src/rosetta_reality/sim/gym_aloha.py",
    "src/rosetta_reality/sim/action_contract.py",
} | {
    f"src/rosetta_reality/eval/gate_diagnostic_{part}.py"
    for part in ("io", "protocol", "capture", "replay", "verify", "training", "report", "runtime")
}


def reference(root, spec):
    path = relative(root, spec["path"])
    if not re.fullmatch("[0-9a-f]{64}", spec.get("sha256", "")) or sha(path) != spec["sha256"]:
        raise ValueError("Registered evidence identity differs: " + spec["path"])
    return path


def validate_plan(plan, root, *, check_files=True):
    if (
        plan.get("schema_version") != 1
        or plan.get("diagnostic_only") is not True
        or plan.get("optimizer_steps") != 0
        or plan.get("hidden_test_loaded") is not False
        or plan.get("checkpoint_step") != 5000
        or not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,100}", plan.get("run_id", ""))
    ):
        raise ValueError("Explicit canonical step-5000 diagnostic scope required")
    if plan.get("rollout") != {
        "seed": 3,
        "policy_noise_seed": 3,
        "maximum_steps": 500,
        "noise_mode": "seeded_standard_normal",
        "project_policy_output": True,
        "frequency_hz": 50,
        "chunk_execution_steps": 1,
    }:
        raise ValueError("Fixed Seed 3 first-action protocol required")
    maximum = plan.get("maximum_evidence_bytes")
    if type(maximum) is not int or not 64 * 1024**2 <= maximum <= 4 * 1024**3:
        raise ValueError("Evidence budget must be 64 MiB through 4 GiB")
    output = relative(root, plan["output"])
    if not output.is_relative_to(Path(root) / "runs"):
        raise ValueError("Diagnostic output must be below runs/")
    if plan.get("status") not in ("draft", "registered"):
        raise ValueError("Unknown registration status")
    if not check_files:
        return {
            "status": "schema_valid",
            "execution_ready": False,
            "file_identities_checked": False,
        }
    if plan["status"] != "registered":
        raise ValueError("Draft is not executable; create a new sealed registration")
    if not REQUIRED_SOURCES <= plan["sources"].keys():
        raise ValueError("Incomplete diagnostic dependency inventory")
    for name, digest in plan["sources"].items():
        reference(root, {"path": name, "sha256": digest})
    if any(
        plan["sources"].get(name) != digest for name, digest in REFERENCE_IMPLEMENTATIONS.items()
    ):
        raise ValueError("Historical Gate call path changed; cannot reuse its acceptance")
    authority_path = reference(root, {"path": AUTHORITY, "sha256": AUTHORITY_SHA})
    authority = load_json(authority_path)
    if (
        plan["checkpoint"]["files"] != authority["endpoint"]["files"]
        or plan["training_plan"]["sha256"] != authority["source_training_plan_sha256"]
        or plan["action_contract"]["sha256"]
        != authority["gate_protocol_authority"]["action_contract_sha256"]
        or plan["artifact_config"]["sha256"] != CONFIG_SHA
    ):
        raise ValueError("Fixed canonical endpoint differs")
    checkpoint = relative(root, plan["checkpoint"]["path"])
    actual = {p.relative_to(checkpoint).as_posix() for p in checkpoint.rglob("*") if p.is_file()}
    if actual != set(plan["checkpoint"]["files"]):
        raise ValueError("Checkpoint file inventory differs")
    for name, digest in plan["checkpoint"]["files"].items():
        reference(checkpoint, {"path": name, "sha256": digest})
    for name in ("training_plan", "action_contract", "artifact_config"):
        reference(root, plan[name])
    manifest = load_json(reference(root, plan["artifact_manifest"]))
    gate = load_json(reference(root, plan["gate3_report"]))
    if (
        manifest.get("status") != "verified"
        or manifest.get("selected_checkpoint_model_sha256") != WEIGHT_SHA
        or manifest.get("reload", {}).get("exact_tensor_equality") is not True
        or manifest.get("hidden_test_loaded") is not False
        or any(
            manifest.get("files", {}).get("pretrained_model/" + name) != digest
            for name, digest in plan["checkpoint"]["files"].items()
        )
        or manifest.get("files", {}).get("config.json") != CONFIG_SHA
    ):
        raise ValueError("Verified canonical artifact/reload required")
    if (
        gate.get("status") != "passed"
        or gate.get("gate") != "m2_gate_3_small_policy_rollout"
        or gate.get("artifact_manifest_sha256") != plan["artifact_manifest"]["sha256"]
        or gate.get("artifact_id") != manifest.get("artifact_id")
        or gate.get("artifact_reload_verified") is not True
        or gate.get("hidden_test_loaded") is not False
        or not gate.get("acceptance_criteria")
        or not all(value is True for value in gate["acceptance_criteria"].values())
    ):
        raise ValueError("Matching passed Gate 3 prerequisite required")
    return {"status": "identity_valid", "execution_ready": False, "file_identities_checked": True}


def authorize(plan, root, stage):
    """Check permission/window before any model import or construction."""
    if stage not in ("collect", "replay", "probe"):
        raise ValueError("Unknown model stage")
    validate_plan(plan, root)
    execution = plan.get("execution") or {}
    if (
        execution.get("authorized") is not True
        or stage not in execution.get("stages", [])
        or not time.time() < execution.get("deadline", 0) <= time.time() + 3600
        or execution.get("runtime") != "autodl_cuda_container_instance"
        or execution.get("nested_docker_used") is not False
    ):
        raise ValueError("New stage-authorized bounded CUDA window required")
    if any(
        os.environ.get(k) != v
        for k, v in {
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "ROSETTA_TORCH_DEVICE": "cuda",
        }.items()
    ):
        raise ValueError("Registered offline CUDA runtime required")
    if not execution.get("gpu_uuid") or not execution.get("package_versions"):
        raise ValueError("Pinned GPU and package identities required")
    from importlib.metadata import version

    for package in ("torch", "lerobot", "gym-aloha", "mujoco"):
        if version(package) != execution["package_versions"].get(package):
            raise ValueError("Runtime package identity differs: " + package)
    from scripts.run_iris_furnace import alive

    watchdog = execution["watchdog"]
    if not alive(watchdog["pid"], watchdog["ticks"]):
        raise ValueError("Live registered shutdown watchdog required")
    return execution
