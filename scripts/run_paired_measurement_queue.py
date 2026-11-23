"""Create-only fixed-endpoint A/B/C reload and independent policy measurements."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
INPUT = ROOT / "configs/diagnostics/paired-ABC-measurement-input-20260928-001.json"
ARMS = ("A", "B", "C")
SOURCES = (
    "scripts/run_paired_measurement_queue.py",
    "scripts/evaluate_paired_policy_v2.py",
    "scripts/diagnose_smolvla_base_five_scene.py",
    "scripts/smolvla_autodl_vfunfreeze_sim_gate.py",
    "scripts/smolvla_sim_gate.py",
    "scripts/restore_paired_branch_v2.py",
    "scripts/check_q0_state_restore.py",
    "src/rosetta_reality/eval/reproducibility_capture.py",
    "src/rosetta_reality/sim/scripted_teacher_observer.py",
    "src/rosetta_reality/sim/geometry_teacher.py",
    "src/rosetta_reality/sim/gym_aloha.py",
    "src/rosetta_reality/vla/action_space.py",
    "src/rosetta_reality/vla/processor.py",
    "src/rosetta_reality/sim/action_contract.py",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def checked(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Only root-relative registered inputs are allowed")
    resolved = (root / path).resolve(strict=True)
    if not resolved.is_relative_to(root.resolve(strict=True)):
        raise ValueError("Registered input escaped durable root")
    return resolved


def inputs() -> tuple[dict, Path]:
    plan = read(INPUT)
    if plan["id"] != "paired-ABC-measurement-input-20260928-001" or plan["arms"] != list(ARMS):
        raise ValueError("Unexpected fixed measurement identity")
    durable = Path(os.environ["ROSETTA_AUTODL_ROOT"]).resolve(strict=True)
    if plan["source_sha256"] != {name: digest(ROOT / name) for name in SOURCES}:
        raise ValueError("Measurement source identity changed")
    return plan, durable


def artifact(job: Path, arm: str) -> Path:
    return job / "artifacts" / arm


def checkpoint(durable: Path, plan: dict, arm: str) -> Path:
    return checked(durable, plan["checkpoint_pretrained"][arm])


def prepare(job: Path) -> None:
    plan, durable = inputs()
    if os.environ.get("ROSETTA_TORCH_DEVICE") != "cuda":
        raise ValueError("Registered CUDA worker required")
    source = checked(durable, plan["source_artifact"])
    training = checked(durable, plan["training_run"])
    bank = checked(durable, plan["bank_root"])
    if digest(source / "manifest.json") != plan["source_artifact_manifest_sha256"]:
        raise ValueError("Canonical interface source changed")
    if digest(source / "normalization.json") != plan["normalization_sha256"]:
        raise ValueError("Common D-only normalization changed")
    if digest(bank / "manifest.json") != plan["bank_manifest_sha256"] or digest(bank / "collection-audit.json") != plan["bank_audit_sha256"]:
        raise ValueError("Common recovery bank identity changed")
    bank_audit = read(bank / "collection-audit.json")
    if bank_audit.get("passed") is not True or bank_audit.get("manifest_sha256") != plan["bank_manifest_sha256"] or bank_audit.get("summary_sha256") != digest(bank / "summary.json"):
        raise ValueError("Recovery bank independent audit is incomplete")
    if read(training / "worker-exited.json")["error_type"] is not None:
        raise ValueError("Three-arm training did not exit cleanly")
    for arm in ARMS:
        completion = read(training / f"training-{arm}" / "completion.json")
        if completion.get("status") != "completed" or completion.get("successful_optimizer_updates") != 5000 or completion.get("arm") != arm or completion.get("normalization_sha256") != plan["normalization_sha256"]:
            raise ValueError(f"Arm {arm} completion identity differs")
        pretrained = checkpoint(durable, plan, arm)
        model_sha = digest(pretrained / "model.safetensors")
        if model_sha != plan["model_sha256"][arm]:
            raise ValueError(f"Arm {arm} fixed endpoint model changed")
        destination = artifact(job, arm)
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "pretrained_model").symlink_to(pretrained, target_is_directory=True)
        for name in ("normalization.json", "action_contract.json"):
            shutil.copyfile(source / name, destination / name)
        config = read(source / "config.json")
        config.update(
            artifact_id=f"paired-ABC-20260928-001-{arm.lower()}-step5000",
            selected_checkpoint_step=5000,
            selected_checkpoint_model_sha256=model_sha,
            formal_plan_sha256=completion["plan_sha256"],
            dataset_view_manifest_sha256=completion["view_manifest_sha256"],
            normalization_report_sha256=plan["normalization_sha256"],
        )
        selection = {"arm": arm, "selected_step": 5000, "model_sha256": model_sha,
                     "selection_rule": "preregistered_final_step_no_metric_search", "optimizer_updates": 0}
        save(job / f"selection-{arm}.json", selection)
        config["selection_report_sha256"] = digest(job / f"selection-{arm}.json")
        save(destination / "config.json", config)
        files = {f"pretrained_model/{p.relative_to(pretrained).as_posix()}": digest(p)
                 for p in sorted(pretrained.rglob("*")) if p.is_file()}
        files.update({name: digest(destination / name) for name in ("config.json", "normalization.json", "action_contract.json")})
        save(destination / "candidate-manifest.json", {
            "schema_version": 1, "status": "prepared", "artifact_id": config["artifact_id"],
            "selected_checkpoint_step": 5000, "selected_checkpoint_model_sha256": model_sha,
            "source_artifact_manifest_sha256": plan["source_artifact_manifest_sha256"],
            "training_completion_sha256": digest(training / f"training-{arm}" / "completion.json"),
            "hidden_test_loaded": False, "files": files,
            "reload": {"verified": False, "exact_tensor_equality": False},
        })
        save(job / f"probe-plan-{arm}.json", {"action_contract": {"path": "configs/sim/aloha_insertion_smolvla.yaml"}})
    save(job / "prepare-completion.json", {"status": "completed", "arms": list(ARMS),
         "bank_manifest_sha256": plan["bank_manifest_sha256"], "optimizer_updates": 0,
         "shutdown_requested": False})


def probe(job: Path, arm: str, number: int) -> None:
    if number not in (1, 2):
        raise ValueError("Two independent probes required")
    plan, durable = inputs()
    destination = artifact(job, arm)
    candidate = read(destination / "candidate-manifest.json")
    if digest(destination / "pretrained_model" / "model.safetensors") != plan["model_sha256"][arm] or candidate["selected_checkpoint_model_sha256"] != plan["model_sha256"][arm]:
        raise ValueError("Probe model identity changed")
    import torch
    from rosetta_reality.sim import GymAlohaEnvironment, load_action_contract
    import smolvla_autodl_vfunfreeze_sim_gate as reference
    from diagnose_smolvla_base_five_scene import parameter_digest

    reference._ACTIVE_SIM_PLAN = job / f"probe-plan-{arm}.json"
    contract = load_action_contract(ROOT / "configs/sim/aloha_insertion_smolvla.yaml")
    if digest(ROOT / "configs/sim/aloha_insertion_smolvla.yaml") != plan["action_contract_sha256"] or read(destination / "action_contract.json") != json.loads(json.dumps(asdict(contract))):
        raise ValueError("Probe Action Contract changed")
    torch.cuda.set_per_process_memory_fraction(.4)
    policy = reference._build_online_cuda_class()(destination, read(destination / "config.json"),
                                                  read(destination / "normalization.json"), contract)
    before = parameter_digest(policy.policy)
    env = GymAlohaEnvironment(contract, maximum_episode_steps=20)
    try:
        observation = env.reset(seed=20260809)
        policy.configure_noise("zeros", None)
        raw, projected = policy.predict(observation, "Insert the peg into the socket.")
    finally:
        env.close()
    after = parameter_digest(policy.policy)
    if before != after or not torch.isfinite(raw).all() or not torch.isfinite(projected).all():
        raise ValueError("Reload probe changed parameters or produced nonfinite actions")
    save(job / f"probe-{arm}-{number}.json", {"pid": os.getpid(), "arm": arm,
         "model_sha256": plan["model_sha256"][arm], "parameter_before": before,
         "parameter_after": after, "raw": raw.tolist(), "projected": projected.tolist(),
         "optimizer_updates": 0, "noise": "zeros", "scene_seed": 20260809})


def reload(job: Path, arm: str) -> int:
    try:
        for number in (1, 2):
            subprocess.run([sys.executable, __file__, "probe", "--job", str(job),
                            "--arm", arm, "--number", str(number)], cwd=ROOT, check=True)
        first, second = (read(job / f"probe-{arm}-{number}.json") for number in (1, 2))
        for key in ("arm", "model_sha256", "parameter_before", "parameter_after", "raw", "projected", "optimizer_updates", "noise", "scene_seed"):
            if first[key] != second[key]:
                raise ValueError(f"Independent {arm} full-array reload mismatch: {key}")
        if first["pid"] == second["pid"] or first["parameter_before"] != first["parameter_after"]:
            raise ValueError("Independent process/parameter condition failed")
        proof = {"status": "passed", "arm": arm, "independent_processes": 2,
                 "probe_1_sha256": digest(job / f"probe-{arm}-1.json"),
                 "probe_2_sha256": digest(job / f"probe-{arm}-2.json"),
                 "full_action_array_exact": True, "optimizer_updates": 0}
        save(job / f"reload-proof-{arm}.json", proof)
        destination = artifact(job, arm)
        candidate = read(destination / "candidate-manifest.json")
        candidate.update(status="verified", reload={"verified": True, "exact_tensor_equality": True,
                         "independent_processes": 2, "proof_sha256": digest(job / f"reload-proof-{arm}.json")})
        save(destination / "manifest.json", candidate)
        plan, durable = inputs()
        bank = checked(durable, plan["bank_root"])
        evaluation = {"schema_version": 1, "plan_id": f"paired-ABC-measurement-20260928-001-{arm}",
            "stage": "independent_paired_policy_measurement", "status": "preregistered",
            "arm": arm, "seeds": [1000, 1001, 1002, 1003, 1004], "maximum_steps": 500,
            "noise": "seeded_standard_normal", "instruction": "Insert the peg into the socket.",
            "optimizer_updates": 0, "hidden_test_loaded": False, "shutdown_requested": False,
            "action_contract": {"path": "configs/sim/aloha_insertion_smolvla.yaml"},
            "action_contract_sha256": plan["action_contract_sha256"],
            "artifact_manifest_sha256": digest(destination / "manifest.json"),
            "model_sha256": plan["model_sha256"][arm],
            "D_normalization_sha256": plan["normalization_sha256"],
            "bank_manifest_sha256": plan["bank_manifest_sha256"],
            "bank_audit_report": "collection-audit.json", "bank_audit_sha256": plan["bank_audit_sha256"],
            "resources": {"maximum_seconds": 1750, "rss_bytes": 8 * 1024**3,
                          "evidence_bytes": 256 * 1024**2, "cuda_allocated_bytes": 8 * 1024**3,
                          "cuda_reserved_bytes": 10 * 1024**3},
            "source_sha256": {name: digest(ROOT / name) for name in SOURCES}}
        if digest(bank / "manifest.json") != evaluation["bank_manifest_sha256"]:
            raise ValueError("Bank identity changed during reload")
        save(job / f"evaluation-plan-{arm}.json", evaluation)
        return 0
    except Exception as exc:
        save(job / f"reload-failure-{arm}.json", {"error_type": type(exc).__name__,
             "message": str(exc), "arm": arm, "shutdown_requested": False})
        return 2


def measure(job: Path, arm: str) -> int:
    plan, durable = inputs()
    evaluation = job / f"evaluation-plan-{arm}.json"
    if not evaluation.exists():
        save(job / f"measurement-refusal-{arm}.json", {"status": "reload_not_verified", "arm": arm})
        return 2
    command = [sys.executable, str(ROOT / "scripts/evaluate_paired_policy_v2.py"),
        "--plan", str(evaluation), "--artifact", str(artifact(job, arm)),
        "--bank", str(checked(durable, plan["bank_root"])),
        "--output", str(job / f"measurement-{arm}")]
    code = subprocess.run(command, cwd=ROOT, check=False).returncode
    save(job / f"measurement-exit-{arm}.json", {"arm": arm, "returncode": code,
         "summary_present": (job / f"measurement-{arm}" / "summary.json").is_file(),
         "shutdown_requested": False})
    return 0 if code == 0 else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "probe", "reload", "measure"))
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--number", type=int)
    args = parser.parse_args()
    job = args.job.resolve(strict=True)
    if args.mode == "prepare":
        prepare(job)
        return 0
    if args.arm is None:
        parser.error("probe/reload/measure require --arm")
    if args.mode == "probe":
        probe(job, args.arm, args.number)
        return 0
    if args.mode == "reload":
        return reload(job, args.arm)
    return measure(job, args.arm)


if __name__ == "__main__":
    raise SystemExit(main())
