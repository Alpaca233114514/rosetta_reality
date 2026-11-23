"""Create-only independent engineering checks; no provider qualification or retry."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time

import run_q0_screen_job as shared

STAGES = ("check_env", "synthetic", "evidence_audit", "state_restore", "refusal_boundaries")
SCHEMA = "rosetta.q0_independent_batch.v1"
PRIOR_JOB = "runs/q0-coupled-translation-20260927-001"
PRIOR_MANIFEST = "b9ef3a845877d68dc5c054a4edfc6052930140d2e72c11c04a984b208f5fad87"
OUTPUTS = {"evidence_audit": "evidence-audit.json", "state_restore": "state-restore.json",
           "refusal_boundaries": "refusal-boundaries.json"}
sha, save = shared.sha, shared.save


def safe_message(exc):
    """Keep useful failure context while removing absolute host references."""
    message = str(exc).replace("\n", " ").replace("\r", " ")
    message = re.sub(r"[A-Za-z]:[\\/][^\s\"'<>]+", "<absolute-path>", message)
    return re.sub(r"(?<![\w:])/(?:[^\s\"'<>]+)", "<absolute-path>", message)[:2000]


def validate(plan):
    if plan.get("schema") != SCHEMA or plan.get("status") != "prepared":
        raise ValueError("Requires a prepared independent batch plan")
    cap, target = plan["total_cap_seconds"], plan["platform_target_unix"]
    if type(cap) not in (int, float) or not math.isfinite(cap) or not 0 < cap <= 600:
        raise ValueError("Batch cap must be finite and at most 600 seconds")
    if type(target) not in (int, float) or not math.isfinite(target):
        raise ValueError("Platform deadline must be a finite launch-time timestamp")
    if plan.get("shutdown_authorized") is not True:
        raise ValueError("Independent guard requires shutdown authorization")
    if plan.get("prior_job_relative") != PRIOR_JOB or plan.get("prior_manifest_sha256") != PRIOR_MANIFEST:
        raise ValueError("Prior evidence identity mismatch")
    image = plan.get("runtime_container_image_id")
    if not isinstance(image, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]+", image):
        raise ValueError("Invalid runtime image identity")
    sources = plan["source_sha256"]
    required = {"scripts/run_q0_independent_batch.py", "scripts/run_q0_screen_job.py",
                "scripts/q0_bounded_guard.py", "scripts/autodl_bounded_guard.py",
                "scripts/run_autodl.sh", "scripts/check_env.py", shared.PROFILE,
                "scripts/audit_q0_screen_evidence.py", "scripts/check_q0_state_restore.py",
                "scripts/check_q0_refusal_boundaries.py", "tests/test_q0_independent_batch_stdlib.py"}
    if not isinstance(sources, dict) or not required <= sources.keys():
        raise ValueError("Incomplete source identity coverage")
    for raw, digest in sources.items():
        shared.relative_path(raw)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid source SHA256")
    commands = plan["commands"]
    if tuple(c["stage"] for c in commands) != STAGES:
        raise ValueError("Exactly five ordered independent phases are required")
    fixed = {
        "check_env": ["python", "scripts/check_env.py"],
        "evidence_audit": ["python", "scripts/audit_q0_screen_evidence.py", "--input", "{prior_job}",
                           "--output", "{job}/evidence-audit.json"],
        "state_restore": ["python", "scripts/check_q0_state_restore.py", "--output", "{job}/state-restore.json"],
        "refusal_boundaries": ["python", "scripts/check_q0_refusal_boundaries.py", "--output", "{job}/refusal-boundaries.json"],
    }
    total = 0
    for command in commands:
        args, seconds = command["args"], command["seconds"]
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 < seconds <= 120:
            raise ValueError("Every phase timeout must be at most 120 seconds")
        total += seconds
        if not isinstance(args, list) or any(not isinstance(a, str) for a in args):
            raise ValueError("Explicit argv required")
        if command["stage"] == "synthetic":
            if args[:4] != ["python", "-m", "pytest", "-q"] or not args[4:]:
                raise ValueError("Synthetic phase requires explicit pytest files")
            if "tests/test_q0_independent_batch_stdlib.py" not in args[4:]:
                raise ValueError("Batch safety tests are mandatory")
            for test in args[4:]:
                if not re.fullmatch(r"tests/test_[A-Za-z0-9_]+(?:_stdlib)?\.py", test) or test not in sources:
                    raise ValueError("Synthetic tests must be explicitly hashed files")
        elif args != fixed[command["stage"]]:
            raise ValueError("Phase argv is outside the fixed engineering whitelist")
    if total > cap:
        raise ValueError("Phase budgets exceed total work cap")


def prior_root(job, plan):
    # The new job and sealed old job must be siblings under durable runs/.
    if job.parent.name != "runs":
        raise ValueError("New job must be directly under durable runs")
    prior = job.parent.parent / shared.relative_path(plan["prior_job_relative"])
    if prior.resolve() == job.resolve():
        raise ValueError("Old evidence cannot be the new output job")
    return prior


def verify_prior(prior, plan):
    manifest_path = prior / "manifest.json"
    if manifest_path.is_symlink() or sha(manifest_path) != plan["prior_manifest_sha256"]:
        raise ValueError("Prior manifest SHA mismatch")
    manifest = json.loads(manifest_path.read_text())
    seen, dependencies, total = set(), set(), 0
    for entry in manifest["files"]:
        raw = entry["path"]
        path = shared.relative_path(raw)
        if raw in seen:
            raise ValueError("Duplicate prior manifest member: " + raw)
        seen.add(raw)
        member = prior / path
        resolved = member.resolve(strict=True)
        if member.is_symlink() or not resolved.is_relative_to(prior.resolve()) or not member.is_file():
            raise ValueError("Unsafe prior manifest member: " + raw)
        if member.stat().st_size != entry["bytes"] or sha(member) != entry["sha256"]:
            raise ValueError("Prior manifest member mismatch: " + raw)
        total += entry["bytes"]
        if path.parts[0] == "dependencies":
            dependencies.add(raw)
    if not dependencies:
        raise ValueError("Sealed dependency overlay is missing")
    actual = set()
    for member in (prior / "dependencies").rglob("*"):
        if member.is_symlink():
            raise ValueError("Symlink in prior dependency overlay")
        if member.is_file():
            actual.add(member.relative_to(prior).as_posix())
    if actual != dependencies:
        raise ValueError("Unsealed file or missing file in dependency overlay")
    if total != manifest["bytes"]:
        raise ValueError("Prior manifest byte total mismatch")
    return {"manifest_sha256": sha(manifest_path), "verified_files": len(seen),
            "verified_bytes": total, "dependency_files": len(dependencies), "old_run_modified": False}


def classify_result(job, stage, returncode):
    value = json.loads((job / OUTPUTS[stage]).read_text())
    if value.get("provider_qualified") is not False:
        raise ValueError("Engineering result must disclaim provider qualification")
    if value.get("infrastructure_failure") is True or value.get("safety_stop") is True:
        raise RuntimeError("Engineering check reported infrastructure failure or safety stop")
    if returncode < 0:
        raise RuntimeError("Engineering process terminated by signal")
    if returncode and value.get("status") not in ("negative", "failed", "check_failed", "unsupported", "mismatch"):
        raise RuntimeError("Nonzero phase lacks a recorded logical negative result")
    if not returncode and (value.get("passed") is False or value.get("status") in (
            "negative", "failed", "check_failed", "unsupported", "mismatch")):
        raise RuntimeError("Successful exit conflicts with recorded negative result")
    if not returncode and value.get("passed") is not True and value.get("status") != "passed":
        raise ValueError("Successful engineering check lacks an explicit passing result")
    return "negative" if returncode else "passed"


def worker(workspace, job, plan_path):
    if os.getpgrp() != os.getpid():
        raise RuntimeError("Worker must own its independent process group")
    def interrupted(_signal, _frame):
        raise TimeoutError("Independent guard interrupted worker")
    signal.signal(signal.SIGTERM, interrupted)
    phases, active, failed = [], "preparation", True
    try:
        reg = json.loads((job / "registration.json").read_text())
        deadline = time.monotonic() + max(0, reg["work_deadline_unix"] - time.time())
        if sha(plan_path) != reg["plan_sha256"]:
            raise ValueError("Plan identity drift")
        plan = json.loads(plan_path.read_text())
        validate(plan)
        shared.check_sources(workspace, plan)
        if sys.platform != "linux" or sys.version_info[:2] != (3, 12):
            raise RuntimeError("Sealed overlay requires Linux Python 3.12")
        prior = prior_root(job, plan)
        save(job / "prior-integrity.json", verify_prior(prior, plan))
        env = dict(os.environ, PYTHONPATH=f"{workspace / 'src'}:{workspace}:{prior / 'dependencies'}",
                   PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MUJOCO_GL="egl",
                   ROSETTA_RUN_ROOT=str(job / "unused-gate-runs"), ROSETTA_TORCH_DEVICE="cpu",
                   ROSETTA_CONTAINER_IMAGE_ID=plan["runtime_container_image_id"])
        save(job / "runtime-observation.json", {"python_version": sys.version.split()[0],
             "runtime_container_image_id": plan["runtime_container_image_id"],
             "image_digest_independently_verified": False, "nested_docker_used": False,
             "cgroup_memory_max": Path("/sys/fs/cgroup/memory.max").read_text().strip(),
             "mount_namespace_identity": os.readlink("/proc/self/ns/mnt"), "provider_qualified": False})
        for command in plan["commands"]:
            active, started = command["stage"], time.monotonic()
            phase = {"stage": active, "started_unix": time.time(), "args": command["args"], "provider_qualified": False}
            phases.append(phase)
            end = min(deadline, started + command["seconds"])
            shared.resources(job)
            if end <= started:
                raise TimeoutError("Work deadline reached")
            argv = [sys.executable] + [a.replace("{job}", str(job)).replace("{prior_job}", str(prior))
                                       for a in command["args"][1:]]
            with (job / (active + ".log")).open("x", encoding="utf-8") as log:
                child = subprocess.Popen(argv, cwd=workspace, env=env, stdin=subprocess.DEVNULL,
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                reader = threading.Thread(target=shared.drain_output, args=(child.stdout, log), daemon=True)
                reader.start()
                try:
                    while child.poll() is None:
                        shared.resources(job)
                        if time.monotonic() >= end:
                            raise TimeoutError("Phase timeout: " + active)
                        time.sleep(min(2, max(0, end - time.monotonic())))
                finally:
                    shared.stop_children()
                    child.wait(timeout=5)
                    reader.join(timeout=5)
                    if reader.is_alive():
                        raise RuntimeError("Child left a live output pipe")
            shared.resources(job)
            phase.update(exit_code=child.returncode, seconds=time.monotonic() - started)
            if active in OUTPUTS:
                phase["status"] = classify_result(job, active, child.returncode)
            elif child.returncode:
                raise RuntimeError("Environment or synthetic prerequisite failed")
            else:
                phase["status"] = "passed"
            save(job / (active + "-phase.json"), phase)
        failed = False
        save(job / "result.json", {"status": "engineering_batch_completed", "provider_qualified": False,
             "phases": phases, "logical_negatives": [p["stage"] for p in phases if p["status"] == "negative"],
             "simulation_step_cap": 6, "optimizer_updates": 0, "model_loaded": False,
             "gate_retries_executed": False, "g3_g4_qualification": False, "collection_started": False,
             "automatic_retry": False})
    except Exception as exc:
        save(job / "failure.json", {"phase": active, "error_type": type(exc).__name__,
             "message": safe_message(exc), "phases": phases, "provider_qualified": False,
             "automatic_retry": False})
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        shared.stop_children()
        entries = [{"path": p.relative_to(job).as_posix(), "sha256": sha(p), "bytes": p.stat().st_size}
                   for p in sorted(job.rglob("*")) if p.is_file() and p.name not in ("guard.log", "supervisor.log")]
        save(job / "manifest.json", {"files": entries, "bytes": sum(e["bytes"] for e in entries),
             "excluded_live_logs": ["guard.log", "supervisor.log"], "provider_qualified": False})
        save(job / "worker-exited.json", {"finished_unix": time.time(), "status": "failed" if failed else "completed",
             "manifest_sha256": sha(job / "manifest.json"), "provider_qualified": False})
        os.sync()


def _launch(workspace, job, plan_path):
    plan = json.loads(plan_path.read_text())
    validate(plan)
    shared.check_sources(workspace, plan)
    verify_prior(prior_root(job, plan), plan)
    from autodl_bounded_guard import WRAPPER_SHA256
    if sha(Path("/usr/bin/shutdown")) != WRAPPER_SHA256:
        raise ValueError("Shutdown wrapper mismatch")
    trash = Path("/root/.local/share/Trash")
    if trash.exists() or trash.is_symlink():
        raise ValueError("Trash exists; shutdown safety check failed")
    for argv, codes in ((["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], (0,)),
                        (["tmux", "list-sessions"], (0, 1))):
        result = subprocess.run(argv, capture_output=True, timeout=10, check=False)
        if result.returncode not in codes or result.stdout.strip():
            raise RuntimeError("Existing GPU work or tmux prevents launch")
    now = time.time()
    work = min(now + plan["total_cap_seconds"], plan["platform_target_unix"] - 300)
    if work <= now or shutil.disk_usage(job.parent).free < shared.RESERVE:
        raise ValueError("No bounded work window or disk reserve")
    save(job / "registration.json", {"started_unix": now, "work_deadline_unix": work,
         "shutdown_deadline_unix": work + 300, "shutdown_authorized": True,
         "platform_target_unix": plan["platform_target_unix"], "platform_target_independently_verified": False,
         "plan_sha256": sha(plan_path), "source_sha256": plan["source_sha256"],
         "runtime_profile_reference": shared.PROFILE, "runtime_profile_sha256": sha(workspace / shared.PROFILE),
         "runtime_profile_config": {"reference": shared.PROFILE, "sha256": sha(workspace / shared.PROFILE)},
         "nested_docker_used": False, "runtime_container_image_id": plan["runtime_container_image_id"],
         "rss_limit_bytes": shared.RSS_LIMIT, "evidence_limit_bytes": shared.EVIDENCE_LIMIT,
         "disk_reserve_bytes": shared.RESERVE, "provider_qualified": False, "automatic_retry": False})
    try:
        with (job / "guard.log").open("x") as log:
            guard = subprocess.Popen([sys.executable, "-u", str(workspace / "scripts/q0_bounded_guard.py"),
                                      "--job", str(job), "--retrieval-grace-seconds", "300"],
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        for _ in range(50):
            if (job / "guard-armed.json").exists():
                break
            if guard.poll() is not None:
                raise RuntimeError("Independent guard failed before arming")
            time.sleep(.1)
        armed = json.loads((job / "guard-armed.json").read_text())
        if armed["pid"] != guard.pid or armed["work_deadline_unix"] != work:
            raise RuntimeError("Independent guard registration mismatch")
        command = [sys.executable, "-u", "scripts/run_q0_independent_batch.py", "worker",
                   "--workspace", str(workspace), "--job", str(job), "--plan", str(plan_path)]
        with (job / "supervisor.log").open("x") as log:
            child = subprocess.Popen(["bash", "scripts/run_autodl.sh", "shell"], cwd=workspace,
                                     stdin=subprocess.PIPE, stdout=log, stderr=log, start_new_session=True)
            ticks = Path(f"/proc/{child.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
            save(job / "worker-pid.json", {"pid": child.pid, "start_ticks": ticks})
            child.stdin.write(("exec " + shlex.join(command) + "\n").encode())
            child.stdin.close()
    except Exception as exc:
        save(job / "launch-failure.json", {"error_type": type(exc).__name__, "message": safe_message(exc),
                                          "provider_qualified": False})
        raise
    print(json.dumps({"worker_pid": child.pid, "guard_pid": guard.pid, "work_deadline_unix": work,
                      "shutdown_deadline_unix": work + 300, "provider_qualified": False}))


def launch(workspace, job, plan_path):
    # Create only the new output identity, so prerequisite failures retain details too.
    # An existing job fails before any write, preserving historical output.
    job.mkdir()
    try:
        _launch(workspace, job, plan_path)
    except Exception as exc:
        save(job / "preparation-failure.json", {"phase": "preparation", "error_type": type(exc).__name__,
             "message": safe_message(exc), "provider_qualified": False, "automatic_retry": False})
        if not (job / "worker-pid.json").exists():
            entries = [{"path": p.relative_to(job).as_posix(), "sha256": sha(p), "bytes": p.stat().st_size}
                       for p in sorted(job.rglob("*")) if p.is_file() and p.name not in ("guard.log", "supervisor.log")]
            save(job / "manifest.json", {"files": entries, "bytes": sum(e["bytes"] for e in entries),
                 "excluded_live_logs": ["guard.log", "supervisor.log"], "provider_qualified": False})
            save(job / "worker-exited.json", {"finished_unix": time.time(), "status": "preparation_failed",
                 "manifest_sha256": sha(job / "manifest.json"), "provider_qualified": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "worker"))
    for key in ("workspace", "job", "plan"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    globals()[args.mode](args.workspace.resolve(strict=True), args.job.resolve(), args.plan.resolve(strict=True))
