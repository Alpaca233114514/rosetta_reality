"""Create-only bounded teacher diagnostic; no model, optimizer or platform shutdown."""
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
import run_q0_independent_batch as prior_batch
import work_only_guard

SCHEMA = "rosetta.teacher_cross_pose_execution.v1"
TEST_FILES = ("tests/test_teacher_diagnostic_job_stdlib.py", "tests/test_teacher_cross_pose_diagnostic.py",
              "tests/test_q0_bounded_guard_stdlib.py")
REQUIRED_SOURCES = {"scripts/run_teacher_diagnostic_job.py", "scripts/work_only_guard.py",
    "scripts/run_q0_screen_job.py", "scripts/run_q0_independent_batch.py", "scripts/q0_bounded_guard.py",
    "scripts/autodl_bounded_guard.py", "scripts/run_autodl.sh", "scripts/diagnose_teacher_cross_pose.py",
    shared.PROFILE, *TEST_FILES}
DIAGNOSTIC_ARGS = ["python", "scripts/diagnose_teacher_cross_pose.py", "--output", "{job}/cross-pose", "--max-seconds", "600"]
sha, save = shared.sha, shared.save
prior_root, verify_prior, safe_message = prior_batch.prior_root, prior_batch.verify_prior, prior_batch.safe_message


def validate(plan):
    if plan.get("schema") != SCHEMA or plan.get("status") != "prepared":
        raise ValueError("Prepared teacher diagnostic plan required")
    if not isinstance(plan.get("id"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", plan["id"]):
        raise ValueError("Invalid job identity")
    cap, target = plan["total_cap_seconds"], plan["platform_target_unix"]
    if type(cap) not in (int, float) or not math.isfinite(cap) or not 0 < cap <= 780:
        raise ValueError("Finite total cap <=780 required")
    if target is not None and (type(target) not in (int, float) or not math.isfinite(target)):
        raise ValueError("Platform target must be null or a finite confirmed deadline")
    if plan.get("shutdown_authorized") is not False:
        raise ValueError("This job has no platform shutdown authority")
    if plan.get("prior_job_relative") != prior_batch.PRIOR_JOB or plan.get("prior_manifest_sha256") != prior_batch.PRIOR_MANIFEST:
        raise ValueError("Prior sealed evidence identity mismatch")
    if not isinstance(plan.get("runtime_container_image_id"), str) or not re.fullmatch(r"[A-Za-z0-9_.:-]+", plan["runtime_container_image_id"]):
        raise ValueError("Invalid runtime image identity")
    sources = plan["source_sha256"]
    if not isinstance(sources, dict) or not REQUIRED_SOURCES <= sources.keys():
        raise ValueError("Incomplete source identity coverage")
    for raw, digest in sources.items():
        shared.relative_path(raw)
        if ":" in raw or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid relative source or SHA256")
    commands = plan["commands"]
    if not isinstance(commands, list) or len(commands) != 2 or [c["stage"] for c in commands] != ["tests", "diagnostic"]:
        raise ValueError("Exactly tests then diagnostic required")
    if commands[0]["args"] != ["python", "-m", "pytest", "-q", *TEST_FILES] or commands[1]["args"] != DIAGNOSTIC_ARGS:
        raise ValueError("Only fixed hashed test and diagnostic argv permitted")
    for command, maximum in zip(commands, (90, 600)):
        seconds = command["seconds"]
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 < seconds <= maximum:
            raise ValueError("Invalid phase budget")
    if sum(c["seconds"] for c in commands) > cap:
        raise ValueError("Phase budgets exceed cap")


def work_deadline(plan, now):
    end = now + plan["total_cap_seconds"]
    if plan["platform_target_unix"] is not None:
        end = min(end, plan["platform_target_unix"] - 300)
    if end - now < 180:
        raise ValueError("Less than 180 seconds available")
    return end


def guard_alive(job, reg):
    armed = json.loads((job / "guard-armed.json").read_text())
    if (armed.get("id") != reg["id"] or armed.get("plan_sha256") != reg["plan_sha256"]
            or armed.get("work_deadline_unix") != reg["work_deadline_unix"]
            or armed.get("shutdown_authorized") is not False
            or work_only_guard.cleanup.original.process_identity(armed["pid"]) != (str(armed["start_ticks"]), armed["pid"])
            or (job / "guard-exited.json").exists()):
        raise RuntimeError("Independent work guard identity lost")


def seal(job, status):
    excluded = {"manifest.json", "worker-exited.json", "guard.log", "supervisor.log"}
    entries = [{"path": p.relative_to(job).as_posix(), "sha256": sha(p), "bytes": p.stat().st_size}
               for p in sorted(job.rglob("*")) if p.is_file() and p.name not in excluded]
    save(job / "manifest.json", {"files": entries, "bytes": sum(e["bytes"] for e in entries),
         "excluded_live_logs": ["guard.log", "supervisor.log"], "provider_qualified": False})
    save(job / "worker-exited.json", {"finished_unix": time.time(), "status": status,
         "manifest_sha256": sha(job / "manifest.json"), "provider_qualified": False, "shutdown_requested": False})


def worker(workspace, job, plan_path):
    if os.getpgrp() != os.getpid():
        raise RuntimeError("Worker must own an independent process group")
    def interrupted(_signum, _frame):
        raise TimeoutError("Work guard interrupted worker")
    signal.signal(signal.SIGTERM, interrupted)
    phases, active, failed = [], "preparation", True
    try:
        reg = json.loads((job / "registration.json").read_text())
        work_only_guard.validate_registration(reg)
        if job.name != reg["id"] or sha(plan_path) != reg["plan_sha256"]:
            raise ValueError("Plan or job identity drift")
        deadline = time.monotonic() + max(0, reg["work_deadline_unix"] - time.time())
        plan = json.loads(plan_path.read_text())
        validate(plan)
        if plan["id"] != reg["id"]:
            raise ValueError("Plan identity drift")
        shared.check_sources(workspace, plan)
        guard_alive(job, reg)
        if sys.platform != "linux" or sys.version_info[:2] != (3, 12):
            raise RuntimeError("Sealed overlay requires Linux Python 3.12")
        prior = prior_root(job, plan)
        save(job / "prior-integrity.json", verify_prior(prior, plan))
        env = dict(os.environ, PYTHONPATH=f"{workspace / 'src'}:{workspace}:{prior / 'dependencies'}",
            PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
            OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MUJOCO_GL="egl",
            ROSETTA_RUN_ROOT=str(job / "unused-gate-runs"), ROSETTA_TORCH_DEVICE="cpu",
            ROSETTA_CONTAINER_IMAGE_ID=plan["runtime_container_image_id"])
        for command in plan["commands"]:
            active, started = command["stage"], time.monotonic()
            phase = {"stage": active, "started_unix": time.time(), "args": command["args"]}
            phases.append(phase)
            end = min(deadline, started + command["seconds"])
            if end <= started:
                raise TimeoutError("Work deadline reached")
            guard_alive(job, reg)
            shared.resources(job)
            argv = [sys.executable] + [a.replace("{job}", str(job)) for a in command["args"][1:]]
            with (job / (active + ".log")).open("x", encoding="utf-8") as log:
                child = subprocess.Popen(argv, cwd=workspace, env=env, stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                reader = threading.Thread(target=shared.drain_output, args=(child.stdout, log), daemon=True)
                reader.start()
                try:
                    while child.poll() is None:
                        guard_alive(job, reg)
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
            phase.update(exit_code=child.returncode, seconds=time.monotonic() - started)
            if active == "diagnostic":
                summary_path = job / "cross-pose/summary.json"
                if summary_path.exists():
                    summary = json.loads(summary_path.read_text())
                    phase["summary_sha256"] = sha(summary_path)
                    phase["diagnostic_status"] = summary.get("status")
                    phase["success_count"] = summary.get("success_count")
                    if (summary.get("provider_qualified") is not False or summary.get("formal_gate_result") is not None
                            or summary.get("registered_seeds") != list(range(1900, 1905))
                            or summary.get("max_steps_per_seed") != 500
                            or summary.get("optimizer_steps") != 0 or summary.get("model_forward_calls") != 0):
                        raise ValueError("Diagnostic violated fixed measurement boundaries")
                    if child.returncode:
                        phase["infrastructure_reason"] = safe_message(json.dumps({
                            "status": summary.get("status"), "episodes": summary.get("episodes", [])}))
                elif not child.returncode:
                    raise ValueError("Diagnostic succeeded without measurement summary")
            save(job / (active + "-phase.json"), phase)
            if child.returncode:
                raise RuntimeError("Infrastructure phase failed: " + active + "; exit=" + str(child.returncode))
        failed = False
        save(job / "result.json", {"status": "diagnostic_completed", "phases": phases,
            "simulation_step_cap": 2500, "seeds": list(range(1900, 1905)), "optimizer_updates": 0,
            "model_calls": 0, "formal_gate_result": None, "provider_qualified": False,
            "shutdown_requested": False, "automatic_retry": False})
    except Exception as exc:
        save(job / "failure.json", {"phase": active, "error_type": type(exc).__name__,
            "message": safe_message(exc), "phases": phases, "provider_qualified": False,
            "formal_gate_result": None, "shutdown_requested": False})
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        shared.stop_children()
        seal(job, "failed" if failed else "completed")
        os.sync()
    return 1 if failed else 0


def launch(workspace, job, plan_path):
    # Historical jobs are never touched; prerequisite failures receive fresh evidence.
    job.mkdir()
    child = None
    try:
        plan = json.loads(plan_path.read_text())
        validate(plan)
        if job.name != plan["id"]:
            raise ValueError("Plan and job identity mismatch")
        shared.check_sources(workspace, plan)
        verify_prior(prior_root(job, plan), plan)
        for argv, codes in ((["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], (0,)),
                            (["tmux", "list-sessions"], (0, 1))):
            check = subprocess.run(argv, capture_output=True, timeout=10, check=False)
            if check.returncode not in codes or check.stdout.strip():
                raise RuntimeError("Existing GPU work or tmux prevents launch")
        now = time.time()
        work = work_deadline(plan, now)
        if shutil.disk_usage(job.parent).free < shared.RESERVE:
            raise RuntimeError("Insufficient disk reserve")
        reg = {"id": plan["id"], "started_unix": now, "work_deadline_unix": work,
            "shutdown_authorized": False, "platform_target_unix": plan["platform_target_unix"],
            "plan_sha256": sha(plan_path), "source_sha256": plan["source_sha256"],
            "runtime_profile_reference": shared.PROFILE, "runtime_profile_sha256": sha(workspace / shared.PROFILE),
            "runtime_container_image_id": plan["runtime_container_image_id"], "nested_docker_used": False,
            "rss_limit_bytes": shared.RSS_LIMIT, "evidence_limit_bytes": shared.EVIDENCE_LIMIT,
            "disk_reserve_bytes": shared.RESERVE, "provider_qualified": False, "formal_gate_result": None}
        save(job / "registration.json", reg)
        with (job / "guard.log").open("x") as log:
            guard = subprocess.Popen([sys.executable, "-u", str(workspace / "scripts/work_only_guard.py"),
                "--job", str(job)], stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        for _ in range(50):
            if (job / "guard-armed.json").exists():
                break
            if guard.poll() is not None:
                raise RuntimeError("Work guard failed before arming")
            time.sleep(.1)
        guard_alive(job, reg)
        armed = json.loads((job / "guard-armed.json").read_text())
        if armed["pid"] != guard.pid:
            raise RuntimeError("Work guard PID mismatch")
        command = [sys.executable, "-u", "scripts/run_teacher_diagnostic_job.py", "worker",
            "--workspace", str(workspace), "--job", str(job), "--plan", str(plan_path)]
        with (job / "supervisor.log").open("x") as log:
            child = subprocess.Popen(["bash", "scripts/run_autodl.sh", "shell"], cwd=workspace,
                stdin=subprocess.PIPE, stdout=log, stderr=log, start_new_session=True)
            ticks = Path(f"/proc/{child.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
            save(job / "worker-pid.json", {"pid": child.pid, "start_ticks": ticks})
            child.stdin.write(("exec " + shlex.join(command) + "\n").encode())
            child.stdin.close()
        print(json.dumps({"worker_pid": child.pid, "guard_pid": guard.pid,
            "work_deadline_unix": work, "shutdown_authorized": False, "provider_qualified": False}))
    except Exception as exc:
        save(job / "preparation-failure.json", {"error_type": type(exc).__name__, "message": safe_message(exc),
            "provider_qualified": False, "shutdown_requested": False})
        if child is not None and (job / "worker-pid.json").exists():
            work_only_guard.cleanup.stop_worker(job)
        elif child is not None and child.poll() is None:
            # An unreaped child returned by this Popen is owned even if reading
            # its /proc identity failed before the durable record was written.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=5)
        if not (job / "worker-exited.json").exists():
            seal(job, "preparation_failed")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "worker"))
    for key in ("workspace", "job", "plan"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    result = globals()[args.mode](args.workspace.resolve(strict=True), args.job.resolve(), args.plan.resolve(strict=True))
    if isinstance(result, int):
        sys.exit(result)
