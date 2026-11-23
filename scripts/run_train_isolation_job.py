"""Bounded AutoDL train-isolation qualification, no model or optimizer."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PLAN = "configs/diagnostics/aloha_train_isolation_20260927_004.json"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1024*1024):
            h.update(block)
    return h.hexdigest()


def save(path, payload):
    with path.open("x") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())


def worker(workspace, job):
    started = time.monotonic()
    root = job.parents[1]
    phases = []
    evidence = job / "evidence"
    evidence.mkdir()
    active = None
    try:
        from importlib.metadata import version
        reg = json.loads((job / "registration.json").read_text())
        if sha(workspace / PLAN) != reg["plan_sha256"]:
            raise ValueError("Plan drift")
        save(evidence / "runtime.json", {"python": sys.version.split()[0], "pyarrow": version("pyarrow"),
             "numpy": version("numpy"), "nested_docker_used": False, "model_loaded": False,
             "runtime_profile_sha256": sha(workspace / "configs/runtime/autodl_rtx4090.yaml"),
             "runtime_profile_reference": "configs/runtime/autodl_rtx4090.yaml", "cuda_doctor_run": False,
             "cgroup_memory_limit": Path("/sys/fs/cgroup/memory.max").read_text().strip(),
             "data_disk_available_bytes": shutil.disk_usage(root).free})
        env = dict(os.environ, PYTHONPATH=f"{workspace / 'src'}:{workspace}", PYTHONDONTWRITEBYTECODE="1",
                   HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        commands = [
            ("regression", [sys.executable, "-m", "unittest", "discover", "-v", "-s", "tests", "-p", "test_execution_qualification_stdlib.py"], 45),
            ("synthetic-isolation", [sys.executable, "-m", "unittest", "discover", "-v", "-s", "tests", "-p", "test_train_isolation.py"], 45),
            ("real-isolation", [sys.executable, "scripts/isolate_aloha_train.py", "run", "--plan", PLAN,
                                "--data-root", str(root), "--output", str(job / "isolation")], 180),
        ]
        for name, command, budget in commands:
            active = name
            if time.time() + budget > reg["work_deadline_unix"] or shutil.disk_usage(root).free < 1024**3:
                raise RuntimeError("Insufficient remaining budget")
            phase_started = time.monotonic()
            with (evidence / (name + ".log")).open("x") as log:
                p = subprocess.run(command, cwd=workspace, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, timeout=budget, check=False)
            phases.append({"phase": name, "exit_code": p.returncode, "seconds": time.monotonic()-phase_started})
            if p.returncode:
                raise RuntimeError("Phase failed")
        # Transfer metadata and train-only audit aggregates, never raw dataset payloads.
        for name, path in (("isolation-manifest.json", job / "isolation/train-only/manifest.json"),
                           ("isolation-verification.json", job / "isolation/train-only/verification.json")):
            with (evidence / name).open("xb") as target:
                target.write(path.read_bytes())
        verified = json.loads((evidence / "isolation-verification.json").read_text())
        save(evidence / "result.json", {"status": "train_isolation_and_row_audit_passed", "phases": phases,
             "elapsed_seconds": time.monotonic()-started, "frames": sum(x["frames"] for x in verified["sources"]),
             "protected_rows_exported": 0, "protected_rows_may_be_decoded_inside_worker": True,
             "training_ready": False, "model_loaded": False, "optimizer_updates": 0, "simulation_rollouts": 0,
             "remaining_blockers": ["video_alignment", "verified_success_and_recovery_labels", "matched_cohort_and_scene_parent_provenance", "base_gate3_failed"]})
    except Exception as exc:
        save(evidence / "failure.json", {"phase": active, "error_type": type(exc).__name__, "phases": phases,
                                       "elapsed_seconds": time.monotonic()-started, "training_ready": False})
    finally:
        save(job / "worker-exited.json", {"finished_unix": time.time(), "status": "failed" if (evidence / "failure.json").exists() else "completed"})
        for name in ("registration.json", "guard-armed.json", "worker-pid.json", "worker-exited.json"):
            with (evidence / name).open("xb") as f:
                f.write((job / name).read_bytes())
        entries = [{"path": p.name, "sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(evidence.iterdir()) if p.is_file()]
        save(job / "handoff-ready.json", {"files": entries, "bytes": sum(x["bytes"] for x in entries), "raw_payload_included": False})
        os.sync()


def launch(workspace, job, platform):
    now = time.time()
    if not 450 <= platform-now <= 900:
        raise ValueError("Platform deadline must leave a bounded 7.5 to 15 minute window")
    job.mkdir()
    save(job / "registration.json", {"run": job.name, "started_unix": now, "work_deadline_unix": now+300,
         "shutdown_deadline_unix": now+420, "shutdown_authorized": True, "platform_deadline_unix": platform,
         "plan_sha256": sha(workspace / PLAN), "guard_sha256": sha(workspace / "scripts/autodl_bounded_guard.py"),
         "worker_sha256": sha(Path(__file__)), "isolation_code_sha256": sha(workspace / "scripts/isolate_aloha_train.py"),
         "source_workspace": str(workspace.relative_to(job.parents[1])), "model_forward_limit": 0, "optimizer_updates": 0,
         "simulation_rollouts": 0, "automatic_retry_resume_training": False, "max_export_bytes": 32*1024**2})
    with (job / "guard.log").open("x") as log:
        guard = subprocess.Popen([sys.executable, "-u", str(workspace / "scripts/autodl_bounded_guard.py"),
                                  "--job", str(job), "--retrieval-grace-seconds", "240"],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(30):
        if (job / "guard-armed.json").exists():
            break
        if guard.poll() is not None:
            raise RuntimeError("Guard failed")
        time.sleep(.1)
    if not (job / "guard-armed.json").exists():
        raise RuntimeError("Guard not armed")
    with (job / "supervisor.log").open("x") as log:
        child = subprocess.Popen([sys.executable, "-u", __file__, "worker", "--workspace", str(workspace), "--job", str(job)],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        ticks = Path(f"/proc/{child.pid}/stat").read_text().rsplit(")",1)[1].split()[19]
        save(job / "worker-pid.json", {"pid": child.pid, "start_ticks": ticks})
    print(json.dumps({"worker_pid": child.pid, "guard_pid": guard.pid, "work_deadline_unix": now+300, "shutdown_deadline_unix": now+420}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("launch", "worker"))
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--platform-deadline-unix", type=float)
    a = p.parse_args()
    if a.mode == "launch":
        launch(a.workspace.resolve(), a.job, a.platform_deadline_unix)
    else:
        worker(a.workspace.resolve(), a.job)
