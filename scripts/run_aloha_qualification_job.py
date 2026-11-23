"""Detached, bounded Q audit supervisor. No model, optimizer, simulation or downloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
from pathlib import Path
import shutil
import subprocess
import sys
import time


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def worker(workspace, job):
    started = time.monotonic()
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024**2, 8 * 1024**2))
    plan = json.loads((workspace / "configs/diagnostics/aloha_qualification_20260927_002.json").read_text())
    root = job.parents[1]
    phases = []
    try:
        env = dict(os.environ, PYTHONPATH=f"{workspace / 'src'}:{workspace}",
                   PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
        from importlib.metadata import version
        save(job / "runtime.json", {"python": sys.version.split()[0], "pyarrow": version("pyarrow"),
             "numpy": version("numpy"), "nested_docker_used": False,
             "runtime_profile_reference": "configs/runtime/autodl_rtx4090.yaml",
             "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"], text=True).strip(),
             "cgroup_memory_limit": Path("/sys/fs/cgroup/memory.max").read_text().strip(),
             "data_disk_available_bytes": shutil.disk_usage(root).free})
        # Source receipts remain untouched. Re-read and hash only the two selected source namespaces.
        ledger = []
        for source in plan["sources"]:
            location = root / source["root"]
            receipt_path = location / "rosetta_download_manifest.json"
            if sha(receipt_path) != source["manifest_sha256"]:
                raise ValueError("Source receipt drift")
            receipt = json.loads(receipt_path.read_text())
            save(job / ("source-" + source["dataset"].rsplit("_", 1)[-1] + "-receipt.json"), receipt)
            for item in receipt["files"]:
                relative = item["path"]
                if Path(relative).is_absolute() or ".." in Path(relative).parts:
                    raise ValueError("Unsafe receipt member")
                path = (location / relative).resolve()
                if not path.is_relative_to(location.resolve()):
                    raise ValueError("Source member escapes namespace")
                actual = sha(path)
                if actual != item["sha256"] or path.stat().st_size != item["bytes"]:
                    raise ValueError("Source payload identity mismatch")
                ledger.append({"dataset": source["dataset"], "revision": source["revision"],
                               "path": relative, "bytes": item["bytes"], "sha256": actual})
                if time.monotonic() - started > 120 or shutil.disk_usage(root).free < 1024**3:
                    raise RuntimeError("Source verification resource/deadline exceeded")
            info = json.loads((location / "meta/info.json").read_text())
            save(job / ("source-" + source["dataset"].rsplit("_", 1)[-1] + "-info.json"), info)
        save(job / "source-verification.json", {"files": ledger, "file_count": len(ledger),
             "bytes": sum(x["bytes"] for x in ledger), "all_matched": True,
             "data_rows_loaded": False, "video_decoded": False})
        commands = [
            ("regression", [sys.executable, "-m", "unittest", "discover", "-v", "-s", "tests", "-p", "test_execution_qualification_stdlib.py"]),
            ("inventory", [sys.executable, "scripts/audit_targeted_aloha.py", "--output", str(job / "inventory"),
                           "inventory", "--plan", "configs/diagnostics/aloha_qualification_20260927_002.json",
                           "--data-root", str(root), "--read-train-rows"]),
        ]
        for name, command in commands:
            with (job / (name + ".log")).open("x") as log:
                result = subprocess.run(command, cwd=workspace, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=180, check=False)
            phases.append({"name": name, "exit_code": result.returncode})
            if result.returncode:
                raise RuntimeError("Q phase failed: " + name)
            peak_rss = 1024 * (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                               + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
            evidence_bytes = sum(p.stat().st_size for p in job.rglob("*") if p.is_file())
            if peak_rss > 1024**3 or evidence_bytes > 8 * 1024**2:
                raise RuntimeError("RSS/evidence budget exceeded at phase boundary")
        report = json.loads((job / "inventory/report.json").read_text())
        observed = sum(x["frames_observed"] for x in report["episode_observations"].values())
        groups = [g for f in report["files"] for g in f["row_groups"]]
        save(job / "result.json", {"run": job.name, "status": "q_inventory_completed_with_qualification_blockers",
            "phases": phases, "elapsed_seconds": time.monotonic() - started,
            "source_files_verified": len(ledger), "source_bytes_verified": sum(x["bytes"] for x in ledger),
            "data_parquet_files": len(report["files"]), "row_groups": len(groups),
            "row_groups_read": sum(g["raw_rows_read"] for g in groups),
            "row_groups_skipped": sum(not g["admitted_by_episode_statistics"] for g in groups),
            "raw_frames_observed": observed, "hidden_loaded": report["hidden_loaded"],
            "qualified_cohort_available": False, "training_ready": False, "optimizer_updates": 0,
            "simulation_rollouts": 0, "model_loaded": False,
            "blockers": ["shared_protected_row_groups_require_isolated_reader",
                         "success_recovery_labels_and_video_alignment_unverified",
                         "protected_scene_parent_inventory_and_matched_N_unsealed"],
            "ordinary_success_not_inferred_from_next_done": True})
        save(job / "worker-exited.json", {"status": "completed_q_inventory", "exit_code": 0,
             "finished_unix": time.time(), "elapsed_seconds": time.monotonic() - started})
    except Exception as exc:
        save(job / "failure.json", {"error_type": type(exc).__name__, "phases": phases,
                                   "elapsed_seconds": time.monotonic() - started, "training_ready": False})
        raise
    finally:
        # Hash evidence only, never models/data. Save a ready marker before retrieval/closeout.
        entries = []
        for path in sorted(job.rglob("*")):
            if path.is_file() and path.name not in {"guard.log", "supervisor.log", "evidence.sha256", "handoff-ready.json"}:
                entries.append((str(path.relative_to(job)), path.stat().st_size, sha(path)))
        with (job / "evidence.sha256").open("x") as stream:
            for name, size, digest in entries:
                stream.write(f"{digest}  {name}\n")
        save(job / "handoff-ready.json", {"file_count": len(entries), "bytes": sum(x[1] for x in entries),
             "manifest_sha256": sha(job / "evidence.sha256"), "finished_unix": time.time()})


def launch(workspace, job, platform_deadline):
    if job.exists():
        raise FileExistsError("New job identity required")
    started = time.time()
    if platform_deadline - started < 390:
        raise ValueError("Less than six and a half minutes remain before platform shutdown")
    job.mkdir()
    save(job / "registration.json", {"run": job.name, "started_unix": started,
        "work_deadline_unix": started + 240, "shutdown_deadline_unix": started + 360,
        "shutdown_authorized": True, "platform_deadline_unix": platform_deadline,
        "source_workspace": str(workspace.relative_to(job.parents[1])),
        "work_seconds": 240, "shutdown_seconds": 360, "retrieval_grace_seconds": 180,
        "maximum_process_group_rss_bytes": 1024**3, "maximum_evidence_bytes": 8 * 1024**2,
        "minimum_free_disk_bytes": 1024**3, "model_forward_limit": 0, "optimizer_updates": 0,
        "simulation_rollouts": 0, "automatic_retry_resume_training": False,
        "plan_sha256": sha(workspace / "configs/diagnostics/aloha_qualification_20260927_002.json"),
        "guard_sha256": sha(workspace / "scripts/autodl_bounded_guard.py"),
        "worker_sha256": sha(workspace / "scripts/run_aloha_qualification_job.py")})
    with (job / "guard.log").open("x") as log:
        guard = subprocess.Popen([sys.executable, "-u", str(workspace / "scripts/autodl_bounded_guard.py"),
                                  "--job", str(job), "--retrieval-grace-seconds", "180"],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(30):
        if (job / "guard-armed.json").is_file():
            break
        if guard.poll() is not None:
            raise RuntimeError("Guard failed before worker launch")
        time.sleep(0.1)
    if not (job / "guard-armed.json").is_file():
        raise RuntimeError("Guard did not arm")
    with (job / "supervisor.log").open("x") as log:
        child = subprocess.Popen([sys.executable, "-u", __file__, "worker", "--workspace", str(workspace), "--job", str(job)],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        ticks = Path(f"/proc/{child.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
        save(job / "worker-pid.json", {"pid": child.pid, "start_ticks": ticks})
    print(json.dumps({"run": job.name, "guard_pid": guard.pid, "worker_pid": child.pid,
                      "work_deadline_unix": started + 240, "shutdown_deadline_unix": started + 360}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["launch", "worker"])
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--platform-deadline-unix", type=float)
    a = p.parse_args()
    if a.mode == "launch":
        launch(a.workspace.resolve(), a.job, a.platform_deadline_unix)
    else:
        worker(a.workspace.resolve(), a.job)
