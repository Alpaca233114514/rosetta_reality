"""Bounded offline CPU subprocess supervisor for the repaired training chain.

Runs only explicit test/script commands, with no production policy optimization.
Evidence is create-only; this runner neither authorizes CUDA nor labels Gates passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def digest(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def group_rss(group):
    total = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(stat[2]) == group:
                total += int(stat[21]) * os.sysconf("SC_PAGE_SIZE")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--commands", type=Path, required=True)
    parser.add_argument("--commands-sha256", required=True)
    args = parser.parse_args()
    if (
        sys.platform != "linux"
        or os.environ.get("HF_HUB_OFFLINE") != "1"
        or os.environ.get("HF_DATASETS_OFFLINE") != "1"
        or digest(args.commands) != args.commands_sha256
    ):
        raise ValueError("Offline Linux and sealed command registration required")
    registration = json.loads(args.commands.read_text())
    if registration["production_optimizer_steps"] != 0:
        raise ValueError("CPU audit never authorizes production optimization")
    for name, sha in registration["source_sha256"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Source identity drift: " + name)
    args.job.mkdir(parents=True, exist_ok=False)
    start = time.time()
    deadline = start + min(registration["seconds"], 1800)
    result = {
        "status": "running", "started_unix": start,
        "command_registration_sha256": args.commands_sha256,
        "production_optimizer_steps": 0, "model_weights_loaded": False,
        "new_gate_measured": False, "stages": [],
    }
    save(args.job / "started.json", result)
    for row in registration["commands"]:
        name, command = row["name"], row["argv"]
        if not name.replace("-", "").isalnum() or command[0] != "python":
            raise ValueError("Invalid registered command")
        began = time.time()
        peak = 0
        reason = None
        with (args.job / (name + ".log")).open("x") as log:
            process = subprocess.Popen(
                [sys.executable, *command[1:]], cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            while process.poll() is None:
                peak = max(peak, group_rss(process.pid))
                if time.time() >= min(deadline, began + row.get("timeout", 600)):
                    reason = "deadline_exceeded"
                elif peak > 1536 * 1024**2:
                    reason = "worker_process_tree_rss_exceeded_1536_MiB"
                if reason:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    break
                time.sleep(1)
            # Ensure a failed test cannot leave its own subprocess running.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        record = {
            "name": name, "exit_code": process.returncode, "stop_reason": reason,
            "seconds": time.time() - began, "peak_process_tree_rss": peak,
            "log_sha256": digest(args.job / (name + ".log")),
        }
        result["stages"].append(record)
        save(args.job / (name + "-exit.json"), record)
        if reason or (process.returncode and not row.get("independent", False)):
            break
    result.update(
        status=("passed" if len(result["stages"]) == len(registration["commands"])
                and all(r["exit_code"] == 0 for r in result["stages"]) else "failed"),
        finished_unix=time.time(),
        source_unchanged=all(digest(ROOT / p) == h
                             for p, h in registration["source_sha256"].items()),
    )
    if not result["source_unchanged"]:
        result["status"] = "failed"
    save(args.job / "result.json", result)
    files = {p.relative_to(args.job).as_posix():
             {"sha256": digest(p), "bytes": p.stat().st_size}
             for p in sorted(args.job.rglob("*")) if p.is_file()}
    save(args.job / "manifest.json", {"files": files})
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
