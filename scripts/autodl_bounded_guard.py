"""Independent bounded AutoDL guard; import has no process or shutdown side effects."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

WRAPPER_SHA256 = "0358e83eeeaf542aa98f64ba9e339c91df46f1e025892d52dba159f4fb1cf027"


def save(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def process_identity(pid: int, proc_root: Path = Path("/proc")) -> tuple[str, int] | None:
    """Read Linux start ticks and process group, tolerating exit during the read."""
    try:
        # comm may contain spaces or parentheses: fields after the LAST ')' start at 3.
        fields = (proc_root / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        return fields[19], int(fields[2])
    except (FileNotFoundError, ProcessLookupError):
        return None


def stop_worker(job: Path, grace_seconds: float = 3) -> str:
    record = job / "worker-pid.json"
    if not record.exists():
        return "not_registered"
    value = json.loads(record.read_text())
    pid = value["pid"]
    if type(pid) is not int or pid <= 1 or pid == os.getpid():
        raise ValueError("Worker PID must identify a separate process group leader")
    expected = (str(value["start_ticks"]), pid)
    if process_identity(pid) != expected:
        return "exited_or_identity_changed"
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return "exited_before_term"
    time.sleep(grace_seconds)
    if process_identity(pid) != expected:
        return "terminated_or_identity_changed"
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return "exited_before_kill"
    return "killed_owned_group"


def deadlines(reg: dict, *, wall: float, monotonic: float) -> tuple[float, float]:
    start, work, shutdown = (reg[k] for k in (
        "started_unix", "work_deadline_unix", "shutdown_deadline_unix"
    ))
    if any(type(v) not in (int, float) or not math.isfinite(v)
           for v in (start, work, shutdown)):
        raise ValueError("Deadlines must be finite timestamps")
    if not start < work <= shutdown or shutdown - start > 24 * 3600 or start > wall + 5:
        raise ValueError("Invalid or unbounded registered deadline")
    # A clock rollback after startup must not extend this guard's lifetime.
    return monotonic + max(0, work - wall), monotonic + max(0, shutdown - wall)


def request_shutdown(job: Path) -> None:
    wrapper = Path("/usr/bin/shutdown")
    if hashlib.sha256(wrapper.read_bytes()).hexdigest() != WRAPPER_SHA256:
        raise RuntimeError("Platform shutdown wrapper changed")
    trash = Path("/root/.local/share/Trash")
    if trash.exists() or trash.is_symlink():
        raise RuntimeError("Shutdown would delete existing Trash; use platform UI")
    gpu = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10, check=False,
    )
    sessions = subprocess.run(
        ["tmux", "list-sessions"], capture_output=True, text=True, timeout=10, check=False,
    )
    if gpu.returncode or gpu.stdout.strip():
        raise RuntimeError("Active GPU worker prevents shutdown")
    if sessions.returncode not in (0, 1) or sessions.stdout.strip():
        raise RuntimeError("Unrelated tmux session prevents shutdown")
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = (proc / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"rosetta" in argv and any(token in argv for token in (
            b"train_", b"run_smolvla", b"evaluate_", b"pytest", b"audit.py",
            b"audit_targeted_aloha.py",
        )):
            raise RuntimeError("Active project worker prevents shutdown")
    save(job / "shutdown-request.json", {
        "observed_unix": time.time(), "shutdown_wrapper_sha256": WRAPPER_SHA256,
        "release_requested": False, "platform_stopped_independently_verified": False,
    })
    os.sync()
    if trash.exists() or trash.is_symlink():
        raise RuntimeError("Trash appeared before shutdown")
    os.execv("/bin/bash", ["bash", str(wrapper)])


def run(job: Path, grace_seconds: float = 120) -> None:
    if not 0 <= grace_seconds <= 300:
        raise ValueError("Retrieval grace must be bounded by 300 seconds")
    reg = json.loads((job / "registration.json").read_text())
    if reg.get("shutdown_authorized") is not True:
        raise PermissionError("Registration must explicitly authorize platform shutdown")
    work, shutdown = deadlines(reg, wall=time.time(), monotonic=time.monotonic())
    save(job / "guard-armed.json", {
        "pid": os.getpid(), "observed_unix": time.time(),
        "work_deadline_unix": reg["work_deadline_unix"],
        "shutdown_deadline_unix": reg["shutdown_deadline_unix"],
        "clock": "monotonic_after_registration", "retrieval_grace_seconds": grace_seconds,
    })
    early = None
    stopped = False
    try:
        while time.monotonic() < shutdown:
            now = time.monotonic()
            if now >= work and not stopped:
                stop_worker(job)
                stopped = True
            if (job / "closeout-now").exists():
                break
            if any((job / name).exists() for name in ("worker-exited.json", "failure.json")):
                early = early if early is not None else min(shutdown, now + grace_seconds)
                if now >= early:
                    break
            time.sleep(min(2, max(0, shutdown - now)))
        stop_worker(job)
    except Exception as exc:
        save(job / "guard-worker-error.json", {"error": str(exc), "observed_unix": time.time()})
    # Worker cleanup failure must not silently skip shutdown safety checks.
    try:
        request_shutdown(job)
    except Exception as exc:
        save(job / "shutdown-blocked.json", {"error": str(exc), "observed_unix": time.time()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--retrieval-grace-seconds", type=float, default=120)
    args = parser.parse_args()
    run(args.job.resolve(strict=True), args.retrieval_grace_seconds)
