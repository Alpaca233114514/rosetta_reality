"""Q0 guard wrapper preserving original deadlines and shutdown safety checks."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import time

import autodl_bounded_guard as original

_shutdown = original.request_shutdown


def members(group, proc_root=Path("/proc")):
    found = []
    for proc in proc_root.iterdir():
        if not proc.name.isdigit():
            continue
        identity = original.process_identity(int(proc.name), proc_root)
        if identity is not None and identity[1] == group:
            try:
                state = (proc / "stat").read_text().rsplit(")", 1)[1].split()[0]
                if state != "Z":
                    found.append((int(proc.name), identity[0]))
            except (FileNotFoundError, ProcessLookupError):
                pass
    return found


def stop_worker(job, grace_seconds=3):
    record = job / "worker-pid.json"
    if not record.exists():
        return "not_registered"
    value = json.loads(record.read_text())
    pid = value["pid"]
    if type(pid) is not int or pid <= 1 or pid == os.getpid():
        raise ValueError("Worker PID must identify a separate group leader")
    expected = (str(value["start_ticks"]), pid)
    identity = original.process_identity(pid)
    owned = members(pid)
    if identity != expected:
        if owned:
            if not (job / "q0-worker-cleanup-blocked.json").exists():
                original.save(job / "q0-worker-cleanup-blocked.json", {
                    "reason": "leader_identity_unverifiable_with_remaining_group_members",
                    "observed_unix": time.time(), "shutdown_blocked": True})
            raise RuntimeError("Cannot prove ownership of remaining worker group; shutdown blocked")
        return "exited_group_empty"
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    time.sleep(grace_seconds)
    # Never rely on the leader surviving TERM: children can ignore TERM independently.
    for child_pid, ticks in owned:
        if original.process_identity(child_pid) == (ticks, pid):
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    # A child forked after our snapshot is not independently identity-bound.
    # Zombies are harmless, but any live remaining member blocks shutdown.
    end = time.monotonic() + 3
    while time.monotonic() < end:
        remaining = []
        for child_pid, _ticks in members(pid):
            try:
                fields = Path(f"/proc/{child_pid}/stat").read_text().rsplit(")", 1)[1].split()
                if fields[0] != "Z":
                    remaining.append(child_pid)
            except (FileNotFoundError, ProcessLookupError):
                pass
        if not remaining:
            return "owned_group_terminated"
        time.sleep(.05)
    if not (job / "q0-worker-cleanup-blocked.json").exists():
        original.save(job / "q0-worker-cleanup-blocked.json", {
            "reason": "live_group_members_after_identity_bound_cleanup", "observed_unix": time.time(),
            "shutdown_blocked": True})
    raise RuntimeError("Remaining live group members; shutdown blocked")


def request_shutdown(job):
    if any((job / name).exists() for name in ("q0-worker-cleanup-blocked.json", "guard-worker-error.json")):
        raise RuntimeError("Q0 cleanup ownership failure blocks shutdown")
    _shutdown(job)


def run(job, grace_seconds=300):
    original.stop_worker = stop_worker
    original.request_shutdown = request_shutdown
    original.run(job, grace_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--retrieval-grace-seconds", type=float, default=300)
    args = parser.parse_args()
    run(args.job.resolve(strict=True), args.retrieval_grace_seconds)
