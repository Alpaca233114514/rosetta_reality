"""Independent work deadline guard. This module never powers off the platform."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import q0_bounded_guard as cleanup
from run_q0_screen_job import save


def validate_registration(reg):
    if reg.get("shutdown_authorized") is not False:
        raise ValueError("Work guard requires shutdown_authorized=false")
    start, end = reg["started_unix"], reg["work_deadline_unix"]
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in (start, end)):
        raise ValueError("Finite work timestamps required")
    if not 0 < end - start <= 780:
        raise ValueError("Work lifetime must be at most 780 seconds")
    if not isinstance(reg.get("id"), str) or not reg["id"]:
        raise ValueError("Missing job identity")


def run(job):
    reg = json.loads((job / "registration.json").read_text())
    validate_registration(reg)
    if job.name != reg["id"]:
        raise ValueError("Job directory identity mismatch")
    identity = cleanup.original.process_identity(os.getpid())
    if identity is None or identity[1] != os.getpid():
        raise RuntimeError("Guard must own an independent process group")
    deadline = time.monotonic() + max(0, reg["work_deadline_unix"] - time.time())
    save(job / "guard-armed.json", {"id": reg["id"], "pid": os.getpid(),
         "start_ticks": identity[0], "plan_sha256": reg["plan_sha256"],
         "work_deadline_unix": reg["work_deadline_unix"], "shutdown_authorized": False})
    status = "failed"
    try:
        while time.monotonic() < deadline:
            if (job / "worker-exited.json").exists():
                # Exit marker alone does not prove the process group is empty.
                record = job / "worker-pid.json"
                if not record.exists():
                    status = "no_worker_started"
                    break
                value = json.loads(record.read_text())
                if not cleanup.members(value["pid"]):
                    status = "worker_exited_group_empty"
                    break
            time.sleep(min(.5, max(0, deadline - time.monotonic())))
        else:
            status = cleanup.stop_worker(job)
        save(job / "guard-work-completed.json", {"id": reg["id"], "status": status,
             "finished_unix": time.time(), "shutdown_requested": False})
    except Exception as exc:
        save(job / "guard-worker-error.json", {"id": reg["id"], "error_type": type(exc).__name__,
             "shutdown_requested": False})
        raise
    finally:
        save(job / "guard-exited.json", {"id": reg["id"], "finished_unix": time.time(),
             "status": status, "shutdown_requested": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    run(parser.parse_args().job.resolve(strict=True))
