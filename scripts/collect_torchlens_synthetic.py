"""Isolated synthetic diagnostic. Run only inside the bounded Linux container."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from rosetta_reality.diagnostics.torchlens_capture import (
    CASES,
    compare_arms,
    digest,
    envelope,
    execute_arm,
    write_new,
)


def run_worker(command, log_path, timeout):
    """Timeout kills and reaps the worker; its create-only log stays available."""
    with log_path.open("xb") as log:
        return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                              timeout=timeout, check=False, env=os.environ.copy())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-id", default="torchlens-synthetic")
    parser.add_argument("--backend", choices=("plain", "torchlens", "reference", "basin-torchlens"),
                        default="torchlens")
    parser.add_argument("--case", choices=CASES, default="normal")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if sys.platform != "linux" or not Path("/.dockerenv").exists():
        raise RuntimeError("Use the WSL-launched Linux Docker runner")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.worker:
        result = envelope(args.run_id)
        try:
            if args.backend == "basin-torchlens":
                from scripts.basin_torchlens_worker import collect

                result = collect(args.case, args.run_id, args.output, result)
            else:
                result = execute_arm(args.backend, args.case, args.run_id, result=result)
            write_new(args.output / "native.json", result)
        except Exception as exc:
            failure = result
            failure["status"] = "incomplete"
            failure["parameters"].update({"backend": args.backend, "case": args.case,
                                          "error_type": type(exc).__name__})
            write_new(args.output / "failure.json", failure)
            raise
        return
    receipt = envelope(args.run_id)
    receipt["parameters"].update({"case": args.case, "arms": {}, "parity": None})
    deadline = time.monotonic() + 175
    try:
        for name, backend in (("control", "plain"), ("repeat", "plain"),
                              ("observed", args.backend)):
            command = [sys.executable, __file__, "--worker", "--backend", backend,
                       "--case", args.case, "--run-id", args.run_id,
                       "--output", str(args.output / name)]
            completed = run_worker(command, args.output / f"{name}.log",
                                   max(0.01, deadline - time.monotonic()))
            receipt["parameters"]["arms"][name] = {"exit_code": completed.returncode}
            if completed.returncode:
                raise RuntimeError(f"{name} worker failed; preserved its log")
        arms = [json.loads((args.output / name / "native.json").read_text())
                for name in ("control", "repeat", "observed")]
        parity = compare_arms(*arms)
        receipt["parameters"]["parity"] = parity
        observed = arms[2]
        observed["parameters"]["consistency"] = parity
        observed["parameters"]["arm_files"] = {
            name: digest((args.output / name / "native.json").read_bytes())
            for name in ("control", "repeat", "observed")}
        observed["status"] = "complete" if parity["passed"] else "incomplete"
        write_new(args.output / "basin-native.json", observed)
        if not parity["passed"]:
            raise RuntimeError("Strict parity failed; no relaxed tolerance is allowed")
        receipt["status"] = "complete"
    except Exception as exc:
        receipt["parameters"]["error_type"] = type(exc).__name__
        raise
    finally:
        receipt["parameters"]["files"] = {
            path.relative_to(args.output).as_posix(): digest(path.read_bytes())
            for path in sorted(args.output.rglob("*")) if path.is_file()}
        write_new(args.output / "receipt.json", receipt)


if __name__ == "__main__":
    main()
