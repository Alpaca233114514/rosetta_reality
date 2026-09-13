"""Guarded same-checkpoint scene K/V ablation supervisor."""

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
sys.path.insert(0, str(ROOT))

from scripts.hestia_q_replay import CONDITIONS, validate_protocol  # noqa: E402


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def ticks(pid):
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def alive(pid, start):
    try:
        return ticks(pid) == start
    except (FileNotFoundError, ProcessLookupError):
        return False


def stop_child(job):
    path = job / "active-child.json"
    if not path.exists():
        return
    value = json.loads(path.read_text())
    if alive(value["pid"], value["ticks"]):
        os.killpg(value["pid"], signal.SIGTERM)
        for _ in range(20):
            if not alive(value["pid"], value["ticks"]):
                return
            time.sleep(0.25)
        if alive(value["pid"], value["ticks"]):
            os.killpg(value["pid"], signal.SIGKILL)


def shutdown(job, template):
    if (job / "shutdown-request.json").exists():
        return
    path = ROOT / "scripts/run_visual_coverage_job.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != template["shutdown_helper_sha256"]:
        raise ValueError("Registered shutdown helper changed")
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_visual_coverage_job as helper

    helper.JOB = job
    helper.shutdown()


def watch(job, template):
    registration = json.loads((job / "registration.json").read_text())
    if (
        registration["model_execution_authorized"] is not True
        or registration["shutdown_authorized"] is not True
    ):
        raise ValueError("Watchdog lacks explicit authorization")
    while time.time() < registration["deadline_unix"]:
        time.sleep(min(10, registration["deadline_unix"] - time.time()))
    if not (job / "worker-exited.json").exists():
        stop_child(job)
    if not (job / "worker-exited.json").exists() and alive(
        registration["parent_pid"], registration["parent_ticks"]
    ):
        os.kill(registration["parent_pid"], signal.SIGTERM)
    while time.time() < registration["shutdown_deadline_unix"]:
        if (job / "shutdown-request.json").exists():
            return
        time.sleep(min(2, registration["shutdown_deadline_unix"] - time.time()))
    save(job / "watchdog-shutdown.json", {"deadline_reached": True, "time": time.time()})
    shutdown(job, template)


def supervise(args, template):
    if not args.execute_authorized or not args.shutdown_authorized:
        raise ValueError("Explicit inference and shutdown authorization required")
    if (
        not os.environ.get("ROSETTA_AUTODL_RUNTIME_PROFILE")
        or os.environ.get("ROSETTA_TORCH_DEVICE") != "cuda"
    ):
        raise ValueError("Launch via registered AutoDL shell")
    job = Path(template["output"])
    if job.is_absolute() or ".." in job.parts:
        raise ValueError("Job path must remain within the fresh workspace")
    job.mkdir(parents=True, exist_ok=False)
    started = time.time()
    registration = {
        "model_execution_authorized": True,
        "training_authorized": False,
        "shutdown_authorized": True,
        "started_unix": started,
        "deadline_unix": started + 4200,
        "shutdown_deadline_unix": started + 4800,
        "template_sha256": hashlib.sha256(args.template.read_bytes()).hexdigest(),
        "allowed_conditions": list(CONDITIONS),
        "parent_pid": os.getpid(),
        "parent_ticks": ticks(os.getpid()),
    }
    save(job / "registration.json", registration)
    with (job / "watchdog.log").open("xb") as stream:
        watchdog = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "watchdog",
                "--template",
                str(args.template.resolve()),
            ],
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    permit = {
        **registration,
        "watchdog_active": True,
        "watchdog_pid": watchdog.pid,
        "watchdog_ticks": ticks(watchdog.pid),
    }
    save(job / "permit.json", permit)

    def terminated(_signum, _frame):
        raise TimeoutError("Shared checkpoint observation deadline reached")

    signal.signal(signal.SIGTERM, terminated)
    completed = []
    error = None
    child = None
    try:
        active = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if active.returncode != 0 or active.stdout.strip():
            raise ValueError("GPU is unavailable or another worker is active")
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                argv = (proc / "cmdline").read_bytes().split(b"\0")
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            if any(b"rosetta" in a for a in argv) and any(
                any(
                    token in a
                    for token in (
                        b"train_",
                        b"evaluate_",
                        b"run_smolvla",
                        b"diagnose_",
                        b"pytest",
                        b"run_hestia_fit.py",
                    )
                )
                for a in argv
            ):
                raise ValueError("Another project worker is active")
        doctor = subprocess.run(
            ["bash", "scripts/run_autodl.sh", "doctor"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        save(
            job / "doctor-check.json",
            {"exit_code": doctor.returncode, "stdout": doctor.stdout, "stderr": doctor.stderr},
        )
        if doctor.returncode != 0:
            raise ValueError("Fresh AutoDL doctor failed")
        normalization = subprocess.run(
            [
                sys.executable,
                "scripts/prepare_hestia_checkpoint_workspace.py",
                "--template",
                str(args.template),
                "--bind-existing",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        save(
            job / "normalization-check.json",
            {
                "exit_code": normalization.returncode,
                "stdout": normalization.stdout,
                "stderr": normalization.stderr,
            },
        )
        if normalization.returncode != 0:
            raise ValueError("Native normalization workspace preflight failed")
        check = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "tests/test_hestia_parameter_crossover.py",
                "tests/test_hestia_parameter_crossover_guard.py",
                "tests/test_hestia_local_checkpoint.py",
                "tests/test_hestia_parameter_crossover_cli.py",
                "tests/test_hestia_kv_split.py",
                "tests/test_hestia_kv_split_cli.py",
                "tests/test_hestia_scene_kv.py",
                "tests/test_hestia_scene_kv_cli.py",
                "tests/test_hestia_q_replay.py",
                "tests/test_hestia_parameter_crossover_transfer.py",
            ],
            capture_output=True,
            text=True,
            timeout=45,
        )
        save(
            job / "cpu-check.json",
            {"exit_code": check.returncode, "stdout": check.stdout, "stderr": check.stderr},
        )
        if check.returncode != 0:
            raise ValueError("Fresh runtime CPU checks failed")
        for condition in registration["allowed_conditions"]:
            if time.time() >= registration["deadline_unix"] or not alive(
                watchdog.pid, permit["watchdog_ticks"]
            ):
                raise TimeoutError("Deadline or watchdog prerequisite failed")
            with (job / f"condition-{condition}.log").open("xb") as stream:
                child = subprocess.Popen(
                    [
                        sys.executable,
                        "scripts/diagnose_hestia_q_replay.py",
                        "--template",
                        str(args.template),
                        "--permit",
                        str(job / "permit.json"),
                        "--condition",
                        condition,
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                active = {"pid": child.pid, "ticks": ticks(child.pid), "condition": condition}
                # Mutable control state is confined to this new job, not evidence.
                temporary = job / "active-child.tmp"
                temporary.write_text(json.dumps(active))
                temporary.replace(job / "active-child.json")
                status = child.wait(timeout=max(1, registration["deadline_unix"] - time.time()))
                if status != 0:
                    raise RuntimeError(f"Condition {condition} collector exit {status}")
                completed.append(condition)
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        # Popen owns this child even if the atomic PID record was interrupted.
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
    finally:
        save(
            job / "worker-exited.json",
            {
                "completed_conditions": completed,
                "error": error,
                "optimizer_steps": 0,
                "time": time.time(),
                "elapsed_seconds": time.time() - started,
            },
        )
        files = {}
        for path in job.rglob("*"):
            if not path.is_file() or path.name in {
                "watchdog.log",
                "active-child.json",
                "active-child.tmp",
            }:
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(job.resolve()):
                raise ValueError("Result path escaped")
            files[path.relative_to(job).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        if sum(v["bytes"] for v in files.values()) > 512 * 1024**2:
            raise ValueError("Result transfer exceeds registered bound")
        save(
            job / "handoff-manifest.json",
            {"files": files, "total_bytes": sum(v["bytes"] for v in files.values())},
        )
        expected_manifest = hashlib.sha256((job / "handoff-manifest.json").read_bytes()).hexdigest()
        stop = min(registration["shutdown_deadline_unix"], time.time() + 240)
        while time.time() < stop:
            receipt_path = job / "transfer-receipt.json"
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if (
                    receipt.get("all_file_sha256_matched") is True
                    and receipt.get("manifest_sha256") == expected_manifest
                ):
                    break
            time.sleep(min(2, stop - time.time()))
        # Only result transfer, never local analysis, may keep this window open.
        shutdown(job, template)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("supervise", "watchdog"))
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--execute-authorized", action="store_true")
    parser.add_argument("--shutdown-authorized", action="store_true")
    args = parser.parse_args()
    template = json.loads(args.template.read_text(encoding="utf-8-sig"))
    validate_protocol(template)
    if args.mode == "watchdog":
        watch(Path(template["output"]), template)
    else:
        supervise(args, template)


if __name__ == "__main__":
    main()
