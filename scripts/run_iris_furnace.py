"""Detached Iris supervisor, independent watchdog and bounded result handoff."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from scripts.iris_protocol import OWNERS, RUN, STAGES, bind_inputs  # noqa: E402
from scripts.iris_runtime import save, sha  # noqa: E402


def ticks(pid):
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def alive(pid, start):
    try:
        return ticks(pid) == start
    except (FileNotFoundError, ProcessLookupError):
        return False


def terminate(pid, start):
    if not alive(pid, start):
        return
    os.killpg(pid, signal.SIGTERM)
    until = time.monotonic() + 5
    while alive(pid, start) and time.monotonic() < until:
        time.sleep(0.25)
    if alive(pid, start):
        os.killpg(pid, signal.SIGKILL)


def process_tree_rss(pgid):
    total = 0
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if os.getpgid(int(proc.name)) != pgid:
                continue
            for line in (proc / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return total


def build_template():
    files = set(OWNERS)
    for directory in ("src", "scripts", "configs", "tests"):
        files.update(
            p.relative_to(ROOT).as_posix()
            for p in (ROOT / directory).rglob("*")
            if p.is_file()
            and p.suffix in (".py", ".sh", ".yaml", ".json")
            and "__pycache__" not in p.parts
        )
    return {
        "id": RUN,
        "stages": list(STAGES),
        "work_seconds": 3600,
        "shutdown_seconds": 4200,
        "required_free_bytes": 23 * 1024**3,
        "maximum_optimizer_steps": 2564,
        "sources": {name: sha(ROOT / name) for name in sorted(files)},
        "hidden_test_loaded": False,
        "retry_allowed": False,
        "m2_complete": False,
        "shutdown_helper_sha256": sha(ROOT / "scripts/run_visual_coverage_job.py"),
    }


def validate_template(value):
    expected = {
        "id": RUN,
        "stages": list(STAGES),
        "work_seconds": 3600,
        "shutdown_seconds": 4200,
        "maximum_optimizer_steps": 2564,
        "required_free_bytes": 23 * 1024**3,
        "hidden_test_loaded": False,
        "retry_allowed": False,
        "m2_complete": False,
    }
    if any(type(value.get(k)) is not type(v) or value.get(k) != v for k, v in expected.items()):
        raise ValueError("Frozen Iris template differs")
    if not isinstance(value.get("sources"), dict) or not set(OWNERS) <= set(value["sources"]):
        raise ValueError("Iris source seal is incomplete")
    for name, expected_sha in value["sources"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or sha(ROOT / relative) != expected_sha:
            raise ValueError("Iris source seal changed: " + name)


def shutdown(job, registration):
    if (job / "shutdown-request.json").exists():
        return
    helper = ROOT / "scripts/run_visual_coverage_job.py"
    if sha(helper) != registration["shutdown_helper_sha256"]:
        raise ValueError("Shutdown helper source changed")
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_visual_coverage_job

    run_visual_coverage_job.JOB = job
    run_visual_coverage_job.shutdown()


def watch(job):
    reg = json.loads((job / "registration.json").read_text())
    stopped = False
    while time.time() < reg["shutdown_deadline"]:
        if (job / "shutdown-request.json").exists():
            return
        if (
            time.time() >= reg["deadline"]
            and not stopped
            and not (job / "worker-exited.json").exists()
        ):
            active = job / "active-child.json"
            if active.exists():
                try:
                    child = json.loads(active.read_text())
                    terminate(child["pid"], child["ticks"])
                except json.JSONDecodeError:
                    pass
            if alive(reg["supervisor_pid"], reg["supervisor_ticks"]):
                os.kill(reg["supervisor_pid"], signal.SIGTERM)
            stopped = True
        time.sleep(min(5, max(0.1, reg["shutdown_deadline"] - time.time())))
    shutdown(job, reg)


def supervise(template, template_sha, *, execute_authorized, shutdown_authorized):
    if not execute_authorized or not shutdown_authorized:
        raise PermissionError("Iris execution and protected shutdown must be authorized")
    validate_template(template)
    profile = os.environ.get("ROSETTA_AUTODL_RUNTIME_PROFILE")
    if (
        os.environ.get("ROSETTA_TORCH_DEVICE") != "cuda"
        or not profile
        or sha(profile) != template["sources"]["configs/runtime/autodl_rtx4090.yaml"]
        or any(os.environ.get(k) != "1" for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"))
    ):
        raise ValueError("Use the registered offline AutoDL shell")
    checkpoint_root = Path(os.environ["ROSETTA_CHECKPOINT_ROOT"]).resolve(strict=True)
    job = ROOT / "runs" / RUN
    job.mkdir(parents=True, exist_ok=False)
    (job / "plans").mkdir()
    started = time.time()
    reg = {
        **template,
        "template_sha256": template_sha,
        "execution_authorized": True,
        "shutdown_authorized": True,
        "started": started,
        "deadline": started + 3600,
        "shutdown_deadline": started + 4200,
        "supervisor_pid": os.getpid(),
        "supervisor_ticks": ticks(os.getpid()),
        "nested_docker_used": False,
    }
    save(job / "registration.json", reg)
    with (job / "watchdog.log").open("xb") as log:
        watchdog = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "watchdog"],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    save(job / "watchdog.json", {"pid": watchdog.pid, "ticks": ticks(watchdog.pid)})
    child = None
    identity = None
    stage = "bind-inputs"
    completed, error = [], None

    def deadline_signal(_signal, _frame):
        raise TimeoutError("Independent watchdog requested stage termination")

    signal.signal(signal.SIGTERM, deadline_signal)
    try:
        if shutil.disk_usage(checkpoint_root).free < template["required_free_bytes"]:
            raise OSError("Full Iris checkpoint/export/transfer budget does not fit")
        if subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        ).strip():
            raise RuntimeError("GPU is already occupied")
        bind_inputs()
        for stage in STAGES:
            if time.time() >= reg["deadline"]:
                raise TimeoutError("Iris work deadline reached")
            with (job / (stage + ".log")).open("xb") as log:
                child = subprocess.Popen(
                    [sys.executable, "scripts/iris_stage.py", "--job", str(job), "--stage", stage],
                    cwd=ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                identity = {"pid": child.pid, "ticks": ticks(child.pid), "stage": stage}
                temporary = job / "active-child.tmp"
                temporary.write_text(json.dumps(identity))
                temporary.replace(job / "active-child.json")
                stage_start = time.time()
                peak_rss = 0
                while child.poll() is None:
                    if time.time() >= reg["deadline"]:
                        raise TimeoutError("Iris stage exceeded shared deadline")
                    peak_rss = max(peak_rss, process_tree_rss(child.pid))
                    if peak_rss > 8 * 1024**3:
                        raise MemoryError("Iris process tree RSS exceeded 8GiB")
                    time.sleep(1)
                code = child.returncode
                if code != 0:
                    raise RuntimeError(f"Iris stage {stage} exited {code}")
                save(
                    job / (stage + ".done.json"),
                    {
                        "stage": stage,
                        "exit_code": code,
                        "seconds": time.time() - stage_start,
                        "peak_process_tree_rss": peak_rss,
                        "log_sha256": sha(job / (stage + ".log")),
                    },
                )
                completed.append(stage)
                print(json.dumps({"stage_completed": stage}), flush=True)
    except BaseException as exc:
        error = {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
        if child is not None and identity is not None and child.poll() is None:
            terminate(child.pid, identity["ticks"])
            child.wait(timeout=10)
    finally:
        save(
            job / "worker-exited.json",
            {
                "completed_stages": completed,
                "error": error,
                "seconds": time.time() - started,
                "m2_complete": False,
            },
        )
        from scripts.iris_delivery import package

        manifest_sha = package(job)
        stop = min(reg["shutdown_deadline"], time.time() + 900)
        while time.time() < stop:
            receipt = job / "transfer-receipt.json"
            if receipt.is_file():
                try:
                    value = json.loads(receipt.read_text())
                    if (
                        value.get("manifest_sha256") == manifest_sha
                        and value.get("all_file_sha256_matched") is True
                    ):
                        break
                except json.JSONDecodeError:
                    pass
            time.sleep(2)
        shutdown(job, reg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "supervise", "watchdog"))
    parser.add_argument("--template", type=Path)
    parser.add_argument("--template-sha256")
    parser.add_argument("--execute-authorized", action="store_true")
    parser.add_argument("--shutdown-authorized", action="store_true")
    args = parser.parse_args()
    if args.mode == "watchdog":
        watch(ROOT / "runs" / RUN)
    elif args.mode == "prepare":
        if args.template is None:
            raise ValueError("Explicit create-only template output required")
        save(args.template, build_template())
        print(sha(args.template))
    else:
        if args.template is None or sha(args.template) != args.template_sha256:
            raise ValueError("Explicit template SHA seal required")
        supervise(
            json.loads(args.template.read_text()),
            args.template_sha256,
            execute_authorized=args.execute_authorized,
            shutdown_authorized=args.shutdown_authorized,
        )


if __name__ == "__main__":
    main()
