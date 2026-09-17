"""Create-only Iris gate supervisor with independent shutdown protection."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.canonical_fullframes_runtime import NAME, PLAN, artifact_path, load  # noqa: E402
from scripts.iris_runtime import save, sha  # noqa: E402
from scripts.run_iris_furnace import (  # noqa: E402
    process_tree_rss,
    shutdown,
    terminate,
    ticks,
    watch,
)


def prepare_output_roots(job):
    results = job / "results"
    trackio = results / "trackio"
    trackio.mkdir(parents=True, exist_ok=False)
    artifacts = job.parent / (job.name + "-artifacts")
    artifacts.mkdir(exist_ok=False)
    os.environ["ROSETTA_RUN_ROOT"] = str(results)
    os.environ["TRACKIO_DIR"] = str(trackio)
    os.environ["ROSETTA_ARTIFACT_ROOT"] = str(artifacts)


def supervise(template, template_sha):
    for name, digest in template["sources"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Gate source seal changed")
    if (
        template["id"] != NAME
        or template["source_manifest_sha256"]
        != "66b8e8dfdb5fe33f5b1ddb03d035280fcd52570a1c331d37d7c66deed3627f96"
        or template["optimizer_steps"] != 0
        or template["work_seconds"] != 3600
        or template["shutdown_seconds"] != 4200
    ):
        raise ValueError("Gate registration differs")
    if os.environ.get("ROSETTA_TORCH_DEVICE") != "cuda" or any(
        os.environ.get(k) != "1" for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")
    ):
        raise ValueError("Offline AutoDL CUDA runtime required")
    from scripts.iris_protocol import bind_inputs

    bind_inputs()
    import shutil

    if shutil.disk_usage(os.environ["ROSETTA_AUTODL_ROOT"]).free < 1024**3:
        raise ValueError("Post-training storage below 1GiB")
    job = ROOT / "runs" / NAME
    job.mkdir(parents=True, exist_ok=False)
    prepare_output_roots(job)
    started = time.time()
    os.environ["ROSETTA_POSTTRAIN_DEADLINE"] = str(started + 3600)
    reg = {
        **template,
        "template_sha256": template_sha,
        "started": started,
        "deadline": started + 3600,
        "shutdown_deadline": started + 4200,
        "supervisor_pid": os.getpid(),
        "supervisor_ticks": ticks(os.getpid()),
        "execution_authorized": True,
        "shutdown_authorized": True,
    }
    save(job / "registration.json", reg)
    with (job / "watchdog.log").open("xb") as log:
        watchdog = subprocess.Popen(
            [sys.executable, __file__, "watchdog"],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    save(job / "watchdog.json", {"pid": watchdog.pid, "ticks": ticks(watchdog.pid)})
    child = None
    identity = None
    outcomes = {}
    error = None

    def interrupted(_sig, _frame):
        raise TimeoutError("Gate watchdog deadline")

    signal.signal(signal.SIGTERM, interrupted)

    def stage(name, args):
        nonlocal child, identity
        with (job / (name + ".log")).open("xb") as log:
            child = subprocess.Popen(
                args,
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            identity = {"pid": child.pid, "ticks": ticks(child.pid), "stage": name}
            temp = job / "active-child.tmp"
            temp.write_text(json.dumps(identity))
            temp.replace(job / "active-child.json")
            while child.poll() is None:
                if time.time() >= reg["deadline"]:
                    raise TimeoutError("Shared gate deadline reached")
                if process_tree_rss(child.pid) > 8 * 1024**3:
                    raise MemoryError("Gate process tree exceeds 8GiB")
                total = sum(p.stat().st_size for p in job.rglob("*") if p.is_file())
                if total > 128 * 1024**2:
                    raise MemoryError("Post-training outputs exceed 128MiB cap")
                if shutil.disk_usage(os.environ["ROSETTA_AUTODL_ROOT"]).free < 512 * 1024**2:
                    raise OSError("Durable safety reserve exhausted")
                time.sleep(1)
        print(json.dumps({"stage": name, "exit_code": child.returncode}), flush=True)
        return child.returncode

    try:
        if subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        ).strip():
            raise ValueError("GPU occupied by another worker")
        if stage(
            "cpu-checks",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "tests/test_canonical_posttrain_repairs.py",
                "tests/test_canonical_posttrain_roots.py",
                "tests/test_canonical_posttrain_identity.py",
                "tests/test_canonical_fullframes_posttrain.py",
                "tests/test_iris_gate.py",
                "tests/test_smolvla_reload_evidence_v2.py",
            ],
        ):
            raise ValueError("Post-training CPU regression failed")
        if stage(
            "doctor",
            [
                sys.executable,
                "scripts/autodl_doctor_cuda.py",
                "--profile",
                "configs/runtime/autodl_rtx4090.yaml",
                "--config",
                "configs/vla/smolvla_450m_aloha_insertion_action_repair_bounded_gripper_003.yaml",
            ],
        ):
            raise ValueError("CUDA doctor failed")

        def run_phase(name, argv):
            if stage(name, [sys.executable, *argv]):
                raise RuntimeError("Post-training stage failed: " + name)

        run_phase("prepare", ["-m", "scripts.canonical_fullframes_stages", "prepare"])
        for name in ("collect-first", "collect-reload", "verify-reload"):
            run_phase(name, ["-m", "scripts.canonical_fullframes_runtime", name])
        plan = load(PLAN)
        data = (
            Path(os.environ["ROSETTA_AUTODL_ROOT"])
            / "runs"
            / plan["source_experiment"]
            / "dataset_views/train-only-3e3c6b9d347e5e71"
        )
        run_phase(
            "offline",
            [
                "-m",
                "scripts.canonical_fullframes_offline",
                "--plan",
                str(PLAN),
                "--artifact",
                str(artifact_path(plan)),
                "--dataset-root",
                str(data),
                "--output",
                str(job / "offline.json"),
            ],
        )
        run_phase("render", ["-m", "scripts.canonical_fullframes_stages", "render"])
        for gate in ("gate3", "gate4"):
            args = [
                sys.executable,
                "-m",
                "scripts.canonical_fullframes_sim_gate",
                gate,
                "--plan",
                str(job / "gate.yaml"),
            ]
            output = job / "results" / plan["source_experiment"] / "gates"
            if gate == "gate4":
                args += ["--gate3-report", str(output / "gate3-smolvla-sim-471.json")]
            code = stage(gate, args)
            report = output / f"{gate}-smolvla-sim-471.json"
            if not report.is_file() or code not in (0, 1):
                raise RuntimeError("Gate runtime failure: " + gate)
            status = load(report)["status"]
            if (status == "passed") != (code == 0):
                raise ValueError("Gate exit/report mismatch")
            if not (job / (gate + "-parameter-check.json")).is_file():
                raise ValueError("Gate parameter equality proof missing")
            outcomes[gate] = status
            if gate == "gate3" and status != "passed":
                outcomes["gate4"] = "not measured: Gate 3 failed"
                break
    except BaseException as exc:  # noqa: BLE001 - preserve evidence and shut down
        error = {"type": type(exc).__name__, "message": str(exc)}
        if child is not None and identity is not None and child.poll() is None:
            terminate(child.pid, identity["ticks"])
            child.wait(timeout=10)
    finally:
        save(
            job / "worker-exited.json",
            {
                "outcomes": outcomes,
                "error": error,
                "optimizer_steps": 0,
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
                value = load(receipt)
                if (
                    value.get("manifest_sha256") == manifest_sha
                    and value.get("all_file_sha256_matched") is True
                ):
                    break
            time.sleep(2)
        shutdown(job, reg)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("supervise", "watchdog"))
    p.add_argument("--template", type=Path)
    p.add_argument("--sha256")
    p.add_argument("--execute-authorized", action="store_true")
    p.add_argument("--shutdown-authorized", action="store_true")
    args = p.parse_args()
    if args.mode == "watchdog":
        watch(ROOT / "runs" / NAME)
    else:
        if (
            not args.execute_authorized
            or not args.shutdown_authorized
            or sha(args.template) != args.sha256
        ):
            raise ValueError("Explicit sealed gate authorization required")
        supervise(load(args.template), args.sha256)


if __name__ == "__main__":
    main()
