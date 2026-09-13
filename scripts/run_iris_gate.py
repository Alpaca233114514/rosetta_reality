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
from scripts.iris_gate import ARMS, MANIFEST_SHA, NAME, load  # noqa: E402
from scripts.iris_runtime import save, sha  # noqa: E402
from scripts.run_iris_furnace import (  # noqa: E402
    build_template,
    process_tree_rss,
    shutdown,
    terminate,
    ticks,
    watch,
)


def supervise(template, template_sha):
    for name, digest in template["sources"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Gate source seal changed")
    if (
        template["id"] != NAME
        or template["source_manifest_sha256"] != MANIFEST_SHA
        or template["optimizer_steps"] != 0
        or template["work_seconds"] != 3600
        or template["shutdown_seconds"] != 4200
    ):
        raise ValueError("Gate registration differs")
    if os.environ.get("ROSETTA_TORCH_DEVICE") != "cuda" or any(
        os.environ.get(k) != "1" for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")
    ):
        raise ValueError("Offline AutoDL CUDA runtime required")
    job = ROOT / "runs" / NAME
    job.mkdir(parents=True, exist_ok=False)
    started = time.time()
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
                time.sleep(1)
        print(json.dumps({"stage": name, "exit_code": child.returncode}), flush=True)
        return child.returncode

    try:
        if subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        ).strip():
            raise ValueError("GPU occupied by another worker")
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
        if stage("render", [sys.executable, "scripts/iris_gate.py", "render", "--job", str(job)]):
            raise ValueError("Artifact acceptance/render failed")
        for arm, suffix in ARMS.items():
            outcomes[arm] = {}
            for gate in ("gate3", "gate4"):
                code = stage(
                    arm + "-" + gate,
                    [sys.executable, "scripts/iris_gate.py", gate, "--job", str(job), "--arm", arm],
                )
                report = (
                    job
                    / "results"
                    / "m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003"
                    / "gates"
                    / f"{gate}-smolvla-sim-{suffix}.json"
                )
                if not report.is_file() or code not in (0, 1):
                    raise RuntimeError("Gate runtime failed: " + arm + "-" + gate)
                status = load(report)["status"]
                if (status == "passed") != (code == 0):
                    raise ValueError("Gate exit/report mismatch")
                if not (job / (arm + "-" + gate + "-parameter-check.json")).is_file():
                    raise ValueError("Gate immutable parameter check missing")
                outcomes[arm][gate] = status
                if gate == "gate3" and status != "passed":
                    outcomes[arm]["gate4"] = "not measured: Gate 3 failed"
                    break
    except BaseException as exc:
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
    p.add_argument("mode", choices=("prepare", "supervise", "watchdog"))
    p.add_argument("--template", type=Path)
    p.add_argument("--sha256")
    p.add_argument("--execute-authorized", action="store_true")
    p.add_argument("--shutdown-authorized", action="store_true")
    args = p.parse_args()
    if args.mode == "prepare":
        t = build_template()
        t.update(id=NAME, optimizer_steps=0, source_manifest_sha256=MANIFEST_SHA)
        t.pop("maximum_optimizer_steps")
        t.pop("required_free_bytes")
        t["stages"] = [
            "doctor",
            "render",
            "control-gate3",
            "control-gate4-if-gate3-passed",
            "treatment-gate3",
            "treatment-gate4-if-gate3-passed",
        ]
        save(args.template, t)
        print(sha(args.template))
    elif args.mode == "watchdog":
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
