"""Detached bounded GPU audit supervisor with sealed sources and protected shutdown."""

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

from scripts.iris_runtime import save, sha  # noqa: E402
from scripts.run_training_chain_ssh_audit import group_rss  # noqa: E402
from scripts.training_chain_gpu_audit import NAME, STAGES  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    assert sha(args.template) == args.sha256
    template = json.loads(args.template.read_text())
    assert template["id"] == NAME and template["maximum_optimizer_steps"] == 2
    assert template["stages"] == list(STAGES)
    for name, expected in template["sources"].items():
        assert sha(ROOT / name) == expected, name
    assert shutil.disk_usage(os.environ["ROSETTA_AUTODL_ROOT"]).free >= 4 * 1024**3
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()
    assert gpu == template["gpu_uuid"]
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    )
    assert not active.strip(), "Unrelated GPU worker exists"
    job = ROOT / "runs" / NAME
    job.mkdir(parents=True, exist_ok=False)
    start = time.time()
    registration = {
        **template,
        "started": start,
        "deadline": start + 2700,
        "execution_authorized": True,
        "shutdown_authorized": True,
    }
    save(job / "registration.json", registration)
    records, error = [], None
    try:
        for stage in STAGES:
            began = time.time()
            peak = 0
            with (job / (stage + ".log")).open("x") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "scripts/training_chain_gpu_audit.py",
                        stage,
                        "--job",
                        str(job),
                    ],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                while process.poll() is None:
                    peak = max(peak, group_rss(process.pid))
                    if time.time() >= registration["deadline"] or peak > 8 * 1024**3:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        raise RuntimeError(
                            "Audit deadline or process-tree memory limit exceeded"
                        )
                    time.sleep(1)
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            row = {
                "stage": stage,
                "exit_code": process.returncode,
                "seconds": time.time() - began,
                "peak_rss": peak,
            }
            records.append(row)
            save(job / (stage + "-exit.json"), row)
            if process.returncode:
                raise RuntimeError("Stage failed: " + stage)
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    save(
        job / "worker-exited.json",
        {
            "stages": records,
            "error": error,
            "finished_unix": time.time(),
            "m2_complete": False,
        },
    )
    # Evidence inventory contains no checkpoint weights; retained smoke recovery is durable.
    roots = [job, job.parent / (NAME + "-gate")]
    files = {
        p.relative_to(ROOT).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size}
        for root in roots
        if root.exists()
        for p in root.rglob("*")
        if p.is_file()
    }
    save(
        job / "handoff-manifest.json", {"files": files, "model_weights_included": False}
    )
    until = time.time() + 300
    receipt = job / "transfer-receipt.json"
    while time.time() < until:
        if receipt.is_file():
            value = json.loads(receipt.read_text())
            if (
                value.get("manifest_sha256") == sha(job / "handoff-manifest.json")
                and value.get("verified") is True
            ):
                break
        time.sleep(5)
    from scripts import run_visual_coverage_job as shutdown_helper

    shutdown_helper.JOB = job
    shutdown_helper.shutdown()


if __name__ == "__main__":
    main()
