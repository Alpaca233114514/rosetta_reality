"""Bounded Hestia GPU preflight; no main training or automatic retry."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path

import run_visual_coverage_job as job
from prepare_visual_coverage import CONTROL, REVIEW, UPSTREAM, build_plans

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

SELF = Path(__file__).resolve()
JOB_REL = Path("runs/hestia-gpu-preflight-002")
JOB = ROOT / JOB_REL
job.SELF, job.JOB_REL, job.JOB = SELF, JOB_REL, JOB
PLAN_DOC = "reports/training/m2-smolvla-hestia-preflight-amendment-002-2026-09-10.md"
QA = ROOT / "runs/hestia-code-validation-002"
NATIVE_TRAINER_SHA = "4d15d283ea54583f552b32088db0b6c195250905ca6daf06d4670383790e2059"
CHECK_FILES = [
    "scripts/run_hestia_preflight.py",
    "scripts/evaluate_visual_fit_evidence.py",
    "src/rosetta_reality/vla/visual_fit.py",
    "src/rosetta_reality/vla/visual_coverage.py",
    "src/rosetta_reality/vla/training/observed_launch.py",
    "tests/test_smolvla_visual_fit.py",
    "tests/test_smolvla_observed_launch.py",
]
TESTS = [
    "visual_fit",
    "observed_launch",
    "training_observation",
    "visual_coverage",
    "fixed_visual_samples",
    "tracking_composition",
    "v2_error_boundaries",
]


def code_identity():
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    return {p: job.digest(ROOT / p) for p in paths if p.endswith((".py", ".yaml", ".json"))}


def verify_code():
    """CPU-only acceptance must complete before preparing a GPU job."""
    QA.mkdir(parents=True, exist_ok=False)
    identity = code_identity()
    upstream = {
        **UPSTREAM,
        "scripts/lerobot_train.py": NATIVE_TRAINER_SHA,
    }
    installed = Path(next(iter(find_spec("lerobot").submodule_search_locations)))
    for name, sha in upstream.items():
        assert job.digest(installed / name) == sha, "Native source changed: " + name
    for name, command in (
        ("ruff", [sys.executable, "-m", "ruff", "check", *CHECK_FILES]),
        (
            "regressions",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                *["tests/test_smolvla_" + x + ".py" for x in TESTS],
                "--junitxml=" + str(QA / "pytest.xml"),
            ],
        ),
    ):
        with (QA / (name + ".log")).open("x") as log:
            subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180
            )
    suite = ET.parse(QA / "pytest.xml").getroot()
    assert not any(
        int(s.attrib.get(k, 0))
        for s in suite.iter("testsuite")
        for k in ("failures", "errors", "skipped")
    )
    assert identity == code_identity(), "Source changed during CPU validation"
    job.save(
        QA / "result.json",
        {
            "status": "passed",
            "source_files": identity,
            "upstream_files": upstream,
            "tests": sum(int(s.attrib["tests"]) for s in suite.iter("testsuite")),
            "packages": {p: version(p) for p in ("torch", "lerobot", "accelerate", "numpy")},
            "smolvla_weights_loaded": False,
            "real_data_loaded": False,
            "gpu_model_execution": False,
            "m2_complete": False,
        },
    )


def check_code_validation():
    result = job.load(QA / "result.json")
    assert result["status"] == "passed" and result["source_files"] == code_identity()
    installed = Path(next(iter(find_spec("lerobot").submodule_search_locations)))
    for name, sha in result["upstream_files"].items():
        assert job.digest(installed / name) == sha, "Native source changed: " + name
    for name, expected in result["packages"].items():
        assert version(name) == expected, "Environment changed since CPU validation"


def supervise():
    registration = job.load(JOB / "registration.json")
    for name, sha in registration["job_sources"].items():
        assert job.digest(ROOT / name) == sha, name
    started = time.time()
    deadline = started + 1200
    job.save(
        JOB / "watchdog.json",
        {
            "pid": os.getpid(),
            "started_unix": started,
            "deadline_unix": deadline,
            "registration_sha256": job.digest(JOB / "registration.json"),
            "external_watchdog_verified": True,
            "shutdown_grace_seconds": 120,
        },
    )
    with (JOB / "worker.log").open("x") as log:
        process = subprocess.Popen(
            [sys.executable, str(SELF), "worker"],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        job.save(
            JOB / "worker-process.json",
            {"pid": process.pid, "pgid": process.pid, "supervisor_pid": os.getpid()},
        )
        try:
            code = process.wait(timeout=max(1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            job.terminate_group(process)
            code = 124
        # A failed worker may have left a subprocess; this group belongs only to this job.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    job.save(
        JOB / "supervisor-exit.json",
        {"time": time.time(), "worker_exit_code": code, "m2_complete": False},
    )
    os.sync()
    time.sleep(120)  # Bounded grace for small evidence collection; independent of SSH.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        job.shutdown()
    except BaseException as error:
        job.save(
            JOB / "shutdown-failed.json",
            {"error_type": type(error).__name__, "instance_release_requested": False},
        )
        raise


def prepare():
    """Seal source, original inputs and three unique plans before execution."""
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise ValueError("A clean immutable source checkout is required")
    check_code_validation()
    control = job.load(ROOT / CONTROL)
    for name, sha in control["implementation_files"].items():
        assert job.digest(ROOT / name) == sha, name
    JOB.mkdir(parents=True, exist_ok=False)
    durable = Path(os.environ["ROSETTA_AUTODL_ROOT"])
    for entry in control["prerequisites"].values():
        target = ROOT / entry["path"]
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write((durable / entry["path"]).read_bytes())
        assert job.digest(target) == entry["sha256"], entry["path"]
    norm = control["normalization"]
    target = ROOT / norm["report"]
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write((durable / norm["report"]).read_bytes())
    assert job.digest(target) == norm["report_sha256"]
    view = ROOT / "runs" / job.EXP / "dataset_views"
    view.parent.mkdir(parents=True, exist_ok=True)
    view.symlink_to(durable / "runs" / job.EXP / "dataset_views", target_is_directory=True)
    assert job.digest(ROOT / norm["dataset_view_manifest"]) == norm["dataset_view_manifest_sha256"]
    incoming = durable / "control/coverage40-unattended-20260910-001/incoming"
    # Small audit scripts/evidence only; no model or dataset copies.
    shutil.copytree(incoming / "visual-native-small-001", ROOT / "runs/visual-native-small-001")
    stages = build_plans(control, job.load(ROOT / REVIEW))
    for stage in ("preflight-b1", "preflight-b4", "smoke2"):
        plan = stages[stage]
        name = "m2-smolvla450m-visual-hestia-" + stage + "-002"
        plan.update(status="preregistered", plan_id=name, run_name=name)
        plan["hypothesis"] = (
            "Verify native Hestia inputs and observed two-step update/reload "
            "before the fit-strength diagnostic."
        )
        plan["training"].update(steps=1280, save_freq=320, checkpoint_steps=[320, 640, 960, 1280])
        plan["training"]["scheduler"]["num_decay_steps"] = 1280
        plan["optimizer_smoke"]["run_name"] = name
        plan["preflight"]["run_name"] = name + "-forward"
        job.save_stage_plan(JOB / "plans" / (stage + ".yaml"), plan)
    free = shutil.disk_usage(durable).free
    required = 11831765228
    assert free >= required, "Candidate checkpoint retention budget no longer fits"
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid,memory.total", "--format=csv,noheader"], text=True
    ).strip()
    assert "RTX 4090" in gpu
    assert not subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip(), "GPU already in use"
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    sources = {p: job.digest(ROOT / p) for p in tracked if p.endswith((".py", ".yaml", ".json"))}
    sources[PLAN_DOC] = job.digest(ROOT / PLAN_DOC)
    # Also bind the inherited supervisor/resource/shutdown helpers.
    job.save(
        JOB / "registration.json",
        {
            "schema_version": 1,
            "scope": "GPU preflight and exactly two optimizer updates",
            "prior_cpu_validation_sha256": job.digest(QA / "result.json"),
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "job_sources": sources,
            "stage_plans": {p.name: job.digest(p) for p in (JOB / "plans").glob("*.yaml")},
            "upstream_files": UPSTREAM,
            "gpu": gpu,
            "compute_seconds": 1200,
            "disk_free_bytes": free,
            "disk_required_bytes": required,
            "main_training_started": False,
            "main_training_steps": 0,
            "shutdown_authorized": True,
            "no_retry": True,
            "hidden_test_loaded": False,
            "nested_docker_used": False,
            "m2_complete": False,
        },
    )


def observed_smoke():
    import run_smolvla_v2
    import torch
    from lerobot.scripts import lerobot_train

    from rosetta_reality.vla.training.observed_launch import run_observed_launch

    watchdog = job.load(JOB / "watchdog.json")
    assert time.time() < watchdog["deadline_unix"]
    os.kill(watchdog["pid"], 0)
    registration = job.load(JOB / "registration.json")
    for name, sha in registration["job_sources"].items():
        assert job.digest(ROOT / name) == sha, name
    plan_path = JOB / "plans/smoke2.yaml"
    assert job.digest(plan_path) == registration["stage_plans"]["smoke2.yaml"]
    plan = job.load(plan_path)
    assert plan["optimizer_smoke"]["steps"] == 2
    schedule = job.load(JOB / "sampler.json")["arms"]["B"]["schedule"][:8]
    original = torch.optim.AdamW.step

    def finite_step(optimizer, *args, **kwargs):
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                    raise FloatingPointError("Nonfinite optimizer gradient")
        return original(optimizer, *args, **kwargs)

    torch.optim.AdamW.step = finite_step
    sys.argv = ["scripts/run_smolvla_v2.py", "smoke", "--plan", str(plan_path)]
    try:
        run_observed_launch(
            lerobot_train,
            run_smolvla_v2.main,
            expected_samples=[(ep, 0) for ep in schedule],
            batch_size=4,
            output=JOB / "observation",
        )
    finally:
        torch.optim.AdamW.step = original


def worker():
    import numpy as np

    watchdog = job.load(JOB / "watchdog.json")
    deadline = watchdog["deadline_unix"]
    os.kill(watchdog["pid"], 0)
    stage = "startup"

    def run(name, argv, *, heavy=False, env=None, minimum_remaining=20):
        nonlocal stage
        stage = name
        assert deadline - time.time() >= minimum_remaining, "Insufficient time for " + name
        job.event("stage_started", stage=name)
        if heavy:
            argv = [sys.executable, str(SELF), "wrap", str(JOB / (name + "-resources.json")), *argv]
        with (JOB / (name + ".log")).open("x") as stream:
            subprocess.run(
                argv,
                cwd=ROOT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=max(1, deadline - time.time()),
            )
        job.event("stage_passed", stage=name)

    try:
        check_code_validation()
        assert (
            job.digest(QA / "result.json")
            == job.load(JOB / "registration.json")["prior_cpu_validation_sha256"]
        )
        job.event("prior_code_validation_verified")
        run("check-env", [sys.executable, "scripts/check_env.py"])
        isolated_root = JOB / "environment-runs"
        (isolated_root / "trackio").mkdir(parents=True)
        isolated = dict(
            os.environ,
            ROSETTA_RUN_ROOT=str(isolated_root),
            TRACKIO_DIR=str(isolated_root / "trackio"),
        )
        parent = job.load(JOB / "plans/smoke2.yaml")["parent_experiment"]["config"]
        run(
            "doctor",
            [
                sys.executable,
                "scripts/autodl_doctor_cuda.py",
                "--profile",
                "configs/runtime/autodl_rtx4090.yaml",
                "--config",
                parent,
            ],
            env=isolated,
        )
        run(
            "data-check",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-m",
                "data",
                "tests/test_smolvla_visual_grounding_data.py",
            ],
        )
        run(
            "benchmark",
            [sys.executable, "scripts/benchmark_smolvla.py", "--config", parent],
            env=isolated,
        )
        for name in ("preflight-b1", "preflight-b4"):
            run(
                name,
                [
                    "scripts/run_smolvla_v2.py",
                    "preflight",
                    "--plan",
                    str(JOB / "plans" / (name + ".yaml")),
                ],
                heavy=True,
            )
        run(
            "sampler",
            [
                sys.executable,
                "scripts/inspect_visual_coverage_sampler.py",
                "--output",
                str(JOB / "sampler.json"),
            ],
        )
        run(
            "sample-contract",
            [
                "scripts/visual_coverage_job_checks.py",
                "samples",
                CONTROL,
                str(JOB / "sample-contract.json"),
            ],
            heavy=True,
        )
        run("smoke2", [str(SELF), "observed-smoke"], heavy=True, minimum_remaining=360)
        observation = job.load(JOB / "observation/result.json")
        assert observation["status"] == "passed"
        smoke_plan = str(JOB / "plans/smoke2.yaml")
        run(
            "smoke-update-audit",
            [
                sys.executable,
                "runs/visual-native-small-001/check-checkpoint-002.py",
                smoke_plan,
                str(JOB / "smoke-update-audit.json"),
            ],
        )
        for suffix in ("first", "reload"):
            run(
                "smoke-" + suffix,
                [
                    "scripts/visual_coverage_job_checks.py",
                    "predict",
                    smoke_plan,
                    str(JOB / ("smoke-" + suffix + ".npz")),
                ],
                heavy=True,
            )
        with (
            np.load(JOB / "smoke-first.npz", allow_pickle=False) as first,
            np.load(JOB / "smoke-reload.npz", allow_pickle=False) as second,
        ):
            assert set(first.files) == set(second.files) == {"normalized", "standard"}
            assert all(
                first[k].dtype == second[k].dtype and np.array_equal(first[k], second[k])
                for k in first.files
            )
        assert (
            job.load(JOB / "smoke-first.npz.json")["pid"]
            != job.load(JOB / "smoke-reload.npz.json")["pid"]
        )
        job.save(
            JOB / "result.json",
            {
                "status": "passed",
                "completed_stage": "smoke-independent-reload",
                "optimizer_updates": 2,
                "full_chunk_reload_exact": True,
                "main_training_steps": 0,
                "development_visual_acceptance": "not measured",
                "hidden_test_loaded": False,
                "m2_complete": False,
            },
        )
    except BaseException as error:
        job.save(
            JOB / "failure.json",
            {
                "stage": stage,
                "error_type": type(error).__name__,
                "status": "failed",
                "main_training_steps": 0,
                "m2_complete": False,
            },
        )
        raise


if __name__ == "__main__":
    {
        "verify-code": verify_code,
        "prepare": prepare,
        "supervise": supervise,
        "worker": worker,
        "wrap": job.wrapped,
        "observed-smoke": observed_smoke,
    }[sys.argv[1]]()
