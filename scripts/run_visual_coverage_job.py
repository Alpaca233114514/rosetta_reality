"""One bounded, fail-closed AutoDL job; never retries or resumes training."""
from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import runpy
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
SELF = Path(__file__).resolve()
JOB_REL = Path("runs/visual-coverage40-unattended-001")
JOB = ROOT / JOB_REL
TEMPLATE = ROOT / "configs/vla/visual-coverage40-20260910-001/execution-contract.template.json"
CONTROL = "configs/vla/m2-smolvla450m-visual-native-b4-pilot-003.yaml"
EXP = "m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def load(path):
    return json.loads(Path(path).read_text())


def evidence(path):
    return {"path": Path(path).relative_to(ROOT).as_posix(), "sha256": digest(path), "accepted": True}


def event(kind, **fields):
    with (JOB / "events.jsonl").open("a") as stream:
        stream.write(json.dumps({"time": time.time(), "kind": kind, **fields}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def inventory(folder):
    return {p.relative_to(folder).as_posix(): digest(p) for p in sorted(folder.rglob("*")) if p.is_file()}


def prepare():
    """No model execution. Preserve originals and seal unique stage plans."""
    JOB.mkdir(parents=True, exist_ok=False)
    durable = Path(os.environ["ROSETTA_AUTODL_ROOT"])
    incoming = durable / "control/coverage40-unattended-20260910-001/incoming"
    template = load(TEMPLATE)
    for name, sha in template["implementation_files"].items():
        assert digest(ROOT / name) == sha, name
    expected = load(ROOT / "reports/training/m2-smolvla-native-visual-small-2026-09-09.json")
    for name, sha in expected["evidence_sha256"].items():
        assert digest(incoming / "visual-native-small-001" / name) == sha, name
    assert digest(incoming / "000256/pretrained_model/model.safetensors") == template["arms"]["A"]["checkpoint_files"]["model.safetensors"]
    cp = Path(os.environ["ROSETTA_CHECKPOINT_ROOT"]) / Path(template["arms"]["A"]["checkpoint_relative_to_root"]).parent
    assert not cp.exists() and not cp.is_symlink()
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.symlink_to(incoming / "000256", target_is_directory=True)
    shutil.copytree(incoming / "visual-native-small-001", ROOT / "runs/visual-native-small-001")
    for stage, item in template["stages"].items():
        assert digest(ROOT / item["path"]) == item["sha256"]
        plan = load(ROOT / item["path"])
        plan["status"] = "preregistered"
        for entry in plan["prerequisites"].values():
            target = ROOT / entry["path"]
            source = durable / entry["path"]
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(source.read_bytes())
            assert digest(target) == entry["sha256"]
        norm = plan["normalization"]
        target = ROOT / norm["report"]
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write((durable / norm["report"]).read_bytes())
        assert digest(target) == norm["report_sha256"]
        link = ROOT / "runs" / EXP / "dataset_views"
        if not link.exists():
            link.symlink_to(durable / "runs" / EXP / "dataset_views", target_is_directory=True)
        assert digest(ROOT / norm["dataset_view_manifest"]) == norm["dataset_view_manifest_sha256"]
        save(JOB / "plans" / f"{stage}.json", plan)
    size = sum(p.stat().st_size for p in (incoming / "000256").rglob("*") if p.is_file())
    needed = size * 3 + 2 * 1024**3
    free = shutil.disk_usage(durable).free
    assert free >= needed, "Insufficient space for smoke, main, temporary save and 2 GiB reserve"
    manifest = {"status": "preregistered", "job": JOB_REL.as_posix(), "compute_seconds": 1800,
                "shutdown_grace_seconds": 120, "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "job_sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in (SELF, ROOT / "scripts/visual_coverage_job_checks.py")},
                "stage_plans": {p.name: digest(p) for p in (JOB / "plans").glob("*.json")},
                "review_plan_sha256": template["review_plan"]["sha256"], "disk_free_bytes": free, "disk_required_bytes": needed,
                "A_checkpoint_files": inventory(incoming / "000256"), "shutdown_authorized": True,
                "authorization": "User authorized task-based computation, unattended server execution, a task-length watchdog and shutdown afterward.",
                "no_retry": True, "B_fresh_base_only": True, "hidden_test_loaded": False, "m2_complete": False,
                "stage_order": template["stage_order"], "order_note": "A input/sampler contract precedes smoke; A new train8 fit is checked before B."}
    save(JOB / "registration.json", manifest)


def wrapped():
    """Observe memory, actual optimizer LR and sampled indices without changing RNG."""
    import torch

    report, script, *args = sys.argv[2:]
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, 10 * 1024**3 / total))
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    stop = threading.Event()
    reasons, learning_rates, sampled_indices = [], [], []

    def handle_term(_signum, _frame):
        raise RuntimeError("Resource/deadline stop: " + "; ".join(reasons))

    signal.signal(signal.SIGTERM, handle_term)

    def monitor():
        while not stop.wait(1):
            if (torch.cuda.max_memory_allocated() > 8 * 1024**3
                    or torch.cuda.max_memory_reserved() > 10 * 1024**3
                    or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 10 * 1024**3):
                reasons.append("registered memory cap exceeded")
                os.kill(os.getpid(), signal.SIGTERM)
                return

    if Path(script).name == "run_smolvla_v2.py" and args[0] == "smoke":
        from rosetta_reality.vla.training import features

        original_step = torch.optim.AdamW.step

        def step(optimizer, *pos, **kw):
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError("Nonfinite optimizer gradient")
            learning_rates.append([group["lr"] for group in optimizer.param_groups])
            return original_step(optimizer, *pos, **kw)

        torch.optim.AdamW.step = step
        original_install = features.FixedFrameSamplerFeature.install

        def install(feature, context):
            original_install(feature, context)
            sampler = features._lerobot_train_module().EpisodeAwareSampler
            original_iter = sampler.__iter__

            def iterate(instance):
                for index in original_iter(instance):
                    sampled_indices.append(int(index))
                    yield index

            sampler.__iter__ = iterate

        features.FixedFrameSamplerFeature.install = install
    threading.Thread(target=monitor, daemon=True).start()
    sys.argv = [script, *args]
    passed = False
    try:
        try:
            runpy.run_path(script, run_name="__main__")
        except SystemExit as error:
            if error.code not in (None, 0):
                raise
        passed = True
    finally:
        stop.set()
        result = {"status": "passed" if passed and not reasons else "failed", "elapsed_seconds": time.time() - start,
                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  "stop_reasons": reasons, "learning_rates": learning_rates, "sampled_indices": sampled_indices}
        save(report, result)
        assert result["peak_cuda_allocated_bytes"] <= 8 * 1024**3
        assert result["peak_cuda_reserved_bytes"] <= 10 * 1024**3
        assert result["peak_host_rss_bytes"] <= 10 * 1024**3


def terminate_group(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def shutdown():
    command = Path("/usr/bin/shutdown")
    expected = "0358e83eeeaf542aa98f64ba9e339c91df46f1e025892d52dba159f4fb1cf027"
    assert digest(command) == expected, "Platform shutdown wrapper changed"
    trash = Path("/root/.local/share/Trash")
    assert not trash.exists() and not trash.is_symlink(), "Shutdown would delete existing Trash"
    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], capture_output=True, text=True, timeout=10, check=False)
    assert gpu.returncode == 0 and not gpu.stdout.strip(), "GPU worker remains active"
    sessions = subprocess.run(["tmux", "list-sessions", "-F", "#{session_name}"], capture_output=True, text=True, timeout=10, check=False)
    assert not sessions.stdout.strip() and sessions.returncode in (0, 1), "Unrelated tmux session present"
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        assert not (any(b"rosetta" in x for x in argv) and any(any(t in x for t in (b"train_", b"evaluate_", b"run_smolvla", b"diagnose_", b"pytest", b"visual_coverage_job_checks")) for x in argv)), "Unrelated project worker present"
    save(JOB / "shutdown-request.json", {"time": time.time(), "shutdown_wrapper_sha256": expected,
                                        "instance_release_requested": False, "platform_billing_independently_verified": False})
    os.sync()
    assert not trash.exists() and not trash.is_symlink()
    os.execv("/bin/bash", ["bash", str(command)])


def supervise():
    # Detached supervisor is independent of SSH and owns only this child's group.
    registration = load(JOB / "registration.json")
    for name, sha in registration["job_sources"].items():
        assert digest(ROOT / name) == sha
    started = time.time()
    save(JOB / "watchdog.json", {"pid": os.getpid(), "started_unix": started,
                                "deadline_unix": started + 1800, "registration_sha256": digest(JOB / "registration.json"),
                                "external_watchdog_verified": True, "shutdown_grace_seconds": 120})
    with (JOB / "worker.log").open("x") as log:
        process = subprocess.Popen([sys.executable, str(SELF), "worker"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save(JOB / "worker-process.json", {"pid": process.pid, "pgid": process.pid, "supervisor_pid": os.getpid()})
        try:
            code = process.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            event("shared_deadline_reached")
            terminate_group(process)
            code = 124
        # A failed child can leave descendants; signal only its dedicated group.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    save(JOB / "supervisor-exit.json", {"time": time.time(), "worker_exit_code": code, "m2_complete": False})
    # Give only the owned worker group a bounded chance to release GPU handles.
    for _ in range(20):
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(1)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        time.sleep(2)
    os.sync()
    try:
        shutdown()
    except Exception:
        save(JOB / "shutdown-blocked.json", {"error": traceback.format_exc(), "time": time.time()})
        raise


def worker():
    import numpy as np

    watchdog = load(JOB / "watchdog.json")
    os.kill(watchdog["pid"], 0)
    deadline = watchdog["deadline_unix"]
    template = load(TEMPLATE)
    template.update(status="authorized_sealed", model_execution_authorized=True,
                    started_unix=watchdog["started_unix"], deadline_unix=deadline, external_watchdog_verified=True)
    template["implementation_files"].update(load(JOB / "registration.json")["job_sources"])
    prerequisites = {}
    stage = "startup"

    def run(name, argv, *, heavy=False, env=None, minimum_remaining=30):
        nonlocal stage
        stage = name
        if deadline - time.time() < minimum_remaining:
            raise TimeoutError("Insufficient registered time remaining for " + name)
        event("stage_started", stage=name, argv=argv)
        if heavy:
            argv = [sys.executable, str(SELF), "wrap", str(JOB / f"{name}-resources.json"), *argv]
        with (JOB / f"{name}.log").open("x") as stream:
            subprocess.run(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=max(1, deadline - time.time()))
        event("stage_passed", stage=name)

    def accept(name, value):
        path = JOB / f"accepted-{name}.json"
        save(path, value)
        prerequisites[name] = evidence(path)

    def checkpoint(stage_name):
        plan = load(JOB / "plans" / f"{stage_name}.json")
        smoke = plan["optimizer_smoke"]
        return Path(os.environ["ROSETTA_CHECKPOINT_ROOT"]) / EXP / "smoke" / smoke["run_name"] / "checkpoints" / f"{smoke['steps']:06d}"

    def seal(name, include_b=False):
        contract = json.loads(json.dumps(template))
        contract["prerequisite_evidence"] = dict(prerequisites)
        for arm in ("A", "B") if include_b else ("A",):
            identity = contract["arms"][arm]
            if arm == "B":
                plan = JOB / "plans/main256.json"
                identity["plan"] = {"path": plan.relative_to(ROOT).as_posix(), "sha256": digest(plan)}
            source = Path(os.environ["ROSETTA_CHECKPOINT_ROOT"]) / identity["checkpoint_relative_to_root"]
            identity["checkpoint_files"] = inventory(source)
        path = JOB / name
        save(path, contract)
        return path

    def native_a(name):
        run(name, ["scripts/evaluate_visual_native_small.py", "--plan", CONTROL, "--source", "checkpoint", "--output", str(JOB / f"{name}.json")], heavy=True)

    try:
        run("check-env", [sys.executable, "scripts/check_env.py"])
        isolated = dict(os.environ, ROSETTA_RUN_ROOT=str(JOB / "environment-runs"))
        run("doctor", [sys.executable, "scripts/autodl_doctor_cuda.py", "--profile", "configs/runtime/autodl_rtx4090.yaml", "--config", load(JOB / "plans/main256.json")["parent_experiment"]["config"]], env=isolated)
        run("static-tests", [sys.executable, "-m", "pytest", "-q", "tests/test_smolvla_visual_coverage.py", "tests/test_smolvla_fixed_visual_samples.py", "tests/test_smolvla_tracking_composition.py", "tests/test_smolvla_v2_error_boundaries.py"])
        run("data-check", [sys.executable, "-m", "pytest", "-q", "-m", "data", "tests/test_smolvla_visual_grounding_data.py"])
        accept("environment", {"status": "passed", "doctor_log_sha256": digest(JOB / "doctor.log"), "test_logs": {n: digest(JOB / (n + ".log")) for n in ("static-tests", "data-check")}, "nested_docker_used": False})
        parent = load(JOB / "plans/main256.json")["parent_experiment"]["config"]
        run("benchmark", [sys.executable, "scripts/benchmark_smolvla.py", "--config", parent], env=isolated)
        for name in ("preflight-b1", "preflight-b4"):
            run(name, ["scripts/run_smolvla_v2.py", "preflight", "--plan", str(JOB / "plans" / f"{name}.json")], heavy=True)
            report = load(JOB / (name + "-resources.json"))
            assert report["status"] == "passed" and not report["learning_rates"]
        accept("resource_preflight", {"status": "passed", "reports": {name: evidence(JOB / f"{name}-resources.json") for name in ("preflight-b1", "preflight-b4")}})
        run("sampler", [sys.executable, "scripts/inspect_visual_coverage_sampler.py", "--output", str(JOB / "sampler.json")])
        run("sample-contract", ["scripts/visual_coverage_job_checks.py", "samples", CONTROL, str(JOB / "sample-contract.json")], heavy=True)
        accept("sample_contract", {"status": "passed", "samples": evidence(JOB / "sample-contract.json"), "schedule": evidence(JOB / "sampler.json")})
        smoke_plan = str(JOB / "plans/smoke2.json")
        run("smoke2", ["scripts/run_smolvla_v2.py", "smoke", "--plan", smoke_plan], heavy=True, minimum_remaining=1200)
        old_check = "runs/visual-native-small-001/check-checkpoint-002.py"
        run("smoke-update-audit", [sys.executable, old_check, smoke_plan, str(JOB / "smoke-update-audit.json")])
        for suffix in ("first", "reload"):
            run("smoke-" + suffix, ["scripts/visual_coverage_job_checks.py", "predict", smoke_plan, str(JOB / f"smoke-{suffix}.npz")], heavy=True)
        with np.load(JOB / "smoke-first.npz") as first, np.load(JOB / "smoke-reload.npz") as second:
            assert set(first.files) == set(second.files) == {"normalized", "standard"}
            assert all(np.array_equal(first[key], second[key]) for key in first.files)
        assert load(JOB / "smoke-first.npz.json")["pid"] != load(JOB / "smoke-reload.npz.json")["pid"]
        accept("two_step_smoke_reload", {"status": "passed", "full_chunk_exact": True, "smoke_audit": evidence(JOB / "smoke-update-audit.json"), "predictions_sha256": [digest(JOB / f"smoke-{s}.npz") for s in ("first", "reload")]})
        native_a("control-reproduction")
        reproduced, historical = load(JOB / "control-reproduction.json"), load(ROOT / "runs/visual-native-small-001/pilot-eval-003.json")
        for key in ("source_sha256", "episodes", "image_sha256", "results", "native_denoising_steps"):
            assert reproduced[key] == historical[key], "Historical control reproduction differs: " + key
        accept("control_reproduction", {"status": "passed", "exact_saved_values": True, "reproduction": evidence(JOB / "control-reproduction.json")})
        before = seal("A-before-B-contract.json")
        run("A-fit", ["scripts/evaluate_visual_coverage.py", "collect", "--contract", str(before), "--arm", "A", "--output", str(JOB / "A-fit")], heavy=True)
        assert load(JOB / "A-fit/metrics.json")["training_fit_passed"], "A train8 prerequisite failed"
        main_plan = str(JOB / "plans/main256.json")
        run("main256", ["scripts/run_smolvla_v2.py", "smoke", "--plan", main_plan], heavy=True, minimum_remaining=720)
        run("B-update-audit", [sys.executable, old_check, main_plan, str(JOB / "B-update-audit.json")])
        cp = checkpoint("main256")
        state = load(cp / "training_state/scheduler_state.json")
        assert state["last_epoch"] == 256 and state["_step_count"] == 257 and state["_last_lr"] == [2.5e-6]
        resources = load(JOB / "main256-resources.json")
        assert resources["status"] == "passed" and len(resources["learning_rates"]) == 256
        for index, values in enumerate(resources["learning_rates"]):
            factor = ((1 / 17 - 1) * (1 - index / 16) + 1) if index < 16 else (0.975 * 0.5 * (1 + math.cos(math.pi * index / 256)) + 0.025)
            assert len(values) == 1 and math.isclose(values[0], 1e-4 * factor, rel_tol=1e-12, abs_tol=1e-16)
        samples, schedule = load(JOB / "sample-contract.json"), load(JOB / "sampler.json")["arms"]["B"]["schedule"]
        mapping = dict(zip(samples["actual_train_episodes"], samples["actual_train_view_indices"], strict=True))
        assert resources["sampled_indices"] == [mapping[episode] for episode in schedule], "Actual 1024 sample order differs"
        metric = load(cp / "rosetta_checkpoint_metrics.json")
        assert metric["step"] == 256 and all(math.isfinite(v) for v in metric["metrics"].values())
        saved = load(cp / "pretrained_model/train_config.json")
        control_cp = Path(os.environ["ROSETTA_CHECKPOINT_ROOT"]) / template["arms"]["A"]["checkpoint_relative_to_root"]
        old_saved = load(control_cp / "train_config.json")
        for key in ("optimizer", "scheduler", "seed", "steps", "batch_size"):
            assert saved[key] == old_saved[key], "Saved training recipe differs: " + key
        assert saved["dataset"]["episodes"] == samples["actual_train_episodes"]
        accept("B_training_integrity", {"status": "passed", "update_audit": evidence(JOB / "B-update-audit.json"), "actual_optimizer_and_sampler": evidence(JOB / "main256-resources.json"), "checkpoint_files": inventory(cp)})
        final_contract = seal("execution-contract.json", include_b=True)
        for arm in ("A", "B"):
            for suffix in ("first", "reload"):
                run(f"{arm}-{suffix}", ["scripts/evaluate_visual_coverage.py", "collect", "--contract", str(final_contract), "--arm", arm, "--output", str(JOB / f"{arm}-{suffix}")], heavy=True)
            run(f"{arm}-reload-check", [sys.executable, "scripts/evaluate_visual_coverage.py", "reload", "--first", str(JOB / f"{arm}-first"), "--second", str(JOB / f"{arm}-reload"), "--output", str(JOB / f"{arm}-reload-check.json")])
        run("compare-arms", [sys.executable, "scripts/evaluate_visual_coverage.py", "compare", "--first", str(JOB / "A-first"), "--second", str(JOB / "B-first"), "--output", str(JOB / "comparison.json")])
        save(JOB / "closure.json", {"status": "completed", "last_stage": stage, "comparison": load(JOB / "comparison.json"), "m2_complete": False, "next_experiment_started": False})
    except BaseException:
        error = traceback.format_exc()
        event("failed", stage=stage, error=error)
        save(JOB / "closure.json", {"status": "stopped", "failed_stage": stage, "error": error, "later_stages": "not measured", "m2_complete": False, "next_experiment_started": False})
        raise
    finally:
        os.sync()


if __name__ == "__main__":
    {"prepare": prepare, "supervise": supervise, "worker": worker, "wrap": wrapped}[sys.argv[1]]()
