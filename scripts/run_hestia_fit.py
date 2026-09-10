"""One bounded Hestia main experiment, with native training and no automatic retry."""

from __future__ import annotations

import argparse
import copy
import os
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import run_visual_coverage_job as io  # noqa: E402

from rosetta_reality.vla import visual_fit as fit  # noqa: E402
from rosetta_reality.vla import visual_fit_contract as checks  # noqa: E402
from rosetta_reality.vla import visual_fit_job as gates  # noqa: E402

SELF = Path(__file__).resolve()
JOB = ROOT / gates.JOB_REL
io.SELF, io.JOB_REL, io.JOB = SELF, gates.JOB_REL, JOB


def source_identity():
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    return {name: io.digest(ROOT / name) for name in names if name}


def stage_inputs():
    """Restore only small registered evidence and a checked durable dataset-view link."""
    durable = Path(os.environ["ROSETTA_AUTODL_ROOT"])
    history = io.load(ROOT / checks.HISTORY)
    control = io.load(ROOT / "configs/vla/m2-smolvla450m-visual-native-b4-pilot-003.yaml")
    refs = [(entry["path"], entry["sha256"]) for entry in control["prerequisites"].values()]
    norm = control["normalization"]
    refs.append((norm["report"], norm["report_sha256"]))
    for name, sha in refs:
        path = checks.relative_file(ROOT, name)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            source = checks.relative_file(durable, name)
            if io.digest(source) != sha:
                raise ValueError("Durable prerequisite hash changed: " + name)
            with path.open("xb") as stream:
                stream.write(source.read_bytes())
        if io.digest(path) != sha:
            raise ValueError("Prerequisite hash changed: " + name)
    b = history["checkpoint_identities"]["B"]
    target = checks.relative_file(ROOT, b["plan"]["path"])
    original_workspace = Path(gates.HISTORICAL_B).parts[:2]
    source = durable.joinpath(*original_workspace) / b["plan"]["path"]
    if io.digest(source) != b["plan"]["sha256"]:
        raise ValueError("Historical B plan changed")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(source.read_bytes())
    if io.digest(target) != b["plan"]["sha256"]:
        raise ValueError("Staged B plan differs")
    view = ROOT / "runs" / io.EXP / "dataset_views"
    actual = durable / "runs" / io.EXP / "dataset_views"
    if not view.exists() and not view.is_symlink():
        view.parent.mkdir(parents=True, exist_ok=True)
        view.symlink_to(actual, target_is_directory=True)
    if view.resolve() != actual.resolve():
        raise ValueError("Dataset view link differs from registered durable storage")
    if io.digest(ROOT / norm["dataset_view_manifest"]) != norm["dataset_view_manifest_sha256"]:
        raise ValueError("Dataset view manifest changed")


def supervise(execute_authorized: bool, shutdown_authorized: bool):
    if not execute_authorized or not shutdown_authorized:
        raise ValueError("Main execution and shutdown must both be explicitly authorized")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise ValueError("Use a clean immutable source checkout")
    for key in ("ROSETTA_AUTODL_ROOT", "ROSETTA_CHECKPOINT_ROOT", "ROSETTA_AUTODL_RUNTIME_PROFILE"):
        if not os.environ.get(key):
            raise ValueError("Use the registered AutoDL runner")
    JOB.mkdir(parents=True, exist_ok=False)
    io.save(JOB / "registration.json", {
        "status": "authorized_worker", "run_name": fit.RUN_NAMES["C"],
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_files": source_identity(), "resource_limits": gates.LIMITS,
        "model_execution_authorized": True, "shutdown_authorized": True,
        "compute_seconds": 1800, "no_retry": True, "instance_release_requested": False,
        "hidden_test_loaded": False, "nested_docker_used": False, "m2_complete": False,
    })
    start = time.time()
    io.save(JOB / "watchdog.json", {
        "pid": os.getpid(), "process_start_ticks": checks.process_start_ticks(os.getpid()),
        "started_unix": start, "deadline_unix": start + 1800,
        "registration_sha256": io.digest(JOB / "registration.json"),
        "external_watchdog_verified": True, "shutdown_grace_seconds": 120,
    })
    with (JOB / "worker.log").open("x") as stream:
        process = subprocess.Popen(
            [sys.executable, str(SELF), "worker"], cwd=ROOT, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
        io.save(JOB / "worker-process.json", {"pid": process.pid, "pgid": process.pid})
        try:
            code = process.wait(timeout=max(1, start + 1800 - time.time()))
        except subprocess.TimeoutExpired:
            io.terminate_group(process)
            code = 124
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    io.save(JOB / "supervisor-exit.json", {
        "worker_exit_code": code, "elapsed_seconds": time.time() - start,
        "m2_complete": False,
    })
    os.sync()
    time.sleep(120)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit() or int(proc.name) == os.getpid():
                continue
            try:
                argv = (proc / "cmdline").read_bytes().split(b"\0")
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            names = {Path(os.fsdecode(arg)).name for arg in argv if arg}
            if names & {"run_hestia_fit.py", "check_hestia_collector_cpu.py"}:
                raise RuntimeError("Another Hestia worker remains active")
        io.shutdown()
    except BaseException as error:
        io.save(JOB / "shutdown-failed.json", {
            "error_type": type(error).__name__, "instance_release_requested": False,
        })
        raise


def observed_train():
    plan_path, samples = gates.authorize_training(ROOT, JOB)
    from run_smolvla_v2 import _resolve_plan

    plan, _, experiment = _resolve_plan(plan_path)
    checks.validate_plan(plan, "C")
    checks.validate_experiment(experiment)
    target = Path(os.environ["ROSETTA_CHECKPOINT_ROOT"]) / io.EXP / "smoke" / fit.RUN_NAMES["C"]
    if target.exists() or target.is_symlink():
        raise FileExistsError("Candidate output already exists; never resume or overwrite")
    import run_smolvla_v2
    import torch
    from lerobot.scripts import lerobot_train

    from rosetta_reality.vla.training.observed_launch import run_observed_launch

    original = torch.optim.AdamW.step

    def finite_step(optimizer, *args, **kwargs):
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                    raise FloatingPointError("Nonfinite main optimizer gradient")
        return original(optimizer, *args, **kwargs)

    torch.optim.AdamW.step = finite_step
    sys.argv = ["scripts/run_smolvla_v2.py", "smoke", "--plan", str(plan_path)]
    try:
        run_observed_launch(
            lerobot_train, run_smolvla_v2.main, expected_samples=samples,
            batch_size=4, output=JOB / "observation",
        )
    finally:
        torch.optim.AdamW.step = original


def worker():
    registration, watch = gates.verify_live_job(ROOT, JOB)
    deadline = watch["deadline_unix"]
    stage = "stage-inputs"

    def run(name, command, *, heavy=False, env=None, limit=180):
        nonlocal stage
        stage = name
        remaining = deadline - time.time()
        if remaining <= 5:
            raise TimeoutError("Shared deadline reached before " + name)
        io.event("stage_started", stage=name)
        if heavy:
            report = str(JOB / (name + "-resources.json"))
            command = [sys.executable, str(SELF), "wrap", report, *command]
        with (JOB / (name + ".log")).open("x") as stream:
            subprocess.run(
                command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                check=True, timeout=min(limit, remaining),
            )
        io.event("stage_passed", stage=name)

    def passed(name, **fields):
        path = JOB / (name + ".json")
        io.save(path, {"status": "passed", **fields})
        return io.evidence(path)

    try:
        stage_inputs()
        checkpoint_root = Path(os.environ["ROSETTA_CHECKPOINT_ROOT"])
        candidate = checkpoint_root / io.EXP / "smoke" / fit.RUN_NAMES["C"]
        if candidate.exists() or candidate.is_symlink():
            raise FileExistsError("Hestia candidate already exists; no resume or retry")
        run("cpu", [sys.executable, "scripts/check_hestia_collector_cpu.py",
                    "--output", str(JOB / "cpu")],
            env=dict(os.environ, ROSETTA_TORCH_DEVICE="cpu"), limit=600)
        if io.load(JOB / "cpu/result.json")["status"] != "passed":
            raise ValueError("Current CPU acceptance failed")
        if io.load(JOB / "cpu/disk.json")["status"] != "passed":
            raise ValueError("Candidate checkpoint retention budget does not fit")
        main_plan = JOB / "cpu/draft/main1280.yaml"
        plan = io.load(main_plan)
        checks.validate_plan(plan, "C")
        stage = "prior-smoke-and-runtime-identity"
        proof = io.load(ROOT / gates.SMOKE_REPORT)
        prior = io.load(ROOT / gates.SMOKE_REGISTRATION)
        gates.verify_smoke_reuse(proof, prior["job_sources"], registration["source_files"])
        prior_cpu = io.load(ROOT / gates.SMOKE_CPU)
        for name, expected in prior_cpu["packages"].items():
            if version(name) != expected:
                raise ValueError("Runtime package differs from the completed smoke: " + name)
        smoke = Path(os.environ["ROSETTA_AUTODL_ROOT"]) / proof["checkpoint"]
        smoke_model_sha = io.digest(smoke / "pretrained_model/model.safetensors")
        if smoke_model_sha != proof["checkpoint_model_sha256"]:
            raise ValueError("Completed smoke model changed")
        prior_reload = proof["reload"]
        old_workspace = proof["remote_workspace_relative_to_durable_root"]
        old_job = Path(os.environ["ROSETTA_AUTODL_ROOT"]) / old_workspace / proof["job"]
        for suffix in ("first", "reload"):
            metadata_sha = io.digest(old_job / ("smoke-" + suffix + ".npz.json"))
            if metadata_sha != prior_reload[suffix + "_metadata_sha256"]:
                raise ValueError("Prior smoke reload metadata changed")
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip()
        if not gpu.startswith(proof["gpu"] + ",") or "\n" in gpu:
            raise ValueError("Expected one GPU of the previously verified model")
        if subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True,
        ).strip():
            raise ValueError("GPU is already in use")
        environment_root = JOB / "environment-runs"
        (environment_root / "trackio").mkdir(parents=True)
        isolated = dict(os.environ, ROSETTA_RUN_ROOT=str(environment_root),
                        TRACKIO_DIR=str(environment_root / "trackio"))
        run("check-env", [sys.executable, "scripts/check_env.py"])
        run("doctor", [sys.executable, "scripts/autodl_doctor_cuda.py", "--profile",
                       "configs/runtime/autodl_rtx4090.yaml", "--config", checks.PARENT_CONFIG],
            env=isolated)
        run("data", [sys.executable, "-m", "pytest", "-q", "-m", "data",
                     "tests/test_smolvla_visual_grounding_data.py",
                     "--junitxml=" + str(JOB / "data.xml")])
        suites = ET.parse(JOB / "data.xml").getroot()
        if any(int(s.attrib.get(k, 0)) for s in suites.iter("testsuite")
               for k in ("failures", "errors", "skipped")):
            raise ValueError("Real-data check failed or skipped")
        run("benchmark", [sys.executable, "scripts/benchmark_smolvla.py",
                          "--config", checks.PARENT_CONFIG], env=isolated)
        prerequisites = {"environment": passed("environment", gpu=gpu,
            runtime_profile_sha256=io.digest(ROOT / "configs/runtime/autodl_rtx4090.yaml"),
            check_env_log=io.evidence(JOB / "check-env.log"),
            doctor_log=io.evidence(JOB / "doctor.log"),
            benchmark_log=io.evidence(JOB / "benchmark.log"),
            data_log=io.evidence(JOB / "data.log"))}
        for batch in (1, 4):
            forward = copy.deepcopy(plan)
            name = fit.RUN_NAMES["C"] + f"-forward-b{batch}"
            forward.update(run_name=name, plan_id=name)
            forward["preflight"].update(run_name=name, batch_size=batch)
            path = JOB / "plans" / f"forward-b{batch}.yaml"
            io.save_stage_plan(path, forward)
            run(f"forward-b{batch}", ["scripts/run_smolvla_v2.py", "preflight",
                                     "--plan", str(path)], heavy=True)
        history = io.load(ROOT / checks.HISTORY)
        b = history["checkpoint_identities"]["B"]
        run("samples", ["scripts/visual_coverage_job_checks.py", "samples",
                        b["plan"]["path"], str(JOB / "samples.json")], heavy=True)
        if io.load(JOB / "samples.json")["status"] != "passed":
            raise ValueError("Current sample contract failed")
        prerequisites["sample_contract"] = io.evidence(JOB / "samples.json")
        for name in ("forward-b1", "forward-b4", "samples"):
            if io.load(JOB / (name + "-resources.json"))["status"] != "passed":
                raise ValueError("Preflight resource check failed")
        prerequisites["resource_preflight"] = passed("resource-preflight",
            reports=[io.evidence(JOB / (name + "-resources.json"))
                     for name in ("forward-b1", "forward-b4", "samples")])
        prerequisites["two_step_smoke_reload"] = passed("prior-smoke-reused",
            prior_report=io.evidence(ROOT / gates.SMOKE_REPORT),
            model_sha256=proof["checkpoint_model_sha256"],
            current_native_training_source_equal=True, optimizer_resumed=False, new_smoke_updates=0)
        prerequisites["B_training_integrity"] = passed("B-integrity",
            historical_report_sha256=io.digest(ROOT / checks.HISTORY),
            model_sha256=fit.CONTROL_MODEL_SHA256,
            saved_recipe=io.evidence(JOB / "cpu/B-saved-recipe.json"))
        needed = io.load(JOB / "cpu/disk.json")["minimum_new_bytes"]
        stage = "training-permit"
        if shutil.disk_usage(checkpoint_root).free < needed:
            raise ValueError("Disk budget changed before main training")
        io.save(JOB / "training-permit.json", {
            "status": "authorized_sealed",
            "registration_sha256": io.digest(JOB / "registration.json"),
            "watchdog_sha256": io.digest(JOB / "watchdog.json"), "plan": io.evidence(main_plan),
            "schedule": io.evidence(JOB / "cpu/schedule.json"),
            "prerequisite_evidence": prerequisites,
        })
        gates.authorize_training(ROOT, JOB)
        run("main1280", [str(SELF), "train"], heavy=True, limit=900)
        run("seal-C", ["scripts/seal_visual_fit_candidate.py", "--plan", str(main_plan),
            "--observation", str(JOB / "observation/result.json"),
            "--schedule", str(JOB / "cpu/schedule.json"),
            "--resources", str(JOB / "main1280-resources.json"),
            "--output", str(JOB / "C-integrity.json")], heavy=True)
        prerequisites["C_training_integrity"] = io.evidence(JOB / "C-integrity.json")
        contract = io.load(JOB / "cpu/draft/execution-contract.template.json")
        contract.update(status="authorized_sealed", model_execution_authorized=True,
            started_unix=watch["started_unix"], deadline_unix=deadline,
            watchdog=io.evidence(JOB / "watchdog.json"), prerequisite_evidence=prerequisites,
            implementation_files=registration["source_files"])
        files = io.load(JOB / "C-integrity.json")["checkpoint_files"]
        contract["arms"]["C"]["checkpoint_files"] = {
            name.removeprefix("pretrained_model/"): sha for name, sha in files.items()
            if name.startswith("pretrained_model/")
        }
        io.save(JOB / "execution-contract.json", contract)
        checks.check_execution(ROOT, contract)
        io.save(JOB / "selection-export.json", {
            "selection_rule": "preregistered_fixed_final_step_1280", "selected_step": 1280,
            "artifact": contract["arms"]["C"], "export_mode": "native_pretrained_artifact_in_place",
            "validation_checkpoint_search_used": False, "reload": "pending",
            "public_upload": False, "m2_complete": False,
        })
        for arm in ("B", "C"):
            for suffix in ("first", "reload"):
                name = arm + "-" + suffix
                run(name, ["scripts/evaluate_visual_fit.py", "collect",
                    "--contract", str(JOB / "execution-contract.json"),
                    "--arm", arm, "--output", str(JOB / name)], heavy=True)
            run(arm + "-reload-check", [sys.executable, "scripts/evaluate_visual_fit.py", "reload",
                "--first", str(JOB / (arm + "-first")), "--second", str(JOB / (arm + "-reload")),
                "--output", str(JOB / (arm + "-reload-check.json"))])
        stage = "historical-B-reproduction"
        from rosetta_reality.vla import visual_coverage

        durable = Path(os.environ["ROSETTA_AUTODL_ROOT"])
        old_manifest = checks.relative_file(durable, gates.HISTORICAL_B)
        if io.digest(old_manifest) != history["evidence_files"][gates.HISTORICAL_B]["sha256"]:
            raise ValueError("Historical B manifest changed")
        old, old_meta = visual_coverage.read_bundle(old_manifest.parent)
        current, current_meta = fit.read_bundle(JOB / "B-first")
        reproduced = gates.compare_historical_control(old, old_meta, current, current_meta)
        io.save(JOB / "historical-B-reproduction.json", reproduced)
        if reproduced["status"] != "passed":
            raise ValueError("Historical B full arrays did not reproduce")
        stage = "B-C-comparison"
        c, cm = fit.read_bundle(JOB / "C-first")
        comparison = fit.compare_arms(current, current_meta, c, cm)
        io.save(JOB / "comparison.json", comparison)
        outcome = (
            "offline_criteria_passed"
            if comparison["offline_metric_criteria_passed"] else "negative_result"
        )
        io.save(JOB / "result.json", {
            "status": outcome,
            "completed_optimizer_updates": 1280, "completed_sample_exposures": 5120,
            "B_and_C_independent_reload": "passed", "historical_B_reproduction": "passed",
            "comparison": io.evidence(JOB / "comparison.json"),
            "candidate_integrity": prerequisites["C_training_integrity"],
            "hidden_test_loaded": False, "task_success": "not measured", "m2_complete": False,
        })
    except BaseException as error:
        io.save(JOB / "failure.json", {"status": "failed", "stage": stage,
            "error_type": type(error).__name__, "no_automatic_retry": True,
            "main_completion": "inspect_actual_observation_and_checkpoints", "m2_complete": False})
        raise


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "wrap":
        io.wrapped()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("mode", choices=("supervise", "worker", "train"))
        parser.add_argument("--execute-authorized", action="store_true")
        parser.add_argument("--shutdown-authorized", action="store_true")
        args = parser.parse_args()
        if args.mode == "supervise":
            supervise(args.execute_authorized, args.shutdown_authorized)
        elif args.mode == "worker":
            worker()
        else:
            observed_train()
