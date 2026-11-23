"""Create-only, bounded AutoDL Q0 screening; import performs no work."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time

STAGES = ("check_env", "synthetic", "benchmark", "register", "g0", "g1", "g2")
PROFILE = "configs/runtime/autodl_rtx4090.yaml"
EXPERIMENT = "m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003"
RSS_LIMIT, EVIDENCE_LIMIT, RESERVE = 8 * 1024**3, 256 * 1024**2, 1024**3


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def save(path, payload):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def relative_path(raw):
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts or "\\" in raw or not raw:
        raise ValueError("Source references must be relative workspace paths")
    return path


def validate(plan):
    cap, target = plan["total_cap_seconds"], plan["platform_target_unix"]
    if type(cap) not in (int, float) or not math.isfinite(cap) or not 0 < cap <= 2700:
        raise ValueError("Work cap must be finite and at most 2700 seconds")
    if type(target) not in (int, float) or not math.isfinite(target):
        raise ValueError("Platform target must be a finite timestamp")
    if plan.get("shutdown_authorized") is not True:
        raise ValueError("Independent guard requires explicit shutdown authorization")
    for key in ("candidate_id", "module", "factory", "observer_module", "runtime_container_image_id"):
        if not isinstance(plan.get(key), str) or not re.fullmatch(r"[A-Za-z0-9_.:-]+", plan[key]):
            raise ValueError("Invalid identity field: " + key)
    sources = plan["source_sha256"]
    required = {"scripts/run_q0_screen_job.py", "scripts/autodl_bounded_guard.py",
                "scripts/q0_bounded_guard.py",
                "scripts/run_autodl.sh", "scripts/run_teacher_gate.py", "scripts/check_env.py",
                "scripts/prepare_q0_dependencies.py", "scripts/benchmark_q0_candidate.py", PROFILE,
                "src/" + plan["module"].replace(".", "/") + ".py",
                "src/" + plan["observer_module"].replace(".", "/") + ".py"}
    if not required <= sources.keys():
        raise ValueError("Missing source identity coverage")
    for raw, digest in sources.items():
        relative_path(raw)
        if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
            raise ValueError("Invalid source SHA256")
    if tuple(command["stage"] for command in plan["commands"]) != STAGES:
        raise ValueError("Exactly the seven ordered Q0 phases are required")
    for command in plan["commands"]:
        args, seconds = command["args"], command["seconds"]
        if (type(seconds) not in (int, float) or not math.isfinite(seconds)
                or not 0 < seconds <= cap):
            raise ValueError("Invalid phase timeout")
        if not isinstance(args, list) or not args or args[0] != "python":
            raise ValueError("Phases must use an explicit Python argv")
        if any(not isinstance(arg, str) or "\n" in arg or "\r" in arg
               or arg.startswith("/") or "\\" in arg for arg in args):
            raise ValueError("Command references must be relative; use {job} placeholder")
        stage = command["stage"]
        if stage in ("register", "g0", "g1", "g2"):
            if args[1:3] != ["scripts/run_teacher_gate.py", stage]:
                raise ValueError("Gate phases must invoke the existing gate harness")
            if "--candidate-id" not in args or args[args.index("--candidate-id") + 1] != plan["candidate_id"]:
                raise ValueError("Gate candidate identity mismatch")
            if stage == "register":
                for flag, key in (("--module", "module"), ("--factory", "factory"),
                                  ("--observer-module", "observer_module")):
                    if flag not in args or args[args.index(flag) + 1] != plan[key]:
                        raise ValueError("Registration identity mismatch")
                if "--simulator-execution" not in args:
                    raise ValueError("Registration must declare simulator execution")
        elif stage == "check_env" and args[1:] != ["scripts/prepare_q0_dependencies.py", "--job", "{job}"]:
            raise ValueError("Environment phase must prepare the isolated dependency overlay and check_env")
        elif stage == "synthetic":
            if args[1:4] != ["-m", "pytest", "-q"] or not args[4:] or any(
                not arg.startswith("tests/test_") or not arg.endswith(".py") or arg not in sources
                for arg in args[4:]
            ):
                raise ValueError("Synthetic phase must use explicitly hashed pytest test files")
        elif stage == "benchmark" and args[1:] != ["scripts/benchmark_q0_candidate.py", "--output", "{job}/benchmark.json"]:
            raise ValueError("Benchmark must use the bounded Q0 benchmark")


def check_sources(workspace, plan):
    for raw, expected in plan["source_sha256"].items():
        path = (workspace / relative_path(raw)).resolve(strict=True)
        if not path.is_relative_to(workspace) or sha(path) != expected:
            raise ValueError("Source identity mismatch: " + raw)


def group_members(group):
    members = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == group:
                members.append((int(proc.name), fields[19], int(fields[21]) * os.sysconf("SC_PAGE_SIZE")))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return members


def stop_children():
    # Children retain our group so the independent guard can kill everything.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid, ticks, _ in group_members(os.getpgrp()):
            if pid == os.getpid():
                continue
            try:
                fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                if fields[19] == ticks and int(fields[2]) == os.getpgrp():
                    os.kill(pid, sig)
            except (FileNotFoundError, ProcessLookupError):
                pass
        if sig == signal.SIGTERM:
            time.sleep(1)


def resources(job):
    size = sum(path.stat().st_size for path in job.rglob("*") if path.is_file())
    rss = sum(entry[2] for entry in group_members(os.getpgrp()))
    if size > EVIDENCE_LIMIT or rss > RSS_LIMIT or shutil.disk_usage(job).free < RESERVE:
        raise RuntimeError("Resource budget exceeded")
    return {"evidence_bytes": size, "group_rss_bytes": rss}


def verify_gate(job, candidate, stage):
    root = job / "gate-runs" / EXPERIMENT / "teacher_gate"
    registration = root / "registrations" / (candidate + ".json")
    paths = list((root / candidate).glob(stage + "-*.json"))
    if len(paths) != 1:
        raise ValueError("Gate requires exactly one fresh attempt")
    value = json.loads(paths[0].read_text())
    episodes = value.get("episodes", [])
    if (value.get("candidate_id") != candidate or value.get("passed") is not True
            or value.get("registration_sha256") != sha(registration)
            or value.get("stage") != stage or not episodes):
        raise ValueError("Gate did not pass with bound identity")
    expected_seeds = [10] if stage == "g1" else [1900, 1901, 1902, 1903, 1904]
    if [episode.get("seed") for episode in episodes] != expected_seeds:
        raise ValueError("Gate episode inventory mismatch")
    for episode in episodes:
        if (episode.get("success") is not True or episode.get("joint_limit_violations") != 0
                or episode.get("unexpected_collisions") != 0 or episode.get("refusals") != []
                or not 0 < episode.get("rollout_length", 0) <= 500):
            raise ValueError("Gate evidence has unsafe or unsuccessful episodes")
    return {"path": str(paths[0].relative_to(job)), "sha256": sha(paths[0])}


def drain_output(pipe, log):
    # Read bounded chunks rather than unbounded lines; never copy absolute host paths.
    while block := pipe.read(4096):
        text = block.decode("utf-8", errors="replace")
        text = re.sub(r"(?<![\w:])/(?:[^\s\"'<>]+)", "<absolute-path>", text)
        log.write(text)
        log.flush()


def worker(workspace, job, plan_path):
    if os.getpgrp() != os.getpid():
        raise RuntimeError("Worker must own its independent process group")
    def interrupted(_signum, _frame):
        raise TimeoutError("Independent guard interrupted worker")
    signal.signal(signal.SIGTERM, interrupted)
    phases, active, failed = [], None, True
    reg = json.loads((job / "registration.json").read_text())
    deadline = time.monotonic() + max(0, reg["work_deadline_unix"] - time.time())
    try:
        if sha(plan_path) != reg["plan_sha256"]:
            raise ValueError("Plan identity drift")
        plan = json.loads(plan_path.read_text())
        validate(plan)
        check_sources(workspace, plan)
        env = dict(os.environ, PYTHONPATH=f"{workspace / 'src'}:{workspace}:{job / 'dependencies'}",
                   PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MUJOCO_GL="egl",
                   ROSETTA_RUN_ROOT=str(job / "gate-runs"),
                   ROSETTA_CONTAINER_IMAGE_ID=plan["runtime_container_image_id"])
        save(job / "runtime-observation.json", {"python_version": sys.version.split()[0],
             "runtime_container_image_id": plan["runtime_container_image_id"],
             "image_digest_independently_verified": False, "nested_docker_used": False,
             "cgroup_memory_max": Path("/sys/fs/cgroup/memory.max").read_text().strip(),
             "mount_namespace_identity": os.readlink("/proc/self/ns/mnt")})
        for command in plan["commands"]:
            active, started = command["stage"], time.monotonic()
            phase = {"stage": active, "started_unix": time.time(), "args": command["args"]}
            phases.append(phase)
            end = min(deadline, started + command["seconds"])
            resources(job)
            if end <= started:
                raise TimeoutError("Work deadline reached")
            args = [sys.executable] + [arg.replace("{job}", str(job)) for arg in command["args"][1:]]
            with (job / (active + ".log")).open("x", encoding="utf-8") as log:
                child = subprocess.Popen(args, cwd=workspace, env=env, stdin=subprocess.DEVNULL,
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                reader = threading.Thread(target=drain_output, args=(child.stdout, log), daemon=True)
                reader.start()
                try:
                    while child.poll() is None:
                        resources(job)
                        if time.monotonic() >= end:
                            raise TimeoutError("Phase timeout")
                        time.sleep(min(2, max(0, end - time.monotonic())))
                finally:
                    # Stop descendants and drain before the log file closes, including timeout paths.
                    stop_children()
                    child.wait(timeout=5)
                    reader.join(timeout=5)
                    if reader.is_alive():
                        raise RuntimeError("Child left a live output pipe")
            phase.update(exit_code=child.returncode, seconds=time.monotonic() - started)
            if child.returncode:
                raise RuntimeError("First failed phase; retries prohibited")
            if active in ("g1", "g2"):
                phase["gate_evidence"] = verify_gate(job, plan["candidate_id"], active)
            save(job / (active + "-phase.json"), phase)
        failed = False
        save(job / "result.json", {"status": "partial_screen_only", "provider_qualified": False,
             "g3_g4_g5_executed": False, "collection_started": False, "optimizer_updates": 0,
             "phases": phases, "automatic_retry": False})
    except Exception as exc:
        save(job / "failure.json", {"phase": active, "error_type": type(exc).__name__,
             "phases": phases, "provider_qualified": False, "automatic_retry": False})
    finally:
        # Finish bounded cleanup when the guard sends TERM to the entire group.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        stop_children()
        entries = [{"path": str(p.relative_to(job)), "sha256": sha(p), "bytes": p.stat().st_size}
                   for p in sorted(job.rglob("*")) if p.is_file() and p.name not in ("guard.log", "supervisor.log")]
        save(job / "manifest.json", {"files": entries, "bytes": sum(x["bytes"] for x in entries),
             "excluded_live_logs": ["guard.log", "supervisor.log"]})
        save(job / "worker-exited.json", {"finished_unix": time.time(), "status": "failed" if failed else "completed",
             "provider_qualified": False, "manifest_sha256": sha(job / "manifest.json")})
        os.sync()


def launch(workspace, job, plan_path):
    plan = json.loads(plan_path.read_text())
    validate(plan)
    check_sources(workspace, plan)
    from autodl_bounded_guard import WRAPPER_SHA256
    if sha(Path("/usr/bin/shutdown")) != WRAPPER_SHA256:
        raise ValueError("Shutdown wrapper mismatch")
    trash = Path("/root/.local/share/Trash")
    if trash.exists() or trash.is_symlink():
        raise ValueError("Trash exists; shutdown safety check failed")
    for args, codes in ((["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], (0,)),
                        (["tmux", "list-sessions"], (0, 1))):
        check = subprocess.run(args, capture_output=True, timeout=10, check=False)
        if check.returncode not in codes or check.stdout.strip():
            raise RuntimeError("Existing GPU work or tmux prevents launch")
    now = time.time()
    work = min(now + plan["total_cap_seconds"], plan["platform_target_unix"] - 300)
    if work <= now or shutil.disk_usage(job.parent).free < RESERVE:
        raise ValueError("No bounded work window or disk reserve")
    job.mkdir()
    save(job / "registration.json", {"started_unix": now, "work_deadline_unix": work,
         "shutdown_deadline_unix": work + 300, "shutdown_authorized": True,
         "platform_target_unix": plan["platform_target_unix"], "platform_target_independently_verified": False,
         "plan_sha256": sha(plan_path), "source_sha256": plan["source_sha256"],
         "runtime_profile_reference": PROFILE, "runtime_profile_sha256": sha(workspace / PROFILE),
         "runtime_profile_config": {"reference": PROFILE, "sha256": sha(workspace / PROFILE)},
         "nested_docker_used": False,
         "runtime_container_image_id": plan["runtime_container_image_id"],
         "rss_limit_bytes": RSS_LIMIT, "evidence_limit_bytes": EVIDENCE_LIMIT,
         "disk_reserve_bytes": RESERVE, "provider_qualified": False, "automatic_retry": False})
    with (job / "guard.log").open("x") as log:
        guard = subprocess.Popen([sys.executable, "-u", str(workspace / "scripts/q0_bounded_guard.py"),
                                  "--job", str(job), "--retrieval-grace-seconds", "300"],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    for _ in range(50):
        if (job / "guard-armed.json").exists():
            break
        if guard.poll() is not None:
            raise RuntimeError("Independent guard failed")
        time.sleep(.1)
    armed = json.loads((job / "guard-armed.json").read_text())
    if armed["pid"] != guard.pid or armed["work_deadline_unix"] != work:
        raise RuntimeError("Independent guard registration mismatch")
    command = [sys.executable, "-u", "scripts/run_q0_screen_job.py", "worker", "--workspace", str(workspace),
               "--job", str(job), "--plan", str(plan_path)]
    with (job / "supervisor.log").open("x") as log:
        child = subprocess.Popen(["bash", "scripts/run_autodl.sh", "shell"], cwd=workspace,
                                 stdin=subprocess.PIPE, stdout=log, stderr=log, start_new_session=True)
        ticks = Path(f"/proc/{child.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
        save(job / "worker-pid.json", {"pid": child.pid, "start_ticks": ticks})
        child.stdin.write(("exec " + shlex.join(command) + "\n").encode())
        child.stdin.close()
    print(json.dumps({"worker_pid": child.pid, "guard_pid": guard.pid, "work_deadline_unix": work,
                      "shutdown_deadline_unix": work + 300, "provider_qualified": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "worker"))
    for key in ("workspace", "job", "plan"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    globals()[args.mode](args.workspace.resolve(strict=True), args.job.resolve(), args.plan.resolve(strict=True))
