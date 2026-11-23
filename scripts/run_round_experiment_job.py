"""Durable work-only round supervisor with identity-bound independent guards."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager, nullcontext

POLL = .1
HANDSHAKE = 10
GRACE = 3
SOURCE = "scripts/run_round_experiment_job.py"
RSS_CEILING = 16 * 1024**3


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def family_lock(path, exclusive=False):
    import fcntl
    with Path(str(path) + ".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def gpu_snapshot():
    """Driver-owned usage, including reserved/graphics overhead; one GPU only."""
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,memory.total,memory.used",
        "--format=csv,noheader,nounits"], text=True, timeout=10).strip().splitlines()
    if len(raw) != 1:
        raise RuntimeError("One registered RTX4090 GPU required")
    device_uuid, name, total, used = [v.strip() for v in raw[0].split(",")]
    if "4090" not in name:
        raise RuntimeError("Registered GPU model changed")
    processes = []
    rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
        "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
    for row in rows.splitlines():
        gpu_uuid, pid, memory = [v.strip() for v in row.split(",")]
        processes.append({"gpu_uuid": gpu_uuid, "pid": int(pid), "bytes": int(memory)*1024**2})
    return {"uuid": device_uuid, "total_bytes": int(total)*1024**2,
            "used_bytes": int(used)*1024**2, "processes": processes}


def family_descriptor(plan):
    item = plan.get("resource_family")
    if item is None:
        return None
    if (not isinstance(item, dict) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", item.get("family_id", ""))
            or not re.fullmatch(r"[a-f0-9]{32}", item.get("member_nonce", ""))):
        raise ValueError("Explicit family ID and predeclared member nonce required")
    path = Path(item["manifest_path"])
    if not path.is_absolute() or path.resolve() != path:
        raise ValueError("Canonical absolute family manifest required")
    gpu_cap = plan.get("maximum_gpu_bytes")
    if type(gpu_cap) is not int or gpu_cap < 0 or (plan.get("gpu_required") is True and gpu_cap == 0):
        raise ValueError("Explicit per-member GPU budget required")
    if plan.get("gpu_required") is not True and gpu_cap != 0:
        raise ValueError("GPU sharing must remain explicitly GPU-required")
    return item


def register_family(spec_path):
    """Reserve new job roots before any member starts; never adopt old jobs."""
    spec = read(spec_path)
    path = Path(spec["manifest_path"])
    if not path.is_absolute() or path.resolve() != path or path.exists():
        raise ValueError("New canonical create-only family manifest required")
    if spec.get("shutdown_authorized") is not False or not spec.get("members"):
        raise ValueError("Explicit work-only family required")
    if type(spec.get("maximum_rss_bytes")) is not int or not 0 < spec["maximum_rss_bytes"] <= RSS_CEILING:
        raise ValueError("Family RSS budget cannot exceed 16GiB")
    gpu_cap = spec.get("maximum_gpu_bytes")
    if type(gpu_cap) is not int or gpu_cap < 0:
        raise ValueError("Explicit family GPU budget required")
    roots, nonces, rss_sum, gpu_sum = [], [], 0, 0
    for row in spec["members"]:
        root = Path(row["job_root"])
        workspace = Path(row["workspace"])
        plan_path = Path(row["plan_path"])
        if (not root.is_absolute() or root.resolve() != root or root.exists()
                or not workspace.is_absolute() or workspace.resolve(strict=True) != workspace
                or not plan_path.is_absolute() or plan_path.resolve(strict=True) != plan_path):
            raise ValueError("Family members require new canonical job roots and fixed inputs")
        plan = read(plan_path); verify(plan, workspace)
        declaration = family_descriptor(plan)
        if (declaration is None or declaration["manifest_path"] != str(path)
                or declaration["family_id"] != spec["family_id"] or declaration["member_nonce"] != row["nonce"]
                or row["plan_sha256"] != file_sha(plan_path)
                or row["maximum_rss_bytes"] != plan["maximum_rss_bytes"]
                or row["maximum_gpu_bytes"] != plan["maximum_gpu_bytes"]
                or plan["source_sha256"][SOURCE] != file_sha(__file__)):
            raise ValueError("Family member source/nonce/plan/resource identity drift")
        roots.append(root); nonces.append(row["nonce"])
        rss_sum += row["maximum_rss_bytes"]; gpu_sum += row["maximum_gpu_bytes"]
    if (len(set(roots)) != len(roots) or len(set(nonces)) != len(nonces)
            or any(a != b and (a.is_relative_to(b) or b.is_relative_to(a)) for a in roots for b in roots)):
        raise ValueError("Duplicate or overlapping family ownership")
    if rss_sum > spec["maximum_rss_bytes"] or gpu_sum > gpu_cap:
        raise ValueError("Sum of predeclared member peaks exceeds joint budget")
    if gpu_sum:
        gpu = gpu_snapshot()
        if (spec["gpu_uuid"] != gpu["uuid"] or spec["gpu_total_bytes"] != gpu["total_bytes"]
                or not 0 < gpu_cap <= gpu["total_bytes"] or gpu["processes"]):
            raise RuntimeError("Family registration requires idle exact GPU and admissible joint peaks")
    elif gpu_cap != 0 or spec.get("gpu_uuid") is not None or spec.get("gpu_total_bytes", 0) != 0:
        raise ValueError("CPU-only family cannot claim a CUDA device")
    path.parent.mkdir(parents=True, exist_ok=True)
    save(path, spec)
    family_sha = file_sha(path)
    with family_lock(path, exclusive=True):
        for root, row in zip(roots, spec["members"]):
            root.mkdir(parents=True, exist_ok=False)
            save(root / "family-reserved.json", {"family_sha256": family_sha,
                "nonce": row["nonce"], "plan_sha256": row["plan_sha256"]})
        save(Path(str(path) + ".ready.json"), {"family_sha256": family_sha, "members": len(roots)})
    return {"family_id": spec["family_id"], "family_sha256": family_sha, "members": len(roots), "shutdown_requested": False}


def family_binding(job, reg):
    declaration = family_descriptor(reg["plan"])
    if declaration is None:
        return None
    path = Path(declaration["manifest_path"]); spec = read(path)
    expected = reg["family_sha256"]
    if (file_sha(path) != expected or read(Path(str(path)+".ready.json"))["family_sha256"] != expected
            or spec["family_id"] != declaration["family_id"]):
        raise ValueError("Family manifest identity changed")
    rows = [row for row in spec["members"] if row["job_root"] == str(job)]
    if len(rows) != 1:
        raise ValueError("Job is not a predeclared member")
    row = rows[0]; reservation = read(job / "family-reserved.json")
    if (reservation != {"family_sha256": expected, "nonce": row["nonce"], "plan_sha256": row["plan_sha256"]}
            or reg["nonce"] != row["nonce"] or reg["plan_sha256"] != row["plan_sha256"]
            or reg["workspace"] != row["workspace"] or reg["plan_path"] != row["plan_path"]
            or file_sha(row["plan_path"]) != row["plan_sha256"] or read(row["plan_path"]) != reg["plan"]):
        raise ValueError("Family member registration drift")
    return spec


def owned_records(job, reg):
    records = {}
    for name in ("worker", "guard"):
        path = job / (name + "-pid.json")
        if path.exists():
            owner = read(path)
            if matches(owner):
                for current in capture(owner)["members"]:
                    records[(current["pid"],current["start_ticks"])] = current
    path = job / "active-stage.json"
    if path.exists():
        active = read(path)
        if reg["plan"].get("resource_family") is not None:
            initial = read(job / (active["stage"] + "-process.json"))
            if (active["nonce"] != reg["nonce"] or active["plan_sha256"] != reg["plan_sha256"]
                    or initial["leader"] != active["leader"] or initial["nonce"] != reg["nonce"]):
                raise ValueError("Stage does not belong to this exact family registration")
        for current in capture(active["leader"], active["members"])["members"]:
            if matches(current):
                records[(current["pid"],current["start_ticks"])] = current
    return records


def rss_bytes(records):
    total = 0
    for current in records.values():
        if not matches(current):
            continue
        try:
            for line in Path(f"/proc/{current['pid']}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1])*1024
        except (FileNotFoundError, ProcessLookupError):
            pass
    return total


def check_family_gpu(spec, records_by_job, snapshot):
    if (snapshot["uuid"] != spec["gpu_uuid"] or snapshot["total_bytes"] != spec["gpu_total_bytes"]
            or snapshot["used_bytes"] > spec["maximum_gpu_bytes"]):
        raise MemoryError("Registered joint GPU device/usage budget exceeded")
    owners = {}
    for root, records in records_by_job.items():
        for current in records.values():
            if matches(current):
                aliases = {current["pid"]}
                try:
                    for line in Path(f"/proc/{current['pid']}/status").read_text().splitlines():
                        if line.startswith("NSpid:"):
                            aliases.update(int(value) for value in line.split()[1:])
                except (FileNotFoundError, ProcessLookupError):
                    continue
                if not matches(current):
                    continue
                for pid in aliases:
                    if pid in owners and owners[pid] != root:
                        raise ValueError("GPU process has ambiguous kernel PID ownership")
                    owners[pid] = root
    usage = {row["job_root"]:0 for row in spec["members"]}
    for row in snapshot["processes"]:
        if row["gpu_uuid"] != spec["gpu_uuid"] or row["pid"] not in owners:
            raise RuntimeError("Unknown/unbound GPU compute process; shared admission refused")
        usage[owners[row["pid"]]] += row["bytes"]
    for row in spec["members"]:
        if usage[row["job_root"]] > row["maximum_gpu_bytes"]:
            raise MemoryError("Member GPU usage exceeded its registered peak")
    return usage


def family_resources(job, reg, *, already_locked=False):
    path = Path(reg["plan"]["resource_family"]["manifest_path"])
    with (nullcontext() if already_locked else family_lock(path)):
        spec = family_binding(job, reg); all_records = {}; per_job = {}
        for row in spec["members"]:
            root = Path(row["job_root"])
            if not (root/"registration.json").exists():
                # A reserved, not-yet-launched member grants no process ownership.
                if read(root/"family-reserved.json") != {"family_sha256":reg["family_sha256"],"nonce":row["nonce"],"plan_sha256":row["plan_sha256"]}:
                    raise ValueError("Pending family member reservation drift")
                continue
            member = registration(root)
            if member.get("family_sha256") != reg["family_sha256"]:
                raise ValueError("Member adopted a different resource family")
            records = owned_records(root, member)
            if (root/"worker-pid.json").exists() and matches(read(root/"worker-pid.json")):
                guard_alive(root, member)
            current_rss = rss_bytes(records)
            if current_rss > row["maximum_rss_bytes"]:
                raise MemoryError("Family member RSS cap exceeded")
            per_job[str(root)] = records
            for identity, current in records.items():
                if identity in all_records:
                    raise ValueError("Family jobs share an owned process")
                all_records[identity] = current
        total_rss = rss_bytes(all_records)
        if total_rss > spec["maximum_rss_bytes"] or total_rss > RSS_CEILING:
            raise MemoryError("Joint RSS budget exceeded")
        if spec["maximum_gpu_bytes"]:
            check_family_gpu(spec, per_job, gpu_snapshot())
        return {"joint_rss_bytes":total_rss,"members_with_processes":len(per_job)}


def save(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    sync_dir(path.parent)


def sync_dir(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def active_save(path, value):
    path = Path(path)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    save(temp, value)
    os.replace(temp, path)
    sync_dir(path.parent)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def process(pid, proc_root=Path("/proc")):
    try:
        fields = (proc_root / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "start_ticks": fields[19], "group": int(fields[2]),
                "session": int(fields[3]), "state": fields[0]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def matches(record):
    current = process(record["pid"])
    return bool(current and current["state"] != "Z" and all(
        current[key] == record[key] for key in ("pid", "start_ticks", "group", "session")))


def members(group):
    result = []
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            current = process(int(path.name))
            if current and current["group"] == group and current["state"] != "Z":
                result.append(current)
    return result


def bind(child):
    current = process(child.pid)
    if current is None or current["group"] != child.pid or current["session"] != child.pid:
        raise RuntimeError("Independent process session identity not established")
    return {key: current[key] for key in ("pid", "start_ticks", "group", "session")}


def capture(leader, known=()):
    retained = {(row["pid"], row["start_ticks"]): row for row in known}
    if matches(leader):
        for current in members(leader["group"]):
            if current["session"] != leader["session"]:
                raise RuntimeError("Group session changed")
            retained[(current["pid"], current["start_ticks"])] = current
    return {"leader": leader, "members": list(retained.values())}


def stop_group(receipt, grace=GRACE):
    """Never signal a reused PID/group; bind survivors individually by start ticks."""
    leader = receipt["leader"]
    if leader["pid"] <= 1 or leader["pid"] == os.getpid():
        raise ValueError("Cleanup requires a separate registered group")
    known = capture(leader, receipt.get("members", []))["members"]
    if matches(leader):
        try:
            os.killpg(leader["group"], signal.SIGTERM)
        except ProcessLookupError:
            pass
    else:
        for member in known:
            if matches(member):
                try:
                    os.kill(member["pid"], signal.SIGTERM)
                except ProcessLookupError:
                    pass
    until = time.monotonic() + grace
    while time.monotonic() < until and members(leader["group"]):
        time.sleep(POLL)
    # Children may ignore TERM even after their leader exits.
    for member in known:
        if matches(member):
            try:
                os.kill(member["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass
    until = time.monotonic() + 3
    while time.monotonic() < until:
        if not members(leader["group"]):
            return {"status": "owned_group_empty", "bound_members": len(known)}
        time.sleep(POLL)
    raise RuntimeError("Unproven or live group members remain after bounded cleanup")


def verify(plan, workspace):
    if plan.get("execution_authorized") is not True or plan.get("shutdown_authorized") is not False:
        raise ValueError("Explicit work-only authorization required")
    cap = plan["maximum_seconds"]
    if type(cap) not in (int, float) or not math.isfinite(cap) or not 0 < cap <= 28800:
        raise ValueError("Finite registered work cap <=28800 required")
    for key in ("minimum_free_bytes", "maximum_rss_bytes", "maximum_output_bytes"):
        if type(plan.get(key)) is not int or plan[key] <= 0:
            raise ValueError("Explicit positive resource budgets required")
    if plan["maximum_rss_bytes"] > 16 * 1024**3:
        raise ValueError("Host memory design budget exceeded")
    family_descriptor(plan)
    sources = plan["source_sha256"]
    if SOURCE not in sources:
        raise ValueError("Supervisor identity must be sealed")
    for raw, expected in sources.items():
        path = Path(raw)
        if path.is_absolute() or ".." in path.parts or hashlib.sha256((workspace / path).read_bytes()).hexdigest() != expected:
            raise ValueError("Source identity drift")
    commands = plan["commands"]
    if not isinstance(commands, list) or not commands:
        raise ValueError("Explicit stages required")
    names = []
    for command in commands:
        name = command.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", name):
            raise ValueError("Path-safe stage names required")
        names.append(name)
        argv = command.get("argv")
        if not isinstance(argv, list) or not argv or any(not isinstance(v, str) for v in argv):
            raise ValueError("Explicit argv required; no shell")
        seconds = command["maximum_seconds"]
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 < seconds <= cap:
            raise ValueError("Invalid stage work cap")
        if command.get("allowed_exit_codes") not in ([0], [0, 2]):
            raise ValueError("Only explicitly registered logical-negative exit codes allowed")
    if len(set(names)) != len(names) or sum(c["maximum_seconds"] for c in commands) > cap:
        raise ValueError("Duplicate stages or stage budget overflow")


def registration(job):
    reg = read(job / "registration.json")
    if reg["shutdown_authorized"] is not False:
        raise ValueError("Work-only registration required")
    verify(reg["plan"], Path(reg["workspace"]))
    if reg["deadline_unix"] - reg["started_unix"] != reg["plan"]["maximum_seconds"]:
        raise ValueError("Registered work deadline changed")
    if "plan_path" in reg and (file_sha(reg["plan_path"]) != reg["plan_sha256"] or read(reg["plan_path"]) != reg["plan"]):
        raise ValueError("Registered plan file changed")
    family_binding(job, reg)
    return reg


def guard_alive(job, reg):
    armed, record = read(job / "guard-armed.json"), read(job / "guard-pid.json")
    if (armed["nonce"] != reg["nonce"] or armed["plan_sha256"] != reg["plan_sha256"]
            or armed["process"] != record or armed["deadline_unix"] != reg["deadline_unix"]
            or not matches(record) or (job / "guard-exited.json").exists()):
        raise RuntimeError("Independent guard identity or liveness lost")


def cleanup_stage(job, actor):
    path = job / "active-stage.json"
    if not path.exists():
        return None
    receipt = read(path)
    reg = registration(job)
    if reg["plan"].get("resource_family") is not None:
        original = read(job / (receipt["stage"] + "-process.json"))
        if (receipt["leader"] != original["leader"] or receipt["nonce"] != reg["nonce"]
                or original["nonce"] != reg["nonce"] or receipt["plan_sha256"] != reg["plan_sha256"]):
            raise ValueError("Refusing cleanup of an unbound shared stage")
    status = stop_group(receipt)
    result = {**status, "stage": receipt["stage"], "actor": actor, "shutdown_requested": False}
    path = job / (receipt["stage"] + "-cleanup-" + actor + ".json")
    if not path.exists():
        save(path, result)
    return result


def guard(job):
    reg = registration(job)
    me = process(os.getpid())
    if me is None or me["group"] != os.getpid() or me["session"] != os.getpid():
        raise RuntimeError("Independent guard process group required")
    handshake = time.monotonic() + HANDSHAKE
    while not (job / "guard-pid.json").exists():
        if time.monotonic() >= handshake:
            raise TimeoutError("Guard PID receipt missing")
        time.sleep(POLL)
    owner = read(job / "guard-pid.json")
    if not matches(owner) or any(owner[key] != me[key] for key in ("pid", "start_ticks", "group", "session")):
        raise RuntimeError("Guard PID/start ticks differ")
    deadline = time.monotonic() + max(0, reg["deadline_unix"] - time.time())
    save(job / "guard-armed.json", {"nonce": reg["nonce"], "plan_sha256": reg["plan_sha256"],
         "process": owner, "deadline_unix": reg["deadline_unix"], "shutdown_authorized": False})
    outcome, error, cleanup = "failed", None, None
    last_resources = 0
    def interrupted(_signum, _frame):
        raise InterruptedError("Guard interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        while time.monotonic() < deadline:
            if (job / "worker-pid.json").exists():
                owner = read(job / "worker-pid.json")
                if not matches(owner):
                    outcome = "worker_exited" if (job / "worker-exited.json").exists() else "worker_vanished"
                    break
            elif time.monotonic() >= handshake:
                raise TimeoutError("Worker never started")
            if time.monotonic() >= last_resources + 1:
                receipt = read(job/"active-stage.json") if (job/"active-stage.json").exists() else None
                resources(job, reg["plan"], receipt, reg=reg)
                last_resources = time.monotonic()
            time.sleep(POLL)
        else:
            outcome = "work_deadline_reached"
    except BaseException as exc:
        error = type(exc).__name__
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            cleanup = cleanup_stage(job, "guard")
            if (job / "worker-pid.json").exists():
                owner = read(job / "worker-pid.json")
                stop_group({"leader": owner, "members": [owner]})
        except BaseException as exc:
            error = error or type(exc).__name__
            save(job / "guard-cleanup-blocked.json", {"error_type": type(exc).__name__, "shutdown_requested": False})
        save(job / "guard-exited.json", {"status": outcome, "error_type": error,
             "cleanup": cleanup, "shutdown_requested": False, "finished_unix": time.time()})


def resources(job, plan, receipt, *, reg=None):
    if shutil.disk_usage(job).free < plan["minimum_free_bytes"]:
        raise OSError("Storage reserve insufficient")
    reg = registration(job) if reg is None else reg
    rss = rss_bytes(owned_records(job, reg))
    if rss > plan["maximum_rss_bytes"]:
        raise MemoryError("Registered tree RSS exceeded")
    if sum(p.stat().st_size for p in job.rglob("*") if p.is_file()) > plan["maximum_output_bytes"]:
        raise OSError("Registered output cap exceeded")
    if plan.get("resource_family") is not None:
        family_resources(job, reg)


def worker(job):
    outcomes, error, cleanup, child = [], None, None, None
    def interrupted(_signum, _frame):
        raise InterruptedError("Worker interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        reg = registration(job)
        plan, workspace = reg["plan"], Path(reg["workspace"])
        deadline = time.monotonic() + max(0, reg["deadline_unix"] - time.time())
        guard_alive(job, reg)
        for command in plan["commands"]:
            guard_alive(job, reg)
            resources(job, plan, None, reg=reg)
            if time.monotonic() >= deadline:
                raise TimeoutError("Round deadline")
            stage_end = min(deadline, time.monotonic() + command["maximum_seconds"])
            environment = dict(os.environ, **plan.get("environment", {}))
            environment.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
            argv = [sys.executable if v == "{python}" else v.replace("{job}", str(job)) for v in command["argv"]]
            with (job / (command["name"] + ".log")).open("xb") as stream:
                child = subprocess.Popen(argv, cwd=workspace, env=environment, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                leader = bind(child)
                receipt = {**capture(leader), "stage": command["name"], "nonce":reg["nonce"], "plan_sha256":reg["plan_sha256"]}
                save(job / (command["name"] + "-process.json"), receipt)
                active_save(job / "active-stage.json", receipt)
                last_resources = 0
                while child.poll() is None:
                    guard_alive(job, reg)
                    if time.monotonic() >= stage_end:
                        raise TimeoutError("Stage deadline")
                    receipt = {**capture(leader, receipt["members"]), "stage": command["name"], "nonce":reg["nonce"], "plan_sha256":reg["plan_sha256"]}
                    active_save(job / "active-stage.json", receipt)
                    if time.monotonic() >= last_resources + 1:
                        resources(job, plan, receipt, reg=reg)
                        last_resources = time.monotonic()
                    time.sleep(POLL)
                guard_alive(job, reg)
                resources(job, plan, receipt, reg=reg)
                if time.monotonic() >= stage_end:
                    raise TimeoutError("Stage completed after registered deadline")
                cleanup = cleanup_stage(job, "worker")
                child.wait(timeout=3)
                outcomes.append({"name": command["name"], "returncode": child.returncode})
                save(job / (command["name"] + "-exited.json"), outcomes[-1])
                guard_alive(job, reg)
                if child.returncode not in command["allowed_exit_codes"]:
                    raise RuntimeError("Unclassified stage failure")
        guard_alive(job, reg)
        resources(job, plan, None, reg=reg)
        if time.monotonic() >= deadline:
            raise TimeoutError("Round completed after registered deadline")
    except BaseException as exc:
        error = type(exc).__name__
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            cleanup = cleanup_stage(job, "worker")
            if child is not None:
                child.wait(timeout=3)
        except BaseException as exc:
            error = error or type(exc).__name__
            save(job / "worker-cleanup-blocked.json", {"error_type": type(exc).__name__, "shutdown_requested": False})
        save(job / "worker-exited.json", {"outcomes": outcomes, "error_type": error,
             "cleanup": cleanup, "shutdown_requested": False, "finished_unix": time.time()})
    return 1 if error else 0


def launch(args):
    plan = read(Path(args.plan))
    declaration = family_descriptor(plan)
    if declaration is None:
        return launch_member(args)
    with family_lock(Path(declaration["manifest_path"]), exclusive=True):
        return launch_member(args)


def launch_member(args):
    workspace, plan_path = Path(args.workspace).resolve(strict=True), Path(args.plan).resolve(strict=True)
    plan = read(plan_path)
    verify(plan, workspace)
    if shutil.disk_usage(workspace).free < plan["minimum_free_bytes"]:
        raise OSError("Storage reserve insufficient")
    declaration = family_descriptor(plan)
    if plan.get("gpu_required") is True and declaration is None:
        occupied = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
        model = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
        if occupied or "4090" not in model:
            raise RuntimeError("RTX4090 worker must be exclusive")
    job = Path(args.job).resolve()
    if declaration is None:
        job.mkdir(parents=True, exist_ok=False)
        nonce = uuid.uuid4().hex
        family_sha = None
    else:
        reservation = read(job/"family-reserved.json")
        if (job/"registration.json").exists() or reservation["nonce"] != declaration["member_nonce"]:
            raise ValueError("Family job root must be unused and explicitly reserved")
        nonce, family_sha = reservation["nonce"],reservation["family_sha256"]
    now = time.time()
    reg = {"plan": plan, "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
           "workspace": str(workspace), "started_unix": now, "deadline_unix": now + plan["maximum_seconds"],
           "shutdown_authorized": False, "nonce": nonce, "plan_path":str(plan_path)}
    if family_sha is not None:
        reg["family_sha256"] = family_sha
        family_binding(job, reg)
        family_resources(job, reg, already_locked=True)
    save(job / "registration.json", reg)
    started = []
    try:
        for mode in ("guard", "worker"):
            if mode == "worker":
                end = time.monotonic() + HANDSHAKE
                while not (job / "guard-armed.json").exists():
                    if not matches(started[0][1]) or time.monotonic() >= end:
                        raise RuntimeError("Guard failed before armed handshake")
                    time.sleep(POLL)
                guard_alive(job, reg)
            with (job / (mode + ".log")).open("xb") as stream:
                child = subprocess.Popen([sys.executable, __file__, mode, "--job", str(job)],
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            owner = bind(child)
            started.append((child, owner))
            save(job / (mode + "-pid.json"), owner)
        guard_alive(job, reg)
    except BaseException as exc:
        failures = []
        for child, owner in reversed(started):
            try:
                stop_group({"leader": owner, "members": [owner]})
                child.wait(timeout=3)
            except BaseException as cleanup_error:
                failures.append(type(cleanup_error).__name__)
        save(job / "launch-failed.json", {"error_type": type(exc).__name__,
             "cleanup_errors": failures, "shutdown_requested": False})
        raise
    print(json.dumps({"job": job.name, "launched": True, "guard_handshake_verified": True,
                      "shutdown_authorized": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("launch", "worker", "guard", "register-family"))
    parser.add_argument("--job")
    parser.add_argument("--workspace")
    parser.add_argument("--plan")
    parser.add_argument("--spec")
    args = parser.parse_args()
    if args.mode == "register-family":
        if not args.spec:parser.error("register-family requires --spec")
        print(json.dumps(register_family(Path(args.spec)),sort_keys=True))
    elif not args.job:
        parser.error("--job required")
    elif args.mode == "launch":
        launch(args)
    elif args.mode == "guard":
        guard(Path(args.job))
    else:
        sys.exit(worker(Path(args.job)))
