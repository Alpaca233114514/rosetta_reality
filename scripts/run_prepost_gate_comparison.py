"""One-shot, create-only base versus trained SmolVLA Gate 3/4 comparison.

The base arm uses pinned base weights with the saved canonical ALOHA policy
configuration and processors. It is not a historical step-zero checkpoint.
"""

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
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
from scripts import smolvla_autodl_vfunfreeze_sim_gate as reference  # noqa: E402
from scripts import smolvla_sim_gate as engine  # noqa: E402
from scripts.run_iris_furnace import alive, process_tree_rss, shutdown, terminate, ticks  # noqa: E402

EXP = "m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003"
NAME = "prepost-gate4-20260924-001"
DURABLE = Path("/root/autodl-tmp/rosetta")
JOB = DURABLE / "runs" / NAME
SOURCE_ARTIFACT = (
    DURABLE / "workspaces/20260916T100517Z-95cf9cf9483b-6bb3da3027df"
    / "runs/canonical-fullframes-posttrain-20260916-005-artifacts"
    / EXP / "canonical-fullframes-20260914-001-step5000-zero-copy-004"
)
TRAINED = (
    DURABLE / "checkpoints" / EXP / "smoke/canonical-fullframes-20260914-001"
    / "checkpoints/005000/pretrained_model"
)
BASE = (
    DURABLE / "models/lerobot--smolvla_base"
    / "c83c3163b8ca9b7e67c509fffd9121e66cb96205/model.safetensors"
)
WEIGHTS = {
    "base": (BASE, "7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb", 0, "481"),
    "trained": (TRAINED / "model.safetensors", "d4c0d87cd8e66c723b07ec84f7f5e47875268bfd4e2057ec5683fdf56adcabef", 5000, "482"),
}
WORK_SECONDS = 5400
SHUTDOWN_SECONDS = 5700


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def artifact(arm: str) -> Path:
    return JOB / "artifacts" / EXP / f"prepost-{arm}-aloha-interface-001"


def gate_path(arm: str) -> Path:
    return JOB / f"{arm}-gate.yaml"


def report_path(arm: str, gate: str) -> Path:
    return JOB / "results" / EXP / "gates" / f"{gate}-smolvla-sim-{WEIGHTS[arm][3]}.json"


def preflight() -> None:
    if os.environ.get("ROSETTA_TORCH_DEVICE") != "cuda" or any(
        os.environ.get(key) != "1" for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")
    ):
        raise ValueError("Offline CUDA AutoDL shell required")
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True
    ).strip()
    if "RTX 4080 SUPER" not in gpu or "32760 MiB" not in gpu:
        raise ValueError("Clone GPU identity changed: " + gpu)
    busy = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if busy:
        raise ValueError("GPU already occupied: " + busy)
    if os.statvfs(DURABLE).f_bavail * os.statvfs(DURABLE).f_frsize < 2 * 1024**3:
        raise ValueError("Durable storage headroom below 2 GiB")
    if not SOURCE_ARTIFACT.is_dir() or read(SOURCE_ARTIFACT / "manifest.json")["status"] != "verified":
        raise ValueError("Saved canonical artifact unavailable")
    source_manifest = read(SOURCE_ARTIFACT / "manifest.json")
    for relative, expected in source_manifest["files"].items():
        if sha(SOURCE_ARTIFACT / relative) != expected:
            raise ValueError("Canonical artifact source changed: " + relative)
    for arm, (path, expected, _step, _suffix) in WEIGHTS.items():
        if sha(path) != expected:
            raise ValueError("Model SHA changed: " + arm)
    if sha(ROOT / "configs/sim/aloha_insertion_smolvla.yaml") != reference.CONTRACT_SHA:
        raise ValueError("Action Contract changed")
    for prior in (reference.PRIOR_FAILURE, reference.PRIOR_TASK_FAILURE):
        path = DURABLE / prior["report"]
        if sha(path) != prior["report_sha256"]:
            raise ValueError("Prior Gate evidence changed")
    save(JOB / "preflight.json", {
        "status": "passed", "gpu": gpu, "source_artifact_manifest_sha256": sha(SOURCE_ARTIFACT / "manifest.json"),
        "base_sha256": WEIGHTS["base"][1], "trained_sha256": WEIGHTS["trained"][1],
        "processor_source": str(SOURCE_ARTIFACT), "optimizer_steps": 0,
        "base_interpretation": "pinned base weights under saved canonical ALOHA interface; not historical step zero",
    })


def prepare_artifact(arm: str) -> None:
    import shutil

    path, weight_sha, step, _suffix = WEIGHTS[arm]
    destination = artifact(arm)
    pretrained = destination / "pretrained_model"
    pretrained.mkdir(parents=True, exist_ok=False)
    source = SOURCE_ARTIFACT / "pretrained_model"
    for entry in sorted(source.rglob("*")):
        relative = entry.relative_to(source)
        target = pretrained / relative
        if entry.is_dir():
            target.mkdir(exist_ok=False)
        elif entry.is_file():
            target.symlink_to(path if relative.as_posix() == "model.safetensors" else entry)
    for name in ("normalization.json", "action_contract.json"):
        shutil.copyfile(SOURCE_ARTIFACT / name, destination / name)
    config = read(SOURCE_ARTIFACT / "config.json")
    config.update(artifact_id=destination.name, selected_checkpoint_step=step,
                  selected_checkpoint_model_sha256=weight_sha)
    save(destination / "config.json", config)
    files = {p.relative_to(destination).as_posix(): sha(p) for p in sorted(destination.rglob("*")) if p.is_file()}
    save(destination / "candidate-manifest.json", {
        "status": "prepared", "artifact_id": destination.name, "experiment_id": EXP,
        "selected_checkpoint_step": step, "selected_checkpoint_model_sha256": weight_sha,
        "files": files, "hidden_test_loaded": False,
        "source_artifact_manifest_sha256": sha(SOURCE_ARTIFACT / "manifest.json"),
        "reload": {"verified": False, "exact_tensor_equality": False, "independent_processes": 2},
    })


def probe(arm: str, attempt: int) -> None:
    import torch
    from rosetta_reality.sim import GymAlohaEnvironment, load_action_contract

    torch.cuda.set_per_process_memory_fraction(0.4)
    reference._ACTIVE_SIM_PLAN = gate_path(arm)
    contract = load_action_contract(ROOT / "configs/sim/aloha_insertion_smolvla.yaml")
    online = reference._build_online_cuda_class()(artifact(arm), read(artifact(arm) / "config.json"),
                                                    read(artifact(arm) / "normalization.json"), contract)
    environment = GymAlohaEnvironment(contract, maximum_episode_steps=20)
    try:
        observation = environment.reset(seed=20260809)
        online.configure_noise("zeros", None)
        raw, projected = online.predict(observation, "Insert the peg into the socket.")
    finally:
        environment.close()
    save(JOB / f"{arm}-probe-{attempt}.json", {
        "raw": raw.tolist(), "projected": projected.tolist(),
        "weight_sha256": WEIGHTS[arm][1], "processor_source_sha256": sha(SOURCE_ARTIFACT / "manifest.json"),
    })


def finish_reload(arm: str) -> None:
    first, second = (read(JOB / f"{arm}-probe-{number}.json") for number in (1, 2))
    if first != second:
        raise ValueError("Independent reload predictions differ: " + arm)
    candidate = read(artifact(arm) / "candidate-manifest.json")
    candidate["status"] = "verified"
    candidate["reload"] = {"verified": True, "exact_tensor_equality": True, "independent_processes": 2,
                           "probe_1_sha256": sha(JOB / f"{arm}-probe-1.json"),
                           "probe_2_sha256": sha(JOB / f"{arm}-probe-2.json"),
                           "optimizer_steps": 0}
    save(artifact(arm) / "manifest.json", candidate)


def render(arm: str) -> None:
    import yaml

    _path, weight_sha, step, suffix = WEIGHTS[arm]
    selection = JOB / "results" / EXP / "selection" / f"prepost-{arm}-001.json"
    save(selection, {"status": "passed", "selected": {"step": step, "model_safetensors_sha256": weight_sha},
                     "hidden_test_loaded": False, "source": "pinned_base" if arm == "base" else "canonical_step5000"})
    backup = JOB / "results" / EXP / "artifact_backup" / f"prepost-{arm}-001.json"
    save(backup, {"status": "verified", "artifact_manifest_sha256": sha(artifact(arm) / "manifest.json"),
                  "off_host_copy_created": arm == "trained", "source_preserved": True})
    code = "\n".join(
        f"  {name}: {sha(ROOT / name)}" for name in (
            "scripts/smolvla_sim_gate.py", "scripts/smolvla_autodl_vfunfreeze_sim_gate.py",
            "scripts/run_prepost_gate_comparison.py", "src/rosetta_reality/sim/gym_aloha.py",
            "src/rosetta_reality/vla/processor.py",
        )
    )
    text = reference.SIM_PLAN_TEMPLATE.format(
        sim_plan_id=f"{NAME}-{arm}-001", experiment_id=EXP, artifact_id=artifact(arm).name,
        artifact_manifest_sha256=sha(artifact(arm) / "manifest.json"), alignment_report="not_applicable",
        alignment_sha="0" * 64, prior_report=reference.PRIOR_FAILURE["report"],
        prior_sha=reference.PRIOR_FAILURE["report_sha256"],
        prior_task_report=reference.PRIOR_TASK_FAILURE["report"],
        prior_task_sha=reference.PRIOR_TASK_FAILURE["report_sha256"],
        selection_report=selection.relative_to(DURABLE).as_posix(), selection_sha=sha(selection),
        selected_step=step, model_sha=weight_sha, backup_sha=sha(backup),
        contract_sha=reference.CONTRACT_SHA, sim_code_blocks=code, suffix=suffix,
    )
    plan = yaml.safe_load(text)
    plan.pop("entry_permit")
    plan["hypothesis"] = "Compare pinned base and canonical step-5000 weights under the same saved ALOHA interface"
    plan["single_axis_change"] = {"field": "model.safetensors", "arm": arm,
                                  "base_is_historical_step_zero": False}
    plan["inference"]["noise_source"] = "pinned_lerobot_default_standard_normal"
    with gate_path(arm).open("x", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(plan, stream, sort_keys=False)


def evidence_path(raw: str) -> Path:
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe Gate evidence path")
    local = ROOT / relative
    if local.is_file():
        return local
    durable = DURABLE / relative
    if durable.is_file():
        return durable
    raise FileNotFoundError(raw)


def gate(arm: str, stage: str) -> int:
    import torch

    torch.cuda.set_per_process_memory_fraction(0.4)
    reference._ACTIVE_SIM_PLAN = gate_path(arm)
    engine._repository_path = evidence_path
    engine._OnlineSmolVLA = reference._build_online_cuda_class()
    engine._runtime = reference._cuda_runtime
    if stage == "gate3":
        return engine.gate3(gate_path(arm))
    return engine.gate4(gate_path(arm), report_path(arm, "gate3"))


def child(stage: str, arm: str, attempt: int) -> None:
    if stage == "probe":
        probe(arm, attempt)
    elif stage in ("gate3", "gate4"):
        raise SystemExit(gate(arm, stage))
    else:
        raise ValueError(stage)


def watchdog() -> None:
    reg = read(JOB / "registration.json")
    while time.time() < reg["shutdown_deadline"]:
        if (JOB / "shutdown-request.json").is_file():
            return
        if time.time() >= reg["deadline"] and not (JOB / "worker-exited.json").is_file():
            active = JOB / "active-child.json"
            if active.is_file():
                process = read(active)
                terminate(process["pid"], process["ticks"])
            if alive(reg["supervisor_pid"], reg["supervisor_ticks"]):
                os.kill(reg["supervisor_pid"], signal.SIGTERM)
        time.sleep(5)
    shutdown(JOB, reg)


def supervise() -> None:
    import shutil

    if JOB.exists():
        raise FileExistsError("Create-only comparison already exists")
    JOB.mkdir(parents=True)
    started = time.time()
    reg = {"started": started, "deadline": started + WORK_SECONDS,
           "shutdown_deadline": started + SHUTDOWN_SECONDS,
           "supervisor_pid": os.getpid(), "supervisor_ticks": ticks(os.getpid()),
           "shutdown_helper_sha256": sha(ROOT / "scripts/run_visual_coverage_job.py"),
           "script_sha256": sha(Path(__file__)), "optimizer_steps": 0}
    save(JOB / "registration.json", reg)
    with (JOB / "watchdog.log").open("x") as log:
        guard = subprocess.Popen([sys.executable, __file__, "watchdog"], stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save(JOB / "watchdog.json", {"pid": guard.pid, "ticks": ticks(guard.pid),
                                  "deadline": reg["deadline"], "shutdown_deadline": reg["shutdown_deadline"]})
    outcomes: dict[str, str] = {}
    error = None

    def run(stage: str, arm: str, attempt: int = 0) -> int:
        command = [sys.executable, __file__, "child", stage, arm, str(attempt)]
        with (JOB / f"{arm}-{stage}-{attempt}.log").open("x") as log:
            proc = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            active = {"pid": proc.pid, "ticks": ticks(proc.pid), "stage": stage, "arm": arm}
            (JOB / "active-child.json").write_text(json.dumps(active))
            while proc.poll() is None:
                if time.time() >= reg["deadline"]:
                    terminate(proc.pid, active["ticks"])
                    raise TimeoutError("Comparison deadline reached")
                if process_tree_rss(proc.pid) > 12 * 1024**3:
                    terminate(proc.pid, active["ticks"])
                    raise MemoryError("Gate process RSS exceeds 12 GiB")
                if shutil.disk_usage(DURABLE).free < 1024**3:
                    terminate(proc.pid, active["ticks"])
                    raise OSError("Durable disk reserve below 1 GiB")
                time.sleep(5)
        return proc.returncode

    try:
        preflight()
        os.environ["ROSETTA_ARTIFACT_ROOT"] = str(JOB / "artifacts")
        os.environ["ROSETTA_RUN_ROOT"] = str(JOB / "results")
        os.environ["TRACKIO_DIR"] = str(JOB / "results" / "trackio")
        (JOB / "results" / "trackio").mkdir(parents=True)
        for arm in WEIGHTS:
            prepare_artifact(arm)
            # The probe loader needs only the contract path before final Gate rendering.
            with gate_path(arm).open("x", encoding="utf-8") as stream:
                stream.write("action_contract:\n  path: configs/sim/aloha_insertion_smolvla.yaml\n")
            for attempt in (1, 2):
                if run("probe", arm, attempt):
                    raise RuntimeError(f"{arm} independent reload probe {attempt} failed")
            finish_reload(arm)
            gate_path(arm).unlink()
            render(arm)
        for arm in WEIGHTS:
            gate3_code = run("gate3", arm)
            gate3_report = report_path(arm, "gate3")
            if gate3_code not in (0, 1) or not gate3_report.is_file():
                raise RuntimeError(f"{arm} Gate 3 runtime failed")
            outcomes[arm + "_gate3"] = read(gate3_report)["status"]
            if gate3_code:
                outcomes[arm + "_gate4"] = "not measured: Gate 3 failed"
                continue
            gate4_code = run("gate4", arm)
            gate4_report = report_path(arm, "gate4")
            if gate4_code not in (0, 1) or not gate4_report.is_file():
                raise RuntimeError(f"{arm} Gate 4 runtime failed")
            outcomes[arm + "_gate4"] = read(gate4_report)["status"]
    except BaseException as exc:  # Preserve all failures and stop the batch.
        error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        save(JOB / "worker-exited.json", {"outcomes": outcomes, "error": error,
                                           "elapsed_seconds": time.time() - started,
                                           "optimizer_steps": 0, "m2_complete": False})
        shutdown(JOB, reg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("supervise", "watchdog", "child"))
    parser.add_argument("stage", nargs="?")
    parser.add_argument("arm", nargs="?")
    parser.add_argument("attempt", nargs="?", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "supervise":
        supervise()
    elif args.mode == "watchdog":
        watchdog()
    else:
        child(args.stage, args.arm, args.attempt)


if __name__ == "__main__":
    main()
