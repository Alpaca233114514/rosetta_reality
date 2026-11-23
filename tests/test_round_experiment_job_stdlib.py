"""Real Linux process lifecycle tests; run only in the authorized container."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_round_experiment_job.py"
spec = importlib.util.spec_from_file_location("round_lifecycle_test", SCRIPT)
round_job = importlib.util.module_from_spec(spec)
spec.loader.exec_module(round_job)
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux process groups required")


def await_file(path, seconds=15):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if path.exists():
            return json.loads(path.read_text())
        time.sleep(.05)
    raise AssertionError("Expected durable lifecycle receipt: " + path.name)


def launch(tmp_path, code, stage_seconds=8, total_seconds=20, script=SCRIPT):
    workspace = tmp_path / "workspace"
    (workspace / "scripts").mkdir(parents=True)
    copied = workspace / "scripts/run_round_experiment_job.py"
    copied.write_bytes(script.read_bytes())
    plan = {"execution_authorized":True,"shutdown_authorized":False,
            "maximum_seconds":total_seconds,"minimum_free_bytes":1,
            "maximum_rss_bytes":256*1024**2,"maximum_output_bytes":16*1024**2,
            "source_sha256":{"scripts/run_round_experiment_job.py":hashlib.sha256(copied.read_bytes()).hexdigest()},
            "commands":[{"name":"stage","argv":["{python}","-c",code,"{job}"],
                         "maximum_seconds":stage_seconds,"allowed_exit_codes":[0]}]}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    job = tmp_path / "job"
    result = subprocess.run([sys.executable,str(copied),"launch","--workspace",str(workspace),
                             "--plan",str(plan_path),"--job",str(job)],capture_output=True,text=True,timeout=20)
    return job,result


def cleanup(job):
    if (job/"active-stage.json").exists():
        round_job.stop_group(json.loads((job/"active-stage.json").read_text()),grace=.2)
    for name in ("worker","guard"):
        path=job/(name+"-pid.json")
        if path.exists():
            owner=json.loads(path.read_text())
            round_job.stop_group({"leader":owner,"members":[owner]},grace=.2)


TREE = '''import pathlib,signal,subprocess,sys,time
signal.signal(signal.SIGTERM,signal.SIG_IGN)
subprocess.Popen([sys.executable,"-c","import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"])
pathlib.Path(sys.argv[1],"started").write_text("ready")
time.sleep(60)
'''


def await_group(job):
    end=time.monotonic()+8
    while time.monotonic()<end:
        path=job/"active-stage.json"
        if path.exists():
            receipt=json.loads(path.read_text())
            if len(receipt["members"])>=2:
                return receipt
        time.sleep(.05)
    raise AssertionError("Stage tree member identities were not retained")


def test_normal_completion_has_guard_handshake_and_final_empty_groups(tmp_path):
    job,result=launch(tmp_path,"import time; time.sleep(.3)")
    try:
        assert result.returncode==0,result.stderr
        assert json.loads(result.stdout)["guard_handshake_verified"] is True
        exited=await_file(job/"worker-exited.json")
        guard=await_file(job/"guard-exited.json")
        assert exited["error_type"] is None
        assert exited["outcomes"]==[{"name":"stage","returncode":0}]
        assert exited["cleanup"]["status"]=="owned_group_empty"
        assert guard["error_type"] is None and guard["shutdown_requested"] is False
        assert not round_job.members(json.loads((job/"stage-process.json").read_text())["leader"]["group"])
    finally:cleanup(job)


def test_killed_guard_triggers_worker_cleanup_of_term_ignoring_tree(tmp_path):
    job,result=launch(tmp_path,TREE)
    try:
        assert result.returncode==0,result.stderr
        tree=await_group(job)
        guard=json.loads((job/"guard-pid.json").read_text())
        assert round_job.matches(guard)
        os.kill(guard["pid"],signal.SIGKILL)
        exited=await_file(job/"worker-exited.json")
        assert exited["error_type"]=="RuntimeError"
        assert exited["cleanup"]["status"]=="owned_group_empty"
        assert not round_job.members(tree["leader"]["group"])
    finally:cleanup(job)


def test_stage_deadline_cleans_every_bound_tree_member(tmp_path):
    job,result=launch(tmp_path,TREE,stage_seconds=2)
    try:
        assert result.returncode==0,result.stderr
        tree=await_group(job)
        exited=await_file(job/"worker-exited.json")
        assert exited["error_type"]=="TimeoutError"
        assert not round_job.members(tree["leader"]["group"])
        await_file(job/"guard-exited.json")
    finally:cleanup(job)


def test_failed_guard_never_starts_worker_or_stage(tmp_path):
    altered=tmp_path/"failed-guard.py"
    altered.write_text(SCRIPT.read_text().replace("guard(Path(args.job))","raise RuntimeError('intentional guard startup failure')"))
    job,result=launch(tmp_path,TREE,script=altered)
    try:
        assert result.returncode!=0
        assert (job/"launch-failed.json").exists()
        assert not (job/"worker-pid.json").exists()
        assert not (job/"active-stage.json").exists()
    finally:cleanup(job)


def test_reused_leader_receipt_does_not_signal_unowned_pid(monkeypatch):
    fake={"pid":99199,"start_ticks":"11","group":99199,"session":99199}
    monkeypatch.setattr(round_job,"process",lambda pid:{**fake,"start_ticks":"12","state":"S"})
    monkeypatch.setattr(round_job,"members",lambda group:[])
    calls=[]
    monkeypatch.setattr(round_job.os,"kill",lambda *args:calls.append(args))
    monkeypatch.setattr(round_job.os,"killpg",lambda *args:calls.append(args))
    assert round_job.stop_group({"leader":fake,"members":[fake]},grace=0)["status"]=="owned_group_empty"
    assert calls==[]
