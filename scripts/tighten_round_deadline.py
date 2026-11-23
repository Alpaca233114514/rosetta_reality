"""Add a shorter, identity-bound work deadline without rewriting a live run."""
from __future__ import annotations
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


def load_core(job):
    raw = (job / 'registration.json').read_bytes()
    reg = json.loads(raw)
    source = Path(reg['workspace']) / 'scripts/run_round_experiment_job.py'
    if hashlib.sha256(source.read_bytes()).hexdigest() != reg['plan']['source_sha256']['scripts/run_round_experiment_job.py']:
        raise ValueError('Original supervisor identity changed')
    spec = importlib.util.spec_from_file_location('bound_round_supervisor', source)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    core.registration(job)
    return core, reg, hashlib.sha256(raw).hexdigest()


def run(job, output):
    core, reg, reg_sha = load_core(job)
    owner = core.read(job / 'worker-pid.json')
    declared = core.read(output / 'deadline-registration.json')
    if declared['original_registration_sha256'] != reg_sha or declared['worker'] != owner:
        raise ValueError('Worker or registration changed')
    deadline = min(reg['deadline_unix'], reg['started_unix'] + 7200)
    if declared['deadline_unix'] != deadline:
        raise ValueError('Deadline changed')
    end = time.monotonic() + max(0, deadline - time.time())
    core.save(output / 'armed.json', {'process': core.process(os.getpid()),
        'original_registration_sha256': reg_sha, 'deadline_unix': deadline, 'shutdown_requested': False})
    status = 'error'
    cleanup = None
    try:
        while core.matches(owner) and not (job / 'worker-exited.json').exists():
            remaining = end - time.monotonic()
            if remaining <= 0:
                status = 'work_deadline_reached'
                if core.matches(owner):
                    os.kill(owner['pid'], signal.SIGTERM)
                grace = time.monotonic() + 30
                while core.matches(owner) and time.monotonic() < grace:
                    time.sleep(.2)
                if core.matches(owner):
                    core.stop_group({'leader': owner, 'members': [owner]})
                cleanup = core.cleanup_stage(job, 'tightened-deadline')
                break
            time.sleep(min(5, remaining))
        else:
            cleanup = core.cleanup_stage(job, 'tightened-deadline')
            status = 'worker_exited_before_deadline'
    finally:
        core.save(output / 'exited.json', {'status': status, 'cleanup': cleanup,
            'finished_unix': time.time(), 'shutdown_requested': False})


def launch(job, output):
    core, reg, reg_sha = load_core(job)
    owner = core.read(job / 'worker-pid.json')
    if not core.matches(owner) or (job / 'worker-exited.json').exists():
        raise RuntimeError('No live registered worker to constrain')
    deadline = min(reg['deadline_unix'], reg['started_unix'] + 7200)
    output.mkdir(exist_ok=False)
    core.save(output / 'deadline-registration.json', {'original_registration_sha256': reg_sha,
        'worker': owner, 'original_deadline_unix': reg['deadline_unix'],
        'started_unix': reg['started_unix'], 'deadline_unix': deadline,
        'maximum_total_seconds': 7200, 'cleanup_grace_seconds': 30,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'shutdown_authorized': False})
    with (output / 'guard.log').open('xb') as log:
        child = subprocess.Popen([sys.executable, __file__, 'run', '--job', str(job), '--output', str(output)],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    process = core.bind(child)
    core.save(output / 'guard-pid.json', process)
    until = time.monotonic() + 10
    while not (output / 'armed.json').exists():
        if child.poll() is not None or time.monotonic() >= until:
            raise RuntimeError('Tightened guard failed to arm')
        time.sleep(.1)
    if not core.matches(process):
        raise RuntimeError('Tightened guard died after handshake')
    print(json.dumps({'armed': True, 'deadline_unix': deadline,
        'maximum_total_seconds': 7200, 'original_run_registration_unchanged': True,
        'shutdown_authorized': False, 'process': process}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['launch','run'])
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    (launch if args.mode == 'launch' else run)(args.job.resolve(strict=True), args.output.resolve())
