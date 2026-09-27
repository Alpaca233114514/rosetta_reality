"""Independent absolute deadline, worker timeout and protected platform shutdown."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

JOB = Path(__file__).resolve().parent
EXPECTED = '0358e83eeeaf542aa98f64ba9e339c91df46f1e025892d52dba159f4fb1cf027'

def save(name, value):
    with (JOB / name).open('x') as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

def stop_worker():
    p = JOB / 'worker-pid.json'
    if not p.exists():
        return
    ident = json.loads(p.read_text())
    proc = Path('/proc') / str(ident['pid'])
    try:
        if proc.joinpath('stat').read_text().split()[21] != ident['start_ticks']:
            return
        os.killpg(ident['pid'], signal.SIGTERM)
        time.sleep(3)
        if proc.exists() and proc.joinpath('stat').read_text().split()[21] == ident['start_ticks']:
            os.killpg(ident['pid'], signal.SIGKILL)
    except ProcessLookupError:
        pass

def shutdown():
    wrapper = Path('/usr/bin/shutdown')
    if hashlib.sha256(wrapper.read_bytes()).hexdigest() != EXPECTED:
        raise RuntimeError('platform shutdown wrapper hash changed')
    trash = Path('/root/.local/share/Trash')
    if trash.exists() or trash.is_symlink():
        raise RuntimeError('platform shutdown would delete existing Trash; manual platform shutdown required')
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
    if gpu.returncode or gpu.stdout.strip():
        raise RuntimeError('unrelated GPU process prevents shutdown')
    sessions = subprocess.run(['tmux', 'list-sessions'], capture_output=True, text=True, timeout=10)
    if sessions.stdout.strip() or sessions.returncode not in (0, 1):
        raise RuntimeError('unrelated tmux session prevents shutdown')
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            argv = p.joinpath('cmdline').read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b'rosetta' in argv and any(x in argv for x in (b'train_', b'run_smolvla', b'evaluate_', b'pytest', b'audit.py')):
            raise RuntimeError('unrelated project worker prevents shutdown')
    save('shutdown-request.json', {'observed_unix': time.time(), 'shutdown_wrapper_sha256': EXPECTED,
                                 'release_requested': False, 'platform_stopped_independently_verified': False})
    os.sync()
    if trash.exists() or trash.is_symlink():
        raise RuntimeError('Trash appeared before shutdown')
    os.execv('/bin/bash', ['bash', str(wrapper)])

reg = json.loads(JOB.joinpath('registration.json').read_text())
save('guard-armed.json', {'pid': os.getpid(), 'work_deadline_unix': reg['work_deadline_unix'],
                        'shutdown_deadline_unix': reg['shutdown_deadline_unix'], 'observed_unix': time.time()})
early = None
while time.time() < reg['shutdown_deadline_unix']:
    if time.time() >= reg['work_deadline_unix']:
        stop_worker()
    if (JOB / 'closeout-now').exists():
        break
    if any((JOB / p).exists() for p in ['worker-exited.json', 'failure.json']):
        early = early or time.time() + 120
        if time.time() >= early:
            break
    time.sleep(2)
stop_worker()
try:
    shutdown()
except BaseException as exc:
    save('shutdown-blocked.json', {'error': str(exc), 'observed_unix': time.time()})
    raise
