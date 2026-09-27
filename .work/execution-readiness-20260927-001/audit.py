"""Bounded metadata-only admission audit. Never loads models or dataset rows."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

JOB = Path(__file__).resolve().parent
ROOT = JOB.parents[1]

def save(name, value):
    with (JOB / name).open('x') as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    start = time.monotonic()
    rows, errors, visited = [], [], 0
    # Report metadata only; ignore symlinks and all model/data payloads.
    for directory, dirs, files in os.walk(ROOT / 'runs', followlinks=False):
        dirs[:] = [d for d in dirs if not Path(directory, d).is_symlink() and d not in {'traces', 'trackio', 'pycache', '__pycache__', 'compiler_cache', 'history'}]
        if time.monotonic() - start > 150:
            raise TimeoutError('metadata inventory exceeded 150 seconds')
        for name in files:
            if not (name.startswith('gate3') and name.endswith('.json')):
                continue
            path = Path(directory, name)
            if path.is_symlink() or path.stat().st_size > 2 * 1024 * 1024:
                continue
            visited += 1
            if visited > 1000:
                raise RuntimeError('gate report count exceeds registered bound')
            try:
                raw = path.read_bytes()
                value = json.loads(raw)
                rows.append({'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(raw).hexdigest(),
                             'status': value.get('status'), 'passed': value.get('passed'),
                             'base_reference': b'prepost-base' in raw or b'7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb' in raw,
                             'code_identity': value.get('code_identity'),
                             'simulation_plan_sha256': value.get('simulation_plan_sha256'),
                             'artifact_manifest_sha256': value.get('artifact_manifest_sha256'),
                             'acceptance_criteria': value.get('acceptance_criteria')})
            except Exception as exc:
                errors.append({'path': str(path.relative_to(ROOT)), 'error': type(exc).__name__})
    dataset = ROOT / 'datasets_external/targeted-20260927-001'
    receipts = {}
    for name in ['result-stream.json', 'independent-verification-stream.json', 'closeout-observation.json', 'exit-code-stream.txt', 'provenance.sha256']:
        p = dataset / name
        receipts[name] = {'exists': p.is_file(), 'sha256': sha(p) if p.is_file() else None}
        if p.is_file() and p.suffix == '.json':
            receipts[name]['value'] = json.loads(p.read_bytes())
    candidates = [row for row in rows if row['base_reference']]
    disk = shutil.disk_usage(ROOT)
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used', '--format=csv,noheader'], capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    result = {'run': JOB.name, 'observed_unix': time.time(), 'status': 'blocked_on_existing_prerequisites',
              'execution_authorized_by_current_user': True, 'model_loaded': False, 'dataset_rows_loaded': False,
              'optimizer_updates': 0, 'simulation_rollouts': 0, 'nested_docker_used': False,
              'runtime_profile_reference': 'configs/runtime/autodl_rtx4090.yaml', 'gpu': gpu,
              'memory_limit': Path('/sys/fs/cgroup/memory.max').read_text().strip(),
              'disk_available_bytes': disk.free, 'gate_reports_scanned': visited, 'gate3_reports': rows,
              'gate_report_parse_errors': errors, 'base_gate3_candidates': candidates,
              'exp1': {'status': 'blocked_on_existing_valid_base_gate3_binding', 'gate4': 'not_measured',
                       'candidate_presence_is_acceptance': False},
              'exp2_exp3': {'status': 'blocked_on_qualified_matched_cohort', 'qualified_cohort_manifest': None,
                            'N': None, 'recovery_bank': None, 'cross_embodiment_substitution_allowed': False},
              'dataset_receipts': receipts,
              'qualification_audit_completed': False,
              'elapsed_seconds': time.monotonic() - start}
    save('result.json', result)
    save('worker-exited.json', {'finished_unix': time.time(), 'status': result['status']})

if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        save('failure.json', {'error': type(exc).__name__, 'message': str(exc), 'observed_unix': time.time()})
        raise
