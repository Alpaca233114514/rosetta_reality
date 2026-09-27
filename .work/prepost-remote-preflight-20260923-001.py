"""Read-only metadata/hash inventory; no torch/model/data imports or writes."""
import hashlib
import importlib.metadata
import json
import pathlib
import struct
import sys
import time

ROOT = pathlib.Path('/root/autodl-tmp/rosetta')
ENV = ROOT / 'envs/smolvla-cuda-001/lib/python3.12/site-packages'
WORKSPACE = ROOT / 'workspaces/20260916T100517Z-95cf9cf9483b-6bb3da3027df'

def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()

def header(path):
    with path.open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        if length > 16 * 1024 * 1024:
            raise ValueError('Oversized header')
        value = json.loads(stream.read(length))
    return {key: {'dtype': data['dtype'], 'shape': data['shape']}
            for key, data in value.items() if key != '__metadata__'}

report = {'id': 'prepost-remote-preflight-20260923-001',
          'scope': 'no_gpu_read_only_metadata', 'time_unix': time.time(),
          'python_version': sys.version.split()[0], 'nested_docker_used': False,
          'model_loaded': False, 'optimizer_steps': 0, 'files': {}, 'packages': {},
          'cgroup': {}, 'sources': {}, 'optimizer_headers': {}}
for name in ('memory.max', 'memory.swap.max', 'cpu.max'):
    report['cgroup'][name] = (pathlib.Path('/sys/fs/cgroup') / name).read_text().strip()
for name in ('torch', 'lerobot', 'transformers', 'safetensors', 'torchlens', 'accelerate'):
    try:
        report['packages'][name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        report['packages'][name] = None
base = ROOT / 'models/lerobot--smolvla_base/c83c3163b8ca9b7e67c509fffd9121e66cb96205'
sources = [(base / 'model.safetensors', '7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb')]
checkpoint_root = ROOT / 'checkpoints/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/smoke/canonical-fullframes-20260914-001/checkpoints'
if not checkpoint_root.exists():
    candidates = list((ROOT / 'checkpoints').glob('**/canonical-fullframes-20260914-001*/checkpoints'))
    if len(candidates) != 1:
        raise ValueError('Canonical checkpoint root is not uniquely identified')
    checkpoint_root = candidates[0]
for step, expected in ((2500, 'a3494600183d0a24c71ab7db0976afbc57567ac714ac678fd572c607e06a2f0c'),
                       (5000, 'd4c0d87cd8e66c723b07ec84f7f5e47875268bfd4e2057ec5683fdf56adcabef')):
    folder = checkpoint_root / f'{step:06d}'
    sources.append((folder / 'pretrained_model/model.safetensors', expected))
    h = header(folder / 'training_state/optimizer_state.safetensors')
    report['optimizer_headers'][str(step)] = {'tensors': len(h), 'first_keys': list(h)[:12],
        'shape720_entries': {key: value for key, value in h.items() if value['shape'] == [720]}}
for path, expected in sources:
    actual = sha(path)
    report['files'][str(path.relative_to(ROOT))] = {
        'bytes': path.stat().st_size, 'sha256': actual, 'expected_sha256': expected,
        'matched': actual == expected}
for name in ('policies/smolvla/modeling_smolvla.py', 'policies/smolvla/smolvlm_with_expert.py',
             'policies/factory.py', 'policies/pretrained.py', 'scripts/lerobot_train.py',
             'optim/optimizers.py'):
    path = ENV / 'lerobot' / name
    report['sources']['lerobot/' + name] = sha(path)
report['runtime_profile_sha256'] = sha(WORKSPACE / 'configs/runtime/autodl_rtx4090.yaml')
report['workspace_archive_marker'] = (WORKSPACE / '.rosetta-workspace.sha256').read_text().split()[0]
report['cuda_available_observed_by_nvidia_smi'] = False
report['cuda_stage_ready'] = False
report['all_model_hashes_match'] = all(item['matched'] for item in report['files'].values())
print(json.dumps(report, indent=2))
if not report['all_model_hashes_match']:
    sys.exit(1)
