"""Real Linux CPU family lifecycle and mocked driver admission, never CUDA proof."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/run_round_experiment_job.py'
spec = importlib.util.spec_from_file_location('round_family', SCRIPT)
job = importlib.util.module_from_spec(spec)
spec.loader.exec_module(job)
pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='Linux lifecycle only')


def wait(path, seconds=20):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if path.exists():
            return json.loads(path.read_text())
        time.sleep(.05)
    raise AssertionError(str(path))


def setup_family(tmp_path, caps=(256*1024**2, 256*1024**2)):
    workspace = tmp_path / 'workspace'
    (workspace / 'scripts').mkdir(parents=True)
    script = workspace / 'scripts/run_round_experiment_job.py'
    script.write_bytes(SCRIPT.read_bytes())
    manifest = tmp_path / 'family.json'
    rows = []
    for i, cap in enumerate(caps):
        nonce = uuid.uuid4().hex
        plan = {'execution_authorized':True, 'shutdown_authorized':False,
                'gpu_required':False, 'maximum_gpu_bytes':0, 'maximum_seconds':25,
                'minimum_free_bytes':1, 'maximum_rss_bytes':cap, 'maximum_output_bytes':16*1024**2,
                'resource_family':{'family_id':'test-family', 'member_nonce':nonce,
                                   'manifest_path':str(manifest)},
                'source_sha256':{'scripts/run_round_experiment_job.py':hashlib.sha256(script.read_bytes()).hexdigest()},
                'commands':[{'name':'stage', 'argv':['{python}','-c',
                    "import pathlib,sys,time;pathlib.Path(sys.argv[1],'started').write_text('ready');time.sleep(4)",'{job}'],
                    'maximum_seconds':10, 'allowed_exit_codes':[0]}]}
        pp = tmp_path / f'plan-{i}.json'
        pp.write_text(json.dumps(plan))
        rows.append({'job_root':str(tmp_path/f'job-{i}'), 'workspace':str(workspace),
                     'plan_path':str(pp), 'plan_sha256':hashlib.sha256(pp.read_bytes()).hexdigest(),
                     'nonce':nonce, 'maximum_rss_bytes':cap, 'maximum_gpu_bytes':0})
    family = {'manifest_path':str(manifest), 'family_id':'test-family', 'shutdown_authorized':False,
              'maximum_rss_bytes':sum(caps), 'maximum_gpu_bytes':0, 'gpu_uuid':None,
              'gpu_total_bytes':0, 'members':rows}
    sp = tmp_path/'spec.json';sp.write_text(json.dumps(family))
    return script, sp, family


def command(script, *args):
    return subprocess.run([sys.executable,str(script),*args],capture_output=True,text=True,timeout=20)


def cleanup(root):
    active = root/'active-stage.json'
    if active.exists():job.stop_group(json.loads(active.read_text()),grace=.1)
    for name in ('worker','guard'):
        path=root/(name+'-pid.json')
        if path.exists():
            owner=json.loads(path.read_text());job.stop_group({'leader':owner,'members':[owner]},grace=.1)


def test_two_registered_cpu_members_overlap_and_close_owned_groups(tmp_path):
    script, sp, family=setup_family(tmp_path)
    result=command(script,'register-family','--spec',str(sp))
    assert result.returncode==0,result.stderr
    roots=[Path(r['job_root']) for r in family['members']]
    try:
        for row in family['members']:
            result=command(script,'launch','--job',row['job_root'],'--workspace',row['workspace'],'--plan',row['plan_path'])
            assert result.returncode==0,result.stderr
        for root in roots:wait(root/'stage-process.json')
        owners=[json.loads((root/'worker-pid.json').read_text()) for root in roots]
        assert all(job.matches(owner) for owner in owners), 'Both workers must overlap'
        for root in roots:
            reg=job.registration(root)
            assert job.family_resources(root,reg)['members_with_processes']==2
        for root in roots:
            result=wait(root/'worker-exited.json');guard=wait(root/'guard-exited.json')
            assert result['error_type'] is None and guard['error_type'] is None
            assert result['outcomes']==[{'name':'stage','returncode':0}]
            assert result['cleanup']['status']=='owned_group_empty'
            assert not job.members(json.loads((root/'stage-process.json').read_text())['leader']['group'])
    finally:
        for root in roots:cleanup(root)


@pytest.mark.parametrize('mutation',['rss','cpu_gpu_claim','occupied_root','plan_sha','nonce'])
def test_invalid_family_never_reserves_or_launches(tmp_path,mutation):
    script, sp, family=setup_family(tmp_path)
    if mutation=='rss':family['maximum_rss_bytes']=16*1024**3+1
    if mutation=='cpu_gpu_claim':family['maximum_gpu_bytes']=1
    if mutation=='occupied_root':Path(family['members'][0]['job_root']).mkdir()
    if mutation=='plan_sha':family['members'][0]['plan_sha256']='0'*64
    if mutation=='nonce':family['members'][0]['nonce']='a'*32
    sp.write_text(json.dumps(family))
    result=command(script,'register-family','--spec',str(sp))
    assert result.returncode!=0
    assert not Path(family['manifest_path']).exists()
    assert not any((Path(r['job_root'])/'registration.json').exists() for r in family['members'])


def test_changed_reserved_plan_is_rejected_before_worker(tmp_path):
    script,sp,family=setup_family(tmp_path)
    assert command(script,'register-family','--spec',str(sp)).returncode==0
    row=family['members'][0];pp=Path(row['plan_path'])
    pp.write_text(pp.read_text()+' ')
    result=command(script,'launch','--job',row['job_root'],'--workspace',row['workspace'],'--plan',row['plan_path'])
    assert result.returncode!=0
    assert not (Path(row['job_root'])/'worker-pid.json').exists()


def gpu_case():
    record=job.process(os.getpid())
    family={'gpu_uuid':'gpu-test','gpu_total_bytes':10000,'maximum_gpu_bytes':9000,
            'members':[{'job_root':'a','maximum_gpu_bytes':5000},{'job_root':'b','maximum_gpu_bytes':4000}]}
    snapshot={'uuid':'gpu-test','total_bytes':10000,'used_bytes':7000,
              'processes':[{'gpu_uuid':'gpu-test','pid':os.getpid(),'bytes':4000}]}
    records={'a':{(record['pid'],record['start_ticks']):record},'b':{}}
    return family,snapshot,records


def test_driver_unknown_pid_and_reused_pid_cannot_gain_ownership():
    family,snapshot,records=gpu_case()
    assert job.check_family_gpu(family,records,snapshot)=={'a':4000,'b':0}
    snapshot['processes'][0]['pid']=987654321
    with pytest.raises(RuntimeError):job.check_family_gpu(family,records,snapshot)
    snapshot['processes'][0]['pid']=os.getpid()
    record=next(iter(records['a'].values()));record['start_ticks']='0'
    with pytest.raises(RuntimeError):job.check_family_gpu(family,records,snapshot)


@pytest.mark.parametrize('kind',['member','joint','device','ambiguous'])
def test_driver_registered_gpu_budget_checks(kind):
    family,snapshot,records=gpu_case()
    if kind=='member':snapshot['processes'][0]['bytes']=5001
    if kind=='joint':snapshot['used_bytes']=9001
    if kind=='device':snapshot['uuid']='different-gpu'
    if kind=='ambiguous':records['b']=dict(records['a'])
    with pytest.raises((MemoryError,ValueError)):job.check_family_gpu(family,records,snapshot)
