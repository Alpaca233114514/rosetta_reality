"""Record final file-only download evidence; never starts model work."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

root=Path('/root/autodl-tmp/rosetta/datasets_external/targeted-20260927-001')
result=json.loads((root/'result-stream.json').read_text())
verification=json.loads((root/'independent-verification-stream.json').read_text())
plan=json.loads((root/'plan.json').read_text())
exit_code=int((root/'exit-code-stream.txt').read_text().strip())
assert result['complete'] and not result['failures'] and exit_code==0
assert result['files']==len(plan['files'])==len(verification['files'])==212
assert result['bytes']==plan['total_bytes']==verification['verified_bytes']==66158775113
assert not verification['failures']
old=[json.loads(line) for line in (root/'existing-verification.jsonl').read_text().splitlines()]
assert len(old)==2 and all(item['verified'] for item in old)
assert sum(item['files'] for item in old)==12 and sum(item['bytes'] for item in old)==171563018
for group in plan['groups']:
    expected=[x for x in plan['files'] if x['dataset']==group['dataset']]
    manifest=json.loads((root/expected[0]['destination']/'rosetta_download_manifest.json').read_text())
    actual={x['path']:(x['bytes'],x['sha256']) for x in manifest['files']}
    verified={x['path']:(x['bytes'],x['sha256']) for x in verification['files'] if x['dataset']==group['dataset']}
    assert actual==verified and len(actual)==group['files']
    assert sum(x[0] for x in actual.values())==group['bytes']

def read_cgroup(name):
    file=Path('/sys/fs/cgroup')/name
    return file.read_text().strip() if file.exists() else None

disk=shutil.disk_usage(root)
events=dict(line.split() for line in read_cgroup('memory.events').splitlines())
memory=dict(line.split() for line in read_cgroup('memory.stat').splitlines())
check=subprocess.run(['pgrep','-f','^python -u /root/autodl-tmp/rosetta/datasets_external/targeted-20260927-001/stream.py$'],capture_output=True,text=True)
assert check.returncode==1, 'download worker must have exited'
observed={'observed_utc':datetime.now(timezone.utc).isoformat(),'runtime_boundary':'autodl_platform_container','nested_docker_used':False,'mode':'no_gpu_dataset_staging','runtime_profile_reference':'configs/runtime/autodl_rtx4090.yaml','gpu_doctor_run':False,'model_work_started':False,'experiment_task_started':False,'python':sys.version.split()[0],'linux_kernel_release':platform.release(),'worker_running':False,'worker_exit_code':exit_code,'result_complete':True,'new_files':212,'new_bytes':66158775113,'old_verified_files':12,'old_verified_bytes':171563018,'disk':{'total':disk.total,'used':disk.used,'available':disk.free},'memory':{'limit':read_cgroup('memory.max'),'current':read_cgroup('memory.current'),'anon':int(memory['anon']),'file_cache':int(memory['file']),'events':{k:int(v) for k,v in events.items()}},'raw_row_and_video_decoding_verified':False,'platform_shutdown_verified':False}
with (root/'closeout-observation.json').open('x') as out:
    json.dump(observed,out,sort_keys=True,indent=2)
names=['download_core.py','worker.py','parallel.py','parallel8.py','stream.py','rosetta_dataset_stream.py','subset-audit.py','source-inventory.py','closeout.py','plan.json','parallel-execution.json','parallel8-execution.json','stream-execution.json','rj45-subset-metadata-audit.json','existing-verification.jsonl','independent-verification-stream.json','result-stream.json','exit-code-stream.txt','closeout-observation.json','download.log','download-parallel.log','download-parallel8.log','download-stream.log']
names.extend(str(path.relative_to(root)) for path in sorted(root.glob('**/rosetta_download_manifest.json')))
with (root/'provenance.sha256').open('x') as out:
    for name in names:
        digest=hashlib.sha256((root/name).read_bytes()).hexdigest()
        out.write(f'{digest}  {name}\n')
print(json.dumps(observed,sort_keys=True),flush=True)
print(json.dumps({'receipt_hashes':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['plan.json','independent-verification-stream.json','result-stream.json','provenance.sha256']}}),flush=True)
