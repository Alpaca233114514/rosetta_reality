"""Resume the sealed selection with eight bounded-memory HTTP streams."""
import concurrent.futures
import importlib.util
import json
from pathlib import Path
import shutil
import time

ROOT=Path('/root/autodl-tmp/rosetta/datasets_external/targeted-20260927-001')
spec=importlib.util.spec_from_file_location('worker',ROOT/'worker.py')
w=importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
import threading
log_lock=threading.Lock()
original_emit=w.emit
def synchronized_emit(**event):
    with log_lock:
        original_emit(**event)
w.emit=synchronized_emit
stream_spec=importlib.util.spec_from_file_location('rosetta_dataset_stream',ROOT/'rosetta_dataset_stream.py')
stream_module=importlib.util.module_from_spec(stream_spec)
stream_spec.loader.exec_module(stream_module)
stream_module.install(w)
plan=json.loads((ROOT/'plan.json').read_text())
started=time.time()
w.seal('stream-execution.json',{'workers':8,'read_block_bytes':8*1024*1024,'request_mode':'continuous-range-resumable','plan_sha256':w.digests(ROOT/'plan.json')['sha256'],'started_unix':started,'historical_execution_log':'download-parallel8.log','resume_partials':True})
records={}
failures=[]

def transfer(item):
    record=w.fetch(item)
    w.emit(status='file_complete',dataset=item['dataset'],path=item['path'],bytes=item['size'],sha256=record['sha256'])
    return item,record

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    jobs={pool.submit(transfer,item):item for item in plan['files']}
    for future in concurrent.futures.as_completed(jobs):
        item=jobs[future]
        try:
            source,result=future.result()
            records[(item['dataset'],item['path'])]=result
        except Exception as exc:
            failure={'dataset':item['dataset'],'path':item['path'],'error_type':type(exc).__name__}
            failures.append(failure)
            w.emit(status='file_failed',**failure)
        if len(records)%10==0:
            w.emit(status='batch_progress',completed_files=len(records),completed_bytes=sum(x['bytes'] for x in records.values()),failed_files=len(failures),elapsed=round(time.time()-started,2))

for group in plan['groups']:
    items=[x for x in plan['files'] if x['dataset']==group['dataset']]
    completed=[records[(x['dataset'],x['path'])] for x in items if (x['dataset'],x['path']) in records]
    directory=ROOT/items[0]['destination']
    manifest=directory/'rosetta_download_manifest.json'
    if len(completed)==len(items):
        if not manifest.exists():
            with manifest.open('x') as out:
                json.dump({'dataset':group['dataset'],'files':completed,'source_files':items,'complete_selected_files':True,'full_dataset':group['dataset']!='REBOOT26/rj45_recovery_install'},out,sort_keys=True,indent=2)
        else:
            prior=json.loads(manifest.read_text())
            prior_hashes={x['path']:x['sha256'] for x in prior['files']}
            if prior_hashes!={x['path']:x['sha256'] for x in completed}:
                raise RuntimeError('existing manifest disagrees')
        w.emit(status='dataset_complete',dataset=group['dataset'],files=len(items),bytes=sum(x['size'] for x in items))

checks=[]
for item in plan['files']:
    if (item['dataset'],item['path']) not in records:
        continue
    result=w.verify(ROOT/item['destination']/item['path'],item)
    if result['sha256']!=records[(item['dataset'],item['path'])]['sha256']:
        raise RuntimeError('independent rehash disagrees')
    checks.append({'dataset':item['dataset'],'path':item['path'],'bytes':item['size'],**result})
w.seal('independent-verification-stream.json',{'files':checks,'failures':failures,'verified_bytes':sum(x['bytes'] for x in checks)})
w.seal('result-stream.json',{'complete':not failures,'failures':failures,'files':len(checks),'bytes':sum(x['bytes'] for x in checks),'free_bytes':shutil.disk_usage(ROOT).free,'elapsed_seconds':time.time()-started})
w.emit(status='all_complete' if not failures else 'incomplete',files=len(checks),bytes=sum(x['bytes'] for x in checks),free_bytes=shutil.disk_usage(ROOT).free)
raise SystemExit(1 if failures else 0)
