"""Remote-only, capacity-bounded dataset staging. No model or training imports."""
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import time
import traceback
import urllib.parse
import urllib.request

ROOT = Path('/root/autodl-tmp/rosetta/datasets_external/targeted-20260927-001')
RESERVE = 10_000_000_000
BLOCK = 32 * 1024 * 1024
spec = importlib.util.spec_from_file_location('download_core', ROOT / 'download_core.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

def emit(**event):
    print(json.dumps(event, sort_keys=True), flush=True)

def seal(name, obj):
    with (ROOT / name).open('x', encoding='utf-8') as out:
        json.dump(obj, out, indent=2, sort_keys=True)
        out.write('\n')

def digests(path):
    size = path.stat().st_size
    sha = hashlib.sha256()
    md5 = hashlib.md5()
    git = hashlib.sha1(f'blob {size}\0'.encode())
    with path.open('rb') as inp:
        while data := inp.read(8 * 1024 * 1024):
            sha.update(data)
            md5.update(data)
            git.update(data)
    return {'sha256':sha.hexdigest(), 'md5_base64':base64.b64encode(md5.digest()).decode(), 'git_blob_sha1':git.hexdigest()}

def verify(path, item):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != item['size']:
        raise RuntimeError('file type/size mismatch: ' + item['path'])
    result = digests(path)
    for key in ('sha256', 'md5_base64', 'git_blob_sha1'):
        if item.get(key) and item[key] != result[key]:
            raise RuntimeError('source checksum mismatch: ' + item['path'])
    return result

def fetch(item):
    target = ROOT / item['destination'] / item['path']
    if target.is_symlink():
        raise RuntimeError('target symlink')
    if target.exists():
        return {'path':item['path'], 'bytes':item['size'], **verify(target,item), 'status':'verified_existing'}
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.partial-20260927-001')
    if partial.is_symlink():
        raise RuntimeError('partial symlink')
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > item['size']:
        raise RuntimeError('oversized partial')
    if shutil.disk_usage(ROOT).free < item['size'] - offset + RESERVE:
        raise RuntimeError('capacity reserve would be violated')
    started = time.monotonic()
    while offset < item['size']:
        end = min(offset + BLOCK, item['size']) - 1
        for attempt in range(8):
            try:
                req = urllib.request.Request(item['url'], headers={'User-Agent':'rosetta-direct-staging/2', 'Range':f'bytes={offset}-{end}'})
                with urllib.request.urlopen(req, timeout=60) as response:
                    if response.status == 206:
                        cr = response.headers.get('Content-Range','')
                        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',cr)
                        if not match or tuple(map(int,match.groups())) != (offset,end,item['size']):
                            raise RuntimeError('range mismatch')
                    elif response.status != 200 or offset != 0:
                        raise RuntimeError('range refused; partial preserved')
                    data = response.read(end-offset+1)
                if len(data) != end-offset+1:
                    raise RuntimeError('short range')
                if shutil.disk_usage(ROOT).free < len(data) + RESERVE:
                    raise RuntimeError('capacity reserve reached')
                with partial.open('ab' if offset else 'xb') as out:
                    out.write(data)
                offset += len(data)
                if offset % (256*1024*1024) == 0 or offset == item['size']:
                    emit(status='progress', dataset=item['dataset'], path=item['path'], bytes=offset, size=item['size'], elapsed=round(time.monotonic()-started,2))
                break
            except Exception as exc:
                emit(status='retry',dataset=item['dataset'],path=item['path'],offset=offset,attempt=attempt+1,error_type=type(exc).__name__)
                if attempt == 7:
                    raise
                time.sleep(min(30,3*(attempt+1)))
    result = verify(partial,item)
    os.link(partial,target)
    partial.unlink()
    return {'path':item['path'], 'bytes':item['size'], **result, 'status':'downloaded'}

def hf_items(repo, revision, chosen):
    entries = core.tree_pages(repo,revision)
    if chosen is not None:
        entries = [x for x in entries if chosen(x['path'])]
    items=[]
    for x in entries:
        path=x['path']
        if path.startswith('/') or '..' in Path(path).parts:
            raise RuntimeError('unsafe source path')
        lfs=x.get('lfs') or {}
        items.append({'dataset':repo,'revision':revision,'destination':repo+'/'+revision,'path':path,'size':int(x['size']),
                      'sha256':lfs.get('oid'), 'git_blob_sha1':None if lfs else x.get('oid'),
                      'url':f'{core.MIRROR}/datasets/{repo}/resolve/{revision}/'+urllib.parse.quote(path,safe='/')})
    return items

def main():
    started=time.time()
    # Complete USB-A; RJ45 data shards sampled across the full ordered corpus.
    selected_ids=sorted({int(round(i*59/23)) for i in range(24)})
    usb=hf_items('REBOOT26/USB-A_recovery_install','cfa3498a3982eb24554e88e77140247871eee3eb',None)
    def rj_selection(path):
        if not path.startswith('data/'):
            return True  # retain every camera video and original metadata
        match=re.fullmatch(r'data/chunk-000/file-(\d+)\.parquet',path)
        return bool(match and int(match.group(1)) in selected_ids)
    rj=hf_items('REBOOT26/rj45_recovery_install','f6acd5b69394ffd4d2d8bd30079306c9e2061bbe',rj_selection)
    human=hf_items('lerobot/aloha_sim_insertion_human','cc571a3c661df81b566dbfde3d5c1e85fcdf7884',None)
    mimic=hf_items('amandlek/mimicgen_datasets','33016f8a62c02334f929f2913af8fdd2a8a129e1',lambda p:p in {'core/square_d0.hdf5','core/square_d1.hdf5','core/square_d2.hdf5','README.md','.gitattributes'})
    objects=list(core.gcs_objects('austin_sirius_dataset_converted_externally_to_rlds/'))
    if len(objects)!=67 or sum(int(x['size']) for x in objects)!=7031694115:
        raise RuntimeError('Sirius inventory drift')
    sirius=[]
    for x in objects:
        path=x['name'].split('/',1)[1]
        if path.startswith('/') or '..' in Path(path).parts:
            raise RuntimeError('unsafe GCS path')
        sirius.append({'dataset':'sirius','destination':'open_x_embodiment/austin_sirius_dataset_converted_externally_to_rlds','path':path,'size':int(x['size']), 'generation':x['generation'],'md5_base64':x.get('md5Hash'), 'url':'https://storage.googleapis.com/'+core.GCS_BUCKET+'/'+urllib.parse.quote(x['name'],safe='/')+'?generation='+x['generation']})
    groups=[human,mimic,sirius,usb,rj]
    total=sum(x['size'] for group in groups for x in group)
    available=shutil.disk_usage(ROOT).free
    if total+RESERVE>available:
        raise RuntimeError('sealed download plan does not fit')
    plan={'run':'targeted-20260927-001','started_unix':started,'remote_only':True,'no_optimizer':True,'reserve_bytes':RESERVE,'initial_free_bytes':available,'total_bytes':total,'groups':[{'dataset':g[0]['dataset'],'files':len(g),'bytes':sum(x['size'] for x in g)} for g in groups], 'rj45_selection':{'data_file_indices':selected_ids,'all_videos':True,'original_metadata_unchanged':True,'full_dataset':False,'episode_alignment':'must inspect episode metadata before use','normalization':'full-corpus stats retained as provenance only; recompute from training split before training'},'files':[x for g in groups for x in g], 'reuse':['targeted-20260925-001/lerobot/aloha_sim_insertion_scripted','targeted-20260925-001/robomimic/robomimic_datasets'], 'deferred':['remaining RJ45 data shards','RACER','AgiBot'], 'contract_warning':'Separate robot contracts; no direct ALOHA training compatibility established.'}
    seal('plan.json',plan)
    emit(status='plan_sealed',bytes=total,free_bytes=available,groups=plan['groups'],rj45_data_file_indices=selected_ids)
    failures=[]
    for group in groups:
        records=[]
        try:
            for item in group:
                record=fetch(item)
                records.append(record)
                emit(status='file_complete',dataset=item['dataset'],path=item['path'],bytes=item['size'],sha256=record['sha256'])
            manifest={'dataset':group[0]['dataset'],'files':records,'source_files':group,'complete_selected_files':True,'full_dataset':group[0]['dataset']!='REBOOT26/rj45_recovery_install'}
            directory=ROOT/group[0]['destination']
            with (directory/'rosetta_download_manifest.json').open('x') as out:
                json.dump(manifest,out,indent=2,sort_keys=True)
            emit(status='dataset_complete',dataset=group[0]['dataset'],files=len(records),bytes=sum(r['bytes'] for r in records))
        except Exception as exc:
            failures.append({'dataset':group[0]['dataset'],'error_type':type(exc).__name__,'completed_files':len(records)})
            emit(status='dataset_failed',**failures[-1])
    # Independently reread every selected payload after download.
    checks=[]
    for group in groups:
        if any(f['dataset']==group[0]['dataset'] for f in failures):
            continue
        for item in group:
            result=verify(ROOT/item['destination']/item['path'],item)
            checks.append({'dataset':item['dataset'],'path':item['path'],'bytes':item['size'],**result})
    seal('independent_verification.json',{'files':checks,'failures':failures,'verified_bytes':sum(x['bytes'] for x in checks)})
    seal('result.json',{'complete':not failures,'failures':failures,'files':len(checks),'bytes':sum(x['bytes'] for x in checks),'free_bytes':shutil.disk_usage(ROOT).free,'elapsed_seconds':time.time()-started})
    emit(status='all_complete' if not failures else 'incomplete',files=len(checks),bytes=sum(x['bytes'] for x in checks),free_bytes=shutil.disk_usage(ROOT).free)
    if failures:
        raise SystemExit(1)

if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        emit(status='fatal',error_type=type(exc).__name__)
        raise SystemExit(1)
