import json
from pathlib import Path
import shutil
root=Path('/root/autodl-tmp/rosetta/datasets_external/targeted-20260927-001')
plan=json.loads((root/'plan.json').read_text())
groups=[]
for group in plan['groups']:
    files=[x for x in plan['files'] if x['dataset']==group['dataset']]
    complete=0
    total=0
    partial=0
    for item in files:
        target=root/item['destination']/item['path']
        if target.is_file():
            complete+=1
            total+=target.stat().st_size
        part=target.with_name(target.name+'.partial-20260927-001')
        if part.is_file():
            partial+=part.stat().st_size
    groups.append({'dataset':group['dataset'],'files':complete,'target_files':len(files),'bytes':total,'partial_bytes':partial})
log=root/'download-parallel8.log'
errors=[json.loads(line) for line in log.read_text().splitlines() if '"status": "file_failed"' in line]
result_path=root/'result-parallel8.json'
print(json.dumps({'groups':groups,'payload_bytes_present':sum(x['bytes']+x['partial_bytes'] for x in groups),'planned_bytes':plan['total_bytes'],'free_bytes':shutil.disk_usage(root).free,'failures':errors,'result':json.loads(result_path.read_text()) if result_path.exists() else None,'log_tail':log.read_text().splitlines()[-2:]},sort_keys=True),flush=True)
