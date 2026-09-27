"""Read source episode metadata only, using the existing remote environment."""
import importlib.util
import json
from pathlib import Path
import pyarrow.parquet as pq

ROOT=Path('/root/autodl-tmp/rosetta/datasets_external/targeted-20260927-001')
spec=importlib.util.spec_from_file_location('worker',ROOT/'worker.py')
w=importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
plan=json.loads((ROOT/'plan.json').read_text())
repo='REBOOT26/rj45_recovery_install'
items=[x for x in plan['files'] if x['dataset']==repo]
audit_metadata=[]
for item in items:
    if not item['path'].startswith('meta/'):
        continue
    local=dict(item,destination='metadata-audit/rj45')
    w.fetch(local)
    audit_metadata.append(local)
table=pq.read_table(ROOT/'metadata-audit/rj45/meta/episodes/chunk-000/file-000.parquet')
required={'episode_index','length','data/chunk_index','data/file_index'}
if not required.issubset(table.column_names):
    raise RuntimeError('episode metadata schema differs')
data_files={x['path'] for x in items if x['path'].startswith('data/')}
video_files={x['path'] for x in items if x['path'].startswith('videos/')}
selected=[]
for row in table.to_pylist():
    data=f"data/chunk-{row['data/chunk_index']:03d}/file-{row['data/file_index']:03d}.parquet"
    if data not in data_files:
        continue
    for key,value in row.items():
        if key.startswith('videos/') and key.endswith('/file_index'):
            prefix=key[:-len('/file_index')]
            chunk=row[prefix+'/chunk_index']
            video=f'{prefix}/chunk-{chunk:03d}/file-{value:03d}.mp4'
            if video not in video_files:
                raise RuntimeError('selected episode camera shard missing from plan')
    selected.append({'episode_index':row['episode_index'],'length':row['length'],'data_path':data,'dataset_from_index':row.get('dataset_from_index'),'dataset_to_index':row.get('dataset_to_index')})
if len({x['data_path'] for x in selected})!=len(data_files):
    raise RuntimeError('some selected data shards have no episode metadata')
result={'source_revision':items[0]['revision'],'source_total_episodes':table.num_rows,'selected_episodes':selected,'selected_episode_count':len(selected),'selected_frames_from_metadata':sum(x['length'] for x in selected),'all_selected_video_references_in_plan':True,'raw_data_rows_and_video_decoding_verified':False,'normalization_recomputed':False,'metadata_source_files':audit_metadata}
w.seal('rj45-subset-metadata-audit.json',result)
print(json.dumps({k:v for k,v in result.items() if k not in ('metadata_source_files','selected_episodes')},sort_keys=True),flush=True)
