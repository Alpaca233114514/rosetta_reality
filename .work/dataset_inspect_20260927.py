import json
import urllib.request
import importlib.util
from pathlib import Path

root = Path('/root/autodl-tmp/rosetta/datasets_external/targeted-20260927-001')
spec = importlib.util.spec_from_file_location('download_core', root / 'download_core.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
for repo, revision, selection in core.HF_SOURCES:
    entries = core.tree_pages(repo, revision)
    print(json.dumps({'repo':repo, 'revision':revision, 'files':[{'path':x['path'],'size':x['size']} for x in entries]}, sort_keys=True), flush=True)
    if repo.startswith('REBOOT'):
        info = core.get_json(f'{core.MIRROR}/datasets/{repo}/resolve/{revision}/meta/info.json')
        print(json.dumps({'repo':repo, 'info':info}), flush=True)
repo='lerobot/aloha_sim_insertion_human'
info=core.get_json(f'{core.MIRROR}/api/datasets/{repo}')
entries=core.tree_pages(repo,info['sha'])
print(json.dumps({'repo':repo,'revision':info['sha'],'count':len(entries),'bytes':sum(x['size'] for x in entries),'files':[{'path':x['path'],'size':x['size']} for x in entries]}),flush=True)
