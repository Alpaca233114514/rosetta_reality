import importlib.util
import json
from pathlib import Path
root=Path('/root/autodl-tmp/rosetta/datasets_external/targeted-20260927-001')
print(json.dumps({'pyarrow_available':importlib.util.find_spec('pyarrow') is not None}),flush=True)
human=Path('/root/autodl-tmp/rosetta/data/lerobot--aloha_sim_insertion_human')
print(json.dumps({'existing_human_metadata':[str(x.relative_to(human)) for x in human.glob('**/info.json')]}),flush=True)
