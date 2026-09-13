"""Read-only follow-up of the failed float64 sentinel approximation."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts.hestia_q_replay import attention_parts
from scripts.verify_hestia_q_replay_analysis import verify

root=Path('runs/hestia-q-replay-recovered-20260913-001/verified')
out=Path('runs/hestia-q-replay-analysis-20260913-001')
analysis=json.loads((out/'result.json').read_text())
action=verify(root,analysis)
action['sentinel_verification']='failed_original_float64_one_ulp_check'
with (out/'action-and-identity-verification.json').open('x') as stream:
    json.dump(action,stream,indent=2)
print(json.dumps(action),flush=True)
records=[]
for path in sorted(root.glob('base*/sentinel-*.npz')):
    with np.load(path,allow_pickle=False) as a:
        tensors={n:torch.from_numpy(a[n].copy()) for n in ('q','k','v','probs','output','mask')}
        for n,dtype in zip(('q','k','v','probs','output'),a['dtypes'],strict=True):
            tensors[n]=tensors[n].to(getattr(torch,str(dtype).split('.')[-1]))
        q,k,v,mask=(tensors[n] for n in ('q','k','v','mask'))
        core=SimpleNamespace(num_attention_heads=q.shape[2],num_key_value_heads=k.shape[2])
        with torch.autocast('cpu',dtype=torch.bfloat16):
            result,p=attention_parts(core,mask,1,q.shape[-1],q,k,v)
        error=np.abs(p.float().numpy()-a['probs'])
        oe=np.abs(result.float().numpy()-a['output'])
        records.append({'file':path.relative_to(root).as_posix(),'dtypes':a['dtypes'].tolist(),
            'prob_max_abs':float(error.max()),'prob_unequal':int(np.count_nonzero(error)),
            'output_max_abs':float(oe.max()),'output_unequal':int(np.count_nonzero(oe)),
            'prob_count':error.size,'output_count':oe.size})
report={'cpu_native_order_replay':records,'new_model_forwards':0,'optimizer_steps':0,
        'original_acceptance_not_changed':True}
with (out/'sentinel-diagnostic.json').open('x') as stream:
    json.dump(report,stream,indent=2)
print(json.dumps({'sentinels':len(records),'prob_exact_files':sum(r['prob_unequal']==0 for r in records),
    'maximum_probability_error':max(r['prob_max_abs'] for r in records),
    'maximum_output_error':max(r['output_max_abs'] for r in records)}),flush=True)
print(json.dumps(records[:3]),flush=True)
