import hashlib
import json
from pathlib import Path
import numpy as np
import torch

root=Path('runs/hestia-q-replay-recovered-20260913-001/verified')
out=Path('runs/hestia-q-replay-analysis-20260913-001')
checked=0
for base in (640,1280):
    native={}
    with (root/f'base{base}/attention.jsonl').open() as stream:
        for line in stream:
            e=json.loads(line)
            if e['mode']=='native' and e['noise']==0 and e['row'] in (0,40):
                native[(e['row'],e['step'],e['layer'])]=e
    for (row,step,layer),e in native.items():
        with np.load(root/f'base{base}/sentinel-{row}-{step}-{layer}.npz',allow_pickle=False) as a:
            for name,dtype in zip(('q','k','v','probs','output'),a['dtypes'],strict=True):
                t=torch.from_numpy(a[name].copy()).to(getattr(torch,str(dtype).split('.')[-1])).contiguous()
                value=hashlib.sha256(str((tuple(t.shape),t.dtype)).encode()+t.view(torch.uint8).numpy().tobytes()).hexdigest()
                assert value==e[name],(base,row,step,layer,name)
                checked+=1
            t=torch.from_numpy(a['mask'].copy()).contiguous()
            value=hashlib.sha256(str((tuple(t.shape),t.dtype)).encode()+t.view(torch.uint8).numpy().tobytes()).hexdigest()
            assert value==e['mask']
            checked+=1
diagnostic=json.loads((out/'sentinel-diagnostic.json').read_text())['cpu_native_order_replay']
summary=json.loads((out/'summary.json').read_text())
ratios={}
for base in (640,1280):
    x=summary['standard'][f'{base}/standard/dev5/full/left_joint']
    ratios[str(base)]={}
    for metric in ('mae','mse'):
        n,f,k=(x[metric][m] for m in ('native','kmean_qnative','kmean'))
        ratios[str(base)][metric]={'fixed_q_gain':n-f,'natural_q_additional_gain':f-k,
            'retained_fraction_of_kmean_gain':(n-f)/(n-k),
            'removed_fraction_of_better_constant_gap':(n-f)/(n-min(x[metric+'_constants'].values()))}
report={'snapshot_tensor_hash_checks':checked,'snapshot_exact_to_cuda_event_hashes':True,
        'cpu_probability_unequal_values':sum(r['prob_unequal'] for r in diagnostic),
        'cpu_probability_values':sum(r['prob_count'] for r in diagnostic),
        'cpu_output_unequal_values':sum(r['output_unequal'] for r in diagnostic),
        'cpu_output_values':sum(r['output_count'] for r in diagnostic),
        'original_float64_one_ulp_acceptance':'failed_not_relaxed',
        'ratios':ratios,'optimizer_steps':0,'new_model_forwards':0}
with (out/'snapshot-identity-and-effect-summary.json').open('x') as stream:
    json.dump(report,stream,indent=2)
print(json.dumps(report,indent=2))
