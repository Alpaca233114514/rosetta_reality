import json
from pathlib import Path
import numpy as np

root=Path('runs/hestia-q-replay-analysis-20260913-001')
r=json.loads((root/'result.json').read_text())
summary={'primary':r['primary'],'standard':{},'left_terms':{},'controls':{}}
for base in (640,1280):
    for split in ('train40','dev5'):
        for window in ('full','first'):
            for group in r['groups']:
                key=f'{base}/standard/{split}/{window}/{group}'
                record={}
                for metric in ('mae','mse'):
                    x=r['metrics'][key+'/'+metric]
                    means={m:float(np.mean(x['by_noise'][m])) for m in ('native','kmean_qnative','kmean')}
                    record[metric]=means
                    record[metric+'_constants']=x['constants']
                summary['standard'][key]=record
    entry=r['metrics'][f'{base}/standard/dev5/full/left_joint/mse']
    summary['left_terms'][str(base)]={m:{k:float(np.mean(v)) for k,v in terms.items()}
        for m,terms in entry['decomposition'].items() if m in ('native','kmean_qnative','kmean')}
    report=json.loads((Path('runs/hestia-q-replay-recovered-20260913-001/verified')/f'base{base}/result.json').read_text())
    summary['controls'][str(base)]={k:report[k] for k in ('model_forwards','exact_control_forwards',
        'all_parameters_unchanged','parameter_tensors','seconds','peak_cuda_allocated','peak_rss_bytes')}
with (root/'summary.json').open('x') as f:
    json.dump(summary,f,indent=2)
print(json.dumps({'primary':summary['primary'],'controls':summary['controls'],'left_terms':summary['left_terms']},indent=2))
for key,v in summary['standard'].items():
    if '/dev5/' in key or '/full/left_joint' in key:
        print(key,json.dumps(v))
