"""Synthetic scalar arithmetic only, not a SmolVLA training or remediation run."""
import hashlib
import json
from pathlib import Path

import torch

records = []
for dtype in (torch.bfloat16, torch.float32):
    parameter = torch.nn.Parameter(torch.ones(1, dtype=dtype))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4, betas=(0.9, 0.95),
                                 eps=1e-8, weight_decay=1e-10)
    parameter.grad = torch.ones_like(parameter)
    before = parameter.detach().float().item()
    optimizer.step()
    after = parameter.detach().float().item()
    state = optimizer.state[parameter]
    records.append({'dtype': str(dtype), 'before': before, 'gradient': 1.0, 'after': after,
                    'actual_delta': after - before, 'exp_avg': state['exp_avg'].float().item(),
                    'exp_avg_sq': state['exp_avg_sq'].float().item(),
                    'moment_dtype': str(state['exp_avg'].dtype)})
assert records[0]['actual_delta'] == 0.0
assert records[0]['exp_avg'] > 0
assert records[1]['actual_delta'] < 0.0
one = torch.tensor(1., dtype=torch.bfloat16)
result = {'scope': 'synthetic_scalar_cpu_only', 'torch': torch.__version__, 'records': records,
          'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'bf16_next_below_one': torch.nextafter(one, torch.tensor(0., dtype=one.dtype)).item(),
          'bf16_next_above_one': torch.nextafter(one, torch.tensor(2., dtype=one.dtype)).item(),
          'real_policy_updates': 0, 'actual_historical_norm_gradient_measured': False,
          'unique_root_cause_established': False}
with Path('/output/result.json').open('x') as f:
    json.dump(result, f, indent=2)
    f.write('\n')
print(json.dumps(result))
