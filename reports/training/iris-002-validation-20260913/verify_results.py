"""Read-only independent physical-error arithmetic on recovered Iris arrays."""
import json
import sys
from pathlib import Path

import numpy as np

job = Path(sys.argv[1])
result = json.loads((job / 'result.json').read_text())
meta = json.loads((job / 'control-first.npz.json').read_text())
arrays = {}
for arm in ('control', 'treatment'):
    with np.load(job / (arm + '-first.npz'), allow_pickle=False) as data:
        arrays[arm] = {name: data[name] for name in data.files}
    with np.load(job / (arm + '-reload.npz'), allow_pickle=False) as reload:
        assert set(reload.files) == set(arrays[arm])
        for name, value in arrays[arm].items():
            np.testing.assert_array_equal(value, reload[name])
for name in ('standard_targets', 'normalized_targets', 'noise', 'valid_mask'):
    np.testing.assert_array_equal(arrays['control'][name], arrays['treatment'][name])
assert arrays['control']['valid_mask'].all(), 'Physical arithmetic requires all slots valid'
target = arrays['control']['standard_targets'].astype(np.float64)
checked = 0
for name, recorded in result['metrics'].items():
    window, group, metric = name.split('/')
    side, kind = group.split('_')
    dims = [i for i, d in enumerate(meta['action_dimensions'])
            if d['name'].startswith(side + '_') and
            (('gripper' in d['name']) == (kind == 'gripper'))]
    assert dims
    slots = [0] if window == 'first' else list(range(50))
    truth = target[40:][:, slots][:, :, dims]
    train = target[:40][:, slots][:, :, dims]
    transform = np.abs if metric == 'mae' else np.square
    per_scene = {}
    for arm in arrays:
        prediction = arrays[arm]['standard_predictions'].astype(np.float64)[:, 40:]
        prediction = prediction[:, :, slots][:, :, :, dims]
        per_scene[arm] = transform(prediction - truth).mean(axis=(2, 3))
        np.testing.assert_allclose(per_scene[arm].mean(axis=1), recorded['by_noise'][arm],
                                   rtol=5e-6, atol=1e-7)
        checked += 4
    for method in ('mean', 'median'):
        constant = getattr(np, method)(train, axis=0)
        actual = float(transform(constant - truth).mean())
        np.testing.assert_allclose(actual, recorded['constants'][method], rtol=5e-6, atol=1e-7)
        checked += 1
    if 'leave_one_scene_out_gain' in recorded:
        gains = per_scene['control'] - per_scene['treatment']
        actual = [[float(np.mean([gains[n, s] for s in range(5) if s != leave]))
                   for n in range(4)] for leave in range(5)]
        np.testing.assert_allclose(actual, recorded['leave_one_scene_out_gain'],
                                   rtol=5e-6, atol=1e-7)
        checked += 20
assert result['m2_complete'] is False
print(json.dumps({'status': 'passed', 'physical_metric_scalar_checks': checked,
                  'both_reload_array_sets_exact': True, 'all_slots_valid': True,
                  'reported_scientific_status': result['status'], 'm2_complete': False}))
