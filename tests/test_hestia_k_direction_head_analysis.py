import copy

import numpy as np
import pytest

from scripts.analyze_hestia_head_attention import head_effects, unchanged_query_heads


def test_gqa_isolation_rejects_changed_neighbor_but_allows_later_propagation():
    native = (np.zeros((2, 8, 15, 64)), np.zeros((2, 8, 15, 8)))
    changed = tuple(a.copy() for a in native)
    for array in changed:
        array[:, 0, 6:9] = 1
        array[:, 1:] = 2
    assert len(unchanged_query_heads(native, changed, 2)) == 12
    changed[0][:, 0, 9] = 1
    with pytest.raises(ValueError, match="Unselected"):
        unchanged_query_heads(native, changed, 2)


def test_ratio_of_means_threshold_also_requires_all_four_noise_signs():
    def entry(values):
        error = {k: dict(mean=float(np.mean(values)), by_noise=values) for k in ("mae", "mse")}
        return dict(standard=dict(dev5=dict(full=dict(left_joint=dict(errors=dict(model=error))))))

    metrics = {"base640_native": entry([10]*4), "base640_kdirection": entry([5]*4)}
    metrics.update({f"base640_kh{h}": entry([7]*4) for h in range(5)})
    assert all(r["qualifies_for_routing_isolation"] for r in head_effects(metrics))
    other = copy.deepcopy(metrics)
    other["base640_kh2"] = entry([11, 5, 5, 5])
    row = head_effects(other)[2]
    assert row["fraction_of_layer1_mse_benefit"] > 0.6
    assert not row["qualifies_for_routing_isolation"]
