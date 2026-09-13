import pytest
import torch

from scripts.hestia_k_head_interaction import CONDITIONS, LAYER, DirectionHeads, condition_spec


def test_single_layer_and_five_heads_are_only_registered_axes():
    assert LAYER in range(1, 16, 2)
    assert len(CONDITIONS) == 7
    assert all(condition_spec(c) == (640, None) for c in CONDITIONS)
    with pytest.raises(ValueError):
        condition_spec("base640_kh5")


def test_head_result_transfer_whitelist_preserves_new_metadata():
    from scripts.verify_hestia_k_head_interaction_results import validate_manifest

    item = {"bytes": 1, "sha256": "a" * 64}
    files = {name: item for name in ("registration.json", "permit.json", "worker-exited.json")}
    files.update({c + "/head-selection.json": item for c in CONDITIONS})
    assert validate_manifest({"files": files, "total_bytes": len(files)}) == files
    for name in ("base640_kd01/result.json", "base640_kh5/result.json", "../secret"):
        with pytest.raises(ValueError):
            validate_manifest({"files": {**files, name: item}, "total_bytes": len(files) + 1})


@pytest.mark.parametrize("heads", [(1,), (2,), (1, 2), (0, 3, 4), tuple(range(5))])
def test_only_selected_head_directions_change(monkeypatch, heads):
    obj = DirectionHeads.__new__(DirectionHeads)
    obj.row, obj.mode, obj.factor_mode = 0, "native", "kdirection"
    obj.selected, obj.selected_heads, obj.head_dim = (LAYER,), heads, 64
    obj.calls, obj.cache, obj.transformed = {}, {}, {}
    obj.reference, obj.layout = None, {"shape": [1, 241, 320]}
    k = torch.zeros(1, 241, 320, dtype=torch.bfloat16)
    k.reshape(1, 241, 5, 64)[..., 0] = 3
    k.reshape(1, 241, 5, 64)[..., 1] = 4
    m = torch.zeros(1, 64, 320, dtype=torch.bfloat16)
    m.reshape(1, 64, 5, 64)[..., 1] = 2
    key = f"{LAYER}_k"
    obj.means = {key: m}
    obj.norms = {key: torch.full((1, 64, 5, 1), 5.0, dtype=torch.float64)}
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda *_: True)
    out = obj.hook(key)(None, (k,), k)
    a = out[:, :64].reshape(1, 64, 5, 64)
    n = k[:, :64].reshape_as(a)
    for h in range(5):
        if h in heads:
            assert (a[:, :, h, 0] == 0).all() and (a[:, :, h, 1] == 5).all()
        else:
            assert torch.equal(a[:, :, h], n[:, :, h])
    assert torch.equal(out[:, 64:], k[:, 64:])
    assert torch.equal(obj.hook(f"{LAYER}_v")(None, (k,), k), k)
    other = 1 if LAYER != 1 else 3
    assert torch.equal(obj.hook(f"{other}_k")(None, (k,), k), k)


@pytest.mark.parametrize("name,heads", [("kh12", (1, 2)), ("kh034", (0, 3, 4))])
def test_real_constructor_selects_pair_and_complement(monkeypatch, name, heads):
    from scripts.hestia_k_head_interaction import SingleHeads

    def parent(self, policy, checkpoint, condition, calibration, episodes, train):
        assert condition == "base640_kh1"
        self.selected_heads = (1,)

    monkeypatch.setattr(SingleHeads, "__init__", parent)
    obj = DirectionHeads(None, "sha", f"base640_{name}", None, [], [])
    assert obj.selected_heads == heads
    assert obj.condition == f"base640_{name}"
