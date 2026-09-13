"""Counterexamples for layer selection and observational attention capture."""

import numpy as np
import pytest
import torch
import torch.nn.functional as functional

from scripts.hestia_k_direction_layer import CONDITIONS, DirectionLayers, selected_layers
from scripts.hestia_scene_kv_mean import LAYERS


def test_layer_design_has_full_controls_and_no_unregistered_subset():
    assert len(CONDITIONS) == len(set(CONDITIONS)) == 11
    assert selected_layers(CONDITIONS[0]) == ()
    assert selected_layers(CONDITIONS[1]) == selected_layers(CONDITIONS[2]) == LAYERS
    assert tuple(selected_layers(c)[0] for c in CONDITIONS[3:]) == LAYERS
    with pytest.raises(ValueError):
        selected_layers("base640_kd02")


def bank():
    obj = DirectionLayers.__new__(DirectionLayers)
    obj.row, obj.current_layer = 0, 1
    obj.calls, obj.attention = {(0, "1_k"): 1}, {}
    mask = torch.zeros(1, 50, 241, dtype=torch.bool)
    mask[:, :, :64] = True
    mask[:, :, 192:200] = True
    mask[:, :, 240] = True
    q = torch.ones(1, 50, 15, 64, dtype=torch.bfloat16)
    k = torch.ones(1, 241, 5, 64, dtype=torch.bfloat16)
    v = torch.arange(241).view(1, 241, 1, 1).expand(1, 241, 5, 64).to(torch.bfloat16)

    def eager(mask, _batch, _dim, _q, _k, v):
        logits = (
            torch.arange(241, dtype=torch.float32).view(1, 1, 1, 241).expand(1, 15, 50, 241) / 100
        )
        logits = torch.where(mask[:, None], logits, torch.finfo(torch.float32).min)
        p = functional.softmax(logits, dim=-1).to(v.dtype)
        expanded = v.repeat_interleave(3, dim=2).permute(0, 2, 1, 3)
        return p @ expanded

    obj.original_eager = eager
    return obj, (mask, 1, 64, q, k, v)


def test_observer_returns_exact_native_output_and_restores_softmax():
    obj, args = bank()
    original = functional.softmax
    reference = obj.original_eager(*args)
    output = obj.eager(*args)
    assert functional.softmax is original
    assert torch.equal(output, reference)
    image, stats = obj.attention[0, 1]
    assert image.shape == (15, 64) and stats.shape == (15, 8)
    assert np.allclose(image.sum(-1), stats[:, 0], atol=1e-6)
    assert np.all(stats[:, 3] == 0)
    assert np.allclose(stats[:, :4].sum(-1), 1, atol=0.004)
    with pytest.raises(ValueError, match="Duplicate"):
        obj.eager(*args)
    assert functional.softmax is original


def test_later_denoise_calls_do_not_repeat_capture():
    obj, args = bank()
    obj.calls[0, "1_k"] = 2
    assert torch.equal(obj.eager(*args), obj.original_eager(*args))
    assert not obj.attention


def test_exception_cannot_leave_global_softmax_replaced():
    obj, args = bank()
    original = functional.softmax

    def fail(*_args):
        raise RuntimeError("synthetic native failure")

    obj.original_eager = fail
    with pytest.raises(RuntimeError):
        obj.eager(*args)
    assert functional.softmax is original


def test_cross_layer_context_always_restored_after_failure():
    obj, _ = bank()
    obj.current_layer = None

    def fail(*_args):
        assert obj.current_layer == 3
        raise RuntimeError("synthetic cross failure")

    obj.original_cross = fail
    with pytest.raises(RuntimeError):
        obj.cross(None, None, 3)
    assert obj.current_layer is None


def test_only_registered_layer_k_changes_other_routes_stay_exact(monkeypatch):
    obj = DirectionLayers.__new__(DirectionLayers)
    obj.row, obj.mode, obj.factor_mode = 0, "native", "kdirection"
    obj.selected, obj.head_dim = (3,), 64
    obj.calls, obj.cache, obj.transformed = {}, {}, {}
    obj.reference, obj.layout = None, {"shape": [1, 241, 320]}
    k = torch.zeros(1, 241, 320, dtype=torch.bfloat16)
    k.reshape(1, 241, 5, 64)[..., 0] = 3
    k.reshape(1, 241, 5, 64)[..., 1] = 4
    m = torch.zeros(1, 64, 320, dtype=torch.bfloat16)
    m.reshape(1, 64, 5, 64)[..., 1] = 2
    obj.means = {"3_k": m}
    obj.norms = {"3_k": torch.full((1, 64, 5, 1), 5.0, dtype=torch.float64)}
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda *_: True)
    for key in ("1_k", "3_v", "5_k"):
        assert torch.equal(obj.hook(key)(None, (k,), k), k)
    changed = obj.hook("3_k")(None, (k,), k)
    assert torch.equal(changed[:, 64:], k[:, 64:])
    assert torch.all(changed[:, :64].reshape(1, 64, 5, 64)[..., 1] == 5)
    assert set(obj.transformed) == {(0, "3_k")}
    with pytest.raises(ValueError, match="varies"):
        obj.hook("3_k")(None, (k + 1,), k)


def test_transfer_allowlist_allows_attention_but_not_other_layers_or_files():
    from scripts.verify_hestia_k_direction_layer_results import validate_manifest

    item = {"bytes": 1, "sha256": "a" * 64}
    files = {key: item for key in ("registration.json", "permit.json", "worker-exited.json")}
    files.update({c + "/attention.npz": item for c in CONDITIONS})
    assert validate_manifest({"files": files, "total_bytes": len(files)}) == files
    for name in ("base640_kd02/attention.npz", "base1280_native/result.json", "../outside"):
        with pytest.raises(ValueError):
            validate_manifest({"files": {**files, name: item}, "total_bytes": len(files) + 1})
