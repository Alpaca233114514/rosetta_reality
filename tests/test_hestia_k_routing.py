"""Independent probability-factor examples and real eager interception controls."""

import pytest
import torch
import torch.nn.functional as functional

from scripts.hestia_k_routing import CONDITIONS, DirectionHeads, factor_probabilities


def probabilities():
    native = torch.zeros(1, 15, 50, 241, dtype=torch.float32)
    native[..., :64] = 0.8 / 64
    native[..., 192] = native[..., 240] = 0.1
    changed = native.clone()
    changed[:, 3:9, :, :64] = 0
    changed[:, 3:9, :, :32] = 0.4 / 32
    changed[:, 3:9, :, 192] = changed[:, 3:9, :, 240] = 0.3
    return native, changed


@pytest.mark.parametrize("mode", ["rrestore", "rfull", "rintra", "rmass"])
def test_factor_mass_and_distribution_with_known_closed_form(mode):
    native, changed = probabilities()
    out, stats = factor_probabilities(native, changed, mode, torch.bfloat16)
    if mode == "rrestore":
        assert out is native
    elif mode == "rfull":
        assert out is changed
    elif mode == "rintra":
        assert torch.allclose(out[:, 3:9, :, :32], torch.full((1, 6, 50, 32), 0.8 / 32))
        assert not out[:, 3:9, :, 32:64].any()
        assert torch.equal(out[..., 64:], native[..., 64:])
    else:
        assert torch.allclose(out[:, 3:9, :, :64], torch.full((1, 6, 50, 64), 0.4 / 64))
        assert torch.equal(out[..., 64:], changed[..., 64:])
    others = [0, 1, 2, 9, 10, 11, 12, 13, 14]
    assert torch.equal(out[:, others], native[:, others])
    assert stats["pre_mass_error"] <= 1e-12


@pytest.mark.parametrize("fault", ["other_head", "padding", "nan", "zero_image", "mass"])
def test_invalid_probability_factors_fail_closed(fault):
    native, changed = probabilities()
    if fault == "other_head":
        changed[:, 0, :, 0] += 0.01
    elif fault == "padding":
        changed[:, 3:9, :, 80] = 0.01
    elif fault == "nan":
        changed[:, 3:9, :, 0] = torch.nan
    elif fault == "zero_image":
        changed[:, 3:9, :, :64] = 0
    else:
        changed[:, 3:9, :, 240] += 0.1
    with pytest.raises(ValueError):
        factor_probabilities(native, changed, "rmass", torch.bfloat16)


def bank(mode):
    obj = DirectionHeads.__new__(DirectionHeads)
    obj.routing_mode, obj.row, obj.current_layer = mode, 0, 1
    obj.routing_calls, obj.routing_maxima, obj.attention = 0, {}, {}
    obj.calls = {(0, "1_k"): 1}
    q = torch.ones(1, 50, 15, 64, dtype=torch.bfloat16)
    native = torch.zeros(1, 241, 5, 64, dtype=torch.bfloat16)
    changed = native.clone()
    changed[:, :32, 1:3, 0] = 8
    obj.cache = {(0, "1_k"): (None, native.reshape(1, 241, 320), None)}
    mask = torch.zeros(1, 50, 241, dtype=torch.bool)
    mask[..., :64] = mask[..., 192:200] = mask[..., 240:] = True
    v = torch.arange(241).reshape(1, 241, 1, 1).expand(1, 241, 5, 64).to(torch.bfloat16)

    def eager(mask, _batch, dim, queries, keys, values):
        expanded = keys.repeat_interleave(3, dim=2).float().transpose(1, 2)
        logits = queries.float().transpose(1, 2) @ expanded.transpose(2, 3)
        logits *= dim**-0.5
        logits = torch.where(mask[:, None], logits, torch.finfo(torch.float32).min)
        p = functional.softmax(logits, dim=-1).to(values.dtype)
        return p @ values.repeat_interleave(3, dim=2).permute(0, 2, 1, 3)

    obj.original_eager = eager
    return obj, (mask, 1, 64, q, changed, v), native


@pytest.mark.parametrize("mode", ["rrestore", "rfull", "rintra", "rmass"])
def test_interceptor_uses_original_kernel_and_restores_global_softmax(mode):
    obj, args, native = bank(mode)
    original = functional.softmax
    reference = obj.original_eager(*args[:4], native if mode == "rrestore" else args[4], args[5])
    actual = obj.eager(*args)
    assert functional.softmax is original
    if mode in {"rrestore", "rfull"}:
        assert torch.equal(actual, reference)
    assert obj.routing_calls == 1 and set(obj.attention) == {(0, 1)}
    obj.calls[0, "1_k"] = 2
    obj.eager(*args)
    assert obj.routing_calls == 2 and len(obj.attention) == 1


def test_second_kernel_failure_restores_softmax():
    obj, args, _ = bank("rintra")
    original, real = functional.softmax, obj.original_eager
    calls = []

    def fail(*values):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("synthetic second kernel failure")
        return real(*values)

    obj.original_eager = fail
    with pytest.raises(RuntimeError, match="second kernel"):
        obj.eager(*args)
    assert functional.softmax is original


def test_transfer_accepts_only_registered_routing_metadata():
    from scripts.verify_hestia_k_routing_results import validate_manifest

    item = {"bytes": 1, "sha256": "a" * 64}
    files = {k: item for k in ("registration.json", "permit.json", "worker-exited.json")}
    files.update({c + "/routing.json": item for c in CONDITIONS})
    assert validate_manifest({"files": files, "total_bytes": len(files)}) == files
    for name in ("base640_kh12/routing.json", "base640_runknown/result.json", "../secret"):
        with pytest.raises(ValueError):
            validate_manifest({"files": {**files, name: item}, "total_bytes": len(files) + 1})


def test_hybrid_artifact_marks_overridden_probabilities(tmp_path, monkeypatch):
    import json

    import numpy as np

    from scripts.hestia_k_factor import KeyFactor

    obj = DirectionHeads.__new__(DirectionHeads)
    obj.routing_mode, obj.routing_calls, obj.routing_maxima = "rintra", 1800, {}
    obj.condition, obj.selected, obj.selected_heads = "base640_rintra", (1,), (1, 2)
    obj.attention = {
        (row, layer): (np.zeros((15, 64)), np.zeros((15, 8)))
        for row in range(45)
        for layer in range(1, 16, 2)
    }
    obj.transformed = {(row, "1_k"): torch.zeros(1, 241, 320) for row in range(45)}
    monkeypatch.setattr(KeyFactor, "finish", lambda *_: {"parent_checked": True})
    assert obj.finish(tmp_path) == {"parent_checked": True}
    record = json.loads((tmp_path / "attention.json").read_text())
    assert record["softmax_return_overridden"] is True
    assert record["routing_mode"] == "rintra"
    assert record["probabilities"].startswith("actual returned")
