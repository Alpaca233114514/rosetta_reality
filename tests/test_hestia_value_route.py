"""Synthetic full-prefix routing checks, including state and masked cameras."""

import pytest
import torch

from scripts.hestia_value_route import CONDITIONS, ROUTES, ValueRoutes, route_output


def test_routes_reconstruct_full_output_except_masked_cameras():
    native = torch.zeros((1, 241, 320), dtype=torch.bfloat16)
    donor = torch.ones_like(native)
    combined = native.clone()
    for mode, (start, stop) in ROUTES.items():
        actual = route_output(native, donor, mode)
        assert torch.equal(actual[:, start:stop], donor[:, start:stop])
        assert not actual[:, :start].any() and not actual[:, stop:].any()
        combined += actual
    assert combined[:, :64].all() and combined[:, 192:].all()
    assert not combined[:, 64:192].any()
    assert route_output(native, donor, "native") is native
    assert route_output(native, donor, "full") is donor
    with pytest.raises(ValueError, match="shape"):
        route_output(native[:, :64], donor[:, :64], "image")
    assert len(CONDITIONS) == len(set(CONDITIONS)) == 10


def test_layout_rejects_old_image_slice_and_visible_empty_camera():
    bank = ValueRoutes.__new__(ValueRoutes)
    bank.layout = bank.reference = None
    mask = torch.ones((1, 241), dtype=torch.bool)
    with pytest.raises(ValueError, match="mask differs"):
        bank.set_layout(mask)
    mask[:, 64:192] = False
    bank.set_layout(mask)
    with pytest.raises(ValueError, match="mask differs"):
        bank.set_layout(mask[:, :64])
    changed = mask.clone()
    changed[:, 200] = False
    with pytest.raises(ValueError, match="varies"):
        bank.set_layout(changed)


def test_actual_hook_caches_full_prefix_and_rejects_state_drift(monkeypatch):
    bank = ValueRoutes.__new__(ValueRoutes)
    bank.row, bank.mode = 0, "state"
    bank.cache, bank.calls, bank.reference, bank.layout = {}, {}, None, None
    bank.donors = {1: 2 * torch.eye(320)}
    mask = torch.ones((1, 241), dtype=torch.bool)
    mask[:, 64:192] = False
    bank.set_layout(mask)
    x = torch.ones((1, 241, 320))
    y = x.bfloat16()
    hook = bank.hook(1)
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda *_: True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = hook(None, (x,), y)
        assert torch.equal(actual[:, :240], y[:, :240])
        assert torch.equal(actual[:, 240:], 2 * y[:, 240:])
        assert torch.equal(hook(None, (x,), y), actual)
        changed = x.clone()
        changed[:, 240:] += 1
        with pytest.raises(ValueError, match="Prefix depends"):
            hook(None, (changed,), y)
    assert len(bank.cache) == 1


def test_transfer_manifest_remains_narrow():
    from scripts.verify_hestia_value_route_results import validate_manifest

    item = {"bytes": 1, "sha256": "a" * 64}
    files = {
        n: item
        for n in (
            "registration.json",
            "permit.json",
            "worker-exited.json",
            "base640_vstate1280/intervention.json",
        )
    }
    assert validate_manifest({"files": files, "total_bytes": len(files)}) == files
    files["base1280/../../unrelated"] = item
    with pytest.raises(ValueError, match="Unexpected"):
        validate_manifest({"files": files, "total_bytes": len(files)})
