"""Analytic component and live-hook checks on tiny synthetic CPU tensors."""

import pytest
import torch

from scripts.hestia_value_component import (
    CONDITIONS,
    ValueComponents,
    component_mode,
    condition_spec,
    digest,
    intervention_output,
    split_update,
)


def test_registered_axis_and_invalid_condition():
    assert len(CONDITIONS) == 10 and len(set(CONDITIONS)) == 10
    for name in CONDITIONS:
        base, donor = condition_spec(name)
        assert base + donor == 1920
        assert component_mode(name) in ("native", "full", "global", "token", "scene")
    with pytest.raises(ValueError):
        condition_spec("base1280_vfake640")


def test_exact_analytic_components_and_reciprocal_sign():
    global_mean = torch.tensor([[[8.0, 16.0]]])
    token = torch.tensor([[[-2.0, -4.0], [2.0, 4.0]]])
    scene = torch.arange(-19.5, 20.5).reshape(40, 1, 1, 1)
    delta = global_mean + token + scene
    g, t = split_update(delta)
    assert torch.equal(g, global_mean) and torch.equal(t, token)
    native = torch.ones((1, 2, 2))
    donor = native + delta[0]
    partials = [
        intervention_output(native, donor, g, t, mode, 1) - native
        for mode in ("global", "token", "scene")
    ]
    assert torch.equal(sum(partials), donor - native)
    reciprocal = [
        intervention_output(donor, native, g, t, mode, -1) - donor
        for mode in ("global", "token", "scene")
    ]
    assert torch.equal(sum(reciprocal), native - donor)
    with pytest.raises(ValueError, match="forty"):
        split_update(torch.cat((delta, delta[:5])))


def test_bf16_full_returns_donor_without_roundtrip():
    base = torch.tensor([1000.0], dtype=torch.bfloat16)
    donor = torch.tensor([0.003], dtype=torch.bfloat16)
    assert not torch.equal((base.float() + donor.float() - base.float()).bfloat16(), donor)
    assert intervention_output(base, donor, None, None, "full", 1) is donor
    assert intervention_output(base, donor, None, None, "native", 1) is base
    assert digest(torch.tensor([0.0])) == digest(torch.tensor([-0.0]))


def test_hook_exact_donor_and_repeated_prefix_guard(monkeypatch):
    # Exercise real hook logic and linear operations without loading a model or GPU.
    bank = ValueComponents.__new__(ValueComponents)
    bank.row, bank.sign, bank.mode = 0, 1, "full"
    bank.cache, bank.calls, bank.reference = {}, {}, None
    bank.donors = {1: 2 * torch.eye(320)}
    x = torch.ones((1, 64, 320))
    native = x.bfloat16()
    hook = bank.hook(1)
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda *_: True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        first = hook(None, (x,), native)
        assert torch.equal(first, 2 * native)
        assert hook(None, (x,), native) is first
        with pytest.raises(ValueError, match="Prefix depends"):
            hook(None, (x + 1,), native)
    assert len(bank.cache) == 1


def test_transfer_accepts_calibration_but_rejects_unregistered_paths():
    from scripts.verify_hestia_value_component_results import validate_manifest

    item = {"bytes": 1, "sha256": "a" * 64}
    files = {
        n: item
        for n in (
            "registration.json",
            "permit.json",
            "worker-exited.json",
            "base1280/calibration.npz",
            "base1280/calibration.json",
            "base1280_vscene640/intervention.json",
        )
    }
    assert validate_manifest({"files": files, "total_bytes": len(files)}) == files
    files["base1280/../../unrelated"] = item
    with pytest.raises(ValueError, match="Unexpected result path"):
        validate_manifest({"files": files, "total_bytes": len(files)})
