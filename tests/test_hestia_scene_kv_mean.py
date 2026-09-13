"""Synthetic counterexamples for train-only calibration and reversible routing."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.hestia_scene_kv_mean import (
    CONDITIONS,
    LAYERS,
    SceneKVMean,
    component_mode,
    condition_spec,
    digest,
    replace_image,
    train_mean,
)


def test_no_cross_checkpoint_donor_and_exact_condition_set():
    assert len(CONDITIONS) == len(set(CONDITIONS)) == 10
    for condition in CONDITIONS:
        base, donor = condition_spec(condition)
        assert base in (640, 1280) and donor is None
        assert component_mode(condition) in {"native", "krestore", "vrestore", "kmean", "vmean"}
    with pytest.raises(ValueError, match="Unregistered"):
        condition_spec("base640_kmean1280")


def test_only_train_rows_enter_mean_and_positions_are_retained():
    values = torch.arange(45 * 3, dtype=torch.float32).reshape(45, 1, 3, 1)
    rows = list(range(39, -1, -1))
    original = train_mean(values, rows)
    values[40:] = 1e10
    assert torch.equal(train_mean(values, rows), original)
    assert original.shape == (1, 3, 1)
    assert torch.equal(original.float(), torch.tensor([[[58.5], [59.5], [60.5]]]))
    with pytest.raises(ValueError, match="duplicate"):
        train_mean(values, [0, 0])
    with pytest.raises(ValueError, match="Invalid"):
        train_mean(values, [45])


@pytest.mark.parametrize("mode", ["mean", "restore"])
def test_full_prefix_and_bf16_restore_do_not_use_subtraction(mode):
    native = torch.ones((1, 241, 320), dtype=torch.bfloat16) * 0.003
    mean = torch.ones((1, 64, 320), dtype=torch.bfloat16) * 1000
    before = native.clone()
    output = replace_image(native, mean, mode)
    assert torch.equal(native, before)
    assert torch.equal(output[:, 64:], native[:, 64:])
    assert torch.equal(output[:, :64], mean if mode == "mean" else native[:, :64])
    assert not torch.equal(
        (mean.float() + native[:, :64].float() - mean.float()).bfloat16(), native[:, :64]
    )


def bank(mode):
    result = SceneKVMean.__new__(SceneKVMean)
    result.row, result.mode = 0, mode
    result.layout = {"shape": [1, 241, 320]}
    result.calls, result.cache, result.reference = {}, {}, None
    result.means = {"1_k": torch.full((1, 64, 320), 2, dtype=torch.bfloat16)}
    result.means["1_v"] = result.means["1_k"]
    return result


@pytest.mark.parametrize("mode", ["kmean", "vmean", "krestore", "vrestore"])
def test_actual_hook_changes_only_selected_route_and_checks_repeats(monkeypatch, mode):
    instance = bank(mode)
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda *_: True)
    x = torch.ones((1, 241, 320))
    native = x.bfloat16()
    for projection in ("k", "v"):
        hook = instance.hook("1_" + projection)
        output = hook(None, (x,), native)
        assert torch.equal(output[:, 64:], native[:, 64:])
        if mode == projection + "mean":
            assert torch.equal(output[:, :64], 2 * native[:, :64])
        else:
            assert torch.equal(output, native)
        assert torch.equal(hook(None, (x,), native), output)
        with pytest.raises(ValueError, match="varies"):
            hook(None, (x + 1,), native)


def test_hook_rejects_slice_confusion_and_wrong_mask(monkeypatch):
    instance = bank("kmean")
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda *_: True)
    short = torch.ones((1, 64, 320), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="Full native"):
        instance.hook("1_k")(None, (short,), short)
    with pytest.raises(ValueError, match="73 active"):
        instance.set_layout(torch.ones((1, 241), dtype=torch.bool))


def calibration(tmp_path):
    modules = {}
    for layer in LAYERS:
        for kind in ("k", "v"):
            name = f"model.vlm_with_expert.lm_expert.layers.{layer}.self_attn.{kind}_proj"
            modules[name] = torch.nn.Linear(320, 320, bias=False)
    arrays = {
        f"{layer}_{kind}": np.zeros((1, 64, 320), dtype=np.float32)
        for layer in LAYERS
        for kind in ("k", "v")
    }
    np.savez(tmp_path / "calibration.npz", **arrays)
    value = {
        "base_step": 640,
        "checkpoint_sha256": "a" * 64,
        "episodes": list(range(45)),
        "training_episodes": list(range(40)),
        "training_rows": list(range(40)),
        "labels_used": False,
        "arrays_sha256": hashlib.sha256((tmp_path / "calibration.npz").read_bytes()).hexdigest(),
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(value))
    result = {
        "condition": "base640_native",
        "scene_kv_mean": {"calibration_sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
        "same_device_control": {
            "passed": True,
            "normalized_predictions": {"max_abs": 0},
            "standard_predictions": {"max_abs": 0},
        },
    }
    (tmp_path / "result.json").write_text(json.dumps(result))
    return SimpleNamespace(named_modules=lambda: modules.items()), modules


def test_calibration_constructor_installs_and_removes_both_projection_hooks(tmp_path):
    policy, modules = calibration(tmp_path)
    instance = SceneKVMean(
        policy, "a" * 64, "base640_kmean", tmp_path, list(range(45)), list(range(40))
    )
    assert len(instance.handles) == 16 and len(instance.means) == 16
    instance.close()
    assert not any(m._forward_hooks for m in modules.values())


@pytest.mark.parametrize("change", ["checkpoint", "base", "split", "manifest", "arrays", "control"])
def test_tampered_or_cross_checkpoint_calibration_rejected(tmp_path, change):
    policy, _ = calibration(tmp_path)
    checkpoint, condition, train = "a" * 64, "base640_vmean", list(range(40))
    if change == "checkpoint":
        checkpoint = "b" * 64
    elif change == "base":
        condition = "base1280_vmean"
    elif change == "split":
        train[-1] = 44
    elif change in {"manifest", "arrays"}:
        path = tmp_path / ("calibration.json" if change == "manifest" else "calibration.npz")
        with path.open("ab") as stream:
            stream.write(b" ")
    else:
        path = tmp_path / "result.json"
        result = json.loads(path.read_text())
        result["same_device_control"]["normalized_predictions"]["max_abs"] = 1e-8
        path.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        SceneKVMean(policy, checkpoint, condition, tmp_path, list(range(45)), train)


def test_digest_preserves_dtype_shape_and_signed_zero():
    assert digest(torch.tensor([0.0])) != digest(torch.tensor([-0.0]))
    assert digest(torch.tensor([0.0])) != digest(torch.tensor([0.0], dtype=torch.bfloat16))


def test_incomplete_hook_coverage_cannot_seal(tmp_path):
    with pytest.raises(ValueError, match="coverage"):
        bank("native").finish(tmp_path)


def test_transfer_allowlist_keeps_new_conditions_and_rejects_old_or_escape():
    from scripts.verify_hestia_scene_kv_mean_results import validate_manifest

    item = {"bytes": 1, "sha256": "a" * 64}
    files = {n: item for n in ("registration.json", "permit.json", "worker-exited.json")}
    files.update({f"{condition}/result.json": item for condition in CONDITIONS})
    assert validate_manifest({"files": files, "total_bytes": len(files)}) == files
    for bad in ("base640/result.json", "base640_native/../../other", "base640_native/private.key"):
        altered = {**files, bad: item}
        with pytest.raises(ValueError):
            validate_manifest({"files": altered, "total_bytes": len(altered)})
