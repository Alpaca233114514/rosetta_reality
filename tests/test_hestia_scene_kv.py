"""Synthetic isolation, shape, precision and rollback checks; no real model/data."""

import numpy as np
import pytest
import torch

from scripts.hestia_scene_kv import (
    CONDITIONS,
    SceneKV,
    calibration_means,
    condition_spec,
    digest,
    replace_image,
    validate_protocol,
)


def test_training_self_exclusion_and_token_order():
    x = np.zeros((40, 64, 320), dtype=np.float32)
    x[3, 12, 2] = 39
    loo, dev = calibration_means(x)
    assert loo[3, 12, 2] == 0
    assert loo[0, 12, 2] == 1
    assert dev[12, 2] == np.float32(39 / 40)
    assert np.count_nonzero(dev) == 1
    # Changing the omitted sample cannot change its own leave-one-out statistic.
    x[3] += 100
    changed, _ = calibration_means(x)
    np.testing.assert_array_equal(changed[3], loo[3])


@pytest.mark.parametrize("count", [39, 41, 45])
def test_calibration_rejects_wrong_or_development_cohort(count):
    with pytest.raises(ValueError, match="Forty"):
        calibration_means(np.zeros((count, 64, 320), dtype=np.float32))


def test_calibration_rejects_nonfinite_and_wrong_precision():
    a = np.zeros((40, 64, 320), dtype=np.float64)
    with pytest.raises(ValueError):
        calibration_means(a)
    a = a.astype(np.float32)
    a[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        calibration_means(a)


def test_replacement_protects_all_other_positions_and_native_tensor():
    native = torch.arange(241).reshape(1, 241, 1).expand(1, 241, 320).bfloat16()
    before = native.clone()
    replacement = torch.full((64, 320), -5, dtype=torch.bfloat16)
    output = replace_image(native, replacement)
    assert torch.equal(output[0, :64], replacement)
    assert torch.equal(output[:, 64:], before[:, 64:])
    assert torch.equal(native, before)
    assert torch.equal(replace_image(native, native[0, :64]), native)


def test_replacement_rejects_the_historical_64_token_slice_error():
    x = torch.zeros((1, 64, 320), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="Full prefix"):
        replace_image(x, x[0])


def test_replacement_rejects_dtype_and_nonfinite():
    x = torch.zeros((1, 241, 320), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="BF16"):
        replace_image(x, torch.zeros((64, 320)))
    with pytest.raises(ValueError, match="finite"):
        replace_image(x, torch.full((64, 320), float("nan"), dtype=torch.bfloat16))


def test_mask_and_native_cpu_rejection():
    bank = SceneKV.__new__(SceneKV)
    bank.layout = bank.reference = None
    bank.row = 0
    mask = torch.ones((1, 241), dtype=torch.bool)
    with pytest.raises(ValueError, match="mask differs"):
        bank.set_layout(mask)
    mask[:, 64:192] = False
    bank.set_layout(mask)
    with pytest.raises(ValueError, match="Full native"):
        bank.set_layout(mask[:, :64])
    changed = mask.clone()
    changed[:, 200] = False
    with pytest.raises(ValueError, match="varies"):
        bank.set_layout(changed)
    with pytest.raises(ValueError, match="Native CUDA"):
        bank.hook(1, "k")(
            None, (torch.zeros((1, 241, 320)),), torch.zeros((1, 241, 320), dtype=torch.bfloat16)
        )


def test_partial_hook_installation_is_restored(tmp_path):
    first = torch.nn.Linear(320, 320, bias=False)
    name = "model.vlm_with_expert.lm_expert.layers.1.self_attn.k_proj"

    class Policy:
        def named_modules(self):
            return [(name, first)]

    with pytest.raises(KeyError):
        SceneKV(Policy(), "base640_native", tmp_path)
    assert len(first._forward_hooks) == 0


def test_digest_keeps_signed_zero_and_dtype():
    assert digest(torch.tensor([0.0])) != digest(torch.tensor([-0.0]))
    assert digest(torch.tensor([0.0])) != digest(torch.tensor([0.0]).bfloat16())


def test_order_is_both_native_then_both_self_then_all_ablations():
    assert CONDITIONS[:4] == ("base640_native", "base1280_native", "base640_self", "base1280_self")
    assert len(set(CONDITIONS)) == 10
    assert condition_spec("base1280_vmean") == (1280, "vmean")
    with pytest.raises(ValueError, match="Unregistered"):
        condition_spec("base640_v1280")


def test_design_cannot_launch():
    with pytest.raises(ValueError, match="non-launchable"):
        validate_protocol({"launchable": False})


def test_manifest_accepts_only_registered_results():
    from scripts.verify_hestia_scene_kv_results import validate_manifest

    item = {"bytes": 1, "sha256": "a" * 64}
    files = {
        key: item
        for key in (
            "registration.json",
            "permit.json",
            "worker-exited.json",
            "base640_native/calibration-train.npz",
        )
    }
    assert validate_manifest({"files": files, "total_bytes": 4}) == files
    files["base640_native/../../secret"] = item
    with pytest.raises(ValueError, match="Unexpected"):
        validate_manifest({"files": files, "total_bytes": 5})


def test_scene_decomposition_separates_wrong_scene_response_from_bias():
    from scripts.analyze_hestia_scene_kv import scene_terms

    target = np.array([-1, 1], dtype=float).reshape(2, 1, 1)
    pred = np.broadcast_to(-target, (4, 2, 1, 1))
    result = scene_terms(pred, target, target)
    assert result["bias"] == [0] * 4
    assert result["variance"] == [1] * 4
    assert result["minus_twice_covariance"] == [2] * 4
    assert result["gap_to_train_mean"] == [3] * 4
    constant = np.full_like(pred, 2)
    result = scene_terms(constant, target, target)
    assert result["variance"] == [0] * 4
    assert result["bias"] == [4] * 4
