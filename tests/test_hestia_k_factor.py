"""Counterexamples for per-head factorization and protected-prefix identity."""

import numpy as np
import pytest
import torch

from scripts.hestia_k_factor import CONDITIONS, component_mode, condition_spec, factors


def example():
    k = torch.zeros(1, 241, 320, dtype=torch.bfloat16)
    k.reshape(1, 241, 5, 64)[..., 0] = 3
    k.reshape(1, 241, 5, 64)[..., 1] = 4
    m = torch.zeros(1, 64, 320, dtype=torch.bfloat16)
    m.reshape(1, 64, 5, 64)[..., 1] = 2
    avg = torch.full((1, 64, 5, 1), 7, dtype=torch.float64)
    return k, m, avg


def test_single_checkpoint_and_complete_factorial():
    assert len(CONDITIONS) == len(set(CONDITIONS)) == 7
    assert all(condition_spec(c) == (640, None) for c in CONDITIONS)
    with pytest.raises(ValueError, match="Unregistered"):
        condition_spec("base1280_kdirection")


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("krestore", (3, 4)),
        ("kmean", (0, 2)),
        ("kdirection", (0, 5)),
        ("kmagnitude", (1.2, 1.6)),
        ("knormavg", (4.2, 5.6)),
        ("kdiravg", (0, 7)),
    ],
)
def test_actual_head_coordinates_and_no_input_mutation(mode, expected):
    k, m, avg = example()
    before = k.clone()
    out = factors(k, m, avg, 64, mode)
    assert torch.equal(k, before)
    assert torch.equal(out[:, 64:], k[:, 64:])
    target = torch.tensor(expected, dtype=torch.bfloat16)
    assert torch.equal(out[:, :64].reshape(1, 64, 5, 64)[..., :2], target.expand(1, 64, 5, 2))
    assert not out[:, :64].reshape(1, 64, 5, 64)[..., 2:].any()


def test_headwise_scaling_is_not_flat_vector_scaling():
    k, m, avg = example()
    k.reshape(1, 241, 5, 64)[:, :, 1] *= 2
    out = factors(k, m, avg, 64, "kdirection").reshape(1, 241, 5, 64)
    assert out[0, 0, 0, 1] == 5 and out[0, 0, 1, 1] == 10


@pytest.mark.parametrize(
    "which", ["native_zero", "mean_zero", "nan", "norm_zero", "norm_shape", "head"]
)
def test_undefined_directions_stop_instead_of_silent_fallback(which):
    k, m, avg = example()
    dim = 64
    if which == "native_zero":
        k[:, :64] = 0
    elif which == "mean_zero":
        m.zero_()
    elif which == "nan":
        m[0, 0, 0] = float("nan")
    elif which == "norm_zero":
        avg.zero_()
    elif which == "norm_shape":
        avg = avg[:, :1]
    else:
        dim = 63
    with pytest.raises(ValueError):
        factors(k, m, avg, dim, "kdirection")


def test_norm_of_mean_and_mean_of_norm_are_distinct():
    # Antipodal components cancel in the mean but not in average norm.
    values = np.array([[3.0, 4.0], [-3.0, 4.0]])
    assert np.linalg.norm(values.mean(0)) == 4
    assert np.linalg.norm(values, axis=-1).mean() == 5


def test_transfer_allowlist_has_no_other_checkpoint_or_unregistered_file():
    from scripts.verify_hestia_k_factor_results import validate_manifest

    item = {"bytes": 1, "sha256": "a" * 64}
    files = {k: item for k in ("registration.json", "permit.json", "worker-exited.json")}
    files.update({c + "/result.json": item for c in CONDITIONS})
    files["base640_native/native-keys.npz"] = item
    assert validate_manifest({"files": files, "total_bytes": len(files)}) == files
    for name in ("base1280_native/result.json", "../private", "base640_native/secret.key"):
        with pytest.raises(ValueError):
            validate_manifest({"files": {**files, name: item}, "total_bytes": len(files) + 1})


def test_every_mode_maps_to_its_registered_name():
    assert all(c == "base640_" + component_mode(c) for c in CONDITIONS)


def test_native_seal_train_only_and_real_hook_roundtrip(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from scripts.hestia_k_factor import LAYERS, KeyFactor
    from scripts.hestia_scene_kv_mean import digest

    modules = {}
    layers = {}
    for layer in LAYERS:
        attn = SimpleNamespace(head_dim=64, q_proj=SimpleNamespace(out_features=960))
        for kind in ("k", "v"):
            module = torch.nn.Linear(320, 320, bias=False)
            setattr(attn, kind + "_proj", module)
            modules[f"model.vlm_with_expert.lm_expert.layers.{layer}.self_attn.{kind}_proj"] = (
                module
            )
        layers[layer] = SimpleNamespace(self_attn=attn)
    core = SimpleNamespace(
        lm_expert=SimpleNamespace(layers=layers),
        num_key_value_heads=5,
        num_attention_heads=15,
        config=SimpleNamespace(text_config=SimpleNamespace(head_dim=64)),
    )
    policy = SimpleNamespace(
        model=SimpleNamespace(vlm_with_expert=core),
        named_modules=lambda: modules.items(),
        parameters=lambda: iter(next(iter(modules.values())).parameters()),
    )
    native = KeyFactor(
        policy, "a" * 64, "base640_native", tmp_path, list(range(45)), list(range(40))
    )
    mask = torch.zeros(1, 241, dtype=torch.bool)
    mask[:, :64] = True
    mask[:, 192:200] = True
    mask[:, 240] = True
    native.set_layout(mask)
    for row in range(45):
        value = torch.full((1, 241, 320), row + 1 if row < 40 else 999, dtype=torch.bfloat16)
        for layer in LAYERS:
            for kind in ("k", "v"):
                pair = row, f"{layer}_{kind}"
                native.cache[pair] = (
                    value,
                    value,
                    {"input": digest(value), "native": digest(value)},
                )
                native.calls[pair] = 40
    native.close()
    report = native.finish(tmp_path)
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "condition": "base640_native",
                "k_factor": report,
                "same_device_control": {
                    "passed": True,
                    "normalized_predictions": {"max_abs": 0},
                    "standard_predictions": {"max_abs": 0},
                },
            }
        )
    )
    with np.load(tmp_path / "calibration.npz") as arrays:
        assert np.all(arrays["1_mean"] == 20.5)
        assert np.all(arrays["1_norm"] == 164.0)
    restored = KeyFactor(
        policy, "a" * 64, "base640_kmagnitude", tmp_path, list(range(45)), list(range(40))
    )
    restored.set_layout(mask)
    restored.row = 0
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda *_: True)
    value = torch.ones(1, 241, 320, dtype=torch.bfloat16)
    actual = restored.hook("1_k")(None, (value,), value)
    assert torch.equal(actual[:, :64], torch.full((1, 64, 320), 20.5, dtype=torch.bfloat16))
    assert torch.equal(actual[:, 64:], value[:, 64:])
    assert torch.equal(restored.hook("1_v")(None, (value,), value), value)
    with pytest.raises(ValueError, match="varies"):
        restored.hook("1_k")(None, (value + 1,), value)
    restored.close()
    assert not any(module._forward_hooks for module in modules.values())
