"""Observer identity, noninterference, drift and authorization counterexamples."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import diagnose_hestia_prefix_path as collector
from scripts.diagnose_hestia_prefix_path import PrefixObserver, validate_permit


def record_all(observer, values):
    for layer in (1, 15):
        for kind in ("k", "v"):
            for stage in ("input", "projected"):
                observer.record(f"layer{layer}_{kind}_{stage}", values)


def test_hooks_return_none_and_do_not_change_forward(tmp_path):
    observer = PrefixObserver(tmp_path, camera_tokens=2, steps=1, rows=1)
    layer = torch.nn.Linear(3, 3)
    x = torch.ones(1, 4, 3)
    expected = layer(x).detach().clone()
    observer.begin(0)
    h1 = layer.register_forward_pre_hook(observer.pre_hook("sample_in"))
    h2 = layer.register_forward_hook(observer.post_hook("sample_out"))
    actual = layer(x)
    h1.remove()
    h2.remove()
    assert torch.equal(expected, actual)
    assert observer.references["sample_in"].shape == (1, 2, 3)


def test_prefix_only_and_across_noise_identity(tmp_path):
    observer = PrefixObserver(tmp_path, camera_tokens=2, steps=1, rows=1)
    x = torch.ones(1, 4, 3)
    for index in range(4):
        observer.begin(index)
        x[:, 2:] = index + 10
        record_all(observer, x)
        observer.finish()
    assert len(observer.records) == 1
    with np.load(tmp_path / "prefix-row-000.npz") as a:
        assert len(a.files) == 8
        assert all(v.shape == (1, 2, 3) and (v == 1).all() for v in a.values())


def test_denoising_drift_rejected(tmp_path):
    o = PrefixObserver(tmp_path, camera_tokens=2)
    o.begin(0)
    o.record("x", torch.ones(1, 2, 3))
    with pytest.raises(ValueError, match="denoising"):
        o.record("x", torch.zeros(1, 2, 3))


def test_cross_noise_drift_rejected(tmp_path):
    o = PrefixObserver(tmp_path, camera_tokens=2, steps=1, rows=1)
    o.begin(0)
    record_all(o, torch.ones(1, 2, 3))
    o.finish()
    o.begin(1)
    record_all(o, torch.zeros(1, 2, 3))
    with pytest.raises(ValueError, match="across"):
        o.finish()


def test_missing_projection_and_output_budget_rejected(tmp_path):
    o = PrefixObserver(tmp_path, camera_tokens=2, steps=1, maximum_bytes=1)
    o.begin(0)
    with pytest.raises(ValueError, match="Missing"):
        o.finish()
    record_all(o, torch.ones(1, 2, 3))
    with pytest.raises(ValueError, match="bound"):
        o.finish()


def permit():
    return dict(
        allowed_steps=[1280],
        model_execution_authorized=True,
        training_authorized=False,
        shutdown_authorized=True,
        template_sha256="a",
        watchdog_active=True,
        started_unix=10,
        deadline_unix=490,
    )


def test_single_checkpoint_and_expired_permit():
    validate_permit("a", permit(), 1280, 20)
    with pytest.raises(ValueError, match="final checkpoint"):
        validate_permit("a", permit(), 640, 20)
    with pytest.raises(ValueError, match="expired"):
        validate_permit("a", permit(), 1280, 500)


@pytest.mark.parametrize(
    "field", ["model_execution_authorized", "shutdown_authorized", "watchdog_active"]
)
def test_authorization_or_watchdog_cannot_be_missing(field):
    p = permit()
    p[field] = False
    with pytest.raises(ValueError):
        validate_permit("a", p, 1280, 20)


@pytest.mark.parametrize("fail", [False, True])
def test_collector_wraps_and_restores_factory_policy_and_hooks(tmp_path, monkeypatch, fail):
    layers = [
        SimpleNamespace(
            self_attn=SimpleNamespace(k_proj=torch.nn.Linear(3, 3), v_proj=torch.nn.Linear(3, 3))
        )
        for _ in range(16)
    ]
    vlm = torch.nn.Linear(1, 1)
    vlm.requires_grad_(False)
    core = SimpleNamespace(
        num_vlm_layers=16,
        num_expert_layers=16,
        self_attn_every_n_layers=2,
        vlm=vlm,
        lm_expert=SimpleNamespace(layers=layers),
        get_vlm_model=lambda: SimpleNamespace(
            vision_model=SimpleNamespace(config=SimpleNamespace(patch_size=16)),
            connector=SimpleNamespace(scale_factor=2),
        ),
    )

    def predict(_batch, **_kwargs):
        if fail:
            raise RuntimeError("synthetic prediction failure")
        x = torch.ones(1, 64, 3)
        for _ in range(10):
            for layer in (1, 15):
                layers[layer].self_attn.k_proj(x)
                layers[layer].self_attn.v_proj(x)
        return torch.zeros(1, 50, 32)

    policy = SimpleNamespace(
        model=SimpleNamespace(vlm_with_expert=core),
        config=SimpleNamespace(add_image_special_tokens=False, attention_mode="cross_attn"),
        predict_action_chunk=predict,
        prepare_images=lambda _batch: (
            [torch.zeros(1, 3, 256, 256)],
            [torch.tensor([True]), torch.tensor([False]), torch.tensor([False])],
        ),
    )

    def factory(**kwargs):
        return policy

    native = SimpleNamespace(make_policy=factory)
    monkeypatch.setitem(sys.modules, "evaluate_visual_native_small", native)

    def collect(_plan, output):
        actual = native.make_policy()
        with torch.inference_mode():
            for _ in range(180):
                assert torch.equal(actual.predict_action_chunk({}), torch.zeros(1, 50, 32))
        (output / "result.json").write_text(
            json.dumps(
                {
                    "same_device_control": {
                        "passed": True,
                        "normalized_predictions": {"max_abs": 0},
                        "standard_predictions": {"max_abs": 0},
                    }
                }
            )
        )

    monkeypatch.setattr(collector, "ORIGINAL_COLLECT", collect)
    if fail:
        with pytest.raises(RuntimeError, match="synthetic"):
            collector.observed_collect({}, tmp_path)
    else:
        collector.observed_collect({}, tmp_path)
        assert (tmp_path / "prefix-observer.json").exists()
        assert len(list(tmp_path.glob("prefix-row-*.npz"))) == 45
    assert native.make_policy is factory
    assert policy.predict_action_chunk is predict
    for layer in (1, 15):
        for kind in ("k_proj", "v_proj"):
            module = getattr(layers[layer].self_attn, kind)
            assert not module._forward_hooks and not module._forward_pre_hooks
