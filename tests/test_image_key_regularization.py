"""Analytic and lifecycle checks; no model weights or dataset required."""

from types import SimpleNamespace

import pytest
import torch

from rosetta_reality.vla.image_key_regularization import (
    LAYERS,
    MARKER,
    install,
    load_calibration,
    regularized_forward,
    restore,
    scene_variance,
    validate_scales,
)


def test_unbiased_value_and_analytic_gradient():
    x = torch.tensor(
        [[[1.0, 3.0]], [[5.0, 7.0]], [[3.0, 2.0]]], dtype=torch.float64, requires_grad=True
    )
    value = scene_variance(x)
    expected = x.var(dim=0, correction=1).mean()
    torch.testing.assert_close(value, expected)
    value.backward()
    torch.testing.assert_close(x.grad, 2 * (x.detach() - x.detach().mean(0)) / 4)


def test_permutation_and_shared_token_pattern():
    x = torch.arange(48, dtype=torch.float64).reshape(4, 6, 2)
    pattern = torch.arange(12, dtype=torch.float64).reshape(1, 6, 2) * 100
    torch.testing.assert_close(scene_variance(x), scene_variance(x[[3, 0, 2, 1]]))
    torch.testing.assert_close(scene_variance(x), scene_variance(x + pattern))
    assert scene_variance(pattern.expand(4, -1, -1)) == 0


@pytest.mark.parametrize("value", [0, -1, 1e-12, float("nan"), float("inf"), True, "1"])
def test_invalid_calibration(value):
    with pytest.raises(ValueError):
        validate_scales(dict.fromkeys(LAYERS, value))


@pytest.mark.parametrize("shape", [(1, 64, 320), (2, 0, 320), (2, 64)])
def test_invalid_shape(shape):
    with pytest.raises(ValueError):
        scene_variance(torch.zeros(shape))


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(add_image_special_tokens=False, prefix_length=0)
        vlm = SimpleNamespace(
            vision_model=SimpleNamespace(config=SimpleNamespace(patch_size=16)),
            connector=SimpleNamespace(scale_factor=2),
        )
        self.model = SimpleNamespace(vlm_with_expert=SimpleNamespace(get_vlm_model=lambda: vlm))
        self.projections = torch.nn.ModuleList(
            [torch.nn.Linear(320, 320, bias=False) for _ in LAYERS]
        )

    def prepare_images(self, x):
        image = torch.zeros(x.shape[0], 3, 256, 256)
        mask = torch.ones(x.shape[0], dtype=torch.bool)
        return [image, image - 1, image - 1], [mask, ~mask, ~mask]

    def named_modules(self, *args, **kwargs):
        return iter(
            (f"model.vlm_with_expert.lm_expert.layers.{layer}.self_attn.k_proj", p)
            for layer, p in zip(LAYERS, self.projections, strict=True)
        )


def native(policy, x, mode="normal", **kwargs):
    policy.prepare_images(x)
    loss = x.sum() * 0
    for i, projection in enumerate(policy.projections):
        if mode == "missing" and i == 7:
            continue
        y = projection(x)
        if mode == "duplicate" and i == 0:
            projection(x)
        loss = loss + y[:, 64:].square().mean()
    if mode == "error":
        raise RuntimeError("native error")
    return loss, {"loss": float(loss.detach())}


def test_full_forward_gradient_and_only_real_image_slice():
    torch.manual_seed(2)
    policy = Policy()
    x = torch.randn(4, 241, 320, requires_grad=True)
    scales = dict.fromkeys(LAYERS, 1.0)
    baseline, _ = native(policy, x)
    baseline_grad = torch.autograd.grad(baseline, x)[0]
    total, metrics = regularized_forward(policy, native, (x,), {}, scales, 0.01)
    grad = torch.autograd.grad(total, x)[0]
    torch.testing.assert_close(grad[:, 64:], baseline_grad[:, 64:], rtol=0, atol=0)
    assert bool((grad[:, :64] != 0).any())
    assert metrics["total_loss"] > metrics["flow_loss"]
    assert metrics["loss"] == metrics["total_loss"] == float(total.detach())
    assert not any(p._forward_hooks for p in policy.projections)
    assert not getattr(policy, MARKER)


def test_actual_regularizer_metrics_pass_unchanged_public_sanitizer():
    from rosetta_reality.tracking.public_payload import sanitize_metrics

    policy = Policy()
    _, metrics = regularized_forward(
        policy, native, (torch.randn(4, 241, 320),), {}, dict.fromkeys(LAYERS, 1.0), 0.01
    )
    public = sanitize_metrics(metrics, mode="train")
    assert public["train/image_k_scene_loss"] == metrics["image_k_scene_loss"]
    for layer in LAYERS:
        name = f"image_k_scene_layer_{layer}"
        assert public["train/" + name] == metrics[name]
    assert public["train/total_loss"] == metrics["total_loss"]
    with pytest.raises(ValueError, match="forbidden sensitive"):
        sanitize_metrics({**metrics, "api_key": 1.0}, mode="train")


@pytest.mark.parametrize(
    "mode,error", [("missing", ValueError), ("duplicate", ValueError), ("error", RuntimeError)]
)
def test_failure_cleans_hooks(mode, error):
    p = Policy()
    x = torch.randn(4, 241, 320)
    with pytest.raises(error):
        regularized_forward(p, native, (x, mode), {}, dict.fromkeys(LAYERS, 1.0), 0.01)
    assert not any(layer._forward_hooks for layer in p.projections)
    assert not getattr(p, MARKER)


def test_eval_and_zero_bypass_preserve_rng_and_output():
    p = Policy()
    x = torch.randn(1, 241, 320)
    before = torch.get_rng_state().clone()
    expected = native(p, x)
    actual = regularized_forward(p, native, (x,), {}, {}, 0)
    assert torch.equal(expected[0], actual[0])
    p.eval()
    actual = regularized_forward(p, native, (x,), {}, {}, 0.01)
    assert torch.equal(expected[0], actual[0])
    assert torch.equal(before, torch.get_rng_state())
    assert not any(layer._forward_hooks for layer in p.projections)


def test_install_zero_exact_function_restore_and_drift(tmp_path):
    import hashlib

    source = tmp_path / "modeling.py"
    source.write_text("pinned stub")

    class Stub:
        forward = native

    modeling = SimpleNamespace(__file__=str(source), SmolVLAPolicy=Stub)
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    original = Stub.forward
    install(modeling, dict.fromkeys(LAYERS, 1.0), 0, upstream_sha256=sha)
    assert Stub.forward is original
    with pytest.raises(RuntimeError):
        install(modeling, dict.fromkeys(LAYERS, 1.0), 0, upstream_sha256=sha)
    restore(modeling)
    assert Stub.forward is original
    with pytest.raises(ValueError):
        install(modeling, dict.fromkeys(LAYERS, 1.0), 0, upstream_sha256="0" * 64)


@pytest.mark.parametrize(
    "mode",
    ["mask_shape", "mask_dtype", "masked_real", "active_empty", "nonfinite", "image_size", "batch"],
)
def test_bad_camera_layout_restores_method_and_hooks(mode):
    policy = Policy()
    original = policy.prepare_images

    def broken(x):
        images, masks = original(x)
        if mode == "mask_shape":
            masks[0] = masks[0][:, None]
        elif mode == "mask_dtype":
            masks[0] = masks[0].float()
        elif mode == "masked_real":
            masks[0][0] = False
        elif mode == "active_empty":
            masks[1][0] = True
        elif mode == "nonfinite":
            images[0][0, 0, 0, 0] = float("nan")
        elif mode == "image_size":
            images = [
                torch.zeros(4, 3, 257, 256),
                torch.full((4, 3, 257, 256), -1.0),
                torch.full((4, 3, 257, 256), -1.0),
            ]
        else:
            images[1] = images[1][:1]
        return images, masks

    policy.prepare_images = broken
    with pytest.raises(ValueError):
        regularized_forward(
            policy, native, (torch.randn(4, 241, 320),), {}, dict.fromkeys(LAYERS, 1.0), 0.01
        )
    assert policy.prepare_images is broken
    assert not any(layer._forward_hooks for layer in policy.projections)
    assert not getattr(policy, MARKER)


@pytest.mark.parametrize("mode", ["bool_count", "duplicate", "alias_layer", "nonfinite", "hash"])
def test_calibration_rejects_ambiguous_or_changed_identity(tmp_path, mode):
    import hashlib
    import json

    path = tmp_path / "calibration.json"
    expected = {"status": "passed", "optimizer_steps": 0, "labels_used": False}
    value = {**expected, "layer_variance": {str(layer): 1.0 for layer in LAYERS}}
    if mode == "bool_count":
        value["optimizer_steps"] = False
    elif mode == "alias_layer":
        value["layer_variance"]["01"] = value["layer_variance"].pop("1")
    elif mode == "nonfinite":
        value["layer_variance"]["1"] = float("nan")
    raw = json.dumps(value)
    if mode == "duplicate":
        raw = raw.replace('"optimizer_steps": 0', '"optimizer_steps": 0, "optimizer_steps": 0')
    path.write_text(raw)
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if mode != "hash" else "0" * 64
    with pytest.raises(ValueError):
        load_calibration(path, digest, expected)


def test_calibration_exact_identity(tmp_path):
    import hashlib
    import json

    path = tmp_path / "calibration.json"
    expected = {"status": "passed", "optimizer_steps": 0, "labels_used": False}
    path.write_text(
        json.dumps({**expected, "layer_variance": {str(layer): 2.0 for layer in LAYERS}})
    )
    assert load_calibration(
        path, hashlib.sha256(path.read_bytes()).hexdigest(), expected
    ) == dict.fromkeys(LAYERS, 2.0)
