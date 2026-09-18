"""Exhaustive pixel values and native-cycle conversion/restore boundaries."""

from types import SimpleNamespace

import pytest
import torch

from rosetta_reality.vla.image_scaling import canonical_rgb_uint8
from rosetta_reality.vla.training import features


def test_all_pixel_values_and_noncontiguous_input_are_preserved():
    image = torch.arange(256, dtype=torch.uint8).reshape(1, 16, 16).expand(3, -1, -1)
    before = image.clone()
    result = canonical_rgb_uint8(image)
    expected = torch.tensor([float(i) / 255 for i in range(256)]).reshape(1, 16, 16)
    assert torch.equal(result, expected.expand(3, -1, -1))
    assert torch.equal(image, before)
    assert result.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_all_pixel_values_equal_cpu_exactly():
    image = torch.arange(256, dtype=torch.uint8).reshape(1, 16, 16).expand(3, -1, -1)
    assert torch.equal(canonical_rgb_uint8(image.cuda()).cpu(), image.float() / 255)


@pytest.mark.parametrize("image", [torch.zeros(3, 2, 2), torch.zeros(2, 2, dtype=torch.uint8),
                                  torch.zeros(1, 2, 2, dtype=torch.uint8)])
def test_rejects_ambiguous_or_non_rgb_inputs(image):
    with pytest.raises(ValueError):
        canonical_rgb_uint8(image)


def test_native_loop_does_not_double_scale_or_change_metadata(monkeypatch):
    raw = {"observation.images.top": torch.full((4, 3, 2, 2), 128, dtype=torch.uint8),
           "frame_index": torch.tensor([0, 249, 450, 499])}
    module = SimpleNamespace(cycle=lambda: iter([raw]))
    original = module.cycle
    monkeypatch.setattr(features, "_lerobot_train_module", lambda: module)
    context = SimpleNamespace(experiment={"dataset": {"rename_map": {
        "observation.images.top": "observation.images.camera1"
    }}})
    feature = features.CanonicalImageScalingFeature({})
    feature.install(context)
    try:
        converted = next(module.cycle())
        assert converted is not raw
        assert converted["frame_index"] is raw["frame_index"]
        image = converted["observation.images.top"]
        # The pinned native trainer scales only uint8, so this branch must be skipped.
        if image.dtype == torch.uint8:
            image = image.float() / 255
        assert torch.equal(image, raw["observation.images.top"].float() / 255)
        assert raw["observation.images.top"].dtype == torch.uint8
        with pytest.raises(RuntimeError, match="already installed"):
            features.CanonicalImageScalingFeature({}).install(context)
    finally:
        feature.restore(context)
    assert module.cycle is original
