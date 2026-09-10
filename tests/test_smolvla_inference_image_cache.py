"""Cache parity, scope and memory tests; no downloaded weights or data."""

import pytest
import torch
from torch import nn

from rosetta_reality.vla.inference_image_cache import FrozenImageEmbeddingCache


class _Vlm(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 2)
        self.calls = 0

    def embed_image(self, image):
        self.calls += 1
        return self.projection(image)


def test_cached_outputs_are_exact_and_returned_tensors_cannot_corrupt_entries():
    model = _Vlm().eval()
    image = torch.ones(1, 3)
    original = model.embed_image
    with torch.inference_mode():
        reference = original(image)
        with FrozenImageEmbeddingCache(model) as cache:
            first = model.embed_image(image)
            first.zero_()
            second = model.embed_image(image.clone())
            assert torch.equal(second, reference)
            second.add_(1)
            assert torch.equal(model.embed_image(image), reference)
            assert cache.report()["hits"] == 2
            assert cache.report()["encoder_calls"] == 1
        assert model.embed_image == original
        assert cache.cached_bytes == 0


def test_changed_pixels_and_autocast_do_not_reuse_an_embedding():
    model = _Vlm().eval()
    with torch.inference_mode(), FrozenImageEmbeddingCache(model) as cache:
        model.embed_image(torch.ones(1, 3))
        model.embed_image(torch.zeros(1, 3))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = model.embed_image(torch.ones(1, 3))
            assert result.dtype == torch.bfloat16
        assert cache.report()["encoder_calls"] == 3


def test_parameter_update_raises_instead_of_using_stale_visual_features():
    model = _Vlm().eval()
    original = model.embed_image
    with torch.no_grad():
        with pytest.raises(RuntimeError, match="parameters changed"):
            with FrozenImageEmbeddingCache(model):
                model.embed_image(torch.ones(1, 3))
                model.projection.weight.add_(1)
                model.embed_image(torch.ones(1, 3))
    assert model.embed_image == original


def test_cache_is_forbidden_in_training_and_autograd():
    model = _Vlm()
    with pytest.raises(RuntimeError, match="eval-mode"):
        with FrozenImageEmbeddingCache(model):
            pass
    model.eval()
    with FrozenImageEmbeddingCache(model):
        with pytest.raises(RuntimeError, match="autograd"):
            model.embed_image(torch.ones(1, 3))
        model.train()
        with torch.no_grad(), pytest.raises(RuntimeError, match="training"):
            model.embed_image(torch.ones(1, 3))


def test_lru_eviction_and_oversize_results_respect_the_byte_budget():
    model = _Vlm().eval()
    with torch.no_grad(), FrozenImageEmbeddingCache(model, max_bytes=8) as cache:
        model.embed_image(torch.ones(1, 3))
        model.embed_image(torch.zeros(1, 3))
        model.embed_image(torch.ones(1, 3))
        model.embed_image(torch.ones(2, 3))
        assert cache.report()["encoder_calls"] == 4
        assert cache.peak_cached_bytes == 8
        assert cache.cached_bytes <= 8
