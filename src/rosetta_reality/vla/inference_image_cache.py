"""Bounded, restorable image-embedding reuse for frozen diagnostic inference.

Cache actual embeddings, including masked placeholders, rather than substituting
zeros or changing attention masks. Each key binds pixel bytes, device, dtype
and autocast. Any parameter identity/version change invalidates the context
with an error. This extension is never installed in a trainer or Gate runner.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any


class FrozenImageEmbeddingCache:
    """Cache one eval-mode VLM's embed_image calls within a no-gradient context."""

    def __init__(self, vlm: Any, *, max_bytes: int = 64 * 1024 * 1024) -> None:
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("Image cache requires a positive byte budget.")
        self.vlm = vlm
        self.max_bytes = max_bytes
        self.entries: OrderedDict[tuple, Any] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.cached_bytes = 0
        self.peak_cached_bytes = 0
        self.original = None
        self.wrapped = None

    def _parameter_identity(self) -> tuple:
        return tuple(
            (name, id(parameter), parameter._version, str(parameter.device), parameter.dtype)
            for name, parameter in self.vlm.named_parameters()
        )

    def __enter__(self):
        import torch

        if self.original is not None or getattr(
            self.vlm.embed_image, "_rosetta_image_cache", False
        ):
            raise RuntimeError("An image cache is already installed.")
        if self.vlm.training:
            raise RuntimeError("Image caching requires an eval-mode VLM.")
        self.identity = self._parameter_identity()
        self.original = self.vlm.embed_image

        def embed_image(image):
            if torch.is_grad_enabled() or self.vlm.training:
                raise RuntimeError("Image caching is forbidden during training or autograd.")
            if self._parameter_identity() != self.identity:
                raise RuntimeError("VLM parameters changed during cached inference.")
            # uint8 byte view also supports bfloat16, which NumPy cannot read directly.
            digest = hashlib.sha256(
                image.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
            ).digest()
            device_type = image.device.type
            key = (
                tuple(image.shape),
                str(image.device),
                image.dtype,
                digest,
                torch.is_autocast_enabled(device_type),
                torch.get_autocast_dtype(device_type),
            )
            if key in self.entries:
                self.hits += 1
                self.entries.move_to_end(key)
                return self.entries[key].clone()
            self.misses += 1
            output = self.original(image)
            size = output.numel() * output.element_size()
            if size <= self.max_bytes:
                while self.entries and self.cached_bytes + size > self.max_bytes:
                    _, removed = self.entries.popitem(last=False)
                    self.cached_bytes -= removed.numel() * removed.element_size()
                self.entries[key] = output.detach().clone()
                self.cached_bytes += size
                self.peak_cached_bytes = max(self.peak_cached_bytes, self.cached_bytes)
            return output

        embed_image._rosetta_image_cache = True
        self.wrapped = embed_image
        self.vlm.embed_image = embed_image
        return self

    def __exit__(self, *_):
        if self.vlm.embed_image is not self.wrapped:
            raise RuntimeError("The image embedding method changed during caching.")
        self.vlm.embed_image = self.original
        self.original = None
        self.wrapped = None
        self.entries.clear()
        self.cached_bytes = 0

    def report(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "encoder_calls": self.misses,
            "max_cache_bytes": self.max_bytes,
            "peak_cache_bytes": self.peak_cached_bytes,
        }
