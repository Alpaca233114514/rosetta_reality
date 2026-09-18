"""Device-independent float32 representation of RGB bytes, without model imports."""

from __future__ import annotations


def canonical_rgb_uint8(image):
    """Use a CPU-defined 256-value table on the input device; never divide on CUDA.

    Native float32 division by 255 rounds differently on CPU and CUDA. Selecting
    identical float32 values from this table preserves CPU preprocessing semantics
    on either device. Input bytes/layout/metadata are never modified in place.
    """
    import torch

    if not isinstance(image, torch.Tensor) or image.dtype != torch.uint8:
        raise ValueError("Canonical RGB scaling requires uint8 input bytes")
    if image.ndim < 3 or image.shape[-3] != 3 or image.numel() == 0:
        raise ValueError("Canonical RGB scaling requires nonempty (...,3,H,W)")
    table = torch.tensor(
        [value / 255.0 for value in range(256)], dtype=torch.float32, device="cpu"
    )
    return table.to(image.device)[image.to(dtype=torch.long)]
