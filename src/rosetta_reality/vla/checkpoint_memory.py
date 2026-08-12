"""Bound transient checkpoint memory without changing live training state."""

from __future__ import annotations

import ctypes
import gc
from functools import wraps
from typing import Any


def release_checkpoint_headroom() -> None:
    """Return unreachable host and unused XPU allocations before serialization."""

    gc.collect()
    import torch

    if torch.xpu.is_available():
        torch.xpu.empty_cache()
    libc = ctypes.CDLL(None)
    malloc_trim = getattr(libc, "malloc_trim", None)
    if malloc_trim is not None:
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        malloc_trim(0)


def install_checkpoint_memory_trim(lerobot_train: Any) -> None:
    """Trim transient allocations after resume load and before every checkpoint."""

    marker = "_rosetta_checkpoint_memory_trim_installed"
    if getattr(lerobot_train, marker, False):
        return
    original_resume = lerobot_train.resume_after_prepare
    original_save = lerobot_train.save_checkpoint

    @wraps(original_resume)
    def resume_after_prepare(*args: Any, **kwargs: Any) -> Any:
        result = original_resume(*args, **kwargs)
        release_checkpoint_headroom()
        return result

    @wraps(original_save)
    def save_checkpoint(*args: Any, **kwargs: Any) -> Any:
        release_checkpoint_headroom()
        return original_save(*args, **kwargs)

    lerobot_train.resume_after_prepare = resume_after_prepare
    lerobot_train.save_checkpoint = save_checkpoint
    setattr(lerobot_train, marker, True)
