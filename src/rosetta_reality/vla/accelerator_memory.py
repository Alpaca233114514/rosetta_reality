"""Small device-neutral helpers for CUDA/XPU synchronization and memory evidence."""

from __future__ import annotations

from typing import Any

_ACCELERATOR_TYPES = frozenset({"cuda", "xpu"})


def device_type(device: Any) -> str:
    """Return a normalized torch device type without importing torch."""

    value = getattr(device, "type", device)
    if not isinstance(value, str) or not value:
        raise ValueError("Accelerator device must have a non-empty string type.")
    return value.split(":", maxsplit=1)[0].lower()


def _backend(torch_module: Any, device: Any) -> tuple[str, Any | None]:
    kind = device_type(device)
    if kind not in _ACCELERATOR_TYPES:
        return kind, None
    backend = getattr(torch_module, kind, None)
    if backend is None:
        raise RuntimeError(f"PyTorch has no {kind.upper()} backend.")
    return kind, backend


def require_available(torch_module: Any, device: Any) -> None:
    """Fail closed when a requested CUDA/XPU backend is unavailable."""

    kind, backend = _backend(torch_module, device)
    if backend is not None and not bool(backend.is_available()):
        raise RuntimeError(f"The requested {kind.upper()} is unavailable.")


def synchronize(torch_module: Any, device: Any) -> None:
    """Synchronize CUDA/XPU and leave CPU execution untouched."""

    _kind, backend = _backend(torch_module, device)
    if backend is not None:
        backend.synchronize()


def reset_peak_memory_stats(torch_module: Any, device: Any) -> None:
    """Reset peak allocation accounting for CUDA/XPU."""

    _kind, backend = _backend(torch_module, device)
    if backend is not None:
        backend.reset_peak_memory_stats()


def memory_snapshot(torch_module: Any, device: Any) -> dict[str, int]:
    """Return the common allocation contract used by local and AutoDL reports."""

    _kind, backend = _backend(torch_module, device)
    if backend is None:
        return {}
    return {
        "allocated_bytes": int(backend.memory_allocated()),
        "reserved_bytes": int(backend.memory_reserved()),
        "maximum_allocated_bytes": int(backend.max_memory_allocated()),
    }


def tracking_memory_metrics(torch_module: Any, device: Any) -> dict[str, int]:
    """Expose generic metrics plus a backend alias for historical dashboards."""

    kind = device_type(device)
    snapshot = memory_snapshot(torch_module, kind)
    if not snapshot:
        return {}
    generic = {
        "accelerator_allocated_bytes": snapshot["allocated_bytes"],
        "accelerator_reserved_bytes": snapshot["reserved_bytes"],
        "accelerator_max_allocated_bytes": snapshot["maximum_allocated_bytes"],
    }
    aliases = {
        f"{kind}_allocated_bytes": snapshot["allocated_bytes"],
        f"{kind}_reserved_bytes": snapshot["reserved_bytes"],
        f"{kind}_max_allocated_bytes": snapshot["maximum_allocated_bytes"],
    }
    return {**generic, **aliases}


def empty_accelerator_cache(torch_module: Any, device: Any | None = None) -> None:
    """Release unused CUDA/XPU cache without assuming one accelerator family."""

    if device is not None:
        _kind, backend = _backend(torch_module, device)
        if backend is not None and bool(backend.is_available()):
            backend.empty_cache()
        return
    for kind in ("cuda", "xpu"):
        backend = getattr(torch_module, kind, None)
        if backend is not None and bool(backend.is_available()):
            backend.empty_cache()
