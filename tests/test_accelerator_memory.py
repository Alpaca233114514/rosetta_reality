from types import SimpleNamespace

import pytest

from rosetta_reality.vla.accelerator_memory import (
    empty_accelerator_cache,
    memory_snapshot,
    require_available,
    reset_peak_memory_stats,
    synchronize,
    tracking_memory_metrics,
)


class FakeBackend:
    def __init__(self, *, available: bool = True) -> None:
        self.available = available
        self.events: list[str] = []

    def is_available(self) -> bool:
        return self.available

    def synchronize(self) -> None:
        self.events.append("synchronize")

    def reset_peak_memory_stats(self) -> None:
        self.events.append("reset")

    def memory_allocated(self) -> int:
        return 10

    def memory_reserved(self) -> int:
        return 20

    def max_memory_allocated(self) -> int:
        return 30

    def empty_cache(self) -> None:
        self.events.append("empty")


@pytest.mark.parametrize("kind", ["cuda", "xpu"])
def test_accelerator_helpers_are_backend_neutral(kind: str) -> None:
    backend = FakeBackend()
    torch_module = SimpleNamespace(**{kind: backend})

    require_available(torch_module, kind)
    reset_peak_memory_stats(torch_module, kind)
    synchronize(torch_module, kind)
    assert memory_snapshot(torch_module, kind) == {
        "allocated_bytes": 10,
        "reserved_bytes": 20,
        "maximum_allocated_bytes": 30,
    }
    assert tracking_memory_metrics(torch_module, kind) == {
        "accelerator_allocated_bytes": 10,
        "accelerator_reserved_bytes": 20,
        "accelerator_max_allocated_bytes": 30,
        f"{kind}_allocated_bytes": 10,
        f"{kind}_reserved_bytes": 20,
        f"{kind}_max_allocated_bytes": 30,
    }
    empty_accelerator_cache(torch_module, kind)
    assert backend.events == ["reset", "synchronize", "empty"]


def test_cpu_helpers_are_noops() -> None:
    torch_module = SimpleNamespace()

    require_available(torch_module, "cpu")
    reset_peak_memory_stats(torch_module, "cpu")
    synchronize(torch_module, "cpu")
    empty_accelerator_cache(torch_module, "cpu")
    assert memory_snapshot(torch_module, "cpu") == {}


def test_unavailable_accelerator_fails_closed() -> None:
    torch_module = SimpleNamespace(cuda=FakeBackend(available=False))

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        require_available(torch_module, "cuda")
