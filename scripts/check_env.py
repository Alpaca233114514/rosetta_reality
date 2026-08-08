"""Print environment capabilities without modifying the environment."""

from __future__ import annotations

import importlib.util
import platform
import sys


def module_available(name: str) -> bool:
    """Return whether a module can be imported without importing it."""

    return importlib.util.find_spec(name) is not None


def main() -> int:
    """Print Python, OS, PyTorch, accelerator, and optional-package details."""

    print(f"Python version: {platform.python_version()}")
    print(f"Python executable: {sys.executable}")
    print(f"OS: {platform.platform()}")

    torch_available = module_available("torch")
    print(f"PyTorch available: {torch_available}")
    if torch_available:
        try:
            import torch
        except Exception as exc:  # pragma: no cover - broken installation path
            print(f"PyTorch import error: {type(exc).__name__}: {exc}")
            print("PyTorch version: unavailable")
            print("CUDA available: unavailable")
            print("CUDA device count: unavailable")
            print("ROCm/HIP version: unavailable")
        else:
            print(f"PyTorch version: {torch.__version__}")
            cuda_available = torch.cuda.is_available()
            print(f"CUDA available: {cuda_available}")
            device_count = torch.cuda.device_count()
            print(f"CUDA device count: {device_count}")
            for device_index in range(device_count):
                try:
                    device_name = torch.cuda.get_device_name(device_index)
                except Exception as exc:  # pragma: no cover - hardware-specific defensive path
                    device_name = f"unavailable ({type(exc).__name__})"
                print(f"CUDA device {device_index}: {device_name}")
            rocm_version = getattr(torch.version, "hip", None)
            print(f"ROCm/HIP version: {rocm_version or 'not reported'}")
    else:
        print("PyTorch version: unavailable")
        print("CUDA available: unavailable")
        print("CUDA device count: unavailable")
        print("ROCm/HIP version: unavailable")

    print(f"Transformers available: {module_available('transformers')}")
    print(f"PEFT available: {module_available('peft')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
