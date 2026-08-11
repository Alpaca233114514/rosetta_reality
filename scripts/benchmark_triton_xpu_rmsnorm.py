"""Benchmark a bounded forward-only fused RMSNorm Triton kernel on XPU."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch
import triton
import triton.language as tl

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rosetta_reality.experiment import stable_hash, workspace_code_identity  # noqa: E402
from rosetta_reality.features import create_json  # noqa: E402


@triton.jit
def _rmsnorm_forward_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    columns: tl.constexpr,
    epsilon: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(axis=0)
    offsets = tl.arange(0, block_size)
    mask = offsets < columns
    values = tl.load(input_pointer + row * columns + offsets, mask=mask, other=0.0)
    values_fp32 = values.to(tl.float32)
    mean_square = tl.sum(values_fp32 * values_fp32, axis=0) / columns
    inverse_root = tl.rsqrt(mean_square + epsilon)
    weights = tl.load(weight_pointer + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        output_pointer + row * columns + offsets,
        values_fp32 * inverse_root * weights,
        mask=mask,
    )


def _benchmark(function: Callable[[], torch.Tensor], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    torch.xpu.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        function()
    torch.xpu.synchronize()
    return (time.perf_counter() - started) * 1000 / iterations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--columns", type=int, default=960)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    if min(args.rows, args.columns, args.warmup, args.iterations) <= 0:
        raise ValueError("RMSNorm benchmark dimensions and repetitions must be positive.")
    if not torch.xpu.is_available():
        raise RuntimeError("The fused RMSNorm benchmark requires an Intel XPU.")
    run_root = Path(os.environ["ROSETTA_RUN_ROOT"])
    if not os.environ.get("TRITON_CACHE_DIR"):
        triton_cache = run_root / "compiler_cache" / "xpu-rmsnorm-probe" / "triton"
        triton_cache.mkdir(parents=True, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = str(triton_cache)

    device = torch.device("xpu")
    torch.manual_seed(args.seed)
    torch.xpu.manual_seed_all(args.seed)
    inputs = torch.randn(args.rows, args.columns, device=device, dtype=torch.bfloat16)
    weights = torch.randn(args.columns, device=device, dtype=torch.bfloat16)
    output = torch.empty_like(inputs)
    epsilon = 1e-6
    block_size = triton.next_power_of_2(args.columns)

    def torch_forward() -> torch.Tensor:
        values = inputs.float()
        inverse_root = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + epsilon)
        return (values * inverse_root * weights.float()).to(inputs.dtype)

    def triton_forward() -> torch.Tensor:
        _rmsnorm_forward_kernel[(args.rows,)](
            inputs,
            weights,
            output,
            columns=args.columns,
            epsilon=epsilon,
            block_size=block_size,
        )
        return output

    reference = torch_forward()
    candidate = triton_forward()
    torch.xpu.synchronize()
    difference = (reference.float() - candidate.float()).abs()
    maximum_absolute = float(difference.max().cpu())
    mean_absolute = float(difference.mean().cpu())
    if not math.isfinite(maximum_absolute) or maximum_absolute > 0.02:
        raise RuntimeError("The fused RMSNorm forward kernel failed numerical parity.")
    torch_ms = _benchmark(torch_forward, args.warmup, args.iterations)
    triton_ms = _benchmark(triton_forward, args.warmup, args.iterations)
    report = {
        "schema_version": 1,
        "status": "complete",
        "stage": "xpu_triton_fused_rmsnorm_forward_microbenchmark",
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "device": torch.xpu.get_device_name(0),
        "dtype": "bfloat16",
        "rows": args.rows,
        "columns": args.columns,
        "block_size": block_size,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "seed": args.seed,
        "maximum_absolute_difference": maximum_absolute,
        "mean_absolute_difference": mean_absolute,
        "torch_eager_milliseconds": torch_ms,
        "triton_fused_milliseconds": triton_ms,
        "microbenchmark_speedup": torch_ms / triton_ms,
        "forward_only": True,
        "training_integration_allowed": False,
        "integration_gate": "profile_hotspot_then_backward_parity_and_end_to_end_speedup",
        "workspace": workspace_code_identity(REPOSITORY_ROOT),
    }
    identity = stable_hash(
        {
            key: report[key]
            for key in (
                "torch_version",
                "triton_version",
                "device",
                "dtype",
                "rows",
                "columns",
                "seed",
                "workspace",
            )
        }
    )
    destination = run_root / "hardware" / f"triton-rmsnorm-{identity[:16]}.json"
    create_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    print(f"Triton RMSNorm report: {destination.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
