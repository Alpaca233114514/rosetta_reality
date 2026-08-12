"""Run a bounded Rosetta forward/backward smoke on an Intel XPU device."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rosetta_reality.experiment import stable_hash, workspace_code_identity  # noqa: E402
from rosetta_reality.features import create_json  # noqa: E402
from rosetta_reality.models import ContinuousActionHead, StateEncoder, VLAPolicy  # noqa: E402
from rosetta_reality.models.backbones import DummyBackbone  # noqa: E402
from rosetta_reality.train.losses import smooth_l1_action_loss  # noqa: E402


def _run_root() -> Path:
    configured = os.environ.get("ROSETTA_RUN_ROOT")
    return Path(configured) if configured else REPOSITORY_ROOT / "runs"


def _device_report(device_index: int) -> dict[str, Any]:
    properties = torch.xpu.get_device_properties(device_index)
    return {
        "index": device_index,
        "name": torch.xpu.get_device_name(device_index),
        "total_memory_bytes": int(getattr(properties, "total_memory", 0)),
        "architecture": str(getattr(properties, "architecture", "unknown")),
    }


def run_smoke(*, steps: int, seed: int) -> Path:
    """Exercise the downstream M2 tensor contract on XPU and persist a report."""

    if steps <= 0:
        raise ValueError("steps must be positive.")
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("Intel XPU is not available in this isolated runtime.")

    device_index = 0
    device = torch.device(f"xpu:{device_index}")
    torch.manual_seed(seed)
    torch.xpu.manual_seed_all(seed)
    feature_dim = 5120
    state_dim = 14
    hidden_dim = 256
    action_dim = 14
    chunk_size = 100
    batch_size = 64

    policy = VLAPolicy(
        backbone=DummyBackbone(input_dim=feature_dim, hidden_size=hidden_dim),
        state_encoder=StateEncoder(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            dropout=0.0,
        ),
        action_head=ContinuousActionHead(
            input_dim=hidden_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dim=hidden_dim,
        ),
    ).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-4, foreach=False)
    observations = {
        "features": torch.randn(batch_size, feature_dim, device=device),
    }
    robot_state = torch.randn(batch_size, state_dim, device=device)
    target = torch.randn(batch_size, chunk_size, action_dim, device=device)

    with torch.no_grad():
        initial_prediction = policy(observations, robot_state)
        initial_loss = smooth_l1_action_loss(initial_prediction, target)
    if initial_prediction.shape != (batch_size, chunk_size, action_dim):
        raise RuntimeError("XPU smoke prediction violated the M2 tensor contract.")
    if not bool(torch.isfinite(initial_prediction).all() and torch.isfinite(initial_loss)):
        raise FloatingPointError("XPU smoke initial output is non-finite.")

    torch.xpu.synchronize(device)
    started = time.perf_counter()
    gradient_l2_squared: dict[str, float] = {}
    step_losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = policy(observations, robot_state)
        loss = smooth_l1_action_loss(prediction, target)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("XPU smoke loss became non-finite.")
        loss.backward()
        gradient_l2_squared = {}
        for name, parameter in policy.named_parameters():
            if parameter.grad is None:
                continue
            if not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError(f"XPU smoke gradient is non-finite: {name}.")
            component = name.split(".", maxsplit=1)[0]
            gradient_l2_squared[component] = gradient_l2_squared.get(component, 0.0) + float(
                parameter.grad.detach().square().sum().cpu()
            )
        clip_grad_norm_(policy.parameters(), 1.0, foreach=False)
        optimizer.step()
        step_losses.append(float(loss.detach().cpu()))
    torch.xpu.synchronize(device)
    elapsed = time.perf_counter() - started

    required_components = {"backbone", "state_encoder", "fusion", "action_head"}
    if any(gradient_l2_squared.get(name, 0.0) <= 0.0 for name in required_components):
        raise RuntimeError("XPU smoke did not propagate gradients through every component.")
    with torch.no_grad():
        final_loss = float(
            smooth_l1_action_loss(policy(observations, robot_state), target).cpu()
        )
    initial_loss_value = float(initial_loss.cpu())
    if not final_loss < initial_loss_value:
        raise RuntimeError("XPU smoke optimizer steps did not reduce the fixed-batch loss.")

    report = {
        "schema_version": 1,
        "status": "passed",
        "gate": "project_xpu_forward_backward_smoke",
        "workspace": workspace_code_identity(REPOSITORY_ROOT),
        "torch_version": torch.__version__,
        "device": _device_report(device_index),
        "seed": seed,
        "steps": steps,
        "tensor_contract": {
            "batch_size": batch_size,
            "feature_dim": feature_dim,
            "state_dim": state_dim,
            "chunk_size": chunk_size,
            "action_dim": action_dim,
        },
        "prediction_shape": list(initial_prediction.shape),
        "initial_loss": initial_loss_value,
        "step_losses": step_losses,
        "final_loss": final_loss,
        "gradient_l2_squared": gradient_l2_squared,
        "elapsed_seconds": elapsed,
        "steps_per_second": steps / elapsed,
    }
    identity = stable_hash(
        {
            key: report[key]
            for key in ("workspace", "torch_version", "device", "seed", "steps", "tensor_contract")
        }
    )
    destination = _run_root() / "hardware" / f"xpu-smoke-{identity[:16]}.json"
    create_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"XPU smoke report: {destination.name}")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args()
    run_smoke(steps=args.steps, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
