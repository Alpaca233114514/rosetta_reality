"""Training entry point with an offline CPU dry-run for M0."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run one synthetic CPU optimizer step with DummyBackbone.",
    )
    return parser


def run_dry_run() -> None:
    """Run a deterministic forward/loss/backward/update cycle on CPU."""

    try:
        import torch
    except ImportError:
        raise SystemExit(
            "Dry-run requires PyTorch. Install the project dependencies in an isolated "
            "environment, then retry; no packages were installed automatically."
        ) from None

    from rosetta_reality.models import ContinuousActionHead, StateEncoder, VLAPolicy
    from rosetta_reality.models.backbones import DummyBackbone
    from rosetta_reality.train import train_step

    torch.manual_seed(7)
    device = torch.device("cpu")
    batch_size = 2
    feature_dim = 16
    state_dim = 9
    hidden_dim = 32
    action_dim = 7
    chunk_size = 8

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
        ),
    ).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    observations = {"features": torch.randn(batch_size, feature_dim, device=device)}
    robot_state = torch.randn(batch_size, state_dim, device=device)
    target_actions = torch.randn(batch_size, chunk_size, action_dim, device=device)

    result = train_step(policy, optimizer, observations, robot_state, target_actions)
    print("Dry-run success")
    print(f"Device: {device}")
    print(f"Prediction shape: {tuple(result.prediction_shape)}")
    print(f"Smooth L1 loss: {result.loss:.6f}")


def main() -> int:
    """Dispatch the safe M0 training mode."""

    parser = build_parser()
    args = parser.parse_args()
    if not args.dry_run:
        parser.error("M0 only supports --dry-run; real training is not implemented.")
    run_dry_run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
