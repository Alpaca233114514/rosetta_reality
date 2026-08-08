"""Training entry point with a configuration-driven offline CPU dry-run."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "train" / "smoke.yaml"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


@dataclass(frozen=True, slots=True)
class DryRunConfig:
    """Validated settings for the offline CPU smoke step."""

    seed: int
    device: str
    steps: int
    batch_size: int
    learning_rate: float
    dummy_input_dim: int
    backbone_hidden_size: int
    state_dim: int
    state_hidden_dim: int
    state_num_layers: int
    state_dropout: float
    action_dim: int
    chunk_size: int


def _required(mapping: dict[str, Any], key: str, context: str) -> Any:
    try:
        return mapping[key]
    except KeyError as error:
        raise ValueError(f"{context} is missing '{key}'.") from error


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping.")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer.") from error


def _positive_integer(value: Any, name: str) -> int:
    parsed = _integer(value, name)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive.")
    return parsed


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive number.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive number.") from error
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive number.")
    return parsed


def load_dry_run_config(path: Path) -> DryRunConfig:
    """Load the checked-in smoke configuration without importing PyTorch."""

    raw = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "Dry-run configuration")
    model = _mapping(_required(raw, "model", "Dry-run configuration"), "model")
    device = str(_required(raw, "device", "Dry-run configuration"))
    if device != "cpu":
        raise ValueError("Dry-run configuration must use the CPU device.")
    dropout = float(_required(model, "state_dropout", "model"))
    if not 0.0 <= dropout < 1.0:
        raise ValueError("model.state_dropout must be in the range [0, 1).")
    return DryRunConfig(
        seed=_integer(_required(raw, "seed", "Dry-run configuration"), "seed"),
        device=device,
        steps=_positive_integer(
            _required(raw, "steps", "Dry-run configuration"), "steps"
        ),
        batch_size=_positive_integer(
            _required(raw, "batch_size", "Dry-run configuration"), "batch_size"
        ),
        learning_rate=_positive_float(
            _required(raw, "learning_rate", "Dry-run configuration"), "learning_rate"
        ),
        dummy_input_dim=_positive_integer(
            _required(model, "dummy_input_dim", "model"), "model.dummy_input_dim"
        ),
        backbone_hidden_size=_positive_integer(
            _required(model, "backbone_hidden_size", "model"),
            "model.backbone_hidden_size",
        ),
        state_dim=_positive_integer(
            _required(model, "state_dim", "model"), "model.state_dim"
        ),
        state_hidden_dim=_positive_integer(
            _required(model, "state_hidden_dim", "model"), "model.state_hidden_dim"
        ),
        state_num_layers=_positive_integer(
            _required(model, "state_num_layers", "model"), "model.state_num_layers"
        ),
        state_dropout=dropout,
        action_dim=_positive_integer(
            _required(model, "action_dim", "model"), "model.action_dim"
        ),
        chunk_size=_positive_integer(
            _required(model, "chunk_size", "model"), "model.chunk_size"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run configured synthetic CPU optimizer steps with DummyBackbone.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the CPU smoke configuration.",
    )
    return parser


def run_dry_run(config_path: Path = DEFAULT_CONFIG) -> None:
    """Run deterministic configured forward/loss/backward/update cycles on CPU."""

    config = load_dry_run_config(config_path)

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

    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    policy = VLAPolicy(
        backbone=DummyBackbone(
            input_dim=config.dummy_input_dim,
            hidden_size=config.backbone_hidden_size,
        ),
        state_encoder=StateEncoder(
            state_dim=config.state_dim,
            hidden_dim=config.state_hidden_dim,
            num_layers=config.state_num_layers,
            dropout=config.state_dropout,
        ),
        action_head=ContinuousActionHead(
            input_dim=config.state_hidden_dim,
            action_dim=config.action_dim,
            chunk_size=config.chunk_size,
        ),
    ).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.learning_rate)
    observations = {
        "features": torch.randn(
            config.batch_size,
            config.dummy_input_dim,
            device=device,
        )
    }
    robot_state = torch.randn(config.batch_size, config.state_dim, device=device)
    target_actions = torch.randn(
        config.batch_size,
        config.chunk_size,
        config.action_dim,
        device=device,
    )

    for _ in range(config.steps):
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
    run_dry_run(args.config.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
