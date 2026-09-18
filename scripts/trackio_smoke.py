"""Create a public-safe local Trackio run before any model optimization."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any

import trackio
import yaml

from rosetta_reality.tracking import validate_public_payload

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs/vla/smolvla_450m_aloha_insertion.yaml"


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("VLA experiment config must be a mapping.")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    experiment = _load(args.config.resolve())
    tracking = experiment["tracking"]
    run_name = f"space-smoke-{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%SZ}"
    public_config = {
        "experiment_id": experiment["experiment_id"],
        "role": "vla",
        "phase": "space_smoke",
        "model_id": experiment["model"]["identifier"],
        "model_revision": experiment["model"]["revision"],
        "dataset_id": experiment["dataset"]["identifier"],
        "dataset_revision": experiment["dataset"]["revision"],
        "synthetic_metrics_only": True,
        "test_split_loaded": False,
    }
    validate_public_payload(public_config, context="trackio_smoke")
    trackio.init(
        project=tracking["project"],
        name=run_name,
        group=f"{experiment['experiment_id']}-space-smoke",
        config=public_config,
        embed=False,
        auto_log_cpu=False,
        auto_log_gpu=False,
    )
    try:
        trackio.log({"system/space_smoke": 1, "system/test_split_loaded": 0}, step=0)
        trackio.log({"system/space_smoke": 1, "system/sanitizer_passed": 1}, step=1)
    finally:
        trackio.finish()
    print(f"Trackio local smoke complete: {run_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
