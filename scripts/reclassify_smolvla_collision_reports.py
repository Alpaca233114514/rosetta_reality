"""Reclassify immutable SmolVLA contact histograms with the current adapter rule."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rosetta_reality.experiment import file_sha256  # noqa: E402
from rosetta_reality.features import create_json  # noqa: E402
from rosetta_reality.sim import GymAlohaEnvironment, load_action_contract  # noqa: E402


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()


def _parse_pair(value: str) -> tuple[str, str]:
    parts = value.split(" <-> ")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Invalid serialized MuJoCo contact pair: {value!r}.")
    return parts[0], parts[1]


def _reclassify_report(
    path: Path, environment: GymAlohaEnvironment
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"Gate episode report lacks metrics: {_relative(path)}.")
    raw_pairs = metrics.get("unexpected_collision_pairs")
    if not isinstance(raw_pairs, dict):
        raise ValueError(f"Gate episode report lacks contact histogram: {_relative(path)}.")
    pairs: list[dict[str, Any]] = []
    reclassified_total = 0
    histogram_total = 0
    for serialized, raw_count in sorted(raw_pairs.items()):
        if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count < 0:
            raise ValueError(f"Invalid contact count for {serialized!r}.")
        first, second = _parse_pair(str(serialized))
        unexpected = environment.is_unexpected_collision_pair(first, second)
        histogram_total += raw_count
        if unexpected:
            reclassified_total += raw_count
        pairs.append(
            {
                "pair": [first, second],
                "count": raw_count,
                "reclassified_unexpected": unexpected,
            }
        )
    recorded_total = metrics.get("unexpected_collisions")
    if recorded_total != histogram_total:
        raise ValueError(f"Contact histogram total differs from {_relative(path)}.")
    return {
        "path": _relative(path),
        "sha256": file_sha256(path),
        "seed": payload.get("seed"),
        "recorded_unexpected_contacts": recorded_total,
        "reclassified_unexpected_contacts": reclassified_total,
        "pairs": pairs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    environment = GymAlohaEnvironment(
        load_action_contract(contract_path), environment=object()
    )
    reports = [
        _reclassify_report(path.resolve(), environment) for path in args.input
    ]
    adapter_path = REPOSITORY_ROOT / "src/rosetta_reality/sim/gym_aloha.py"
    result = {
        "schema_version": 1,
        "status": "complete",
        "stage": "smolvla_collision_metric_reclassification",
        "action_contract": _relative(contract_path),
        "action_contract_sha256": file_sha256(contract_path),
        "adapter": _relative(adapter_path),
        "adapter_sha256": file_sha256(adapter_path),
        "source_reports_mutated": False,
        "fresh_gate_required_for_acceptance": True,
        "reports": reports,
        "aggregate": {
            "recorded_unexpected_contacts": sum(
                int(report["recorded_unexpected_contacts"]) for report in reports
            ),
            "reclassified_unexpected_contacts": sum(
                int(report["reclassified_unexpected_contacts"]) for report in reports
            ),
        },
    }
    destination = args.output.resolve()
    create_json(destination, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Reclassification: {_relative(destination)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
