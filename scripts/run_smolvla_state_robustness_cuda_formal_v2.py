"""Validate CUDA compile safety before delegating a registered Way formal plan."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import run_smolvla_state_robustness_cuda_formal as formal_runner  # noqa: E402

from rosetta_reality.vla.runtime_compatibility import (  # noqa: E402
    validate_cuda_compile_contract,
)


def _plan_path() -> Path:
    try:
        return Path(sys.argv[sys.argv.index("--plan") + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise ValueError("Way CUDA formal v2 requires an explicit --plan path.") from error


def _cuda_graph_smoke_accepted(plan: dict[str, Any]) -> bool:
    smoke = plan.get("cuda_smoke")
    if not isinstance(smoke, dict):
        raise ValueError("The Way formal plan has no CUDA smoke identity.")
    return smoke.get("cuda_graph_capture_accepted") is True


def main() -> int:
    plan_path = _plan_path()
    plan = formal_runner._load_yaml(plan_path)
    training = plan.get("training")
    if not isinstance(training, dict) or not isinstance(training.get("policy"), dict):
        raise ValueError("The Way formal plan has no policy runtime contract.")
    validate_cuda_compile_contract(
        training["policy"],
        cuda_graph_smoke_accepted=_cuda_graph_smoke_accepted(plan),
    )
    return formal_runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
