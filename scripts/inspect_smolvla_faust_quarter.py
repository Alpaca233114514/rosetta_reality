"""Inspect only one registered Faust quarter checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import inspect_smolvla_quarter as inspector  # noqa: E402
import run_smolvla_action_repair_formal as formal_runner  # noqa: E402


def main() -> int:
    inspector.formal_runner = formal_runner
    return inspector.main()


if __name__ == "__main__":
    raise SystemExit(main())
