"""Verify execution identities, separate from static/historical plan inspection."""

from __future__ import annotations

from pathlib import Path

from rosetta_reality.experiment import file_sha256
from rosetta_reality.vla.training.plan import is_sha256, repository_relative_path

CORE_IMPLEMENTATION = frozenset(
    {
        "scripts/run_smolvla_v2.py",
        "scripts/train_smolvla_v2.py",
        "src/rosetta_reality/vla/training/plan.py",
        "src/rosetta_reality/vla/training/launch.py",
        "src/rosetta_reality/vla/training/features.py",
        "src/rosetta_reality/vla/training/integrity.py",
    }
)


def validate_local_implementation(plan, repository_root):
    declared = plan.get("implementation_files")
    if not isinstance(declared, dict) or not declared:
        raise ValueError("Implementation inventory is missing")
    root = Path(repository_root).resolve()
    verified = {}
    for name, expected in declared.items():
        relative = repository_relative_path(name, context="Implementation file")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or not is_sha256(expected):
            raise ValueError(f"Implementation identity is invalid: {name}")
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"Implementation checksum changed: {name}")
        verified[name] = actual
    if not CORE_IMPLEMENTATION <= set(verified):
        raise ValueError("Implementation inventory must bind the complete v2 launch core")
    return verified
