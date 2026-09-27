"""Check the actual cgroup envelope before reading saved research arrays."""

from __future__ import annotations

import os
from pathlib import Path


def verify_resource_envelope(plan, *, cgroup=Path("/sys/fs/cgroup"), environment=None):
    environment = os.environ if environment is None else environment
    limits = plan["resources"]
    memory = (cgroup / "memory.max").read_text().strip()
    quota, period = (cgroup / "cpu.max").read_text().split()
    if memory == "max" or quota == "max":
        raise ValueError("Bounded memory and CPU cgroups required")
    memory, quota, period = int(memory), int(quota), int(period)
    wall = int(environment.get("ROSETTA_ROOT_ENFORCED_WALL_SECONDS", "0"))
    if not 0 < memory <= limits["memory_mib"] * 1024**2:
        raise ValueError("Actual cgroup memory exceeds registered budget")
    if period <= 0 or not 0 < quota / period <= limits["cpu_threads"]:
        raise ValueError("Actual CPU quota exceeds registered budget")
    if not 0 < wall <= limits["wall_seconds"]:
        raise ValueError("Runner timeout absent or exceeds registered deadline")
    return {
        "memory_max_bytes": memory,
        "cpu_quota": quota / period,
        "command_timeout_seconds": wall,
        "verified_before_array_load": True,
    }
