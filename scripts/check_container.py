"""Verify Docker resource, mount, privilege, and network boundaries read-only."""

from __future__ import annotations

import json
import os
from pathlib import Path


def _cgroup(name: str) -> str:
    return (Path("/sys/fs/cgroup") / name).read_text(encoding="utf-8").strip()


def _read_only(path: str) -> bool:
    return bool(os.statvfs(path).f_flag & os.ST_RDONLY)


def main() -> int:
    memory_max = _cgroup("memory.max")
    memory_swap_max = _cgroup("memory.swap.max")
    cpu_quota, cpu_period = _cgroup("cpu.max").split()
    pids_max = _cgroup("pids.max")
    interfaces = sorted(path.name for path in Path("/sys/class/net").iterdir())
    writable_mounts = ("/workspace/data", "/workspace/feature_cache", "/workspace/runs")
    report = {
        "memory_max": memory_max,
        "memory_swap_max": memory_swap_max,
        "cpu_max": [cpu_quota, cpu_period],
        "pids_max": pids_max,
        "root_read_only": _read_only("/"),
        "workspace_read_only": _read_only("/workspace"),
        "declared_output_mounts_writable": {
            path: not _read_only(path) for path in writable_mounts
        },
        "network_interfaces": interfaces,
        "no_new_privileges": "NoNewPrivs:\t1"
        in Path("/proc/self/status").read_text(encoding="utf-8"),
    }
    expected_memory = 5 * 1024**3
    passed = (
        memory_max == str(expected_memory)
        and memory_swap_max == "0"
        and cpu_quota != "max"
        and int(cpu_quota) / int(cpu_period) <= 2.0
        and pids_max == "512"
        and report["root_read_only"]
        and report["workspace_read_only"]
        and all(report["declared_output_mounts_writable"].values())
        and interfaces == ["lo"]
        and report["no_new_privileges"]
    )
    report["status"] = "passed" if passed else "failed"
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
