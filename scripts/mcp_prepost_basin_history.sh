#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
readonly BASIN_ROOT="${ROSETTA_BASIN_ROOT:-${REPOSITORY_ROOT}/../basin}"
readonly HISTORY="${REPOSITORY_ROOT}/runs/prepost-basin-received-20260924-001/prepost-basin-history-20260924-001/history"

[[ -f "${BASIN_ROOT}/basin/__main__.py" ]] || {
    printf 'Basin source is missing\n' >&2
    exit 2
}
[[ -f "${HISTORY}/prepost-gate4-20260924-001-trained-gate4/manifest.json" ]] || {
    printf 'Pre/post Basin history is missing\n' >&2
    exit 2
}

export PYTHONPATH="${BASIN_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec /usr/bin/python3 -m basin --store "${HISTORY}" mcp
