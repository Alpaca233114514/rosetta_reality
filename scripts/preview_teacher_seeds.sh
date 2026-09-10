#!/usr/bin/env bash
# Read-only seed preview for the geometric teacher candidate: replays the
# registered seeds through scripts/diagnose_geometric_teacher_g1.py without
# writing gate evidence.  Usage: scripts/preview_teacher_seeds.sh [seed...]
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
cd "${REPOSITORY_ROOT}"

seeds=("$@")
if [[ ${#seeds[@]} -eq 0 ]]; then
    seeds=(10 1900 1901 1902 1903 1904)
fi

for seed in "${seeds[@]}"; do
    echo "=== seed ${seed} ==="
    scripts/run_m2_container.sh vla-sim-xpu \
        python scripts/diagnose_geometric_teacher_g1.py --seed "${seed}" 2>&1 |
        grep -E 'done|final|unexpected contacts|REFUSAL' || true
done
