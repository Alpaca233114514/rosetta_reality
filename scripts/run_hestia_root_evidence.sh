#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
root=$(wslpath -m "$PWD")
image=sha256:fb3c1bbda42881fac9b7725d3acb436119934f0c60af29747339d5951c3039da
mode=${1:?format check or run}
test "$(docker.exe image inspect "$image" --format '{{.Id}}' | tr -d '\r')" = "$image"
mounts=()
if [[ $mode == format ]]; then
 for path in scripts/analyze_hestia_root_evidence.py tests/test_hestia_root_evidence.py; do mounts+=(--mount "type=bind,source=$root/$path,target=/workspace/$path"); done
fi
docker.exe run --rm --network none --memory 2g --memory-swap 2g --cpus 2 --pids-limit 128 --read-only --tmpfs /tmp:rw,nosuid,size=128m \
 --mount "type=bind,source=$root,target=/workspace,readonly" \
 --mount "type=bind,source=$root/runs/hestia-root-evidence-20260913-002,target=/workspace/runs/hestia-root-evidence-20260913-002" \
 "${mounts[@]}" -w /workspace -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e HF_HUB_OFFLINE=1 -e PYTHONPATH=/workspace/src:/workspace -e "MODE=$mode" -e "ROSETTA_CONTAINER_IMAGE_ID=$image" "$image" timeout --kill-after=10 300 bash -lc '
 set -Eeuo pipefail
 paths="scripts/analyze_hestia_root_evidence.py tests/test_hestia_root_evidence.py"
 if [[ $MODE == format ]]; then
  python -m ruff format --no-cache $paths
 elif [[ $MODE == check ]]; then
  python scripts/check_env.py
  python -m ruff check --no-cache $paths
  python -m ruff format --check --no-cache $paths
  python -m pytest -q -p no:cacheprovider tests/test_hestia_root_evidence.py
 elif [[ $MODE == run ]]; then
  python scripts/analyze_hestia_root_evidence.py
 else exit 2
 fi'
