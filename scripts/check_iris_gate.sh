#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
repo=$(wslpath -m "$PWD")
mode=${1:-check}
for file in scripts/launch_iris_gate_worker.sh scripts/launch_iris_gate_from_wsl.sh scripts/receive_iris_gate_from_wsl.sh; do
 bash -n "$file"
done
mounts=()
if [[ "$mode" == format ]]; then
 for file in scripts/iris_gate.py scripts/run_iris_gate.py tests/test_iris_gate.py; do
  mounts+=(--mount "type=bind,source=$repo/$file,target=/workspace/$file")
 done
elif [[ "$mode" == prepare ]]; then
 mkdir -p runs/iris-gate-preparation-20260913-001
 mounts+=(--mount "type=bind,source=$repo/runs/iris-gate-preparation-20260913-001,target=/workspace/runs/iris-gate-preparation-20260913-001")
fi
docker.exe run --rm --network none --memory 2g --memory-swap 2g --cpus 2 --pids-limit 128 \
 --read-only --tmpfs /tmp:rw,nosuid,size=128m \
 --mount "type=bind,source=$repo,target=/workspace,readonly" "${mounts[@]}" \
 -w /workspace -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/workspace/src:/workspace \
 -e OMP_NUM_THREADS=2 -e "MODE=$mode" \
 sha256:fb3c1bbda42881fac9b7725d3acb436119934f0c60af29747339d5951c3039da \
 timeout 180 bash -lc '
set -Eeuo pipefail
files=(scripts/iris_gate.py scripts/run_iris_gate.py tests/test_iris_gate.py)
if [[ "$MODE" == format ]]; then
 python -m ruff format --no-cache "${files[@]}"
 python -m ruff check --fix --no-cache "${files[@]}"
elif [[ "$MODE" == prepare ]]; then
 python scripts/run_iris_gate.py prepare --template runs/iris-gate-preparation-20260913-001/template.json
else
 python scripts/check_env.py
 python -m pytest -q -p no:cacheprovider tests/test_iris_gate.py tests/test_sim_gate_protocol.py tests/test_iris_runtime.py
 python -m ruff check --no-cache "${files[@]}"
 python -m ruff format --check --no-cache "${files[@]}"
fi
'
