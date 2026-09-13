#!/usr/bin/env bash
set -Eeuo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$repo"
root=$(wslpath -m "$repo")
image=sha256:fb3c1bbda42881fac9b7725d3acb436119934f0c60af29747339d5951c3039da
mode=${1:?format or check}
test "$(docker.exe image inspect "$image" --format '{{.Id}}' | tr -d '\r')" = "$image"
files=(scripts/hestia_q_replay.py scripts/diagnose_hestia_q_replay.py scripts/run_hestia_q_replay.py scripts/stream_hestia_q_replay_results.py scripts/verify_hestia_q_replay_results.py scripts/analyze_hestia_q_replay.py scripts/verify_hestia_q_replay_analysis.py tests/test_hestia_q_replay.py)
mounts=()
if [[ "$mode" == format ]]; then
 for path in "${files[@]}"; do mounts+=(--mount "type=bind,source=$root/$path,target=/workspace/$path"); done
fi
docker.exe run --rm --network none --memory 2g --memory-swap 2g --cpus 2 --pids-limit 128 --read-only --tmpfs /tmp:rw,nosuid,size=128m \
 --mount "type=bind,source=$root,target=/workspace,readonly" \
 "${mounts[@]}" -w /workspace -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 \
 -e HF_HUB_OFFLINE=1 -e PYTHONPATH=/workspace/src:/workspace -e "MODE=$mode" "$image" timeout --kill-after=10 180 bash -lc '
 set -Eeuo pipefail
 files=(scripts/hestia_q_replay.py scripts/diagnose_hestia_q_replay.py scripts/run_hestia_q_replay.py scripts/stream_hestia_q_replay_results.py scripts/verify_hestia_q_replay_results.py scripts/analyze_hestia_q_replay.py scripts/verify_hestia_q_replay_analysis.py tests/test_hestia_q_replay.py)
 if [[ "$MODE" == format ]]; then
  python -m ruff format --no-cache "${files[@]}"
  python -m ruff check --fix --no-cache "${files[@]}"
 else
  python scripts/check_env.py
  python -m ruff check --no-cache "${files[@]}"
  python -m ruff format --check --no-cache "${files[@]}"
  python -m pytest -q -p no:cacheprovider tests/test_hestia_parameter_crossover.py tests/test_hestia_parameter_crossover_guard.py tests/test_hestia_local_checkpoint.py tests/test_hestia_parameter_crossover_cli.py tests/test_hestia_kv_split.py tests/test_hestia_kv_split_cli.py tests/test_hestia_v_layer.py tests/test_hestia_v_layer_cli.py tests/test_hestia_scene_kv.py tests/test_hestia_scene_kv_cli.py tests/test_hestia_parameter_crossover_transfer.py tests/test_hestia_q_replay.py
 fi'
