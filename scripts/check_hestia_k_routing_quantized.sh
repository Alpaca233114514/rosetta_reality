#!/usr/bin/env bash
set -Eeuo pipefail
source_root=${1:?source repository required}
mode=${2:?format or check}
cd "$(dirname "${BASH_SOURCE[0]}")/.."
root=$(wslpath -m "$PWD")
source_win=$(wslpath -m "$source_root")
image=sha256:fb3c1bbda42881fac9b7725d3acb436119934f0c60af29747339d5951c3039da
test "$(docker.exe image inspect "$image" --format '{{.Id}}' | tr -d '\r')" = "$image"
files=(scripts/hestia_k_routing_quantized.py scripts/diagnose_hestia_k_routing_quantized.py
 scripts/run_hestia_k_routing_quantized.py scripts/stream_hestia_k_routing_quantized_results.py
 scripts/verify_hestia_k_routing_quantized_results.py tests/test_hestia_k_routing_quantized.py
 tests/test_hestia_k_routing_quantized_cli.py)
mounts=()
for path in "${files[@]}"; do
 suffix=,readonly
 if [[ $mode == format ]]; then suffix=; fi
 mounts+=(--mount "type=bind,source=$root/$path,target=/new/$path$suffix")
done
docker.exe run --rm --network none --memory 2g --memory-swap 2g --cpus 2 --pids-limit 128 \
 --read-only --tmpfs /tmp:rw,nosuid,size=256m --tmpfs /workspace:rw,nosuid,size=256m \
 --mount "type=bind,source=$source_win,target=/source,readonly" \
 --mount "type=bind,source=$root,target=/new,readonly" "${mounts[@]}" \
 -w /new -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 \
 -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 -e PYTHONPATH=/workspace/src:/workspace \
 -e "ROSETTA_CONTAINER_IMAGE_ID=$image" -e "MODE=$mode" "$image" \
 timeout --kill-after=10 300 bash -lc '
set -Eeuo pipefail
files=(scripts/hestia_k_routing_quantized.py scripts/diagnose_hestia_k_routing_quantized.py
 scripts/run_hestia_k_routing_quantized.py scripts/stream_hestia_k_routing_quantized_results.py
 scripts/verify_hestia_k_routing_quantized_results.py tests/test_hestia_k_routing_quantized.py
 tests/test_hestia_k_routing_quantized_cli.py)
if [[ $MODE == format ]]; then
 python -m ruff format --no-cache "${files[@]}"
 python -m ruff check --fix --no-cache "${files[@]}"
elif [[ $MODE == check ]]; then
 python /source/scripts/check_env.py
 python -m ruff check --no-cache "${files[@]}"
 python -m ruff format --check --no-cache "${files[@]}"
 cp -a /source/scripts /source/tests /workspace/
 cp /new/scripts/hestia_k_direction_head.py /workspace/scripts/hestia_k_direction_head.py
 cp /new/scripts/hestia_k_direction_layer.py /workspace/scripts/hestia_k_direction_layer.py
 cp /new/scripts/hestia_k_factor.py /workspace/scripts/hestia_k_factor.py
 cp /new/scripts/hestia_scene_kv_mean.py /workspace/scripts/hestia_scene_kv_mean.py
 cp /source/pyproject.toml /workspace/pyproject.toml
 ln -s /source/src /workspace/src
 ln -s /source/configs /workspace/configs
 for path in "${files[@]}"; do
  test ! -e "/workspace/$path"
  cp "/new/$path" "/workspace/$path"
 done
 cd /workspace
 python -m pytest -q -p no:cacheprovider tests/test_hestia_k_routing_quantized.py tests/test_hestia_k_routing_quantized_cli.py
 python -m pytest -q -p no:cacheprovider tests/test_hestia_parameter_crossover.py \
  tests/test_hestia_parameter_crossover_guard.py tests/test_hestia_local_checkpoint.py \
  tests/test_hestia_parameter_crossover_cli.py tests/test_hestia_kv_split.py \
  tests/test_hestia_kv_split_cli.py tests/test_hestia_parameter_crossover_transfer.py
else exit 2
fi'
