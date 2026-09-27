#!/usr/bin/env bash
# Synthetic-only tests, existing digest, read-only source, no network or downloads.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
output=$(realpath -m -- "${1:?New output directory under .cache required}")
shift
[[ "$output" == "$repo/.cache/"* && ! -e "$output" ]] || exit 2
image=sha256:fb3c1bbda42881fac9b7725d3acb436119934f0c60af29747339d5951c3039da
docker=$(command -v docker.exe)
"$docker" image inspect "$image" >/dev/null
mkdir -p -- "$output"
repo_host=$(wslpath -w "$repo")
output_host=$(wslpath -w "$output")
basin=$(realpath -- "${DUAL_AXIS_BASIN_ROOT:-$repo/../basin}")
basin_host=$(wslpath -w "$basin")
if (( $# == 0 )); then
  set -- python -m pytest -q -p no:cacheprovider tests/test_dual_axis.py tests/test_dual_axis_runtime.py
fi
set +e
"$docker" run --rm --pull never --network none --read-only --cpus 2 \
  --memory 4g --memory-swap 4g --pids-limit 256 --cap-drop ALL \
  --security-opt no-new-privileges --tmpfs /tmp:rw,size=1g \
  --mount "type=bind,source=$repo_host,target=/source,readonly" \
  --mount "type=bind,source=$basin_host,target=/basin,readonly" \
  --mount "type=bind,source=$output_host,target=/output" \
  -e PYTHONDONTWRITEBYTECODE=1 -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
  -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -e RUFF_CACHE_DIR=/tmp/ruff -e PYTHONPATH=/source/src:/source:/source/scripts:/source/tests:/basin \
  --workdir /source --entrypoint bash "$image" -c \
  'exec timeout --signal=TERM --kill-after=5s 300 "$@"' bash "$@" 2>&1 | tee "$output/container.log"
status=${PIPESTATUS[0]}
set -e
printf '{"container_exit_code":%s,"scope":"synthetic-only","image":"%s"}\n' \
  "$status" "$image" > "$output/container-exit.json"
exit "$status"
