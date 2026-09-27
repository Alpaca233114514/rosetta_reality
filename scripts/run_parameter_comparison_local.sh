#!/usr/bin/env bash
# WSL entry, offline pinned CPU container, immutable inputs and create-only output.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
output=$(realpath -m -- "${1:?new output under .cache required}")
runs_host=${2:?resolved Windows runs directory required}
shift 2
[[ "$output" == "$repo/.cache/"* && ! -e "$output" && $# -gt 0 ]] || exit 2
image='sha256:fb3c1bbda42881fac9b7725d3acb436119934f0c60af29747339d5951c3039da'
docker=$(command -v docker.exe)
"$docker" image inspect "$image" >/dev/null
mkdir -p -- "$output"
set +e
"$docker" run --rm --pull never --network none --read-only --cpus 2 \
  --memory 4g --memory-swap 4g --pids-limit 256 --cap-drop ALL \
  --security-opt no-new-privileges --tmpfs /tmp:rw,size=256m \
  --mount "type=bind,source=$(wslpath -w "$repo"),target=/source,readonly" \
  --mount "type=bind,source=$runs_host,target=/source/runs,readonly" \
  --mount "type=bind,source=$(wslpath -w "$output"),target=/output" \
  -e PYTHONDONTWRITEBYTECODE=1 -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
  -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -e PYTHONPATH=/source:/source/src \
  -e "ROSETTA_CONTAINER_IMAGE_ID=$image" -e ROSETTA_ROOT_ENFORCED_WALL_SECONDS=600 \
  --workdir /source --entrypoint timeout "$image" --signal=TERM --kill-after=5s 600 \
  "$@" 2>&1 | tee "$output/container.log"
status=${PIPESTATUS[0]}
set -e
printf '{"exit_code":%s,"image":"%s","scope":"saved_weight_algebra_only"}\n' \
  "$status" "$image" > "$output/container-exit.json"
exit "$status"
