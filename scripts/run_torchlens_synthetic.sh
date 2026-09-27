#!/usr/bin/env bash
# WSL entry: no image pulls, no network, no live weights/data, no host ML execution.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
output=${1:?Pass a new output directory under the repository .cache directory}
shift
if (( $# == 0 )); then
  set -- scripts/collect_torchlens_synthetic.py --output /output/capture
fi
output=$(realpath -m -- "$output")
[[ "$output" == "$repo/.cache/"* && ! -e "$output" ]] || exit 2
image='sha256:fb3c1bbda42881fac9b7725d3acb436119934f0c60af29747339d5951c3039da'
docker=$(command -v docker.exe)
"$docker" image inspect "$image" >/dev/null
mkdir -p -- "$output"
repo_host=$(wslpath -w "$repo")
output_host=$(wslpath -w "$output")
basin=$(realpath "$repo/../basin")
basin_host=$(wslpath -w "$basin")
set +e
"$docker" run --rm --pull never --network none --read-only --cpus 2 \
  --memory 4g --memory-swap 4g --pids-limit 256 --cap-drop ALL \
  --security-opt no-new-privileges --tmpfs /tmp:rw,size=512m \
  --mount "type=bind,source=$repo_host,target=/source,readonly" \
  --mount "type=bind,source=$basin_host,target=/basin,readonly" \
  --mount "type=bind,source=$output_host,target=/output" \
  -e PYTHONDONTWRITEBYTECODE=1 -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
  -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -e BASIN_SOURCE=/basin \
  --workdir /source --entrypoint bash "$image" -c '
    set -euo pipefail
    python -m pip install --no-index --no-deps --require-hashes --target /tmp/overlay \
      --find-links /source/.work/torchlens/wheels \
      -r /source/requirements/torchlens-diagnostics.txt
    export PYTHONPATH=/tmp/overlay:/source/src:/source:/basin
    timeout --signal=TERM --kill-after=5s 180 python "$@"
  ' bash "$@" 2>&1 | tee "$output/container.log"
status=${PIPESTATUS[0]}
set -e
printf '{"container_exit_code":%s,"scope":"synthetic-only","image":"%s"}\n' \
  "$status" "$image" > "$output/container-exit.json"
exit "$status"
