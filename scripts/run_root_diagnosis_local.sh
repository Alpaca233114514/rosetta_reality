#!/usr/bin/env bash
# WSL-only offline CPU runner; materialize code to avoid Windows junction escapes.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
docker=$(command -v docker.exe)
image='sha256:fb3c1bbda42881fac9b7725d3acb436119934f0c60af29747339d5951c3039da'
repo_host=$(wslpath -w "$repo")
runs_host=${1:?Pass the verified Windows path to the existing runs directory}
shift
memory_mib=${ROSETTA_ROOT_MEMORY_MIB:-3072}
wall_seconds=${ROSETTA_ROOT_WALL_SECONDS:-180}
[[ "$memory_mib" =~ ^[0-9]+$ && "$wall_seconds" =~ ^[0-9]+$ ]] || exit 2
(( memory_mib >= 512 && memory_mib <= 3072 && wall_seconds >= 1 && wall_seconds <= 600 )) || exit 2
"$docker" image inspect "$image" >/dev/null
exec "$docker" run --rm --network none --cpus 2 --memory "${memory_mib}m" --memory-swap "${memory_mib}m" \
  --pids-limit 256 --env OMP_NUM_THREADS=2 --env MKL_NUM_THREADS=2 \
  --env "ROSETTA_ROOT_ENFORCED_WALL_SECONDS=$wall_seconds" \
  --env HF_HUB_OFFLINE=1 --env HF_DATASETS_OFFLINE=1 --env PYTHONDONTWRITEBYTECODE=1 \
  --mount "type=bind,source=$repo_host,target=/source,readonly" \
  --mount "type=bind,source=$runs_host,target=/work/runs" \
  --mount "type=bind,source=$repo_host/reports,target=/work/reports,readonly" \
  --workdir /work "$image" bash -c '
    for name in src scripts tests configs docs pyproject.toml; do cp -a "/source/$name" /work/; done
    export PYTHONPATH=/work/src:/work:/work/scripts
    exec timeout --signal=TERM --kill-after=5s "$ROSETTA_ROOT_ENFORCED_WALL_SECONDS" "$@"
  ' bash "$@"
