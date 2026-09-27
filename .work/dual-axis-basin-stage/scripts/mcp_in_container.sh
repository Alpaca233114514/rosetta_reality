#!/usr/bin/env bash
# Read-only access to Basin's imported evidence. No Rosetta mount or network needed.
set -euo pipefail
cd "$(dirname "$0")/.."
basin_dir="$(pwd)"
docker_cli="${BASIN_DOCKER_CLI:-docker}"
if ! command -v "$docker_cli" >/dev/null 2>&1 || ! "$docker_cli" version >/dev/null 2>&1; then
  docker_cli='/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe'
fi
if [[ "$docker_cli" == *.exe ]]; then
  basin_mount="$(wslpath -w "$basin_dir")"
else
  basin_mount="$basin_dir"
fi
exec "$docker_cli" run --rm -i --pull never --network none --read-only \
  --cpus 2 --memory 1g --memory-swap 1g --cap-drop ALL \
  --security-opt no-new-privileges --pids-limit 128 \
  --mount "type=bind,source=$basin_mount,target=/basin,readonly" \
  --workdir /basin -e PYTHONDONTWRITEBYTECODE=1 --entrypoint python \
  sha256:fb3c1bbda42881fac9b7725d3acb436119934f0c60af29747339d5951c3039da \
  -m basin --store "${1:-/basin/outputs/rosetta-demo-v2/history}" mcp
