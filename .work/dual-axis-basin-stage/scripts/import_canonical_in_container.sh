#!/usr/bin/env bash
# Create a new Basin history from retained Rosetta evidence; no model execution.
set -euo pipefail
cd "$(dirname "$0")/.."
basin_dir="$(pwd)"
rosetta_dir="$(cd ../rosetta_reality && pwd)"
runs_dir="$(realpath /mnt/d/rosetta_reality/offload-20260920/runs)"
output_parent="$basin_dir/outputs"
output_name="${1:-rosetta-canonical-20260922-001}"
if [[ "$output_name" != rosetta-canonical-* || "$output_name" == */* || -e "$output_parent/$output_name" ]]; then
  echo 'Expected a new rosetta-canonical-* output directory name' >&2
  exit 2
fi
mkdir -p "$output_parent"
docker_cli="${BASIN_DOCKER_CLI:-docker}"
if ! command -v "$docker_cli" >/dev/null 2>&1 || ! "$docker_cli" version >/dev/null 2>&1; then
  docker_cli='/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe'
fi
if [[ "$docker_cli" == *.exe ]]; then
  basin_mount="$(wslpath -w "$basin_dir")"
  rosetta_mount="$(wslpath -w "$rosetta_dir")"
  runs_mount="$(wslpath -w "$runs_dir")"
  output_mount="$(wslpath -w "$output_parent")"
else
  basin_mount="$basin_dir"
  rosetta_mount="$rosetta_dir"
  runs_mount="$runs_dir"
  output_mount="$output_parent"
fi
image='sha256:fb3c1bbda42881fac9b7725d3acb436119934f0c60af29747339d5951c3039da'
"$docker_cli" run --rm --pull never --network none --read-only --cpus 2 --memory 1g --memory-swap 1g \
  --cap-drop ALL --security-opt no-new-privileges --pids-limit 128 --tmpfs /tmp:rw,size=128m \
  --mount "type=bind,source=$basin_mount,target=/basin,readonly" \
  --mount "type=bind,source=$rosetta_mount,target=/rosetta,readonly" \
  --mount "type=bind,source=$runs_mount,target=/runs,readonly" \
  --mount "type=bind,source=$output_mount,target=/out" \
  --workdir /basin -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/basin \
  --entrypoint python "$image" /basin/scripts/import_canonical_history.py \
  --rosetta /rosetta --runs /runs --output "/out/$output_name"
