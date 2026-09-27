#!/usr/bin/env bash
# Append verified units to the existing MCP history, with create-only receipts.
set -euo pipefail
cd "$(dirname "$0")/.."
basin_dir="$(pwd)"
source_dir="$basin_dir/outputs/rosetta-canonical-20260922-004/history"
destination_dir="$basin_dir/outputs/rosetta-demo-v2/history"
receipt_dir="$basin_dir/outputs"
preflight_name='rosetta-canonical-append-20260922-001-preflight.json'
receipt_name='rosetta-canonical-append-20260922-001-receipt.json'
if [[ ! -d "$source_dir" || ! -d "$destination_dir" || -e "$receipt_dir/$preflight_name" || -e "$receipt_dir/$receipt_name" ]]; then
  echo 'Source/destination missing or append receipt already exists' >&2
  exit 2
fi
docker_cli="${BASIN_DOCKER_CLI:-docker}"
if ! command -v "$docker_cli" >/dev/null 2>&1 || ! "$docker_cli" version >/dev/null 2>&1; then
  docker_cli='/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe'
fi
if [[ "$docker_cli" == *.exe ]]; then
  basin_mount="$(wslpath -w "$basin_dir")"
  source_mount="$(wslpath -w "$source_dir")"
  destination_mount="$(wslpath -w "$destination_dir")"
  receipt_mount="$(wslpath -w "$receipt_dir")"
else
  basin_mount="$basin_dir"
  source_mount="$source_dir"
  destination_mount="$destination_dir"
  receipt_mount="$receipt_dir"
fi
image='sha256:fb3c1bbda42881fac9b7725d3acb436119934f0c60af29747339d5951c3039da'
"$docker_cli" run --rm --pull never --network none --read-only --cpus 2 --memory 1g --memory-swap 1g \
  --cap-drop ALL --security-opt no-new-privileges --pids-limit 128 --tmpfs /tmp:rw,size=128m \
  --mount "type=bind,source=$basin_mount,target=/basin,readonly" \
  --mount "type=bind,source=$source_mount,target=/source,readonly" \
  --mount "type=bind,source=$destination_mount,target=/destination" \
  --mount "type=bind,source=$receipt_mount,target=/receipts" \
  --workdir /basin -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/basin \
  --entrypoint python "$image" /basin/scripts/append_canonical_history.py \
  --source /source --destination /destination \
  --preflight "/receipts/$preflight_name" --receipt "/receipts/$receipt_name"
