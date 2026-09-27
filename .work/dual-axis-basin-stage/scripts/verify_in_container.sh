#!/usr/bin/env bash
# Invoke from wsl.exe bash. Uses existing Linux image; never pulls or installs.
set -euo pipefail
cd "$(dirname "$0")/.."
basin_dir="$(pwd)"
rosetta_dir="$(cd ../rosetta_reality && pwd)"
evidence_dir="$(realpath "$rosetta_dir/runs/canonical-posttrain-received-20260916-005-ssh/verified")"
docker_cli="${BASIN_DOCKER_CLI:-docker}"
if ! command -v "$docker_cli" >/dev/null 2>&1 || ! "$docker_cli" version >/dev/null 2>&1; then
  docker_cli='/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe'
fi
image='sha256:fb3c1bbda42881fac9b7725d3acb436119934f0c60af29747339d5951c3039da'
if [[ "$docker_cli" == *.exe ]]; then
  basin_mount="$(wslpath -w "$basin_dir")"
  rosetta_mount="$(wslpath -w "$rosetta_dir")"
  evidence_mount="$(wslpath -w "$evidence_dir")"
else
  basin_mount="$basin_dir"
  rosetta_mount="$rosetta_dir"
  evidence_mount="$evidence_dir"
fi
common=(run --rm --pull never --network none --read-only --cpus 2 --memory 1g --memory-swap 1g
  --cap-drop ALL --security-opt no-new-privileges --pids-limit 128 --tmpfs /tmp:rw,size=128m
  --mount "type=bind,source=$basin_mount,target=/basin"
  --mount "type=bind,source=$rosetta_mount,target=/rosetta,readonly"
  --mount "type=bind,source=$evidence_mount,target=/evidence,readonly"
  --workdir /basin -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/basin --entrypoint python)
"$docker_cli" "${common[@]}" "$image" -m unittest discover -s tests -v
if [[ "${1:-}" == '--tests-only' ]]; then
  exit 0
fi
if [[ "${1:-}" == '--api-demo' ]]; then
  "$docker_cli" "${common[@]}" "$image" scripts/agent_api_demo.py
  exit 0
fi
"$docker_cli" "${common[@]}" "$image" scripts/demo.py --output "${1:-outputs/rosetta-demo-v2}" --rosetta /rosetta --trace-root /evidence
