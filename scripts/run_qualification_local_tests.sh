#!/usr/bin/env bash
# Synthetic stdlib checks only; existing local image, no downloads or real data.
set -Eeuo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
output=${1:?Pass a NEW output directory under repository .cache}
image=${2:?Pass the verified local stdlib image digest}
output=$(realpath -m "$output")
[[ "$output" == "$repo/.cache/"* && ! -e "$output" ]] || exit 2
[[ "$image" =~ ^sha256:[0-9a-f]{64}$ ]] || exit 2
docker.exe image inspect "$image" >/dev/null
mkdir -p "$output"
set +e
docker.exe run --rm --pull never --network none --read-only --cpus 2 \
    --memory 512m --memory-swap 512m --pids-limit 128 --cap-drop ALL \
    --security-opt no-new-privileges --tmpfs /tmp:rw,size=128m \
    --mount "type=bind,source=$(wslpath -w "$repo"),target=/source,readonly" \
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/source/src:/source \
    --workdir /source "$image" -m unittest discover -v -s tests \
    -p test_execution_qualification_stdlib.py 2>&1 | tee "$output/tests.log"
status=${PIPESTATUS[0]}
set -e
printf '{"scope":"stdlib_synthetic_only","image":"%s","exit_code":%s}\n' \
    "$image" "$status" > "$output/result.json"
exit "$status"
