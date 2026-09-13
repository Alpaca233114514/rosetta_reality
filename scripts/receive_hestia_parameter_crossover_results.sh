#!/usr/bin/env bash
# Start only after the user supplies current SSH access and authorizes the worker.
set -Eeuo pipefail
host=${1:?SSH user@host required}
port=${2:?SSH port required}
workspace=${3:?Fresh registered remote workspace required}
[[ "$host" =~ ^[a-zA-Z0-9_-]+@[a-zA-Z0-9.-]+$ ]]
[[ "$port" =~ ^[0-9]+$ ]]
[[ "$workspace" =~ ^/root/autodl-tmp/rosetta/workspaces/[A-Za-z0-9._-]+$ ]]
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$repo"
root=runs/hestia-parameter-crossover-receiver-20260912-001
output=runs/hestia-parameter-crossover-recovered-20260912-001
mkdir "$root"
failure() { printf 'receiver_failed_exit=%s\n' "$?" > "$root/failure.txt"; }
trap failure ERR
ssh -p "$port" "$host" "cd '$workspace' && python3 scripts/stream_hestia_parameter_crossover_results.py" \
  > "$root/results.tar" 2> "$root/stream.log"
winroot=$(wslpath -m "$repo")
image=sha256:fb3c1bbda42881fac9b7725d3acb436119934f0c60af29747339d5951c3039da
mkdir "$output"
# Verifier creates its child destination atomically; enclosing output is our new mount.
docker.exe run --rm --network none --memory 2g --memory-swap 2g --cpus 2 \
  --pids-limit 128 --read-only --tmpfs /tmp:rw,nosuid,size=128m \
  --mount "type=bind,source=$winroot,target=/workspace,readonly" \
  --mount "type=bind,source=$winroot/$root,target=/workspace/$root" \
  --mount "type=bind,source=$winroot/$output,target=/workspace/$output" \
  -w /workspace -e PYTHONDONTWRITEBYTECODE=1 -e HF_HUB_OFFLINE=1 \
  "$image" python scripts/verify_hestia_parameter_crossover_results.py \
  --archive "$root/results.tar" --output "$output/verified" --receipt "$root/receipt.json" \
  > "$root/verify.log" 2>&1
ssh -p "$port" "$host" "cd '$workspace' && python3 -c \"import json,pathlib,sys; data=json.load(sys.stdin); p=pathlib.Path('runs/hestia-parameter-crossover-20260912-001/transfer-receipt.json'); f=p.open('x'); json.dump(data,f); f.close()\"" \
  < "$root/receipt.json" > "$root/ack.log" 2>&1
printf '%s\n' 'verified_results_returned_and_shutdown_receipt_sent' > "$root/completed.txt"
