#!/usr/bin/env bash
# Run in a hidden, detached Git Bash process on the local Windows host.
set -Eeuo pipefail
export MSYS_NO_PATHCONV=1
port="${1:?SSH port required}"
workspace="${2:?Registered remote workspace required}"
[[ "$port" =~ ^[0-9]+$ ]]
[[ "$workspace" =~ ^/root/autodl-tmp/rosetta/dev/[a-z0-9-]+$ ]]
root=runs/hestia-cuda-curve-receiver-20260911-001
mkdir "$root"
failure() { printf 'receiver_failed_exit=%s\n' "$?" > "$root/failure.txt"; }
trap failure ERR
ssh_helper=runs/coverage40-unattended-20260910-001/ssh.sh
bash "$ssh_helper" "$port" "cd '$workspace' && python3 scripts/stream_hestia_checkpoint_results.py" > "$root/results.tar" 2> "$root/stream.log"
wsl.exe bash -lc 'cd /mnt/c/Users/Logan/Documents/GitHub/rosetta_reality && export PATH="/mnt/c/Program Files/Docker/Docker/resources/bin:$PATH" && export ROSETTA_VLA_XPU_IMAGE=sha256:fb3c1bbda42881fac9b7725d3acb436119934f0c60af29747339d5951c3039da ROSETTA_VLA_DOCKER_MEMORY=2g && bash scripts/run_m2_container.sh vla-cpu bash -lc "python scripts/verify_hestia_checkpoint_results.py --archive runs/hestia-cuda-curve-receiver-20260911-001/results.tar --output runs/hestia-cuda-curve-recovered-20260911-001 --receipt runs/hestia-cuda-curve-receiver-20260911-001/receipt.json"' > "$root/verify.log" 2>&1
# Receipt is data on stdin; no credential or JSON payload is interpolated into a command.
bash "$ssh_helper" "$port" "cd '$workspace' && python3 -c \"import json,pathlib,sys; data=json.load(sys.stdin); p=pathlib.Path('runs/hestia-cuda-checkpoint-curve-20260911-001/transfer-receipt.json'); t=p.with_suffix('.tmp'); f=t.open('x'); json.dump(data,f); f.close(); t.replace(p)\"" < "$root/receipt.json" > "$root/ack.log" 2>&1
printf '%s\n' 'verified_results_returned_and_shutdown_receipt_sent' > "$root/completed.txt"
