#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
workspace=${1:?Fresh Iris workspace required}
[[ "$workspace" =~ ^/root/autodl-tmp/rosetta/workspaces/[0-9]{8}T[0-9]{6}Z-[a-f0-9]{12}-[a-f0-9]{12}$ ]]
out=runs/iris-gate-recovered-20260913-003
mkdir "$out"
ssh_registered() {
 /mnt/c/Windows/System32/OpenSSH/ssh.exe -o BatchMode=yes -o ConnectTimeout=15 \
  -p 20497 root@connect.cqa1.seetacloud.com "$@"
}
ssh_registered "cat '$workspace/runs/iris-002-gate34-20260913-003/package.json'" > "$out/package.json"
size=$(ssh_registered "stat -c %s '$workspace/runs/iris-002-gate34-20260913-003.tar'" | tr -d '\r')
[[ "$size" =~ ^[0-9]+$ && "$size" -le 5402263552 ]]
ssh_registered "cat '$workspace/runs/iris-002-gate34-20260913-003.tar'" > "$out/archive.tar"
repo=$(wslpath -m "$PWD")
docker.exe run --rm --network none --memory 2g --memory-swap 2g --cpus 2 --pids-limit 128 \
 --read-only --tmpfs /tmp:rw,nosuid,size=128m \
 --mount "type=bind,source=$repo,target=/workspace,readonly" \
 --mount "type=bind,source=$repo/$out,target=/workspace/$out" \
 -w /workspace -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/workspace/src:/workspace \
 sha256:fb3c1bbda42881fac9b7725d3acb436119934f0c60af29747339d5951c3039da \
 python scripts/iris_delivery.py --archive "$out/archive.tar" --destination "$out/verified" --package-json "$out/package.json"
ssh_registered "test ! -e '$workspace/runs/iris-002-gate34-20260913-003/transfer-receipt.json' && cat > '$workspace/runs/iris-002-gate34-20260913-003/transfer-receipt.tmp' && mv '$workspace/runs/iris-002-gate34-20260913-003/transfer-receipt.tmp' '$workspace/runs/iris-002-gate34-20260913-003/transfer-receipt.json'" < "$out/verified/transfer-receipt.json"
printf 'Iris result files verified and receipt delivered\n'
