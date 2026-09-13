#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
out=runs/iris-attention-backend-recovered-20260913-001
mkdir "$out"
remote=/root/autodl-tmp/rosetta/workspaces/20260913T121956Z-300c10938db1-acb2d84d4d48/runs/iris-attention-backend-20260913-001
for file in result.json audit.log exit-code; do
 /mnt/c/Windows/System32/OpenSSH/ssh.exe -o BatchMode=yes -p 20497 root@connect.cqa1.seetacloud.com "cat '$remote/$file'" > "$out/$file"
done
/mnt/c/Windows/System32/OpenSSH/ssh.exe -o BatchMode=yes -p 20497 root@connect.cqa1.seetacloud.com "cd '$remote' && sha256sum result.json audit.log exit-code" | tr -d '\r' > "$out/SHA256SUMS"
cd "$out"
sha256sum --check SHA256SUMS
