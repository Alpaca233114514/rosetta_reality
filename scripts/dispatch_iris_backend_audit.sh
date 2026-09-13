#!/usr/bin/env bash
set -Eeuo pipefail
/mnt/c/Windows/System32/OpenSSH/ssh.exe -o BatchMode=yes -o ConnectTimeout=15 \
 -p 20497 root@connect.cqa1.seetacloud.com 'bash -s' <<'REMOTE'
set -Eeuo pipefail
cd /root/autodl-tmp/rosetta/workspaces/20260913T121956Z-300c10938db1-acb2d84d4d48
test "$(sha256sum scripts/audit_iris_attention_backend.py | cut -d' ' -f1)" = bb14d3cfb552604f96d1b4f1d0ae6788686aa5dcbb416e72ced8291b241309a3
test "$(sha256sum configs/runtime/autodl_rtx4090.yaml | cut -d' ' -f1)" = be2bfc3ea2a518c85e56410ba3ea1da6f744236d51b6f4a5f6a7b73927e9f992
mkdir -p runs
mkdir runs/iris-attention-backend-20260913-001
tmux new-session -d -s iris-backend-20260913-001 "source /root/autodl-tmp/rosetta/envs/smolvla-cuda-001/bin/activate; timeout --kill-after=10 180 bash scripts/run_autodl.sh preflight python scripts/audit_iris_attention_backend.py --root /root/autodl-tmp/rosetta/workspaces/20260913T112429Z-300c10938db1-2e532a4aa31f/runs/hestia-q-replay-20260913-001 --output runs/iris-attention-backend-20260913-001/result.json >runs/iris-attention-backend-20260913-001/audit.log 2>&1; echo \$? >runs/iris-attention-backend-20260913-001/exit-code"
printf 'Dispatched bounded tensor-only audit\n'
REMOTE
