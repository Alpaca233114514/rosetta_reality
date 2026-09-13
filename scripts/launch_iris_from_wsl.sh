#!/usr/bin/env bash
set -Eeuo pipefail
workspace=${1:?Fresh staged workspace required}
template=${2:?Sealed template relative path required}
expected=${3:?Sealed template SHA required}
[[ "$workspace" =~ ^/root/autodl-tmp/rosetta/workspaces/[0-9]{8}T[0-9]{6}Z-[a-f0-9]{12}-[a-f0-9]{12}$ ]]
[[ "$template" =~ ^reports/training/[A-Za-z0-9._-]+\.json$ ]]
[[ "$expected" =~ ^[a-f0-9]{64}$ ]]
/mnt/c/Windows/System32/OpenSSH/ssh.exe -o BatchMode=yes -o ConnectTimeout=15 \
 -p 20497 root@connect.cqa1.seetacloud.com "bash -s -- '$workspace' '$template' '$expected'" <<'REMOTE'
set -Eeuo pipefail
cd "$1"
test ! -e runs/iris-k-scene-20260913-001
test ! -e runs/iris-k-scene-20260913-001-supervisor.log
test "$(sha256sum "$2" | cut -d' ' -f1)" = "$3"
mkdir -p runs
nohup bash scripts/launch_iris_worker.sh "$2" "$3" \
 >runs/iris-k-scene-20260913-001-supervisor.log 2>&1 </dev/null &
printf 'IRIS_SUPERVISOR_PID=%s\n' "$!"
REMOTE
