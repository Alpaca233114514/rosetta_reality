#!/usr/bin/env bash
set -Eeuo pipefail
cd /mnt/c/Users/Logan/Documents/GitHub/rosetta_reality
sshbin=/mnt/c/Windows/System32/OpenSSH/ssh.exe
remote=/root/autodl-tmp/rosetta/runs/execution-readiness-20260927-001
local=runs/execution-readiness-received-20260927-001
test ! -e "$local"
mkdir "$local"
"$sshbin" -o BatchMode=yes -o ConnectTimeout=15 -p 20497 root@connect.cqa1.seetacloud.com "cd '$remote' && test -f result.json && (set -C; sha256sum result.json worker-exited.json registration.json guard-armed.json worker-pid.json audit.py guard.py launch.sh payload.sha256 source/* > evidence.sha256) && tar -cf - result.json worker-exited.json registration.json guard-armed.json worker-pid.json audit.py guard.py launch.sh payload.sha256 source evidence.sha256" > "$local/results.tar"
tar -tf "$local/results.tar" > "$local/members.txt"
if grep -E '(^/|(^|/)\.\.(/|$))' "$local/members.txt"; then exit 2; fi
mkdir "$local/verified"
tar --no-same-owner --no-same-permissions -xf "$local/results.tar" -C "$local/verified"
(cd "$local/verified" && sha256sum -c evidence.sha256)
sha256sum "$local/results.tar" > "$local/archive.sha256"
