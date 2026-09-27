#!/usr/bin/env bash
set -Eeuo pipefail
cd /mnt/c/Users/Logan/Documents/GitHub/rosetta_reality
sshbin=/mnt/c/Windows/System32/OpenSSH/ssh.exe
job=/root/autodl-tmp/rosetta/runs/execution-readiness-20260927-001
archive=.work/execution-readiness-20260927-001.tar
test ! -e "$archive"
bash -n .work/execution-readiness-20260927-001/launch.sh
tar -C .work/execution-readiness-20260927-001 -cf "$archive" .
digest=$(sha256sum "$archive" | cut -d' ' -f1)
printf '%s  %s\n' "$digest" "$archive" > .work/execution-readiness-20260927-001.archive.sha256
"$sshbin" -o BatchMode=yes -o ConnectTimeout=15 -p 20497 root@connect.cqa1.seetacloud.com "test ! -e '$job' && mkdir '$job' && (set -C; cat > '$job/payload.tar')" < "$archive"
"$sshbin" -o BatchMode=yes -o ConnectTimeout=15 -p 20497 root@connect.cqa1.seetacloud.com "cd '$job' && printf '%s  payload.tar\\n' '$digest' | sha256sum -c - && tar -xf payload.tar && sha256sum -c payload.sha256 && bash launch.sh"
