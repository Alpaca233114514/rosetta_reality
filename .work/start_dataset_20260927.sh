#!/usr/bin/env bash
set -euo pipefail
sshbin=/mnt/c/Windows/System32/OpenSSH/ssh.exe
remote_root=/root/autodl-tmp/rosetta/datasets_external/targeted-20260927-001
cat .work/dataset_worker_20260927.py | "$sshbin" -o BatchMode=yes -o ConnectTimeout=15 -p 20497 root@connect.cqa1.seetacloud.com "(set -C; cat > $remote_root/worker.py)"
"$sshbin" -o BatchMode=yes -o ConnectTimeout=15 -p 20497 root@connect.cqa1.seetacloud.com "python -m py_compile $remote_root/worker.py; test ! -e $remote_root/download.log; tmux new-session -d -s targeted-20260927-001 'python -u $remote_root/worker.py > $remote_root/download.log 2>&1; code=\$?; (set -C; printf \"%s\\n\" \"\$code\" > $remote_root/exit-code.txt)'; sleep 3; tmux has-session -t targeted-20260927-001; tail -n 3 $remote_root/download.log"
