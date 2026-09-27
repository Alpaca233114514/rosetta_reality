#!/usr/bin/env bash
set -euo pipefail
sshbin=/mnt/c/Windows/System32/OpenSSH/ssh.exe
remote_root=/root/autodl-tmp/rosetta/datasets_external/targeted-20260927-001
cat .work/dataset_parallel8_20260927.py | "$sshbin" -o BatchMode=yes -p 20497 root@connect.cqa1.seetacloud.com "(set -C; cat > $remote_root/parallel8.py)"
cat <<'REMOTE' | "$sshbin" -o BatchMode=yes -p 20497 root@connect.cqa1.seetacloud.com 'bash -s'
set -euo pipefail
root=/root/autodl-tmp/rosetta/datasets_external/targeted-20260927-001
test ! -e "$root/download-parallel8.log"
pid=$(pgrep -f '^python -u /root/autodl-tmp/rosetta/datasets_external/targeted-20260927-001/parallel.py$')
test -n "$pid"
kill -TERM "$pid"
for attempt in 1 2 3 4 5; do
    if ! kill -0 "$pid" 2>/dev/null; then break; fi
    sleep 1
done
! kill -0 "$pid" 2>/dev/null
python -m py_compile "$root/parallel8.py"
tmux new-session -d -s targeted-20260927-parallel8 "python -u $root/parallel8.py > $root/download-parallel8.log 2>&1; code=\$?; (set -C; printf '%s\n' \"\$code\" > $root/exit-code-parallel8.txt)"
sleep 3
tmux has-session -t targeted-20260927-parallel8
tail -n 2 "$root/download-parallel8.log"
REMOTE
