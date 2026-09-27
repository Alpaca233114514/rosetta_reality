#!/usr/bin/env bash
set -euo pipefail
sshbin=/mnt/c/Windows/System32/OpenSSH/ssh.exe
remote_root=/root/autodl-tmp/rosetta/datasets_external/targeted-20260927-001
cat .work/dataset_parallel_20260927.py | "$sshbin" -o BatchMode=yes -p 20497 root@connect.cqa1.seetacloud.com "(set -C; cat > $remote_root/parallel.py)"
cat <<'REMOTE' | "$sshbin" -o BatchMode=yes -p 20497 root@connect.cqa1.seetacloud.com 'bash -s'
set -euo pipefail
root=/root/autodl-tmp/rosetta/datasets_external/targeted-20260927-001
test -f "$root/plan.json"
test ! -e "$root/download-parallel.log"
pid=$(pgrep -f '^python -u /root/autodl-tmp/rosetta/datasets_external/targeted-20260927-001/worker.py$')
test -n "$pid"
kill -TERM "$pid"
for attempt in 1 2 3 4 5; do
    if ! kill -0 "$pid" 2>/dev/null; then break; fi
    sleep 1
done
! kill -0 "$pid" 2>/dev/null
python -m py_compile "$root/parallel.py"
tmux new-session -d -s targeted-20260927-parallel "python -u $root/parallel.py > $root/download-parallel.log 2>&1; code=\$?; (set -C; printf '%s\n' \"\$code\" > $root/exit-code-parallel.txt)"
sleep 3
tmux has-session -t targeted-20260927-parallel
tail -n 3 "$root/download-parallel.log"
REMOTE
