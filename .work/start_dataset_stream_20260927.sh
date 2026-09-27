#!/usr/bin/env bash
set -euo pipefail
sshbin=/mnt/c/Windows/System32/OpenSSH/ssh.exe
remote_root=/root/autodl-tmp/rosetta/datasets_external/targeted-20260927-001
for mapping in rosetta_dataset_stream_20260927.py:rosetta_dataset_stream.py dataset_stream_20260927.py:stream.py dataset_status_stream_20260927.py:staging-status-stream.py; do
    cat ".work/${mapping%%:*}" | "$sshbin" -o BatchMode=yes -p 20497 root@connect.cqa1.seetacloud.com "(set -C; cat > $remote_root/${mapping##*:})"
done
cat <<'REMOTE' | "$sshbin" -o BatchMode=yes -p 20497 root@connect.cqa1.seetacloud.com 'bash -s'
set -euo pipefail
root=/root/autodl-tmp/rosetta/datasets_external/targeted-20260927-001
test ! -e "$root/download-stream.log"
pid=$(pgrep -f '^python -u /root/autodl-tmp/rosetta/datasets_external/targeted-20260927-001/parallel8.py$')
test -n "$pid"
kill -TERM "$pid"
for attempt in 1 2 3 4 5; do
    if ! kill -0 "$pid" 2>/dev/null; then break; fi
    sleep 1
done
! kill -0 "$pid" 2>/dev/null
python -m py_compile "$root/stream.py" "$root/rosetta_dataset_stream.py"
tmux new-session -d -s targeted-20260927-stream "python -u $root/stream.py > $root/download-stream.log 2>&1; code=\$?; (set -C; printf '%s\n' \"\$code\" > $root/exit-code-stream.txt)"
sleep 3
tmux has-session -t targeted-20260927-stream
tail -n 2 "$root/download-stream.log"
REMOTE
