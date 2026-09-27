#!/usr/bin/env bash
set -euo pipefail
set -C
worker_dir=/root/autodl-tmp/rosetta/datasets_external/targeted-20260925-001/worker
worker=$worker_dir/remote_download_selected_datasets_20260925.py
log=$worker_dir/download.log
pidfile=$worker_dir/download.pid
test -f "$worker"
test ! -e "$log"
test ! -e "$pidfile"
free_bytes=$(df -B1 --output=avail /root/autodl-tmp | tail -1 | tr -d ' ')
test "$free_bytes" -ge 95000000000
nohup python -u "$worker" > "$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$pidfile"
sleep 2
kill -0 "$pid"
printf 'pid=%s free_bytes=%s log=%s\n' "$pid" "$free_bytes" "$log"
