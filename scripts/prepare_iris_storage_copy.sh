#!/usr/bin/env bash
set -Eeuo pipefail
/mnt/c/Windows/System32/OpenSSH/ssh.exe -o BatchMode=yes -o ConnectTimeout=15 \
 -p 20497 root@connect.cqa1.seetacloud.com 'bash -s' <<'REMOTE'
set -Eeuo pipefail
source_dir=/root/autodl-tmp/closed-axis-runtime-archive
copy_dir=/root/iris-preserved-closed-axis-20260913-001
test "$(realpath "$source_dir")" = "$source_dir"
test ! -e "$copy_dir"
needed=$(du -sb "$source_dir" | cut -f1)
available=$(df -B1 --output=avail /root | tail -1 | tr -d ' ')
test "$available" -gt "$((needed + 2147483648))"
mkdir "$copy_dir"
cp -a --reflink=auto "$source_dir/." "$copy_dir/"
cd "$source_dir"
find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > /root/iris-storage-source-20260913-001.sha256
cd "$copy_dir"
sha256sum --check --quiet /root/iris-storage-source-20260913-001.sha256
diff -u <(cd "$source_dir" && find . -printf '%y %p %l\n' | LC_ALL=C sort) <(find . -printf '%y %p %l\n' | LC_ALL=C sort)
printf 'COPY_VERIFIED bytes=%s files=%s manifest_sha256=' "$needed" "$(wc -l </root/iris-storage-source-20260913-001.sha256)"
sha256sum /root/iris-storage-source-20260913-001.sha256
df -h /root /root/autodl-tmp
REMOTE
