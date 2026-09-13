#!/usr/bin/env bash
set -Eeuo pipefail
/mnt/c/Windows/System32/OpenSSH/ssh.exe -o BatchMode=yes -o ConnectTimeout=15 \
 -p 20497 root@connect.cqa1.seetacloud.com 'bash -s' <<'REMOTE'
set -Eeuo pipefail
source_dir=/root/autodl-tmp/closed-axis-runtime-archive
copy_dir=/root/iris-preserved-closed-axis-20260913-001
manifest=/root/iris-storage-source-20260913-001.sha256
test -d "$source_dir" && test ! -L "$source_dir"
test -d "$copy_dir" && test ! -L "$copy_dir"
test "$(realpath "$source_dir")" = /root/autodl-tmp/closed-axis-runtime-archive
test "$(realpath "$copy_dir")" = /root/iris-preserved-closed-axis-20260913-001
test "$(sha256sum "$manifest" | cut -d' ' -f1)" = b383e6d6e04e031ef5bebedc0d005dc8b2f7033e53c02f6b37c6660c1b19acfb
cd "$copy_dir"
sha256sum --check --quiet "$manifest"
diff -u <(cd "$source_dir" && find . -printf '%y %p %l\n' | LC_ALL=C sort) <(find . -printf '%y %p %l\n' | LC_ALL=C sort)
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
available_before=$(df -B1 --output=avail /root/autodl-tmp | tail -1 | tr -d ' ')
# User explicitly approved this exact original directory and replacement link.
# The verified copy is on another filesystem and is never a deletion target.
rm -rf --one-file-system -- /root/autodl-tmp/closed-axis-runtime-archive
ln -s -- /root/iris-preserved-closed-axis-20260913-001 /root/autodl-tmp/closed-axis-runtime-archive
test "$(realpath /root/autodl-tmp/closed-axis-runtime-archive)" = "$copy_dir"
available_after=$(df -B1 --output=avail /root/autodl-tmp | tail -1 | tr -d ' ')
printf 'CLEANUP_COMPLETE before=%s after=%s freed=%s files=232 manifest_sha256=b383e6d6e04e031ef5bebedc0d005dc8b2f7033e53c02f6b37c6660c1b19acfb\n' \
 "$available_before" "$available_after" "$((available_after - available_before))"
df -h /root/autodl-tmp /root
REMOTE
