#!/usr/bin/env bash
# Opt-in narrow transfer: existing remote source plus an inspected local allowlist.
set -Eeuo pipefail
source_only=0
if [[ "${1:-}" == --source-only ]]; then
    source_only=1
    shift
fi
host=${1:?SSH alias required}
base=${2:?Existing verified remote workspace required}
base_sha=${3:?Existing base identity SHA required}
archive=${4:?Previously packed narrow archive required}
archive_sha=${5:?Reviewed archive SHA required}
files=${6:?Reviewed exact file list required}
[[ "$host" =~ ^[A-Za-z0-9._-]+$ ]]
[[ "$base" =~ ^/root/autodl-tmp/rosetta/workspaces/[0-9]{8}T[0-9]{6}Z-[a-f0-9]{12}-[a-f0-9]{12}$ ]]
[[ "$base_sha" =~ ^[a-f0-9]{64}$ && "$archive_sha" =~ ^[a-f0-9]{64}$ ]]
test "$(sha256sum "$archive" | cut -d' ' -f1)" = "$archive_sha"
diff -u <(LC_ALL=C sort "$files") <(tar -tf "$archive" | LC_ALL=C sort)
while IFS= read -r path; do
    [[ "$path" =~ ^[A-Za-z0-9_./-]+$ ]]
    [[ "$path" != /* && "$path" != ../* && "$path" != */../* && "$path" != */.. ]]
done < "$files"
identity=$(printf 'base=%s\ndelta=%s\n' "$base_sha" "$archive_sha" | sha256sum | cut -d' ' -f1)
head=$(git rev-parse --short=12 HEAD)
remote="/root/autodl-tmp/rosetta/workspaces/$(date -u +%Y%m%dT%H%M%SZ)-$head-${identity:0:12}"
ssh "$host" "bash -s -- '$base' '$base_sha' '$remote' '$archive_sha' '$identity' '$source_only'" <<'REMOTE'
set -Eeuo pipefail
base=$1
base_sha=$2
remote=$3
delta_sha=$4
identity=$5
source_only=$6
if [[ -f "$base/.rosetta-workspace.sha256" ]]; then
    test "$(cut -d' ' -f1 "$base/.rosetta-workspace.sha256")" = "$base_sha"
    base_kind=archive
else
    test "$(sed -n 's/^composite_sha256=//p' "$base/.rosetta-composite-workspace.txt")" = "$base_sha"
    base_kind=composite
fi
test ! -e "$remote"
mkdir "$remote"
# Clone only already-present source on the same host; runs/bindings are excluded.
tar -C "$base" --exclude=./runs --exclude=./.transport --exclude=./.rosetta-workspace.sha256 \
    --exclude=./.rosetta-composite-workspace.txt -cf - . \
    | tar -C "$remote" --keep-old-files -xf -
if [[ "$source_only" == 0 ]]; then
tar -C "$base" -cf - runs/hestia-recovered-20260911-001/C-first \
    runs/hestia-recovered-20260911-001/cpu/draft/main1280.yaml \
    | tar -C "$remote" --keep-old-files -xf -
if [[ "$base_kind" == composite ]]; then
    tar -C "$base" -cf - runs/hestia-cuda-curve-recovered-20260912-001/000640/arrays.npz \
        runs/hestia-cuda-curve-recovered-20260912-001/001280/arrays.npz \
        runs/hestia-parameter-crossover-audit-20260912-001/result.json \
        | tar -C "$remote" --keep-old-files -xf -
fi
fi
printf 'base_identity_kind=%s\nbase_identity_sha256=%s\ndelta_archive_sha256=%s\ncomposite_sha256=%s\n' \
    "$base_kind" "$base_sha" "$delta_sha" "$identity" > "$remote/.rosetta-composite-workspace.txt"
REMOTE
if [[ "$source_only" == 1 ]]; then
    # Replacements apply only inside the new independent clone, never to the base.
    # Refuse symlink parents before applying an allowlisted source delta.
    mapfile -t allowed_paths < "$files"
    ssh "$host" bash -s -- "$remote" "${allowed_paths[@]}" <<'CHECK'
set -Eeuo pipefail
root=$1
shift
for relative in "$@"; do
cursor=$root
IFS=/ read -ra parts <<< "$relative"
for part in "${parts[@]}"; do
    cursor="$cursor/$part"
    test ! -L "$cursor"
done
done
CHECK
    ssh "$host" "tar --overwrite --extract --file=- --directory='$remote'" < "$archive"
else
    ssh "$host" "tar --keep-old-files --extract --file=- --directory='$remote'" < "$archive"
fi
printf 'Remote workspace created: %s\n' "$remote"
printf 'Base identity SHA-256: %s\n' "$base_sha"
printf 'Narrow delta archive SHA-256: %s\n' "$archive_sha"
printf 'Composite workspace SHA-256: %s\n' "$identity"
