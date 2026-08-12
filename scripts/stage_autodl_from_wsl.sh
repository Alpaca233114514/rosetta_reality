#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

usage() {
    cat <<'EOF'
Usage: scripts/stage_autodl_from_wsl.sh SSH_HOST [REMOTE_RELEASE_ID]

Creates a new versioned remote workspace without deleting or replacing any
existing remote workspace. SSH_HOST may be an ~/.ssh/config alias. Large data
and model caches are intentionally transferred by separate explicit rsync calls.
EOF
}

host="${1:-}"
[[ -n "$host" ]] || { usage; exit 2; }
[[ "$host" =~ ^[A-Za-z0-9._-]+$ ]] || { printf 'error: unsafe SSH host alias\n' >&2; exit 2; }

head="$(git -C "$REPOSITORY_ROOT" rev-parse --short=12 HEAD)"
tree_hash="$(git -C "$REPOSITORY_ROOT" status --porcelain=v1 | sha256sum | cut -c1-12)"
release_id="${2:-$(date -u +%Y%m%dT%H%M%SZ)-${head}-${tree_hash}}"
[[ "$release_id" =~ ^[A-Za-z0-9._-]+$ ]] || { printf 'error: unsafe release id\n' >&2; exit 2; }
remote_root="/root/autodl-tmp/rosetta/workspaces/${release_id}"

if ssh "$host" "test -e '$remote_root'"; then
    printf 'error: remote release already exists: %s\n' "$remote_root" >&2
    exit 2
fi
ssh "$host" "mkdir -p '$remote_root'"
git -C "$REPOSITORY_ROOT" ls-files --cached --others --exclude-standard -z \
    | tar --directory="$REPOSITORY_ROOT" --null --files-from=- --create --gzip --file=- \
    | ssh "$host" "tar --extract --gzip --file=- --directory='$remote_root'"

printf 'Remote workspace created: %s\n' "$remote_root"
printf 'Next, transfer only the approved immutable data/model cache roots, then run bootstrap_autodl.sh.\n'
