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
archive_path="$(mktemp "${TMPDIR:-/tmp}/rosetta-autodl-workspace.XXXXXX.tar")"
trap 'rm -f -- "$archive_path"' EXIT
git -C "$REPOSITORY_ROOT" ls-files --cached --others --exclude-standard -z \
    | tar --directory="$REPOSITORY_ROOT" --null --files-from=- --sort=name \
        --mtime='@0' --owner=0 --group=0 --numeric-owner \
        --create --file="$archive_path"
workspace_sha256="$(sha256sum "$archive_path" | cut -d' ' -f1)"
tree_hash="${workspace_sha256:0:12}"
release_id="${2:-$(date -u +%Y%m%dT%H%M%SZ)-${head}-${tree_hash}}"
[[ "$release_id" =~ ^[A-Za-z0-9._-]+$ ]] || { printf 'error: unsafe release id\n' >&2; exit 2; }
remote_root="/root/autodl-tmp/rosetta/workspaces/${release_id}"

if ssh "$host" "test -e '$remote_root'"; then
    printf 'error: remote release already exists: %s\n' "$remote_root" >&2
    exit 2
fi
ssh "$host" "mkdir -p '$remote_root'"
ssh "$host" "tar --extract --file=- --directory='$remote_root'" <"$archive_path"
printf '%s  workspace.tar\n' "$workspace_sha256" \
    | ssh "$host" "cat > '$remote_root/.rosetta-workspace.sha256'"

printf 'Remote workspace created: %s\n' "$remote_root"
printf 'Workspace archive SHA-256: %s\n' "$workspace_sha256"
printf 'Next, transfer only the approved immutable data/model cache roots, then run bootstrap_autodl.sh.\n'
