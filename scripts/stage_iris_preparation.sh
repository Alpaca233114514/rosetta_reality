#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
ssh() {
    [[ "$1" == iris-registered ]] || return 2
    shift
    /mnt/c/Windows/System32/OpenSSH/ssh.exe -o BatchMode=yes -o ConnectTimeout=15 \
        -p 20497 root@connect.cqa1.seetacloud.com "$@"
}
export -f ssh
bash scripts/stage_autodl_from_wsl.sh iris-registered
