#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
template=${1:?Template path required}
expected=${2:?Template SHA required}
[[ "$template" =~ ^reports/training/[A-Za-z0-9._-]+\.json$ ]]
[[ "$expected" =~ ^[a-f0-9]{64}$ ]]
test "$(sha256sum "$template" | cut -d' ' -f1)" = "$expected"
source /root/autodl-tmp/rosetta/envs/smolvla-cuda-001/bin/activate
bash scripts/run_autodl.sh shell <<EOF
exec python scripts/run_iris_gate.py supervise --template '$template' --sha256 '$expected' --execute-authorized --shutdown-authorized
EOF
