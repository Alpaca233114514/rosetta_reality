#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
template=${1:?Template path required}
expected=${2:?Template SHA required}
[[ "$template" =~ ^reports/training/[A-Za-z0-9._-]+\.json$ ]]
[[ "$expected" =~ ^[a-f0-9]{64}$ ]]
test "$(sha256sum "$template" | cut -d' ' -f1)" = "$expected"
source /root/autodl-tmp/rosetta/envs/smolvla-cuda-001/bin/activate
run_name=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["id"])' "$template")
[[ "$run_name" =~ ^canonical-fullframes-posttrain-[0-9]{8}-[0-9]{3}$ ]]
mkdir -p runs
set -o noclobber
exec > "runs/${run_name}-launcher.log" 2>&1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
bash scripts/run_autodl.sh shell <<EOF
export PYTHONPATH="\$PWD:\$PWD/src:\$PWD/scripts"
exec python -m scripts.run_canonical_fullframes_posttrain supervise --template '$template' --sha256 '$expected' --execute-authorized --shutdown-authorized
EOF
