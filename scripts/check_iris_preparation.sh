#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
for script in scripts/launch_iris_worker.sh scripts/launch_iris_from_wsl.sh scripts/receive_iris_from_wsl.sh; do
 bash -n "$script"
done
repo=$(wslpath -m "$PWD")
image=sha256:fb3c1bbda42881fac9b7725d3acb436119934f0c60af29747339d5951c3039da
mounts=()
if [[ "${1:-check}" == prepare ]]; then
 mkdir -p runs/iris-furnace-preparation-20260913-002
 mounts+=(--mount "type=bind,source=$repo/runs/iris-furnace-preparation-20260913-002,target=/workspace/runs/iris-furnace-preparation-20260913-002")
fi
if [[ "${1:-check}" == format ]]; then
 for file in src/rosetta_reality/vla/image_key_regularization.py src/rosetta_reality/vla/training/features.py src/rosetta_reality/vla/training/plan.py scripts/audit_iris_attention_backend.py scripts/iris_runtime.py scripts/iris_protocol.py scripts/iris_calibration.py scripts/iris_stage.py scripts/run_iris_furnace.py scripts/iris_delivery.py scripts/analyze_iris.py tests/test_iris_furnace.py tests/test_iris_runtime.py tests/test_image_key_regularization.py tests/test_smolvla_training_features.py; do
  mounts+=(--mount "type=bind,source=$repo/$file,target=/workspace/$file")
 done
fi
docker.exe run --rm --network none --memory 2g --memory-swap 2g --cpus 2 --pids-limit 128 \
 --read-only --tmpfs /tmp:rw,nosuid,size=128m \
 --mount "type=bind,source=$repo,target=/workspace,readonly" \
 "${mounts[@]}" -e "MODE=${1:-check}" -w /workspace -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/workspace/src:/workspace \
 -e OMP_NUM_THREADS=2 -e HF_HUB_OFFLINE=1 "$image" timeout 180 bash -lc '
set -Eeuo pipefail
if [[ "$MODE" == prepare ]]; then
 python scripts/run_iris_furnace.py prepare --template runs/iris-furnace-preparation-20260913-002/template.json
 exit
fi
files=(src/rosetta_reality/vla/image_key_regularization.py src/rosetta_reality/vla/training/features.py src/rosetta_reality/vla/training/plan.py scripts/audit_iris_attention_backend.py scripts/iris_runtime.py scripts/iris_protocol.py scripts/iris_calibration.py scripts/iris_stage.py scripts/run_iris_furnace.py scripts/iris_delivery.py scripts/analyze_iris.py tests/test_iris_furnace.py tests/test_iris_runtime.py tests/test_image_key_regularization.py tests/test_smolvla_training_features.py)
if [[ "$MODE" == format ]]; then
 python -m ruff format --no-cache "${files[@]}"
 python -m ruff check --fix --no-cache "${files[@]}"
 exit
fi
python scripts/check_env.py
python -m pytest -q -p no:cacheprovider tests/test_image_key_regularization.py tests/test_iris_runtime.py tests/test_iris_furnace.py tests/test_smolvla_training_features.py tests/test_smolvla_training_plan_schema.py
python -m ruff check --no-cache "${files[@]}"
python -m ruff format --check --no-cache "${files[@]}"
'
