#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
readonly PLATFORM_ROOT="${AUTODL_TMP:-/root/autodl-tmp}"
readonly DURABLE_ROOT="${ROSETTA_AUTODL_ROOT:-${PLATFORM_ROOT}/rosetta}"
readonly ENV_ROOT="${ROSETTA_AUTODL_ENV_ROOT:-${DURABLE_ROOT}/envs/smolvla-cuda-001}"
readonly LEROBOT_REVISION="c903b114a90e703b3f7d0c46cb38727c328c55ff"

[[ "$(uname -s)" == "Linux" ]] || { printf 'error: AutoDL bootstrap requires Linux\n' >&2; exit 2; }
[[ -d "$PLATFORM_ROOT" ]] || { printf 'error: AutoDL data disk is missing: %s\n' "$PLATFORM_ROOT" >&2; exit 2; }
command -v nvidia-smi >/dev/null 2>&1 || { printf 'error: nvidia-smi is unavailable\n' >&2; exit 2; }

if command -v apt-get >/dev/null 2>&1; then
    apt-get -o Acquire::Retries=5 update
    apt-get -o Acquire::Retries=5 install --yes --no-install-recommends \
        ca-certificates \
        libegl1 \
        libgl1 \
        libgles2 \
        libglfw3 \
        libosmesa6
fi

# The selected AutoDL image must provide CUDA PyTorch. The bootstrap deliberately
# preserves it instead of installing or upgrading torch, CUDA, cuDNN or drivers.
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("error: choose an AutoDL PyTorch image with CUDA enabled")
print(f"Using preinstalled torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}")
PY

mkdir -p -- "$DURABLE_ROOT/envs"
if [[ ! -x "$ENV_ROOT/bin/python" ]]; then
    python -m venv --system-site-packages "$ENV_ROOT"
fi

source "$ENV_ROOT/bin/activate"
readonly CONSTRAINTS_FILE="$(mktemp)"
trap 'rm -f -- "$CONSTRAINTS_FILE"' EXIT
python - "$CONSTRAINTS_FILE" <<'PY'
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

pins = []
for package in ("torch", "torchvision"):
    try:
        pins.append(f"{package}=={version(package)}")
    except PackageNotFoundError:
        if package == "torch":
            raise SystemExit("error: the AutoDL image has no preinstalled torch")
Path(sys.argv[1]).write_text("\n".join(pins) + "\n", encoding="utf-8")
PY

PIP_CONSTRAINT="$CONSTRAINTS_FILE" python -m pip install \
    "lerobot[training,smolvla] @ https://github.com/huggingface/lerobot/archive/${LEROBOT_REVISION}.zip" \
    "trackio==0.28.0" \
    "gym-aloha==0.1.4" \
    "pytest==9.1.1" \
    "ruff==0.16.2"
PIP_CONSTRAINT="$CONSTRAINTS_FILE" python -m pip install --no-deps "$REPOSITORY_ROOT"
python -m pip check
python - <<'PY'
import torch
from importlib.metadata import version
assert torch.cuda.is_available()
assert version("lerobot") == "0.6.2"
assert version("trackio") == "0.28.0"
assert version("gym-aloha") == "0.1.4"
print(f"AutoDL environment ready: torch={torch.__version__}, CUDA={torch.version.cuda}")
PY

printf 'Activate with: source %s/bin/activate\n' "$ENV_ROOT"
printf 'Then run: scripts/run_autodl.sh doctor\n'
