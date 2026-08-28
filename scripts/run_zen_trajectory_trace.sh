#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
readonly EXPERIMENT_ID="m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003"
readonly ORCHESTRATION_ROOT="${REPOSITORY_ROOT}/runs/${EXPERIMENT_ID}/orchestration"
readonly STATUS_PATH="${ORCHESTRATION_ROOT}/zen-trajectory-trace-001.status"
readonly LOG_PATH="${ORCHESTRATION_ROOT}/zen-trajectory-trace-001.log"

mkdir -p -- "$ORCHESTRATION_ROOT"
[[ ! -e "$STATUS_PATH" && ! -e "$LOG_PATH" ]] \
    || { printf 'error: Zen trajectory trace status or log already exists\n' >&2; exit 3; }

write_status() {
    local exit_code="$1"
    (
        set -o noclobber
        printf 'state=finished\nexit_code=%s\n' "$exit_code" >"$STATUS_PATH"
    )
}

cd -- "$REPOSITORY_ROOT"
set +e
ROSETTA_VLA_SIM_XPU_IMAGE=sha256:f4a71c4020cd54d2a878f01628d591af9572f0784458f4c821008f8aea30393c \
ROSETTA_VLA_DOCKER_MEMORY=6g \
ROSETTA_CONTAINER_NAME=zen-trajectory-trace-001 \
    scripts/run_m2_container.sh vla-sim-xpu \
    python -u scripts/diagnose_smolvla_zen_trajectory.py \
    --artifact-id m2-smolvla450m-zen-cuda-b64-firstaction-001-step0316-deploy-001 \
    --episode 2 \
    --seed 10 \
    --policy-noise-seed 10 \
    --maximum-steps 320 \
    >"$LOG_PATH" 2>&1
exit_code="$?"
set -e
write_status "$exit_code"
exit "$exit_code"
