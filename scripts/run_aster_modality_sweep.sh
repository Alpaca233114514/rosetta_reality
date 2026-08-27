#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
readonly EXPERIMENT_ID="m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003"
readonly PLAN_PATH="/workspace/configs/vla/smolvla_450m_aloha_insertion_aster_batch8_003.yaml"
readonly ORCHESTRATION_ROOT="${REPOSITORY_ROOT}/runs/${EXPERIMENT_ID}/orchestration"
readonly STATUS_PATH="${ORCHESTRATION_ROOT}/aster-modality-sweep-001.status"
readonly LOG_PATH="${ORCHESTRATION_ROOT}/aster-modality-sweep-001.log"
readonly -a CHECKPOINT_STEPS=(625 1250 1875)

mkdir -p -- "$ORCHESTRATION_ROOT"
[[ ! -e "$STATUS_PATH" && ! -e "$LOG_PATH" ]] \
    || { printf 'error: Aster modality sweep status or log already exists\n' >&2; exit 3; }

write_status() {
    local state="$1"
    local exit_code="$2"
    local checkpoint_step="$3"
    local temporary="${STATUS_PATH}.partial"
    printf 'state=%s\nexit_code=%s\ncheckpoint_step=%s\n' \
        "$state" "$exit_code" "$checkpoint_step" >"$temporary"
    mv -- "$temporary" "$STATUS_PATH"
}

cd -- "$REPOSITORY_ROOT"
write_status running -1 0
for checkpoint_step in "${CHECKPOINT_STEPS[@]}"; do
    write_status running -1 "$checkpoint_step"
    set +e
    ROSETTA_VLA_XPU_IMAGE=sha256:2696f8e0430050951ffdd16721e41af19473bd741de4c5547cfb330d6f08580b \
    ROSETTA_VLA_DOCKER_MEMORY=8g \
    ROSETTA_CONTAINER_NAME="aster-modality-${checkpoint_step}-001" \
        scripts/run_m2_container.sh vla-xpu \
        python -u scripts/diagnose_smolvla_aster_modalities.py \
        --plan "$PLAN_PATH" \
        --checkpoint-step "$checkpoint_step" \
        >>"$LOG_PATH" 2>&1
    exit_code="$?"
    set -e
    if [[ "$exit_code" -ne 0 ]]; then
        write_status failed "$exit_code" "$checkpoint_step"
        exit "$exit_code"
    fi
done
write_status finished 0 "${CHECKPOINT_STEPS[-1]}"
