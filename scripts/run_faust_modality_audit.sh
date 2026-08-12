#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
readonly EXPERIMENT_ID="m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003"
readonly ORCHESTRATION_ROOT="${REPOSITORY_ROOT}/runs/${EXPERIMENT_ID}/orchestration"
readonly STATUS_PATH="${ORCHESTRATION_ROOT}/faust-b8-modality-audit-001.status"
readonly LOG_PATH="${ORCHESTRATION_ROOT}/faust-b8-modality-audit-001.log"

mkdir -p -- "$ORCHESTRATION_ROOT"
[[ ! -e "$STATUS_PATH" && ! -e "$LOG_PATH" ]] \
    || { printf 'error: Faust modality status or log already exists\n' >&2; exit 3; }

write_status() {
    local exit_code="$1"
    (
        set -o noclobber
        printf 'state=finished\nexit_code=%s\n' "$exit_code" >"$STATUS_PATH"
    )
}

cd -- "$REPOSITORY_ROOT"
set +e
ROSETTA_VLA_XPU_IMAGE=sha256:2696f8e0430050951ffdd16721e41af19473bd741de4c5547cfb330d6f08580b \
ROSETTA_VLA_DOCKER_MEMORY=8g \
    scripts/run_m2_container.sh vla-xpu \
    python -u scripts/diagnose_smolvla_action_repair_modalities.py \
    --plan /workspace/configs/vla/smolvla_450m_aloha_insertion_faust_batch8_002.yaml \
    --checkpoint-step 1875 \
    --shuffle-seed 20260812 \
    >"$LOG_PATH" 2>&1
exit_code="$?"
set -e
write_status "$exit_code"
exit "$exit_code"
