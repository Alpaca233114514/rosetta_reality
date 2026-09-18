#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
readonly EXPERIMENT_ID="m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003"
readonly PLAN_PATH="/workspace/configs/vla/smolvla_450m_aloha_insertion_aster_zero_noise_sim_004.yaml"
readonly ORCHESTRATION_ROOT="${REPOSITORY_ROOT}/runs/${EXPERIMENT_ID}/orchestration"

[[ "$#" -eq 1 ]] || { printf 'usage: %s gate3|gate4\n' "$0" >&2; exit 2; }
readonly PHASE="$1"
case "$PHASE" in
    gate3)
        readonly STATUS_PATH="${ORCHESTRATION_ROOT}/aster-zero-noise-gate3-004.status"
        readonly LOG_PATH="${ORCHESTRATION_ROOT}/aster-zero-noise-gate3-004.log"
        readonly CONTAINER_NAME="aster-zero-noise-gate3-004"
        readonly -a GATE_ARGUMENTS=(
            gate3
            --plan "$PLAN_PATH"
        )
        ;;
    gate4)
        readonly STATUS_PATH="${ORCHESTRATION_ROOT}/aster-zero-noise-gate4-004.status"
        readonly LOG_PATH="${ORCHESTRATION_ROOT}/aster-zero-noise-gate4-004.log"
        readonly CONTAINER_NAME="aster-zero-noise-gate4-004"
        readonly -a GATE_ARGUMENTS=(
            gate4
            --plan "$PLAN_PATH"
            --gate3-report "/workspace/runs/${EXPERIMENT_ID}/gates/gate3-smolvla-sim-004.json"
        )
        ;;
    *)
        printf 'error: phase must be gate3 or gate4\n' >&2
        exit 2
        ;;
esac

mkdir -p -- "$ORCHESTRATION_ROOT"
[[ ! -e "$STATUS_PATH" && ! -e "$LOG_PATH" ]] \
    || { printf 'error: Aster zero-noise status or log already exists\n' >&2; exit 3; }

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
ROSETTA_CONTAINER_NAME="$CONTAINER_NAME" \
    scripts/run_m2_container.sh vla-sim-xpu \
    python -u scripts/smolvla_action_repair_sim_gate.py "${GATE_ARGUMENTS[@]}" \
    >"$LOG_PATH" 2>&1
exit_code="$?"
set -e
write_status "$exit_code"
exit "$exit_code"
