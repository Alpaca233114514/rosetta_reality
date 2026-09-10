#!/usr/bin/env bash
# Vcdropout Gate 3/4 local execution runner (create-only status/log identity).
# Usage: scripts/run_vcdropout_sim_gate.sh gate3
#        scripts/run_vcdropout_sim_gate.sh gate4
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly REPOSITORY_ROOT="${SCRIPT_DIR}"
readonly EXPERIMENT_ID="m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003"
readonly ORCHESTRATION_ROOT="${REPOSITORY_ROOT}/runs/${EXPERIMENT_ID}/orchestration"

MODE="${1:-}"
case "${MODE}" in
    render) ;;
    gate3|gate4) ;;
    *) printf 'usage: %s render|gate3|gate4\n' "$0" >&2; exit 2; ;;
esac

readonly STATUS_PATH="${ORCHESTRATION_ROOT}/vcd-gate34-${MODE}-001.status"
readonly LOG_PATH="${ORCHESTRATION_ROOT}/vcd-gate34-${MODE}-001.log"
mkdir -p -- "${ORCHESTRATION_ROOT}"

ARTIFACT_ID="m2-smolvla450m-vcdropout-cuda-b64-001-step0237-deploy-001"
ARGS=(--artifact-id "${ARTIFACT_ID}")
if [[ "${MODE}" == "gate4" ]]; then
    GATE3_REPORT="runs/${EXPERIMENT_ID}/gates/gate3-smolvla-sim-433.json"
    [[ -e "${REPOSITORY_ROOT}/${GATE3_REPORT}" ]] \
        || { printf 'error: passed Gate 3 report missing: %s\n' "${GATE3_REPORT}" >&2; exit 4; }
    ARGS+=(--gate3-report "${GATE3_REPORT}")
fi

write_status() {
    local exit_code="$1"
    (
        set -o noclobber
        printf 'state=finished\nexit_code=%s\n' "${exit_code}" >"${STATUS_PATH}"
    )
}

cd -- "${REPOSITORY_ROOT}"
if [[ "${MODE}" == "render" ]]; then
    # Render is idempotent and drift-checked; no create-only status identity.
    ROSETTA_VLA_SIM_XPU_IMAGE=sha256:f4a71c4020cd54d2a878f01628d591af9572f0784458f4c821008f8aea30393c \
    ROSETTA_VLA_DOCKER_MEMORY=6g \
    ROSETTA_CONTAINER_NAME=vcd-gate34-render-001 \
        scripts/run_m2_container.sh vla-sim-xpu \
        python -u scripts/smolvla_vcdropout_sim_gate.py render "${ARGS[@]}"
    exit $?
fi

[[ ! -e "${STATUS_PATH}" && ! -e "${LOG_PATH}" ]] \
    || { printf 'error: vcdropout %s status or log already exists\n' "${MODE}" >&2; exit 3; }

set +e
ROSETTA_VLA_SIM_XPU_IMAGE=sha256:f4a71c4020cd54d2a878f01628d591af9572f0784458f4c821008f8aea30393c \
ROSETTA_VLA_DOCKER_MEMORY=6g \
ROSETTA_CONTAINER_NAME=vcd-gate34-${MODE}-001 \
    scripts/run_m2_container.sh vla-sim-xpu \
    python -u scripts/smolvla_vcdropout_sim_gate.py "${MODE}" "${ARGS[@]}" \
    >"${LOG_PATH}" 2>&1
exit_code="$?"
set -e
write_status "${exit_code}"
exit "${exit_code}"
