#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
readonly EXPERIMENT_ID="${ROSETTA_EXPERIMENT_ID:-m2-qwen08b-frozen-001}"
readonly CONFIG_RELATIVE="${ROSETTA_EXPERIMENT_CONFIG:-configs/experiments/m2_qwen08b_frozen_001.yaml}"
readonly ARTIFACT_ID="${ROSETTA_ARTIFACT_ID:-m2-qwen08b-frozen-001-base-dc7cdfe2}"
readonly ACCELERATOR="${ROSETTA_GATE4_ACCELERATOR:-cpu}"
readonly MAXIMUM_STEPS="${ROSETTA_GATE4_MAXIMUM_STEPS:-500}"
readonly MINIMUM_SUCCESS_RATE="${ROSETTA_GATE4_MINIMUM_SUCCESS_RATE:-0.2}"
readonly MAXIMUM_UNEXPECTED_COLLISIONS="${ROSETTA_GATE4_MAXIMUM_UNEXPECTED_COLLISIONS:-0}"
readonly SEEDS_TEXT="${ROSETTA_GATE4_SEEDS:-1000 1001 1002 1003 1004}"
readonly RUN_ROOT="${ROSETTA_RUN_ROOT:-${REPOSITORY_ROOT}/runs}"
readonly MONITOR_ROOT="${RUN_ROOT}/runtime/${EXPERIMENT_ID}"
readonly STARTED_AT="$(date -u +%Y%m%dT%H%M%SZ)"
readonly RUN_TOKEN="${STARTED_AT}-$$"
readonly CONTAINER_NAME="rosetta-m2-gate4-${RUN_TOKEN}"
readonly LOG_RELATIVE="runs/runtime/${EXPERIMENT_ID}/gate4-${RUN_TOKEN}.log"
readonly STATUS_RELATIVE="runs/runtime/${EXPERIMENT_ID}/gate4-${RUN_TOKEN}.status"
readonly LOG_PATH="${MONITOR_ROOT}/gate4-${RUN_TOKEN}.log"
readonly STATUS_PATH="${MONITOR_ROOT}/gate4-${RUN_TOKEN}.status"
CONFIG_PATH=""
SIM_COMMAND=""
DOCKER_BIN=()
GATE4_SEEDS=()

select_docker() {
    local candidate
    local -a candidates=()
    if [[ -n "${ROSETTA_DOCKER_COMMAND:-}" ]]; then
        candidates+=("$ROSETTA_DOCKER_COMMAND")
    fi
    candidates+=(docker.exe docker)
    for candidate in "${candidates[@]}"; do
        if command -v "$candidate" >/dev/null 2>&1 && "$candidate" info >/dev/null 2>&1; then
            DOCKER_BIN=("$candidate")
            return
        fi
    done
    printf 'error: Docker Desktop Linux engine is not ready\n' >&2
    exit 2
}

container_running() {
    "${DOCKER_BIN[@]}" inspect --format '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null \
        | grep -qx true
}

stop_owned_container() {
    if container_running; then
        "${DOCKER_BIN[@]}" stop --timeout 10 "$CONTAINER_NAME" >/dev/null || true
    fi
}

write_status() {
    local state="$1"
    local exit_code="$2"
    printf 'state=%s\nexit_code=%s\ncontainer=%s\naccelerator=%s\nlog=%s\nstarted_at=%s\nupdated_at=%s\n' \
        "$state" \
        "$exit_code" \
        "$CONTAINER_NAME" \
        "$ACCELERATOR" \
        "$LOG_RELATIVE" \
        "$STARTED_AT" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        >"$STATUS_PATH"
}

handle_signal() {
    write_status interrupted 130
    stop_owned_container
    exit 130
}

[[ -d "${ROSETTA_MODEL_ROOT:-}" ]] \
    || { printf 'error: ROSETTA_MODEL_ROOT must name the verified local Base model directory\n' >&2; exit 2; }
case "$ACCELERATOR" in
    cpu) SIM_COMMAND=sim ;;
    xpu) SIM_COMMAND=sim-xpu ;;
    *)
        printf 'error: ROSETTA_GATE4_ACCELERATOR must be cpu or xpu\n' >&2
        exit 2
        ;;
esac
[[ "$CONFIG_RELATIVE" != /* && "$CONFIG_RELATIVE" != *'\'* ]] \
    || { printf 'error: ROSETTA_EXPERIMENT_CONFIG must be repository-relative\n' >&2; exit 2; }
case "/${CONFIG_RELATIVE}/" in
    */../*)
        printf 'error: ROSETTA_EXPERIMENT_CONFIG must not contain .. path components\n' >&2
        exit 2
        ;;
esac
CONFIG_PATH="$(realpath -m -- "${REPOSITORY_ROOT}/${CONFIG_RELATIVE}")"
[[ "$CONFIG_PATH" == "${REPOSITORY_ROOT}/"* && -f "$CONFIG_PATH" ]] \
    || { printf 'error: verified experiment config is missing or outside the repository\n' >&2; exit 2; }
config_experiment_id="$(sed -nE 's/^[[:space:]]*experiment_id:[[:space:]]*([^[:space:]#]+).*$/\1/p' "$CONFIG_PATH" | head -n 1)"
[[ "$config_experiment_id" == "$EXPERIMENT_ID" ]] \
    || { printf 'error: experiment config identity does not match ROSETTA_EXPERIMENT_ID\n' >&2; exit 2; }
[[ "$MAXIMUM_STEPS" =~ ^[1-9][0-9]*$ ]] \
    || { printf 'error: ROSETTA_GATE4_MAXIMUM_STEPS must be a positive integer\n' >&2; exit 2; }
[[ "$MAXIMUM_UNEXPECTED_COLLISIONS" =~ ^[0-9]+$ ]] \
    || { printf 'error: ROSETTA_GATE4_MAXIMUM_UNEXPECTED_COLLISIONS must be non-negative\n' >&2; exit 2; }
[[ -d "${REPOSITORY_ROOT}/artifacts/${EXPERIMENT_ID}/${ARTIFACT_ID}" ]] \
    || { printf 'error: verified artifact directory is missing\n' >&2; exit 2; }
read -r -a GATE4_SEEDS <<<"$SEEDS_TEXT"
[[ "${#GATE4_SEEDS[@]}" -gt 0 ]] \
    || { printf 'error: ROSETTA_GATE4_SEEDS must contain at least one seed\n' >&2; exit 2; }
for seed in "${GATE4_SEEDS[@]}"; do
    [[ "$seed" =~ ^-?[0-9]+$ ]] \
        || { printf 'error: every Gate 4 seed must be an integer\n' >&2; exit 2; }
done

mkdir -p -- "$MONITOR_ROOT"
select_docker
if "${DOCKER_BIN[@]}" inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    printf 'error: container name already exists: %s\n' "$CONTAINER_NAME" >&2
    exit 2
fi

trap handle_signal INT TERM HUP
write_status running pending
printf 'monitor_status=%s\nmonitor_log=%s\ncontainer=%s\naccelerator=%s\n' \
    "$STATUS_RELATIVE" "$LOG_RELATIVE" "$CONTAINER_NAME" "$ACCELERATOR"

set +e
ROSETTA_CONTAINER_NAME="$CONTAINER_NAME" \
    "${SCRIPT_DIR}/run_m2_container.sh" "$SIM_COMMAND" \
    python -u scripts/sim_gate.py task-eval \
    --config "/workspace/${CONFIG_RELATIVE}" \
    --artifact "/workspace/artifacts/${EXPERIMENT_ID}/${ARTIFACT_ID}" \
    --maximum-steps "$MAXIMUM_STEPS" \
    --seeds "${GATE4_SEEDS[@]}" \
    --minimum-task-success-rate "$MINIMUM_SUCCESS_RATE" \
    --maximum-unexpected-collisions "$MAXIMUM_UNEXPECTED_COLLISIONS" \
    2>&1 | tee "$LOG_PATH"
pipeline_status=("${PIPESTATUS[@]}")
set -e

exit_code="${pipeline_status[0]}"
if [[ "$exit_code" == "0" ]]; then
    write_status completed "$exit_code"
else
    write_status failed "$exit_code"
fi
exit "$exit_code"
