#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
readonly EXPERIMENT_ID="${ROSETTA_EXPERIMENT_ID:-m2-qwen08b-frozen-002-spatial}"
readonly CONFIG_RELATIVE="${ROSETTA_EXPERIMENT_CONFIG:-configs/experiments/m2_qwen08b_frozen_002_spatial.yaml}"
readonly CONFIG_PATH="${REPOSITORY_ROOT}/${CONFIG_RELATIVE}"
readonly RUN_ROOT="${ROSETTA_RUN_ROOT:-${REPOSITORY_ROOT}/runs}"
readonly MONITOR_ROOT="${RUN_ROOT}/${EXPERIMENT_ID}/runtime"
readonly STARTED_AT="$(date -u +%Y%m%dT%H%M%SZ)"
readonly RUN_TOKEN="${STARTED_AT}-$$"
readonly CONTAINER_NAME="rosetta-m2-cache-${RUN_TOKEN}"
readonly LOG_RELATIVE="runs/${EXPERIMENT_ID}/runtime/feature-cache-${RUN_TOKEN}.log"
readonly STATUS_RELATIVE="runs/${EXPERIMENT_ID}/runtime/feature-cache-${RUN_TOKEN}.status"
readonly LOG_PATH="${MONITOR_ROOT}/feature-cache-${RUN_TOKEN}.log"
readonly STATUS_PATH="${MONITOR_ROOT}/feature-cache-${RUN_TOKEN}.status"
readonly LOCK_RELATIVE="runs/${EXPERIMENT_ID}/runtime/feature-cache.lock"
readonly LOCK_PATH="${MONITOR_ROOT}/feature-cache.lock"
DOCKER_BIN=()

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
    printf 'state=%s\nexit_code=%s\ncontainer=%s\nlog=%s\nstarted_at=%s\nupdated_at=%s\n' \
        "$state" \
        "$exit_code" \
        "$CONTAINER_NAME" \
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

[[ "$EXPERIMENT_ID" != "." && "$EXPERIMENT_ID" != ".." ]]
[[ "$EXPERIMENT_ID" != *[[:space:]/\\]* ]] \
    || { printf 'error: ROSETTA_EXPERIMENT_ID must be a path-safe token\n' >&2; exit 2; }
[[ "$CONFIG_RELATIVE" != /* && "$CONFIG_RELATIVE" != *".."* ]]
[[ -f "$CONFIG_PATH" ]] \
    || { printf 'error: experiment config is missing\n' >&2; exit 2; }
declared_experiment_id="$(awk '$1 == "experiment_id:" { print $2; exit }' "$CONFIG_PATH")"
[[ "$declared_experiment_id" == "$EXPERIMENT_ID" ]] \
    || { printf 'error: experiment ID and config disagree\n' >&2; exit 2; }
[[ -d "${ROSETTA_MODEL_ROOT:-}" ]] \
    || { printf 'error: ROSETTA_MODEL_ROOT must name the verified local Base model directory\n' >&2; exit 2; }

mkdir -p -- "$MONITOR_ROOT"
exec 9>"$LOCK_PATH"
flock -n 9 \
    || { printf 'error: another feature-cache build holds %s\n' "$LOCK_RELATIVE" >&2; exit 2; }
select_docker
if "${DOCKER_BIN[@]}" inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    printf 'error: container name already exists: %s\n' "$CONTAINER_NAME" >&2
    exit 2
fi

trap handle_signal INT TERM HUP
write_status running pending
printf 'monitor_status=%s\nmonitor_log=%s\ncontainer=%s\n' \
    "$STATUS_RELATIVE" "$LOG_RELATIVE" "$CONTAINER_NAME"

set +e
ROSETTA_CONTAINER_NAME="$CONTAINER_NAME" \
    "${SCRIPT_DIR}/run_m2_container.sh" ml \
    python -u scripts/cache_features.py build \
    --config "/workspace/${CONFIG_RELATIVE}" \
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
