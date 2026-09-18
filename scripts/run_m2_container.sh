#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
readonly ML_IMAGE="${ROSETTA_ML_IMAGE:-rosetta-reality-m2:local}"
readonly ML_XPU_IMAGE="${ROSETTA_ML_XPU_IMAGE:-rosetta-reality-m2-xpu:local}"
readonly VLA_XPU_IMAGE="${ROSETTA_VLA_XPU_IMAGE:-rosetta-reality-smolvla-xpu:local}"
readonly VLA_SIM_XPU_IMAGE="${ROSETTA_VLA_SIM_XPU_IMAGE:-rosetta-reality-smolvla-sim-xpu:local}"
readonly SIM_IMAGE="${ROSETTA_SIM_IMAGE:-rosetta-reality-sim:local}"
readonly SIM_XPU_IMAGE="${ROSETTA_SIM_XPU_IMAGE:-rosetta-reality-sim-xpu:local}"
readonly MEMORY_LIMIT="${ROSETTA_DOCKER_MEMORY:-5g}"
readonly VLA_MEMORY_LIMIT="${ROSETTA_VLA_DOCKER_MEMORY:-6g}"
readonly CPU_LIMIT="${ROSETTA_DOCKER_CPUS:-2}"
readonly PIDS_LIMIT="${ROSETTA_DOCKER_PIDS:-512}"
DOCKER_BIN=()
DOCKER_USES_WINDOWS_PATHS=0

usage() {
    cat <<'EOF'
Usage: scripts/run_m2_container.sh COMMAND [ARG...]

Commands:
  build             Build both pinned Linux images.
  build-ml          Build the Python 3.13 data/Qwen/training image.
  build-ml-xpu      Build the digest-pinned Intel XPU training image.
  build-vla-xpu     Build the pinned LeRobot/SmolVLA/Trackio Intel XPU image.
  build-vla-sim-xpu Build the pinned SmolVLA/Gym-ALOHA Intel XPU image.
  build-sim         Build the Python 3.11 Gym-ALOHA/MuJoCo image.
  build-sim-xpu     Build the Intel XPU Gym-ALOHA/MuJoCo image.
  data ARG...       Run a data-preparation command with network access.
  model ARG...      Run a model-preparation command with network and models/ write access.
  model-adopt ...   Validate an existing local snapshot offline and add a create-only manifest.
  model-inspect ... Run a model inspection command offline with models/ read-only.
  ml ARG...         Run a training/evaluation command offline.
  ml-xpu ARG...     Run an Intel XPU training/evaluation command offline.
  vla-prepare ...   Download/verify SmolVLA dependencies with models/ write access.
  vla-cpu ARG...    Run a SmolVLA diagnostic offline on CPU.
  vla-xpu ARG...    Run SmolVLA training/evaluation offline on Intel XPU.
  vla-sim-xpu ...   Run SmolVLA MuJoCo evaluation offline on Intel XPU.
  trackio-local ... Write/query the durable local Trackio store offline.
  trackio-sync ...  Sync a sanitized snapshot using a read-only HF token file.
  sim ARG...        Run a simulation command offline.
  sim-xpu ARG...    Run a simulation command on Intel XPU offline.
  shell-ml          Open an offline shell in the ML image.
  shell-ml-xpu      Open an offline shell in the Intel XPU image.
  shell-sim         Open an offline shell in the simulation image.
  shell-sim-xpu     Open an offline shell in the Intel XPU simulation image.

Environment:
  ROSETTA_MODEL_ROOT      Local model directory mounted read-only at /model.
  ROSETTA_MODELS_ROOT     Model store for `model` (default: repository models/).
  ROSETTA_HF_TOKEN_FILE   Existing host HF token file mounted read-only for Trackio sync.
  ROSETTA_DATA_ROOT       Dataset cache (default: repository data/lerobot_m2/).
  ROSETTA_FEATURE_ROOT    Feature cache (default: repository feature_cache/).
  ROSETTA_CHECKPOINT_ROOT Checkpoints (default: repository checkpoints/).
  ROSETTA_ARTIFACT_ROOT   Exports (default: repository artifacts/).
  ROSETTA_RUN_ROOT        Logs and metrics (default: repository runs/).
  ROSETTA_DOCKER_MEMORY   Hard memory and memory+swap limit (default: 5g).
  ROSETTA_VLA_DOCKER_MEMORY SmolVLA memory and memory+swap limit (default: 6g).
  ROSETTA_DOCKER_CPUS     CPU quota (default: 2).
  ROSETTA_DOCKER_PIDS     PID limit (default: 512).
  ROSETTA_XPU_DEVICE_PATH WSL GPU bridge device (default: /dev/dxg).
  ROSETTA_WSL_LIB_ROOT    WSL host-library root (default: /usr/lib/wsl).
  ROSETTA_CONTAINER_NAME  Optional explicit Docker container name for monitoring.
  ROSETTA_VLA_CONTAINER_USER Optional numeric uid:gid for VLA artifact filesystem ownership.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

select_docker() {
    local candidate
    local -a candidates=()
    if [[ -n "${ROSETTA_DOCKER_COMMAND:-}" ]]; then
        candidates+=("$ROSETTA_DOCKER_COMMAND")
    fi
    # Prefer the WSL-native Docker CLI. Docker Desktop can leave a stale
    # docker.exe interop shim in PATH after a restart even while the Linux CLI
    # and engine are healthy.
    candidates+=(docker docker.exe)
    for candidate in "${candidates[@]}"; do
        if command -v "$candidate" >/dev/null 2>&1 && "$candidate" info >/dev/null 2>&1; then
            DOCKER_BIN=("$candidate")
            [[ "$candidate" == "docker.exe" ]] && DOCKER_USES_WINDOWS_PATHS=1
            return 0
        fi
    done
    die "Docker Desktop Linux engine is not ready"
}

docker_command() {
    "${DOCKER_BIN[@]}" "$@"
}

image_exists() {
    local image="$1"
    if docker_command image inspect "$image" >/dev/null 2>&1; then
        return 0
    fi
    if [[ "$image" != */* ]] \
        && docker_command image inspect "docker.io/library/$image" >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

docker_host_path() {
    local path="$1"
    if [[ "$DOCKER_USES_WINDOWS_PATHS" == "1" ]]; then
        wslpath -w "$path"
    else
        printf '%s\n' "$path"
    fi
}

ensure_directory() {
    local path="$1"
    mkdir -p -- "$path"
    [[ -d "$path" ]] || die "not a directory: $path"
}

build_image() {
    local image="$1"
    local dockerfile="$2"
    local attempts="${ROSETTA_DOCKER_BUILD_ATTEMPTS:-3}"
    local attempt
    local dockerfile_host
    local repository_host
    dockerfile_host="$(docker_host_path "$dockerfile")"
    repository_host="$(docker_host_path "$REPOSITORY_ROOT")"
    for ((attempt = 1; attempt <= attempts; attempt++)); do
        if docker_command build \
            --memory "$MEMORY_LIMIT" \
            --file "$dockerfile_host" \
            --tag "$image" \
            "$repository_host"; then
            return
        fi
        printf 'build attempt %d/%d failed for %s\n' "$attempt" "$attempts" "$image" >&2
    done
    die "Docker image build failed after $attempts attempts: $image"
}

base_run_args() {
    local network="$1"
    local memory_limit="${2:-$MEMORY_LIMIT}"
    local memory_swap_limit="${3:-$memory_limit}"
    local repository_host
    repository_host="$(docker_host_path "$REPOSITORY_ROOT")"
    printf '%s\0' \
        --rm \
        --init \
        --read-only \
        --cap-drop=ALL \
        --security-opt=no-new-privileges \
        --memory="$memory_limit" \
        --memory-swap="$memory_swap_limit" \
        --cpus="$CPU_LIMIT" \
        --pids-limit="$PIDS_LIMIT" \
        --shm-size=256m \
        --network="$network" \
        --tmpfs=/tmp:rw,nosuid,nodev,noexec,size=1g \
        --workdir=/workspace \
        --env=HOME=/tmp/home \
        --env=HF_HUB_DISABLE_TELEMETRY=1 \
        --env=TOKENIZERS_PARALLELISM=false \
        --env=PYTHONPYCACHEPREFIX=/tmp/pycache \
        --env=RUFF_CACHE_DIR=/tmp/ruff-cache \
        --env=XDG_CACHE_HOME=/tmp/cache \
        --env=OMP_NUM_THREADS=2 \
        --env=MKL_NUM_THREADS=2 \
        --volume="$repository_host:/workspace:ro"
    if [[ -n "${ROSETTA_CONTAINER_NAME:-}" ]]; then
        [[ "$ROSETTA_CONTAINER_NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]+$ ]] \
            || die "invalid ROSETTA_CONTAINER_NAME"
        printf '%s\0' --name="$ROSETTA_CONTAINER_NAME"
    fi
}

run_model_prepare() {
    local models_root="${ROSETTA_MODELS_ROOT:-${REPOSITORY_ROOT}/models}"
    local models_host
    local -a args=()

    ensure_directory "$models_root"
    models_host="$(docker_host_path "$models_root")"
    while IFS= read -r -d '' argument; do
        args+=("$argument")
    done < <(base_run_args bridge)
    args+=(
        --volume="$models_host:/workspace/models:rw"
        --env=ROSETTA_MODELS_ROOT=/workspace/models
        --env=HF_HUB_DISABLE_XET=1
        --env=HF_HUB_ETAG_TIMEOUT=30
        --env=HF_HUB_DOWNLOAD_TIMEOUT=120
    )
    image_exists "$ML_IMAGE" \
        || die "image is missing; run build-ml first"
    docker_command run "${args[@]}" "$ML_IMAGE" "$@"
}

run_model_inspect() {
    local models_root="${ROSETTA_MODELS_ROOT:-${REPOSITORY_ROOT}/models}"
    local models_host
    local -a args=()

    [[ -d "$models_root" ]] || die "models root is missing: $models_root"
    models_host="$(docker_host_path "$models_root")"
    while IFS= read -r -d '' argument; do
        args+=("$argument")
    done < <(base_run_args none)
    args+=(
        --volume="$models_host:/workspace/models:ro"
        --env=ROSETTA_MODELS_ROOT=/workspace/models
        --env=HF_HUB_OFFLINE=1
    )
    image_exists "$ML_IMAGE" \
        || die "image is missing; run build-ml first"
    docker_command run "${args[@]}" "$ML_IMAGE" "$@"
}

run_model_adopt() {
    local model_root="${ROSETTA_MODEL_ROOT:-}"
    local model_host
    local -a args=()

    [[ -n "$model_root" && -d "$model_root" ]] \
        || die "ROSETTA_MODEL_ROOT must identify the existing snapshot to adopt"
    model_host="$(docker_host_path "$model_root")"
    while IFS= read -r -d '' argument; do
        args+=("$argument")
    done < <(base_run_args none)
    args+=(
        --user="$(id -u):$(id -g)"
        --volume="$model_host:/model:rw"
        --env=ROSETTA_MODEL_ROOT=/model
        --env=HF_HUB_OFFLINE=1
    )
    image_exists "$ML_IMAGE" \
        || die "image is missing; run build-ml first"
    docker_command run "${args[@]}" "$ML_IMAGE" "$@"
}

run_container() {
    local image="$1"
    local network="$2"
    local accelerator="$3"
    shift 3

    local data_root="${ROSETTA_DATA_ROOT:-${REPOSITORY_ROOT}/data/lerobot_m2}"
    local feature_root="${ROSETTA_FEATURE_ROOT:-${REPOSITORY_ROOT}/feature_cache}"
    local checkpoint_root="${ROSETTA_CHECKPOINT_ROOT:-${REPOSITORY_ROOT}/checkpoints}"
    local artifact_root="${ROSETTA_ARTIFACT_ROOT:-${REPOSITORY_ROOT}/artifacts}"
    local run_root="${ROSETTA_RUN_ROOT:-${REPOSITORY_ROOT}/runs}"
    local data_host
    local feature_host
    local checkpoint_host
    local artifact_host
    local run_host
    local -a args=()

    ensure_directory "$data_root"
    ensure_directory "$feature_root"
    ensure_directory "$checkpoint_root"
    ensure_directory "$artifact_root"
    ensure_directory "$run_root"
    data_host="$(docker_host_path "$data_root")"
    feature_host="$(docker_host_path "$feature_root")"
    checkpoint_host="$(docker_host_path "$checkpoint_root")"
    artifact_host="$(docker_host_path "$artifact_root")"
    run_host="$(docker_host_path "$run_root")"

    while IFS= read -r -d '' argument; do
        args+=("$argument")
    done < <(base_run_args "$network")

    args+=(
        --volume="$data_host:/workspace/data:rw"
        --volume="$feature_host:/workspace/feature_cache:rw"
        --volume="$checkpoint_host:/workspace/checkpoints:rw"
        --volume="$artifact_host:/workspace/artifacts:rw"
        --volume="$run_host:/workspace/runs:rw"
        --env=ROSETTA_DATA_ROOT=/workspace/data
        --env=ROSETTA_FEATURE_ROOT=/workspace/feature_cache
        --env=ROSETTA_CHECKPOINT_ROOT=/workspace/checkpoints
        --env=ROSETTA_ARTIFACT_ROOT=/workspace/artifacts
        --env=ROSETTA_RUN_ROOT=/workspace/runs
    )

    if [[ -n "${ROSETTA_MODEL_ROOT:-}" ]]; then
        local model_host
        [[ -d "$ROSETTA_MODEL_ROOT" ]] || die "ROSETTA_MODEL_ROOT is not a directory"
        model_host="$(docker_host_path "$ROSETTA_MODEL_ROOT")"
        args+=(--volume="$model_host:/model:ro" --env=ROSETTA_MODEL_ROOT=/model)
    fi

    if [[ "$network" == "none" ]]; then
        args+=(--env=HF_HUB_OFFLINE=1 --env=HF_DATASETS_OFFLINE=1)
    fi

    if [[ "$accelerator" == "xpu" ]]; then
        local xpu_device="${ROSETTA_XPU_DEVICE_PATH:-/dev/dxg}"
        local wsl_lib_root="${ROSETTA_WSL_LIB_ROOT:-/usr/lib/wsl}"
        [[ -c "$xpu_device" ]] || die "XPU bridge is not a character device: $xpu_device"
        [[ -d "$wsl_lib_root/lib" ]] || die "WSL GPU libraries are missing: $wsl_lib_root/lib"
        args+=(
            --device="$xpu_device"
            --volume="$wsl_lib_root:$wsl_lib_root:ro"
            --env="LD_LIBRARY_PATH=$wsl_lib_root/lib"
            --env=ROSETTA_TORCH_DEVICE=xpu
        )
    elif [[ "$accelerator" != "cpu" ]]; then
        die "unsupported accelerator: $accelerator"
    fi

    image_exists "$image" \
        || die "image is missing; run the matching build command first: $image"
    docker_command run "${args[@]}" "$image" "$@"
}

run_vla_container() {
    local image="$1"
    local network="$2"
    local accelerator="$3"
    shift 3

    local data_root="${ROSETTA_DATA_ROOT:-${REPOSITORY_ROOT}/data/lerobot_m2}"
    local models_root="${ROSETTA_MODELS_ROOT:-${REPOSITORY_ROOT}/models}"
    local hf_home_root="${models_root}/hf_home"
    local checkpoint_root="${ROSETTA_CHECKPOINT_ROOT:-${REPOSITORY_ROOT}/checkpoints}"
    local artifact_root="${ROSETTA_ARTIFACT_ROOT:-${REPOSITORY_ROOT}/artifacts}"
    local run_root="${ROSETTA_RUN_ROOT:-${REPOSITORY_ROOT}/runs}"
    local data_host models_host hf_home_host checkpoint_host artifact_host run_host
    local -a args=()

    ensure_directory "$data_root"
    ensure_directory "$models_root"
    ensure_directory "$hf_home_root"
    ensure_directory "$checkpoint_root"
    ensure_directory "$artifact_root"
    ensure_directory "$run_root"
    data_host="$(docker_host_path "$data_root")"
    models_host="$(docker_host_path "$models_root")"
    hf_home_host="$(docker_host_path "$hf_home_root")"
    checkpoint_host="$(docker_host_path "$checkpoint_root")"
    artifact_host="$(docker_host_path "$artifact_root")"
    run_host="$(docker_host_path "$run_root")"
    while IFS= read -r -d '' argument; do
        args+=("$argument")
    done < <(base_run_args "$network" "$VLA_MEMORY_LIMIT" "$VLA_MEMORY_LIMIT")
    if [[ -n "${ROSETTA_VLA_CONTAINER_USER:-}" ]]; then
        [[ "$ROSETTA_VLA_CONTAINER_USER" =~ ^[0-9]+:[0-9]+$ ]] \
            || die "ROSETTA_VLA_CONTAINER_USER must be a numeric uid:gid"
        args+=(--user="$ROSETTA_VLA_CONTAINER_USER")
    fi
    args+=(
        --volume="$data_host:/workspace/data:ro"
        --volume="$models_host:/workspace/models:ro"
        --volume="$hf_home_host:/workspace/models/hf_home:rw"
        --volume="$checkpoint_host:/workspace/checkpoints:rw"
        --volume="$artifact_host:/workspace/artifacts:rw"
        --volume="$run_host:/workspace/runs:rw"
        --env=ROSETTA_DATA_ROOT=/workspace/data
        --env=ROSETTA_MODELS_ROOT=/workspace/models
        --env=ROSETTA_CHECKPOINT_ROOT=/workspace/checkpoints
        --env=ROSETTA_ARTIFACT_ROOT=/workspace/artifacts
        --env=ROSETTA_RUN_ROOT=/workspace/runs
        --env=TRACKIO_DIR=/workspace/runs/trackio
        --env=HF_HOME=/workspace/models/hf_home
        --env=ROSETTA_DOCKER_MEMORY_LIMIT="$VLA_MEMORY_LIMIT"
        --env=ROSETTA_DOCKER_MEMORY_SWAP_LIMIT="$VLA_MEMORY_LIMIT"
    )
    if [[ "$network" == "none" ]]; then
        args+=(--env=HF_HUB_OFFLINE=1 --env=HF_DATASETS_OFFLINE=1)
    fi
    if [[ "$accelerator" == "xpu" ]]; then
        local xpu_device="${ROSETTA_XPU_DEVICE_PATH:-/dev/dxg}"
        local wsl_lib_root="${ROSETTA_WSL_LIB_ROOT:-/usr/lib/wsl}"
        [[ -c "$xpu_device" ]] || die "XPU bridge is not a character device: $xpu_device"
        [[ -d "$wsl_lib_root/lib" ]] || die "WSL GPU libraries are missing: $wsl_lib_root/lib"
        args+=(
            --device="$xpu_device"
            --volume="$wsl_lib_root:$wsl_lib_root:ro"
            --env="LD_LIBRARY_PATH=$wsl_lib_root/lib"
            --env=ROSETTA_TORCH_DEVICE=xpu
        )
    elif [[ "$accelerator" == "cpu" ]]; then
        args+=(--env=ROSETTA_TORCH_DEVICE=cpu)
    else
        die "unsupported VLA accelerator: $accelerator"
    fi
    image_exists "$image" || die "the requested VLA image is missing"
    local image_id
    image_id="$(docker_command image inspect --format '{{.Id}}' "$image")"
    [[ "$image_id" == sha256:* ]] || die "the requested VLA image has no immutable image id"
    args+=(--env=ROSETTA_CONTAINER_IMAGE_ID="$image_id")
    docker_command run "${args[@]}" "$image" "$@"
}

run_vla_prepare() {
    local data_root="${ROSETTA_DATA_ROOT:-${REPOSITORY_ROOT}/data/lerobot_m2}"
    local models_root="${ROSETTA_MODELS_ROOT:-${REPOSITORY_ROOT}/models}"
    local data_host models_host
    local -a args=()

    ensure_directory "$data_root"
    ensure_directory "$models_root"
    data_host="$(docker_host_path "$data_root")"
    models_host="$(docker_host_path "$models_root")"
    while IFS= read -r -d '' argument; do
        args+=("$argument")
    done < <(base_run_args bridge)
    args+=(
        --volume="$data_host:/workspace/data:ro"
        --volume="$models_host:/workspace/models:rw"
        --env=ROSETTA_DATA_ROOT=/workspace/data
        --env=ROSETTA_MODELS_ROOT=/workspace/models
        --env=HF_HOME=/workspace/models/hf_home
        --env=HF_HUB_DISABLE_XET=1
        --env=HF_HUB_ETAG_TIMEOUT=30
        --env=HF_HUB_DOWNLOAD_TIMEOUT=120
        --env=ROSETTA_TORCH_DEVICE=cpu
    )
    image_exists "$VLA_XPU_IMAGE" || die "image is missing; run build-vla-xpu first"
    docker_command run "${args[@]}" "$VLA_XPU_IMAGE" "$@"
}

run_trackio_local() {
    local run_root="${ROSETTA_RUN_ROOT:-${REPOSITORY_ROOT}/runs}"
    local run_host
    local -a args=()

    ensure_directory "$run_root"
    run_host="$(docker_host_path "$run_root")"
    while IFS= read -r -d '' argument; do
        args+=("$argument")
    done < <(base_run_args none)
    args+=(
        --volume="$run_host:/workspace/runs:rw"
        --env=ROSETTA_RUN_ROOT=/workspace/runs
        --env=TRACKIO_DIR=/workspace/runs/trackio
    )
    image_exists "$VLA_XPU_IMAGE" || die "image is missing; run build-vla-xpu first"
    docker_command run "${args[@]}" "$VLA_XPU_IMAGE" "$@"
}

run_trackio_sync() {
    local token_file="${ROSETTA_HF_TOKEN_FILE:-}"
    local run_root="${ROSETTA_RUN_ROOT:-${REPOSITORY_ROOT}/runs}"
    local token_host run_host
    local -a args=()

    [[ -n "$token_file" && -f "$token_file" ]] \
        || die "ROSETTA_HF_TOKEN_FILE must identify the existing HF CLI token file"
    ensure_directory "$run_root"
    token_host="$(docker_host_path "$token_file")"
    run_host="$(docker_host_path "$run_root")"
    while IFS= read -r -d '' argument; do
        args+=("$argument")
    done < <(base_run_args bridge)
    args+=(
        --volume="$run_host:/workspace/runs:rw"
        --mount="type=bind,src=$token_host,dst=/tmp/home/.cache/huggingface/token,readonly"
        --env=ROSETTA_RUN_ROOT=/workspace/runs
        --env=TRACKIO_DIR=/workspace/runs/trackio
        --env=HF_TOKEN_PATH=/tmp/home/.cache/huggingface/token
    )
    image_exists "$VLA_XPU_IMAGE" || die "image is missing; run build-vla-xpu first"
    docker_command run "${args[@]}" "$VLA_XPU_IMAGE" "$@"
}

select_docker

command_name="${1:-}"
[[ -n "$command_name" ]] || { usage; exit 2; }
shift

case "$command_name" in
    build)
        build_image "$ML_IMAGE" "$REPOSITORY_ROOT/docker/Dockerfile.m2"
        build_image "$SIM_IMAGE" "$REPOSITORY_ROOT/docker/Dockerfile.sim"
        ;;
    build-ml)
        build_image "$ML_IMAGE" "$REPOSITORY_ROOT/docker/Dockerfile.m2"
        ;;
    build-ml-xpu)
        build_image "$ML_XPU_IMAGE" "$REPOSITORY_ROOT/docker/Dockerfile.m2-xpu"
        ;;
    build-vla-xpu)
        build_image "$VLA_XPU_IMAGE" "$REPOSITORY_ROOT/docker/Dockerfile.smolvla-xpu"
        ;;
    build-vla-sim-xpu)
        build_image "$VLA_SIM_XPU_IMAGE" "$REPOSITORY_ROOT/docker/Dockerfile.smolvla-sim-xpu"
        ;;
    build-sim)
        build_image "$SIM_IMAGE" "$REPOSITORY_ROOT/docker/Dockerfile.sim"
        ;;
    build-sim-xpu)
        build_image "$SIM_XPU_IMAGE" "$REPOSITORY_ROOT/docker/Dockerfile.sim-xpu"
        ;;
    data)
        [[ "$#" -gt 0 ]] || die "data requires a command"
        run_container "$ML_IMAGE" bridge cpu "$@"
        ;;
    model)
        [[ "$#" -gt 0 ]] || die "model requires a command"
        run_model_prepare "$@"
        ;;
    model-adopt)
        [[ "$#" -gt 0 ]] || die "model-adopt requires a command"
        run_model_adopt "$@"
        ;;
    model-inspect)
        [[ "$#" -gt 0 ]] || die "model-inspect requires a command"
        run_model_inspect "$@"
        ;;
    ml)
        [[ "$#" -gt 0 ]] || die "ml requires a command"
        run_container "$ML_IMAGE" none cpu "$@"
        ;;
    ml-xpu)
        [[ "$#" -gt 0 ]] || die "ml-xpu requires a command"
        run_container "$ML_XPU_IMAGE" none xpu "$@"
        ;;
    vla-prepare)
        [[ "$#" -gt 0 ]] || die "vla-prepare requires a command"
        run_vla_prepare "$@"
        ;;
    vla-cpu)
        [[ "$#" -gt 0 ]] || die "vla-cpu requires a command"
        run_vla_container "$VLA_XPU_IMAGE" none cpu "$@"
        ;;
    vla-xpu)
        [[ "$#" -gt 0 ]] || die "vla-xpu requires a command"
        run_vla_container "$VLA_XPU_IMAGE" none xpu "$@"
        ;;
    vla-sim-xpu)
        [[ "$#" -gt 0 ]] || die "vla-sim-xpu requires a command"
        run_vla_container "$VLA_SIM_XPU_IMAGE" none xpu "$@"
        ;;
    trackio-local)
        [[ "$#" -gt 0 ]] || die "trackio-local requires a command"
        run_trackio_local "$@"
        ;;
    trackio-sync)
        [[ "$#" -gt 0 ]] || die "trackio-sync requires a command"
        run_trackio_sync "$@"
        ;;
    sim)
        [[ "$#" -gt 0 ]] || die "sim requires a command"
        run_container "$SIM_IMAGE" none cpu "$@"
        ;;
    sim-xpu)
        [[ "$#" -gt 0 ]] || die "sim-xpu requires a command"
        run_container "$SIM_XPU_IMAGE" none xpu "$@"
        ;;
    shell-ml)
        run_container "$ML_IMAGE" none cpu bash
        ;;
    shell-ml-xpu)
        run_container "$ML_XPU_IMAGE" none xpu bash
        ;;
    shell-sim)
        run_container "$SIM_IMAGE" none cpu bash
        ;;
    shell-sim-xpu)
        run_container "$SIM_XPU_IMAGE" none xpu bash
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage >&2
        die "unknown command: $command_name"
        ;;
esac
