#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
readonly PLATFORM_ROOT="${AUTODL_TMP:-/root/autodl-tmp}"
readonly DURABLE_ROOT="${ROSETTA_AUTODL_ROOT:-${PLATFORM_ROOT}/rosetta}"
readonly PROFILE="${ROSETTA_AUTODL_PROFILE:-${REPOSITORY_ROOT}/configs/runtime/autodl_rtx4090.yaml}"
readonly EXPERIMENT_CONFIG="${ROSETTA_VLA_EXPERIMENT_CONFIG:-${REPOSITORY_ROOT}/configs/vla/smolvla_450m_aloha_insertion_action_repair_bounded_gripper_003.yaml}"

usage() {
    cat <<'EOF'
Usage: scripts/run_autodl.sh COMMAND [ARG...]

Commands:
  init-dirs              Create only the durable AutoDL directory skeleton.
  doctor                 Verify RTX 4090, packages and immutable caches offline.
  benchmark              Run immutable pre-training baselines; no model weights/optimizer.
  preflight ARG...       Run an explicitly supplied no-optimizer command after doctor+benchmark.
  smoke ARG...           Run the plan-bound two-step Way CUDA smoke; formal remains locked.
  gate ARG...            Run a preregistered Way Gate 3/4 after artifact backup.
  shell                  Open an offline shell with the registered durable roots.
  formal ARG...          Run only an explicit, separately preregistered Way formal plan.

The AutoDL container instance is the Linux container boundary. This runner never
tries to start nested Docker. Model/dataset download and Trackio public sync are
not provided by this offline runner.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

init_dirs() {
    [[ "$(uname -s)" == "Linux" ]] || die "AutoDL runner requires Linux"
    [[ -d "$PLATFORM_ROOT" ]] || die "AutoDL data disk is missing: $PLATFORM_ROOT"
    mkdir -p -- \
        "$DURABLE_ROOT/data" \
        "$DURABLE_ROOT/models/hf_home" \
        "$DURABLE_ROOT/checkpoints" \
        "$DURABLE_ROOT/artifacts" \
        "$DURABLE_ROOT/runs/trackio" \
        "$DURABLE_ROOT/envs"
}

export_runtime() {
    init_dirs
    export ROSETTA_AUTODL_PLATFORM_ROOT="$PLATFORM_ROOT"
    export ROSETTA_AUTODL_ROOT="$DURABLE_ROOT"
    export ROSETTA_DATA_ROOT="$DURABLE_ROOT/data"
    export ROSETTA_MODELS_ROOT="$DURABLE_ROOT/models"
    export ROSETTA_CHECKPOINT_ROOT="$DURABLE_ROOT/checkpoints"
    export ROSETTA_ARTIFACT_ROOT="$DURABLE_ROOT/artifacts"
    export ROSETTA_RUN_ROOT="$DURABLE_ROOT/runs"
    export TRACKIO_DIR="$DURABLE_ROOT/runs/trackio"
    export HF_HOME="$DURABLE_ROOT/models/hf_home"
    export HF_HUB_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export HF_HUB_DISABLE_TELEMETRY=1
    export TOKENIZERS_PARALLELISM=false
    export ROSETTA_TORCH_DEVICE=cuda
    export ROSETTA_DOCKER_MEMORY_LIMIT=autodl_platform_container
    export ROSETTA_DOCKER_MEMORY_SWAP_LIMIT=autodl_platform_container
    export MUJOCO_GL=egl
    export ROSETTA_AUTODL_RUNTIME_PROFILE="$PROFILE"
    export PYTHONPYCACHEPREFIX="$DURABLE_ROOT/runs/pycache"
    export TORCHINDUCTOR_CACHE_DIR="$DURABLE_ROOT/runs/compiler_cache/cuda/inductor"
    export TRITON_CACHE_DIR="$DURABLE_ROOT/runs/compiler_cache/cuda/triton"
    mkdir -p -- "$PYTHONPYCACHEPREFIX" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
}

run_doctor() {
    python scripts/autodl_doctor_cuda.py --profile "$PROFILE" --config "$EXPERIMENT_CONFIG"
}

run_benchmark() {
    run_doctor
    python scripts/benchmark_smolvla.py --config "$EXPERIMENT_CONFIG"
}

command="${1:-}"
[[ -n "$command" ]] || { usage; exit 2; }
shift

case "$command" in
    init-dirs)
        init_dirs
        ;;
    doctor)
        export_runtime
        run_doctor "$@"
        ;;
    benchmark)
        export_runtime
        run_benchmark "$@"
        ;;
    preflight)
        [[ "$#" -gt 0 ]] || die "preflight requires an explicit no-optimizer command"
        export_runtime
        run_benchmark
        export ROSETTA_AUTODL_NO_OPTIMIZER_AUTHORIZED=1
        "$@"
        ;;
    smoke)
        [[ "$#" -gt 0 ]] || die "smoke requires explicit plan and evidence arguments"
        export_runtime
        export ROSETTA_AUTODL_TWO_STEP_SMOKE_AUTHORIZED=1
        python scripts/run_smolvla_state_robustness_cuda_smoke.py "$@"
        ;;
    gate)
        [[ "$#" -gt 0 ]] || die "gate requires gate3/gate4 and an explicit plan"
        export_runtime
        [[ "${ROSETTA_AUTODL_ARTIFACT_BACKUP_VERIFIED:-}" == "1" ]] \
            || die "gate requires verified deploy-artifact backup evidence"
        python scripts/smolvla_autodl_way_sim_gate_runtime_repair.py "$@"
        ;;
    shell)
        export_runtime
        exec bash --noprofile --norc
        ;;
    formal)
        [[ "$#" -gt 0 ]] || die "formal requires an explicit plan and evidence arguments"
        explicit_plan=0
        for argument in "$@"; do
            [[ "$argument" == "--plan" ]] && explicit_plan=1
        done
        [[ "$explicit_plan" == "1" ]] \
            || die "formal requires an explicit separately preregistered --plan"
        export_runtime
        export ROSETTA_AUTODL_FORMAL_AUTHORIZED=1
        python scripts/run_smolvla_state_robustness_cuda_formal.py "$@"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage >&2
        die "unknown command: $command"
        ;;
esac
