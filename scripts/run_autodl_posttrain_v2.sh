#!/usr/bin/env bash
set -euo pipefail

readonly REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PROFILE="${ROSETTA_AUTODL_RUNTIME_PROFILE:-$REPOSITORY_ROOT/configs/runtime/autodl_rtx4090.yaml}"
readonly DURABLE_ROOT="${ROSETTA_AUTODL_ROOT:-${AUTODL_TMP:-/root/autodl-tmp}/rosetta}"

die() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

usage() {
    cat <<'EOF'
Usage: bash scripts/run_autodl_posttrain_v2.sh COMMAND ARG...

Commands:
  validate    Run future Way validation through the shared compatibility boundary.
  export      Run future Way export through the shared compatibility boundary.
  gate        Run future Way Gate 3/4 after durable artifact-backup verification.

This runner is for newly registered post-Way plans. It does not authorize training,
download assets, or alter completed Way evidence.
EOF
}

export_runtime() {
    [[ "$DURABLE_ROOT" = /* ]] || die "ROSETTA_AUTODL_ROOT must be absolute"
    [[ -d "$DURABLE_ROOT" ]] || die "durable AutoDL root is missing"
    export ROSETTA_AUTODL_ROOT="$DURABLE_ROOT"
    export ROSETTA_RUN_ROOT="$DURABLE_ROOT/runs"
    export ROSETTA_ARTIFACT_ROOT="$DURABLE_ROOT/artifacts"
    export ROSETTA_CHECKPOINT_ROOT="$DURABLE_ROOT/checkpoints"
    export ROSETTA_MODELS_ROOT="$DURABLE_ROOT/models"
    export ROSETTA_DATA_ROOT="$DURABLE_ROOT/data"
    export HF_HOME="$DURABLE_ROOT/models/hf_home"
    export HF_HUB_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_HUB_DISABLE_TELEMETRY=1
    export TOKENIZERS_PARALLELISM=false
    export ROSETTA_TORCH_DEVICE=cuda
    export MUJOCO_GL=egl
    export ROSETTA_AUTODL_RUNTIME_PROFILE="$PROFILE"
    export PYTHONPYCACHEPREFIX="$DURABLE_ROOT/runs/pycache"
    export TORCHINDUCTOR_CACHE_DIR="$DURABLE_ROOT/runs/compiler_cache/cuda/inductor"
    export TRITON_CACHE_DIR="$DURABLE_ROOT/runs/compiler_cache/cuda/triton"
    mkdir -p \
        "$ROSETTA_RUN_ROOT" \
        "$ROSETTA_ARTIFACT_ROOT" \
        "$ROSETTA_CHECKPOINT_ROOT" \
        "$PYTHONPYCACHEPREFIX" \
        "$TORCHINDUCTOR_CACHE_DIR" \
        "$TRITON_CACHE_DIR"
}

command="${1:-help}"
if [[ "$#" -gt 0 ]]; then
    shift
fi
cd "$REPOSITORY_ROOT"

case "$command" in
    validate)
        [[ "$#" -gt 0 ]] || die "validate requires an explicit formal plan"
        export_runtime
        python scripts/evaluate_smolvla_way_validation_v2.py "$@"
        ;;
    export)
        [[ "$#" -gt 0 ]] || die "export requires explicit selection and formal plan arguments"
        export_runtime
        python scripts/export_smolvla_way_v2.py "$@"
        ;;
    gate)
        [[ "$#" -gt 0 ]] || die "gate requires gate3/gate4 and an explicit plan"
        export_runtime
        [[ "${ROSETTA_AUTODL_ARTIFACT_BACKUP_VERIFIED:-}" == "1" ]] \
            || die "gate requires verified deploy-artifact backup evidence"
        python scripts/smolvla_autodl_way_sim_gate_v2.py "$@"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        die "unknown post-training command: $command"
        ;;
esac
