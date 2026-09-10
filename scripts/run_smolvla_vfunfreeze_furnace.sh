#!/usr/bin/env bash
# Vfunfreeze formal furnace driver — preregistered single-arm ladder inside tmux.
# Mirrors run_smolvla_vcdropout_furnace.sh for the vision-front-end-unfreeze
# candidate. Adds one phase the axis requires: after the two-step optimizer
# smoke, verify (weights-level, against the pinned base snapshot) that the
# visual front end received non-zero updates and the language model stayed
# bit-identical. Stops after export: the corrected frame-0 paired alignment
# probe and Gate 3/4 run LOCALLY against the transferred artifact and require
# their own registrations. Durable outputs act as their own resume guards:
# re-invoking skips completed phases instead of burning create-only identities.
set -uo pipefail

REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPOSITORY_ROOT"

DURABLE_ROOT="${ROSETTA_AUTODL_ROOT:-/root/autodl-tmp/rosetta}"
RUNTIME_ROOT="${VFU_RUNTIME_ROOT:-/root/vfunfreeze-runtime}"
RUN_ROOT="$DURABLE_ROOT/runs"
ORCH="$RUN_ROOT/orchestration"
EXPID=m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003
mkdir -p "$RUNTIME_ROOT/checkpoints" "$DURABLE_ROOT/artifacts" "$ORCH"

PLAN=configs/vla/smolvla_450m_aloha_insertion_vfunfreeze_cuda_b32_003.yaml
RUN=m2-smolvla450m-vfunfreeze-cuda-b32-003
VAL_PREFIX=m2-smolvla450m-vfunfreeze-val
PREFLIGHT_REPORT="$RUN_ROOT/$EXPID/preflight/m2-smolvla450m-vfunfreeze-preflight-003.json"
# The batch-32 smoke evidence lives on the system disk (bound to the current
# plan hash); a relocated runtime root must still be able to reuse it.
SMOKE_GUARD="${VFU_SMOKE_GUARD:-$RUNTIME_ROOT/checkpoints/$EXPID/smoke/m2-smolvla450m-vfunfreeze-smoke-003/checkpoints/000002/pretrained_model/model.safetensors}"
SMOKE_VERIFY_REPORT="$RUN_ROOT/$EXPID/diagnostics/vfunfreeze-smoke-update-verification.json"
FORMAL_GUARD="$RUNTIME_ROOT/checkpoints/$EXPID/formal/$RUN/checkpoints/000632/pretrained_model/model.safetensors"
SEL_REPORT="$RUN_ROOT/$EXPID/selection/$RUN-selection.json"

mark() { printf '%s\n' "$1" >> "$ORCH/vfunfreeze-furnace-events.jsonl"; }
fail() { mark "{\"event\":\"failure\",\"phase\":\"$1\",\"at\":\"$(date -u +%FT%TZ)\"}"; exit 1; }

phase() {
    local name="$1"; shift
    local guard="$1"; shift
    if [[ -n "$guard" && -e "$guard" ]]; then
        mark "{\"event\":\"skipped\",\"phase\":\"$name\"}"
        return 0
    fi
    local log="$ORCH/vfunfreeze-phase-$name.log"
    echo "[driver] $(date -u +%FT%TZ) start $name"
    if "$@" >"$log" 2>&1; then
        [[ -n "$guard" && "$guard" == /root/vfunfreeze-runtime/.stage-* ]] && touch "$guard"
        mark "{\"event\":\"done\",\"phase\":\"$name\"}"
        return 0
    fi
    tail -30 "$log"
    fail "$name"
}

mark "{\"event\":\"furnace_start\",\"workspace_tree\":\"$(cat .rosetta-workspace.sha256 2>/dev/null | cut -d' ' -f1)\",\"pid\":$$}"

source /root/autodl-tmp/rosetta/envs/smolvla-cuda-001/bin/activate

export AUTODL_TMP=/root/autodl-tmp
export ROSETTA_AUTODL_ROOT=$DURABLE_ROOT
export ROSETTA_DATA_ROOT=$DURABLE_ROOT/data
export ROSETTA_MODELS_ROOT=$DURABLE_ROOT/models
export ROSETTA_CHECKPOINT_ROOT=$RUNTIME_ROOT/checkpoints
export ROSETTA_ARTIFACT_ROOT=$DURABLE_ROOT/artifacts
export ROSETTA_RUN_ROOT=$RUN_ROOT
export TRACKIO_DIR=$RUN_ROOT/trackio HF_HOME=$DURABLE_ROOT/models/hf_home
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1
export ROSETTA_TORCH_DEVICE=cuda
export ROSETTA_DOCKER_MEMORY_LIMIT=autodl_platform_container ROSETTA_DOCKER_MEMORY_SWAP_LIMIT=autodl_platform_container
export MUJOCO_GL=egl ROSETTA_AUTODL_RUNTIME_PROFILE=configs/runtime/autodl_rtx4090.yaml
mkdir -p "$RUN_ROOT/pycache" "$RUN_ROOT/compiler_cache/cuda/inductor" "$RUN_ROOT/compiler_cache/cuda/triton"
rm -rf "$RUN_ROOT/pycache"/* >/dev/null 2>&1 || true

EXPV="$RUN_ROOT/$EXPID"

phase doctor "$RUNTIME_ROOT/.stage-doctor-done" scripts/run_autodl.sh doctor
phase benchmark "$RUNTIME_ROOT/.stage-benchmark-done" scripts/run_autodl.sh benchmark

phase preflight "$PREFLIGHT_REPORT" python scripts/run_smolvla_v2.py preflight --plan "$PLAN"
phase smoke "$SMOKE_GUARD" python scripts/run_smolvla_v2.py smoke --plan "$PLAN"

phase verify-smoke-updates "$SMOKE_VERIFY_REPORT" python scripts/verify_vfunfreeze_smoke_updates.py \
    --checkpoint "$SMOKE_GUARD" \
    --output "$SMOKE_VERIFY_REPORT"
# The report file alone is not a pass marker: a written FAILED verdict must
# stop the ladder even though the create-only file exists.
grep -q '"status": "passed"' "$SMOKE_VERIFY_REPORT" \
    || { echo "[driver] smoke update verification verdict is not passed"; fail verify-smoke-updates-verdict; }

BASEVAL_GUARD="$EXPV/validation/${VAL_PREFIX}-base.json"
phase baseval "$BASEVAL_GUARD" python scripts/smolvla_vfunfreeze_validate.py \
    --plan "$PLAN" --preflight-report "$PREFLIGHT_REPORT"

phase formal-train "$FORMAL_GUARD" python scripts/run_smolvla_v2.py train --plan "$PLAN" \
    --preflight-report "$PREFLIGHT_REPORT" \
    --base-validation-report "$BASEVAL_GUARD"

for step in 158 316 474 632; do
    phase "val-step-$step" "$EXPV/validation/${VAL_PREFIX}-step-$(printf '%06d' "$step").json" \
        python scripts/smolvla_vfunfreeze_validate.py --plan "$PLAN" \
        --preflight-report "$PREFLIGHT_REPORT" --checkpoint-step "$step"
done

phase select "$SEL_REPORT" python scripts/select_smolvla_vfunfreeze_checkpoint.py \
    --plan "$PLAN" --run-root "$RUN_ROOT"

selected_step=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['selected_checkpoint_step'])" "$SEL_REPORT")
artifact_id="${RUN}-step$(printf '%04d' "$selected_step")-deploy-001"
manifest="$DURABLE_ROOT/artifacts/$EXPID/$artifact_id/manifest.json"
phase export "$manifest" python scripts/export_smolvla_vfunfreeze.py \
    --plan "$PLAN" --selection-report "$SEL_REPORT" \
    --run-root "$RUN_ROOT" \
    --checkpoint-root "$RUNTIME_ROOT/checkpoints" \
    --artifact-root "$DURABLE_ROOT/artifacts"

mark "{\"event\":\"furnace_complete\",\"at\":\"$(date -u +%FT%TZ)\",\"artifact_id\":\"$artifact_id\"}"
echo "[driver] FURNACE COMPLETE (through export only)"
echo "[driver] artifact: $artifact_id"
echo "[driver] NEXT (local control plane):"
echo "[driver]   1. transfer artifacts/$EXPID/$artifact_id to the local artifact root (SHA256-verified)"
echo "[driver]   2. rerun the corrected frame0 paired alignment probe on the artifact (alignment gate)"
echo "[driver]   3. a pass permits a separately registered Gate 3/4 comparison; a fail closes the axis"
echo "[driver]   4. shut the instance down under the registered procedure (keep checkpoints until backup)"
