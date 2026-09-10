#!/usr/bin/env bash
# Vcdropout formal furnace driver — preregistered single-arm ladder inside tmux.
# Mirrors run_smolvla_zen_furnace.sh for the visual-conditioning state-dropout
# candidate. Stops after export: the frozen offset-250 gradient gate runs
# LOCALLY on XPU against the transferred artifact (same runtime as the frozen
# Zen-uniform baseline), and Gate 3/4 require a separate registration.
# Durable outputs act as their own resume guards: re-invoking skips completed
# phases instead of burning create-only identities.
set -uo pipefail

REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPOSITORY_ROOT"

DURABLE_ROOT="${ROSETTA_AUTODL_ROOT:-/root/autodl-tmp/rosetta}"
RUNTIME_ROOT="${VCD_RUNTIME_ROOT:-/root/vcd-runtime}"
RUN_ROOT="$DURABLE_ROOT/runs"
ORCH="$RUN_ROOT/orchestration"
EXPID=m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003
mkdir -p "$RUNTIME_ROOT/checkpoints" "$DURABLE_ROOT/artifacts" "$ORCH"

PLAN=configs/vla/smolvla_450m_aloha_insertion_vcdropout_cuda_b64_001.yaml
RUN=m2-smolvla450m-vcdropout-cuda-b64-001
VAL_PREFIX=m2-smolvla450m-vcdropout-val
PREFLIGHT_REPORT="$RUN_ROOT/$EXPID/preflight/m2-smolvla450m-vcdropout-preflight-001.json"
SMOKE_GUARD="$RUNTIME_ROOT/checkpoints/$EXPID/smoke/m2-smolvla450m-vcdropout-smoke-001/checkpoints/000002/pretrained_model/model.safetensors"
FORMAL_GUARD="$RUNTIME_ROOT/checkpoints/$EXPID/formal/$RUN/checkpoints/000316/pretrained_model/model.safetensors"
SEL_REPORT="$RUN_ROOT/$EXPID/selection/$RUN-selection.json"

mark() { printf '%s\n' "$1" >> "$ORCH/vcd-furnace-events.jsonl"; }
fail() { mark "{\"event\":\"failure\",\"phase\":\"$1\",\"at\":\"$(date -u +%FT%TZ)\"}"; exit 1; }

phase() {
    local name="$1"; shift
    local guard="$1"; shift
    if [[ -n "$guard" && -e "$guard" ]]; then
        mark "{\"event\":\"skipped\",\"phase\":\"$name\"}"
        return 0
    fi
    local log="$ORCH/vcd-phase-$name.log"
    echo "[driver] $(date -u +%FT%TZ) start $name"
    if "$@" >"$log" 2>&1; then
        [[ -n "$guard" && "$guard" == /root/vcd-runtime/.stage-* ]] && touch "$guard"
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

phase doctor /root/vcd-runtime/.stage-doctor-done scripts/run_autodl.sh doctor
phase benchmark /root/vcd-runtime/.stage-benchmark-done scripts/run_autodl.sh benchmark

phase preflight "$PREFLIGHT_REPORT" python scripts/run_smolvla_v2.py preflight --plan "$PLAN"
phase smoke "$SMOKE_GUARD" python scripts/run_smolvla_v2.py smoke --plan "$PLAN"

BASEVAL_GUARD="$EXPV/validation/${VAL_PREFIX}-base.json"
phase baseval "$BASEVAL_GUARD" python scripts/smolvla_vcdropout_validate.py \
    --plan "$PLAN" --preflight-report "$PREFLIGHT_REPORT"

phase formal-train "$FORMAL_GUARD" python scripts/run_smolvla_v2.py train --plan "$PLAN" \
    --preflight-report "$PREFLIGHT_REPORT" \
    --base-validation-report "$BASEVAL_GUARD"

for step in 79 158 237 316; do
    phase "val-step-$step" "$EXPV/validation/${VAL_PREFIX}-step-$(printf '%06d' "$step").json" \
        python scripts/smolvla_vcdropout_validate.py --plan "$PLAN" \
        --preflight-report "$PREFLIGHT_REPORT" --checkpoint-step "$step"
done

phase select "$SEL_REPORT" python scripts/select_smolvla_vcdropout_checkpoint.py \
    --plan "$PLAN" --run-root "$RUN_ROOT"

selected_step=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['selected_checkpoint_step'])" "$SEL_REPORT")
artifact_id="${RUN}-step$(printf '%04d' "$selected_step")-deploy-001"
manifest="$DURABLE_ROOT/artifacts/$EXPID/$artifact_id/manifest.json"
phase export "$manifest" python scripts/export_smolvla_vcdropout.py \
    --plan "$PLAN" --selection-report "$SEL_REPORT" \
    --run-root "$RUN_ROOT" \
    --checkpoint-root "$RUNTIME_ROOT/checkpoints" \
    --artifact-root "$DURABLE_ROOT/artifacts"

mark "{\"event\":\"furnace_complete\",\"at\":\"$(date -u +%FT%TZ)\",\"artifact_id\":\"$artifact_id\"}"
echo "[driver] FURNACE COMPLETE (through export only)"
echo "[driver] artifact: $artifact_id"
echo "[driver] NEXT (local control plane):"
echo "[driver]   1. transfer artifacts/$EXPID/$artifact_id to the local artifact root (SHA256-verified)"
echo "[driver]   2. run scripts/gate_smolvla_vcdropout_visual_conditioning.py --plan $PLAN --artifact-id $artifact_id on XPU"
echo "[driver]   3. a gate pass only permits a separately registered Gate 3/4 comparison; a fail closes the axis"
echo "[driver]   4. shut the instance down under the registered procedure (keep checkpoints until backup)"
