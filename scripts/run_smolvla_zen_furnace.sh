#!/usr/bin/env bash
# Zen formal furnace driver — preregistered two-arm ladder inside tmux.
# Durable outputs act as their own resume guards: re-invoking skips completed
# phases instead of burning create-only identities.
set -uo pipefail

REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPOSITORY_ROOT"

DURABLE_ROOT="${ROSETTA_AUTODL_ROOT:-/root/autodl-tmp/rosetta}"
ZEN_RUNTIME="${ZEN_RUNTIME_ROOT:-/root/zen-runtime}"
RUN_ROOT="$DURABLE_ROOT/runs"
ORCH="$RUN_ROOT/orchestration"
EXPID=m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003
mkdir -p "$ZEN_RUNTIME/checkpoints" "$ZEN_RUNTIME/artifacts" "$ORCH"

ARM_A_PLAN=configs/vla/smolvla_450m_aloha_insertion_zen_cuda_b64_uniform_002.yaml
ARM_B_PLAN=configs/vla/smolvla_450m_aloha_insertion_zen_cuda_b64_firstaction_001.yaml

mark() { printf '%s\n' "$1" >> "$ORCH/zen-furnace-events.jsonl"; }
fail() { mark "{\"event\":\"failure\",\"phase\":\"$1\",\"at\":\"$(date -u +%FT%TZ)\"}"; exit 1; }

phase() {
    local name="$1"; shift
    local guard="$1"; shift
    if [[ -n "$guard" && -e "$guard" ]]; then
        mark "{\"event\":\"skipped\",\"phase\":\"$name\"}"
        return 0
    fi
    local log="$ORCH/zen-phase-$name.log"
    echo "[driver] $(date -u +%FT%TZ) start $name"
    if "$@" >"$log" 2>&1; then
        [[ -n "$guard" && "$guard" == /root/zen-runtime/.stage-* ]] && touch "$guard"
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
export ROSETTA_CHECKPOINT_ROOT=$ZEN_RUNTIME/checkpoints
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
CHECKS="$ZEN_RUNTIME/checkpoints/$EXPID"

STAGE_DOCTOR=/root/zen-runtime/.stage-doctor-done
STAGE_BENCH=/root/zen-runtime/.stage-benchmark-done

A_TAG=uniform
B_TAG=firstaction
A_PLAN=$ARM_A_PLAN
B_PLAN=$ARM_B_PLAN
A_RUN=m2-smolvla450m-zen-cuda-b64-uniform-001
B_RUN=m2-smolvla450m-zen-cuda-b64-firstaction-001
A_VAL_PREFIX=m2-smolvla450m-zen-cuda-b64-uniform-val
B_VAL_PREFIX=m2-smolvla450m-zen-cuda-b64-firstaction-val
A_PREFLIGHT="$EXPV/preflight/m2-smolvla450m-zen-${A_TAG}-preflight-001.json"
B_PREFLIGHT="$EXPV/preflight/m2-smolvla450m-zen-${B_TAG}-preflight-001.json"
A_SMOKE_GUARD="$CHECKS/smoke/m2-smolvla450m-zen-${A_TAG}-smoke-001/checkpoints/000002/pretrained_model/model.safetensors"
B_SMOKE_GUARD="$CHECKS/smoke/m2-smolvla450m-zen-${B_TAG}-smoke-001/checkpoints/000002/pretrained_model/model.safetensors"
A_FORMAL_GUARD="$CHECKS/formal/$A_RUN/checkpoints/000316/pretrained_model/model.safetensors"
B_FORMAL_GUARD="$CHECKS/formal/$B_RUN/checkpoints/000316/pretrained_model/model.safetensors"
A_SEL_REPORT="$EXPV/selection/$A_RUN-selection.json"
B_SEL_REPORT="$EXPV/selection/$B_RUN-selection.json"

phase doctor "$STAGE_DOCTOR" scripts/run_autodl.sh doctor
phase benchmark "$STAGE_BENCH" scripts/run_autodl.sh benchmark

arm_ladder() {
    local tag="$1" plan="$2" preflight_report="$3" smoke_guard="$4" formal_guard="$5" prefix="$6"

    if [[ ! -f "$preflight_report" ]]; then
        phase "preflight-$tag" "" python scripts/run_smolvla_v2.py preflight --plan "$plan"
        [[ -f "$preflight_report" ]] || fail "preflight-report-missing-$tag"
    else
        mark "{\"event\":\"skipped\",\"phase\":\"preflight-$tag\"}"
    fi

    if [[ ! -e "$smoke_guard" ]]; then
        phase "smoke-$tag" "" python scripts/run_smolvla_v2.py smoke --plan "$plan"
        [[ -e "$smoke_guard" ]] || fail "smoke-guard-missing-$tag"
    else
        mark "{\"event\":\"skipped\",\"phase\":\"smoke-$tag\"}"
    fi

    local baseval_guard="$EXPV/validation/${prefix}-base.json"
    if [[ ! -e "$baseval_guard" ]]; then
        echo "[driver] $(date -u +%FT%TZ) start baseval-$tag"
        python scripts/smolvla_zen_validate.py --plan "$plan" \
            --preflight-report "$preflight_report" \
            >"$ORCH/zen-phase-baseval-$tag.log" 2>&1 \
            || { tail -30 "$ORCH/zen-phase-baseval-$tag.log"; fail "baseval-$tag"; }
        mark "{\"event\":\"done\",\"phase\":\"baseval-$tag\"}"
    else
        mark "{\"event\":\"skipped\",\"phase\":\"baseval-$tag\"}"
    fi

    if [[ ! -e "$formal_guard" ]]; then
        echo "[driver] $(date -u +%FT%TZ) start formal-train-$tag"
        python scripts/run_smolvla_v2.py train --plan "$plan" \
            --preflight-report "$preflight_report" \
            --base-validation-report "$baseval_guard" \
            >"$ORCH/zen-phase-formal-$tag.log" 2>&1 \
            || { tail -40 "$ORCH/zen-phase-formal-$tag.log"; fail "formal-train-$tag"; }
        mark "{\"event\":\"done\",\"phase\":\"formal-train-$tag\"}"
    else
        mark "{\"event\":\"skipped\",\"phase\":\"formal-train-$tag\"}"
    fi
}

posttrain_arm() {
    local tag="$1" plan="$2" prefix="$3" sel_report="$4"

    for step in "" 79 158 237 316; do
        local label="base"
        [[ -n "$step" ]] && label="step-$(printf '%06d' "$step")"
        local rep="$EXPV/validation/${prefix}-${label}.json"
        [[ -e "$rep" ]] && { mark "{\"event\":\"skipped\",\"phase\":\"val-$tag-$label\"}"; continue; }
        echo "[driver] $(date -u +%FT%TZ) start val-$tag-$label"
        local step_args=()
        [[ -n "$step" ]] && step_args=(--checkpoint-step "$step")
        python scripts/smolvla_zen_validate.py --plan "$plan" \
            --preflight-report "$EXPV/preflight/m2-smolvla450m-zen-${tag}-preflight-001.json" \
            "${step_args[@]+"${step_args[@]}"}" \
            >"$ORCH/zen-phase-val-$tag-$label.log" 2>&1 \
            || { tail -20 "$ORCH/zen-phase-val-$tag-$label.log"; fail "val-$tag-$label"; }
        mark "{\"event\":\"done\",\"phase\":\"val-$tag-$label\"}"
    done

    phase "select-$tag" "$sel_report" python scripts/select_smolvla_zen_checkpoint.py \
        --plan "$plan" --run-root "$RUN_ROOT"

    local selected_step manifest artifact_id_final gate_suffix
    selected_step=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['selected_checkpoint_step'])" "$sel_report")
    artifact_id_final="m2-smolvla450m-zen-cuda-b64-${tag}-001-step$(printf '%04d' "$selected_step")-deploy-001"
    case "$tag" in uniform) gate_suffix=411 ;; firstaction) gate_suffix=422 ;; esac
    manifest="$DURABLE_ROOT/artifacts/$EXPID/$artifact_id_final/manifest.json"
    if [[ ! -f "$manifest" ]]; then
        phase "export-$tag" "" python scripts/export_smolvla_zen.py \
            --plan "$plan" --selection-report "$sel_report" \
            --run-root "$RUN_ROOT" \
            --checkpoint-root "$ZEN_RUNTIME/checkpoints" \
            --artifact-root "$DURABLE_ROOT/artifacts"
        [[ -f "$manifest" ]] || fail "export-manifest-missing-$tag"
    else
        mark "{\"event\":\"skipped\",\"phase\":\"export-$tag\"}"
    fi

    export ROSETTA_AUTODL_ARTIFACT_BACKUP_VERIFIED=1
    phase "gate3-$tag" "$EXPV/gates/gate3-smolvla-sim-$gate_suffix.json" \
        python scripts/smolvla_autodl_zen_sim_gate.py gate3 \
        --plan "$plan" --artifact-id "$artifact_id_final" \
        --run-root "$RUN_ROOT" --artifact-root "$DURABLE_ROOT/artifacts"
    local gate3_json
    gate3_json="$EXPV/gates/gate3-smolvla-sim-$gate_suffix.json"
    phase "gate4-$tag" "$EXPV/gates/gate4-smolvla-sim-$gate_suffix.json" \
        python scripts/smolvla_autodl_zen_sim_gate.py gate4 \
        --plan "$plan" --artifact-id "$artifact_id_final" \
        --run-root "$RUN_ROOT" --artifact-root "$DURABLE_ROOT/artifacts" \
        --gate3-report "$gate3_json"
}

arm_ladder "$A_TAG" "$A_PLAN" "$A_PREFLIGHT" "$A_SMOKE_GUARD" "$A_FORMAL_GUARD" "$A_VAL_PREFIX"
arm_ladder "$B_TAG" "$B_PLAN" "$B_PREFLIGHT" "$B_SMOKE_GUARD" "$B_FORMAL_GUARD" "$B_VAL_PREFIX"

posttrain_arm "$A_TAG" "$A_PLAN" "$A_VAL_PREFIX" "$A_SEL_REPORT"
posttrain_arm "$B_TAG" "$B_PLAN" "$B_VAL_PREFIX" "$B_SEL_REPORT"

mark "{\"event\":\"furnace_complete\",\"at\":\"$(date -u +%FT%TZ)\"}"
echo "[driver] FURNACE COMPLETE"
