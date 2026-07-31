#!/usr/bin/env bash
# ============================================================================
# run_followup.sh: Run all follow-up experiments in one shot.
#
# PHASE 1 (CPU, seconds): pathology / heterogeneous-ensemble / calibration
#                          analyses on existing predictions.
# PHASE 2 (GPU, ~30 min): TTA UQ for 3 archs x 5 folds.
# PHASE 3 (GPU, ~30 min): synthetic distortion injection for 3 archs x 5 folds.
#
# All outputs land in $RUNS as uq_tta_*, syndist_*, plus a combined text log.
# Inference-only (no training) so disk impact is small (JSON/CSV/PNG per run).
#
# Usage:
#   bash scripts/run_followup.sh [GPU]      # GPU defaults to 0
#
# Run inside tmux so it survives disconnects.
# ============================================================================

set -uo pipefail

REPO="/raid/candi/michal/miqa/repo"
WORKDIR="/raid/candi/michal/miqa"
DATA="repo/dataset/index_full.csv"
RUNS="/raid/candi/michal/miqa/runs"
LOGS="/raid/candi/michal/miqa/logs"
GPU="${1:-0}"
ARCHS=(resnet18 resnet50 densenet121)
FOLDS=(0 1 2 3 4)

cd "$WORKDIR"
mkdir -p "$LOGS"
TS=$(date +%Y%m%d_%H%M%S)
RESULTS="$LOGS/followup_results_${TS}.txt"

log() { echo "$@" | tee -a "$RESULTS"; }

log "============================================================"
log "MIQA FOLLOW-UP EXPERIMENTS - $(date)"
log "GPU=$GPU  Results=$RESULTS"
log "============================================================"

# ---------------- PHASE 1: CPU analyses ----------------
log ""
log "### PHASE 1: CPU ANALYSES ###"
for script in analyze_densenet_pathology analyze_heterogeneous_ensemble analyze_calibration; do
    log ""
    log ">>> $script"
    python "$REPO/scripts/${script}.py" 2>&1 | tee -a "$RESULTS"
done

# ---------------- PHASE 2: TTA UQ ----------------
log ""
log "### PHASE 2: TEST-TIME AUGMENTATION (3 archs x 5 folds) ###"
for arch in "${ARCHS[@]}"; do
    for f in "${FOLDS[@]}"; do
        ckpt="$RUNS/poster100_${arch}_dwi_roi_f${f}/best_auroc.pt"
        outdir="$RUNS/uq_tta_${arch}_f${f}"
        if [ ! -f "$ckpt" ]; then
            log "SKIP TTA $arch f$f (no checkpoint: $ckpt)"
            continue
        fi
        log ">>> TTA $arch fold $f"
        CUDA_VISIBLE_DEVICES=$GPU python "$REPO/scripts/evaluate_uncertainty.py" \
            --tta_ckpt "$ckpt" \
            --index_csv "$DATA" \
            --outdir "$outdir" \
            --n_folds 5 --fold "$f" --n_forward 50 \
            > "$LOGS/tta_${arch}_f${f}_${TS}.log" 2>&1 \
            && log "    done -> $outdir" \
            || log "    FAILED (see $LOGS/tta_${arch}_f${f}_${TS}.log)"
    done
done

# ---------------- PHASE 3: synthetic distortion ----------------
log ""
log "### PHASE 3: SYNTHETIC DISTORTION INJECTION (3 archs x 5 folds) ###"
for arch in "${ARCHS[@]}"; do
    for f in "${FOLDS[@]}"; do
        ckpt="$RUNS/poster100_${arch}_dwi_roi_f${f}/best_auroc.pt"
        outdir="$RUNS/syndist_${arch}_f${f}"
        if [ ! -f "$ckpt" ]; then
            log "SKIP syndist $arch f$f (no checkpoint)"
            continue
        fi
        log ">>> syndist $arch fold $f"
        CUDA_VISIBLE_DEVICES=$GPU python "$REPO/scripts/evaluate_synthetic_distortion.py" \
            --ckpt "$ckpt" \
            --index_csv "$DATA" \
            --outdir "$outdir" \
            --n_folds 5 --fold "$f" \
            > "$LOGS/syndist_${arch}_f${f}_${TS}.log" 2>&1 \
            && log "    done -> $outdir" \
            || log "    FAILED (see $LOGS/syndist_${arch}_f${f}_${TS}.log)"
    done
done

log ""
log "============================================================"
log "ALL FOLLOW-UP EXPERIMENTS DONE - $(date)"
log "Combined log: $RESULTS"
log "TTA outputs:        $RUNS/uq_tta_*"
log "Synthetic outputs:  $RUNS/syndist_*"
log "============================================================"
