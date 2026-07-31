#!/usr/bin/env bash
# ============================================================================
# run_dual_training.sh: launches improved-recipe + distillation in parallel.
#
# Phase A (GPU 0, ~8-10 h): improved training recipe: SWA + focal + label
#   smoothing + MixUp on 3 archs x 5 folds = 15 runs.
# Phase B (GPU 1, ~4-5 h): knowledge distillation: heterogeneous 9-model
#   ensemble -> ResNet-18 student on 5 folds.
#
# Inference-only side-effects: each run creates ONE small dir
# (improved_<arch>_f<fold>/ or distill_<student>_f<fold>/) so existing
# results are untouched.
#
# Usage (inside tmux):
#   bash scripts/run_dual_training.sh
# ============================================================================

set -uo pipefail

REPO="/raid/candi/michal/miqa/repo"
WORKDIR="/raid/candi/michal/miqa"
DATA="repo/dataset/index_full.csv"
RUNS="/raid/candi/michal/miqa/runs"
LOGS="/raid/candi/michal/miqa/logs"
ARCHS=(resnet18 resnet50 densenet121)
FOLDS=(0 1 2 3 4)

cd "$WORKDIR"
mkdir -p "$LOGS"
TS=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOGS/dual_training_${TS}.log"

log() { echo "$@" | tee -a "$MAIN_LOG"; }

log "============================================================"
log "MIQA DUAL TRAINING - $(date)"
log "Phase A: improved recipe on GPU 0 (15 runs, ~8-10 h)"
log "Phase B: distillation   on GPU 1 (5 runs,  ~4-5 h)"
log "Main log: $MAIN_LOG"
log "============================================================"

# ---------------- Phase A: improved recipe (background, GPU 0) ----------------
(
    PHASE_LOG="$LOGS/improved_${TS}.log"
    echo ">>> Phase A starting on GPU 0  ($(date))" > "$PHASE_LOG"
    for arch in "${ARCHS[@]}"; do
        for f in "${FOLDS[@]}"; do
            outdir="$RUNS/improved_${arch}_f${f}"
            if [ -f "$outdir/summary.json" ] && grep -q '"test"' "$outdir/summary.json" 2>/dev/null; then
                echo "[$(date +%H:%M:%S)] SKIP improved $arch f$f (already done)" >> "$PHASE_LOG"
                continue
            fi
            echo "[$(date +%H:%M:%S)] improved $arch f$f -> $outdir" >> "$PHASE_LOG"
            CUDA_VISIBLE_DEVICES=0 python "$REPO/scripts/train_improved.py" \
                --index_csv "$DATA" \
                --outdir "$outdir" \
                --model "$arch" \
                --n_folds 5 --fold "$f" \
                --epochs 100 --swa_start 60 \
                --focal_gamma 2.0 --label_smoothing 0.1 --mixup_alpha 0.2 \
                >> "$PHASE_LOG" 2>&1 \
                && echo "[$(date +%H:%M:%S)]   done" >> "$PHASE_LOG" \
                || echo "[$(date +%H:%M:%S)]   FAILED" >> "$PHASE_LOG"
        done
    done
    echo ">>> Phase A finished ($(date))" >> "$PHASE_LOG"
) &
A_PID=$!
log "Phase A launched (background pid $A_PID, see $LOGS/improved_${TS}.log)"

# ---------------- Phase B: distillation (foreground, GPU 1) ------------------
log "Phase B starting on GPU 1..."
PHASE_B_LOG="$LOGS/distill_${TS}.log"
echo ">>> Phase B starting on GPU 1  ($(date))" > "$PHASE_B_LOG"
for f in "${FOLDS[@]}"; do
    outdir="$RUNS/distill_resnet18_f${f}"
    if [ -f "$outdir/summary.json" ] && grep -q '"test_student"' "$outdir/summary.json" 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] SKIP distill f$f (already done)" >> "$PHASE_B_LOG"
        continue
    fi
    echo "[$(date +%H:%M:%S)] distill resnet18 f$f -> $outdir" >> "$PHASE_B_LOG"
    CUDA_VISIBLE_DEVICES=1 python "$REPO/scripts/train_distill.py" \
        --index_csv "$DATA" \
        --outdir "$outdir" \
        --runs_root "$RUNS" \
        --student resnet18 \
        --n_folds 5 --fold "$f" \
        --epochs 100 --kd_alpha 0.7 --kd_beta 0.5 \
        >> "$PHASE_B_LOG" 2>&1 \
        && echo "[$(date +%H:%M:%S)]   done" >> "$PHASE_B_LOG" \
        || echo "[$(date +%H:%M:%S)]   FAILED" >> "$PHASE_B_LOG"
done
echo ">>> Phase B finished ($(date))" >> "$PHASE_B_LOG"
log "Phase B finished."

# ---------------- Wait for phase A to complete ---------------------------
log "Waiting for Phase A (pid $A_PID) to finish..."
wait $A_PID
log "Phase A finished."

log "============================================================"
log "DUAL TRAINING DONE - $(date)"
log "Improved runs:  $RUNS/improved_*"
log "Distill  runs:  $RUNS/distill_*"
log "Main log:       $MAIN_LOG"
log "Phase A log:    $LOGS/improved_${TS}.log"
log "Phase B log:    $LOGS/distill_${TS}.log"
log "============================================================"
