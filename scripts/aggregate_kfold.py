"""Aggregate per-fold test predictions into full-dataset evaluation.

Combines test_predictions.csv from all K folds of a configuration into a
single evaluation over all cases (e.g. 245 for the full UCL dataset).

Usage:
    # Explicit fold directories:
    python scripts/aggregate_kfold.py \
        --run_dirs runs/ucl_r18_..._f0_* runs/ucl_r18_..._f1_* ... \
        --outdir results/aggregated/ucl_r18/

    # Auto-discover folds from a glob pattern:
    python scripts/aggregate_kfold.py \
        --pattern "runs/ucl_r18_dwi_roi_s224_aug_cosine_seed1_f*" \
        --outdir results/aggregated/
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    balanced_accuracy_score,
)


def bootstrap_ci(y_true, y_prob, metric_fn, n_boot=2000, seed=42, ci=0.95):
    """Compute bootstrap confidence interval for a metric."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            scores.append(metric_fn(y_true[idx], y_prob[idx]))
        except (ValueError, ZeroDivisionError):
            continue
    if not scores:
        return (np.nan, np.nan)
    alpha = (1 - ci) / 2
    lo = np.percentile(scores, 100 * alpha)
    hi = np.percentile(scores, 100 * (1 - alpha))
    return (round(float(lo), 4), round(float(hi), 4))


def paired_comparison(aurocs_a, aurocs_b):
    """Wilcoxon signed-rank test on paired per-fold metric values.

    Returns dict with p-value, effect size (Cohen's d), and test details.
    """
    a = np.array(aurocs_a, dtype=float)
    b = np.array(aurocs_b, dtype=float)
    diff = a - b

    if len(a) < 5 or np.all(diff == 0):
        return {
            "test": "wilcoxon",
            "n_folds": len(a),
            "mean_diff": round(float(diff.mean()), 4),
            "p_value": 1.0,
            "cohens_d": 0.0,
            "note": "insufficient data or no difference",
        }

    stat, p_value = stats.wilcoxon(a, b)
    pooled_std = np.sqrt((a.std()**2 + b.std()**2) / 2)
    cohens_d = float(diff.mean() / pooled_std) if pooled_std > 0 else 0.0

    return {
        "test": "wilcoxon",
        "n_folds": len(a),
        "mean_a": round(float(a.mean()), 4),
        "mean_b": round(float(b.mean()), 4),
        "mean_diff": round(float(diff.mean()), 4),
        "statistic": round(float(stat), 4),
        "p_value": round(float(p_value), 6),
        "cohens_d": round(cohens_d, 4),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate k-fold test predictions")
    ap.add_argument("--run_dirs", nargs="+", default=[],
                    help="Explicit list of fold run directories")
    ap.add_argument("--pattern", type=str, default="",
                    help="Glob pattern to auto-discover fold directories")
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--n_boot", type=int, default=2000,
                    help="Bootstrap iterations for CIs")

    # Optional: compare two configurations
    ap.add_argument("--compare_dirs", nargs="+", default=[],
                    help="Second set of fold dirs for paired comparison")
    ap.add_argument("--compare_pattern", type=str, default="",
                    help="Glob pattern for second config's fold dirs")

    args = ap.parse_args()

    # --- Discover fold directories ---
    if args.run_dirs:
        fold_dirs = [Path(d) for d in args.run_dirs]
    elif args.pattern:
        fold_dirs = sorted(Path(p) for p in glob.glob(args.pattern))
    else:
        ap.error("Specify --run_dirs or --pattern")

    if not fold_dirs:
        print(f"ERROR: No directories found.")
        sys.exit(1)

    print(f"Found {len(fold_dirs)} fold directories:")
    for d in fold_dirs:
        print(f"  {d}")

    # --- Load and concatenate predictions ---
    dfs = []
    per_fold_metrics = []
    for d in fold_dirs:
        pred_path = d / "test_predictions.csv"
        if not pred_path.exists():
            print(f"WARNING: {pred_path} not found, skipping")
            continue
        df = pd.read_csv(pred_path)
        dfs.append(df)

        # Per-fold metric
        y_t = df["y_true"].values
        y_p = df["y_prob"].values
        try:
            fold_auroc = float(roc_auc_score(y_t, y_p))
        except ValueError:
            fold_auroc = np.nan
        per_fold_metrics.append({
            "dir": str(d),
            "n_samples": len(df),
            "auroc": round(fold_auroc, 4),
        })

    if not dfs:
        print("ERROR: No test_predictions.csv found in any directory.")
        sys.exit(1)

    all_preds = pd.concat(dfs, ignore_index=True)

    # Check for duplicates
    n_unique = all_preds["case_id"].nunique()
    n_total = len(all_preds)
    if n_unique != n_total:
        print(f"WARNING: {n_total} predictions but {n_unique} unique case_ids "
              f"({n_total - n_unique} duplicates)")

    print(f"\nAggregated: {n_total} predictions, {n_unique} unique cases")

    y_true = all_preds["y_true"].values
    y_prob = all_preds["y_prob"].values
    y_pred = (y_prob >= 0.5).astype(int)

    # --- Aggregate metrics ---
    has_both = len(np.unique(y_true)) > 1
    auroc = float(roc_auc_score(y_true, y_prob)) if has_both else 0.0
    auprc = float(average_precision_score(y_true, y_prob)) if has_both else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    sensitivity = round(tp / (tp + fn) if (tp + fn) > 0 else 0.0, 4)
    specificity = round(tn / (tn + fp) if (tn + fp) > 0 else 0.0, 4)

    # Bootstrap CIs
    auroc_ci = bootstrap_ci(y_true, y_prob, roc_auc_score,
                            n_boot=args.n_boot) if has_both else (0.0, 0.0)
    auprc_ci = bootstrap_ci(y_true, y_prob, average_precision_score,
                            n_boot=args.n_boot) if has_both else (0.0, 0.0)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    summary = {
        "n_folds": len(dfs),
        "n_predictions": n_total,
        "n_unique_cases": n_unique,
        "aggregate_metrics": {
            "auroc": round(auroc, 4),
            "auroc_95ci": auroc_ci,
            "auprc": round(auprc, 4),
            "auprc_95ci": auprc_ci,
            "f1": round(f1, 4),
            "bal_acc": round(bal_acc, 4),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        },
        "per_fold": per_fold_metrics,
    }

    # --- Paired comparison (optional) ---
    compare_dirs = []
    if args.compare_dirs:
        compare_dirs = [Path(d) for d in args.compare_dirs]
    elif args.compare_pattern:
        compare_dirs = sorted(Path(p) for p in glob.glob(args.compare_pattern))

    if compare_dirs:
        aurocs_a = [m["auroc"] for m in per_fold_metrics
                     if not np.isnan(m["auroc"])]
        aurocs_b = []
        for d in compare_dirs:
            pred_path = d / "test_predictions.csv"
            if not pred_path.exists():
                continue
            df_b = pd.read_csv(pred_path)
            try:
                aurocs_b.append(float(roc_auc_score(
                    df_b["y_true"].values, df_b["y_prob"].values)))
            except ValueError:
                pass

        if len(aurocs_a) == len(aurocs_b) and len(aurocs_a) >= 3:
            comparison = paired_comparison(aurocs_a, aurocs_b)
            summary["paired_comparison"] = comparison
            print(f"\nPaired comparison: {json.dumps(comparison, indent=2)}")

    # --- Save ---
    all_preds.to_csv(out / "aggregate_predictions.csv", index=False)
    (out / "aggregate_summary.json").write_text(
        json.dumps(summary, indent=2))

    print(f"\nAggregate metrics:")
    print(f"  AUROC:       {auroc:.4f}  95% CI: {auroc_ci}")
    print(f"  AUPRC:       {auprc:.4f}  95% CI: {auprc_ci}")
    print(f"  F1:          {f1:.4f}")
    print(f"  Bal Acc:     {bal_acc:.4f}")
    print(f"  Sensitivity: {sensitivity}")
    print(f"  Specificity: {specificity}")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
