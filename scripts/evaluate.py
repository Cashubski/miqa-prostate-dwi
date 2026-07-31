"""Error analysis + calibration for UCL distortion classification.

Loads a trained checkpoint, runs inference on the test split,
saves test_predictions.csv, error_summary.json, and optionally
calibration_summary.json + reliability_diagram.png.
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score,
                             f1_score, average_precision_score,
                             roc_curve, precision_recall_curve)

from miqa.models import build_model
from miqa.datasets import UCLDistortionDataset
from miqa.metrics import bootstrap_ci, expected_calibration_error
from miqa.calibration import plot_reliability_diagram
from miqa.utils import load_test_split


def main():
    ap = argparse.ArgumentParser(
        description="Error analysis + calibration for UCL distortion cls")
    ap.add_argument("--index_csv", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--run_dir", type=str, default="",
                    help="Run dir with splits.json (auto-detected from ckpt)")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--n_boot", type=int, default=1000,
                    help="Bootstrap iterations for CIs")
    ap.add_argument("--calibration", action="store_true", default=True,
                    help="Run calibration analysis (default: on)")
    ap.add_argument("--no_calibration", dest="calibration",
                    action="store_false")
    ap.add_argument("--n_bins", type=int, default=10,
                    help="Number of bins for calibration")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # Auto-detect run_dir from checkpoint path
    run_dir = args.run_dir or str(Path(args.ckpt).parent)

    df = pd.read_csv(args.index_csv)

    # Recover model config from checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    model_name = ckpt_args.get("model", "resnet18")
    k_slices = ckpt_args.get("k_slices", 3)
    modality = ckpt_args.get("modality", "dwi")
    use_mask_crop = ckpt_args.get("use_mask_crop", True)
    split_seed = ckpt_args.get("split_seed", args.split_seed)

    in_chans = k_slices * 2 if modality == "fusion" else k_slices

    te = load_test_split(df, run_dir=run_dir, split_seed=split_seed)
    print(f"Test set: {len(te)} samples, "
          f"labels: {dict(te['label'].value_counts().sort_index())}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(model_name, in_chans=in_chans,
                        pretrained=False).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    ds = UCLDistortionDataset(te, modality=modality,
                              use_mask_crop=use_mask_crop,
                              k_slices=k_slices, img_size=args.img_size,
                              return_case_id=True)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False,
                    num_workers=2, pin_memory=True)

    # --- Per-sample inference (needs case_ids for predictions CSV) ---
    ys, probs, case_ids = [], [], []
    with torch.no_grad():
        for x, y, cids in dl:
            x = x.to(device, non_blocking=True)
            logit = model(x).cpu().numpy().reshape(-1)
            prob = 1.0 / (1.0 + np.exp(-logit))
            ys.append(y.numpy().reshape(-1))
            probs.append(prob)
            case_ids.extend(list(cids))

    y = np.concatenate(ys)
    p = np.concatenate(probs)
    pred = (p >= 0.5).astype(int)

    bval_map = dict(zip(te["case_id"], te["b_value"]))

    res = pd.DataFrame({
        "case_id": case_ids,
        "b_value": [bval_map.get(c, "") for c in case_ids],
        "y_true": y.astype(int),
        "y_prob": np.round(p, 4),
        "y_pred": pred,
    })

    # Sort: FN first (most confident wrong), then FP, then correct
    res["error_type"] = "correct"
    res.loc[(res["y_true"] == 1) & (res["y_pred"] == 0), "error_type"] = "FN"
    res.loc[(res["y_true"] == 0) & (res["y_pred"] == 1), "error_type"] = "FP"
    order = {"FN": 0, "FP": 1, "correct": 2}
    res["_sort"] = res["error_type"].map(order)
    res["_conf"] = np.where(res["error_type"] == "FN", 1 - res["y_prob"],
                            np.where(res["error_type"] == "FP",
                                     res["y_prob"], 0))
    res = res.sort_values(["_sort", "_conf"], ascending=[True, False])
    res = res.drop(columns=["_sort", "_conf"])
    res.to_csv(out / "test_predictions.csv", index=False)

    # --- Confusion matrix + overall metrics ---
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    has_both = len(np.unique(y)) > 1
    auroc = float(roc_auc_score(y, p)) if has_both else 0.0
    auprc = float(average_precision_score(y, p)) if has_both else 0.0

    summary = {
        "n_test": int(len(y)),
        "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "sensitivity": round(tp / (tp + fn) if (tp + fn) > 0 else 0.0, 4),
        "specificity": round(tn / (tn + fp) if (tn + fp) > 0 else 0.0, 4),
        "class_0_count": int((y == 0).sum()),
        "class_1_count": int((y == 1).sum()),
    }

    # --- Per-b-value stratification ---
    bvals = np.array([bval_map.get(c, None) for c in case_ids])
    unique_bvals = sorted(set(v for v in bvals if v is not None and v != ""))
    per_bval = {}
    for bv in unique_bvals:
        mask = bvals == bv
        yb = y[mask]
        pb = p[mask]
        predb = pred[mask]
        if len(np.unique(yb)) < 2:
            per_bval[str(bv)] = "insufficient data"
            continue
        tp_b = int(((predb == 1) & (yb == 1)).sum())
        tn_b = int(((predb == 0) & (yb == 0)).sum())
        fp_b = int(((predb == 1) & (yb == 0)).sum())
        fn_b = int(((predb == 0) & (yb == 1)).sum())
        per_bval[str(bv)] = {
            "n": int(mask.sum()),
            "auroc": round(float(roc_auc_score(yb, pb)), 4),
            "auprc": round(float(average_precision_score(yb, pb)), 4),
            "bal_acc": round(float(balanced_accuracy_score(yb, predb)), 4),
            "f1": round(float(f1_score(yb, predb, zero_division=0)), 4),
            "sensitivity": round(tp_b / (tp_b + fn_b)
                                 if (tp_b + fn_b) > 0 else 0.0, 4),
            "specificity": round(tn_b / (tn_b + fp_b)
                                 if (tn_b + fp_b) > 0 else 0.0, 4),
        }
    summary["per_b_value"] = per_bval

    # --- Bootstrap CIs ---
    summary["bootstrap_ci"] = bootstrap_ci(y, p, n_boot=args.n_boot)

    (out / "error_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # --- ROC curve data (for thesis plots) ---
    if has_both:
        fpr, tpr, roc_thresholds = roc_curve(y, p)
        roc_df = pd.DataFrame({
            "fpr": np.round(fpr, 6),
            "tpr": np.round(tpr, 6),
            "threshold": np.round(roc_thresholds, 6),
        })
        roc_df.to_csv(out / "roc_curve.csv", index=False)
        print(f"ROC curve: {len(roc_df)} points → roc_curve.csv")

    # --- PR curve data ---
    if has_both:
        precision, recall, pr_thresholds = precision_recall_curve(y, p)
        pr_df = pd.DataFrame({
            "precision": np.round(precision[:-1], 6),
            "recall": np.round(recall[:-1], 6),
            "threshold": np.round(pr_thresholds, 6),
        })
        pr_df.to_csv(out / "pr_curve.csv", index=False)
        print(f"PR curve: {len(pr_df)} points → pr_curve.csv")

    # --- Threshold sweep (sensitivity/specificity at different thresholds) ---
    thresh_rows = []
    for t in np.arange(0.05, 1.0, 0.05):
        prd = (p >= t).astype(int)
        tp_t = int(((prd == 1) & (y == 1)).sum())
        tn_t = int(((prd == 0) & (y == 0)).sum())
        fp_t = int(((prd == 1) & (y == 0)).sum())
        fn_t = int(((prd == 0) & (y == 1)).sum())
        sens_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
        spec_t = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0.0
        f1_t = (2 * tp_t / (2 * tp_t + fp_t + fn_t)
                if (2 * tp_t + fp_t + fn_t) > 0 else 0.0)
        thresh_rows.append({
            "threshold": round(float(t), 2),
            "sensitivity": round(sens_t, 4),
            "specificity": round(spec_t, 4),
            "f1": round(f1_t, 4),
            "tp": tp_t, "tn": tn_t, "fp": fp_t, "fn": fn_t,
        })
    pd.DataFrame(thresh_rows).to_csv(
        out / "threshold_analysis.csv", index=False)
    print(f"Threshold analysis: 19 thresholds → threshold_analysis.csv")

    # --- Calibration (on by default) ---
    if args.calibration:
        ece, bins = expected_calibration_error(y, p, n_bins=args.n_bins)
        cal_summary = {
            "n_samples": int(len(y)),
            "n_bins": args.n_bins,
            "ece": ece,
            "bins": bins,
        }
        (out / "calibration_summary.json").write_text(
            json.dumps(cal_summary, indent=2))
        print(f"ECE = {ece:.4f}")

        plot_reliability_diagram(y, p, out / "reliability_diagram.png",
                                 n_bins=args.n_bins)


if __name__ == "__main__":
    main()
