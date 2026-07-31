"""Uncertainty quantification inference and analysis.

Supports two modes:
  1. MC Dropout: --mc_dropout_ckpt  (single checkpoint, T stochastic passes)
  2. Deep Ensemble: --ensemble_ckpts or --ensemble_dir (M checkpoints)

Outputs:
  - predictions_uncertainty.csv: per-sample uncertainty data
  - uncertainty_summary.json: aggregate metrics + selective prediction
  - selective_prediction_curve.png
  - uncertainty_histogram.png
  - uncertainty_calibration.png
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score,
                             f1_score)

from miqa.datasets import UCLDistortionDataset
from miqa.models import build_model
from miqa.uncertainty import (
    build_model_with_dropout,
    mc_dropout_predict,
    ensemble_predict,
    tta_predict,
    uncertainty_error_auroc,
    selective_prediction_curve,
    plot_selective_prediction,
    plot_uncertainty_histogram,
    plot_uncertainty_calibration,
)
from miqa.utils import load_test_split


def main():
    ap = argparse.ArgumentParser(
        description="Uncertainty quantification analysis")

    # Data
    ap.add_argument("--index_csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--n_folds", type=int, default=0,
                    help="K-fold CV (0=disabled)")
    ap.add_argument("--fold", type=int, default=0,
                    help="Which fold to use for test split (0-indexed)")

    # MC Dropout mode
    ap.add_argument("--mc_dropout_ckpt", type=str, default="",
                    help="Path to MC Dropout checkpoint")
    ap.add_argument("--n_forward", type=int, default=50,
                    help="Number of MC/TTA forward passes (default: 50)")

    # Deep Ensemble mode
    ap.add_argument("--ensemble_ckpts", nargs="+", default=[],
                    help="Explicit list of checkpoint paths for ensemble")
    ap.add_argument("--ensemble_dir", type=str, default="",
                    help="Directory containing seed-variant runs; "
                         "auto-discovers best_auroc.pt in subdirectories")

    # Test-Time Augmentation mode
    ap.add_argument("--tta_ckpt", type=str, default="",
                    help="Path to a standard checkpoint for TTA inference")

    # Selective prediction
    ap.add_argument("--n_sel_points", type=int, default=20,
                    help="Points on selective prediction curve")

    args = ap.parse_args()

    # Validate: exactly one mode
    has_mc = bool(args.mc_dropout_ckpt)
    has_ens = bool(args.ensemble_ckpts) or bool(args.ensemble_dir)
    has_tta = bool(args.tta_ckpt)
    if sum([has_mc, has_ens, has_tta]) == 0:
        ap.error("Specify one of --mc_dropout_ckpt / "
                 "--ensemble_ckpts|--ensemble_dir / --tta_ckpt")
    if sum([has_mc, has_ens, has_tta]) > 1:
        ap.error("Specify exactly one UQ mode "
                 "(MC Dropout, Deep Ensemble, or TTA)")

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Resolve checkpoint(s) and model config ---
    dropout_rate = None
    if has_mc:
        mode = "mc_dropout"
        ckpt = torch.load(args.mc_dropout_ckpt, map_location="cpu")
        ckpt_args = ckpt.get("args", {})
        model_name = ckpt_args.get("model", "resnet18")
        k_slices = ckpt_args.get("k_slices", 3)
        modality = ckpt_args.get("modality", "dwi")
        use_mask_crop = ckpt_args.get("use_mask_crop", True)
        split_seed = ckpt_args.get("split_seed", args.split_seed)
        dropout_rate = ckpt_args.get("dropout_rate", 0.2)
        run_dir = str(Path(args.mc_dropout_ckpt).parent)
    elif has_tta:
        mode = "tta"
        ckpt = torch.load(args.tta_ckpt, map_location="cpu")
        ckpt_args = ckpt.get("args", {})
        model_name = ckpt_args.get("model", "resnet18")
        k_slices = ckpt_args.get("k_slices", 3)
        modality = ckpt_args.get("modality", "dwi")
        use_mask_crop = ckpt_args.get("use_mask_crop", True)
        split_seed = ckpt_args.get("split_seed", args.split_seed)
        run_dir = str(Path(args.tta_ckpt).parent)
    else:
        mode = "deep_ensemble"
        # Collect checkpoint paths
        if args.ensemble_ckpts:
            ckpt_paths = args.ensemble_ckpts
        else:
            ens_dir = Path(args.ensemble_dir)
            ckpt_paths = sorted(str(p) for p in ens_dir.rglob("best_auroc.pt"))
            if not ckpt_paths:
                raise FileNotFoundError(
                    f"No best_auroc.pt found under {ens_dir}")
            print(f"Auto-discovered {len(ckpt_paths)} checkpoints "
                  f"under {ens_dir}")

        # Config from first checkpoint
        ckpt = torch.load(ckpt_paths[0], map_location="cpu")
        ckpt_args = ckpt.get("args", {})
        model_name = ckpt_args.get("model", "resnet18")
        k_slices = ckpt_args.get("k_slices", 3)
        modality = ckpt_args.get("modality", "dwi")
        use_mask_crop = ckpt_args.get("use_mask_crop", True)
        split_seed = ckpt_args.get("split_seed", args.split_seed)
        run_dir = str(Path(ckpt_paths[0]).parent)

    in_chans = k_slices * 2 if modality == "fusion" else k_slices

    # --- Data ---
    df = pd.read_csv(args.index_csv)
    group_col = "patient_id" if "patient_id" in df.columns else None
    if args.n_folds > 0:
        from miqa.utils import load_or_create_kfold_splits
        _, _, te = load_or_create_kfold_splits(
            df, args.n_folds, args.fold, split_seed, out,
            group_col=group_col)
    else:
        te = load_test_split(df, run_dir=run_dir, split_seed=split_seed,
                             group_col=group_col)
    print(f"Test set: {len(te)} samples")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"

    ds = UCLDistortionDataset(te, modality=modality,
                              use_mask_crop=use_mask_crop,
                              k_slices=k_slices, img_size=args.img_size,
                              return_case_id=True)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False,
                    num_workers=0, pin_memory=False)

    # --- Inference ---
    print(f"Mode: {mode}")
    if mode == "mc_dropout":
        model = build_model_with_dropout(
            model_name, in_chans=in_chans, pretrained=False,
            dropout_rate=dropout_rate,
        ).to(device)
        model.load_state_dict(ckpt["model"], strict=True)
        print(f"MC Dropout: {args.n_forward} forward passes, "
              f"dropout_rate={dropout_rate}")
        result = mc_dropout_predict(model, dl, device,
                                    n_forward=args.n_forward,
                                    amp_enabled=amp_enabled)
    elif mode == "tta":
        model = build_model(
            model_name, in_chans=in_chans, pretrained=False,
        ).to(device)
        model.load_state_dict(ckpt["model"], strict=True)
        print(f"TTA: {args.n_forward} augmented forward passes")
        result = tta_predict(model, dl, device,
                             n_forward=args.n_forward,
                             amp_enabled=amp_enabled)
    else:
        print(f"Deep Ensemble: {len(ckpt_paths)} checkpoints")
        for i, cp in enumerate(ckpt_paths):
            print(f"  [{i+1}] {cp}")
        result = ensemble_predict(ckpt_paths, model_name, in_chans, dl,
                                  device, amp_enabled=amp_enabled)

    y_true = result["y_true"]
    mean_prob = result["mean_prob"]
    std_prob = result["std_prob"]
    entropy = result["predictive_entropy"]
    mi = result["mutual_information"]
    case_ids = result["case_ids"]
    y_pred = (mean_prob >= 0.5).astype(int)

    # --- Per-sample CSV ---
    bval_map = dict(zip(te["case_id"], te["b_value"]))
    pred_df = pd.DataFrame({
        "case_id": case_ids if case_ids else range(len(y_true)),
        "b_value": [bval_map.get(c, "") for c in case_ids] if case_ids else "",
        "y_true": y_true.astype(int),
        "mean_prob": np.round(mean_prob, 4),
        "std_prob": np.round(std_prob, 4),
        "entropy": np.round(entropy, 4),
        "mutual_information": np.round(mi, 4),
        "y_pred": y_pred,
    })
    pred_df.to_csv(out / "predictions_uncertainty.csv", index=False)
    print(f"Saved per-sample predictions to {out / 'predictions_uncertainty.csv'}")

    # --- Aggregate metrics ---
    has_both = len(np.unique(y_true)) > 1
    auroc = float(roc_auc_score(y_true, mean_prob)) if has_both else 0.0
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Uncertainty-error AUROC (can uncertainty predict errors?)
    ue_auroc_std = uncertainty_error_auroc(y_true, y_pred, std_prob)
    ue_auroc_ent = uncertainty_error_auroc(y_true, y_pred, entropy)
    ue_auroc_mi = uncertainty_error_auroc(y_true, y_pred, mi)

    # Selective prediction
    sel_data = selective_prediction_curve(y_true, mean_prob, std_prob,
                                          n_points=args.n_sel_points)

    summary = {
        "mode": mode,
        "n_test": int(len(y_true)),
        "n_forward": args.n_forward if mode in ("mc_dropout", "tta") else None,
        "n_ensemble": len(ckpt_paths) if mode == "deep_ensemble" else None,
        "model": model_name,
        "dropout_rate": dropout_rate if mode == "mc_dropout" else None,
        "classification": {
            "auroc": round(auroc, 4),
            "bal_acc": round(bal_acc, 4),
            "f1": round(f1, 4),
        },
        "uncertainty_error_auroc": {
            "std": round(ue_auroc_std, 4) if not np.isnan(ue_auroc_std) else None,
            "entropy": round(ue_auroc_ent, 4) if not np.isnan(ue_auroc_ent) else None,
            "mutual_information": round(ue_auroc_mi, 4) if not np.isnan(ue_auroc_mi) else None,
        },
        "uncertainty_stats": {
            "std_mean": round(float(std_prob.mean()), 4),
            "std_median": round(float(np.median(std_prob)), 4),
            "entropy_mean": round(float(entropy.mean()), 4),
            "mi_mean": round(float(mi.mean()), 4),
        },
        "selective_prediction": sel_data,
    }

    # --- Per-b-value uncertainty breakdown ---
    if case_ids:
        bvals = np.array([bval_map.get(c, None) for c in case_ids])
        unique_bvals = sorted(set(v for v in bvals if v is not None and v != ""))
        per_bval = {}
        for bv in unique_bvals:
            mask = bvals == bv
            per_bval[str(bv)] = {
                "n": int(mask.sum()),
                "std_mean": round(float(std_prob[mask].mean()), 4),
                "entropy_mean": round(float(entropy[mask].mean()), 4),
                "mi_mean": round(float(mi[mask].mean()), 4),
                "accuracy": round(float((y_true[mask] == y_pred[mask]).mean()), 4),
            }
        summary["per_b_value"] = per_bval

    (out / "uncertainty_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # --- Plots ---
    plot_selective_prediction(sel_data,
                              str(out / "selective_prediction_curve.png"),
                              title=f"Selective Prediction ({mode})")

    plot_uncertainty_histogram(y_true, y_pred, std_prob,
                               str(out / "uncertainty_histogram.png"),
                               title=f"Uncertainty Distribution ({mode})")

    if len(np.unique(std_prob)) > 1:
        plot_uncertainty_calibration(y_true, y_pred, std_prob,
                                     str(out / "uncertainty_calibration.png"),
                                     title=f"Uncertainty Calibration ({mode})")

    print(f"\nDone. Results in {out}")


if __name__ == "__main__":
    main()
