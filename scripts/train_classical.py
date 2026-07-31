"""Classical (non-DL) baselines for UCL prostate DWI distortion detection.

Extracts handcrafted image features per case and trains:
  - Logistic Regression
  - Random Forest
  - SVM (RBF kernel)

Uses the same 5-fold CV splits as the deep learning experiments
(via load_or_create_kfold_splits) so results are directly comparable.

Outputs test_predictions.csv per model for aggregation.

Usage:
    python scripts/train_classical.py --index_csv /path/to/index.csv \
        --outdir runs/classical --n_folds 5 --fold 0

Runs on CPU in seconds, no GPU needed.
"""
import argparse, json, time, warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage, stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, f1_score, balanced_accuracy_score,
                             average_precision_score)

from miqa.utils import (set_seed, load_or_create_splits,
                         load_or_create_kfold_splits)


def extract_features(row: pd.Series, use_mask_crop: bool = True) -> dict:
    """Extract handcrafted image features from a single DWI volume."""
    dwi = nib.load(row["dwi_path"]).get_fdata().astype(np.float32)
    mask = nib.load(row["mask_path"]).get_fdata().astype(np.float32)

    if use_mask_crop:
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            # Fallback to full volume
            roi = dwi
        else:
            mins = coords.min(axis=0)
            maxs = coords.max(axis=0) + 1
            roi = dwi[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
    else:
        roi = dwi

    # Flatten to 1D for statistics
    vals = roi.ravel()
    vals = vals[vals > 0]  # ignore background zeros
    if len(vals) == 0:
        vals = np.array([0.0])

    feats = {}

    # Intensity statistics
    feats["mean"] = float(np.mean(vals))
    feats["std"] = float(np.std(vals))
    feats["median"] = float(np.median(vals))
    feats["skewness"] = float(stats.skew(vals))
    feats["kurtosis"] = float(stats.kurtosis(vals))
    feats["p5"] = float(np.percentile(vals, 5))
    feats["p95"] = float(np.percentile(vals, 95))
    feats["iqr"] = float(np.percentile(vals, 75) - np.percentile(vals, 25))
    feats["range"] = float(np.max(vals) - np.min(vals))
    feats["coeff_var"] = float(np.std(vals) / (np.mean(vals) + 1e-8))

    # Edge / sharpness features (Laplacian variance per mid-slice)
    mid_z = roi.shape[2] // 2
    mid_slice = roi[:, :, mid_z]
    lap = ndimage.laplace(mid_slice)
    feats["laplacian_var"] = float(np.var(lap))
    feats["laplacian_mean"] = float(np.mean(np.abs(lap)))

    # Gradient magnitude (Sobel)
    sx = ndimage.sobel(mid_slice, axis=0)
    sy = ndimage.sobel(mid_slice, axis=1)
    grad_mag = np.sqrt(sx**2 + sy**2)
    feats["gradient_mean"] = float(np.mean(grad_mag))
    feats["gradient_std"] = float(np.std(grad_mag))

    # Spatial symmetry (left-right difference in mid-slice)
    h, w = mid_slice.shape
    left = mid_slice[:, :w // 2]
    right = mid_slice[:, w // 2:w // 2 + left.shape[1]]
    if left.shape == right.shape:
        right_flipped = right[:, ::-1]
        feats["lr_asymmetry"] = float(np.mean(np.abs(left - right_flipped)))
    else:
        feats["lr_asymmetry"] = 0.0

    # Volume-level: slice-to-slice intensity variation
    slice_means = [float(np.mean(roi[:, :, z][roi[:, :, z] > 0]))
                   for z in range(roi.shape[2])
                   if np.any(roi[:, :, z] > 0)]
    if len(slice_means) > 1:
        feats["slice_mean_std"] = float(np.std(slice_means))
        feats["slice_mean_range"] = float(max(slice_means) - min(slice_means))
    else:
        feats["slice_mean_std"] = 0.0
        feats["slice_mean_range"] = 0.0

    # Mask shape features
    mask_coords = np.argwhere(mask > 0)
    if len(mask_coords) > 0:
        feats["mask_volume"] = float(len(mask_coords))
        bbox_size = mask_coords.max(axis=0) - mask_coords.min(axis=0) + 1
        feats["mask_bbox_vol"] = float(np.prod(bbox_size))
        feats["mask_solidity"] = feats["mask_volume"] / (feats["mask_bbox_vol"] + 1e-8)
    else:
        feats["mask_volume"] = 0.0
        feats["mask_bbox_vol"] = 0.0
        feats["mask_solidity"] = 0.0

    return feats


def main():
    ap = argparse.ArgumentParser(
        description="Classical ML baselines for distortion detection")

    ap.add_argument("--index_csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--n_folds", type=int, default=0,
                    help="K-fold CV (0=disabled)")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--use_mask_crop", action="store_true", default=True)
    ap.add_argument("--no_mask_crop", dest="use_mask_crop",
                    action="store_false")
    ap.add_argument("--max_cases", type=int, default=0)

    args = ap.parse_args()
    set_seed(args.seed)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args).copy()
    (out / "config.json").write_text(json.dumps(config, indent=2))

    # --- Data split ---
    df = pd.read_csv(args.index_csv)
    group_col = "patient_id" if "patient_id" in df.columns else None
    if args.max_cases > 0:
        ids = sorted(df["case_id"].unique())
        rng = np.random.RandomState(args.split_seed)
        rng.shuffle(ids)
        ids = ids[:args.max_cases]
        df = df[df["case_id"].isin(set(ids))]

    if args.n_folds > 0:
        tr, va, te = load_or_create_kfold_splits(
            df, args.n_folds, args.fold, args.split_seed, out,
            group_col=group_col)
    else:
        tr, va, te = load_or_create_splits(df, args.split_seed, out,
                                           group_col=group_col)

    print(f"Split: train={len(tr)}, val={len(va)}, test={len(te)}")

    # --- Feature extraction ---
    t0 = time.time()

    def extract_df(split_df, name):
        feats_list = []
        for i, (_, row) in enumerate(split_df.iterrows()):
            print(f"  [{name}] {i+1}/{len(split_df)}: {row['case_id']}",
                  end="\r", flush=True)
            f = extract_features(row, use_mask_crop=args.use_mask_crop)
            f["case_id"] = row["case_id"]
            f["label"] = int(row["label"])
            feats_list.append(f)
        print()
        return pd.DataFrame(feats_list)

    print("Extracting features...")
    tr_feats = extract_df(tr, "train")
    va_feats = extract_df(va, "val")
    te_feats = extract_df(te, "test")
    feat_time = round(time.time() - t0, 1)
    print(f"Feature extraction: {feat_time}s")

    # Save features
    tr_feats.to_csv(out / "features_train.csv", index=False)
    te_feats.to_csv(out / "features_test.csv", index=False)

    # --- Prepare arrays ---
    feat_cols = [c for c in tr_feats.columns if c not in ("case_id", "label")]
    X_train = tr_feats[feat_cols].values
    y_train = tr_feats["label"].values
    X_val = va_feats[feat_cols].values
    y_val = va_feats["label"].values
    X_test = te_feats[feat_cols].values
    y_test = te_feats["label"].values

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Handle NaN/inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Features: {len(feat_cols)} ({feat_cols})")

    # --- Train classifiers ---
    classifiers = {
        "logistic_regression": LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000,
            random_state=args.seed),
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", max_depth=10,
            random_state=args.seed),
        "svm_rbf": SVC(
            kernel="rbf", C=1.0, class_weight="balanced",
            probability=True, random_state=args.seed),
    }

    results = {}
    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        t1 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_train, y_train)
        train_secs = round(time.time() - t1, 2)

        # Predictions
        y_prob_test = clf.predict_proba(X_test)[:, 1]
        y_pred_test = (y_prob_test >= 0.5).astype(int)

        y_prob_val = clf.predict_proba(X_val)[:, 1]
        y_pred_val = (y_prob_val >= 0.5).astype(int)

        # Metrics
        has_both = len(np.unique(y_test)) > 1
        test_auroc = float(roc_auc_score(y_test, y_prob_test)) if has_both else 0.0
        test_auprc = float(average_precision_score(y_test, y_prob_test)) if has_both else 0.0
        test_f1 = float(f1_score(y_test, y_pred_test, zero_division=0))
        test_bal_acc = float(balanced_accuracy_score(y_test, y_pred_test))

        has_both_val = len(np.unique(y_val)) > 1
        val_auroc = float(roc_auc_score(y_val, y_prob_val)) if has_both_val else 0.0

        metrics = {
            "train_secs": train_secs,
            "val_auroc": round(val_auroc, 4),
            "test_auroc": round(test_auroc, 4),
            "test_auprc": round(test_auprc, 4),
            "test_f1": round(test_f1, 4),
            "test_bal_acc": round(test_bal_acc, 4),
        }
        results[name] = metrics
        print(f"  {name}: AUROC={test_auroc:.4f}, F1={test_f1:.4f}, "
              f"Bal.Acc={test_bal_acc:.4f} ({train_secs}s)")

        # Save per-sample test predictions
        pred_df = pd.DataFrame({
            "case_id": te_feats["case_id"].values,
            "y_true": y_test.astype(int),
            "y_prob": np.round(y_prob_test, 6),
        })
        pred_df.to_csv(out / f"test_predictions_{name}.csv", index=False)

        # Also save feature importances for RF
        if name == "random_forest":
            imp = pd.DataFrame({
                "feature": feat_cols,
                "importance": clf.feature_importances_,
            }).sort_values("importance", ascending=False)
            imp.to_csv(out / "feature_importances_rf.csv", index=False)
            print(f"  Top features: {imp.head(5).to_dict('records')}")

    # --- Summary ---
    summary = {
        "n_features": len(feat_cols),
        "feature_names": feat_cols,
        "n_train": len(tr),
        "n_val": len(va),
        "n_test": len(te),
        "feat_extraction_secs": feat_time,
        "results": results,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "n_folds": args.n_folds,
        "fold": args.fold,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to {out / 'summary.json'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
