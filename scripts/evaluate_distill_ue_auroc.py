"""Compute UE-AUROC + selective accuracy for distilled ResNet-18 students.

Loops over the five distill_resnet18_f{fold}/best_auroc.pt checkpoints, runs
inference on each fold's test split, and reports:
  - AUROC per fold + overall (should match Ch 8.7's 0.782)
  - UE-AUROC per fold + overall (predictive entropy as uncertainty)
  - UE-AUROC using |p-0.5| (softmax-margin proxy) as uncertainty
  - Selective accuracy at 70% coverage
  - 95% bootstrap CIs on the pooled numbers

Output: prints to stdout + saves logs/distill_student_ue_auroc.json

Run:
  cd /raid/candi/michal/miqa
  python repo/scripts/evaluate_distill_ue_auroc.py

No CLI flags; uses fixed paths matching the project layout.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from miqa.datasets import UCLDistortionDataset
from miqa.models import build_model

ROOT = Path("/raid/candi/michal/miqa")
INDEX_CSV = ROOT / "repo" / "dataset" / "index_full.csv"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


def load_fold_test_split(df, run_dir, fold):
    """Load fold-specific test split from kfold_splits.json.
    Splits are stored by patient_study id (e.g. 'Patient017320846_study_0');
    case_ids in the dataframe are patient_study + '__<bval>-<idx>' (e.g.
    '...study_0__1k4-0'). We match by stripping the '__<bval>-<idx>' suffix.
    """
    splits_path = Path(run_dir) / "kfold_splits.json"
    with open(splits_path) as f:
        splits = json.load(f)
    test_patient_studies = set(splits["folds"][str(fold)]["test"])
    patient_study = df["case_id"].str.rsplit("__", n=1).str[0]
    te = df[patient_study.isin(test_patient_studies)].reset_index(drop=True)
    print(f"  kfold_splits fold {fold}: {len(test_patient_studies)} patient-studies -> {len(te)} scans")
    return te


def bootstrap_ci(y_true, scores, n_boot=1000, seed=0):
    """95% percentile bootstrap CI on AUROC."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aurocs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aurocs.append(roc_auc_score(y_true[idx], scores[idx]))
    return float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))


def main():
    df = pd.read_csv(INDEX_CSV)
    print(f"Loaded index: {len(df)} rows")

    all_y, all_p, all_entropy, all_margin, all_fold = [], [], [], [], []

    per_fold = []

    for fold in range(5):
        run_dir = ROOT / "runs" / f"distill_resnet18_f{fold}"
        ckpt_path = run_dir / "best_auroc.pt"
        if not ckpt_path.exists():
            print(f"WARN: {ckpt_path} missing -- skipping fold {fold}")
            continue

        print(f"\n=== Fold {fold} ===")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt_args = ckpt.get("args", {})
        model_name = ckpt_args.get("model", "resnet18")
        k_slices = ckpt_args.get("k_slices", 3)
        modality = ckpt_args.get("modality", "dwi")
        use_mask_crop = ckpt_args.get("use_mask_crop", True)
        in_chans = k_slices * 2 if modality == "fusion" else k_slices

        te = load_fold_test_split(df, run_dir, fold)
        print(f"  test size: {len(te)}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_model(model_name, in_chans=in_chans,
                            pretrained=False).to(device)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()

        ds = UCLDistortionDataset(te, modality=modality,
                                  use_mask_crop=use_mask_crop,
                                  k_slices=k_slices, img_size=224,
                                  return_case_id=True)
        dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2,
                        pin_memory=True)

        ys, probs = [], []
        with torch.no_grad():
            for batch in dl:
                x = batch[0].to(device, non_blocking=True)
                y = batch[1]
                logit = model(x).cpu().numpy().reshape(-1)
                prob = 1.0 / (1.0 + np.exp(-logit))
                ys.append(y.numpy().reshape(-1))
                probs.append(prob)

        y = np.concatenate(ys).astype(int)
        p = np.concatenate(probs)
        pred = (p >= 0.5).astype(int)
        correct = (pred == y).astype(int)

        # Two uncertainty measures
        eps = 1e-9
        entropy = -p * np.log(p + eps) - (1 - p) * np.log(1 - p + eps)
        margin_unc = 1.0 - np.abs(p - 0.5) * 2  # higher = more uncertain

        # AUROC (should match ~0.782 from thesis Ch 8.7)
        auroc = roc_auc_score(y, p)

        # UE-AUROC: uncertainty ranks errors above corrects
        errors = 1 - correct
        if errors.sum() > 0 and correct.sum() > 0:
            ue_entropy = roc_auc_score(errors, entropy)
            ue_margin = roc_auc_score(errors, margin_unc)
        else:
            ue_entropy, ue_margin = np.nan, np.nan

        # Selective prediction at 70% coverage: keep top-70% by confidence
        conf = -entropy  # higher = more confident (lower entropy)
        n_keep = int(0.7 * len(conf))
        order = np.argsort(conf)[::-1]
        keep = order[:n_keep]
        sel_acc = correct[keep].mean()

        print(f"  AUROC       = {auroc:.4f}")
        print(f"  UE-AUROC    = {ue_entropy:.4f}  (entropy)")
        print(f"  UE-AUROC    = {ue_margin:.4f}  (|p-0.5| margin)")
        print(f"  sel_acc@70% = {sel_acc:.4f}")

        per_fold.append({
            "fold": fold,
            "n_test": int(len(y)),
            "auroc": float(auroc),
            "ue_auroc_entropy": float(ue_entropy),
            "ue_auroc_margin": float(ue_margin),
            "sel_acc_70": float(sel_acc),
        })

        all_y.append(y)
        all_p.append(p)
        all_entropy.append(entropy)
        all_margin.append(margin_unc)
        all_fold.append(np.full_like(y, fold))

    # --- Pooled analysis ---
    y_all = np.concatenate(all_y)
    p_all = np.concatenate(all_p)
    ent_all = np.concatenate(all_entropy)
    mar_all = np.concatenate(all_margin)
    correct_all = ((p_all >= 0.5).astype(int) == y_all).astype(int)
    errors_all = 1 - correct_all

    pooled_auroc = roc_auc_score(y_all, p_all)
    pooled_ue_ent = roc_auc_score(errors_all, ent_all)
    pooled_ue_mar = roc_auc_score(errors_all, mar_all)

    # Selective @ 70%
    conf_all = -ent_all
    n_keep = int(0.7 * len(conf_all))
    keep_all = np.argsort(conf_all)[::-1][:n_keep]
    pooled_sel = correct_all[keep_all].mean()

    # 95% bootstrap CIs
    auroc_ci = bootstrap_ci(y_all, p_all, n_boot=1000)
    ue_ent_ci = bootstrap_ci(errors_all, ent_all, n_boot=1000)
    ue_mar_ci = bootstrap_ci(errors_all, mar_all, n_boot=1000)

    aurocs = np.array([r["auroc"] for r in per_fold])
    ue_ents = np.array([r["ue_auroc_entropy"] for r in per_fold])
    ue_mars = np.array([r["ue_auroc_margin"] for r in per_fold])
    sels = np.array([r["sel_acc_70"] for r in per_fold])

    print()
    print("=" * 60)
    print("DISTILLED STUDENT (ResNet-18) SUMMARY -- 5 folds")
    print("=" * 60)
    print(f"AUROC per-fold mean +/- std : {aurocs.mean():.4f} +/- {aurocs.std():.4f}")
    print(f"AUROC pooled (95% CI)       : {pooled_auroc:.4f} [{auroc_ci[0]:.4f}, {auroc_ci[1]:.4f}]")
    print()
    print(f"UE-AUROC per-fold (entropy) : {ue_ents.mean():.4f} +/- {ue_ents.std():.4f}")
    print(f"UE-AUROC pooled (entropy)   : {pooled_ue_ent:.4f} [{ue_ent_ci[0]:.4f}, {ue_ent_ci[1]:.4f}]")
    print()
    print(f"UE-AUROC per-fold (|p-0.5|) : {ue_mars.mean():.4f} +/- {ue_mars.std():.4f}")
    print(f"UE-AUROC pooled (|p-0.5|)   : {pooled_ue_mar:.4f} [{ue_mar_ci[0]:.4f}, {ue_mar_ci[1]:.4f}]")
    print()
    print(f"Sel acc @ 70% coverage      : {sels.mean():.4f} +/- {sels.std():.4f}")
    print(f"Sel acc pooled              : {pooled_sel:.4f}")
    print()
    print(f"Reference (heterogeneous ensemble teacher): AUROC 0.802, UE-AUROC 0.685, sel_acc 0.836")

    out = LOG_DIR / "distill_student_ue_auroc.json"
    payload = {
        "per_fold": per_fold,
        "summary": {
            "n_folds": len(per_fold),
            "auroc_mean": float(aurocs.mean()),
            "auroc_std": float(aurocs.std()),
            "auroc_pooled": float(pooled_auroc),
            "auroc_95ci": list(auroc_ci),
            "ue_auroc_entropy_mean": float(ue_ents.mean()),
            "ue_auroc_entropy_std": float(ue_ents.std()),
            "ue_auroc_entropy_pooled": float(pooled_ue_ent),
            "ue_auroc_entropy_95ci": list(ue_ent_ci),
            "ue_auroc_margin_mean": float(ue_mars.mean()),
            "ue_auroc_margin_std": float(ue_mars.std()),
            "ue_auroc_margin_pooled": float(pooled_ue_mar),
            "ue_auroc_margin_95ci": list(ue_mar_ci),
            "sel_acc_70_mean": float(sels.mean()),
            "sel_acc_70_std": float(sels.std()),
            "sel_acc_70_pooled": float(pooled_sel),
        },
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
