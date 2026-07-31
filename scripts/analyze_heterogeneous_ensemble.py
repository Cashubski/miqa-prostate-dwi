"""Heterogeneous architecture ensembles vs homogeneous seed ensembles.

Hypothesis: combining DIFFERENT architectures (DenseNet + ResNet-18 + ResNet-50)
produces more diverse predictions than 3 seeds of the SAME architecture, which
restores informative ensemble uncertainty (higher UE-AUROC) even where a
homogeneous ensemble fails (e.g. DenseNet, UE-AUROC ~0.52).

Pure analysis on existing test_predictions.csv (no GPU). Per fold, the test set
is identical across architectures (same split_seed=42, n_folds=5), so we can
align predictions by case_id and combine across architectures.

Compares (per fold, aggregated over 5 folds):
  - Homogeneous DenseNet   : 3 seeds of densenet121
  - Homogeneous ResNet-18  : 3 seeds of resnet18
  - Heterogeneous (1 seed) : densenet121 + resnet18 + resnet50, seed 42
  - Heterogeneous (3 seed) : all 3 seeds x all 3 archs (9 models)

Metrics: AUROC, UE-AUROC (uncertainty-error AUROC via std), selective-prediction
accuracy at 70/80/90% coverage.

Usage:
    python repo/scripts/analyze_heterogeneous_ensemble.py
"""
import glob
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RUNS = os.environ.get("MIQA_RUNS", "/raid/candi/michal/miqa/runs")
SEED_SUFFIXES = ["", "_s43", "_s44"]
FOLDS = [0, 1, 2, 3, 4]


def find_pred_csv(arch, suffix, fold):
    pat = f"{RUNS}/poster100_{arch}_dwi_roi{suffix}_f{fold}*/test_predictions.csv"
    hits = sorted(glob.glob(pat))
    return hits[0] if hits else None


def load_preds(path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    prob_col = next((cols[c] for c in cols if "prob" in c), None)
    true_col = next((cols[c] for c in cols if "true" in c or c == "label"), None)
    id_col = next((cols[c] for c in cols if "case" in c or "id" in c), None)
    out = df[[id_col, true_col, prob_col]].copy()
    out.columns = ["case_id", "y_true", "y_prob"]
    return out.set_index("case_id")


def ue_auroc(y_true, y_pred, uncertainty):
    """Can uncertainty rank errors? AUROC of uncertainty predicting is_error."""
    errors = (y_true != y_pred).astype(int)
    if len(np.unique(errors)) < 2:
        return np.nan
    return roc_auc_score(errors, uncertainty)


def selective_accuracy(y_true, y_prob, uncertainty, coverage):
    """Accuracy on the (coverage) fraction of most-confident (lowest-unc) cases."""
    n_keep = max(1, int(round(coverage * len(y_true))))
    order = np.argsort(uncertainty)  # ascending uncertainty
    keep = order[:n_keep]
    y_pred = (y_prob[keep] >= 0.5).astype(int)
    return np.mean(y_pred == y_true[keep])


def collect(spec):
    """spec: list of (arch, suffix). Returns per-fold aggregated metrics
    by treating each (arch, suffix) as one ensemble member."""
    aurocs, ueaurocs = [], []
    sel = {0.9: [], 0.8: [], 0.7: []}
    for fold in FOLDS:
        members = []
        true_ser = None
        for arch, suf in spec:
            p = find_pred_csv(arch, suf, fold)
            if p:
                d = load_preds(p)
                members.append(d["y_prob"])
                true_ser = d["y_true"]
        if len(members) < 2 or true_ser is None:
            continue
        M = pd.concat(members, axis=1, join="inner")
        idx = M.index
        probs = M.values  # (n_cases, n_members)
        mean_prob = probs.mean(axis=1)
        std_prob = probs.std(axis=1)
        y_true = true_ser.reindex(idx).values
        y_pred = (mean_prob >= 0.5).astype(int)
        if len(np.unique(y_true)) > 1:
            aurocs.append(roc_auc_score(y_true, mean_prob))
        ueaurocs.append(ue_auroc(y_true, y_pred, std_prob))
        for cov in sel:
            sel[cov].append(selective_accuracy(y_true, mean_prob, std_prob, cov))
    return aurocs, ueaurocs, sel


def report(name, spec):
    aurocs, ueaurocs, sel = collect(spec)
    ue = [u for u in ueaurocs if not np.isnan(u)]
    a_m = np.mean(aurocs) if aurocs else float("nan")
    a_s = np.std(aurocs) if aurocs else float("nan")
    u_m = np.mean(ue) if ue else float("nan")
    u_s = np.std(ue) if ue else float("nan")
    s70 = np.mean(sel[0.7]) if sel[0.7] else float("nan")
    s80 = np.mean(sel[0.8]) if sel[0.8] else float("nan")
    s90 = np.mean(sel[0.9]) if sel[0.9] else float("nan")
    print(f"{name:<32}{a_m:.3f}+/-{a_s:.3f}  {u_m:.3f}+/-{u_s:.3f}   "
          f"{s90:.3f}  {s80:.3f}  {s70:.3f}")


def main():
    print("=" * 86)
    print("HETEROGENEOUS vs HOMOGENEOUS ENSEMBLES")
    print("=" * 86)
    print(f"{'ensemble':<32}{'AUROC':<14}{'UE-AUROC':<14}"
          f"{'sel@90':<8}{'sel@80':<8}{'sel@70'}")
    print("-" * 86)

    # Homogeneous: 3 seeds of one architecture
    report("Homogeneous DenseNet (3 seed)",
           [("densenet121", s) for s in SEED_SUFFIXES])
    report("Homogeneous ResNet-18 (3 seed)",
           [("resnet18", s) for s in SEED_SUFFIXES])
    report("Homogeneous ResNet-50 (3 seed)",
           [("resnet50", s) for s in SEED_SUFFIXES])

    print("-" * 86)
    # Heterogeneous: different architectures
    report("Heterogeneous 3-arch (seed42)",
           [("densenet121", ""), ("resnet18", ""), ("resnet50", "")])
    report("Heterogeneous 9-model (3arch x 3seed)",
           [(a, s) for a in ["densenet121", "resnet18", "resnet50"]
            for s in SEED_SUFFIXES])

    print()
    print("=" * 86)
    print("KEY COMPARISON:")
    print(" If Heterogeneous UE-AUROC >> Homogeneous DenseNet UE-AUROC (~0.52),")
    print(" then architecture diversity restores informative uncertainty.")
    print("=" * 86)


if __name__ == "__main__":
    main()
