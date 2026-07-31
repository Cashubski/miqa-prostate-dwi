"""Calibration analysis: Expected Calibration Error (ECE) per architecture,
before and after temperature scaling.

ECE measures whether predicted probabilities match empirical accuracy
(a model predicting 0.8 should be correct 80% of the time). Temperature
scaling divides logits by a learned scalar T>1 to soften overconfident
predictions. We recover logits from saved probabilities via the logit
function, fit T on each fold's predictions (minimising NLL), and recompute ECE.

Pure analysis on existing ensemble-mean test_predictions.csv (no GPU).

Usage:
    python repo/scripts/analyze_calibration.py
"""
import glob
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

RUNS = os.environ.get("MIQA_RUNS", "/raid/candi/michal/miqa/runs")
ARCHS = ["densenet121", "resnet18", "resnet50"]
SEED_SUFFIXES = ["", "_s43", "_s44"]
FOLDS = [0, 1, 2, 3, 4]
N_BINS = 10
EPS = 1e-6


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


def ece(probs, labels, n_bins=N_BINS):
    """Expected Calibration Error (binary)."""
    conf = np.where(probs >= 0.5, probs, 1 - probs)
    pred = (probs >= 0.5).astype(int)
    acc = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    n = len(probs)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 0:
            e += (m.sum() / n) * abs(acc[m].mean() - conf[m].mean())
    return e


def nll_temp(T, logits, labels):
    """Negative log-likelihood after dividing logits by T (binary)."""
    z = logits / T
    p = 1.0 / (1.0 + np.exp(-z))
    p = np.clip(p, EPS, 1 - EPS)
    return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))


def main():
    print("=" * 78)
    print("CALIBRATION ANALYSIS: ECE before/after temperature scaling")
    print("=" * 78)
    print(f"{'arch':<14}{'ECE_raw':<12}{'ECE_tempscaled':<16}{'mean_T':<10}{'improvement'}")
    print("-" * 78)

    for arch in ARCHS:
        ece_raw_folds, ece_ts_folds, T_folds = [], [], []
        for fold in FOLDS:
            seed_dfs, true_ser = [], None
            for suf in SEED_SUFFIXES:
                p = find_pred_csv(arch, suf, fold)
                if p:
                    d = load_preds(p)
                    seed_dfs.append(d["y_prob"])
                    true_ser = d["y_true"]
            if not seed_dfs or true_ser is None:
                continue
            mean_prob = pd.concat(seed_dfs, axis=1, join="inner").mean(axis=1)
            labels = true_ser.reindex(mean_prob.index).values.astype(float)
            probs = np.clip(mean_prob.values, EPS, 1 - EPS)

            # raw ECE
            ece_raw_folds.append(ece(probs, labels))

            # recover logits, fit temperature
            logits = np.log(probs / (1 - probs))
            res = minimize_scalar(nll_temp, bounds=(0.1, 10.0),
                                  args=(logits, labels), method="bounded")
            T = res.x
            T_folds.append(T)
            p_ts = 1.0 / (1.0 + np.exp(-logits / T))
            ece_ts_folds.append(ece(p_ts, labels))

        if ece_raw_folds:
            er = np.mean(ece_raw_folds)
            et = np.mean(ece_ts_folds)
            tm = np.mean(T_folds)
            imp = er - et
            tag = "improved" if imp > 0.005 else "no change" if abs(imp) <= 0.005 else "worse"
            print(f"{arch:<14}{er:<12.4f}{et:<16.4f}{tm:<10.3f}{imp:+.4f} ({tag})")

    print()
    print("=" * 78)
    print("READING THE RESULTS:")
    print(" - Higher ECE_raw = more miscalibrated (predicted prob != accuracy).")
    print(" - T > 1 = model was overconfident (temperature softens it).")
    print(" - If DenseNet has highest ECE_raw and highest T, that quantifies")
    print("   its overconfidence (links to the UQ pathology finding).")
    print(" - If temp scaling improves ECE but NOT selective prediction,")
    print("   that shows calibration != useful uncertainty ranking.")
    print("=" * 78)


if __name__ == "__main__":
    main()
