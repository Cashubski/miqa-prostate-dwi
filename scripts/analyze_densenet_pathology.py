"""Why does DenseNet-121's selective prediction stay flat (UE-AUROC ~0.52)?

Checks three explanations from the saved test_predictions.csv (no GPU):
  H1 homogeneity: the 3 seed variants predict almost identically, so the
     ensemble std barely reflects error. Measured by pairwise prob correlation.
  H2 overconfidence: predictions pile up near 0/1. Measured by the fraction
     falling in [0, 0.1] or [0.9, 1].
  H3 confidence-error decoupling: confidence on correct vs wrong cases looks
     similar, so it cannot flag errors. Measured by mean |p-0.5| on each.

Reads poster100_<arch>_dwi_roi[_s43|_s44]_f<fold>/test_predictions.csv
(seed 42 has no suffix; s43 and s44 are the other two). Compares densenet121
against resnet18 and resnet50.

Run: python scripts/analyze_densenet_pathology.py
"""
import glob
import os
import numpy as np
import pandas as pd

RUNS = os.environ.get("MIQA_RUNS", "/raid/candi/michal/miqa/runs")
ARCHS = ["densenet121", "resnet18", "resnet50"]
SEED_SUFFIXES = ["", "_s43", "_s44"]  # seed 42 (no suffix), 43, 44
FOLDS = [0, 1, 2, 3, 4]


def find_pred_csv(arch, suffix, fold):
    """Find test_predictions.csv for one arch/seed/fold. The run dir may carry
    a trailing timestamp, so glob for it."""
    pat = f"{RUNS}/poster100_{arch}_dwi_roi{suffix}_f{fold}*/test_predictions.csv"
    hits = sorted(glob.glob(pat))
    return hits[0] if hits else None


def load_preds(path):
    df = pd.read_csv(path)
    # normalise column names
    cols = {c.lower(): c for c in df.columns}
    prob_col = next((cols[c] for c in cols if "prob" in c), None)
    true_col = next((cols[c] for c in cols if "true" in c or c == "label"), None)
    id_col = next((cols[c] for c in cols if "case" in c or "id" in c), None)
    if prob_col is None or true_col is None or id_col is None:
        raise ValueError(f"Unexpected columns in {path}: {list(df.columns)}")
    out = df[[id_col, true_col, prob_col]].copy()
    out.columns = ["case_id", "y_true", "y_prob"]
    return out


def main():
    print("=" * 78)
    print("DENSENET UQ PATHOLOGY INVESTIGATION")
    print("=" * 78)

    # ---------- H1: ensemble homogeneity (inter-seed correlation) ----------
    print("\n[H1] ENSEMBLE HOMOGENEITY: pairwise correlation of seed predictions")
    print("     (higher means more homogeneous, so ensemble std is less informative)")
    print("-" * 78)
    print(f"{'arch':<14}{'mean_pairwise_r':<18}{'std':<10}{'interpretation'}")
    homogeneity = {}
    for arch in ARCHS:
        fold_corrs = []
        for fold in FOLDS:
            seed_dfs = []
            for suf in SEED_SUFFIXES:
                p = find_pred_csv(arch, suf, fold)
                if p:
                    seed_dfs.append(load_preds(p).set_index("case_id")["y_prob"])
            if len(seed_dfs) < 2:
                continue
            merged = pd.concat(seed_dfs, axis=1, join="inner")
            merged.columns = [f"s{i}" for i in range(merged.shape[1])]
            # mean of all pairwise correlations
            corr = merged.corr().values
            iu = np.triu_indices_from(corr, k=1)
            fold_corrs.append(np.nanmean(corr[iu]))
        if fold_corrs:
            m, s = np.mean(fold_corrs), np.std(fold_corrs)
            homogeneity[arch] = m
            tag = "HOMOGENEOUS" if m > 0.85 else "diverse" if m < 0.7 else "moderate"
            print(f"{arch:<14}{m:<18.4f}{s:<10.4f}{tag}")

    # ---------- H2: overconfidence (probability saturation) ----------
    print("\n[H2] OVERCONFIDENCE: fraction of predictions saturated to 0/1")
    print("     (higher means more overconfident; ensemble-mean probabilities)")
    print("-" * 78)
    print(f"{'arch':<14}{'frac_saturated':<18}{'mean_conf|p-0.5|':<18}")
    for arch in ARCHS:
        all_probs = []
        for fold in FOLDS:
            # average the available seeds = ensemble-mean probability
            seed_dfs = []
            for suf in SEED_SUFFIXES:
                p = find_pred_csv(arch, suf, fold)
                if p:
                    seed_dfs.append(load_preds(p).set_index("case_id")["y_prob"])
            if not seed_dfs:
                continue
            mean_prob = pd.concat(seed_dfs, axis=1, join="inner").mean(axis=1)
            all_probs.extend(mean_prob.tolist())
        if all_probs:
            arr = np.array(all_probs)
            frac_sat = np.mean((arr < 0.1) | (arr > 0.9))
            mean_conf = np.mean(np.abs(arr - 0.5))
            print(f"{arch:<14}{frac_sat:<18.4f}{mean_conf:<18.4f}")

    # ---------- H3: confidence-error decoupling ----------
    print("\n[H3] CONFIDENCE-ERROR DECOUPLING: confidence on correct vs wrong")
    print("     (if correct and wrong look alike, uncertainty cannot flag errors)")
    print("-" * 78)
    print(f"{'arch':<14}{'conf_correct':<16}{'conf_wrong':<14}{'gap':<10}{'interpretation'}")
    for arch in ARCHS:
        conf_correct, conf_wrong = [], []
        for fold in FOLDS:
            seed_dfs, true_ser = [], None
            for suf in SEED_SUFFIXES:
                p = find_pred_csv(arch, suf, fold)
                if p:
                    d = load_preds(p).set_index("case_id")
                    seed_dfs.append(d["y_prob"])
                    true_ser = d["y_true"]
            if not seed_dfs or true_ser is None:
                continue
            mean_prob = pd.concat(seed_dfs, axis=1, join="inner").mean(axis=1)
            y_true = true_ser.reindex(mean_prob.index)
            y_pred = (mean_prob >= 0.5).astype(int)
            conf = np.abs(mean_prob - 0.5)
            correct = y_pred == y_true
            conf_correct.extend(conf[correct].tolist())
            conf_wrong.extend(conf[~correct].tolist())
        if conf_correct and conf_wrong:
            cc, cw = np.mean(conf_correct), np.mean(conf_wrong)
            gap = cc - cw
            tag = "GOOD (correct>>wrong)" if gap > 0.08 else "DECOUPLED (~equal)" if gap < 0.04 else "moderate"
            print(f"{arch:<14}{cc:<16.4f}{cw:<14.4f}{gap:+.4f}   {tag}")

    print()
    print("=" * 78)
    print("READING THE RESULTS:")
    print(" - Higher DenseNet inter-seed correlation (H1) than ResNet-18 means a")
    print("   homogeneous ensemble, so std is uninformative and selective stays flat.")
    print(" - Higher DenseNet saturation (H2) would mean it is overconfident.")
    print(" - A smaller correct-vs-wrong confidence gap (H3) would mean its")
    print("   confidence cannot distinguish errors.")
    print("=" * 78)


if __name__ == "__main__":
    main()
