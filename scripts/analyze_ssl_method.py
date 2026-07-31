"""Paired Wilcoxon test: SimCLR vs BYOL for SSL fine-tuning.

Compares ft_simclr_<arch>_<init>_seed{S}_f{F} vs ft_byol_<arch>_<init>_seed{S}_f{F}
across all (seed, fold) pairs within each (arch, init) cell.

Usage:
    python repo/scripts/analyze_ssl_method.py
"""
import json
import glob
import numpy as np
from scipy.stats import wilcoxon

RUNS = "/raid/candi/michal/miqa/runs"
ARCHS = ["resnet18", "resnet50"]
INITS = ["rnd", "imn"]
SEEDS = [1, 2, 3, 4, 5]
FOLDS = [0, 1, 2, 3, 4]


def load(path):
    try:
        d = json.load(open(path))
        return None if "error" in d else d
    except Exception:
        return None


def get_test_auroc(d):
    return d.get("test", {}).get("auroc")


def main():
    print("=" * 78)
    print("PAIRED WILCOXON: SimCLR vs BYOL (within arch x init)")
    print("=" * 78)
    print(f"{'Config':<25} {'n_pairs':<10} {'mean(simclr)':<14} {'mean(byol)':<12} "
          f"{'delta':<10} {'p-value':<10} {'sig?':<5}")
    print("-" * 78)

    for arch in ARCHS:
        for init in INITS:
            sim_aurocs, byo_aurocs = [], []
            for seed in SEEDS:
                for fold in FOLDS:
                    s_glob = glob.glob(
                        f"{RUNS}/ft_simclr_{arch}_{init}_seed{seed}_f{fold}_*/summary.json"
                    )
                    b_glob = glob.glob(
                        f"{RUNS}/ft_byol_{arch}_{init}_seed{seed}_f{fold}_*/summary.json"
                    )
                    if not s_glob or not b_glob:
                        continue
                    sd = load(s_glob[0])
                    bd = load(b_glob[0])
                    if sd is None or bd is None:
                        continue
                    sa = get_test_auroc(sd)
                    ba = get_test_auroc(bd)
                    if sa is None or ba is None:
                        continue
                    sim_aurocs.append(sa)
                    byo_aurocs.append(ba)

            if not sim_aurocs:
                print(f"{arch} {init}: no matched pairs")
                continue

            sim_arr = np.array(sim_aurocs)
            byo_arr = np.array(byo_aurocs)
            delta = np.mean(sim_arr - byo_arr)
            try:
                stat, p = wilcoxon(sim_arr, byo_arr)
            except ValueError:
                p = float("nan")

            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            label = f"{arch} {init}"
            print(f"{label:<25} {len(sim_arr):<10} {sim_arr.mean():<14.4f} "
                  f"{byo_arr.mean():<12.4f} {delta:+.4f}    {p:<10.4f} {sig}")

    print()
    print("Legend: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    print("Positive delta = SimCLR outperforms BYOL")


if __name__ == "__main__":
    main()
