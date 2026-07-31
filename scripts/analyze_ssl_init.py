"""Paired Wilcoxon test: Random init vs ImageNet init for SSL fine-tuning.

Compares ft_<method>_<arch>_rnd_seed{S}_f{F} vs ft_<method>_<arch>_imn_seed{S}_f{F}
across all (seed, fold) pairs within each (arch, method) cell.

Usage:
    python repo/scripts/analyze_ssl_init.py
"""
import json
import glob
import numpy as np
from scipy.stats import wilcoxon

RUNS = "/raid/candi/michal/miqa/runs"
ARCHS = ["resnet18", "resnet50"]
METHODS = ["simclr", "byol"]
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
    print("PAIRED WILCOXON: ImageNet init vs Random init (within arch x SSL method)")
    print("=" * 78)
    print(f"{'Config':<25} {'n_pairs':<10} {'mean(imn)':<12} {'mean(rnd)':<12} "
          f"{'delta':<10} {'p-value':<10} {'sig?':<5}")
    print("-" * 78)

    for arch in ARCHS:
        for method in METHODS:
            rnd_aurocs, imn_aurocs = [], []
            for seed in SEEDS:
                for fold in FOLDS:
                    rnd_glob = glob.glob(
                        f"{RUNS}/ft_{method}_{arch}_rnd_seed{seed}_f{fold}_*/summary.json"
                    )
                    imn_glob = glob.glob(
                        f"{RUNS}/ft_{method}_{arch}_imn_seed{seed}_f{fold}_*/summary.json"
                    )
                    if not rnd_glob or not imn_glob:
                        continue
                    rd = load(rnd_glob[0])
                    id_ = load(imn_glob[0])
                    if rd is None or id_ is None:
                        continue
                    ra = get_test_auroc(rd)
                    ia = get_test_auroc(id_)
                    if ra is None or ia is None:
                        continue
                    rnd_aurocs.append(ra)
                    imn_aurocs.append(ia)

            if not rnd_aurocs:
                print(f"{arch} {method:<6}: no matched pairs")
                continue

            rnd_arr = np.array(rnd_aurocs)
            imn_arr = np.array(imn_aurocs)
            delta = np.mean(imn_arr - rnd_arr)
            try:
                stat, p = wilcoxon(imn_arr, rnd_arr)
            except ValueError:
                p = float("nan")

            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            label = f"{arch} {method}"
            print(f"{label:<25} {len(rnd_arr):<10} {imn_arr.mean():<12.4f} "
                  f"{rnd_arr.mean():<12.4f} {delta:+.4f}    {p:<10.4f} {sig}")

    print()
    print("Legend: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    print("Positive delta = ImageNet+SSL outperforms SSL-from-random-init")


if __name__ == "__main__":
    main()
