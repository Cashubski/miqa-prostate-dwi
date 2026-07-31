"""Generate all key thesis figures from existing JSON/CSV results.

Produces PNG + PDF for each figure into figures/thesis/.

Uses MIQA_RUNS env var (defaults to pt8 path). Override locally:
    MIQA_RUNS="/Volumes/Extreme SSD/UCL/miqa_backup/runs" \
        python repo/scripts/make_thesis_figures.py

Each figure is produced by an independent function so you can iterate quickly
on style. Re-run after edits to refresh.
"""
import json
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Consistent colour palette across all figures
COLORS = {
    "densenet121": "#1f77b4",   # blue
    "resnet18":    "#ff7f0e",   # orange
    "resnet50":    "#2ca02c",   # green
    "efficientnet_b0": "#9467bd",  # purple
    "deep_ensemble":   "#1f77b4",
    "mc_dropout":      "#d62728",   # red
    "tta":             "#2ca02c",
    "heterogeneous":   "#ff7f0e",
}
ARCH_LABEL = {
    "densenet121": "DenseNet-121",
    "resnet18":    "ResNet-18",
    "resnet50":    "ResNet-50",
    "efficientnet_b0": "EfficientNet-B0",
}

RUNS = os.environ.get("MIQA_RUNS", "/raid/candi/michal/miqa/runs")
OUT = Path(os.environ.get("MIQA_FIGS", "figures/thesis"))
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig, name):
    """Save figure as both PNG and PDF."""
    p_png = OUT / f"{name}.png"
    p_pdf = OUT / f"{name}.pdf"
    fig.savefig(p_png)
    fig.savefig(p_pdf)
    plt.close(fig)
    print(f"  -> {p_png.name} (+ .pdf)")


def _interp_sel(d, cov):
    """Interpolate selective prediction accuracy at the given coverage."""
    c = d["selective_prediction"]["coverage"][::-1]
    a = d["selective_prediction"]["accuracy"][::-1]
    return float(np.interp(cov, c, a))


def _load_jsons(pattern):
    """Glob a pattern and return a list of parsed JSON dicts (skipping errors)."""
    files = sorted(glob.glob(pattern))
    out = []
    for f in files:
        try:
            d = json.load(open(f))
            if "error" not in d:
                out.append(d)
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# Figure 4.1: Architecture comparison
# ---------------------------------------------------------------------------

def fig4_architecture_comparison():
    """Bar chart: classical baselines vs DL architectures (mean +/- std AUROC)."""
    # Search common locations for poster_results.json
    candidates = [
        Path(os.environ.get("MIQA_POSTER_JSON", "")) if os.environ.get("MIQA_POSTER_JSON") else None,
        Path(RUNS).parent / "poster_results.json",
        Path(RUNS).parent.parent / "poster_results.json",
        Path.cwd() / "poster_results.json",
    ]
    poster_json = next((p for p in candidates if p and p.exists()), None)
    if poster_json is not None:
        data = json.load(open(poster_json))
        labels, means, stds = [], [], []
        order = [("classical_logistic_regression", "Logistic Regression"),
                 ("classical_random_forest", "Random Forest"),
                 ("classical_svm_rbf", "SVM-RBF"),
                 ("resnet50_dwi_roi", "ResNet-50"),
                 ("efficientnet_b0_dwi_roi", "EfficientNet-B0"),
                 ("resnet18_dwi_roi", "ResNet-18"),
                 ("densenet121_dwi_roi", "DenseNet-121")]
        for key, label in order:
            if key in data:
                labels.append(label)
                means.append(data[key]["auroc"]["mean"])
                stds.append(data[key]["auroc"]["std"])
        n = len(labels)
        colors = (["#888"] * 3 + ["#1f77b4"] * (n - 3))[:n]
    else:
        print("  (no poster_results.json found; skipping)")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(n)
    ax.barh(x, means, xerr=stds, color=colors, alpha=0.85, capsize=3,
            edgecolor="black", linewidth=0.6)
    ax.set_yticks(x)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Test AUROC (mean ± std across folds)")
    ax.set_xlim(0.5, 0.95)
    ax.axvline(0.5, color="grey", linewidth=0.7, linestyle=":", alpha=0.7)
    ax.set_title("Architecture comparison on prostate DWI distortion detection")
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + s + 0.005, i, f"{m:.2f}", va="center", fontsize=9)
    plt.tight_layout()
    _save(fig, "fig4_architecture_comparison")


# ---------------------------------------------------------------------------
# Figure 7.1: SSL 4-way comparison
# ---------------------------------------------------------------------------

def fig7_ssl_four_way():
    """Grouped bar chart: random/ImageNet x SimCLR/BYOL across architectures."""
    rows = []
    for arch in ["resnet18", "resnet50"]:
        for method in ["simclr", "byol"]:
            for init in ["rnd", "imn"]:
                files = sorted(glob.glob(
                    f"{RUNS}/ft_{method}_{arch}_{init}_seed*_f*_*/summary.json"))
                aurocs = []
                for f in files:
                    try:
                        d = json.load(open(f))
                        if "error" in d:
                            continue
                        a = d.get("test", {}).get("auroc")
                        if a is not None:
                            aurocs.append(a)
                    except Exception:
                        pass
                if aurocs:
                    rows.append({"arch": arch, "method": method, "init": init,
                                 "mean": np.mean(aurocs), "std": np.std(aurocs),
                                 "n": len(aurocs)})
    df = pd.DataFrame(rows)
    if df.empty:
        print("  (no SSL results found; skipping)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, arch in zip(axes, ["resnet18", "resnet50"]):
        sub = df[df["arch"] == arch]
        methods = ["simclr", "byol"]
        inits = ["rnd", "imn"]
        x = np.arange(len(methods))
        w = 0.35
        for j, init in enumerate(inits):
            vals = [sub[(sub["method"] == m) & (sub["init"] == init)]
                    ["mean"].values[0] for m in methods]
            errs = [sub[(sub["method"] == m) & (sub["init"] == init)]
                    ["std"].values[0] for m in methods]
            col = "#888" if init == "rnd" else COLORS[arch]
            label = "Random init" if init == "rnd" else "ImageNet init"
            ax.bar(x + (j - 0.5) * w, vals, w, yerr=errs, label=label,
                   color=col, alpha=0.85, capsize=3,
                   edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["SimCLR", "BYOL"])
        ax.set_title(ARCH_LABEL[arch])
        ax.set_ylim(0.70, 0.85)
        ax.set_ylabel("Test AUROC (mean ± std, n=25)")
        ax.legend(loc="lower right")
    fig.suptitle("Self-supervised pretraining: 4-way comparison")
    plt.tight_layout()
    _save(fig, "fig7_ssl_four_way")


# ---------------------------------------------------------------------------
# Figure 8.1: 3-way UQ selective-prediction overlay
# ---------------------------------------------------------------------------

def fig8_uq_three_way():
    """Three-way UQ method overlay: Deep Ensemble vs MC Dropout vs TTA.

    For each method, plots mean selective-prediction curve (accuracy vs coverage)
    aggregated across 5 folds, with shaded +/- std band, for the BEST arch
    (ResNet-18 for std-based UE-AUROC).
    """
    methods = {
        "Deep Ensemble":  f"{RUNS}/uq_ensemble_resnet18_f*/uncertainty_summary.json",
        "MC Dropout (p=0.2)": f"{RUNS}/uq_mc_resnet18_d2e-01_seed*_f*_*/uncertainty_summary.json",
        "TTA": f"{RUNS}/uq_tta_resnet18_f*/uncertainty_summary.json",
    }
    fig, ax = plt.subplots(figsize=(7, 5))

    for label, pat in methods.items():
        ds = _load_jsons(pat)
        if not ds:
            print(f"  (no data for {label})")
            continue
        # Build a common coverage grid 0.10-1.00 and interpolate each fold
        cov_grid = np.linspace(0.10, 1.00, 91)
        acc_mat = []
        for d in ds:
            c = np.array(d["selective_prediction"]["coverage"])
            a = np.array(d["selective_prediction"]["accuracy"])
            ord_ = np.argsort(c)
            c, a = c[ord_], a[ord_]
            acc_mat.append(np.interp(cov_grid, c, a))
        acc_mat = np.stack(acc_mat, axis=0)
        m = acc_mat.mean(axis=0)
        s = acc_mat.std(axis=0)
        col = {"Deep Ensemble": COLORS["deep_ensemble"],
               "MC Dropout (p=0.2)": COLORS["mc_dropout"],
               "TTA": COLORS["tta"]}[label]
        ax.plot(cov_grid, m, color=col, linewidth=2.0,
                label=f"{label} (n={len(ds)} folds)")
        ax.fill_between(cov_grid, m - s, m + s, color=col, alpha=0.15)

    ax.set_xlabel("Coverage (fraction retained)")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0.10, 1.02)
    ax.set_ylim(0.5, 1.0)
    ax.invert_xaxis()  # high coverage on the left, rejecting more rightward
    ax.set_title("Selective prediction (ResNet-18): three UQ methods")
    ax.legend(loc="lower left")
    plt.tight_layout()
    _save(fig, "fig8_uq_three_way_selective_prediction")


# ---------------------------------------------------------------------------
# Figure 8.2: UE-AUROC bar chart per arch x method
# ---------------------------------------------------------------------------

def fig8_ue_auroc_bar():
    """Bar chart of UE-AUROC (std) for each arch under each UQ method."""
    archs = ["resnet18", "resnet50", "densenet121"]
    methods = ["Deep Ensemble", "TTA"]

    def collect(pat_fn):
        out = {}
        for arch in archs:
            ds = _load_jsons(pat_fn(arch))
            ues = [d["uncertainty_error_auroc"]["std"] for d in ds
                   if d.get("uncertainty_error_auroc", {}).get("std") is not None]
            out[arch] = (np.mean(ues), np.std(ues), len(ues)) if ues else (np.nan, 0, 0)
        return out

    de = collect(lambda a: f"{RUNS}/uq_ensemble_{a}_f*/uncertainty_summary.json")
    tta = collect(lambda a: f"{RUNS}/uq_tta_{a}_f*/uncertainty_summary.json")

    # MC Dropout: only run for r18/r50, p=0.2 representative
    mc = {}
    for arch in ["resnet18", "resnet50"]:
        ds = _load_jsons(f"{RUNS}/uq_mc_{arch}_d2e-01_seed*_f*_*/uncertainty_summary.json")
        ues = [d["uncertainty_error_auroc"]["std"] for d in ds
               if d.get("uncertainty_error_auroc", {}).get("std") is not None]
        mc[arch] = (np.mean(ues), np.std(ues), len(ues)) if ues else (np.nan, 0, 0)
    mc["densenet121"] = (np.nan, 0, 0)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = np.arange(len(archs))
    w = 0.27
    for i, (lbl, src, col) in enumerate([
        ("Deep Ensemble", de, COLORS["deep_ensemble"]),
        ("MC Dropout (p=0.2)", mc, COLORS["mc_dropout"]),
        ("TTA", tta, COLORS["tta"]),
    ]):
        means = [src[a][0] for a in archs]
        stds = [src[a][1] for a in archs]
        ax.bar(x + (i - 1) * w, means, w, yerr=stds, color=col,
               alpha=0.85, capsize=3, label=lbl,
               edgecolor="black", linewidth=0.5)

    ax.axhline(0.5, color="grey", linestyle="--", alpha=0.7,
               linewidth=0.8, label="Chance (uninformative)")
    ax.set_xticks(x)
    ax.set_xticklabels([ARCH_LABEL[a] for a in archs])
    ax.set_ylabel("UE-AUROC (uncertainty vs error)")
    ax.set_ylim(0.45, 0.80)
    ax.set_title("Uncertainty quality across architectures and UQ methods")
    ax.legend(loc="upper right", ncol=2)
    plt.tight_layout()
    _save(fig, "fig8_ue_auroc_per_method_arch")


# ---------------------------------------------------------------------------
# Figure 8.3: Heterogeneous vs homogeneous ensembles
# ---------------------------------------------------------------------------

def fig8_heterogeneous_vs_homogeneous():
    """Bar chart contrasting homogeneous (3-seed) vs heterogeneous (9-model)
    ensembles on AUROC and UE-AUROC."""
    # These numbers come from analyze_heterogeneous_ensemble.py (run locally)
    # Hard-coded here for plotting; can be regenerated.
    data = {
        "Homogeneous DenseNet (3 seed)":     (0.819, 0.049, 0.522, 0.083),
        "Homogeneous ResNet-18 (3 seed)":    (0.801, 0.061, 0.645, 0.080),
        "Homogeneous ResNet-50 (3 seed)":    (0.739, 0.075, 0.617, 0.064),
        "Heterogeneous 3-arch (1 seed)":     (0.797, 0.061, 0.652, 0.071),
        "Heterogeneous 9-model (3 arch x 3 seed)": (0.802, 0.060, 0.685, 0.057),
    }
    labels = list(data.keys())
    auroc_m = [data[k][0] for k in labels]
    auroc_s = [data[k][1] for k in labels]
    ue_m    = [data[k][2] for k in labels]
    ue_s    = [data[k][3] for k in labels]
    colors_ = ["#888"] * 3 + [COLORS["heterogeneous"]] * 2

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(labels))
    # AUROC
    axes[0].barh(x, auroc_m, xerr=auroc_s, color=colors_, alpha=0.85, capsize=3,
                 edgecolor="black", linewidth=0.5)
    axes[0].set_yticks(x)
    axes[0].set_yticklabels(labels)
    axes[0].set_xlabel("AUROC (mean ± std, 5 folds)")
    axes[0].set_xlim(0.65, 0.90)
    axes[0].set_title("Classification accuracy")
    for i, (m, s) in enumerate(zip(auroc_m, auroc_s)):
        axes[0].text(m + s + 0.005, i, f"{m:.3f}", va="center", fontsize=9)
    # UE-AUROC
    axes[1].barh(x, ue_m, xerr=ue_s, color=colors_, alpha=0.85, capsize=3,
                 edgecolor="black", linewidth=0.5)
    axes[1].set_yticks(x)
    axes[1].set_yticklabels([])  # share with left
    axes[1].axvline(0.5, color="grey", linestyle="--", alpha=0.6, linewidth=0.8)
    axes[1].set_xlabel("UE-AUROC (uncertainty vs error)")
    axes[1].set_xlim(0.45, 0.80)
    axes[1].set_title("Uncertainty quality")
    for i, (m, s) in enumerate(zip(ue_m, ue_s)):
        axes[1].text(m + s + 0.005, i, f"{m:.3f}", va="center", fontsize=9)

    fig.suptitle("Heterogeneous architecture ensembles improve uncertainty quality"
                 " without sacrificing accuracy", fontsize=12)
    plt.tight_layout()
    _save(fig, "fig8_heterogeneous_vs_homogeneous")


# ---------------------------------------------------------------------------
# Figure 8.4: DenseNet UQ pathology (inter-seed correlation)
# ---------------------------------------------------------------------------

def fig8_pathology():
    """Bar chart: inter-seed prediction correlation per architecture (the
    quantitative evidence for DenseNet's ensemble homogeneity)."""
    # Numbers from analyze_densenet_pathology.py (H1, mean over 5 folds)
    archs = ["densenet121", "resnet18", "resnet50"]
    means = [0.7447, 0.6827, 0.6392]
    stds  = [0.0292, 0.0285, 0.0869]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    cols = [COLORS[a] for a in archs]
    ax.bar([ARCH_LABEL[a] for a in archs], means, yerr=stds, color=cols,
           alpha=0.85, capsize=4, edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Mean pairwise prediction correlation\nacross seed-variants (5 folds)")
    ax.set_ylim(0.5, 0.85)
    ax.axhline(0.85, color="red", linestyle=":", alpha=0.5,
               label="Threshold (homogeneous)")
    ax.set_title("Ensemble homogeneity: DenseNet seeds are more correlated\n"
                 "→ less informative ensemble uncertainty")
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.005, f"r={m:.2f}", ha="center", fontsize=10)
    ax.legend(loc="lower left")
    plt.tight_layout()
    _save(fig, "fig8_pathology_inter_seed_correlation")


# ---------------------------------------------------------------------------
# Figure 8.5: Calibration / ECE
# ---------------------------------------------------------------------------

def fig8_calibration():
    """Bar chart: ECE before and after temperature scaling per architecture."""
    archs = ["densenet121", "resnet18", "resnet50"]
    ece_raw = [0.0826, 0.0651, 0.0634]
    ece_ts  = [0.0572, 0.0430, 0.0545]
    T = [1.065, 1.266, 1.059]
    x = np.arange(len(archs))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w / 2, ece_raw, w, color="#888", alpha=0.85, label="Raw",
           edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, ece_ts, w, color=COLORS["deep_ensemble"], alpha=0.85,
           label="After temperature scaling",
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([ARCH_LABEL[a] for a in archs])
    ax.set_ylabel("Expected Calibration Error (ECE)")
    ax.set_title("Calibration: temperature scaling reduces ECE for all archs")
    for i, t in enumerate(T):
        ax.text(i, max(ece_raw[i], ece_ts[i]) + 0.005, f"T={t:.2f}",
                ha="center", fontsize=9)
    ax.legend()
    plt.tight_layout()
    _save(fig, "fig8_calibration_ece")


# ---------------------------------------------------------------------------
# Figure 9.1: Synthetic distortion response
# ---------------------------------------------------------------------------

def fig9_synthetic_distortion():
    """Mean P(distorted) vs injected distortion severity per architecture."""
    archs = ["densenet121", "resnet18", "resnet50"]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for arch in archs:
        ds = _load_jsons(f"{RUNS}/syndist_{arch}_f*/distortion_summary.json")
        if not ds:
            print(f"  (no syndist data for {arch})")
            continue
        sev = ds[0]["severities"]
        per_sev = np.array([[d["response_curve"][i]["mean_p_distorted"]
                             for i in range(len(sev))] for d in ds])
        m = per_sev.mean(axis=0)
        s = per_sev.std(axis=0)
        col = COLORS[arch]
        ax.errorbar(sev, m, yerr=s, marker="o", color=col,
                    label=ARCH_LABEL[arch], linewidth=1.5, capsize=3,
                    markersize=5)
    ax.axhline(0.5, color="grey", linestyle="--", alpha=0.6,
               label="Decision threshold")
    ax.set_xlabel("Injected vertical distortion magnitude (max pixels)")
    ax.set_ylabel("Mean P(distorted) on undistorted test cases")
    ax.set_title("Synthetic distortion response: monotonic detection\n"
                 "(physics validation that models respond to geometric distortion)")
    ax.set_ylim(0, 0.7)
    ax.legend(loc="upper left")
    plt.tight_layout()
    _save(fig, "fig9_synthetic_distortion_response")


# ===========================================================================
# CHAPTER 4: additional architecture / ablation figures
# ===========================================================================

def fig4_confusion_matrices():
    """Confusion matrices for each architecture on the test set (5-fold pooled)."""
    archs = ["densenet121", "resnet18", "resnet50", "efficientnet_b0"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, arch in zip(axes, archs):
        # Pool test predictions across the 5 folds
        files = sorted(glob.glob(f"{RUNS}/poster100_{arch}_dwi_roi_f*/test_predictions.csv"))
        if not files:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            ax.set_title(ARCH_LABEL[arch]); continue
        ys, ps = [], []
        for f in files:
            df = pd.read_csv(f)
            ys.append(df["y_true"].values); ps.append(df["y_prob"].values)
        y = np.concatenate(ys); p = np.concatenate(ps)
        yhat = (p >= 0.5).astype(int)
        cm = np.array([[((y == 0) & (yhat == 0)).sum(), ((y == 0) & (yhat == 1)).sum()],
                       [((y == 1) & (yhat == 0)).sum(), ((y == 1) & (yhat == 1)).sum()]])
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.2f})",
                        ha="center", va="center",
                        color="white" if cm_norm[i, j] > 0.5 else "black",
                        fontsize=9)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Undist.", "Dist."])
        ax.set_yticklabels(["Undist.", "Dist."])
        if ax is axes[0]:
            ax.set_ylabel("True label")
        ax.set_xlabel("Predicted")
        ax.set_title(ARCH_LABEL[arch], fontsize=10)
    fig.suptitle("Confusion matrices (test set, 5-fold pooled)")
    plt.tight_layout()
    _save(fig, "fig4_confusion_matrices")


def fig4_roc_curves():
    """ROC curves overlay for each architecture (5-fold pooled)."""
    from sklearn.metrics import roc_curve, auc
    archs = ["densenet121", "resnet18", "resnet50", "efficientnet_b0"]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for arch in archs:
        files = sorted(glob.glob(f"{RUNS}/poster100_{arch}_dwi_roi_f*/test_predictions.csv"))
        if not files:
            continue
        ys, ps = [], []
        for f in files:
            df = pd.read_csv(f)
            ys.append(df["y_true"].values); ps.append(df["y_prob"].values)
        y = np.concatenate(ys); p = np.concatenate(ps)
        fpr, tpr, _ = roc_curve(y, p)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLORS[arch],
                label=f"{ARCH_LABEL[arch]} (AUC={roc_auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.7, alpha=0.5, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves: per-architecture (5-fold pooled)")
    ax.set_xlim(-0.02, 1.0); ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    plt.tight_layout()
    _save(fig, "fig4_roc_curves")


def fig4_modality_comparison():
    """DWI vs T2 vs Fusion comparison (ResNet-18 baselines)."""
    poster = json.load(open(_find_poster_json()))
    keys = [("resnet18_t2_roi", "T2 only"),
            ("resnet18_fusion_roi", "DWI+T2 fusion"),
            ("resnet18_dwi_roi", "DWI only")]
    means = [poster[k]["auroc"]["mean"] for k, _ in keys if k in poster]
    stds  = [poster[k]["auroc"]["std"] for k, _ in keys if k in poster]
    labels = [l for k, l in keys if k in poster]
    fig, ax = plt.subplots(figsize=(6, 4))
    cols = ["#999", "#4c72b0", COLORS["resnet18"]]
    ax.bar(labels, means, yerr=stds, color=cols, alpha=0.85, capsize=4,
           edgecolor="black", linewidth=0.5)
    ax.axhline(0.5, color="grey", linestyle=":", alpha=0.7, label="Chance")
    ax.set_ylabel("Test AUROC")
    ax.set_ylim(0.5, 0.85)
    ax.set_title("Modality ablation (ResNet-18): DWI carries the signal")
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.005, f"{m:.2f}", ha="center", fontsize=10)
    plt.tight_layout()
    _save(fig, "fig4_modality_comparison")


def fig4_roi_vs_global():
    """ROI crop vs Global input."""
    poster = json.load(open(_find_poster_json()))
    keys = [("resnet18_dwi_global", "Global input"),
            ("resnet18_dwi_roi", "Mask-cropped ROI")]
    means = [poster[k]["auroc"]["mean"] for k, _ in keys if k in poster]
    stds = [poster[k]["auroc"]["std"] for k, _ in keys if k in poster]
    labels = [l for k, l in keys if k in poster]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, means, yerr=stds,
           color=["#999", COLORS["resnet18"]], alpha=0.85, capsize=4,
           edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Test AUROC")
    ax.set_ylim(0.5, 0.85)
    ax.set_title("ROI vs Global input (ResNet-18): distortion extends beyond gland")
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.005, f"{m:.3f}", ha="center", fontsize=10)
    plt.tight_layout()
    _save(fig, "fig4_roi_vs_global")


def fig4_pretrained_vs_scratch():
    """ImageNet pretraining vs random init across 2 archs."""
    poster = json.load(open(_find_poster_json()))
    archs = ["resnet18", "resnet50"]
    paired = {}
    for a in archs:
        pre_k = f"{a}_dwi_roi"
        sc_k = f"{a}_dwi_roi_scratch"
        if pre_k in poster and sc_k in poster:
            paired[a] = (poster[sc_k]["auroc"]["mean"], poster[sc_k]["auroc"]["std"],
                         poster[pre_k]["auroc"]["mean"], poster[pre_k]["auroc"]["std"])
    if not paired:
        return
    x = np.arange(len(paired))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sc_m = [v[0] for v in paired.values()]
    sc_s = [v[1] for v in paired.values()]
    pre_m = [v[2] for v in paired.values()]
    pre_s = [v[3] for v in paired.values()]
    ax.bar(x - w / 2, sc_m, w, yerr=sc_s, color="#999", alpha=0.85,
           label="From scratch", capsize=4, edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, pre_m, w, yerr=pre_s, color=COLORS["deep_ensemble"],
           alpha=0.85, label="ImageNet pretrained", capsize=4,
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([ARCH_LABEL[a] for a in paired.keys()])
    ax.set_ylabel("Test AUROC")
    ax.set_ylim(0.5, 0.85)
    ax.set_title("ImageNet pretraining transfers to prostate DWI QA")
    ax.legend()
    plt.tight_layout()
    _save(fig, "fig4_pretrained_vs_scratch")


def fig4_per_fold_heatmap():
    """Heatmap: architecture x fold of test AUROC, showing consistency."""
    archs = ["densenet121", "resnet18", "resnet50", "efficientnet_b0"]
    matrix = np.full((len(archs), 5), np.nan)
    for i, arch in enumerate(archs):
        for f in range(5):
            files = glob.glob(f"{RUNS}/poster100_{arch}_dwi_roi_f{f}*/summary.json")
            files += glob.glob(f"{RUNS}/poster_{arch}_dwi_roi_f{f}*/summary.json")
            if files:
                try:
                    d = json.load(open(files[0]))
                    matrix[i, f] = d.get("test", {}).get("auroc", np.nan)
                except Exception:
                    pass
    fig, ax = plt.subplots(figsize=(6.5, 4))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto", vmin=0.5, vmax=0.95)
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"Fold {f}" for f in range(5)])
    ax.set_yticks(range(len(archs)))
    ax.set_yticklabels([ARCH_LABEL[a] for a in archs])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        color="white" if matrix[i, j] < 0.75 else "black",
                        fontsize=9)
    ax.set_title("Per-fold test AUROC: cross-fold consistency")
    plt.colorbar(im, ax=ax, label="Test AUROC")
    plt.tight_layout()
    _save(fig, "fig4_per_fold_heatmap")


def fig4_effect_sizes():
    """Cohen's d for key comparisons."""
    stats_json = (Path(_find_poster_json()).parent / "stats_results.json")
    if not stats_json.exists():
        return
    stats = json.load(open(stats_json))
    items = list(stats.items())
    labels = [k for k, _ in items]
    d_vals = [v["cohen_d"] for _, v in items]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    cols = ["#d62728" if d > 0.8 else "#888" for d in d_vals]
    ax.barh(range(len(labels)), d_vals, color=cols, alpha=0.85,
            edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.axvline(0.2, color="grey", linestyle=":", alpha=0.5, label="Small (0.2)")
    ax.axvline(0.5, color="grey", linestyle="--", alpha=0.5, label="Medium (0.5)")
    ax.axvline(0.8, color="red", linestyle="-", alpha=0.5, label="Large (0.8)")
    ax.set_xlabel("Cohen's d (effect size)")
    ax.set_title("Effect sizes across key comparisons")
    ax.legend(loc="lower right")
    for i, d in enumerate(d_vals):
        ax.text(d + 0.05, i, f"d={d:.2f}", va="center", fontsize=9)
    plt.tight_layout()
    _save(fig, "fig4_effect_sizes")


# ===========================================================================
# CHAPTER 7: additional SSL figures
# ===========================================================================

def fig7_per_fold_boxplot():
    """Distribution of test AUROC per SSL config across (5 seeds x 5 folds)=25 evals."""
    rows = []
    for arch in ["resnet18", "resnet50"]:
        for method in ["simclr", "byol"]:
            for init in ["rnd", "imn"]:
                files = sorted(glob.glob(
                    f"{RUNS}/ft_{method}_{arch}_{init}_seed*_f*_*/summary.json"))
                for f in files:
                    try:
                        d = json.load(open(f))
                        if "error" in d:
                            continue
                        a = d.get("test", {}).get("auroc")
                        if a is not None:
                            rows.append({"arch": arch, "method": method,
                                         "init": init, "auroc": a})
                    except Exception:
                        pass
    df = pd.DataFrame(rows)
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, arch in zip(axes, ["resnet18", "resnet50"]):
        sub = df[df["arch"] == arch]
        groups = [(m, i) for m in ["simclr", "byol"] for i in ["rnd", "imn"]]
        data = [sub[(sub["method"] == m) & (sub["init"] == i)]["auroc"].values
                for m, i in groups]
        labs = [f"{m}\n{i}" for m, i in groups]
        bp = ax.boxplot(data, labels=labs, widths=0.6, patch_artist=True)
        for patch, (m, i) in zip(bp["boxes"], groups):
            patch.set_facecolor("#bbb" if i == "rnd" else COLORS[arch])
            patch.set_alpha(0.85)
        ax.set_title(ARCH_LABEL[arch])
        ax.set_ylabel("Test AUROC (25 evals each)")
        ax.set_ylim(0.6, 0.95)
    fig.suptitle("SSL fine-tune AUROC: distribution across seeds × folds")
    plt.tight_layout()
    _save(fig, "fig7_per_fold_boxplot")


def fig7_wilcoxon_significance():
    """Paired Wilcoxon p-values heatmap: ImageNet vs Random init."""
    from scipy.stats import wilcoxon
    archs = ["resnet18", "resnet50"]
    methods = ["simclr", "byol"]
    pvals = np.full((len(archs), len(methods)), np.nan)
    deltas = np.full((len(archs), len(methods)), np.nan)
    for i, arch in enumerate(archs):
        for j, m in enumerate(methods):
            rnd, imn = [], []
            for s in [1, 2, 3, 4, 5]:
                for f in [0, 1, 2, 3, 4]:
                    rg = glob.glob(f"{RUNS}/ft_{m}_{arch}_rnd_seed{s}_f{f}_*/summary.json")
                    ig = glob.glob(f"{RUNS}/ft_{m}_{arch}_imn_seed{s}_f{f}_*/summary.json")
                    if rg and ig:
                        try:
                            r = json.load(open(rg[0])); ii = json.load(open(ig[0]))
                            if "error" not in r and "error" not in ii:
                                ra = r.get("test", {}).get("auroc")
                                ia = ii.get("test", {}).get("auroc")
                                if ra is not None and ia is not None:
                                    rnd.append(ra); imn.append(ia)
                        except Exception:
                            pass
            if len(rnd) >= 5:
                try:
                    _, p = wilcoxon(imn, rnd)
                    pvals[i, j] = p
                    deltas[i, j] = np.mean(np.array(imn) - np.array(rnd))
                except Exception:
                    pass

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    # Heatmap of delta
    im0 = axes[0].imshow(deltas, cmap="RdBu_r", vmin=-0.05, vmax=0.05,
                          aspect="auto")
    axes[0].set_xticks(range(len(methods)))
    axes[0].set_xticklabels([m.upper() for m in methods])
    axes[0].set_yticks(range(len(archs)))
    axes[0].set_yticklabels([ARCH_LABEL[a] for a in archs])
    axes[0].set_title("Δ AUROC (ImageNet − Random)")
    for i in range(deltas.shape[0]):
        for j in range(deltas.shape[1]):
            if not np.isnan(deltas[i, j]):
                axes[0].text(j, i, f"{deltas[i,j]:+.3f}",
                             ha="center", va="center", fontsize=10)
    plt.colorbar(im0, ax=axes[0])

    # Heatmap of -log10(p)
    nl = -np.log10(np.clip(pvals, 1e-6, 1.0))
    im1 = axes[1].imshow(nl, cmap="Greens", aspect="auto", vmin=0, vmax=4)
    axes[1].set_xticks(range(len(methods)))
    axes[1].set_xticklabels([m.upper() for m in methods])
    axes[1].set_yticks(range(len(archs)))
    axes[1].set_yticklabels([ARCH_LABEL[a] for a in archs])
    axes[1].set_title("−log₁₀(p)  (paired Wilcoxon, n=25)")
    for i in range(pvals.shape[0]):
        for j in range(pvals.shape[1]):
            p = pvals[i, j]
            if not np.isnan(p):
                star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                axes[1].text(j, i, f"p={p:.3g}\n{star}",
                             ha="center", va="center", fontsize=9,
                             color="white" if nl[i, j] > 2 else "black")
    plt.colorbar(im1, ax=axes[1])
    fig.suptitle("Statistical significance: ImageNet pretraining helps SSL")
    plt.tight_layout()
    _save(fig, "fig7_wilcoxon_significance")


# ===========================================================================
# CHAPTER 8: additional UQ figures
# ===========================================================================

def fig8_mc_dropout_sweep():
    """AUROC and UE-AUROC vs dropout rate (ResNet-18 / ResNet-50)."""
    rates = [0.1, 0.2, 0.3, 0.5]
    rate_tags = ["d1e-01", "d2e-01", "d3e-01", "d5e-01"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for arch in ["resnet18", "resnet50"]:
        ams, asd, ums, usd = [], [], [], []
        for tag in rate_tags:
            files = sorted(glob.glob(
                f"{RUNS}/uq_mc_{arch}_{tag}_seed*_f*_*/uncertainty_summary.json"))
            aurocs = [json.load(open(f))["classification"]["auroc"] for f in files]
            ues = [json.load(open(f))["uncertainty_error_auroc"]["std"]
                   for f in files if json.load(open(f))["uncertainty_error_auroc"]["std"]]
            if aurocs:
                ams.append(np.mean(aurocs)); asd.append(np.std(aurocs))
                ums.append(np.mean(ues)); usd.append(np.std(ues))
            else:
                ams.append(np.nan); asd.append(0); ums.append(np.nan); usd.append(0)
        axes[0].errorbar(rates, ams, yerr=asd, marker="o", linewidth=2,
                         capsize=3, label=ARCH_LABEL[arch], color=COLORS[arch])
        axes[1].errorbar(rates, ums, yerr=usd, marker="s", linewidth=2,
                         capsize=3, label=ARCH_LABEL[arch], color=COLORS[arch])
    axes[0].set_xlabel("Dropout rate"); axes[0].set_ylabel("Test AUROC")
    axes[0].set_title("MC Dropout: accuracy vs dropout rate")
    axes[0].set_ylim(0.65, 0.85); axes[0].legend()
    axes[1].set_xlabel("Dropout rate"); axes[1].set_ylabel("UE-AUROC (std)")
    axes[1].axhline(0.5, color="grey", linestyle="--", alpha=0.5)
    axes[1].set_title("MC Dropout: uncertainty quality vs dropout rate")
    axes[1].set_ylim(0.5, 0.75); axes[1].legend()
    plt.tight_layout()
    _save(fig, "fig8_mc_dropout_sweep")


def fig8_probability_histograms():
    """Histogram of predicted probabilities per architecture (saturation check)."""
    archs = ["densenet121", "resnet18", "resnet50"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, arch in zip(axes, archs):
        all_p = []
        for f in glob.glob(f"{RUNS}/poster100_{arch}_dwi_roi_f*/test_predictions.csv"):
            try:
                df = pd.read_csv(f)
                all_p.append(df["y_prob"].values)
            except Exception:
                pass
        if not all_p:
            ax.text(0.5, 0.5, "no data"); ax.set_title(ARCH_LABEL[arch]); continue
        p = np.concatenate(all_p)
        ax.hist(p, bins=30, color=COLORS[arch], alpha=0.8, edgecolor="black",
                linewidth=0.4)
        ax.axvline(0.5, color="grey", linestyle="--", alpha=0.5)
        sat = (np.mean((p < 0.1) | (p > 0.9)))
        ax.set_title(f"{ARCH_LABEL[arch]}\n(saturated frac: {sat:.2f})")
        ax.set_xlabel("Predicted probability")
        if ax is axes[0]:
            ax.set_ylabel("Count")
    fig.suptitle("Predicted probability distributions (test set)")
    plt.tight_layout()
    _save(fig, "fig8_probability_histograms")


def fig8_reliability_diagrams():
    """Reliability diagrams (calibration) per architecture."""
    archs = ["densenet121", "resnet18", "resnet50"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    mids = (bins[:-1] + bins[1:]) / 2
    for ax, arch in zip(axes, archs):
        all_p, all_y = [], []
        for f in glob.glob(f"{RUNS}/poster100_{arch}_dwi_roi_f*/test_predictions.csv"):
            try:
                df = pd.read_csv(f)
                all_p.append(df["y_prob"].values); all_y.append(df["y_true"].values)
            except Exception:
                pass
        if not all_p:
            continue
        p = np.concatenate(all_p); y = np.concatenate(all_y)
        # Confidence = max(p, 1-p); accuracy = (yhat == y)
        conf = np.where(p >= 0.5, p, 1 - p)
        yhat = (p >= 0.5).astype(int)
        acc = (yhat == y).astype(float)
        bin_acc, bin_conf, bin_n = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() > 0:
                bin_acc.append(acc[m].mean()); bin_conf.append(conf[m].mean()); bin_n.append(m.sum())
            else:
                bin_acc.append(np.nan); bin_conf.append((lo + hi) / 2); bin_n.append(0)
        ece = 0.0; total = len(p)
        for lo, hi, a, c, n in zip(bins[:-1], bins[1:], bin_acc, bin_conf, bin_n):
            if n > 0:
                ece += (n / total) * abs(a - c)
        bin_acc = np.array(bin_acc); bin_n = np.array(bin_n)
        ax.bar(mids, bin_acc, width=1 / n_bins, color=COLORS[arch], alpha=0.8,
               edgecolor="black", linewidth=0.5)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.6)
        ax.set_xlim(0.4, 1.02); ax.set_ylim(0, 1.02)
        ax.set_xlabel("Mean confidence")
        if ax is axes[0]:
            ax.set_ylabel("Accuracy")
        ax.set_title(f"{ARCH_LABEL[arch]}  (ECE={ece:.3f})")
    fig.suptitle("Reliability diagrams (calibration): test set, 5-fold pooled")
    plt.tight_layout()
    _save(fig, "fig8_reliability_diagrams")


def fig8_per_bvalue_uncertainty():
    """Per-b-value uncertainty + accuracy for the Deep Ensemble (ResNet-18)."""
    files = sorted(glob.glob(f"{RUNS}/uq_ensemble_resnet18_f*/uncertainty_summary.json"))
    if not files:
        return
    agg = {}
    for f in files:
        d = json.load(open(f))
        for bv, vals in (d.get("per_b_value") or {}).items():
            agg.setdefault(bv, {"std": [], "acc": [], "n": []})
            agg[bv]["std"].append(vals.get("std_mean", np.nan))
            agg[bv]["acc"].append(vals.get("accuracy", np.nan))
            agg[bv]["n"].append(vals.get("n", 0))
    bvs = sorted(agg.keys())
    std_m = [np.mean(agg[bv]["std"]) for bv in bvs]
    std_s = [np.std(agg[bv]["std"]) for bv in bvs]
    acc_m = [np.mean(agg[bv]["acc"]) for bv in bvs]
    acc_s = [np.std(agg[bv]["acc"]) for bv in bvs]
    n_total = [int(np.sum(agg[bv]["n"])) for bv in bvs]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].errorbar(bvs, std_m, yerr=std_s, marker="o", linewidth=2,
                     capsize=3, color=COLORS["deep_ensemble"])
    axes[0].set_ylabel("Mean uncertainty (std)")
    axes[0].set_xlabel("b-value group")
    axes[0].set_title("Per-b-value: uncertainty")
    axes[1].errorbar(bvs, acc_m, yerr=acc_s, marker="s", linewidth=2,
                     capsize=3, color=COLORS["resnet18"])
    axes[1].set_ylabel("Accuracy")
    axes[1].set_xlabel("b-value group")
    axes[1].set_title("Per-b-value: accuracy")
    for i, (bv, n) in enumerate(zip(bvs, n_total)):
        axes[0].text(i, std_m[i] + 0.005, f"n={n}", ha="center", fontsize=8)
    fig.suptitle("ResNet-18 Deep Ensemble: per-b-value breakdown")
    plt.tight_layout()
    _save(fig, "fig8_per_bvalue_uncertainty")


# ===========================================================================
# CHAPTER 9: additional cross-cutting figures
# ===========================================================================

def fig9_prior_work_comparison():
    """Comparison with prior published prostate MRI QA work."""
    works = [
        ("Alis et al. 2023\n(bi-parametric prostate MRI QA)", 0.78, "#888"),
        ("This work — best single arch\n(DenseNet-121)",   0.819, COLORS["deep_ensemble"]),
        ("This work — heterogeneous ensemble", 0.802, COLORS["heterogeneous"]),
        ("This work — SSL best\n(ResNet-50 SimCLR+ImageNet)", 0.808, COLORS["resnet50"]),
    ]
    labels = [w[0] for w in works]
    vals = [w[1] for w in works]
    cols = [w[2] for w in works]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(range(len(labels)), vals, color=cols, alpha=0.9,
            edgecolor="black", linewidth=0.6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Test AUROC")
    ax.set_xlim(0.7, 0.86)
    for i, v in enumerate(vals):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=10)
    ax.set_title("Comparison with prior published work on prostate MRI QA")
    plt.tight_layout()
    _save(fig, "fig9_prior_work_comparison")


def fig9_lobo_generalisation():
    """Leave-one-b-value-out: in-distribution vs held-out AUROC per arch."""
    # Pull from lobo runs
    archs = ["resnet18", "resnet50"]
    bvs = ["1k4", "1k6", "2k"]
    bv_labels = {"1k4": "b=1400", "1k6": "b=1600", "2k": "b=2000"}
    # In-distribution baselines (from poster baselines, same archs)
    in_dist = {"resnet18": 0.766, "resnet50": 0.729}

    data = {}
    for arch in archs:
        per_bv = {}
        for bv in bvs:
            files = glob.glob(f"{RUNS}/lobo_{arch}_excl{bv}_*/summary.json")
            aurocs = []
            for f in files:
                try:
                    d = json.load(open(f))
                    if "error" in d:
                        continue
                    a = d.get("test", {}).get("auroc")
                    if a is not None:
                        aurocs.append(a)
                except Exception:
                    pass
            if aurocs:
                per_bv[bv] = (np.mean(aurocs), np.std(aurocs), len(aurocs))
        data[arch] = per_bv

    if not any(data.values()):
        print("  (no LOBO data found; skipping)")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(bvs))
    w = 0.36
    for i, arch in enumerate(archs):
        means = [data[arch].get(bv, (np.nan, 0, 0))[0] for bv in bvs]
        stds = [data[arch].get(bv, (np.nan, 0, 0))[1] for bv in bvs]
        ax.bar(x + (i - 0.5) * w, means, w, yerr=stds, color=COLORS[arch],
               alpha=0.88, capsize=4, label=ARCH_LABEL[arch],
               edgecolor="black", linewidth=0.5)
        # Horizontal reference line for in-distribution baseline
        ax.axhline(in_dist[arch], color=COLORS[arch], linestyle="--", alpha=0.55,
                   linewidth=1.2,
                   label=f"{ARCH_LABEL[arch]} in-distribution baseline ({in_dist[arch]:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels([bv_labels[bv] for bv in bvs])
    ax.set_xlabel("Held-out b-value group (excluded from training)")
    ax.set_ylabel("Test AUROC on held-out b-value")
    ax.set_ylim(0.5, 0.85)
    ax.set_title("Leave-one-b-value-out generalisation\n"
                 "ResNet-18 generalises better across acquisitions than ResNet-50")
    ax.legend(loc="lower right", fontsize=8)
    for i, arch in enumerate(archs):
        for j, bv in enumerate(bvs):
            if bv in data[arch]:
                m, s, n = data[arch][bv]
                ax.text(j + (i - 0.5) * w, m + s + 0.005, f"{m:.2f}",
                        ha="center", fontsize=8)
    plt.tight_layout()
    _save(fig, "fig9_lobo_generalisation")


def fig9_learning_curves():
    """Learning curves per architecture across 25/50/75/100% data fractions."""
    archs = ["resnet18", "resnet50", "densenet121"]
    fracs = [("025", 0.25), ("050", 0.50), ("075", 0.75), ("100", 1.00)]

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted_any = False
    for arch in archs:
        xs, ms, ss = [], [], []
        for frac_str, frac_val in fracs:
            files = glob.glob(f"{RUNS}/lc_{arch}_frac{frac_str}_seed*_f*/summary.json")
            aurocs = []
            for f in files:
                try:
                    d = json.load(open(f))
                    if "error" in d:
                        continue
                    a = d.get("test", {}).get("auroc")
                    if a is not None:
                        aurocs.append(a)
                except Exception:
                    pass
            if aurocs:
                xs.append(frac_val * 100)
                ms.append(np.mean(aurocs))
                ss.append(np.std(aurocs))
        if xs:
            plotted_any = True
            ax.errorbar(xs, ms, yerr=ss, marker="o", linewidth=2,
                        capsize=4, color=COLORS[arch], label=ARCH_LABEL[arch],
                        markersize=7)

    if not plotted_any:
        print("  (no learning-curve data found; skipping)")
        plt.close(fig)
        return

    ax.set_xlabel("Training data fraction (%)")
    ax.set_ylabel("Test AUROC (5-fold mean ± std)")
    ax.set_xlim(15, 105)
    ax.set_ylim(0.6, 0.85)
    ax.set_title("Learning curves: would more data help?\n"
                 "ResNet-18 plateaus; ResNet-50/DenseNet still climbing at 100%")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "fig9_learning_curves")


def fig9_statistical_summary():
    """Master heatmap of all paired comparisons with effect sizes."""
    # Pull from stats_results.json if present; else recompute the SSL ones
    stats_json_path = Path(_find_poster_json()).parent / "stats_results.json"
    if not stats_json_path.exists():
        return
    stats = json.load(open(stats_json_path))
    rows = [(k, v["delta"], v["cohen_d"]) for k, v in stats.items()]
    rows.sort(key=lambda r: abs(r[2]), reverse=True)
    labels = [r[0] for r in rows]
    deltas = [r[1] for r in rows]
    cohens = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors_ = ["#d62728" if c > 0.8 else "#ff7f0e" if c > 0.5 else "#bbb" for c in cohens]
    bars = ax.barh(range(len(labels)), cohens, color=colors_, alpha=0.85,
                   edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for i, (d, c) in enumerate(zip(deltas, cohens)):
        ax.text(c + 0.05, i, f"d={c:.2f}\nΔ={d:+.3f}", va="center", fontsize=8)
    ax.axvline(0.2, color="grey", linestyle=":", alpha=0.5)
    ax.axvline(0.5, color="grey", linestyle="--", alpha=0.5)
    ax.axvline(0.8, color="red", linestyle="-", alpha=0.5, label="Large (d=0.8)")
    ax.set_xlabel("Cohen's d (effect size, sorted)")
    ax.set_title("Master summary: effect sizes across all key comparisons")
    ax.legend(loc="lower right")
    plt.tight_layout()
    _save(fig, "fig9_statistical_summary")


# ---------------------------------------------------------------------------
# Utility to locate poster_results.json for ablation figures
# ---------------------------------------------------------------------------

def _find_poster_json():
    candidates = [
        Path(os.environ.get("MIQA_POSTER_JSON", "")) if os.environ.get("MIQA_POSTER_JSON") else None,
        Path(RUNS).parent / "poster_results.json",
        Path(RUNS).parent.parent / "poster_results.json",
        Path.cwd() / "poster_results.json",
    ]
    p = next((p for p in candidates if p and p.exists()), None)
    if p is None:
        raise FileNotFoundError("poster_results.json not found")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Reading from RUNS = {RUNS}")
    print(f"Saving to    OUT  = {OUT}\n")

    figs = [
        ("Fig 4.1  Architecture comparison",        fig4_architecture_comparison),
        ("Fig 4.2  Confusion matrices grid",        fig4_confusion_matrices),
        ("Fig 4.3  ROC curves",                     fig4_roc_curves),
        ("Fig 4.4  Modality ablation",              fig4_modality_comparison),
        ("Fig 4.5  ROI vs Global",                  fig4_roi_vs_global),
        ("Fig 4.6  Pretrained vs Scratch",          fig4_pretrained_vs_scratch),
        ("Fig 4.7  Per-fold AUROC heatmap",         fig4_per_fold_heatmap),
        ("Fig 4.8  Effect sizes",                   fig4_effect_sizes),
        ("Fig 7.1  SSL 4-way comparison",           fig7_ssl_four_way),
        ("Fig 7.2  SSL per-fold boxplot",           fig7_per_fold_boxplot),
        ("Fig 7.3  SSL Wilcoxon significance",      fig7_wilcoxon_significance),
        ("Fig 8.1  UQ 3-way selective prediction",  fig8_uq_three_way),
        ("Fig 8.2  UE-AUROC per method x arch",     fig8_ue_auroc_bar),
        ("Fig 8.3  Heterogeneous vs homogeneous",   fig8_heterogeneous_vs_homogeneous),
        ("Fig 8.4  Pathology — inter-seed correlation", fig8_pathology),
        ("Fig 8.5  Calibration / ECE",              fig8_calibration),
        ("Fig 8.6  MC Dropout rate sweep",          fig8_mc_dropout_sweep),
        ("Fig 8.7  Probability histograms",         fig8_probability_histograms),
        ("Fig 8.8  Reliability diagrams",           fig8_reliability_diagrams),
        ("Fig 8.9  Per-b-value uncertainty",        fig8_per_bvalue_uncertainty),
        ("Fig 9.1  Synthetic distortion response",  fig9_synthetic_distortion),
        ("Fig 9.2  LOBO generalisation",            fig9_lobo_generalisation),
        ("Fig 9.3  Learning curves",                fig9_learning_curves),
        ("Fig 9.4  Prior work comparison",          fig9_prior_work_comparison),
        ("Fig 9.5  Statistical summary (effect sizes)", fig9_statistical_summary),
    ]
    for label, fn in figs:
        print(f"[{label}]")
        try:
            fn()
        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")

    print(f"\nDone. {len(figs)} figures attempted. Output: {OUT.resolve()}")


if __name__ == "__main__":
    main()
