"""Reliability diagram plotting."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_reliability_diagram(y_true, y_prob, out_path, n_bins=10):
    """Plot reliability diagram (top) + prediction histogram (bottom)."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = 1.0 / n_bins

    accs = []
    confs = []
    counts = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        if lo == bin_edges[-2]:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        count = int(mask.sum())
        counts.append(count)
        if count == 0:
            accs.append(0.0)
            confs.append(0.0)
        else:
            accs.append(float(y_true[mask].mean()))
            confs.append(float(y_prob[mask].mean()))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7),
                                    gridspec_kw={"height_ratios": [3, 1]})

    # Reliability diagram
    ax1.bar(bin_centers, accs, width=bin_width, alpha=0.6,
            edgecolor="black", label="Accuracy")
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Fraction of positives")
    ax1.set_title("Reliability Diagram")
    ax1.legend(loc="upper left")

    # Histogram of predictions
    ax2.bar(bin_centers, counts, width=bin_width, alpha=0.6,
            edgecolor="black", color="tab:orange")
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Mean predicted probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Prediction histogram")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved reliability diagram to {out_path}")
