"""Generate thesis schematic / dataset-overview / pipeline figures.

Most can be auto-generated from the real NIfTI volumes plus index_full.csv
rather than hand-drawn in PowerPoint/Inkscape.

Produces:
  fig3_dataset_overview.pdf       : sample distorted vs undistorted slices per b-value
  fig3_label_distribution.pdf     : bar chart of label balance per b-value
  fig3_kfold_schematic.pdf        : patient-level 5-fold CV visualization
  fig3_2_5d_stacking.pdf          : k=3 axial slices forming a 2.5D input
  fig3_preprocessing_pipeline.pdf : schematic with real sample slices at each step

Reads from MIQA_DATA_ROOT (defaults to MIQA_RUNS's parent; set to the
backup root, e.g. "/Volumes/Extreme SSD/UCL/miqa_backup").

Usage:
    MIQA_DATA_ROOT="/Volumes/Extreme SSD/UCL/miqa_backup" \
        MIQA_FIGS="figures/thesis" \
        python repo/scripts/make_schematic_figures.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
from matplotlib.gridspec import GridSpec

rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

DATA_ROOT = Path(os.environ.get("MIQA_DATA_ROOT",
                                 "/raid/candi/michal/miqa"))
INDEX_CSV = DATA_ROOT / "repo" / "dataset" / "index_full.csv"
OUT = Path(os.environ.get("MIQA_FIGS", "figures/thesis"))
OUT.mkdir(parents=True, exist_ok=True)


def _save(fig, name):
    p_png = OUT / f"{name}.png"
    p_pdf = OUT / f"{name}.pdf"
    fig.savefig(p_png)
    fig.savefig(p_pdf)
    plt.close(fig)
    print(f"  -> {p_png.name} (+ .pdf)")


def _load_volume(rel_path):
    """Load a NIfTI from a path relative to DATA_ROOT and return (D,H,W) array."""
    p = DATA_ROOT / rel_path
    if not p.exists():
        return None
    vol = nib.load(str(p)).get_fdata().astype(np.float32)
    return vol


def _percentile_norm(vol):
    lo, hi = np.percentile(vol, 0.5), np.percentile(vol, 99.5)
    vol = np.clip(vol, lo, hi)
    return (vol - lo) / (hi - lo + 1e-8)


def _mask_bbox(mask):
    coords = np.argwhere(mask > 0)
    if not len(coords):
        return None
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    return mins, maxs


# ---------------------------------------------------------------------------
# Fig 3.1  Dataset overview: distorted vs undistorted across b-values
# ---------------------------------------------------------------------------

def fig3_dataset_overview():
    df = pd.read_csv(INDEX_CSV)
    bvals = sorted(df["b_value"].unique(), key=str)
    ncols = len(bvals)
    fig, axes = plt.subplots(2, ncols, figsize=(2.4 * ncols, 5.2))
    if ncols == 1:
        axes = axes[:, None]

    for col, bv in enumerate(bvals):
        # Pick one distorted and one undistorted case at this b-value
        sub_d = df[(df["b_value"] == bv) & (df["label"] == 1)].head(1)
        sub_u = df[(df["b_value"] == bv) & (df["label"] == 0)].head(1)
        for row, sub in enumerate([sub_u, sub_d]):
            ax = axes[row, col]
            ax.axis("off")
            if sub.empty:
                ax.text(0.5, 0.5, "—", ha="center", va="center")
                continue
            r = sub.iloc[0]
            vol = _load_volume(r["dwi_path"])
            mask = _load_volume(r["mask_path"])
            if vol is None or mask is None:
                ax.text(0.5, 0.5, "(missing)", ha="center", va="center")
                continue
            # Pick the middle slice within the prostate mask
            bb = _mask_bbox(mask > 0)
            if bb is None:
                z = vol.shape[2] // 2
            else:
                z = (bb[0][2] + bb[1][2]) // 2
            sl = _percentile_norm(vol[:, :, z])
            ax.imshow(sl.T, cmap="gray", origin="lower")
            # Overlay mask outline
            m = mask[:, :, z].T > 0
            ax.contour(m, levels=[0.5], colors="yellow", linewidths=0.7)
            if col == 0:
                ax.set_ylabel(["Undistorted", "Distorted"][row],
                              rotation=90, labelpad=10,
                              fontsize=12, fontweight="bold")
                ax.yaxis.set_visible(True); ax.set_yticks([])
            if row == 0:
                ax.set_title(f"b={bv}")

    fig.suptitle("Prostate DWI dataset: representative slices "
                 "(yellow contour = prostate mask)", fontsize=12)
    plt.tight_layout()
    _save(fig, "fig3_dataset_overview")


# ---------------------------------------------------------------------------
# Fig 3.2  Label distribution per b-value
# ---------------------------------------------------------------------------

def fig3_label_distribution():
    df = pd.read_csv(INDEX_CSV)
    counts = (df.groupby(["b_value", "label"]).size().unstack(fill_value=0)
              .rename(columns={0: "Undistorted", 1: "Distorted"}))
    bvals = list(counts.index)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(bvals))
    w = 0.55
    ax.bar(x, counts["Undistorted"], w, label="Undistorted", color="#4c72b0",
           alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.bar(x, counts["Distorted"], w, bottom=counts["Undistorted"],
           label="Distorted", color="#c44e52", alpha=0.85,
           edgecolor="black", linewidth=0.5)
    for i, bv in enumerate(bvals):
        tot = counts.loc[bv].sum()
        ax.text(i, tot + 5, f"n={tot}", ha="center", fontsize=9)
        # Inline counts
        u = int(counts.loc[bv, "Undistorted"])
        d = int(counts.loc[bv, "Distorted"])
        ax.text(i, u / 2, f"{u}", ha="center", va="center", color="white",
                fontsize=9)
        ax.text(i, u + d / 2, f"{d}", ha="center", va="center", color="white",
                fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([str(b) for b in bvals])
    ax.set_xlabel("b-value group")
    ax.set_ylabel("Number of acquisitions")
    ax.set_title(f"Dataset composition  (total n = {len(df)} acquisitions, "
                 f"{df['patient_id'].nunique()} patients)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    _save(fig, "fig3_label_distribution")


# ---------------------------------------------------------------------------
# Fig 3.3  Patient-level 5-fold CV schematic
# ---------------------------------------------------------------------------

def fig3_kfold_schematic():
    fig, ax = plt.subplots(figsize=(9, 4.5))
    n_folds = 5
    # Fold layout: test = fold k, val = (k+1)%5, train = remaining 3
    cmap = {"train": "#4c72b0", "val": "#dd8452", "test": "#c44e52"}
    for k in range(n_folds):
        for j in range(n_folds):
            if j == k:
                role = "test"
            elif j == (k + 1) % n_folds:
                role = "val"
            else:
                role = "train"
            rect = Rectangle((j, n_folds - 1 - k), 1, 1, color=cmap[role],
                             alpha=0.85, ec="black", lw=0.6)
            ax.add_patch(rect)
            ax.text(j + 0.5, n_folds - 1 - k + 0.5, role[0].upper(),
                    ha="center", va="center", color="white", fontsize=11,
                    fontweight="bold")
    ax.set_xlim(0, n_folds); ax.set_ylim(0, n_folds)
    ax.set_xticks(np.arange(n_folds) + 0.5)
    ax.set_xticklabels([f"Fold {f}" for f in range(n_folds)])
    ax.set_yticks(np.arange(n_folds) + 0.5)
    ax.set_yticklabels([f"Iter {k+1}" for k in range(n_folds - 1, -1, -1)])
    ax.set_aspect("equal")
    ax.set_title("Patient-level 5-fold cross-validation\n"
                 "Patients split once into 5 stratified groups; "
                 "every iteration uses 1 fold for test and 1 for val")
    ax.grid(False)
    # Legend
    handles = [Rectangle((0, 0), 1, 1, color=c, alpha=0.85) for c in cmap.values()]
    ax.legend(handles, ["Train", "Val", "Test"], loc="upper right",
              bbox_to_anchor=(1.18, 1.0), frameon=True)
    plt.tight_layout()
    _save(fig, "fig3_kfold_schematic")


# ---------------------------------------------------------------------------
# Fig 3.4  2.5D slice stacking visualization
# ---------------------------------------------------------------------------

def fig3_2_5d_stacking():
    df = pd.read_csv(INDEX_CSV)
    sub = df[df["label"] == 1].head(1)  # pick one distorted case
    if sub.empty:
        return
    r = sub.iloc[0]
    vol = _load_volume(r["dwi_path"])
    mask = _load_volume(r["mask_path"])
    if vol is None or mask is None:
        return
    bb = _mask_bbox(mask > 0)
    if bb is not None:
        r0, r1, c0, c1 = bb[0][0], bb[1][0], bb[0][1], bb[1][1]
        vol = vol[r0:r1, c0:c1, bb[0][2]:bb[1][2]]
        mask = mask[r0:r1, c0:c1, bb[0][2]:bb[1][2]]
    vol = _percentile_norm(vol)
    nz = vol.shape[2]
    center = nz // 2
    k_offsets = [-1, 0, 1]  # k=3
    fig = plt.figure(figsize=(11, 4.2))
    gs = GridSpec(1, 5, width_ratios=[1, 0.3, 1, 0.3, 1.2],
                  hspace=0.3, wspace=0.3)

    # Left: volume strip (5 slices to give context)
    ax_vol = fig.add_subplot(gs[0])
    show_ids = np.linspace(0, nz - 1, 5).astype(int)
    composite = np.concatenate([vol[:, :, z].T for z in show_ids], axis=1)
    ax_vol.imshow(composite, cmap="gray", origin="lower", aspect="auto")
    ax_vol.set_title("3D DWI volume\n(axial slices)", fontsize=10)
    ax_vol.set_xticks([]); ax_vol.set_yticks([])

    # Middle: extract k=3 around center
    ax_mid = fig.add_subplot(gs[2])
    k_slices = [vol[:, :, np.clip(center + o, 0, nz - 1)].T for o in k_offsets]
    composite2 = np.concatenate(k_slices, axis=1)
    ax_mid.imshow(composite2, cmap="gray", origin="lower", aspect="auto")
    for i, o in enumerate(k_offsets):
        ax_mid.text((i + 0.5) * k_slices[0].shape[1], -8,
                    f"slice z={center+o}", ha="center", fontsize=9)
    ax_mid.set_title("Extract k=3 central axial slices", fontsize=10)
    ax_mid.set_xticks([]); ax_mid.set_yticks([])

    # Right: stacked-as-channels visualization (pseudo-RGB)
    ax_r = fig.add_subplot(gs[4])
    rgb = np.stack(k_slices, axis=-1)
    ax_r.imshow(rgb / max(1e-6, rgb.max()), origin="lower")
    ax_r.set_title("Stack as 3-channel input\n(2.5D representation)",
                   fontsize=10)
    ax_r.set_xticks([]); ax_r.set_yticks([])

    # Arrows between panels
    for ax_a, ax_b in [(ax_vol, ax_mid), (ax_mid, ax_r)]:
        a_pos = ax_a.get_position()
        b_pos = ax_b.get_position()
        arrow = FancyArrowPatch(
            (a_pos.x1 + 0.005, (a_pos.y0 + a_pos.y1) / 2),
            (b_pos.x0 - 0.005, (b_pos.y0 + b_pos.y1) / 2),
            transform=fig.transFigure, arrowstyle="-|>",
            mutation_scale=18, color="black", linewidth=1.5)
        fig.add_artist(arrow)

    fig.suptitle("2.5D input construction: k=3 axial slices stacked as channels")
    plt.tight_layout()
    _save(fig, "fig3_2_5d_stacking")


# ---------------------------------------------------------------------------
# Fig 3.5  Preprocessing pipeline with real sample at each step
# ---------------------------------------------------------------------------

def fig3_preprocessing_pipeline():
    df = pd.read_csv(INDEX_CSV)
    sub = df[df["label"] == 1].head(1)
    if sub.empty:
        return
    r = sub.iloc[0]
    vol = _load_volume(r["dwi_path"])
    mask = _load_volume(r["mask_path"])
    if vol is None or mask is None:
        return

    nz = vol.shape[2]
    z = nz // 2

    # Step 1: raw slice
    s_raw = vol[:, :, z]
    # Step 2: mask bbox crop
    bb = _mask_bbox(mask > 0)
    if bb is None:
        s_crop = s_raw
    else:
        r0, r1, c0, c1 = bb[0][0], bb[1][0], bb[0][1], bb[1][1]
        s_crop = vol[r0:r1, c0:c1, z]
    # Step 3: percentile normalise
    s_norm = _percentile_norm(s_crop)
    # Step 4: pad to square + resize visually
    h, w = s_norm.shape
    side = max(h, w)
    pad = np.zeros((side, side), dtype=np.float32)
    ph = (side - h) // 2; pw = (side - w) // 2
    pad[ph:ph + h, pw:pw + w] = s_norm
    s_pad = pad
    # Step 5: 224x224 (resized); use the same image but represent as 'final'
    from numpy import interp
    # simple nearest-resize for visualisation
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(s_pad).float()[None, None]
    s_final = F.interpolate(t, size=(224, 224), mode="bilinear",
                            align_corners=False).squeeze().numpy()

    steps = [
        ("Raw axial slice",       s_raw.T),
        ("Mask bounding-box crop", s_crop.T),
        ("Percentile-normalise\n[0.5, 99.5]%", s_norm.T),
        ("Pad to square",         s_pad.T),
        ("Resize 224×224",        s_final.T),
    ]
    fig, axes = plt.subplots(1, len(steps), figsize=(15, 3.3))
    for ax, (title, img) in zip(axes, steps):
        ax.imshow(img, cmap="gray", origin="lower", aspect="auto")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    # Add arrows between panels
    for i in range(len(steps) - 1):
        a = axes[i].get_position()
        b = axes[i + 1].get_position()
        arrow = FancyArrowPatch(
            (a.x1 + 0.002, (a.y0 + a.y1) / 2),
            (b.x0 - 0.002, (b.y0 + b.y1) / 2),
            transform=fig.transFigure, arrowstyle="-|>",
            mutation_scale=16, color="black", linewidth=1.4)
        fig.add_artist(arrow)
    fig.suptitle("Preprocessing pipeline (per slice)", fontsize=12)
    plt.tight_layout()
    _save(fig, "fig3_preprocessing_pipeline")


# ---------------------------------------------------------------------------
# Architecture diagrams: pure schematic (no NIfTI needed)
# ---------------------------------------------------------------------------

def fig3_architecture_schematic():
    """Simplified block diagram of the 4 CNNs used."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4)
    ax.axis("off")

    # Input box
    ax.add_patch(FancyBboxPatch((0.2, 1.6), 1.4, 0.8, boxstyle="round,pad=0.08",
                                 fc="#dcdcdc", ec="black", lw=1))
    ax.text(0.9, 2.0, "Input\n3×224×224", ha="center", va="center", fontsize=9)

    # Backbones (4)
    backbones = [
        ("ResNet-18\n(11M)",    "#ff7f0e"),
        ("ResNet-50\n(25M)",    "#2ca02c"),
        ("DenseNet-121\n(8M)",  "#1f77b4"),
        ("EffNet-B0\n(5M)",     "#9467bd"),
    ]
    x_back = 2.5
    y_top = 3.4
    for i, (lbl, col) in enumerate(backbones):
        y = y_top - i * 0.85
        ax.add_patch(FancyBboxPatch((x_back, y - 0.35), 2.3, 0.7,
                                     boxstyle="round,pad=0.08",
                                     fc=col, ec="black", lw=1, alpha=0.85))
        ax.text(x_back + 1.15, y, lbl, ha="center", va="center", fontsize=9,
                color="white", fontweight="bold")
        # Arrow from input to backbone
        ax.add_patch(FancyArrowPatch((1.6, 2.0), (x_back, y),
                                      arrowstyle="-|>", mutation_scale=12,
                                      lw=1, color="black"))

    # GAP + FC head
    x_head = 6.0
    ax.add_patch(FancyBboxPatch((x_head, 1.6), 1.6, 0.8, boxstyle="round,pad=0.08",
                                 fc="#f5deb3", ec="black", lw=1))
    ax.text(x_head + 0.8, 2.0,
            "Global avg pool\n+ FC head\n(1 logit)",
            ha="center", va="center", fontsize=9)
    for i in range(4):
        y = y_top - i * 0.85
        ax.add_patch(FancyArrowPatch((x_back + 2.3, y),
                                      (x_head, 2.0),
                                      arrowstyle="-|>", mutation_scale=12,
                                      lw=1, color="black"))

    # Output
    x_out = 8.6
    ax.add_patch(FancyBboxPatch((x_out, 1.6), 2.0, 0.8, boxstyle="round,pad=0.08",
                                 fc="#c44e52", ec="black", lw=1, alpha=0.85))
    ax.text(x_out + 1.0, 2.0,
            "P(distorted)\nbinary classifier",
            ha="center", va="center", fontsize=9, color="white",
            fontweight="bold")
    ax.add_patch(FancyArrowPatch((x_head + 1.6, 2.0), (x_out, 2.0),
                                  arrowstyle="-|>", mutation_scale=14,
                                  lw=1, color="black"))

    ax.set_title("Architecture schematic: four CNN backbones, shared input "
                 "and head", fontsize=11)
    plt.tight_layout()
    _save(fig, "fig3_architecture_schematic")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"DATA_ROOT = {DATA_ROOT}")
    print(f"INDEX_CSV = {INDEX_CSV} (exists: {INDEX_CSV.exists()})")
    print(f"OUT       = {OUT}\n")

    figs = [
        ("Fig 3.1  Dataset overview",      fig3_dataset_overview),
        ("Fig 3.2  Label distribution",    fig3_label_distribution),
        ("Fig 3.3  Patient-level CV",      fig3_kfold_schematic),
        ("Fig 3.4  2.5D stacking",         fig3_2_5d_stacking),
        ("Fig 3.5  Preprocessing pipeline", fig3_preprocessing_pipeline),
        ("Fig 3.6  Architecture schematic", fig3_architecture_schematic),
    ]
    for label, fn in figs:
        print(f"[{label}]")
        try:
            fn()
        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")

    print(f"\nDone. Figures in: {OUT.resolve()}")


if __name__ == "__main__":
    main()
