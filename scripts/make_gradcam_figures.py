"""Grad-CAM attention maps for the thesis (Chapter 9 / interpretability section).

Produces:
  fig9_gradcam_per_arch.pdf   : 3 archs x 4 case types (TP/FP/FN/TN) with
                                heatmap overlays + true label + predicted prob.
  fig9_gradcam_uncertain.pdf  : Grad-CAM for the most-uncertain ensemble cases
                                (visualizes where models look when unsure).

Reads model checkpoints from MIQA_RUNS and NIfTI data from MIQA_DATA_ROOT.

Usage:
    MIQA_RUNS="/Volumes/Extreme SSD/UCL/miqa_backup/runs" \
        MIQA_DATA_ROOT="/Volumes/Extreme SSD/UCL/miqa_backup" \
        MIQA_FIGS="figures/thesis" \
        python repo/scripts/make_gradcam_figures.py
"""
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import nibabel as nib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import cm

from miqa.models import build_model

rcParams.update({
    "font.family": "serif", "font.size": 11,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

RUNS = Path(os.environ.get("MIQA_RUNS", "/raid/candi/michal/miqa/runs"))
DATA_ROOT = Path(os.environ.get("MIQA_DATA_ROOT", "/raid/candi/michal/miqa"))
OUT = Path(os.environ.get("MIQA_FIGS", "figures/thesis"))
OUT.mkdir(parents=True, exist_ok=True)

ARCHS = ["densenet121", "resnet18", "resnet50"]
ARCH_LABEL = {"densenet121": "DenseNet-121", "resnet18": "ResNet-18",
              "resnet50": "ResNet-50"}


# ---------------------------------------------------------------------------
# Grad-CAM implementation
# ---------------------------------------------------------------------------

class GradCAM:
    """Gradient-weighted Class Activation Mapping (Selvaraju et al. 2017).

    For binary classification with a single logit, the "class" is the
    positive class (distortion); we backprop the logit value itself.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        self._fwd_handle = target_layer.register_forward_hook(self._fwd_hook)
        # full_backward_hook is the modern API (forward+backward gradient correct)
        self._bwd_handle = target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self.activations = out.detach()

    def _bwd_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def compute(self, input_tensor, target="positive"):
        """Return (cam_normalized [H,W], sigmoid(logit) scalar)."""
        self.model.zero_grad()
        input_tensor = input_tensor.clone().requires_grad_(True)
        logit = self.model(input_tensor)
        if logit.ndim == 1:
            logit = logit.unsqueeze(-1)
        # Score: positive logit (encourages backprop wrt distortion class)
        score = logit[0, 0]
        if target == "negative":
            score = -score
        score.backward()
        # Channel weights = global-average-pooled gradients
        if self.gradients is None or self.activations is None:
            return None, float(torch.sigmoid(logit).detach().item())
        weights = self.gradients.mean(dim=(2, 3))           # (1, C)
        cam = (weights[:, :, None, None] * self.activations).sum(dim=1)  # (1, h, w)
        cam = F.relu(cam)
        cam = F.interpolate(cam.unsqueeze(1), size=input_tensor.shape[-2:],
                            mode="bilinear", align_corners=False).squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam, float(torch.sigmoid(logit).detach().item())

    def close(self):
        try:
            self._fwd_handle.remove()
        except Exception:
            pass
        try:
            self._bwd_handle.remove()
        except Exception:
            pass


def get_target_layer(model, arch):
    """Return the last convolutional block of each architecture."""
    if arch in ("resnet18", "resnet50"):
        return model.layer4[-1]
    if arch == "densenet121":
        return model.features.denseblock4
    if arch == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(f"Unknown arch: {arch}")


# ---------------------------------------------------------------------------
# Preprocessing: matches UCLDistortionDataset's transforms exactly
# ---------------------------------------------------------------------------

def _percentile_norm(vol):
    lo, hi = np.percentile(vol, 0.5), np.percentile(vol, 99.5)
    vol = np.clip(vol, lo, hi)
    return (vol - lo) / (hi - lo + 1e-8)


def preprocess_case(row, k_slices=3, img_size=224):
    """Mirror UCLDistortionDataset preprocessing for a single case."""
    p_dwi = DATA_ROOT / row["dwi_path"]
    p_mask = DATA_ROOT / row["mask_path"]
    if not (p_dwi.exists() and p_mask.exists()):
        return None, None
    vol = nib.load(str(p_dwi)).get_fdata().astype(np.float32)
    mask = nib.load(str(p_mask)).get_fdata().astype(np.float32)
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None, None
    mins = coords.min(axis=0); maxs = coords.max(axis=0) + 1
    r0, r1, c0, c1, z0, z1 = mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]
    vol_crop = _percentile_norm(vol[r0:r1, c0:c1, z0:z1])
    nz = vol_crop.shape[2]
    center = nz // 2
    half_k = k_slices // 2
    slices = []
    for o in range(-half_k, half_k + 1):
        z = int(np.clip(center + o, 0, nz - 1))
        slices.append(vol_crop[:, :, z])
    stack = np.stack(slices, axis=0)
    _, h, w = stack.shape
    side = max(h, w)
    padded = np.zeros((k_slices, side, side), dtype=np.float32)
    ph, pw = (side - h) // 2, (side - w) // 2
    padded[:, ph:ph + h, pw:pw + w] = stack
    x = torch.from_numpy(padded).float().unsqueeze(0)        # (1, k, S, S)
    x = F.interpolate(x, size=(img_size, img_size), mode="bilinear",
                      align_corners=False)
    # Display image: central slice resized to img_size
    central = stack[k_slices // 2]
    disp = np.zeros((side, side), dtype=np.float32)
    disp[ph:ph + h, pw:pw + w] = central
    d = torch.from_numpy(disp).unsqueeze(0).unsqueeze(0).float()
    d = F.interpolate(d, size=(img_size, img_size), mode="bilinear",
                      align_corners=False).squeeze().numpy()
    return x, d


# ---------------------------------------------------------------------------
# Case selection: pick representative TP / FP / FN / TN per architecture
# ---------------------------------------------------------------------------

def select_cases(arch):
    """Return one case_id per type {TP, FP, FN, TN} for the given arch.

    Pooled across the 5 folds' test_predictions.csv; picks extreme-confidence
    examples (highest-confidence-correct and highest-confidence-wrong).
    """
    files = sorted(glob.glob(str(RUNS / f"poster100_{arch}_dwi_roi_f*/test_predictions.csv")))
    if not files:
        return {}
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs).drop_duplicates("case_id")
    df["pred"] = (df["y_prob"] >= 0.5).astype(int)
    return {
        "TP": df[(df["pred"] == 1) & (df["y_true"] == 1)]
              .sort_values("y_prob", ascending=False).iloc[0]["case_id"]
              if ((df["pred"] == 1) & (df["y_true"] == 1)).any() else None,
        "FP": df[(df["pred"] == 1) & (df["y_true"] == 0)]
              .sort_values("y_prob", ascending=False).iloc[0]["case_id"]
              if ((df["pred"] == 1) & (df["y_true"] == 0)).any() else None,
        "FN": df[(df["pred"] == 0) & (df["y_true"] == 1)]
              .sort_values("y_prob").iloc[0]["case_id"]
              if ((df["pred"] == 0) & (df["y_true"] == 1)).any() else None,
        "TN": df[(df["pred"] == 0) & (df["y_true"] == 0)]
              .sort_values("y_prob").iloc[0]["case_id"]
              if ((df["pred"] == 0) & (df["y_true"] == 0)).any() else None,
    }


def get_model_for_case(arch, case_id):
    """Find which fold's checkpoint contains this case_id in its test set.

    Returns (model, fold_idx) or (None, None) if not found.
    """
    for f in range(5):
        csv = RUNS / f"poster100_{arch}_dwi_roi_f{f}" / "test_predictions.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        if case_id in df["case_id"].values:
            ckpt = RUNS / f"poster100_{arch}_dwi_roi_f{f}" / "best_auroc.pt"
            if not ckpt.exists():
                continue
            model = build_model(arch, in_chans=3, pretrained=False)
            ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
            model.load_state_dict(ck["model"])
            model.eval()
            return model, f
    return None, None


# ---------------------------------------------------------------------------
# Grid figure
# ---------------------------------------------------------------------------

def fig_gradcam_grid():
    """3 archs (rows) x 4 case types (cols) = 12 panels."""
    case_types = ["TP", "FP", "FN", "TN"]
    df_idx = pd.read_csv(DATA_ROOT / "repo" / "dataset" / "index_full.csv").set_index("case_id")

    fig, axes = plt.subplots(len(ARCHS), len(case_types),
                              figsize=(3.2 * len(case_types), 3.0 * len(ARCHS)))
    for i, arch in enumerate(ARCHS):
        cases = select_cases(arch)
        for j, ct in enumerate(case_types):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            cid = cases.get(ct)
            if cid is None or cid not in df_idx.index:
                ax.text(0.5, 0.5, f"(no {ct})", ha="center", va="center")
                if i == 0: ax.set_title(ct, fontsize=11, fontweight="bold")
                if j == 0: ax.set_ylabel(ARCH_LABEL[arch], fontsize=11,
                                          fontweight="bold")
                continue
            row = df_idx.loc[cid]
            model, fold = get_model_for_case(arch, cid)
            if model is None:
                ax.text(0.5, 0.5, "(no ckpt)", ha="center", va="center")
                continue
            target = get_target_layer(model, arch)
            gc = GradCAM(model, target)
            x, disp = preprocess_case(row)
            if x is None:
                ax.text(0.5, 0.5, "(load fail)", ha="center", va="center")
                gc.close(); continue
            cam, prob = gc.compute(x, target="positive")
            gc.close()
            ax.imshow(disp.T, cmap="gray", origin="lower")
            if cam is not None:
                ax.imshow(cam.T, cmap="jet", alpha=0.4, origin="lower")
            true_lbl = "Distorted" if int(row["label"]) == 1 else "Undistorted"
            ax.set_title(f"{ct}: P={prob:.2f}\n(true: {true_lbl}, fold {fold})",
                         fontsize=9)
            if i == 0:
                # column header above is already in the title; leave bold col labels off
                pass
            if j == 0:
                ax.set_ylabel(ARCH_LABEL[arch], fontsize=11, fontweight="bold",
                              rotation=90, labelpad=12)
    fig.suptitle("Grad-CAM attention on representative test cases\n"
                 "(red = high attention; overlaid on input slice)", fontsize=12)
    plt.tight_layout()
    p_png = OUT / "fig9_gradcam_per_arch.png"
    p_pdf = OUT / "fig9_gradcam_per_arch.pdf"
    fig.savefig(p_png); fig.savefig(p_pdf); plt.close(fig)
    print(f"  -> {p_png.name} (+ .pdf)")


# ---------------------------------------------------------------------------
# High-uncertainty figure
# ---------------------------------------------------------------------------

def fig_gradcam_uncertain():
    """For each arch, find the most-uncertain ensemble case and show Grad-CAM.

    Uses uq_ensemble_<arch>_f<fold>/predictions_uncertainty.csv (output of
    evaluate_uncertainty.py in Deep Ensemble mode) to identify high-uncertainty
    cases, then runs Grad-CAM on one fold's checkpoint.
    """
    df_idx = pd.read_csv(DATA_ROOT / "repo" / "dataset" / "index_full.csv").set_index("case_id")
    fig, axes = plt.subplots(len(ARCHS), 3, figsize=(10, 3.0 * len(ARCHS)))

    for i, arch in enumerate(ARCHS):
        # Aggregate uncertainty across folds
        u_files = sorted(glob.glob(
            str(RUNS / f"uq_ensemble_{arch}_f*/predictions_uncertainty.csv")))
        if not u_files:
            for j in range(3):
                axes[i, j].text(0.5, 0.5, "(no UQ data)", ha="center")
                axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            continue
        dfs = [pd.read_csv(f) for f in u_files]
        u = pd.concat(dfs).drop_duplicates("case_id")
        # Pick the 3 most-uncertain (highest std)
        top = u.sort_values("std_prob", ascending=False).head(3)
        for j, (_, r) in enumerate(top.iterrows()):
            cid = r["case_id"]
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            if cid not in df_idx.index:
                ax.text(0.5, 0.5, "(case missing)", ha="center"); continue
            row = df_idx.loc[cid]
            model, fold = get_model_for_case(arch, cid)
            if model is None:
                ax.text(0.5, 0.5, "(no ckpt)", ha="center"); continue
            target = get_target_layer(model, arch)
            gc = GradCAM(model, target)
            x, disp = preprocess_case(row)
            if x is None:
                ax.text(0.5, 0.5, "(load fail)", ha="center"); gc.close(); continue
            cam, prob = gc.compute(x, target="positive")
            gc.close()
            ax.imshow(disp.T, cmap="gray", origin="lower")
            if cam is not None:
                ax.imshow(cam.T, cmap="jet", alpha=0.4, origin="lower")
            true_lbl = "Distorted" if int(row["label"]) == 1 else "Undistorted"
            ax.set_title(f"σ={r['std_prob']:.2f}, mean P={r['mean_prob']:.2f}\n"
                         f"(true: {true_lbl})", fontsize=9)
            if j == 0:
                ax.set_ylabel(ARCH_LABEL[arch], fontsize=11, fontweight="bold")
    fig.suptitle("Grad-CAM on the most-uncertain ensemble cases\n"
                 "(visualises where the model attends when ensemble disagrees)",
                 fontsize=12)
    plt.tight_layout()
    p_png = OUT / "fig9_gradcam_uncertain.png"
    p_pdf = OUT / "fig9_gradcam_uncertain.pdf"
    fig.savefig(p_png); fig.savefig(p_pdf); plt.close(fig)
    print(f"  -> {p_png.name} (+ .pdf)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"RUNS={RUNS}\nDATA_ROOT={DATA_ROOT}\nOUT={OUT}\n")
    print("[Grad-CAM grid (TP/FP/FN/TN x 3 archs)]")
    fig_gradcam_grid()
    print("[Grad-CAM on uncertain ensemble cases]")
    fig_gradcam_uncertain()
    print(f"\nDone. Figures in {OUT.resolve()}")


if __name__ == "__main__":
    main()
