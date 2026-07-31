"""Generate Grad-CAM heatmaps for model interpretability analysis.

Overlays activation maps on DWI slices to show what the model attends to.
Categorises outputs into: correct predictions, incorrect predictions,
and high-uncertainty cases (if uncertainty data available).

Usage:
    python scripts/generate_gradcam.py \
        --ckpt runs/best_model/best_auroc.pt \
        --index_csv /path/to/index.csv \
        --outdir figures/gradcam

Outputs:
    - gradcam_<case_id>_<correct|incorrect>.png per test case
    - gradcam_summary.json with case-level metadata
    - montage_correct.png / montage_incorrect.png composite images
"""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from miqa.models import build_model
from miqa.datasets import UCLDistortionDataset
from miqa.utils import load_test_split, load_or_create_kfold_splits


def _get_target_layer(model, model_name):
    """Return the last convolutional layer for Grad-CAM."""
    if model_name in ("resnet18", "resnet50"):
        return model.layer4[-1]
    elif model_name == "densenet121":
        return model.features.denseblock4
    elif model_name == "efficientnet_b0":
        return model.features[-1]
    else:
        raise ValueError(f"Unknown model for Grad-CAM: {model_name}")


class GradCAM:
    """Grad-CAM for binary classification (single logit output)."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(self, x, device):
        """Generate Grad-CAM heatmap for input tensor x (B, C, H, W).

        Returns: heatmaps (B, H, W) in [0, 1], logits (B,)
        """
        self.model.eval()
        x = x.to(device)
        x.requires_grad_(True)

        logit = self.model(x).squeeze(-1)  # (B,)

        # Backward pass: gradient of logit w.r.t. target layer
        self.model.zero_grad()
        logit.sum().backward()

        # Grad-CAM computation
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1)  # (B, H_feat, W_feat)
        cam = F.relu(cam)

        # Upsample to input size
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=(x.shape[2], x.shape[3]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)  # (B, H, W)

        # Normalize per sample
        for i in range(cam.shape[0]):
            c = cam[i]
            cmin, cmax = c.min(), c.max()
            if cmax - cmin > 1e-8:
                cam[i] = (c - cmin) / (cmax - cmin)
            else:
                cam[i] = torch.zeros_like(c)

        return cam.cpu().numpy(), logit.detach().cpu().numpy()


def make_overlay(image_2d, heatmap, alpha=0.4):
    """Overlay heatmap on grayscale image. Returns RGB array (H, W, 3)."""
    # Normalize image to [0, 1]
    img = image_2d.copy()
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())

    # Create RGB from grayscale
    rgb = np.stack([img, img, img], axis=-1)

    # Apply colormap to heatmap
    cmap = plt.cm.jet
    heatmap_colored = cmap(heatmap)[:, :, :3]  # (H, W, 3)

    # Blend
    overlay = (1 - alpha) * rgb + alpha * heatmap_colored
    overlay = np.clip(overlay, 0, 1)
    return overlay


def main():
    ap = argparse.ArgumentParser(description="Grad-CAM heatmap generation")

    ap.add_argument("--ckpt", required=True,
                    help="Path to model checkpoint (best_auroc.pt)")
    ap.add_argument("--index_csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--n_folds", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--max_cases", type=int, default=0,
                    help="Limit cases for testing (0=all)")
    ap.add_argument("--alpha", type=float, default=0.4,
                    help="Heatmap overlay transparency")
    ap.add_argument("--n_montage", type=int, default=16,
                    help="Max cases per montage grid")
    ap.add_argument("--uncertainty_csv", type=str, default="",
                    help="Optional predictions_uncertainty.csv for "
                         "high-uncertainty case selection")

    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Load checkpoint and config ---
    ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    model_name = ckpt_args.get("model", "resnet18")
    k_slices = ckpt_args.get("k_slices", 3)
    modality = ckpt_args.get("modality", "dwi")
    use_mask_crop = ckpt_args.get("use_mask_crop", True)
    split_seed = ckpt_args.get("split_seed", args.split_seed)

    in_chans = k_slices * 2 if modality == "fusion" else k_slices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Model ---
    model = build_model(model_name, in_chans=in_chans, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    target_layer = _get_target_layer(model, model_name)
    gcam = GradCAM(model, target_layer)

    # --- Data ---
    df = pd.read_csv(args.index_csv)
    group_col = "patient_id" if "patient_id" in df.columns else None
    if args.max_cases > 0:
        ids = sorted(df["case_id"].unique())
        rng = np.random.RandomState(split_seed)
        rng.shuffle(ids)
        ids = ids[:args.max_cases]
        df = df[df["case_id"].isin(set(ids))]

    run_dir = str(Path(args.ckpt).parent)
    if args.n_folds > 0:
        _, _, te = load_or_create_kfold_splits(
            df, args.n_folds, args.fold, split_seed, out,
            group_col=group_col)
    else:
        te = load_test_split(df, run_dir=run_dir, split_seed=split_seed,
                             group_col=group_col)

    print(f"Test set: {len(te)} cases")
    print(f"Model: {model_name}, modality: {modality}, k_slices: {k_slices}")

    ds = UCLDistortionDataset(te, modality=modality,
                              use_mask_crop=use_mask_crop,
                              k_slices=k_slices, img_size=args.img_size,
                              augment=False, return_case_id=True)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)

    # --- Optional uncertainty data ---
    uncertainty_map = {}
    if args.uncertainty_csv and Path(args.uncertainty_csv).exists():
        uq_df = pd.read_csv(args.uncertainty_csv)
        for _, row in uq_df.iterrows():
            uncertainty_map[str(row["case_id"])] = float(row.get("std_prob", 0))

    # --- Generate Grad-CAMs ---
    records = []
    correct_images = []
    incorrect_images = []
    high_unc_images = []

    for x, y, cids in dl:
        heatmaps, logits = gcam.generate(x, device)
        probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
        preds = (probs >= 0.5).astype(int)
        labels = y.numpy().astype(int).ravel()

        for i in range(x.shape[0]):
            cid = cids[i]
            label = int(labels[i])
            pred = int(preds[i])
            prob = float(probs[i])
            correct = pred == label
            category = "correct" if correct else "incorrect"

            # Use middle slice channel for visualization
            mid_ch = x.shape[1] // 2
            img_2d = x[i, mid_ch].numpy()
            hmap = heatmaps[i]

            overlay = make_overlay(img_2d, hmap, alpha=args.alpha)

            # Save individual image
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img_2d, cmap="gray")
            axes[0].set_title(f"Input (ch={mid_ch})")
            axes[0].axis("off")

            axes[1].imshow(hmap, cmap="jet", vmin=0, vmax=1)
            axes[1].set_title("Grad-CAM")
            axes[1].axis("off")

            axes[2].imshow(overlay)
            tag = "CORRECT" if correct else "WRONG"
            axes[2].set_title(
                f"{tag} | true={label} pred={pred} p={prob:.2f}")
            axes[2].axis("off")

            fname = f"gradcam_{cid}_{category}.png"
            fig.tight_layout()
            fig.savefig(out / fname, dpi=100, bbox_inches="tight")
            plt.close(fig)

            # Collect for montage
            if correct:
                correct_images.append((overlay, cid, prob, label))
            else:
                incorrect_images.append((overlay, cid, prob, label))

            unc = uncertainty_map.get(str(cid), None)
            if unc is not None and unc > np.median(
                    list(uncertainty_map.values())):
                high_unc_images.append((overlay, cid, prob, label, unc))

            records.append({
                "case_id": cid,
                "y_true": label,
                "y_pred": pred,
                "y_prob": round(prob, 4),
                "correct": correct,
                "uncertainty": unc,
            })

    # --- Montage grids ---
    def save_montage(images_info, path, title, n_max):
        if not images_info:
            return
        n = min(len(images_info), n_max)
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes[np.newaxis, :]
        elif cols == 1:
            axes = axes[:, np.newaxis]

        for idx in range(rows * cols):
            r, c = idx // cols, idx % cols
            axes[r, c].axis("off")
            if idx < n:
                img_data = images_info[idx]
                overlay = img_data[0]
                cid = img_data[1]
                prob = img_data[2]
                label = img_data[3]
                axes[r, c].imshow(overlay)
                axes[r, c].set_title(f"{cid}\ntrue={label} p={prob:.2f}",
                                     fontsize=8)

        fig.suptitle(title, fontsize=14)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    save_montage(correct_images, out / "montage_correct.png",
                 "Correct Predictions (Grad-CAM)", args.n_montage)
    save_montage(incorrect_images, out / "montage_incorrect.png",
                 "Incorrect Predictions (Grad-CAM)", args.n_montage)
    if high_unc_images:
        save_montage(high_unc_images, out / "montage_high_uncertainty.png",
                     "High Uncertainty Cases (Grad-CAM)", args.n_montage)

    # --- Summary ---
    summary = {
        "model": model_name,
        "modality": modality,
        "n_test": len(records),
        "n_correct": sum(1 for r in records if r["correct"]),
        "n_incorrect": sum(1 for r in records if not r["correct"]),
        "n_high_uncertainty": len(high_unc_images),
        "checkpoint": args.ckpt,
    }
    (out / "gradcam_summary.json").write_text(json.dumps(summary, indent=2))

    # Per-case records
    pd.DataFrame(records).to_csv(out / "gradcam_cases.csv", index=False)

    print(f"\nDone. {len(records)} Grad-CAM images saved to {out}")
    print(f"  Correct: {summary['n_correct']}, "
          f"Incorrect: {summary['n_incorrect']}")
    if high_unc_images:
        print(f"  High uncertainty: {len(high_unc_images)}")


if __name__ == "__main__":
    main()
