"""Synthetic distortion injection: physics-motivated validation.

Tests whether the trained model genuinely detects geometric distortion (rather
than dataset-specific shortcuts) by injecting controlled B0-inhomogeneity-style
distortion into UNDISTORTED test cases at increasing severity and measuring the
model's predicted P(distorted).

Distortion model: B0 field inhomogeneity in EPI/DWI shifts pixels along the
phase-encode (assumed vertical) direction proportionally to local off-resonance.
We approximate the off-resonance field with a smooth low-frequency field (sum of
random low-order sinusoids) and warp the image vertically by `severity` pixels
(max displacement). This is a 2D approximation applied to the preprocessed
slice tensor, a deliberate simplification (noted as a limitation).

Outputs:
  - distortion_response.csv     per-severity mean P(distorted) + detection rate
  - distortion_response.png     severity-response curve
  - distortion_summary.json     detection threshold + curve

Usage:
    python repo/scripts/evaluate_synthetic_distortion.py \
        --ckpt runs/poster100_densenet121_dwi_roi_f0/best_auroc.pt \
        --index_csv repo/dataset/index_full.csv \
        --outdir runs/syndist_densenet121_f0 \
        --n_folds 5 --fold 0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from miqa.datasets import UCLDistortionDataset
from miqa.models import build_model
from miqa.utils import amp_autocast, load_test_split


# ---------------------------------------------------------------------------
# Distortion model (pure functions, unit-testable without data/model)
# ---------------------------------------------------------------------------

def make_displacement_field(H, W, severity, seed=0, n_modes=3):
    """Smooth vertical displacement field of shape (H, W), in pixels.

    Sum of a few low-frequency 2D sinusoids (mimicking a slowly varying
    off-resonance field), normalised to unit max-abs then scaled to `severity`
    pixels of maximum displacement.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W),
                         indexing="ij")
    field = np.zeros((H, W), dtype=np.float32)
    for _ in range(n_modes):
        fx = int(rng.integers(1, 4))
        fy = int(rng.integers(1, 4))
        phase = float(rng.uniform(0, 2 * np.pi))
        amp = float(rng.uniform(0.5, 1.0))
        field += amp * np.sin(2 * np.pi * (fx * xx + fy * yy) + phase)
    field /= (np.abs(field).max() + 1e-8)
    return (field * severity).astype(np.float32)


def warp_vertical(img, disp_pixels):
    """Warp a (C, H, W) tensor by a vertical displacement field (H, W) [pixels].

    Uses grid_sample with border padding. Positive displacement samples from
    further down the image (pile-up/stretch along the phase-encode axis).
    """
    C, H, W = img.shape
    ys = torch.linspace(-1, 1, H)
    xs = torch.linspace(-1, 1, W)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    disp = torch.from_numpy(disp_pixels).float() * (2.0 / max(H - 1, 1))
    gy = gy + disp
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # (1, H, W, 2)
    out = F.grid_sample(img.unsqueeze(0), grid, mode="bilinear",
                        padding_mode="border", align_corners=True)
    return out.squeeze(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Synthetic distortion injection validation")
    ap.add_argument("--ckpt", required=True,
                    help="Trained standard checkpoint (best_auroc.pt)")
    ap.add_argument("--index_csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--n_folds", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--max_cases", type=int, default=0,
                    help="Limit number of undistorted cases (0 = all)")
    ap.add_argument("--severities", type=float, nargs="+",
                    default=[0, 1, 2, 3, 4, 6, 8, 10, 14],
                    help="Max vertical displacement (pixels) per level")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Model config from checkpoint ---
    ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    model_name = ckpt_args.get("model", "resnet18")
    k_slices = ckpt_args.get("k_slices", 3)
    modality = ckpt_args.get("modality", "dwi")
    use_mask_crop = ckpt_args.get("use_mask_crop", True)
    split_seed = ckpt_args.get("split_seed", args.split_seed)
    run_dir = str(Path(args.ckpt).parent)
    in_chans = k_slices * 2 if modality == "fusion" else k_slices

    # --- Data: undistorted test cases only ---
    df = pd.read_csv(args.index_csv)
    group_col = "patient_id" if "patient_id" in df.columns else None
    if args.n_folds > 0:
        from miqa.utils import load_or_create_kfold_splits
        _, _, te = load_or_create_kfold_splits(
            df, args.n_folds, args.fold, split_seed, out, group_col=group_col)
    else:
        te = load_test_split(df, run_dir=run_dir, split_seed=split_seed,
                             group_col=group_col)

    te_undist = te[te["label"] == 0].reset_index(drop=True)
    if args.max_cases > 0:
        te_undist = te_undist.iloc[:args.max_cases].reset_index(drop=True)
    print(f"Undistorted test cases: {len(te_undist)} "
          f"(of {len(te)} total in fold)")
    if len(te_undist) == 0:
        sys.exit("No undistorted cases in this split, nothing to distort.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"

    model = build_model(model_name, in_chans=in_chans, pretrained=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()

    ds = UCLDistortionDataset(te_undist, modality=modality,
                              use_mask_crop=use_mask_crop, augment=False,
                              k_slices=k_slices, img_size=args.img_size,
                              return_case_id=True)

    # --- Sweep severities ---
    print(f"Model: {model_name} | severities: {args.severities}")
    rows = []
    # cache the clean tensors once (distortion is applied per severity)
    clean = []  # list of (case_id, tensor)
    for i in range(len(ds)):
        x, _, cid = ds[i]
        clean.append((cid, x))

    H = W = args.img_size
    with torch.no_grad():
        for sev in args.severities:
            probs = []
            for idx, (cid, x) in enumerate(clean):
                if sev > 0:
                    field = make_displacement_field(H, W, sev, seed=idx)
                    xd = warp_vertical(x, field)
                else:
                    xd = x
                xb = xd.unsqueeze(0).to(device)
                with amp_autocast(amp_enabled):
                    logit = model(xb)
                probs.append(float(torch.sigmoid(logit).cpu().reshape(-1)[0]))
            probs = np.array(probs)
            mean_p = float(probs.mean())
            det_rate = float((probs >= 0.5).mean())
            rows.append({"severity": sev, "mean_p_distorted": round(mean_p, 4),
                         "detection_rate": round(det_rate, 4),
                         "n_cases": len(probs)})
            print(f"  severity={sev:<5}  mean P(distorted)={mean_p:.4f}  "
                  f"detection_rate={det_rate:.4f}")

    resp = pd.DataFrame(rows)
    resp.to_csv(out / "distortion_response.csv", index=False)

    # --- Detection threshold: lowest severity where mean_p crosses 0.5 ---
    thr = None
    for _, r in resp.iterrows():
        if r["mean_p_distorted"] >= 0.5:
            thr = float(r["severity"])
            break

    summary = {
        "model": model_name,
        "fold": args.fold,
        "n_undistorted_cases": int(len(clean)),
        "severities": list(args.severities),
        "baseline_p_at_severity0": float(resp.iloc[0]["mean_p_distorted"]),
        "detection_threshold_pixels": thr,
        "response_curve": rows,
    }
    (out / "distortion_summary.json").write_text(json.dumps(summary, indent=2))

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(resp["severity"], resp["mean_p_distorted"], "o-", linewidth=2,
            markersize=6, color="tab:red", label="Mean P(distorted)")
    ax.plot(resp["severity"], resp["detection_rate"], "s--", linewidth=1.5,
            markersize=5, color="tab:blue", alpha=0.7, label="Detection rate")
    ax.axhline(y=0.5, color="grey", linestyle=":", alpha=0.6,
               label="Decision threshold")
    if thr is not None:
        ax.axvline(x=thr, color="green", linestyle="--", alpha=0.6,
                   label=f"Detection threshold: {thr} px")
    ax.set_xlabel("Injected distortion severity (max vertical displacement, px)")
    ax.set_ylabel("Model output")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Synthetic Distortion Response ({model_name})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "distortion_response.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nBaseline P(distorted) at severity 0: "
          f"{summary['baseline_p_at_severity0']:.4f}")
    print(f"Detection threshold: {thr} pixels"
          if thr else "Never crossed 0.5 within tested severities")
    print(f"Results in {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        _outdir = None
        for i, a in enumerate(sys.argv):
            if a == "--outdir" and i + 1 < len(sys.argv):
                _outdir = Path(sys.argv[i + 1])
                break
        if _outdir and _outdir.exists():
            (_outdir / "summary.json").write_text(json.dumps({
                "error": type(exc).__name__, "message": str(exc),
                "traceback": traceback.format_exc()}, indent=2))
        raise
