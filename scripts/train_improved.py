"""Improved training recipe: adds calibration and regularisation on top of the
standard train.py pipeline.

Additions over train.py:
  - Focal loss (gamma=2 by default) instead of plain BCE: down-weights easy
    examples and tends to calibrate better than BCE.
  - Label smoothing (eps=0.1 by default): prevents weight saturation and
    improves calibration (addresses the DenseNet overconfidence seen in the
    UQ chapter).
  - Stochastic Weight Averaging (SWA): averages weights over the last N epochs
    to find flatter loss minima; improves generalization and calibration.
  - MixUp augmentation (alpha=0.2 by default): interpolates input/label pairs
    as implicit regularisation on a small dataset.

Checkpoints use the same format as train.py, so downstream tools (ensemble UQ,
calibration analysis) work unchanged. After SWA finishes, the reported "best"
checkpoint is the SWA-averaged model.

Usage:
    python repo/scripts/train_improved.py \
        --index_csv repo/dataset/index_full.csv \
        --outdir runs/improved_resnet18_f0 \
        --model resnet18 --n_folds 5 --fold 0 \
        --epochs 100 --swa_start 60 --focal_gamma 2.0 \
        --label_smoothing 0.1 --mixup_alpha 0.2
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score

from miqa.datasets import UCLDistortionDataset
from miqa.metrics import evaluate
from miqa.models import build_model
from miqa.utils import (set_seed, print_hw_banner, load_or_create_kfold_splits,
                        amp_autocast, amp_grad_scaler)


# ---------------------------------------------------------------------------
# Loss + augmentation helpers
# ---------------------------------------------------------------------------

class FocalLossWithSmoothing(nn.Module):
    """Binary focal loss with label smoothing and optional positive weighting.

    Standard BCE:   -[y log p + (1-y) log(1-p)]
    Focal:          BCE * (1-pt)^gamma   where pt = p if y=1 else 1-p
    Label smooth:   y' = y * (1-eps) + 0.5 * eps  (binary symmetric form)
    """

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0,
                 pos_weight: torch.Tensor = None):
        super().__init__()
        self.gamma = float(gamma)
        self.eps = float(label_smoothing)
        if pos_weight is not None and not torch.is_tensor(pos_weight):
            pos_weight = torch.tensor(float(pos_weight))
        self.register_buffer("pos_weight",
                             pos_weight if pos_weight is not None
                             else torch.tensor(1.0))

    def forward(self, logits, targets):
        # Smooth labels (binary symmetric: 0 to eps/2, 1 to 1-eps/2)
        if self.eps > 0:
            targets = targets * (1.0 - self.eps) + 0.5 * self.eps
        # Per-element BCE with pos_weight on positives
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight)
        # Focal weighting using unsmoothed-ish pt approximation
        with torch.no_grad():
            p = torch.sigmoid(logits)
            pt = p * targets + (1.0 - p) * (1.0 - targets)
            focal_w = (1.0 - pt).clamp_(min=0.0, max=1.0).pow(self.gamma)
        return (focal_w * bce).mean()


def mixup_batch(x, y, alpha: float):
    """MixUp on a batched (B, C, H, W) tensor + (B, 1) labels.

    Returns mixed (x, y); set alpha<=0 to disable.
    """
    if alpha <= 0.0:
        return x, y
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    x = lam * x + (1.0 - lam) * x[idx]
    y = lam * y + (1.0 - lam) * y[idx]
    return x, y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Improved training: SWA + focal + label smoothing + MixUp")
    # Data
    ap.add_argument("--index_csv", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--modality", type=str, default="dwi",
                    choices=["dwi", "t2", "fusion"])
    ap.add_argument("--use_mask_crop", action="store_true", default=True)
    ap.add_argument("--no_mask_crop", dest="use_mask_crop",
                    action="store_false")
    ap.add_argument("--k_slices", type=int, default=3)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--max_cases", type=int, default=0)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--fold", type=int, default=0)
    # Model
    ap.add_argument("--model", type=str, default="resnet18",
                    choices=["resnet18", "resnet50", "densenet121",
                             "efficientnet_b0"])
    ap.add_argument("--no_pretrained", action="store_true")
    # Training (standard knobs)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10**9,
                    help="Early stop patience (default disabled, let SWA run)")
    # Improvements (the experimental knobs)
    ap.add_argument("--focal_gamma", type=float, default=2.0,
                    help="Focal loss focusing parameter (0 = plain BCE)")
    ap.add_argument("--label_smoothing", type=float, default=0.1,
                    help="Binary label smoothing epsilon (0 = no smoothing)")
    ap.add_argument("--mixup_alpha", type=float, default=0.2,
                    help="MixUp Beta(alpha,alpha) parameter (0 = no MixUp)")
    ap.add_argument("--swa_start", type=int, default=60,
                    help="Epoch to begin SWA averaging (set > epochs to disable)")
    ap.add_argument("--swa_lr", type=float, default=1e-4,
                    help="Constant LR used during SWA phase")
    # Hardware
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no_amp", dest="amp", action="store_false")
    ap.add_argument("--num_workers", type=int, default=0)
    # Reproducibility
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split_seed", type=int, default=42)

    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    # Device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
        args.amp = False
    print_hw_banner(device, args.amp)

    # Splits
    df = pd.read_csv(args.index_csv)
    group_col = "patient_id" if "patient_id" in df.columns else None
    tr, va, te = load_or_create_kfold_splits(
        df, args.n_folds, args.fold, args.split_seed, out,
        group_col=group_col)
    if args.max_cases > 0:
        tr = tr.iloc[: args.max_cases]
        va = va.iloc[: max(8, args.max_cases // 3)]
        te = te.iloc[: max(8, args.max_cases // 3)]
    print(f"Split: train={len(tr)}, val={len(va)}, test={len(te)}")
    n_pos = int((tr["label"] == 1).sum())
    n_neg = int((tr["label"] == 0).sum())
    pos_weight = (n_neg / max(1, n_pos)) if n_pos > 0 else 1.0
    print(f"pos_weight: {pos_weight:.2f}  (neg={n_neg}, pos={n_pos})")

    # Data
    ds_kwargs = dict(modality=args.modality, use_mask_crop=args.use_mask_crop,
                     k_slices=args.k_slices, img_size=args.img_size)
    tr_ds = UCLDistortionDataset(tr, augment=True, **ds_kwargs)
    va_ds = UCLDistortionDataset(va, augment=False, **ds_kwargs)
    te_ds = UCLDistortionDataset(te, augment=False, **ds_kwargs)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                       num_workers=args.num_workers, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch, shuffle=False,
                       num_workers=args.num_workers)
    te_dl = DataLoader(te_ds, batch_size=args.batch, shuffle=False,
                       num_workers=args.num_workers)

    # Model
    in_chans = args.k_slices * 2 if args.modality == "fusion" else args.k_slices
    model = build_model(args.model, in_chans=in_chans,
                        pretrained=not args.no_pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}, params={n_params:,}")

    # SWA model (created early; updated only after swa_start)
    swa_model = AveragedModel(model)

    # Loss + optimizer
    loss_fn = FocalLossWithSmoothing(
        gamma=args.focal_gamma, label_smoothing=args.label_smoothing,
        pos_weight=torch.tensor(pos_weight, device=device),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd)
    swa_scheduler = SWALR(opt, swa_lr=args.swa_lr, anneal_epochs=5,
                          anneal_strategy="linear")
    scaler = amp_grad_scaler(enabled=args.amp)

    # Config snapshot
    config = vars(args).copy()
    config["task"] = "improved_train"
    config["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["n_params"] = n_params
    config["pos_weight"] = pos_weight
    (out / "config.json").write_text(json.dumps(config, indent=2))
    logf = (out / "metrics.jsonl").open("a")

    # Training loop
    best_val = {"epoch": -1, "auroc": 0.0}
    t0 = time.time()
    n_iters_per_epoch = max(1, len(tr_dl))
    print(f"Training: {args.epochs} epochs, batch={args.batch}, "
          f"focal_gamma={args.focal_gamma}, label_smoothing={args.label_smoothing}, "
          f"mixup_alpha={args.mixup_alpha}, swa_start={args.swa_start}, "
          f"swa_lr={args.swa_lr}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = 0.0
        ep_n = 0
        for batch in tr_dl:
            x = batch[0].to(device, non_blocking=True)
            y = batch[1].to(device, non_blocking=True).float()
            if y.ndim == 1:
                y = y.unsqueeze(-1)
            x, y = mixup_batch(x, y, args.mixup_alpha)
            opt.zero_grad(set_to_none=True)
            with amp_autocast(args.amp):
                logit = model(x)
                if logit.ndim == 1:
                    logit = logit.unsqueeze(-1)
                loss = loss_fn(logit, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ep_loss += float(loss.detach()) * x.size(0)
            ep_n += x.size(0)
        train_loss = ep_loss / max(1, ep_n)

        # After swa_start: maintain running average + SWA LR
        if epoch >= args.swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        # Validate
        val_metrics = evaluate(model, va_dl, device, amp_enabled=args.amp)
        record = {"epoch": epoch, "train_loss": round(train_loss, 5),
                  "val_auroc": val_metrics.get("auroc"),
                  "val_bal_acc": val_metrics.get("bal_acc"),
                  "val_f1": val_metrics.get("f1"),
                  "lr": opt.param_groups[0]["lr"]}
        logf.write(json.dumps(record) + "\n"); logf.flush()
        if val_metrics.get("auroc", 0) > best_val["auroc"]:
            best_val = {"epoch": epoch, **val_metrics}
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "epoch": epoch, "val": val_metrics},
                       out / "best_auroc.pt")
        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  loss={train_loss:.4f}  "
                  f"val_auroc={val_metrics.get('auroc', 0):.4f}")

    # SWA finalisation: update BN stats then evaluate
    eval_model = model
    swa_active = args.swa_start <= args.epochs
    if swa_active:
        print(f"Updating BN stats on SWA-averaged model...")
        swa_model = swa_model.to(device)
        update_bn(tr_dl, swa_model, device=device)
        eval_model = swa_model
        swa_val = evaluate(swa_model, va_dl, device, amp_enabled=args.amp)
        print(f"SWA validation AUROC: {swa_val.get('auroc', 0):.4f}  "
              f"(vs non-SWA best {best_val['auroc']:.4f})")
        # If SWA is better, treat its checkpoint as the reported best
        if swa_val.get("auroc", 0) >= best_val["auroc"]:
            torch.save({"model": swa_model.module.state_dict(),
                        "args": vars(args), "epoch": args.epochs,
                        "val": swa_val, "swa": True},
                       out / "best_auroc.pt")
            best_val = {"epoch": args.epochs, **swa_val, "swa": True}

    logf.close()

    # Test with the chosen eval model (SWA if better else best-by-AUROC)
    print("Loading reported best checkpoint for test evaluation...")
    ckpt = torch.load(out / "best_auroc.pt", map_location=device,
                      weights_only=False)
    test_model = build_model(args.model, in_chans=in_chans, pretrained=False)
    test_model.load_state_dict(ckpt["model"])
    test_model.to(device).eval()

    test_metrics = evaluate(test_model, te_dl, device, amp_enabled=args.amp)

    # Save test predictions (same schema as train.py)
    te_pred_ds = UCLDistortionDataset(te, augment=False, return_case_id=True,
                                       **ds_kwargs)
    te_pred_dl = DataLoader(te_pred_ds, batch_size=args.batch, shuffle=False)
    all_ids, all_true, all_prob = [], [], []
    with torch.no_grad():
        for x, y, cid in te_pred_dl:
            with amp_autocast(args.amp):
                logit = test_model(x.to(device))
            if logit.ndim == 1:
                logit = logit.unsqueeze(-1)
            probs = torch.sigmoid(logit).cpu().numpy().ravel()
            all_ids.extend(cid)
            all_true.extend(y.numpy().ravel())
            all_prob.extend(probs)
    pd.DataFrame({
        "case_id": all_ids,
        "y_true": np.array(all_true, dtype=int),
        "y_prob": np.round(all_prob, 6),
    }).to_csv(out / "test_predictions.csv", index=False)

    elapsed = round(time.time() - t0, 2)
    gpu_mem = (round(torch.cuda.max_memory_allocated(device) / 1e6, 1)
               if device.type == "cuda" else None)
    summary = {
        "run_dir": str(out),
        "task": "improved_train",
        "device": str(device),
        "torch": torch.__version__,
        "secs": elapsed,
        "n_params": n_params,
        "arch": args.model,
        "img_size": args.img_size,
        "k_slices": args.k_slices,
        "modality": args.modality,
        "lr": args.lr,
        "wd": args.wd,
        "batch": args.batch,
        "epochs": args.epochs,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "focal_gamma": args.focal_gamma,
        "label_smoothing": args.label_smoothing,
        "mixup_alpha": args.mixup_alpha,
        "swa_start": args.swa_start,
        "swa_lr": args.swa_lr,
        "swa_used": bool(swa_active),
        "gpu_mem_peak_mb": gpu_mem,
        "best_epoch": int(best_val.get("epoch", -1)),
        "best_val": {k: v for k, v in best_val.items() if k != "epoch"},
        "test": test_metrics,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Summary:", json.dumps(summary, indent=2))


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
