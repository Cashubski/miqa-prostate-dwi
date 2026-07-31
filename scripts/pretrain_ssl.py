"""Self-supervised pretraining on unlabeled 2.5D medical image slices.

Supports SimCLR and BYOL.  Saves backbone weights for downstream
fine-tuning via scripts/train_ssl_finetune.py.

Usage:
    python scripts/pretrain_ssl.py --method simclr --model resnet18 \
        --index_csv data/index.csv --outdir runs/ssl_simclr_r18
"""
import argparse, json, math, os, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from miqa.utils import (set_seed, print_hw_banner,
                         amp_autocast, amp_grad_scaler)
from miqa.ssl import (SimCLRModel, BYOLModel, NTXentLoss, byol_loss,
                       SSLSliceDataset)
from miqa.schedulers import build_scheduler


def main():
    ap = argparse.ArgumentParser(
        description="Self-supervised pretraining for medical image slices")

    # Data
    g = ap.add_argument_group("Data")
    g.add_argument("--index_csv", type=str, required=True)
    g.add_argument("--outdir", type=str, required=True)
    g.add_argument("--modality", type=str, default="dwi",
                   choices=["dwi", "t2", "fusion"])
    g.add_argument("--use_mask_crop", action="store_true", default=True)
    g.add_argument("--no_mask_crop", dest="use_mask_crop",
                   action="store_false")
    g.add_argument("--k_slices", type=int, default=3)
    g.add_argument("--img_size", type=int, default=224)
    g.add_argument("--slices_per_case", type=int, default=10,
                   help="Slices extracted per volume per epoch")
    g.add_argument("--max_cases", type=int, default=0)

    # SSL method
    g = ap.add_argument_group("SSL")
    g.add_argument("--method", type=str, default="simclr",
                   choices=["simclr", "byol"])
    g.add_argument("--temperature", type=float, default=0.5,
                   help="SimCLR temperature")
    g.add_argument("--proj_dim", type=int, default=128,
                   help="Projection head output dim")
    g.add_argument("--proj_hidden", type=int, default=2048,
                   help="Projection head hidden dim")
    g.add_argument("--ema_decay", type=float, default=0.996,
                   help="BYOL target EMA decay")
    g.add_argument("--aug_strength", type=str, default="strong",
                   choices=["strong", "weak"])

    # Model
    g = ap.add_argument_group("Model")
    g.add_argument("--model", type=str, default="resnet18",
                   choices=["resnet18", "resnet50", "densenet121",
                            "efficientnet_b0"])
    g.add_argument("--imagenet_init", action="store_true",
                   help="Initialize backbone from ImageNet before SSL")

    # Training
    g = ap.add_argument_group("Training")
    g.add_argument("--epochs", type=int, default=200)
    g.add_argument("--batch", type=int, default=0)
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--wd", type=float, default=1e-4)
    g.add_argument("--scheduler", type=str, default="cosine",
                   choices=["none", "cosine"])
    g.add_argument("--warmup_epochs", type=int, default=10)
    g.add_argument("--save_every", type=int, default=50)

    # Hardware
    g = ap.add_argument_group("Hardware")
    g.add_argument("--device", type=str, default="cuda")
    g.add_argument("--gpu", type=int, default=0)
    g.add_argument("--amp", action="store_true", default=True)
    g.add_argument("--no_amp", dest="amp", action="store_false")
    g.add_argument("--num_workers", type=int, default=0)
    g.add_argument("--pin_memory", action="store_true", default=True)
    g.add_argument("--no_pin_memory", dest="pin_memory",
                   action="store_false")

    # Reproducibility
    g = ap.add_argument_group("Reproducibility")
    g.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    # --- Auto batch size ---
    if args.batch == 0:
        _base = {"resnet18": 128, "resnet50": 64,
                 "densenet121": 64, "efficientnet_b0": 96}
        args.batch = _base.get(args.model, 64)
        if args.device == "cuda" and torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(
                args.gpu).total_memory / 1e9
            if vram_gb >= 30:
                args.batch *= 2
            print(f"GPU VRAM: {vram_gb:.0f} GB -> auto batch={args.batch}")
        if args.img_size >= 384:
            args.batch = max(16, args.batch // 2)

    # --- Auto workers ---
    if args.num_workers == 0:
        args.num_workers = min(8, max(1, os.cpu_count() // 2))

    # --- Device ---
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
        args.amp = False

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    print_hw_banner(device, args.amp)

    # --- Config ---
    config = vars(args).copy()
    config["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["task"] = f"ssl_{args.method}"
    (out / "config.json").write_text(json.dumps(config, indent=2))

    # --- Data (all cases, no train/test split for SSL) ---
    df = pd.read_csv(args.index_csv)
    if args.max_cases > 0:
        ids = sorted(df["case_id"].unique())[:args.max_cases]
        df = df[df["case_id"].isin(set(ids))]

    ssl_ds = SSLSliceDataset(
        df, modality=args.modality, use_mask_crop=args.use_mask_crop,
        k_slices=args.k_slices, img_size=args.img_size,
        slices_per_case=args.slices_per_case,
        aug_strength=args.aug_strength)

    dl_kwargs = dict(batch_size=args.batch, shuffle=True,
                     num_workers=args.num_workers,
                     pin_memory=args.pin_memory, drop_last=True)
    if args.num_workers > 0:
        dl_kwargs["persistent_workers"] = True
    ssl_dl = torch.utils.data.DataLoader(ssl_ds, **dl_kwargs)

    print(f"SSL dataset: {len(ssl_ds)} samples/epoch "
          f"({len(df)} cases x {args.slices_per_case} slices)")

    # --- Model ---
    in_chans = (args.k_slices * 2 if args.modality == "fusion"
                else args.k_slices)

    if args.method == "simclr":
        model = SimCLRModel(
            backbone_name=args.model, in_chans=in_chans,
            pretrained=args.imagenet_init,
            proj_hidden=args.proj_hidden, proj_dim=args.proj_dim,
        ).to(device)
        criterion = NTXentLoss(temperature=args.temperature)
    else:  # byol
        model = BYOLModel(
            backbone_name=args.model, in_chans=in_chans,
            pretrained=args.imagenet_init,
            proj_hidden=args.proj_hidden, proj_dim=args.proj_dim,
            ema_decay=args.ema_decay,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters()
                      if p.requires_grad)
    print(f"Model: {args.model} ({args.method}), "
          f"params={n_params:,}, trainable={n_trainable:,}")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.wd)

    scaler = amp_grad_scaler(enabled=args.amp)

    steps_per_epoch = len(ssl_dl)
    scheduler = build_scheduler(opt, args.scheduler, args.epochs,
                                args.warmup_epochs, steps_per_epoch)

    logf = (out / "metrics.jsonl").open("a")

    print(f"Training: {args.epochs} epochs, batch={args.batch}, "
          f"lr={args.lr}, method={args.method}")

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.time()
        model.train()
        running_loss = 0.0
        n_batches = 0

        for view1, view2 in ssl_dl:
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)

            with amp_autocast(args.amp):
                if args.method == "simclr":
                    _, z1 = model(view1)
                    _, z2 = model(view2)
                    loss = criterion(z1, z2)
                else:
                    p1, tz2, p2, tz1 = model(view1, view2)
                    loss = byol_loss(p1, tz2, p2, tz1)

            # NaN guard
            if math.isnan(loss.item()):
                print(f"NaN loss at epoch {epoch}, aborting")
                logf.close()
                meta = {"error": "nan_loss", "epoch": epoch}
                (out / "summary.json").write_text(json.dumps(meta, indent=2))
                return

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if scheduler is not None:
                scheduler.step()

            if args.method == "byol":
                model.update_target()

            running_loss += float(loss.item())
            n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)
        epoch_secs = round(time.time() - epoch_t0, 2)
        current_lr = opt.param_groups[0]["lr"]

        gpu_mem_mb = None
        if device.type == "cuda":
            gpu_mem_mb = round(
                torch.cuda.max_memory_allocated(device) / 1e6, 1)

        rec = {"epoch": epoch, "epoch_secs": epoch_secs,
               "loss": round(avg_loss, 5), "lr": round(current_lr, 8)}
        if gpu_mem_mb is not None:
            rec["gpu_mem_mb"] = gpu_mem_mb
        print(rec)
        logf.write(json.dumps(rec) + "\n")
        logf.flush()

        # Periodic save
        if args.save_every > 0 and epoch % args.save_every == 0:
            ckpt = {"model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch, "args": config}
            if scheduler is not None:
                ckpt["scheduler"] = scheduler.state_dict()
            torch.save(ckpt, out / f"epoch_{epoch:04d}.pt")

    logf.close()

    # --- Final checkpoint ---
    final = {"model": model.state_dict(),
             "optimizer": opt.state_dict(),
             "scaler": scaler.state_dict(),
             "epoch": args.epochs, "args": config}
    if scheduler is not None:
        final["scheduler"] = scheduler.state_dict()
    torch.save(final, out / "last.pt")

    elapsed = round(time.time() - t0, 2)
    gpu_peak = None
    if device.type == "cuda":
        gpu_peak = round(
            torch.cuda.max_memory_allocated(device) / 1e6, 1)

    meta = {
        "task": f"ssl_{args.method}",
        "method": args.method,
        "arch": args.model,
        "imagenet_init": args.imagenet_init,
        "in_chans": in_chans,
        "img_size": args.img_size,
        "k_slices": args.k_slices,
        "modality": args.modality,
        "n_cases": len(df),
        "slices_per_case": args.slices_per_case,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "secs": elapsed,
        "n_params": n_params,
        "gpu_mem_peak_mb": gpu_peak,
        "ckpt_last": str(out / "last.pt"),
    }
    (out / "summary.json").write_text(json.dumps(meta, indent=2))
    print("Summary:", json.dumps(meta, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import sys, traceback
        _outdir = None
        for i, a in enumerate(sys.argv):
            if a == "--outdir" and i + 1 < len(sys.argv):
                _outdir = Path(sys.argv[i + 1])
                break
        if _outdir and _outdir.exists():
            err_meta = {
                "error": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            (_outdir / "summary.json").write_text(
                json.dumps(err_meta, indent=2))
        raise
