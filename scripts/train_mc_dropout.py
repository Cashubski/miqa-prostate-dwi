"""Binary classification training with explicit dropout for MC Dropout UQ.

Clone of train.py with 3 changes:
  1. Imports build_model_with_dropout from miqa.uncertainty
  2. Adds --dropout_rate argument
  3. Uses build_model_with_dropout() instead of build_model()

Checkpoint state dict keys differ from standard models (e.g. fc.0.weight
vs fc.weight for ResNet), so MC Dropout checkpoints are separate.
"""
import argparse, json, math, os, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (roc_auc_score, f1_score, balanced_accuracy_score,
                             average_precision_score)

from miqa.utils import (set_seed, print_hw_banner, load_or_create_splits,
                         amp_autocast, amp_grad_scaler)
from miqa.uncertainty import build_model_with_dropout
from miqa.datasets import UCLDistortionDataset
from miqa.metrics import evaluate
from miqa.schedulers import build_scheduler


def main():
    ap = argparse.ArgumentParser(
        description="Train binary classifier with MC Dropout for UQ")

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
    g.add_argument("--max_cases", type=int, default=0,
                   help="Limit total cases (0=all). For smoke tests.")

    # Model
    g = ap.add_argument_group("Model")
    g.add_argument("--model", type=str, default="resnet18",
                   choices=["resnet18", "resnet50", "densenet121",
                            "efficientnet_b0"])
    g.add_argument("--no_pretrained", action="store_true")
    g.add_argument("--dropout_rate", type=float, default=0.2,
                   help="Dropout rate for MC Dropout (default: 0.2)")

    # Training
    g = ap.add_argument_group("Training")
    g.add_argument("--epochs", type=int, default=30)
    g.add_argument("--batch", type=int, default=0,
                   help="Batch size (0=auto: r18->64, r50->32)")
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--wd", type=float, default=1e-4)
    g.add_argument("--grad_accum", type=int, default=1)
    g.add_argument("--clip_grad_norm", type=float, default=0.0,
                   help="Max grad norm (0=disabled)")
    g.add_argument("--no_aug", action="store_true")
    g.add_argument("--patience", type=int, default=7)
    g.add_argument("--scheduler", type=str, default="none",
                   choices=["none", "cosine"])
    g.add_argument("--warmup_epochs", type=int, default=5)
    g.add_argument("--eval_every", type=int, default=1,
                   help="Evaluate every N epochs")
    g.add_argument("--save_every", type=int, default=0,
                   help="Save checkpoint every N epochs (0=disabled)")

    # Hardware
    g = ap.add_argument_group("Hardware")
    g.add_argument("--device", type=str, default="cuda")
    g.add_argument("--gpu", type=int, default=0)
    g.add_argument("--amp", action="store_true", default=True)
    g.add_argument("--no_amp", dest="amp", action="store_false")
    g.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers (0=auto)")
    g.add_argument("--pin_memory", action="store_true", default=True)
    g.add_argument("--no_pin_memory", dest="pin_memory",
                   action="store_false")
    g.add_argument("--persistent_workers", action="store_true", default=True)
    g.add_argument("--no_persistent_workers", dest="persistent_workers",
                   action="store_false")
    g.add_argument("--prefetch_factor", type=int, default=4)

    # Reproducibility
    g = ap.add_argument_group("Reproducibility")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--split_seed", type=int, default=42)
    g.add_argument("--n_folds", type=int, default=0,
                   help="K-fold CV (0=disabled, use split_seed instead)")
    g.add_argument("--fold", type=int, default=0,
                   help="Which fold to run (0-indexed)")
    g.add_argument("--deterministic", action="store_true")

    # Resume
    ap.add_argument("--resume", type=str, default="",
                    help="Path to checkpoint to resume from")

    args = ap.parse_args()

    # --- Auto batch size (GPU-memory-aware) ---
    if args.batch == 0:
        _base_bs = {"resnet18": 64, "resnet50": 32,
                    "densenet121": 32, "efficientnet_b0": 48}
        args.batch = _base_bs.get(args.model, 32)

        if args.device == "cuda" and torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(
                args.gpu).total_memory / 1e9
            if vram_gb >= 30:
                args.batch *= 4
            elif vram_gb >= 20:
                args.batch *= 3
            elif vram_gb >= 14:
                args.batch *= 2
            print(f"GPU VRAM: {vram_gb:.0f} GB -> auto batch={args.batch}")

        if args.img_size >= 384:
            args.batch = max(8, args.batch // 2)

    # --- Auto num_workers ---
    if args.num_workers == 0:
        args.num_workers = min(8, max(1, os.cpu_count() // 2))

    # --- Device setup ---
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
        args.amp = False

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed, deterministic=args.deterministic)
    print_hw_banner(device, args.amp)

    # --- Save config ---
    config = vars(args).copy()
    config["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["run_dir"] = str(out)
    (out / "config.json").write_text(json.dumps(config, indent=2))

    # --- Data ---
    df = pd.read_csv(args.index_csv)
    group_col = "patient_id" if "patient_id" in df.columns else None
    if args.max_cases > 0:
        ids = sorted(df["case_id"].unique())
        rng = np.random.RandomState(args.split_seed)
        rng.shuffle(ids)
        ids = ids[:args.max_cases]
        df = df[df["case_id"].isin(set(ids))]
        print(f"[smoke] Limited to {len(df)} cases")

    if args.n_folds > 0:
        from miqa.utils import load_or_create_kfold_splits
        tr, va, te = load_or_create_kfold_splits(
            df, args.n_folds, args.fold, args.split_seed, out,
            group_col=group_col)
    else:
        tr, va, te = load_or_create_splits(df, args.split_seed, out,
                                           group_col=group_col)
    print(f"Split: train={len(tr)}, val={len(va)}, test={len(te)}")
    print(f"Train labels: {dict(tr['label'].value_counts().sort_index())}")

    ds_kwargs = dict(modality=args.modality, use_mask_crop=args.use_mask_crop,
                     k_slices=args.k_slices, img_size=args.img_size)
    train_ds = UCLDistortionDataset(tr, augment=(not args.no_aug), **ds_kwargs)
    val_ds = UCLDistortionDataset(va, augment=False, **ds_kwargs)
    test_ds = UCLDistortionDataset(te, augment=False, **ds_kwargs)

    dl_kwargs = dict(
        batch_size=args.batch,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    if args.num_workers > 0:
        dl_kwargs["persistent_workers"] = args.persistent_workers
        dl_kwargs["prefetch_factor"] = args.prefetch_factor

    train_dl = torch.utils.data.DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_dl = torch.utils.data.DataLoader(val_ds, shuffle=False, **dl_kwargs)
    test_dl = torch.utils.data.DataLoader(test_ds, shuffle=False, **dl_kwargs)

    # --- Model (with dropout) ---
    in_chans = args.k_slices * 2 if args.modality == "fusion" else args.k_slices
    pretrained = not args.no_pretrained
    model = build_model_with_dropout(
        args.model, in_chans=in_chans, pretrained=pretrained,
        dropout_rate=args.dropout_rate,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model} (MC Dropout p={args.dropout_rate}), "
          f"params={n_params:,}, trainable={n_trainable:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd)

    # Weighted loss
    n_neg = int((tr["label"] == 0).sum())
    n_pos = int((tr["label"] == 1).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)],
                              dtype=torch.float32).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"pos_weight: {pos_weight.item():.2f}  (neg={n_neg}, pos={n_pos})")

    scaler = amp_grad_scaler(enabled=args.amp)

    # --- Scheduler ---
    steps_per_epoch = len(train_dl)
    scheduler = build_scheduler(opt, args.scheduler, args.epochs,
                                args.warmup_epochs, steps_per_epoch)
    effective_patience = 10**9 if args.scheduler != "none" else args.patience

    start_epoch = 1

    # --- Resume ---
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and args.amp:
            scaler.load_state_dict(ckpt["scaler"])
        if "scheduler" in ckpt and scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    best_auroc_val = {"auroc": -1.0}
    best_bal_acc_val = {"bal_acc": -1.0}
    best_epoch = 0
    patience_counter = 0
    logf = (out / "metrics.jsonl").open("a")

    eff_batch = args.batch * args.grad_accum
    print(f"Training: {args.epochs} epochs, batch={args.batch}, "
          f"grad_accum={args.grad_accum}, eff_batch={eff_batch}, "
          f"lr={args.lr}, wd={args.wd}")

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_t0 = time.time()
        model.train()
        running_loss = 0.0
        all_logits, all_labels = [], []
        grad_norms = []
        opt.zero_grad(set_to_none=True)

        for step, (x, y) in enumerate(train_dl):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with amp_autocast(args.amp):
                logit = model(x)
                loss = loss_fn(logit, y) / args.grad_accum

            # NaN guard
            if math.isnan(loss.item()):
                print(f"NaN loss at epoch {epoch} step {step}, aborting")
                logf.close()
                meta = {"error": "nan_loss", "epoch": epoch, "step": step}
                (out / "summary.json").write_text(json.dumps(meta, indent=2))
                return

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_dl):
                scaler.unscale_(opt)
                total_norm = nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.clip_grad_norm if args.clip_grad_norm > 0 else float('inf'))
                grad_norms.append(float(total_norm))
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()

            running_loss += float(loss.item()) * args.grad_accum * x.size(0)
            all_logits.append(logit.detach())
            all_labels.append(y.detach())

        train_loss = running_loss / len(train_ds)
        current_lr = opt.param_groups[0]["lr"]

        # Compute train-set metrics
        _logits = torch.cat(all_logits).cpu()
        _labels = torch.cat(all_labels).cpu()
        _probs = torch.sigmoid(_logits).numpy()
        _preds = (_probs >= 0.5).astype(int)
        _y = _labels.numpy()
        try:
            train_auroc = round(float(roc_auc_score(_y, _probs)), 4)
            train_auprc = round(float(average_precision_score(_y, _probs)), 4)
        except ValueError:
            train_auroc = 0.0
            train_auprc = 0.0
        train_f1 = round(float(f1_score(_y, _preds, zero_division=0)), 4)
        train_bal_acc = round(float(balanced_accuracy_score(_y, _preds)), 4)
        _tp = int(((_preds == 1) & (_y == 1)).sum())
        _tn = int(((_preds == 0) & (_y == 0)).sum())
        _fp = int(((_preds == 1) & (_y == 0)).sum())
        _fn = int(((_preds == 0) & (_y == 1)).sum())
        train_sens = round(_tp / (_tp + _fn) if (_tp + _fn) > 0 else 0.0, 4)
        train_spec = round(_tn / (_tn + _fp) if (_tn + _fp) > 0 else 0.0, 4)

        avg_grad_norm = round(float(np.mean(grad_norms)), 4) if grad_norms else 0.0
        max_grad_norm = round(float(np.max(grad_norms)), 4) if grad_norms else 0.0

        epoch_secs = round(time.time() - epoch_t0, 2)

        gpu_mem_mb = None
        if device.type == "cuda":
            gpu_mem_mb = round(
                torch.cuda.max_memory_allocated(device) / 1e6, 1)

        # --- Evaluation ---
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_metrics = evaluate(model, val_dl, device,
                                   amp_enabled=args.amp, loss_fn=loss_fn)
            rec = {
                "epoch": epoch,
                "epoch_secs": epoch_secs,
                "train_loss": round(train_loss, 5),
                "train_auroc": train_auroc,
                "train_auprc": train_auprc,
                "train_f1": train_f1,
                "train_bal_acc": train_bal_acc,
                "train_sensitivity": train_sens,
                "train_specificity": train_spec,
                "train_tp": _tp, "train_tn": _tn,
                "train_fp": _fp, "train_fn": _fn,
                "lr": round(current_lr, 8),
                "grad_norm_avg": avg_grad_norm,
                "grad_norm_max": max_grad_norm,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            if gpu_mem_mb is not None:
                rec["gpu_mem_mb"] = gpu_mem_mb
            print(rec)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()

            ckpt_payload = {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "args": config,
            }
            if scheduler is not None:
                ckpt_payload["scheduler"] = scheduler.state_dict()

            improved = False
            if val_metrics["auroc"] > best_auroc_val["auroc"]:
                best_auroc_val = val_metrics.copy()
                best_epoch = epoch
                torch.save(ckpt_payload, out / "best_auroc.pt")
                improved = True

            if val_metrics["bal_acc"] > best_bal_acc_val["bal_acc"]:
                best_bal_acc_val = val_metrics.copy()
                torch.save(ckpt_payload, out / "best_bal_acc.pt")
                improved = True

            if improved:
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= effective_patience:
                    print(f"Early stopping at epoch {epoch} "
                          f"(patience={effective_patience})")
                    break
        else:
            rec = {
                "epoch": epoch,
                "epoch_secs": epoch_secs,
                "train_loss": round(train_loss, 5),
                "train_auroc": train_auroc,
                "train_auprc": train_auprc,
                "train_f1": train_f1,
                "train_bal_acc": train_bal_acc,
                "train_sensitivity": train_sens,
                "train_specificity": train_spec,
                "train_tp": _tp, "train_tn": _tn,
                "train_fp": _fp, "train_fn": _fn,
                "lr": round(current_lr, 8),
                "grad_norm_avg": avg_grad_norm,
                "grad_norm_max": max_grad_norm,
            }
            if gpu_mem_mb is not None:
                rec["gpu_mem_mb"] = gpu_mem_mb
            print(rec)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()

        # Periodic save
        if args.save_every > 0 and epoch % args.save_every == 0:
            _periodic = {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "args": config,
            }
            if scheduler is not None:
                _periodic["scheduler"] = scheduler.state_dict()
            torch.save(_periodic, out / f"epoch_{epoch:03d}.pt")

    logf.close()

    # --- Save last checkpoint ---
    _last = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "args": config,
    }
    if scheduler is not None:
        _last["scheduler"] = scheduler.state_dict()
    torch.save(_last, out / "last.pt")

    # --- Final test on best AUROC checkpoint ---
    best_path = out / "best_auroc.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model.to(device)
    test_metrics = evaluate(model, test_dl, device, amp_enabled=args.amp,
                            loss_fn=loss_fn)

    # Save per-sample test predictions for cross-fold aggregation
    model.eval()
    test_pred_ds = UCLDistortionDataset(te, augment=False, return_case_id=True,
                                         **ds_kwargs)
    test_pred_dl = torch.utils.data.DataLoader(
        test_pred_ds, batch_size=args.batch, shuffle=False)
    all_ids, all_true, all_prob = [], [], []
    with torch.no_grad():
        for x, y, cid in test_pred_dl:
            logits = model(x.to(device)).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            all_ids.extend(cid)
            all_true.extend(y.numpy().ravel())
            all_prob.extend(probs)
    pd.DataFrame({
        "case_id": all_ids,
        "y_true": np.array(all_true, dtype=int),
        "y_prob": np.round(all_prob, 6),
    }).to_csv(out / "test_predictions.csv", index=False)

    elapsed = round(time.time() - t0, 2)
    gpu_mem_peak_mb = None
    if device.type == "cuda":
        gpu_mem_peak_mb = round(
            torch.cuda.max_memory_allocated(device) / 1e6, 1)

    meta = {
        "run_dir": str(out),
        "device": str(device),
        "gpu0": (torch.cuda.get_device_name(args.gpu)
                 if torch.cuda.is_available() else None),
        "torch": torch.__version__,
        "secs": elapsed,
        "n_params": n_params,
        "n_trainable": n_trainable,
        "arch": args.model,
        "dropout_rate": args.dropout_rate,
        "img_size": args.img_size,
        "k_slices": args.k_slices,
        "modality": args.modality,
        "use_mask_crop": args.use_mask_crop,
        "lr": args.lr,
        "wd": args.wd,
        "batch": args.batch,
        "epochs": args.epochs,
        "scheduler": args.scheduler,
        "aug": not args.no_aug,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "n_train": len(tr),
        "n_val": len(va),
        "n_test": len(te),
        "gpu_mem_peak_mb": gpu_mem_peak_mb,
        "best_epoch": best_epoch,
        "best_val": best_auroc_val,
        "test": test_metrics,
        "ckpt_best_auroc": str(best_path),
        "ckpt_last": str(out / "last.pt"),
    }
    (out / "summary.json").write_text(json.dumps(meta, indent=2))
    print("Summary:", json.dumps(meta, indent=2))


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
            err_meta = {
                "error": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            (_outdir / "summary.json").write_text(
                json.dumps(err_meta, indent=2))
        raise
