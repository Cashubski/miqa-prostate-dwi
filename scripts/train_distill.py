"""Knowledge distillation: heterogeneous 9-model ensemble into a single ResNet-18.

The heterogeneous ensemble (DenseNet-121 + ResNet-18 + ResNet-50, 3 seeds each
= 9 models) has the best uncertainty quality (UE-AUROC 0.685, selective
prediction acc@70% = 0.836) but needs 9 forward passes per case at inference.
Distilling it into a single ResNet-18 student gives ensemble-quality
predictions at single-model inference cost.

Distillation loss:
  L = (1 - alpha) * BCE(student_logit, true_label)
      + alpha * KL(sigmoid(student_logit) || teacher_mean_prob)
      + beta * MSE(student_uncertainty, teacher_std)

  teacher_mean_prob and teacher_std come from running all 9 teacher models on
  each training case. Re-uses the existing poster100 checkpoints; teachers are
  not retrained.

Per fold:
  1. Load 9 teacher checkpoints (3 archs x 3 seeds).
  2. Run inference on the fold's train + val + test sets to get per-case
     teacher_mean and teacher_std. Cache to teacher_preds.csv.
  3. Train ResNet-18 student with the combined loss.
  4. Evaluate student; compare to teacher ensemble on the same test split.

Usage:
    python repo/scripts/train_distill.py \
        --index_csv repo/dataset/index_full.csv \
        --outdir runs/distill_resnet18_f0 \
        --n_folds 5 --fold 0 --epochs 100 --kd_alpha 0.7 --kd_beta 0.5
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from miqa.datasets import UCLDistortionDataset
from miqa.metrics import evaluate
from miqa.models import build_model
from miqa.utils import (set_seed, print_hw_banner, load_or_create_kfold_splits,
                        amp_autocast, amp_grad_scaler)


TEACHERS = [
    # (arch, dir_pattern_for_fold)
    ("densenet121", "poster100_densenet121_dwi_roi_f{f}"),
    ("densenet121", "poster100_densenet121_dwi_roi_s43_f{f}"),
    ("densenet121", "poster100_densenet121_dwi_roi_s44_f{f}"),
    ("resnet18",   "poster100_resnet18_dwi_roi_f{f}"),
    ("resnet18",   "poster100_resnet18_dwi_roi_s43_f{f}"),
    ("resnet18",   "poster100_resnet18_dwi_roi_s44_f{f}"),
    ("resnet50",   "poster100_resnet50_dwi_roi_f{f}"),
    ("resnet50",   "poster100_resnet50_dwi_roi_s43_f{f}"),
    ("resnet50",   "poster100_resnet50_dwi_roi_s44_f{f}"),
]
EPS = 1e-7


@torch.no_grad()
def get_teacher_predictions(runs_root, fold, df_split, ds_kwargs, batch,
                            device, amp_enabled):
    """Run each of the 9 teacher models on df_split, return aggregated
    mean and std per case (in the order they appear in df_split)."""
    ds = UCLDistortionDataset(df_split, augment=False, return_case_id=True,
                              **ds_kwargs)
    dl = DataLoader(ds, batch_size=batch, shuffle=False)
    # Collect labels and case_ids once
    case_ids, y_true = [], []
    for batchd in dl:
        case_ids.extend(batchd[2])
        y_true.append(batchd[1].numpy().reshape(-1))
    y_true = np.concatenate(y_true)

    all_probs = []  # one (N,) array per teacher
    in_chans = ds_kwargs["k_slices"] * (
        2 if ds_kwargs["modality"] == "fusion" else 1)

    for i, (arch, pat) in enumerate(TEACHERS):
        ckpt_dir = runs_root / pat.format(f=fold)
        ckpt_path = ckpt_dir / "best_auroc.pt"
        if not ckpt_path.exists():
            print(f"  [WARN] missing teacher {i+1}/9: {ckpt_path}")
            continue
        m = build_model(arch, in_chans=in_chans, pretrained=False)
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        m.load_state_dict(ck["model"])
        m.to(device).eval()
        probs = []
        for batchd in dl:
            x = batchd[0].to(device)
            with amp_autocast(amp_enabled):
                logit = m(x)
            if logit.ndim == 1:
                logit = logit.unsqueeze(-1)
            probs.append(torch.sigmoid(logit).cpu().numpy().ravel())
        all_probs.append(np.concatenate(probs))
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  teacher {i+1}/9 ({arch}) done")

    if not all_probs:
        raise RuntimeError("No teacher checkpoints found for this fold.")
    probs_matrix = np.stack(all_probs, axis=1)  # (N, M)
    return {"case_id": case_ids, "y_true": y_true,
            "teacher_mean": probs_matrix.mean(axis=1),
            "teacher_std":  probs_matrix.std(axis=1),
            "n_teachers": probs_matrix.shape[1]}


def distill_loss(student_logit, true_label, teacher_mean, teacher_std,
                 alpha, beta):
    """Combined distillation loss.

    BCE on true label + KL on teacher soft labels + MSE on teacher std.
    """
    if student_logit.ndim == 1:
        student_logit = student_logit.unsqueeze(-1)
    bce = F.binary_cross_entropy_with_logits(student_logit, true_label,
                                             reduction="mean")
    # KL(teacher || student) on binary distribution
    p_s = torch.sigmoid(student_logit).clamp(EPS, 1.0 - EPS)
    p_t = teacher_mean.clamp(EPS, 1.0 - EPS)
    kl = p_t * (p_t.log() - p_s.log()) + \
         (1.0 - p_t) * ((1.0 - p_t).log() - (1.0 - p_s).log())
    kl = kl.mean()
    loss = (1.0 - alpha) * bce + alpha * kl
    # Optional: student attempts to match teacher std via a confidence proxy.
    # We use abs(p_s - 0.5) inverted (higher = uncertain), encouraging the
    # student to be uncertain where the teacher ensemble was uncertain.
    if beta > 0:
        student_unc = 2.0 * p_s * (1.0 - p_s)  # in [0, 0.5], peaks at p=0.5
        target_unc = (4.0 * teacher_std).clamp(0.0, 1.0)
        mse = F.mse_loss(student_unc, target_unc)
        loss = loss + beta * mse
    return loss, {"bce": float(bce.detach()),
                  "kl": float(kl.detach()),
                  "total": float(loss.detach())}


def main():
    ap = argparse.ArgumentParser(
        description="Distill heterogeneous ensemble into a ResNet-18 student")
    ap.add_argument("--index_csv", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--runs_root", type=str,
                    default="/raid/candi/michal/miqa/runs")
    ap.add_argument("--modality", type=str, default="dwi")
    ap.add_argument("--use_mask_crop", action="store_true", default=True)
    ap.add_argument("--no_mask_crop", dest="use_mask_crop",
                    action="store_false")
    ap.add_argument("--k_slices", type=int, default=3)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--max_cases", type=int, default=0)
    # Student + training
    ap.add_argument("--student", type=str, default="resnet18",
                    choices=["resnet18", "resnet50"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    # KD knobs
    ap.add_argument("--kd_alpha", type=float, default=0.7,
                    help="Weight on KL (vs (1-alpha) on BCE-true)")
    ap.add_argument("--kd_beta", type=float, default=0.5,
                    help="Weight on uncertainty-matching MSE (0 = off)")
    # Hardware
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no_amp", dest="amp", action="store_false")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--teacher_archs", type=str, default="all",
                    help=("Comma-separated list of teacher archs to keep "
                          "(e.g. 'densenet121' for a homogeneous-teacher "
                          "control). Default 'all' uses all 9 teachers."))

    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    runs_root = Path(args.runs_root)

    # Optional teacher-arch filter (used for homogeneous-teacher control)
    if args.teacher_archs != "all":
        allowed = set(a.strip() for a in args.teacher_archs.split(","))
        global TEACHERS
        original_n = len(TEACHERS)
        TEACHERS = [(a, p) for a, p in TEACHERS if a in allowed]
        print(f"Filtered teachers: {original_n} -> {len(TEACHERS)} "
              f"(kept archs: {sorted(allowed)})")
        if not TEACHERS:
            raise ValueError(f"No teachers match --teacher_archs "
                             f"{args.teacher_archs}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available()
                          else "cpu")
    amp_enabled = (device.type == "cuda") and args.amp
    print_hw_banner(device, amp_enabled)

    # Splits
    df = pd.read_csv(args.index_csv)
    group_col = "patient_id" if "patient_id" in df.columns else None
    tr, va, te = load_or_create_kfold_splits(
        df, args.n_folds, args.fold, args.split_seed, out,
        group_col=group_col)
    if args.max_cases > 0:
        tr = tr.iloc[: args.max_cases]
        va = va.iloc[: max(4, args.max_cases // 3)]
        te = te.iloc[: max(4, args.max_cases // 3)]
    print(f"Split: train={len(tr)}, val={len(va)}, test={len(te)}")

    ds_kwargs = dict(modality=args.modality, use_mask_crop=args.use_mask_crop,
                     k_slices=args.k_slices, img_size=args.img_size)

    # === PHASE 1: compute teacher predictions on train + val + test ===
    print("=== Phase 1: precomputing teacher ensemble predictions ===")
    teacher_cache = out / "teacher_preds.csv"
    if teacher_cache.exists():
        print(f"Loading cached teacher predictions from {teacher_cache}")
        tp = pd.read_csv(teacher_cache)
    else:
        rows = []
        for split_name, split_df in [("train", tr), ("val", va), ("test", te)]:
            print(f" Teachers on {split_name} ({len(split_df)} cases)...")
            agg = get_teacher_predictions(runs_root, args.fold, split_df,
                                          ds_kwargs, args.batch, device,
                                          amp_enabled)
            for cid, yt, m, s in zip(agg["case_id"], agg["y_true"],
                                     agg["teacher_mean"], agg["teacher_std"]):
                rows.append({"case_id": cid, "split": split_name,
                             "y_true": int(yt), "teacher_mean": float(m),
                             "teacher_std": float(s)})
        tp = pd.DataFrame(rows)
        tp.to_csv(teacher_cache, index=False)
        print(f"Cached -> {teacher_cache}")
    print(f"  loaded {len(tp)} teacher predictions")

    # === PHASE 2: train student ===
    print("=== Phase 2: training student ===")
    in_chans = (args.k_slices * 2 if args.modality == "fusion"
                else args.k_slices)
    student = build_model(args.student, in_chans=in_chans,
                          pretrained=True).to(device)
    n_params = sum(p.numel() for p in student.parameters())
    print(f"Student: {args.student}  params={n_params:,}")

    # Map case_id to (teacher_mean, teacher_std) for fast lookup at train time
    tp_train = tp[tp["split"] == "train"].set_index("case_id")
    teacher_lookup = {cid: (float(r.teacher_mean), float(r.teacher_std))
                      for cid, r in tp_train.iterrows()}

    tr_ds = UCLDistortionDataset(tr, augment=True, return_case_id=True,
                                  **ds_kwargs)
    va_ds = UCLDistortionDataset(va, augment=False, **ds_kwargs)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                       num_workers=args.num_workers, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch, shuffle=False,
                       num_workers=args.num_workers)

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr,
                            weight_decay=args.wd)
    scaler = amp_grad_scaler(enabled=amp_enabled)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    config = vars(args).copy()
    config["task"] = "knowledge_distillation"
    config["n_params"] = n_params
    config["n_teachers"] = 9
    config["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out / "config.json").write_text(json.dumps(config, indent=2))
    logf = (out / "metrics.jsonl").open("a")

    best_val = {"epoch": -1, "auroc": 0.0}
    t0 = time.time()
    print(f"Training: {args.epochs} epochs, alpha={args.kd_alpha}, "
          f"beta={args.kd_beta}")

    for epoch in range(1, args.epochs + 1):
        student.train()
        ep_loss = 0.0
        ep_n = 0
        for x, y, cids in tr_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float()
            if y.ndim == 1:
                y = y.unsqueeze(-1)
            # Look up teacher predictions for this batch
            t_m, t_s = [], []
            for cid in cids:
                tm, ts = teacher_lookup.get(cid, (None, None))
                t_m.append(0.5 if tm is None else tm)
                t_s.append(0.0 if ts is None else ts)
            t_m = torch.tensor(t_m, device=device, dtype=torch.float32).unsqueeze(-1)
            t_s = torch.tensor(t_s, device=device, dtype=torch.float32).unsqueeze(-1)
            opt.zero_grad(set_to_none=True)
            with amp_autocast(amp_enabled):
                logit = student(x)
                if logit.ndim == 1:
                    logit = logit.unsqueeze(-1)
                loss, _ = distill_loss(logit, y, t_m, t_s,
                                       args.kd_alpha, args.kd_beta)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ep_loss += float(loss.detach()) * x.size(0)
            ep_n += x.size(0)
        sched.step()
        train_loss = ep_loss / max(1, ep_n)

        # Validate
        val_metrics = evaluate(student, va_dl, device, amp_enabled=amp_enabled)
        record = {"epoch": epoch, "train_loss": round(train_loss, 5),
                  "val_auroc": val_metrics.get("auroc"),
                  "val_bal_acc": val_metrics.get("bal_acc"),
                  "lr": opt.param_groups[0]["lr"]}
        logf.write(json.dumps(record) + "\n"); logf.flush()
        if val_metrics.get("auroc", 0) > best_val["auroc"]:
            best_val = {"epoch": epoch, **val_metrics}
            torch.save({"model": student.state_dict(), "args": vars(args),
                        "epoch": epoch, "val": val_metrics},
                       out / "best_auroc.pt")
        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  loss={train_loss:.4f}  "
                  f"val_auroc={val_metrics.get('auroc', 0):.4f}")

    logf.close()
    print("Loading best student for test evaluation...")
    ckpt = torch.load(out / "best_auroc.pt", map_location=device,
                      weights_only=False)
    student.load_state_dict(ckpt["model"])
    student.eval()

    te_ds = UCLDistortionDataset(te, augment=False, return_case_id=True,
                                  **ds_kwargs)
    te_dl = DataLoader(te_ds, batch_size=args.batch, shuffle=False)
    test_metrics = evaluate(student, te_dl, device, amp_enabled=amp_enabled)

    # Save test predictions (student) + comparison to teacher ensemble
    all_ids, all_true, all_prob = [], [], []
    with torch.no_grad():
        for x, y, cid in te_dl:
            with amp_autocast(amp_enabled):
                logit = student(x.to(device))
            if logit.ndim == 1:
                logit = logit.unsqueeze(-1)
            all_ids.extend(cid)
            all_true.extend(y.numpy().ravel())
            all_prob.extend(torch.sigmoid(logit).cpu().numpy().ravel())
    pd.DataFrame({"case_id": all_ids,
                  "y_true": np.array(all_true, dtype=int),
                  "y_prob": np.round(all_prob, 6)
                  }).to_csv(out / "test_predictions.csv", index=False)

    # Teacher-ensemble metrics on the same test set (for comparison)
    tp_test = tp[tp["split"] == "test"].set_index("case_id")
    teacher_test_probs = [float(tp_test.loc[c, "teacher_mean"])
                          if c in tp_test.index else 0.5 for c in all_ids]
    from sklearn.metrics import roc_auc_score
    try:
        teacher_auroc = float(roc_auc_score(all_true, teacher_test_probs))
    except Exception:
        teacher_auroc = None

    elapsed = round(time.time() - t0, 2)
    summary = {
        "run_dir": str(out),
        "task": "knowledge_distillation",
        "device": str(device),
        "torch": torch.__version__,
        "secs": elapsed,
        "n_params_student": n_params,
        "n_teachers": int(tp["case_id"].nunique() and 9),
        "arch": args.student,
        "kd_alpha": args.kd_alpha,
        "kd_beta": args.kd_beta,
        "epochs": args.epochs,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "best_epoch": int(best_val.get("epoch", -1)),
        "best_val": {k: v for k, v in best_val.items() if k != "epoch"},
        "test_student": test_metrics,
        "test_teacher_ensemble_auroc": teacher_auroc,
        "k_slices": args.k_slices,
        "modality": args.modality,
        "img_size": args.img_size,
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
