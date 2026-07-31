"""Reload best_auroc.pt from a run directory and reconstruct summary.json.

Used to recover from the pandas bug that overwrote summary.json with an error
even though training and test metrics had completed successfully.

Usage:
    python scripts/eval_from_checkpoint.py --run_dir runs/poster_resnet18_dwi_roi_f0
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from miqa.utils import amp_autocast, load_or_create_kfold_splits, load_or_create_splits
from miqa.models import build_model
from miqa.datasets import UCLDistortionDataset
from miqa.metrics import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--index_csv", type=str,
                    default="repo/dataset/index_full.csv")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    config_path = run_dir / "config.json"
    if not config_path.exists():
        print(f"ERROR: {config_path} does not exist")
        sys.exit(1)

    config = json.loads(config_path.read_text())
    print(f"=== {run_dir.name} ===")

    # Load data + splits using the same logic as train.py
    df = pd.read_csv(args.index_csv)
    group_col = "patient_id" if "patient_id" in df.columns else None

    if config.get("n_folds", 0) > 0:
        tr, va, te = load_or_create_kfold_splits(
            df, config["n_folds"], config["fold"], config["split_seed"],
            run_dir, group_col=group_col)
    else:
        tr, va, te = load_or_create_splits(
            df, config["split_seed"], run_dir, group_col=group_col)

    print(f"  train={len(tr)}, val={len(va)}, test={len(te)}")

    # Build model and load best checkpoint
    in_chans = (config["k_slices"] * 2
                if config["modality"] == "fusion" else config["k_slices"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["model"], in_chans=in_chans, pretrained=False).to(device)

    best_path = run_dir / "best_auroc.pt"
    if not best_path.exists():
        print(f"ERROR: {best_path} does not exist")
        sys.exit(1)
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    # Build test dataloader
    ds_kwargs = dict(
        modality=config["modality"],
        use_mask_crop=config.get("use_mask_crop", True),
        k_slices=config["k_slices"],
        img_size=config.get("img_size", 224),
    )
    test_ds = UCLDistortionDataset(te, augment=False, **ds_kwargs)
    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=config["batch"], shuffle=False)

    # Evaluate
    t0 = time.time()
    test_metrics = evaluate(model, test_dl, device,
                            amp_enabled=config.get("amp", True))
    eval_secs = round(time.time() - t0, 2)

    # Per-sample predictions
    test_pred_ds = UCLDistortionDataset(te, augment=False, return_case_id=True,
                                         **ds_kwargs)
    test_pred_dl = torch.utils.data.DataLoader(
        test_pred_ds, batch_size=config["batch"], shuffle=False)
    all_ids, all_true, all_prob = [], [], []
    with torch.no_grad():
        for x, y, cid in test_pred_dl:
            logits = model(x.to(device)).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_ids.extend(list(cid))
            all_true.extend(y.numpy().flatten().tolist())
            all_prob.extend(probs.tolist())
    try:
        pd.DataFrame({
            "case_id": all_ids,
            "y_true": [int(v) for v in all_true],
            "y_prob": [round(float(v), 6) for v in all_prob],
        }).to_csv(run_dir / "test_predictions.csv", index=False)
    except Exception as e:
        print(f"WARNING: could not save test_predictions.csv: {e}")

    # Rewrite summary.json with real metrics
    meta = {
        "run_dir": str(run_dir),
        "recovered_from_checkpoint": True,
        "arch": config["model"],
        "modality": config["modality"],
        "use_mask_crop": config.get("use_mask_crop", True),
        "k_slices": config["k_slices"],
        "img_size": config.get("img_size", 224),
        "lr": config["lr"],
        "wd": config.get("wd", 1e-4),
        "batch": config["batch"],
        "epochs": config["epochs"],
        "scheduler": config["scheduler"],
        "seed": config["seed"],
        "split_seed": config["split_seed"],
        "n_folds": config.get("n_folds", 0),
        "fold": config.get("fold", 0),
        "n_train": len(tr),
        "n_val": len(va),
        "n_test": len(te),
        "eval_secs": eval_secs,
        "test": test_metrics,
    }
    (run_dir / "summary.json").write_text(json.dumps(meta, indent=2))
    print(f"  test AUROC={test_metrics['auroc']:.4f}, "
          f"AUPRC={test_metrics['auprc']:.4f}, "
          f"F1={test_metrics['f1']:.4f}, "
          f"bal_acc={test_metrics['bal_acc']:.4f}")
    print(f"  saved to {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
