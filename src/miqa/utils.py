"""Shared utilities: seeding, hardware info, patient splits, AMP compat."""

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# AMP compatibility (torch 2.0 to 2.2 vs 2.3+)
# ---------------------------------------------------------------------------
_torch_ge_2_3 = (
    tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    >= (2, 3)
)

if _torch_ge_2_3:
    amp_autocast = lambda enabled: torch.amp.autocast("cuda", enabled=enabled)
    amp_grad_scaler = lambda enabled: torch.amp.GradScaler("cuda", enabled=enabled)
else:
    amp_autocast = lambda enabled: torch.cuda.amp.autocast(enabled=enabled)
    amp_grad_scaler = lambda enabled: torch.cuda.amp.GradScaler(enabled=enabled)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Hardware banner
# ---------------------------------------------------------------------------
def print_hw_banner(device, amp_enabled):
    print("=" * 60)
    print(f"torch {torch.__version__}  |  CUDA avail: {torch.cuda.is_available()}"
          f"  |  device: {device}  |  AMP: {amp_enabled}")
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        print(f"GPU {idx}: {torch.cuda.get_device_name(idx)}"
              f"  |  VRAM: {torch.cuda.get_device_properties(idx).total_memory / 1e9:.1f} GB")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Patient-level split
# ---------------------------------------------------------------------------
def make_patient_split(df: pd.DataFrame, seed: int, group_col=None):
    """70/15/15 split, stratified on label.

    When group_col is set (e.g. 'patient_id'), splitting is done at the
    group level so all acquisitions for a patient land in the same split.
    A group is assigned to the positive stratum if any of its rows has
    label=1.  When group_col is None, splits by case_id (original behaviour).
    """
    group = group_col or "case_id"
    rng = np.random.RandomState(seed)
    train_ids, val_ids, test_ids = [], [], []

    # Per-group label: max(label), if any acquisition is distorted, group=1
    group_label = df.groupby(group)["label"].max()

    for lab in sorted(group_label.unique()):
        ids = sorted(group_label[group_label == lab].index.tolist())
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)
        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train:n_train + n_val])
        test_ids.extend(ids[n_train + n_val:])

    tr = df[df[group].isin(set(train_ids))]
    va = df[df[group].isin(set(val_ids))]
    te = df[df[group].isin(set(test_ids))]
    return tr, va, te


def load_or_create_splits(df: pd.DataFrame, split_seed: int, out: Path,
                          group_col=None):
    """Load splits.json if present, else create and save it."""
    group = group_col or "case_id"
    splits_path = out / "splits.json"
    if splits_path.exists():
        splits = json.loads(splits_path.read_text())
        print(f"Loaded existing splits from {splits_path}")
        tr = df[df[group].isin(splits["train"])]
        va = df[df[group].isin(splits["val"])]
        te = df[df[group].isin(splits["test"])]
        return tr, va, te

    tr, va, te = make_patient_split(df, seed=split_seed, group_col=group_col)
    splits = {
        "split_seed": split_seed,
        "train": sorted(tr[group].unique().tolist()),
        "val": sorted(va[group].unique().tolist()),
        "test": sorted(te[group].unique().tolist()),
    }
    splits_path.write_text(json.dumps(splits, indent=2))
    print(f"Created and saved splits to {splits_path}")
    return tr, va, te


def make_kfold_splits(df: pd.DataFrame, n_folds: int = 5, seed: int = 42,
                      group_col=None):
    """Create n stratified folds.  Returns list of (train_df, val_df, test_df).

    For each fold k:
      - test = fold k groups (~N/n_folds groups)
      - val  = fold (k+1) % n_folds groups
      - train = remaining folds

    Every group appears in exactly one test fold.
    Stratified by label within each fold.

    When group_col is set (e.g. 'patient_id'), all acquisitions for a
    group land in the same fold.  A group is positive if any row has label=1.
    """
    group = group_col or "case_id"
    rng = np.random.RandomState(seed)
    # Build per-label group lists, shuffle, and split into n_folds chunks
    all_folds = [[] for _ in range(n_folds)]

    # Per-group label: max(label)
    group_label = df.groupby(group)["label"].max()

    for lab in sorted(group_label.unique()):
        ids = sorted(group_label[group_label == lab].index.tolist())
        rng.shuffle(ids)
        # Distribute ids round-robin into folds
        for i, gid in enumerate(ids):
            all_folds[i % n_folds].append(gid)

    # Build (train, val, test) splits for each fold
    splits = []
    for k in range(n_folds):
        test_ids = set(all_folds[k])
        val_ids = set(all_folds[(k + 1) % n_folds])
        train_ids = set()
        for j in range(n_folds):
            if j != k and j != (k + 1) % n_folds:
                train_ids.update(all_folds[j])

        tr = df[df[group].isin(train_ids)]
        va = df[df[group].isin(val_ids)]
        te = df[df[group].isin(test_ids)]
        splits.append((tr, va, te))

    return splits


def load_or_create_kfold_splits(df: pd.DataFrame, n_folds: int, fold: int,
                                 seed: int, out: Path, group_col=None):
    """Load or create kfold_splits.json, return (train_df, val_df, test_df).

    Saves to out/kfold_splits.json with structure:
    {
      "n_folds": 5, "seed": 42,
      "folds": {
        "0": {"train": [...], "val": [...], "test": [...]},
        ...
      }
    }

    When group_col is set, the JSON stores group IDs (e.g. patient_ids)
    and filtering uses df[group_col].
    """
    group = group_col or "case_id"
    splits_path = out / "kfold_splits.json"
    if splits_path.exists():
        data = json.loads(splits_path.read_text())
        print(f"Loaded existing k-fold splits from {splits_path}")
        fold_data = data["folds"][str(fold)]
        tr = df[df[group].isin(fold_data["train"])]
        va = df[df[group].isin(fold_data["val"])]
        te = df[df[group].isin(fold_data["test"])]
        return tr, va, te

    all_splits = make_kfold_splits(df, n_folds=n_folds, seed=seed,
                                   group_col=group_col)
    data = {"n_folds": n_folds, "seed": seed, "folds": {}}
    for k, (tr_k, va_k, te_k) in enumerate(all_splits):
        data["folds"][str(k)] = {
            "train": sorted(tr_k[group].unique().tolist()),
            "val": sorted(va_k[group].unique().tolist()),
            "test": sorted(te_k[group].unique().tolist()),
        }
    splits_path.write_text(json.dumps(data, indent=2))
    print(f"Created and saved k-fold splits to {splits_path}")

    fold_data = data["folds"][str(fold)]
    tr = df[df[group].isin(fold_data["train"])]
    va = df[df[group].isin(fold_data["val"])]
    te = df[df[group].isin(fold_data["test"])]
    return tr, va, te


def load_test_split(df: pd.DataFrame, run_dir=None, split_seed: int = 42,
                    group_col=None):
    """Load test IDs from splits.json if available, else recompute."""
    group = group_col or "case_id"
    if run_dir:
        splits_path = Path(run_dir) / "splits.json"
        if splits_path.exists():
            splits = json.loads(splits_path.read_text())
            te = df[df[group].isin(splits["test"])].reset_index(drop=True)
            print(f"Loaded test split from {splits_path} ({len(te)} cases)")
            return te

    # Fallback: recompute
    _, _, te = make_patient_split(df, seed=split_seed, group_col=group_col)
    te = te.reset_index(drop=True)
    print(f"Recomputed test split (seed={split_seed}, {len(te)} cases)")
    return te
