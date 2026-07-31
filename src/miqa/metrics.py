"""Evaluation metrics: evaluate loop, bootstrap CIs, ECE."""

import numpy as np
import torch
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score,
                             f1_score, average_precision_score)

from miqa.utils import amp_autocast


@torch.no_grad()
def evaluate(model, loader, device, amp_enabled=False, loss_fn=None):
    """Run inference on a DataLoader and return classification metrics.

    Works with both 2-tuple (x, y) and 3-tuple (x, y, case_id) loaders.
    If loss_fn is provided, also computes and returns 'loss'.
    """
    model.eval()
    ys, probs = [], []
    total_loss, total_n = 0.0, 0
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        y = batch[1]
        with amp_autocast(amp_enabled):
            logit = model(x)
            if loss_fn is not None:
                y_dev = y.to(device, non_blocking=True)
                total_loss += float(loss_fn(logit, y_dev).item()) * x.size(0)
                total_n += x.size(0)
        prob = torch.sigmoid(logit).cpu().numpy().reshape(-1)
        ys.append(y.numpy().reshape(-1))
        probs.append(prob)

    y = np.concatenate(ys)
    p = np.concatenate(probs)
    pred = (p >= 0.5).astype(int)

    has_both = len(np.unique(y)) > 1
    auroc = float(roc_auc_score(y, p)) if has_both else 0.0
    auprc = float(average_precision_score(y, p)) if has_both else 0.0
    bal_acc = float(balanced_accuracy_score(y, pred))
    f1 = float(f1_score(y, pred, zero_division=0))

    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    result = {
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "bal_acc": round(bal_acc, 4),
        "f1": round(f1, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }
    if loss_fn is not None and total_n > 0:
        result["loss"] = round(total_loss / total_n, 5)
    return result


def bootstrap_ci(y_true, y_prob, n_boot=1000, seed=42):
    """Compute bootstrapped CIs for auroc, auprc, bal_acc, f1."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    results = {k: [] for k in ["auroc", "auprc", "bal_acc", "f1"]}

    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yb = y_true[idx]
        pb = y_prob[idx]
        predb = (pb >= 0.5).astype(int)

        if len(np.unique(yb)) < 2:
            continue
        results["auroc"].append(float(roc_auc_score(yb, pb)))
        results["auprc"].append(float(average_precision_score(yb, pb)))
        results["bal_acc"].append(float(balanced_accuracy_score(yb, predb)))
        results["f1"].append(float(f1_score(yb, predb, zero_division=0)))

    out = {}
    for k, vals in results.items():
        if len(vals) < 10:
            out[k] = {"mean": None, "ci_lo": None, "ci_hi": None}
            continue
        arr = np.array(vals)
        out[k] = {
            "mean": round(float(arr.mean()), 4),
            "ci_lo": round(float(np.percentile(arr, 2.5)), 4),
            "ci_hi": round(float(np.percentile(arr, 97.5)), 4),
        }
    return out


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Compute ECE and return per-bin data."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    n = len(y_true)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        if lo == bin_edges[-2]:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        count = int(mask.sum())
        if count == 0:
            bins.append({
                "bin_lo": round(float(lo), 3),
                "bin_hi": round(float(hi), 3),
                "count": 0,
                "avg_confidence": None,
                "avg_accuracy": None,
            })
            continue
        avg_conf = float(y_prob[mask].mean())
        avg_acc = float(y_true[mask].mean())
        ece += (count / n) * abs(avg_acc - avg_conf)
        bins.append({
            "bin_lo": round(float(lo), 3),
            "bin_hi": round(float(hi), 3),
            "count": count,
            "avg_confidence": round(avg_conf, 4),
            "avg_accuracy": round(avg_acc, 4),
        })
    return round(float(ece), 4), bins
