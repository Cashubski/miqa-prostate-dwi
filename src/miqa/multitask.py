"""Multi-task model: distortion classification + b-value prediction.

Shared backbone (ResNet/DenseNet/EfficientNet) with two heads:
  - Distortion head: FC to 1 (binary, BCEWithLogitsLoss)
  - B-value head:   FC to num_classes (CrossEntropyLoss)

Combined loss: L = L_dist + lambda * L_bval
"""

import numpy as np
import torch
import torch.nn as nn

from miqa.models import build_backbone
from miqa.utils import amp_autocast

# b-value class mapping, merge 1k4+1k5 (1k5 has only 1 sample)
BVAL_MAP = {"1k4": 0, "1k5": 0, "1k6": 1, "2k": 2}
NUM_BVAL_CLASSES = 3
BVAL_NAMES = ["1k4-1k5", "1k6", "2k"]


class MultiTaskModel(nn.Module):
    """Shared backbone with distortion head (binary) and b-value head (3-class)."""

    def __init__(self, backbone_name: str = "resnet18", in_chans: int = 3,
                 num_bval_classes: int = NUM_BVAL_CLASSES,
                 pretrained: bool = True, dropout: float = 0.0):
        super().__init__()
        self.backbone, feat_dim = build_backbone(
            backbone_name, in_chans, pretrained)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.dist_head = nn.Linear(feat_dim, 1)
        self.bval_head = nn.Linear(feat_dim, num_bval_classes)

    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        dist_logit = self.dist_head(features)
        bval_logit = self.bval_head(features)
        return dist_logit, bval_logit

    def forward_distortion_only(self, x):
        """For inference: only return distortion logit."""
        features = self.backbone(x)
        return self.dist_head(features)


@torch.no_grad()
def evaluate_multitask(model, loader, device, amp_enabled=False,
                       dist_loss_fn=None, bval_loss_fn=None):
    """Evaluate multi-task model. Returns dict with distortion + b-value metrics.

    Loader must yield (x, y_dist, y_bval) or (x, y_dist, y_bval, case_id).
    """
    model.eval()
    ys_dist, probs_dist = [], []
    ys_bval, preds_bval = [], []
    total_dist_loss, total_bval_loss, total_n = 0.0, 0.0, 0

    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        y_dist = batch[1]
        y_bval = batch[2]
        with amp_autocast(amp_enabled):
            dist_logit, bval_logit = model(x)
            if dist_loss_fn is not None:
                yd = y_dist.to(device, non_blocking=True)
                total_dist_loss += float(
                    dist_loss_fn(dist_logit, yd).item()) * x.size(0)
            if bval_loss_fn is not None:
                yb = y_bval.to(device, non_blocking=True)
                total_bval_loss += float(
                    bval_loss_fn(bval_logit, yb).item()) * x.size(0)
            total_n += x.size(0)

        prob = torch.sigmoid(dist_logit).cpu().numpy().reshape(-1)
        ys_dist.append(y_dist.numpy().reshape(-1))
        probs_dist.append(prob)

        bval_pred = bval_logit.argmax(dim=1).cpu().numpy()
        ys_bval.append(y_bval.numpy().reshape(-1))
        preds_bval.append(bval_pred)

    y = np.concatenate(ys_dist)
    p = np.concatenate(probs_dist)
    pred = (p >= 0.5).astype(int)

    from sklearn.metrics import (roc_auc_score, balanced_accuracy_score,
                                 f1_score, average_precision_score)

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

    # B-value accuracy
    yb_all = np.concatenate(ys_bval)
    pb_all = np.concatenate(preds_bval)
    bval_acc = float((yb_all == pb_all).mean())
    bval_bal_acc = float(balanced_accuracy_score(yb_all, pb_all))

    result = {
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "bal_acc": round(bal_acc, 4),
        "f1": round(f1, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "bval_acc": round(bval_acc, 4),
        "bval_bal_acc": round(bval_bal_acc, 4),
    }
    if dist_loss_fn is not None and total_n > 0:
        result["dist_loss"] = round(total_dist_loss / total_n, 5)
    if bval_loss_fn is not None and total_n > 0:
        result["bval_loss"] = round(total_bval_loss / total_n, 5)
    return result
