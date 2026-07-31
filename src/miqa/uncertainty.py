"""Uncertainty quantification: MC Dropout, Deep Ensembles, metrics, plots.

Two UQ methods:
  1. MC Dropout: single model with dropout kept active at inference
  2. Deep Ensembles: multiple independently trained models

Both return the same result schema (dict with y_true, probs_matrix,
mean_prob, std_prob, predictive_entropy, mutual_information, case_ids).
"""

import numpy as np
import torch
import torch.nn as nn

from miqa.models import build_model
from miqa.utils import amp_autocast


# ---------------------------------------------------------------------------
# Model construction with explicit dropout
# ---------------------------------------------------------------------------

def _get_head_info(model, name):
    """Return (attr_chain, in_features) for the classification head."""
    if name in ("resnet18", "resnet50"):
        return "fc", model.fc.in_features
    elif name == "densenet121":
        return "classifier", model.classifier.in_features
    elif name == "efficientnet_b0":
        # EfficientNet classifier is nn.Sequential(Dropout, Linear)
        return "classifier", model.classifier[1].in_features
    else:
        raise ValueError(f"Unknown model: {name}")


def build_model_with_dropout(name: str = "resnet18", in_chans: int = 3,
                             pretrained: bool = True,
                             dropout_rate: float = 0.2):
    """Build a classification model with explicit Dropout before the FC head.

    For ResNet: replaces model.fc with Sequential(Dropout, Linear)
    For DenseNet: replaces model.classifier with Sequential(Dropout, Linear)
    For EfficientNet: replaces model.classifier with Sequential(Dropout, Linear)
    """
    model = build_model(name, in_chans=in_chans, pretrained=pretrained)
    attr, feat_dim = _get_head_info(model, name)

    new_head = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(feat_dim, 1),
    )

    if name in ("resnet18", "resnet50"):
        model.fc = new_head
    elif name == "densenet121":
        model.classifier = new_head
    elif name == "efficientnet_b0":
        model.classifier = new_head

    return model


# ---------------------------------------------------------------------------
# MC Dropout inference
# ---------------------------------------------------------------------------

def enable_mc_dropout(model):
    """Set all Dropout modules to train() while the rest stays eval().

    Call model.eval() first, then enable_mc_dropout(model).
    """
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def mc_dropout_predict(model, loader, device, n_forward=50,
                       amp_enabled=False):
    """Run MC Dropout inference: T stochastic forward passes per sample.

    Returns dict with:
        y_true          (N,)    ground-truth labels
        probs_matrix    (N, T)  sigmoid probabilities per pass
        mean_prob       (N,)    mean predicted probability
        std_prob        (N,)    std of predicted probabilities
        predictive_entropy (N,) entropy of mean prediction
        mutual_information (N,) MI = H[mean] - mean(H[individual])
        case_ids        list[str] or None
    """
    model.eval()
    enable_mc_dropout(model)

    # Collect data in a single pass to get labels and case_ids
    all_x, all_y, all_ids = [], [], []
    for batch in loader:
        x = batch[0]
        y = batch[1]
        all_x.append(x)
        all_y.append(y.numpy().reshape(-1))
        if len(batch) >= 3:
            all_ids.extend(batch[2])

    y_true = np.concatenate(all_y)
    case_ids = all_ids if all_ids else None

    # Run T forward passes
    all_probs = []  # list of (N,) arrays
    for _ in range(n_forward):
        pass_probs = []
        for batch in loader:
            x = batch[0].to(device, non_blocking=True)
            with amp_autocast(amp_enabled):
                logit = model(x)
            prob = torch.sigmoid(logit).cpu().numpy().reshape(-1)
            pass_probs.append(prob)
        all_probs.append(np.concatenate(pass_probs))

    probs_matrix = np.stack(all_probs, axis=1)  # (N, T)
    mean_prob = probs_matrix.mean(axis=1)
    std_prob = probs_matrix.std(axis=1)

    return {
        "y_true": y_true,
        "probs_matrix": probs_matrix,
        "mean_prob": mean_prob,
        "std_prob": std_prob,
        "predictive_entropy": predictive_entropy(mean_prob),
        "mutual_information": mutual_information(probs_matrix),
        "case_ids": case_ids,
    }


# ---------------------------------------------------------------------------
# Deep Ensemble inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def ensemble_predict(checkpoints, model_name, in_chans, loader, device,
                     amp_enabled=False):
    """Run Deep Ensemble inference: one forward pass per checkpoint.

    Args:
        checkpoints: list of checkpoint file paths
        model_name: architecture name for build_model
        in_chans: number of input channels
        loader: DataLoader
        device: torch device
        amp_enabled: use AMP autocast

    Returns: same dict schema as mc_dropout_predict.
    """
    # Collect labels and case_ids once
    all_y, all_ids = [], []
    for batch in loader:
        y = batch[1]
        all_y.append(y.numpy().reshape(-1))
        if len(batch) >= 3:
            all_ids.extend(batch[2])

    y_true = np.concatenate(all_y)
    case_ids = all_ids if all_ids else None

    # Forward pass per checkpoint
    all_probs = []
    for ckpt_path in checkpoints:
        model = build_model(model_name, in_chans=in_chans, pretrained=False)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model.to(device)
        model.eval()

        pass_probs = []
        for batch in loader:
            x = batch[0].to(device, non_blocking=True)
            with amp_autocast(amp_enabled):
                logit = model(x)
            prob = torch.sigmoid(logit).cpu().numpy().reshape(-1)
            pass_probs.append(prob)
        all_probs.append(np.concatenate(pass_probs))

    probs_matrix = np.stack(all_probs, axis=1)  # (N, M)
    mean_prob = probs_matrix.mean(axis=1)
    std_prob = probs_matrix.std(axis=1)

    return {
        "y_true": y_true,
        "probs_matrix": probs_matrix,
        "mean_prob": mean_prob,
        "std_prob": std_prob,
        "predictive_entropy": predictive_entropy(mean_prob),
        "mutual_information": mutual_information(probs_matrix),
        "case_ids": case_ids,
    }


# ---------------------------------------------------------------------------
# Test-Time Augmentation (TTA) inference
# ---------------------------------------------------------------------------

def _apply_tta_augment(x):
    """Apply stochastic augmentation to a batched tensor (B, C, H, W).

    Mirrors the train-time augmentation in UCLDistortionDataset (flips, additive
    Gaussian noise, 90-degree rotations) but operates on a batched tensor, so
    spatial dims are [2, 3] rather than [1, 2].
    """
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[3])           # horizontal flip
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[2])           # vertical flip
    if torch.rand(1).item() < 0.3:
        x = (x + 0.02 * torch.randn_like(x)).clamp(0.0, 1.0)
    if torch.rand(1).item() < 0.3:
        k = int(torch.randint(1, 4, (1,)).item())
        x = torch.rot90(x, k, dims=[2, 3])
    return x


@torch.no_grad()
def tta_predict(model, loader, device, n_forward=50, amp_enabled=False,
                include_identity=True):
    """Run Test-Time Augmentation inference: T augmented forward passes.

    Uncertainty arises from input perturbations (augmentations), an axis
    orthogonal to MC Dropout (weight perturbations) and Deep Ensembles
    (initialization perturbations). The first pass uses the un-augmented input
    when include_identity=True so the central prediction is anchored.

    Returns the same dict schema as mc_dropout_predict / ensemble_predict.
    """
    model.eval()

    # Collect labels and case_ids once
    all_y, all_ids = [], []
    for batch in loader:
        all_y.append(batch[1].numpy().reshape(-1))
        if len(batch) >= 3:
            all_ids.extend(batch[2])
    y_true = np.concatenate(all_y)
    case_ids = all_ids if all_ids else None

    all_probs = []
    for t in range(n_forward):
        pass_probs = []
        for batch in loader:
            x = batch[0].to(device, non_blocking=True)
            if not (include_identity and t == 0):
                x = _apply_tta_augment(x)
            with amp_autocast(amp_enabled):
                logit = model(x)
            prob = torch.sigmoid(logit).cpu().numpy().reshape(-1)
            pass_probs.append(prob)
        all_probs.append(np.concatenate(pass_probs))

    probs_matrix = np.stack(all_probs, axis=1)  # (N, T)
    mean_prob = probs_matrix.mean(axis=1)
    std_prob = probs_matrix.std(axis=1)

    return {
        "y_true": y_true,
        "probs_matrix": probs_matrix,
        "mean_prob": mean_prob,
        "std_prob": std_prob,
        "predictive_entropy": predictive_entropy(mean_prob),
        "mutual_information": mutual_information(probs_matrix),
        "case_ids": case_ids,
    }


# ---------------------------------------------------------------------------
# Uncertainty metrics (pure numpy)
# ---------------------------------------------------------------------------

def predictive_entropy(p):
    """Binary entropy H(p) = -p*log(p) - (1-p)*log(1-p).

    Args:
        p: ndarray of probabilities in [0, 1]
    Returns:
        ndarray of same shape with entropy values
    """
    p = np.asarray(p, dtype=np.float64)
    eps = 1e-12
    p = np.clip(p, eps, 1.0 - eps)
    return -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)


def mutual_information(probs_matrix):
    """Mutual information: MI = H[mean] - mean(H[individual]).

    Measures epistemic uncertainty (model disagreement).

    Args:
        probs_matrix: (N, T) array of probabilities from T passes/models
    Returns:
        (N,) array of MI values
    """
    probs_matrix = np.asarray(probs_matrix, dtype=np.float64)
    mean_prob = probs_matrix.mean(axis=1)
    h_mean = predictive_entropy(mean_prob)
    h_individual = predictive_entropy(probs_matrix)  # (N, T)
    mean_h = h_individual.mean(axis=1)
    return h_mean - mean_h


def uncertainty_error_auroc(y_true, y_pred, uncertainty):
    """AUROC for using uncertainty to predict classification errors.

    Higher AUROC means uncertainty is better at identifying errors.
    """
    from sklearn.metrics import roc_auc_score
    errors = (y_true != y_pred).astype(int)
    if len(np.unique(errors)) < 2:
        return float("nan")
    return float(roc_auc_score(errors, uncertainty))


def selective_prediction_curve(y_true, mean_prob, uncertainty,
                               n_points=20):
    """Compute selective prediction (risk-coverage) curve.

    Progressively reject the most uncertain samples and measure
    accuracy on the retained set.

    Returns dict with:
        coverage   (n_points,) fraction of data retained
        accuracy   (n_points,) accuracy on retained data
        auroc      (n_points,) AUROC on retained data (NaN if not computable)
        n_retained (n_points,) number of samples retained
    """
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    mean_prob = np.asarray(mean_prob)
    uncertainty = np.asarray(uncertainty)

    n = len(y_true)
    y_pred = (mean_prob >= 0.5).astype(int)

    # Sort by uncertainty (ascending = most certain first)
    order = np.argsort(uncertainty)

    coverages = np.linspace(1.0, 0.1, n_points)
    results = {"coverage": [], "accuracy": [], "auroc": [], "n_retained": []}

    for cov in coverages:
        k = max(1, int(np.ceil(cov * n)))
        idx = order[:k]
        yt = y_true[idx]
        yp = y_pred[idx]
        mp = mean_prob[idx]

        acc = float((yt == yp).mean())
        try:
            auc = float(roc_auc_score(yt, mp))
        except ValueError:
            auc = float("nan")

        results["coverage"].append(round(float(cov), 4))
        results["accuracy"].append(round(acc, 4))
        results["auroc"].append(round(auc, 4) if not np.isnan(auc) else None)
        results["n_retained"].append(int(k))

    return results


# ---------------------------------------------------------------------------
# Plotting (follows calibration.py pattern: Agg backend, savefig, print)
# ---------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_selective_prediction(data, out_path, title=""):
    """Plot selective prediction curve: accuracy vs coverage."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(data["coverage"], data["accuracy"], "o-", linewidth=2,
            markersize=5, color="tab:blue")
    ax.set_xlabel("Coverage (fraction retained)")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0.05, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.axhline(y=data["accuracy"][0], color="grey", linestyle="--",
               alpha=0.5, label=f"Full coverage: {data['accuracy'][0]:.3f}")
    ax.legend()
    ax.set_title(title or "Selective Prediction Curve")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved selective prediction curve to {out_path}")


def plot_uncertainty_histogram(y_true, y_pred, uncertainty, out_path,
                               title=""):
    """Plot uncertainty distributions for correct vs incorrect predictions."""
    correct = y_true == y_pred
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(uncertainty[correct], bins=30, alpha=0.6, label="Correct",
            color="tab:green", density=True)
    ax.hist(uncertainty[~correct], bins=30, alpha=0.6, label="Incorrect",
            color="tab:red", density=True)
    ax.set_xlabel("Uncertainty (std)")
    ax.set_ylabel("Density")
    ax.set_title(title or "Uncertainty Distribution: Correct vs Incorrect")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved uncertainty histogram to {out_path}")


def plot_uncertainty_calibration(y_true, y_pred, uncertainty, out_path,
                                 n_bins=10, title=""):
    """Plot uncertainty calibration: binned error rate vs mean uncertainty."""
    errors = (y_true != y_pred).astype(float)
    bin_edges = np.linspace(uncertainty.min(), uncertainty.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    error_rates = []
    counts = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        if lo == bin_edges[-2]:
            mask = (uncertainty >= lo) & (uncertainty <= hi)
        else:
            mask = (uncertainty >= lo) & (uncertainty < hi)
        count = int(mask.sum())
        counts.append(count)
        if count == 0:
            error_rates.append(0.0)
        else:
            error_rates.append(float(errors[mask].mean()))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7),
                                    gridspec_kw={"height_ratios": [3, 1]})

    ax1.bar(bin_centers, error_rates, width=(bin_edges[1] - bin_edges[0]),
            alpha=0.6, edgecolor="black")
    ax1.plot([uncertainty.min(), uncertainty.max()],
             [uncertainty.min(), uncertainty.max()],
             "k--", linewidth=1, label="Perfect calibration")
    ax1.set_xlabel("Mean uncertainty")
    ax1.set_ylabel("Error rate")
    ax1.set_title(title or "Uncertainty Calibration")
    ax1.legend()

    ax2.bar(bin_centers, counts, width=(bin_edges[1] - bin_edges[0]),
            alpha=0.6, edgecolor="black", color="tab:orange")
    ax2.set_xlabel("Mean uncertainty")
    ax2.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved uncertainty calibration to {out_path}")
