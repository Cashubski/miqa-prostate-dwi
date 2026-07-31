"""Tests for miqa.uncertainty: dropout models, MC inference, metrics, plots."""

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from miqa.uncertainty import (
    build_model_with_dropout,
    enable_mc_dropout,
    mc_dropout_predict,
    ensemble_predict,
    predictive_entropy,
    mutual_information,
    uncertainty_error_auroc,
    selective_prediction_curve,
    plot_selective_prediction,
    plot_uncertainty_histogram,
    plot_uncertainty_calibration,
)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "resnet18", "resnet50", "densenet121", "efficientnet_b0",
])
def test_dropout_model_output_shape(name):
    model = build_model_with_dropout(name, in_chans=3, pretrained=False,
                                     dropout_rate=0.3)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1)


@pytest.mark.parametrize("name", [
    "resnet18", "resnet50", "densenet121", "efficientnet_b0",
])
def test_dropout_model_has_dropout_module(name):
    model = build_model_with_dropout(name, in_chans=3, pretrained=False,
                                     dropout_rate=0.25)
    dropouts = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    assert len(dropouts) >= 1
    assert dropouts[-1].p == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# MC Dropout behaviour
# ---------------------------------------------------------------------------

def test_enable_mc_dropout_outputs_vary():
    """With MC dropout enabled, repeated eval passes should differ."""
    model = build_model_with_dropout("resnet18", in_chans=3, pretrained=False,
                                     dropout_rate=0.5)
    model.eval()
    enable_mc_dropout(model)
    x = torch.randn(4, 3, 224, 224)
    outs = []
    for _ in range(5):
        with torch.no_grad():
            outs.append(model(x).clone())
    # At least some outputs should differ (very high dropout makes this near-certain)
    stacked = torch.stack(outs)  # (5, 4, 1)
    std_per_sample = stacked.std(dim=0)
    assert std_per_sample.mean().item() > 0.0


def test_without_mc_dropout_outputs_constant():
    """Standard eval (no MC dropout) should be deterministic."""
    model = build_model_with_dropout("resnet18", in_chans=3, pretrained=False,
                                     dropout_rate=0.5)
    model.eval()  # Dropout disabled in eval
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        out1 = model(x).clone()
        out2 = model(x).clone()
    assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------

def test_entropy_zero_for_confident():
    p = np.array([0.001, 0.999])
    h = predictive_entropy(p)
    assert np.all(h < 0.02)


def test_entropy_maximal_at_half():
    p = np.array([0.5])
    h = predictive_entropy(p)
    assert abs(h[0] - math.log(2)) < 1e-6


def test_entropy_symmetric():
    p = np.array([0.3, 0.7])
    h = predictive_entropy(p)
    assert abs(h[0] - h[1]) < 1e-10


# ---------------------------------------------------------------------------
# Mutual information
# ---------------------------------------------------------------------------

def test_mi_zero_for_identical_predictions():
    """If all T passes agree perfectly, MI should be ~0."""
    probs = np.ones((10, 5)) * 0.8  # 10 samples, 5 passes, all = 0.8
    mi = mutual_information(probs)
    assert np.allclose(mi, 0.0, atol=1e-10)


def test_mi_positive_for_disagreeing():
    """If passes disagree, MI should be > 0."""
    rng = np.random.RandomState(42)
    probs = rng.rand(10, 20)  # 10 samples, 20 random passes
    mi = mutual_information(probs)
    assert np.all(mi > 0.0)


# ---------------------------------------------------------------------------
# Selective prediction
# ---------------------------------------------------------------------------

def test_selective_prediction_monotonic():
    """Accuracy should generally improve as we reject uncertain samples.

    With perfectly correlated uncertainty-errors, accuracy is strictly
    non-decreasing as coverage decreases.
    """
    rng = np.random.RandomState(123)
    n = 200
    y_true = rng.randint(0, 2, size=n)
    # Make probabilities correlated with labels but noisy
    mean_prob = y_true * 0.8 + (1 - y_true) * 0.2 + rng.randn(n) * 0.1
    mean_prob = np.clip(mean_prob, 0.01, 0.99)
    # Uncertainty = distance from 0.5 (inverted: close to 0.5 = uncertain)
    uncertainty = 1.0 - np.abs(mean_prob - 0.5)

    data = selective_prediction_curve(y_true, mean_prob, uncertainty,
                                      n_points=10)
    # Accuracy at lowest coverage should be >= accuracy at full coverage
    assert data["accuracy"][-1] >= data["accuracy"][0] - 0.05


def test_uncertainty_error_auroc_perfect():
    """If uncertainty perfectly predicts errors, AUROC should be 1.0."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1])  # errors at index 1, 3
    uncertainty = np.array([0.1, 0.9, 0.1, 0.9, 0.1, 0.1])
    auroc = uncertainty_error_auroc(y_true, y_pred, uncertainty)
    assert auroc == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Integration: mc_dropout_predict output shape
# ---------------------------------------------------------------------------

def test_mc_dropout_predict_output_shape():
    model = build_model_with_dropout("resnet18", in_chans=3, pretrained=False,
                                     dropout_rate=0.2)
    device = torch.device("cpu")
    model.to(device)

    # Synthetic dataloader (2-tuple)
    xs = torch.randn(8, 3, 224, 224)
    ys = torch.randint(0, 2, (8, 1)).float()
    ds = torch.utils.data.TensorDataset(xs, ys)
    loader = torch.utils.data.DataLoader(ds, batch_size=4)

    result = mc_dropout_predict(model, loader, device, n_forward=3,
                                amp_enabled=False)
    assert result["y_true"].shape == (8,)
    assert result["probs_matrix"].shape == (8, 3)
    assert result["mean_prob"].shape == (8,)
    assert result["std_prob"].shape == (8,)
    assert result["predictive_entropy"].shape == (8,)
    assert result["mutual_information"].shape == (8,)
    assert result["case_ids"] is None


# ---------------------------------------------------------------------------
# Integration: ensemble with identical models
# ---------------------------------------------------------------------------

def test_ensemble_identical_models():
    """Two copies of the same checkpoint should give std ~= 0."""
    from miqa.models import build_model
    model = build_model("resnet18", in_chans=3, pretrained=False)

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "ckpt.pt"
        torch.save({"model": model.state_dict()}, ckpt_path)

        xs = torch.randn(4, 3, 224, 224)
        ys = torch.randint(0, 2, (4, 1)).float()
        ds = torch.utils.data.TensorDataset(xs, ys)
        loader = torch.utils.data.DataLoader(ds, batch_size=4)

        result = ensemble_predict(
            checkpoints=[str(ckpt_path), str(ckpt_path)],
            model_name="resnet18",
            in_chans=3,
            loader=loader,
            device=torch.device("cpu"),
        )

    assert np.allclose(result["std_prob"], 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Plotting (test: just check files are created, no visual check)
# ---------------------------------------------------------------------------

def test_plot_selective_prediction():
    data = {
        "coverage": [1.0, 0.8, 0.6, 0.4],
        "accuracy": [0.75, 0.80, 0.88, 0.95],
        "auroc": [0.80, 0.85, None, None],
        "n_retained": [100, 80, 60, 40],
    }
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "sel_pred.png"
        plot_selective_prediction(data, str(out))
        assert out.exists()


def test_plot_uncertainty_histogram():
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 2, 50)
    y_pred = rng.randint(0, 2, 50)
    uncertainty = rng.rand(50)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "hist.png"
        plot_uncertainty_histogram(y_true, y_pred, uncertainty, str(out))
        assert out.exists()
