"""Tests for miqa.metrics: evaluate, bootstrap_ci, expected_calibration_error."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from miqa.metrics import evaluate, bootstrap_ci, expected_calibration_error


class _TinyModel(torch.nn.Module):
    """Minimal model that returns logits = 0 for every input."""
    def forward(self, x):
        return torch.zeros(x.shape[0], 1)


def _make_loader(n=20):
    x = torch.randn(n, 3, 32, 32)
    y = torch.cat([torch.zeros(n // 2, 1), torch.ones(n // 2, 1)])
    return DataLoader(TensorDataset(x, y), batch_size=8)


def test_evaluate_returns_correct_keys():
    model = _TinyModel()
    loader = _make_loader()
    result = evaluate(model, loader, device="cpu")
    expected_keys = {"auroc", "auprc", "bal_acc", "f1",
                     "sensitivity", "specificity",
                     "tp", "tn", "fp", "fn"}
    assert set(result.keys()) == expected_keys


def test_evaluate_with_loss_fn():
    """When loss_fn is provided, result includes 'loss' key."""
    model = _TinyModel()
    loader = _make_loader()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    result = evaluate(model, loader, device="cpu", loss_fn=loss_fn)
    assert "loss" in result
    assert result["loss"] >= 0.0


def test_evaluate_values_in_range():
    model = _TinyModel()
    loader = _make_loader()
    result = evaluate(model, loader, device="cpu")
    for k, v in result.items():
        if k in ("tp", "tn", "fp", "fn"):
            assert v >= 0
        else:
            assert 0.0 <= v <= 1.0


def test_evaluate_single_class_no_crash():
    """All-negative input gives auroc=0.0, no crash."""
    model = _TinyModel()
    x = torch.randn(10, 3, 32, 32)
    y = torch.zeros(10, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=5)
    result = evaluate(model, loader, device="cpu")
    assert result["auroc"] == 0.0


def test_bootstrap_ci_ordering():
    rng = np.random.RandomState(0)
    y = np.array([0]*50 + [1]*50)
    p = rng.rand(100)
    ci = bootstrap_ci(y, p, n_boot=200, seed=0)
    for metric in ["auroc", "auprc", "bal_acc", "f1"]:
        assert ci[metric]["ci_lo"] <= ci[metric]["mean"] <= ci[metric]["ci_hi"]


def test_ece_perfect_calibration():
    """If y_prob == y_true, ECE should be 0."""
    y_true = np.array([0.0, 0.0, 1.0, 1.0])
    y_prob = np.array([0.0, 0.0, 1.0, 1.0])
    ece, bins = expected_calibration_error(y_true, y_prob, n_bins=10)
    assert ece == 0.0
