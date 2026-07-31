"""Tests for miqa.multitask: MultiTaskModel, evaluate_multitask."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from miqa.multitask import (MultiTaskModel, evaluate_multitask,
                            BVAL_MAP, NUM_BVAL_CLASSES, BVAL_NAMES)


@pytest.mark.parametrize("arch", ["resnet18", "resnet50",
                                   "densenet121", "efficientnet_b0"])
def test_multitask_output_shapes(arch):
    model = MultiTaskModel(arch, in_chans=3, pretrained=False)
    x = torch.randn(2, 3, 32, 32)
    dist_logit, bval_logit = model(x)
    assert dist_logit.shape == (2, 1)
    assert bval_logit.shape == (2, NUM_BVAL_CLASSES)


def test_distortion_only_forward():
    model = MultiTaskModel("resnet18", in_chans=3, pretrained=False)
    x = torch.randn(2, 3, 32, 32)
    dist_logit = model.forward_distortion_only(x)
    assert dist_logit.shape == (2, 1)


def test_multitask_six_channel():
    model = MultiTaskModel("resnet18", in_chans=6, pretrained=False)
    x = torch.randn(2, 6, 32, 32)
    dist_logit, bval_logit = model(x)
    assert dist_logit.shape == (2, 1)
    assert bval_logit.shape == (2, NUM_BVAL_CLASSES)


def test_multitask_with_dropout():
    model = MultiTaskModel("resnet18", in_chans=3, pretrained=False,
                           dropout=0.5)
    x = torch.randn(2, 3, 32, 32)
    model.train()
    dist_logit, bval_logit = model(x)
    assert dist_logit.shape == (2, 1)


def _make_multitask_loader(n=20):
    x = torch.randn(n, 3, 32, 32)
    y_dist = torch.cat([torch.zeros(n // 2, 1), torch.ones(n // 2, 1)])
    y_bval = torch.randint(0, NUM_BVAL_CLASSES, (n,))
    return DataLoader(TensorDataset(x, y_dist, y_bval), batch_size=8)


def test_evaluate_multitask_keys():
    model = MultiTaskModel("resnet18", in_chans=3, pretrained=False)
    loader = _make_multitask_loader()
    result = evaluate_multitask(model, loader, device="cpu")
    expected = {"auroc", "auprc", "bal_acc", "f1",
                "sensitivity", "specificity",
                "tp", "tn", "fp", "fn",
                "bval_acc", "bval_bal_acc"}
    assert expected == set(result.keys())


def test_evaluate_multitask_with_losses():
    model = MultiTaskModel("resnet18", in_chans=3, pretrained=False)
    loader = _make_multitask_loader()
    dist_fn = torch.nn.BCEWithLogitsLoss()
    bval_fn = torch.nn.CrossEntropyLoss()
    result = evaluate_multitask(model, loader, device="cpu",
                                dist_loss_fn=dist_fn, bval_loss_fn=bval_fn)
    assert "dist_loss" in result
    assert "bval_loss" in result
    assert result["dist_loss"] >= 0.0
    assert result["bval_loss"] >= 0.0


def test_bval_map_consistency():
    assert len(BVAL_NAMES) == NUM_BVAL_CLASSES
    assert set(BVAL_MAP.values()) == set(range(NUM_BVAL_CLASSES))
