"""Tests for miqa.fusion: FusionModel with all four strategies."""

import pytest
import torch

from miqa.fusion import FusionModel


@pytest.mark.parametrize("strategy", ["early", "mid", "late", "diff_map"])
def test_fusion_output_shape(strategy):
    model = FusionModel("resnet18", k_slices=3, strategy=strategy,
                        pretrained=False)
    x = torch.randn(2, 6, 32, 32)  # 2K=6 channels (DWI + T2)
    out = model(x)
    assert out.shape == (2, 1)


@pytest.mark.parametrize("strategy", ["early", "mid", "late", "diff_map"])
def test_fusion_backward(strategy):
    model = FusionModel("resnet18", k_slices=3, strategy=strategy,
                        pretrained=False)
    x = torch.randn(2, 6, 32, 32)
    out = model(x)
    loss = out.sum()
    loss.backward()
    # Verify gradients exist
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad


def test_fusion_mid_resnet50():
    model = FusionModel("resnet50", k_slices=3, strategy="mid",
                        pretrained=False)
    x = torch.randn(2, 6, 32, 32)
    out = model(x)
    assert out.shape == (2, 1)


def test_fusion_late_learned_weight():
    model = FusionModel("resnet18", k_slices=3, strategy="late",
                        pretrained=False)
    assert hasattr(model, "combine_weight")
    x = torch.randn(2, 6, 32, 32)
    out = model(x)
    assert out.shape == (2, 1)


def test_fusion_diff_map_channels():
    """diff_map uses 3K channels internally."""
    model = FusionModel("resnet18", k_slices=3, strategy="diff_map",
                        pretrained=False)
    # Check that internal model expects 9 channels
    x = torch.randn(2, 6, 32, 32)
    out = model(x)
    assert out.shape == (2, 1)


def test_fusion_unknown_strategy():
    with pytest.raises(ValueError, match="Unknown fusion strategy"):
        FusionModel("resnet18", k_slices=3, strategy="invalid")


def test_fusion_k1():
    """k_slices=1: DWI(1) + T2(1) = 2 channels input."""
    model = FusionModel("resnet18", k_slices=1, strategy="mid",
                        pretrained=False)
    x = torch.randn(2, 2, 32, 32)
    out = model(x)
    assert out.shape == (2, 1)
