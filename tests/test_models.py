"""Tests for miqa.models: output shapes, channel adaptation, error handling."""

import pytest
import torch

from miqa.models import build_model


@pytest.mark.parametrize("name", ["resnet18", "resnet50", "densenet121", "efficientnet_b0"])
def test_output_shape(name):
    model = build_model(name, in_chans=3, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1)


def test_six_channel_input():
    """Fusion modality uses 6-channel input (2*K=2*3)."""
    model = build_model("resnet18", in_chans=6, pretrained=False)
    model.eval()
    x = torch.randn(2, 6, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1)


def test_pretrained_weight_tiling():
    """Pretrained weight tiling for non-3-channel input should not crash."""
    model = build_model("resnet18", in_chans=5, pretrained=True)
    model.eval()
    x = torch.randn(1, 5, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1)


def test_unknown_model():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("vgg16")
