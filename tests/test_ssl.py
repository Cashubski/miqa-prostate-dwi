"""Tests for miqa.ssl: SimCLR, BYOL, augmentations, dataset, extraction."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from miqa.ssl import (
    SSLAugmentation,
    SSLSliceDataset,
    SimCLRModel,
    SimCLRProjector,
    NTXentLoss,
    BYOLModel,
    BYOLProjector,
    byol_loss,
    extract_backbone_state_dict,
)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def test_ssl_augmentation_shape():
    aug = SSLAugmentation(img_size=64, strength="strong")
    x = torch.rand(3, 64, 64)
    out = aug(x)
    assert out.shape == (3, 64, 64)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_ssl_augmentation_weak():
    aug = SSLAugmentation(img_size=32, strength="weak")
    x = torch.rand(3, 48, 48)
    out = aug(x)
    assert out.shape == (3, 32, 32)


# ---------------------------------------------------------------------------
# SimCLR
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch", ["resnet18", "resnet50",
                                   "densenet121", "efficientnet_b0"])
def test_simclr_output_shapes(arch):
    model = SimCLRModel(arch, in_chans=3, pretrained=False,
                        proj_hidden=64, proj_dim=32)
    x = torch.randn(2, 3, 32, 32)
    h, z = model(x)
    assert h.shape[0] == 2
    assert z.shape == (2, 32)


def test_simclr_encode():
    model = SimCLRModel("resnet18", in_chans=3, pretrained=False)
    x = torch.randn(2, 3, 32, 32)
    h = model.encode(x)
    assert h.shape[0] == 2
    assert h.ndim == 2


def test_simclr_six_channel():
    model = SimCLRModel("resnet18", in_chans=6, pretrained=False,
                        proj_hidden=64, proj_dim=32)
    x = torch.randn(2, 6, 32, 32)
    h, z = model(x)
    assert z.shape == (2, 32)


def test_ntxent_loss():
    criterion = NTXentLoss(temperature=0.5)
    z1 = torch.randn(8, 32)
    z2 = torch.randn(8, 32)
    loss = criterion(z1, z2)
    assert loss.shape == ()
    assert loss.item() > 0


def test_ntxent_loss_identical_pairs():
    """Identical pairs should give lower loss than random."""
    criterion = NTXentLoss(temperature=0.5)
    z = torch.randn(8, 32)
    loss_identical = criterion(z, z)
    z2 = torch.randn(8, 32)
    loss_random = criterion(z, z2)
    # Identical should be very low (near 0)
    assert loss_identical.item() < loss_random.item()


# ---------------------------------------------------------------------------
# BYOL
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch", ["resnet18", "resnet50"])
def test_byol_output_shapes(arch):
    model = BYOLModel(arch, in_chans=3, pretrained=False,
                      proj_hidden=64, proj_dim=32, pred_hidden=64)
    x1 = torch.randn(2, 3, 32, 32)
    x2 = torch.randn(2, 3, 32, 32)
    p1, tz2, p2, tz1 = model(x1, x2)
    assert p1.shape == (2, 32)
    assert tz2.shape == (2, 32)
    assert p2.shape == (2, 32)
    assert tz1.shape == (2, 32)


def test_byol_encode():
    model = BYOLModel("resnet18", in_chans=3, pretrained=False,
                      proj_hidden=64, proj_dim=32, pred_hidden=64)
    x = torch.randn(2, 3, 32, 32)
    h = model.encode(x)
    assert h.ndim == 2
    assert h.shape[0] == 2


def test_byol_ema_update():
    model = BYOLModel("resnet18", in_chans=3, pretrained=False,
                      proj_hidden=64, proj_dim=32, pred_hidden=64,
                      ema_decay=0.99)
    # Get initial target param
    tp_before = list(model.target_backbone.parameters())[0].clone()
    # Do a forward + backward to update online params
    x1 = torch.randn(2, 3, 32, 32)
    x2 = torch.randn(2, 3, 32, 32)
    p1, tz2, p2, tz1 = model(x1, x2)
    loss = byol_loss(p1, tz2, p2, tz1)
    loss.backward()
    # Manually update online params (simulate optimizer step)
    with torch.no_grad():
        for p in model.online_backbone.parameters():
            p.data -= 0.1 * p.grad
    # Now EMA update
    model.update_target()
    tp_after = list(model.target_backbone.parameters())[0]
    # Target should have changed
    assert not torch.allclose(tp_before, tp_after)


def test_byol_loss_value():
    p1 = torch.randn(4, 32)
    tz2 = torch.randn(4, 32)
    p2 = torch.randn(4, 32)
    tz1 = torch.randn(4, 32)
    loss = byol_loss(p1, tz2, p2, tz1)
    assert loss.shape == ()
    assert 0.0 <= loss.item() <= 4.0  # cosine sim in [-1, 1]


def test_byol_six_channel():
    model = BYOLModel("resnet18", in_chans=6, pretrained=False,
                      proj_hidden=64, proj_dim=32, pred_hidden=64)
    x1 = torch.randn(2, 6, 32, 32)
    x2 = torch.randn(2, 6, 32, 32)
    p1, tz2, p2, tz1 = model(x1, x2)
    assert p1.shape == (2, 32)


# ---------------------------------------------------------------------------
# SSL dataset
# ---------------------------------------------------------------------------

def test_ssl_dataset_length(sample_dataframe):
    ds = SSLSliceDataset(sample_dataframe, modality="dwi",
                         k_slices=3, img_size=64, slices_per_case=5)
    assert len(ds) == 4 * 5  # 4 cases x 5 slices


def test_ssl_dataset_returns_pair(sample_dataframe):
    ds = SSLSliceDataset(sample_dataframe, modality="dwi",
                         k_slices=3, img_size=64, slices_per_case=2)
    v1, v2 = ds[0]
    assert v1.shape == (3, 64, 64)
    assert v2.shape == (3, 64, 64)
    assert v1.dtype == torch.float32


def test_ssl_dataset_views_differ(sample_dataframe):
    """Two views of the same slice should differ (augmentation)."""
    ds = SSLSliceDataset(sample_dataframe, modality="dwi",
                         k_slices=3, img_size=64, slices_per_case=2)
    v1, v2 = ds[0]
    # Very unlikely to be identical after random augmentations
    assert not torch.allclose(v1, v2)


# ---------------------------------------------------------------------------
# Backbone extraction
# ---------------------------------------------------------------------------

def test_extract_backbone_simclr(tmp_path):
    """Save a SimCLR checkpoint and extract backbone weights."""
    model = SimCLRModel("resnet18", in_chans=3, pretrained=False,
                        proj_hidden=64, proj_dim=32)
    ckpt_path = tmp_path / "simclr.pt"
    torch.save({"model": model.state_dict()}, ckpt_path)

    sd = extract_backbone_state_dict(str(ckpt_path), method="simclr")
    assert "conv1.weight" in sd
    assert not any(k.startswith("projector.") for k in sd)


def test_extract_backbone_byol(tmp_path):
    """Save a BYOL checkpoint and extract backbone weights."""
    model = BYOLModel("resnet18", in_chans=3, pretrained=False,
                      proj_hidden=64, proj_dim=32, pred_hidden=64)
    ckpt_path = tmp_path / "byol.pt"
    torch.save({"model": model.state_dict()}, ckpt_path)

    sd = extract_backbone_state_dict(str(ckpt_path), method="byol")
    assert "conv1.weight" in sd
    assert not any(k.startswith("online_projector.") for k in sd)
    assert not any(k.startswith("predictor.") for k in sd)


def test_extracted_weights_loadable(tmp_path):
    """Extracted SimCLR backbone should load into build_model."""
    from miqa.models import build_model

    model_ssl = SimCLRModel("resnet18", in_chans=3, pretrained=False,
                            proj_hidden=64, proj_dim=32)
    ckpt_path = tmp_path / "simclr.pt"
    torch.save({"model": model_ssl.state_dict()}, ckpt_path)

    sd = extract_backbone_state_dict(str(ckpt_path), method="simclr")

    # Build classification model and load backbone
    model_cls = build_model("resnet18", in_chans=3, pretrained=False)
    missing, unexpected = model_cls.load_state_dict(sd, strict=False)
    # Only fc.weight and fc.bias should be missing
    assert all("fc" in k for k in missing)
    assert len(unexpected) == 0
