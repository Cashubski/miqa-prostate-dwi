"""Multi-modal fusion models for DWI + T2 quality assessment.

Four strategies, all accepting (B, 2K, H, W) input from the existing
UCLDistortionDataset(modality="fusion"). The model splits channels
internally, so no dataset changes are required.

Strategies:
  "early":    Channel concatenation, single backbone (Ch4 default)
  "mid":      Dual backbone, feature concat, FC head
  "late":     Dual backbone, dual heads, learned combination
  "diff_map": |DWI - T2| appended, single backbone (3K channels)
"""

import torch
import torch.nn as nn

from miqa.models import build_model, build_backbone


class FusionModel(nn.Module):
    """Unified fusion model supporting four multi-modal strategies."""

    def __init__(self, backbone_name: str = "resnet18", k_slices: int = 3,
                 strategy: str = "mid", pretrained: bool = True):
        super().__init__()
        if strategy not in ("early", "mid", "late", "diff_map"):
            raise ValueError(f"Unknown fusion strategy: {strategy}")

        self.strategy = strategy
        self.k = k_slices

        if strategy == "early":
            self.model = build_model(
                backbone_name, in_chans=2 * k_slices, pretrained=pretrained)

        elif strategy == "mid":
            self.backbone_dwi, feat_dim = build_backbone(
                backbone_name, k_slices, pretrained)
            self.backbone_t2, _ = build_backbone(
                backbone_name, k_slices, pretrained)
            self.fc = nn.Linear(feat_dim * 2, 1)

        elif strategy == "late":
            self.model_dwi = build_model(
                backbone_name, in_chans=k_slices, pretrained=pretrained)
            self.model_t2 = build_model(
                backbone_name, in_chans=k_slices, pretrained=pretrained)
            self.combine_weight = nn.Parameter(torch.tensor(0.0))

        elif strategy == "diff_map":
            self.model = build_model(
                backbone_name, in_chans=3 * k_slices, pretrained=pretrained)

    def forward(self, x):
        """x: (B, 2K, H, W); first K channels = DWI, last K = T2."""
        if self.strategy == "early":
            return self.model(x)

        x_dwi = x[:, :self.k]
        x_t2 = x[:, self.k:]

        if self.strategy == "mid":
            h_dwi = self.backbone_dwi(x_dwi)
            h_t2 = self.backbone_t2(x_t2)
            return self.fc(torch.cat([h_dwi, h_t2], dim=1))

        elif self.strategy == "late":
            logit_dwi = self.model_dwi(x_dwi)
            logit_t2 = self.model_t2(x_t2)
            w = torch.sigmoid(self.combine_weight)
            return w * logit_dwi + (1 - w) * logit_t2

        elif self.strategy == "diff_map":
            diff = torch.abs(x_dwi - x_t2)
            return self.model(torch.cat([x_dwi, x_t2, diff], dim=1))
