"""Model construction: ResNet18/50, DenseNet121, EfficientNet-B0.

build_model:    full classification model (backbone + FC to 1)
build_backbone: backbone only (FC to Identity), returns (backbone, feat_dim)
"""

import torch.nn as nn
from torchvision import models


def _adapt_first_conv(conv: nn.Conv2d, in_chans: int, pretrained: bool):
    """Replace a pretrained 3-channel conv with an in_chans-channel conv,
    tiling averaged pretrained weights when available."""
    new_conv = nn.Conv2d(
        in_chans, conv.out_channels,
        kernel_size=conv.kernel_size, stride=conv.stride,
        padding=conv.padding, bias=(conv.bias is not None),
    )
    if pretrained and conv.weight.data.shape[1] == 3:
        w = conv.weight.data
        new_conv.weight.data = (w.mean(dim=1, keepdim=True)
                                .expand(-1, in_chans, -1, -1) / in_chans * 3)
    return new_conv


def build_model(name: str = "resnet18", in_chans: int = 3,
                pretrained: bool = True):
    if name == "resnet18":
        m = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.conv1 = _adapt_first_conv(m.conv1, in_chans, pretrained)
        m.fc = nn.Linear(m.fc.in_features, 1)
    elif name == "resnet50":
        m = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.conv1 = _adapt_first_conv(m.conv1, in_chans, pretrained)
        m.fc = nn.Linear(m.fc.in_features, 1)
    elif name == "densenet121":
        m = models.densenet121(
            weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        m.features.conv0 = _adapt_first_conv(m.features.conv0, in_chans, pretrained)
        m.classifier = nn.Linear(m.classifier.in_features, 1)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.features[0][0] = _adapt_first_conv(m.features[0][0], in_chans, pretrained)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    else:
        raise ValueError(f"Unknown model: {name}")
    return m


def build_backbone(name: str = "resnet18", in_chans: int = 3,
                   pretrained: bool = True):
    """Build a backbone without the final FC layer.

    Returns (backbone, feat_dim) where backbone.fc is Identity().
    """
    if name == "resnet18":
        m = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.conv1 = _adapt_first_conv(m.conv1, in_chans, pretrained)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
    elif name == "resnet50":
        m = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.conv1 = _adapt_first_conv(m.conv1, in_chans, pretrained)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
    elif name == "densenet121":
        m = models.densenet121(
            weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        m.features.conv0 = _adapt_first_conv(m.features.conv0, in_chans, pretrained)
        feat_dim = m.classifier.in_features
        m.classifier = nn.Identity()
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.features[0][0] = _adapt_first_conv(m.features[0][0], in_chans, pretrained)
        feat_dim = m.classifier[1].in_features
        m.classifier = nn.Identity()
    else:
        raise ValueError(f"Unknown backbone: {name}")
    return m, feat_dim
