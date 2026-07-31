"""Self-supervised pretraining: SimCLR and BYOL for medical image slices.

SimCLR: Chen et al., "A Simple Framework for Contrastive Learning
        of Visual Representations", ICML 2020.
BYOL:   Grill et al., "Bootstrap Your Own Latent", NeurIPS 2020.

Workflow:
  1. Pretrain on unlabeled 2.5D slices: scripts/pretrain_ssl.py
  2. Fine-tune for distortion detection: scripts/train_ssl_finetune.py
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import nibabel as nib

from miqa.models import build_backbone


# ---------------------------------------------------------------------------
# SSL Augmentations (operates on tensors, no PIL dependency)
# ---------------------------------------------------------------------------

class SSLAugmentation:
    """Stochastic augmentation pipeline for contrastive learning.

    Input:  tensor of shape (C, H, W) with values in [0, 1].
    Output: augmented tensor of same shape.
    """

    def __init__(self, img_size: int = 224, strength: str = "strong"):
        self.img_size = img_size
        self.strength = strength

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        C, H, W = x.shape

        # 1. Random resized crop (most important for SimCLR)
        lo_scale = 0.5 if self.strength == "strong" else 0.7
        scale = torch.empty(1).uniform_(lo_scale, 1.0).item()
        crop_h = max(1, int(H * scale))
        crop_w = max(1, int(W * scale))
        top = int(torch.randint(0, max(H - crop_h, 1) + 1, (1,)).item())
        left = int(torch.randint(0, max(W - crop_w, 1) + 1, (1,)).item())
        x = x[:, top:top + crop_h, left:left + crop_w]

        # Resize to target
        x = F.interpolate(x.unsqueeze(0), size=(self.img_size, self.img_size),
                          mode="bilinear", align_corners=False).squeeze(0)

        # 2. Random horizontal flip
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[2])

        # 3. Random vertical flip
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[1])

        # 4. Random 90-degree rotation
        if torch.rand(1).item() < 0.3:
            k = int(torch.randint(1, 4, (1,)).item())
            x = torch.rot90(x, k, dims=[1, 2])

        # 5. Intensity jitter (brightness + contrast for grayscale)
        if torch.rand(1).item() < 0.8:
            brightness = torch.empty(1).uniform_(-0.2, 0.2).item()
            x = x + brightness
            contrast = torch.empty(1).uniform_(0.7, 1.3).item()
            x = (x - x.mean()) * contrast + x.mean()

        # 6. Gaussian blur
        if torch.rand(1).item() < 0.5:
            ks = 2 * int(torch.randint(1, 4, (1,)).item()) + 1  # 3, 5, or 7
            sigma = torch.empty(1).uniform_(0.1, 2.0).item()
            x = _gaussian_blur(x, ks, sigma)

        # 7. Gaussian noise
        if torch.rand(1).item() < 0.3:
            noise_std = torch.empty(1).uniform_(0.01, 0.05).item()
            x = x + noise_std * torch.randn_like(x)

        return x.clamp(0.0, 1.0)


def _gaussian_blur(x: torch.Tensor, kernel_size: int,
                   sigma: float) -> torch.Tensor:
    """Apply Gaussian blur to (C, H, W) tensor."""
    channels = x.shape[0]
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    k1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    k1d = k1d / k1d.sum()
    k2d = (k1d[:, None] * k1d[None, :]).expand(channels, 1, -1, -1)
    padding = kernel_size // 2
    return F.conv2d(x.unsqueeze(0), k2d.to(x.device),
                    padding=padding, groups=channels).squeeze(0)


# ---------------------------------------------------------------------------
# SSL Slice Dataset
# ---------------------------------------------------------------------------

class SSLSliceDataset(Dataset):
    """Yields (view1, view2) augmented pairs from NIfTI volumes.

    Length = n_cases * slices_per_case.  Each __getitem__ extracts a random
    k-slice stack and applies two independent augmentations.
    """

    def __init__(self, df, modality="dwi", use_mask_crop=True,
                 k_slices=3, img_size=224, slices_per_case=10,
                 aug_strength="strong", cache_volumes=True):
        self.df = df.reset_index(drop=True)
        self.modality = modality
        self.use_mask_crop = use_mask_crop
        self.k = k_slices
        self.img_size = img_size
        self.slices_per_case = slices_per_case

        self.aug1 = SSLAugmentation(img_size, strength=aug_strength)
        self.aug2 = SSLAugmentation(img_size, strength=aug_strength)

        self._cache = {}
        if cache_volumes:
            paths = set(self.df["mask_path"].tolist())
            if modality in ("dwi", "fusion"):
                paths.update(self.df["dwi_path"].tolist())
            if modality in ("t2", "fusion"):
                paths.update(self.df["t2_path"].tolist())
            for p in sorted(paths):
                self._cache[p] = nib.load(p).get_fdata().astype(np.float32)

    def __len__(self):
        return len(self.df) * self.slices_per_case

    def _load_nii(self, path):
        if path in self._cache:
            return self._cache[path]
        return nib.load(path).get_fdata().astype(np.float32)

    @staticmethod
    def _normalize(vol):
        lo = np.percentile(vol, 0.5)
        hi = np.percentile(vol, 99.5)
        vol = np.clip(vol, lo, hi)
        return (vol - lo) / (hi - lo + 1e-8)

    @staticmethod
    def _mask_bbox(mask):
        coords = np.argwhere(mask > 0)
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0) + 1
        return mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]

    def _extract_slices(self, vol_crop, center):
        nz = vol_crop.shape[2]
        half_k = self.k // 2
        slices = []
        for offset in range(-half_k, half_k + 1):
            if self.k % 2 == 0 and offset == half_k:
                continue
            si = np.clip(center + offset, 0, nz - 1)
            slices.append(vol_crop[:, :, si])
        return slices[:self.k]

    def __getitem__(self, idx):
        case_idx = idx // self.slices_per_case
        row = self.df.iloc[case_idx]

        mask = self._load_nii(row["mask_path"])

        if self.use_mask_crop:
            r0, r1, c0, c1, z0, z1 = self._mask_bbox(mask)
        else:
            r0, r1, c0, c1 = 0, mask.shape[0], 0, mask.shape[1]
            z_nz = np.argwhere(mask.sum(axis=(0, 1)) > 0).reshape(-1)
            z0, z1 = ((int(z_nz.min()), int(z_nz.max()) + 1)
                      if len(z_nz) > 0 else (0, mask.shape[2]))

        if self.modality == "fusion":
            dwi = self._load_nii(row["dwi_path"])
            t2 = self._load_nii(row["t2_path"])
            dwi_crop = self._normalize(dwi[r0:r1, c0:c1, z0:z1])
            t2_crop = self._normalize(t2[r0:r1, c0:c1, z0:z1])
            nz = dwi_crop.shape[2]
        else:
            path_key = "t2_path" if self.modality == "t2" else "dwi_path"
            vol = self._load_nii(row[path_key])
            vol_crop = self._normalize(vol[r0:r1, c0:c1, z0:z1])
            nz = vol_crop.shape[2]

        # Random center slice
        half_k = self.k // 2
        lo_z = half_k
        hi_z = max(nz - 1 - half_k, half_k)
        center = int(torch.randint(lo_z, hi_z + 1, (1,)).item())

        # Gather slices
        if self.modality == "fusion":
            stack = np.stack(
                self._extract_slices(dwi_crop, center)
                + self._extract_slices(t2_crop, center), axis=0)
        else:
            stack = np.stack(self._extract_slices(vol_crop, center), axis=0)

        # Pad to square
        h, w = stack.shape[1], stack.shape[2]
        side = max(h, w)
        if h != side or w != side:
            padded = np.zeros((stack.shape[0], side, side), dtype=np.float32)
            ph, pw = (side - h) // 2, (side - w) // 2
            padded[:, ph:ph + h, pw:pw + w] = stack
            stack = padded

        # Resize
        x = torch.from_numpy(stack).float()
        x = F.interpolate(x.unsqueeze(0), size=(self.img_size, self.img_size),
                          mode="bilinear", align_corners=False).squeeze(0)

        # Two independent augmentations
        return self.aug1(x), self.aug2(x)


# ---------------------------------------------------------------------------
# SimCLR
# ---------------------------------------------------------------------------

class SimCLRProjector(nn.Module):
    """MLP projection head: feat_dim to hidden to out_dim."""

    def __init__(self, feat_dim, hidden_dim=2048, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimCLRModel(nn.Module):
    """SimCLR: backbone + projection head."""

    def __init__(self, backbone_name="resnet18", in_chans=3,
                 pretrained=False, proj_hidden=2048, proj_dim=128):
        super().__init__()
        self.backbone, self.feat_dim = build_backbone(
            backbone_name, in_chans, pretrained)
        self.projector = SimCLRProjector(self.feat_dim, proj_hidden, proj_dim)

    def forward(self, x):
        h = self.backbone(x)
        z = self.projector(h)
        return h, z

    def encode(self, x):
        """Return backbone features only (for downstream)."""
        return self.backbone(x)


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss (SimCLR).

    Given 2N embeddings from N pairs, computes contrastive loss.
    """

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        N = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)
        z = F.normalize(z, dim=1)

        sim = torch.mm(z, z.t()) / self.temperature
        mask = torch.eye(2 * N, device=z.device, dtype=torch.bool)
        sim.masked_fill_(mask, torch.finfo(sim.dtype).min)

        labels = torch.cat([
            torch.arange(N, 2 * N, device=z.device),
            torch.arange(0, N, device=z.device),
        ])
        return F.cross_entropy(sim, labels)


# ---------------------------------------------------------------------------
# BYOL
# ---------------------------------------------------------------------------

class BYOLProjector(nn.Module):
    """BYOL projection / prediction MLP."""

    def __init__(self, in_dim, hidden_dim=4096, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class BYOLModel(nn.Module):
    """Bootstrap Your Own Latent.

    Online network: backbone + projector + predictor
    Target network: EMA of backbone + projector (no gradient)
    """

    def __init__(self, backbone_name="resnet18", in_chans=3,
                 pretrained=False, proj_hidden=4096, proj_dim=256,
                 pred_hidden=4096, ema_decay=0.996):
        super().__init__()
        self.ema_decay = ema_decay

        self.online_backbone, self.feat_dim = build_backbone(
            backbone_name, in_chans, pretrained)
        self.online_projector = BYOLProjector(
            self.feat_dim, proj_hidden, proj_dim)
        self.predictor = BYOLProjector(proj_dim, pred_hidden, proj_dim)

        # Target (EMA copy, no gradients)
        self.target_backbone = copy.deepcopy(self.online_backbone)
        self.target_projector = copy.deepcopy(self.online_projector)
        for p in self.target_backbone.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target(self):
        """Exponential moving average update of target network."""
        tau = self.ema_decay
        for op, tp in zip(self.online_backbone.parameters(),
                          self.target_backbone.parameters()):
            tp.data = tau * tp.data + (1 - tau) * op.data
        for op, tp in zip(self.online_projector.parameters(),
                          self.target_projector.parameters()):
            tp.data = tau * tp.data + (1 - tau) * op.data

    def forward(self, x1, x2):
        h1 = self.online_backbone(x1)
        h2 = self.online_backbone(x2)
        z1 = self.online_projector(h1)
        z2 = self.online_projector(h2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        with torch.no_grad():
            tz1 = self.target_projector(self.target_backbone(x1)).detach()
            tz2 = self.target_projector(self.target_backbone(x2)).detach()

        return p1, tz2, p2, tz1

    def encode(self, x):
        """Return online backbone features (for downstream)."""
        return self.online_backbone(x)


def byol_loss(p1, tz2, p2, tz1):
    """BYOL loss: negative cosine similarity, symmetrized."""
    def _cos(p, z):
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return 2 - 2 * (p * z).sum(dim=1).mean()
    return (_cos(p1, tz2) + _cos(p2, tz1)) / 2


# ---------------------------------------------------------------------------
# Utility: extract backbone weights for fine-tuning
# ---------------------------------------------------------------------------

def extract_backbone_state_dict(ssl_ckpt_path, method="simclr"):
    """Load an SSL checkpoint and return the backbone state_dict.

    Returns a dict compatible with models built by build_model /
    build_backbone (same key names: conv1.weight, layer1.*, etc.).
    """
    ckpt = torch.load(ssl_ckpt_path, map_location="cpu")
    model_state = ckpt.get("model", ckpt)

    prefix = {"simclr": "backbone.", "byol": "online_backbone."}
    if method not in prefix:
        raise ValueError(f"Unknown SSL method: {method}")

    pfx = prefix[method]
    return {k[len(pfx):]: v for k, v in model_state.items()
            if k.startswith(pfx)}
