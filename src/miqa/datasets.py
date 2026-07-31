"""Unified 2.5D dataset for UCL prostate MRI distortion detection.

Training:  UCLDistortionDataset(df, augment=True)          returns (x, y)
Inference: UCLDistortionDataset(df, return_case_id=True)   returns (x, y, case_id)

Volume caching (on by default): all NIfTI volumes are loaded into RAM
during __init__, eliminating per-epoch disk I/O. For 245 cases this
uses ~40 MB, negligible on any training machine.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import nibabel as nib


class UCLDistortionDataset(Dataset):
    """2.5D dataset: K adjacent axial slices from mask-cropped NIfTI volumes."""

    def __init__(self, df: pd.DataFrame, modality: str = "dwi",
                 use_mask_crop: bool = True, k_slices: int = 3,
                 img_size: int = 224, augment: bool = False,
                 return_case_id: bool = False,
                 return_bval: bool = False,
                 bval_map: dict = None,
                 cache_volumes: bool = True):
        self.df = df.reset_index(drop=True)
        self.modality = modality
        self.use_mask_crop = use_mask_crop
        self.k = k_slices
        self.img_size = img_size
        self.augment = augment
        self.return_case_id = return_case_id
        self.return_bval = return_bval
        self.bval_map = bval_map or {}

        # Pre-cache all volumes into RAM
        self._cache = {}
        if cache_volumes:
            paths = set(self.df["mask_path"].tolist())
            if modality in ("dwi", "fusion"):
                paths.update(self.df["dwi_path"].tolist())
            if modality in ("t2", "fusion"):
                paths.update(self.df["t2_path"].tolist())
            print(f"Caching {len(paths)} NIfTI volumes into RAM ... ",
                  end="", flush=True)
            for p in sorted(paths):
                self._cache[p] = nib.load(p).get_fdata().astype(np.float32)
            mb = sum(v.nbytes for v in self._cache.values()) / 1e6
            print(f"done ({mb:.0f} MB)")

    def __len__(self):
        return len(self.df)

    def _load_nii(self, path: str) -> np.ndarray:
        if path in self._cache:
            return self._cache[path]
        return nib.load(path).get_fdata().astype(np.float32)

    @staticmethod
    def _mask_bbox(mask: np.ndarray):
        """Return (row_min, row_max, col_min, col_max, z_min, z_max)."""
        coords = np.argwhere(mask > 0)
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0) + 1
        return mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]

    @staticmethod
    def _normalize(vol: np.ndarray) -> np.ndarray:
        """Percentile clipping (0.5-99.5) then scale to [0,1]."""
        lo = np.percentile(vol, 0.5)
        hi = np.percentile(vol, 99.5)
        vol = np.clip(vol, lo, hi)
        vol = (vol - lo) / (hi - lo + 1e-8)
        return vol

    def _extract_slices(self, vol_crop, center):
        """Extract K adjacent slices around center from a cropped volume."""
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
        row = self.df.iloc[idx]
        label = int(row["label"])

        # Load mask for bbox
        mask = self._load_nii(row["mask_path"])

        # Compute crop bounds once (shared for fusion)
        if self.use_mask_crop:
            r0, r1, c0, c1, z0, z1 = self._mask_bbox(mask)
        else:
            r0, r1, c0, c1 = 0, mask.shape[0], 0, mask.shape[1]
            z_nz = np.argwhere(mask.sum(axis=(0, 1)) > 0).reshape(-1)
            z0, z1 = (int(z_nz.min()), int(z_nz.max()) + 1) if len(z_nz) > 0 else (0, mask.shape[2])

        if self.modality == "fusion":
            dwi = self._load_nii(row["dwi_path"])
            t2 = self._load_nii(row["t2_path"])
            dwi_crop = self._normalize(dwi[r0:r1, c0:c1, z0:z1])
            t2_crop = self._normalize(t2[r0:r1, c0:c1, z0:z1])
            nz = dwi_crop.shape[2]
        else:
            if self.modality == "t2":
                vol = self._load_nii(row["t2_path"])
            else:
                vol = self._load_nii(row["dwi_path"])
            vol_crop = self._normalize(vol[r0:r1, c0:c1, z0:z1])
            nz = vol_crop.shape[2]

        # Select center slice index in z
        half_k = self.k // 2
        if self.augment:
            lo_z = half_k
            hi_z = max(nz - 1 - half_k, half_k)
            center = int(torch.randint(lo_z, hi_z + 1, (1,)).item())
        else:
            center = nz // 2

        # Gather slices
        if self.modality == "fusion":
            dwi_slices = self._extract_slices(dwi_crop, center)
            t2_slices = self._extract_slices(t2_crop, center)
            stack = np.stack(dwi_slices + t2_slices, axis=0)  # (2*K, H, W)
        else:
            slices = self._extract_slices(vol_crop, center)
            stack = np.stack(slices, axis=0)  # (K, H, W)

        # Pad to square
        h, w = stack.shape[1], stack.shape[2]
        side = max(h, w)
        if h != side or w != side:
            padded = np.zeros((stack.shape[0], side, side), dtype=np.float32)
            ph = (side - h) // 2
            pw = (side - w) // 2
            padded[:, ph:ph+h, pw:pw+w] = stack
            stack = padded

        # To tensor and resize
        x = torch.from_numpy(stack).float()
        x = x.unsqueeze(0)
        x = F.interpolate(x, size=(self.img_size, self.img_size),
                          mode="bilinear", align_corners=False)
        x = x.squeeze(0)

        # Augmentation
        if self.augment:
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[2])
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[1])
            if torch.rand(1).item() < 0.3:
                x = (x + 0.02 * torch.randn_like(x)).clamp(0.0, 1.0)
            if torch.rand(1).item() < 0.3:
                k = int(torch.randint(1, 4, (1,)).item())
                x = torch.rot90(x, k, dims=[1, 2])

        y = torch.tensor([label], dtype=torch.float32)

        if self.return_bval:
            bval_str = str(row.get("b_value", ""))
            bval_idx = self.bval_map.get(bval_str, 0)
            bval_tensor = torch.tensor(bval_idx, dtype=torch.long)
            if self.return_case_id:
                return x, y, bval_tensor, str(row["case_id"])
            return x, y, bval_tensor

        if self.return_case_id:
            return x, y, str(row["case_id"])
        return x, y
