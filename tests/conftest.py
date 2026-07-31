"""Shared test fixtures: synthetic NIfTI volumes and sample DataFrames."""

import numpy as np
import pandas as pd
import pytest
import nibabel as nib


@pytest.fixture
def synthetic_volume():
    return np.random.rand(64, 64, 20).astype(np.float32)


@pytest.fixture
def synthetic_mask():
    mask = np.zeros((64, 64, 20), dtype=np.float32)
    mask[10:50, 10:50, 5:15] = 1.0
    return mask


@pytest.fixture
def sample_dataframe(tmp_path):
    """Create 4 minimal NIfTI files and return a DataFrame."""
    rows = []
    for i in range(4):
        label = i % 2
        case_id = f"case_{i:03d}"

        vol = np.random.rand(64, 64, 20).astype(np.float32)
        mask = np.zeros((64, 64, 20), dtype=np.float32)
        mask[10:50, 10:50, 5:15] = 1.0

        dwi_path = tmp_path / f"{case_id}_dwi.nii.gz"
        t2_path = tmp_path / f"{case_id}_t2.nii.gz"
        mask_path = tmp_path / f"{case_id}_mask.nii.gz"

        affine = np.eye(4)
        nib.save(nib.Nifti1Image(vol, affine), str(dwi_path))
        nib.save(nib.Nifti1Image(vol, affine), str(t2_path))
        nib.save(nib.Nifti1Image(mask, affine), str(mask_path))

        rows.append({
            "case_id": case_id,
            "label": label,
            "b_value": "1k4" if i < 2 else "2k",
            "dwi_path": str(dwi_path),
            "t2_path": str(t2_path),
            "mask_path": str(mask_path),
        })

    return pd.DataFrame(rows)
