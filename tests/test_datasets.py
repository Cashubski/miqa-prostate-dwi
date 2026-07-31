"""Tests for miqa.datasets: shapes, return_case_id, augmentation, caching."""

import pytest
import torch

from miqa.datasets import UCLDistortionDataset


def test_dwi_shape(sample_dataframe):
    ds = UCLDistortionDataset(sample_dataframe, modality="dwi",
                              k_slices=3, img_size=64)
    x, y = ds[0]
    assert x.shape == (3, 64, 64)


def test_fusion_shape(sample_dataframe):
    ds = UCLDistortionDataset(sample_dataframe, modality="fusion",
                              k_slices=3, img_size=64)
    x, y = ds[0]
    assert x.shape == (6, 64, 64)


def test_return_case_id_true(sample_dataframe):
    ds = UCLDistortionDataset(sample_dataframe, return_case_id=True,
                              img_size=64)
    out = ds[0]
    assert len(out) == 3
    assert isinstance(out[2], str)


def test_return_case_id_false(sample_dataframe):
    ds = UCLDistortionDataset(sample_dataframe, return_case_id=False,
                              img_size=64)
    out = ds[0]
    assert len(out) == 2


def test_augmentation_preserves_shape(sample_dataframe):
    ds = UCLDistortionDataset(sample_dataframe, augment=True,
                              k_slices=3, img_size=64)
    x, y = ds[0]
    assert x.shape == (3, 64, 64)


def test_no_cache(sample_dataframe):
    """Uncached path still works."""
    ds = UCLDistortionDataset(sample_dataframe, modality="dwi",
                              k_slices=3, img_size=64,
                              cache_volumes=False)
    x, y = ds[0]
    assert x.shape == (3, 64, 64)


def test_cache_populates(sample_dataframe):
    """Cached path loads all volumes into _cache dict."""
    ds = UCLDistortionDataset(sample_dataframe, modality="dwi",
                              k_slices=3, img_size=64,
                              cache_volumes=True)
    # Should have cached mask + dwi paths = 4 + 4 = 8 unique files
    assert len(ds._cache) > 0
    x, y = ds[0]
    assert x.shape == (3, 64, 64)


def test_return_bval(sample_dataframe):
    """return_bval=True returns 3-tuple (x, y, bval_idx)."""
    bval_map = {"1k4": 0, "1k5": 0, "1k6": 1, "2k": 2}
    ds = UCLDistortionDataset(sample_dataframe, modality="dwi",
                              k_slices=3, img_size=64,
                              return_bval=True, bval_map=bval_map)
    out = ds[0]
    assert len(out) == 3
    assert out[2].dtype == torch.long


def test_return_bval_and_case_id(sample_dataframe):
    """return_bval=True + return_case_id=True returns 4-tuple."""
    bval_map = {"1k4": 0, "1k5": 0, "1k6": 1, "2k": 2}
    ds = UCLDistortionDataset(sample_dataframe, modality="dwi",
                              k_slices=3, img_size=64,
                              return_bval=True, bval_map=bval_map,
                              return_case_id=True)
    out = ds[0]
    assert len(out) == 4
    assert out[2].dtype == torch.long
    assert isinstance(out[3], str)
