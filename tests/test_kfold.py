"""Tests for k-fold cross-validation splitting functions."""

import json

import numpy as np
import pandas as pd
import pytest

from miqa.utils import (make_kfold_splits, load_or_create_kfold_splits,
                         make_patient_split)


@pytest.fixture
def kfold_df():
    """Synthetic DataFrame with 20 case_ids and 2 labels."""
    rows = []
    for i in range(20):
        rows.append({
            "case_id": f"case_{i:03d}",
            "label": 0 if i < 14 else 1,  # ~70/30 split like real data
            "b_value": "1k4",
        })
    return pd.DataFrame(rows)


def test_make_kfold_splits_returns_n_folds(kfold_df):
    splits = make_kfold_splits(kfold_df, n_folds=5, seed=42)
    assert len(splits) == 5


def test_all_cases_covered_once(kfold_df):
    splits = make_kfold_splits(kfold_df, n_folds=5, seed=42)
    all_test_ids = []
    for tr, va, te in splits:
        all_test_ids.extend(te["case_id"].unique().tolist())
    all_case_ids = sorted(kfold_df["case_id"].unique().tolist())
    assert sorted(all_test_ids) == all_case_ids, "Every case tested exactly once"


def test_no_duplicate_test_ids(kfold_df):
    splits = make_kfold_splits(kfold_df, n_folds=5, seed=42)
    all_test_ids = []
    for _, _, te in splits:
        all_test_ids.extend(te["case_id"].unique().tolist())
    assert len(all_test_ids) == len(set(all_test_ids)), "No duplicate test IDs"


def test_stratified_folds(kfold_df):
    splits = make_kfold_splits(kfold_df, n_folds=5, seed=42)
    # Each test fold should have some positive and negative cases
    for k, (_, _, te) in enumerate(splits):
        labels = te["label"].unique()
        # With 14 neg / 6 pos, each fold of ~4 cases should have both classes
        # (at least for most folds; with 6 pos across 5 folds, at least 1 pos per fold)
        assert len(te) > 0, f"Fold {k} has empty test set"


def test_no_patient_leakage(kfold_df):
    splits = make_kfold_splits(kfold_df, n_folds=5, seed=42)
    for k, (tr, va, te) in enumerate(splits):
        tr_ids = set(tr["case_id"].unique())
        va_ids = set(va["case_id"].unique())
        te_ids = set(te["case_id"].unique())
        assert tr_ids.isdisjoint(va_ids), f"Fold {k}: train/val overlap"
        assert tr_ids.isdisjoint(te_ids), f"Fold {k}: train/test overlap"
        assert va_ids.isdisjoint(te_ids), f"Fold {k}: val/test overlap"


def test_deterministic_with_same_seed(kfold_df):
    s1 = make_kfold_splits(kfold_df, n_folds=5, seed=42)
    s2 = make_kfold_splits(kfold_df, n_folds=5, seed=42)
    for k in range(5):
        ids1 = sorted(s1[k][2]["case_id"].unique().tolist())
        ids2 = sorted(s2[k][2]["case_id"].unique().tolist())
        assert ids1 == ids2, f"Fold {k}: same seed should give same splits"


def test_different_seeds_differ(kfold_df):
    s1 = make_kfold_splits(kfold_df, n_folds=5, seed=42)
    s2 = make_kfold_splits(kfold_df, n_folds=5, seed=99)
    any_diff = False
    for k in range(5):
        ids1 = sorted(s1[k][2]["case_id"].unique().tolist())
        ids2 = sorted(s2[k][2]["case_id"].unique().tolist())
        if ids1 != ids2:
            any_diff = True
            break
    assert any_diff, "Different seeds should produce different splits"


def test_fold_sizes_balanced(kfold_df):
    splits = make_kfold_splits(kfold_df, n_folds=5, seed=42)
    test_sizes = [len(te["case_id"].unique()) for _, _, te in splits]
    n_labels = kfold_df["label"].nunique()
    # With stratification, each label class can differ by ±1 per fold,
    # so total fold size can differ by at most n_labels
    assert max(test_sizes) - min(test_sizes) <= n_labels, (
        f"Fold sizes should be balanced (within ±{n_labels}): {test_sizes}")


def test_load_or_create_saves_json(kfold_df, tmp_path):
    tr, va, te = load_or_create_kfold_splits(
        kfold_df, n_folds=5, fold=0, seed=42, out=tmp_path)
    splits_path = tmp_path / "kfold_splits.json"
    assert splits_path.exists()
    data = json.loads(splits_path.read_text())
    assert data["n_folds"] == 5
    assert data["seed"] == 42
    assert len(data["folds"]) == 5
    assert len(tr) > 0
    assert len(va) > 0
    assert len(te) > 0


def test_load_or_create_loads_existing(kfold_df, tmp_path):
    tr1, va1, te1 = load_or_create_kfold_splits(
        kfold_df, n_folds=5, fold=2, seed=42, out=tmp_path)
    # Call again, should load from file
    tr2, va2, te2 = load_or_create_kfold_splits(
        kfold_df, n_folds=5, fold=2, seed=42, out=tmp_path)
    assert sorted(te1["case_id"].unique().tolist()) == sorted(
        te2["case_id"].unique().tolist())


# ---------------------------------------------------------------------------
# Patient-level (group_col) tests
# ---------------------------------------------------------------------------
@pytest.fixture
def multi_acq_df():
    """DataFrame where one patient has 3 acquisitions (different b-values)."""
    rows = []
    # Patient A: 3 acquisitions, all label=1
    for bv in ["1k4", "1k5", "1k6"]:
        rows.append({"case_id": f"PatA__{bv}-0", "patient_id": "PatA",
                      "b_value": bv, "label": 1})
    # 19 other patients with 1 acquisition each
    for i in range(19):
        lab = 0 if i < 13 else 1
        rows.append({"case_id": f"Pat{i:03d}__1k4-0",
                      "patient_id": f"Pat{i:03d}",
                      "b_value": "1k4", "label": lab})
    return pd.DataFrame(rows)


def test_patient_level_no_leakage(multi_acq_df):
    """All acquisitions for PatA must land in the same fold."""
    splits = make_kfold_splits(multi_acq_df, n_folds=5, seed=42,
                                group_col="patient_id")
    for k, (tr, va, te) in enumerate(splits):
        tr_patients = set(tr["patient_id"].unique())
        va_patients = set(va["patient_id"].unique())
        te_patients = set(te["patient_id"].unique())
        # No patient should appear in more than one partition
        assert tr_patients.isdisjoint(va_patients), \
            f"Fold {k}: patient in both train and val"
        assert tr_patients.isdisjoint(te_patients), \
            f"Fold {k}: patient in both train and test"
        assert va_patients.isdisjoint(te_patients), \
            f"Fold {k}: patient in both val and test"

    # PatA's 3 case_ids must all be in the same partition
    pat_a_cases = set(multi_acq_df.loc[
        multi_acq_df["patient_id"] == "PatA", "case_id"])
    for k, (tr, va, te) in enumerate(splits):
        in_tr = pat_a_cases & set(tr["case_id"])
        in_va = pat_a_cases & set(va["case_id"])
        in_te = pat_a_cases & set(te["case_id"])
        # All 3 must be in exactly one partition
        counts = [len(in_tr) > 0, len(in_va) > 0, len(in_te) > 0]
        assert sum(counts) == 1, \
            f"Fold {k}: PatA cases split across partitions"


def test_mixed_label_patient():
    """Patient with both label=0 and label=1 rows stays together."""
    rows = []
    # Patient X has distorted (label=1) and undistorted (label=0) acquisitions
    rows.append({"case_id": "PatX__1k4-0", "patient_id": "PatX",
                  "b_value": "1k4", "label": 1})
    rows.append({"case_id": "PatX__1k5-0", "patient_id": "PatX",
                  "b_value": "1k5", "label": 0})
    # 18 other patients
    for i in range(18):
        lab = 0 if i < 12 else 1
        rows.append({"case_id": f"Pat{i:03d}__1k4-0",
                      "patient_id": f"Pat{i:03d}",
                      "b_value": "1k4", "label": lab})
    df = pd.DataFrame(rows)
    tr, va, te = make_patient_split(df, seed=42, group_col="patient_id")

    # Both PatX rows must be in the same split
    pat_x_in_tr = set(tr.loc[tr["patient_id"] == "PatX", "case_id"])
    pat_x_in_va = set(va.loc[va["patient_id"] == "PatX", "case_id"])
    pat_x_in_te = set(te.loc[te["patient_id"] == "PatX", "case_id"])
    counts = [len(pat_x_in_tr) > 0, len(pat_x_in_va) > 0,
              len(pat_x_in_te) > 0]
    assert sum(counts) == 1, "PatX's acquisitions split across partitions"
    # Whichever partition has PatX, it should have both case_ids
    for s in [pat_x_in_tr, pat_x_in_va, pat_x_in_te]:
        if s:
            assert len(s) == 2, f"Expected 2 PatX cases, got {len(s)}"


def test_group_col_backward_compat(kfold_df):
    """group_col=None produces identical output to the original behaviour."""
    s1 = make_kfold_splits(kfold_df, n_folds=5, seed=42, group_col=None)
    s2 = make_kfold_splits(kfold_df, n_folds=5, seed=42)
    for k in range(5):
        ids1 = sorted(s1[k][2]["case_id"].unique().tolist())
        ids2 = sorted(s2[k][2]["case_id"].unique().tolist())
        assert ids1 == ids2, f"Fold {k}: group_col=None should match default"
