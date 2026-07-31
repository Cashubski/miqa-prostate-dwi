#!/usr/bin/env python3
"""Build index_full.csv from raw distorted/undistorted directories.

Scans images(distorted)/ and images(undistorted)/ to produce a 7-column CSV
with one row per acquisition:

  case_id, patient_id, b_value, label, dwi_path, t2_path, mask_path

Usage:
  python scripts/build_ucl_index_full.py \
      --data_root dataset/ \
      --out_csv dataset/index_full.csv

  # With masks:
  python scripts/build_ucl_index_full.py \
      --data_root dataset/ \
      --out_csv dataset/index_full.csv \
      --mask_dir dataset/prostate_mask
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


def parse_filename(fname: str):
    """Extract patient_id, b_value, and index from a DWI filename.

    Expected format: PatientXXX_study_0__bval-idx__NNNNNN.nii.gz
    e.g. Patient002215610_study_0__1k6-0__000002.nii.gz

    Returns (patient_id, b_value, acq_index) or None if unparseable.
    """
    stem = fname.replace(".nii.gz", "").replace(".nii", "")
    # Split on double-underscore
    parts = stem.split("__")
    if len(parts) < 2:
        return None
    patient_id = parts[0]  # e.g. Patient002215610_study_0
    bval_idx = parts[1]    # e.g. 1k6-0
    # Split b-value and index
    m = re.match(r"^(.+)-(\d+)$", bval_idx)
    if m:
        b_value = m.group(1)
        acq_idx = m.group(2)
    else:
        b_value = bval_idx
        acq_idx = "0"
    case_id = f"{patient_id}__{b_value}-{acq_idx}"
    return patient_id, b_value, acq_idx, case_id


def scan_directory(data_root: Path, subdir: str, label: int):
    """Scan a distorted or undistorted directory for DWI/T2 pairs.

    Args:
        data_root: Root data directory
        subdir: 'images(distorted)' or 'images(undistorted)'
        label: 1 for distorted, 0 for undistorted

    Returns list of row dicts.
    """
    base = data_root / subdir
    dwi_dir = base / "dwi"

    # T2 capitalization differs: distorted uses 't2', undistorted uses 'T2'
    # Must detect the ACTUAL name on disk (macOS is case-insensitive, Linux is not)
    t2_dir = None
    for entry in base.iterdir():
        if entry.is_dir() and entry.name.lower() == "t2":
            t2_dir = entry
            break
    if t2_dir is None:
        print(f"WARNING: No t2/T2 directory found in {base}")

    if not dwi_dir.is_dir():
        print(f"WARNING: No dwi directory found in {base}")
        return []

    # Collect DWI files
    dwi_files = sorted(f for f in dwi_dir.iterdir()
                       if f.name.endswith((".nii.gz", ".nii")))
    # Build T2 lookup
    t2_lookup = {}
    if t2_dir and t2_dir.is_dir():
        for f in t2_dir.iterdir():
            if f.name.endswith((".nii.gz", ".nii")):
                t2_lookup[f.name] = f

    rows = []
    for dwi_path in dwi_files:
        parsed = parse_filename(dwi_path.name)
        if parsed is None:
            print(f"WARNING: Could not parse filename: {dwi_path.name}")
            continue

        patient_id, b_value, acq_idx, case_id = parsed

        # Find matching T2
        t2_path = t2_lookup.get(dwi_path.name, "")
        if t2_path:
            t2_path = str(t2_path)
        else:
            t2_path = ""

        rows.append({
            "case_id": case_id,
            "patient_id": patient_id,
            "b_value": b_value,
            "label": label,
            "dwi_path": str(dwi_path),
            "t2_path": t2_path,
            "mask_path": "",  # filled in later if --mask_dir provided
        })

    # Validate DWI == T2 count
    n_dwi = len(dwi_files)
    n_t2 = len(t2_lookup)
    if n_dwi != n_t2:
        print(f"WARNING: {subdir} has {n_dwi} DWI files but {n_t2} T2 files")

    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Build index_full.csv from raw distorted/undistorted dirs")
    ap.add_argument("--data_root", type=str, required=True,
                    help="Root data directory containing images(distorted)/ "
                         "and images(undistorted)/")
    ap.add_argument("--out_csv", type=str, required=True,
                    help="Output CSV path (e.g. dataset/index_full.csv)")
    ap.add_argument("--mask_dir", type=str, default="",
                    help="Directory containing prostate masks "
                         "(PatientXXX_study_0.nii.gz). Optional.")
    args = ap.parse_args()

    data_root = Path(args.data_root)

    # Scan both directories
    rows = []
    rows.extend(scan_directory(data_root, "images(distorted)", label=1))
    rows.extend(scan_directory(data_root, "images(undistorted)", label=0))

    if not rows:
        print("ERROR: No acquisitions found. Check --data_root path.")
        sys.exit(1)

    df = pd.DataFrame(rows)

    # Validate no duplicate case_ids
    dupes = df[df["case_id"].duplicated(keep=False)]
    if len(dupes) > 0:
        print(f"ERROR: {len(dupes)} duplicate case_ids found:")
        print(dupes[["case_id", "label", "dwi_path"]].to_string(index=False))
        sys.exit(1)

    # Fill mask paths if --mask_dir provided
    if args.mask_dir:
        mask_dir = Path(args.mask_dir)
        if not mask_dir.is_dir():
            print(f"WARNING: --mask_dir {mask_dir} does not exist")
        else:
            mask_lookup = {}
            for f in mask_dir.iterdir():
                if f.name.endswith((".nii.gz", ".nii")):
                    # Key is the patient_id (filename without extension)
                    key = f.name.replace(".nii.gz", "").replace(".nii", "")
                    mask_lookup[key] = str(f)

            n_matched = 0
            for idx, row in df.iterrows():
                mask = mask_lookup.get(row["patient_id"], "")
                if mask:
                    df.at[idx, "mask_path"] = mask
                    n_matched += 1

            n_patients = df["patient_id"].nunique()
            print(f"Masks: matched {n_matched}/{len(df)} acquisitions "
                  f"({len(mask_lookup)} masks for {n_patients} patients)")
    else:
        print("WARNING: No --mask_dir provided. mask_path column will be empty.")

    # Sort by case_id for determinism
    df = df.sort_values("case_id").reset_index(drop=True)

    # Summary
    n_distorted = (df["label"] == 1).sum()
    n_undistorted = (df["label"] == 0).sum()
    n_patients = df["patient_id"].nunique()
    n_bvalues = df["b_value"].nunique()

    print(f"\nDataset summary:")
    print(f"  Total acquisitions: {len(df)}")
    print(f"  Distorted (label=1): {n_distorted}")
    print(f"  Undistorted (label=0): {n_undistorted}")
    print(f"  Unique patients: {n_patients}")
    print(f"  Unique b-values: {n_bvalues} ({sorted(df['b_value'].unique())})")

    # Check for patients appearing in both categories
    dist_patients = set(df.loc[df["label"] == 1, "patient_id"])
    undist_patients = set(df.loc[df["label"] == 0, "patient_id"])
    overlap = dist_patients & undist_patients
    if overlap:
        print(f"  Patients in BOTH distorted AND undistorted: {len(overlap)}")
        print(f"    (These patients have different b-values with different labels)")

    # Acquisitions per patient distribution
    acq_per_patient = df.groupby("patient_id").size()
    print(f"  Acquisitions per patient: "
          f"min={acq_per_patient.min()}, "
          f"max={acq_per_patient.max()}, "
          f"mean={acq_per_patient.mean():.1f}")

    # Write CSV
    df.to_csv(args.out_csv, index=False)
    print(f"\nWrote {len(df)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
