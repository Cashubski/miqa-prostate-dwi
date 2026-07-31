"""Build index CSV for UCL distortion classification dataset.

Scans dataset root for DWI, T2, and prostate mask NIfTI volumes,
reads labels from sampled_file_list.xlsx, and writes a deterministic
CSV with columns: case_id, b_value, label, dwi_path, t2_path, mask_path
"""
import argparse
import csv
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Build index CSV for UCL distortion dataset")
    ap.add_argument("--data_root", type=str, required=True,
                    help="Root directory containing dwi/, t2/, prostate_mask/, sampled_file_list.xlsx")
    ap.add_argument("--out_csv", type=str, required=True,
                    help="Output CSV path")
    args = ap.parse_args()

    root = Path(args.data_root)
    xlsx = root / "sampled_file_list.xlsx"
    assert xlsx.exists(), f"Label file not found: {xlsx}"

    df = pd.read_excel(xlsx)
    assert set(df.columns) >= {"case_id", "b_value", "score"}, \
        f"Expected columns case_id, b_value, score; got {list(df.columns)}"

    dwi_dir = root / "dwi"
    t2_dir = root / "t2"
    mask_dir = root / "prostate_mask"

    rows = []
    warnings = []
    for _, r in df.iterrows():
        cid = str(r["case_id"])
        fname = f"{cid}.nii.gz"

        dwi_path = dwi_dir / fname
        t2_path = t2_dir / fname
        mask_path = mask_dir / fname

        missing = []
        if not dwi_path.exists():
            missing.append("dwi")
        if not t2_path.exists():
            missing.append("t2")
        if not mask_path.exists():
            missing.append("mask")

        if missing:
            warnings.append(f"  {cid}: missing {', '.join(missing)}")
            continue

        rows.append((
            cid,
            str(r["b_value"]),
            int(r["score"]),
            str(dwi_path),
            str(t2_path),
            str(mask_path),
        ))

    rows.sort(key=lambda x: x[0])

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "b_value", "label", "dwi_path", "t2_path", "mask_path"])
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out}")
    label_counts = {}
    for r in rows:
        label_counts[r[2]] = label_counts.get(r[2], 0) + 1
    print(f"Label distribution: {label_counts}")

    if warnings:
        print(f"\nWarnings ({len(warnings)} cases skipped):")
        for w in warnings:
            print(w)


if __name__ == "__main__":
    main()
