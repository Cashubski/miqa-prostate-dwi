"""Save PNG slices for worst false negatives and false positives.

Reads test_predictions.csv from error_analysis_cls.py, loads NIfTI
volumes, extracts center slice from mask bbox, saves PNGs.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import imageio.v2 as imageio


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Normalize image to [0,255] uint8."""
    img = np.asarray(img, dtype=np.float32)
    lo = float(np.min(img))
    hi = float(np.max(img))
    img = (img - lo) / (hi - lo + 1e-8)
    return (img * 255.0).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True,
                    help="test_predictions.csv from error_analysis_cls.py")
    ap.add_argument("--index_csv", required=True,
                    help="index.csv with file paths")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--topk", type=int, default=10,
                    help="Number of FN and FP each to save")
    ap.add_argument("--modality", type=str, default="dwi",
                    choices=["dwi", "t2"])
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(args.pred_csv)
    index = pd.read_csv(args.index_csv)
    path_col = "t2_path" if args.modality == "t2" else "dwi_path"
    path_map = dict(zip(index["case_id"], index[path_col]))
    mask_map = dict(zip(index["case_id"], index["mask_path"]))

    fn = pred[pred["error_type"] == "FN"].head(args.topk)
    fp = pred[pred["error_type"] == "FP"].head(args.topk)

    count = 0
    for rank, (errors, tag) in enumerate([(fn, "fn"), (fp, "fp")]):
        for i, (_, row) in enumerate(errors.iterrows()):
            cid = row["case_id"]
            vol_path = path_map.get(cid)
            mask_path = mask_map.get(cid)
            if vol_path is None or mask_path is None:
                continue

            vol = nib.load(vol_path).get_fdata().astype(np.float32)
            mask = nib.load(mask_path).get_fdata().astype(np.float32)

            # Center slice from mask bbox
            coords = np.argwhere(mask > 0)
            z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
            z_center = (z_min + z_max) // 2

            r_min, r_max = coords[:, 0].min(), coords[:, 0].max() + 1
            c_min, c_max = coords[:, 1].min(), coords[:, 1].max() + 1

            slc = vol[r_min:r_max, c_min:c_max, z_center]
            u8 = to_uint8(slc)

            bval = row.get("b_value", "")
            y_true = int(row["y_true"])
            y_prob = float(row["y_prob"])
            fname = f"{i:02d}_{tag}_case{cid}_b{bval}_true{y_true}_prob{y_prob:.2f}.png"
            imageio.imwrite(out / fname, u8)
            count += 1

    print(f"Wrote {count} PNGs to {out}")


if __name__ == "__main__":
    main()
