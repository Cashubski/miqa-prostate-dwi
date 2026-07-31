import argparse, json, pathlib, csv, re

ap = argparse.ArgumentParser()
ap.add_argument("--runs_dir", required=True)
ap.add_argument("--out_csv", required=True)
args = ap.parse_args()

runs_dir = pathlib.Path(args.runs_dir)
rows = []
for summ in runs_dir.rglob("summary.json"):
    run = summ.parent
    s = json.loads(summ.read_text())
    row = {
        "run": run.name,
        "path": str(run),
        "arch": s.get("arch"),
        "modality": s.get("modality"),
        "use_mask_crop": s.get("use_mask_crop"),
        "k_slices": s.get("k_slices"),
        "lr": s.get("lr"),
        "wd": s.get("wd"),
        "batch": s.get("batch"),
        "epochs": s.get("epochs"),
        "scheduler": s.get("scheduler"),
        "img_size": s.get("img_size"),
        "aug": s.get("aug"),
        "seed": s.get("seed"),
        "split_seed": s.get("split_seed"),
        "n_train": s.get("n_train"),
        "n_val": s.get("n_val"),
        "n_test": s.get("n_test"),
        "n_params": s.get("n_params"),
        "n_trainable": s.get("n_trainable"),
        "best_epoch": s.get("best_epoch"),
        "secs": s.get("secs"),
        "gpu_mem_peak_mb": s.get("gpu_mem_peak_mb"),
        "gpu0": s.get("gpu0"),
        "torch": s.get("torch"),
    }
    for split in ["best_val", "test"]:
        d = s.get(split, {})
        for k,v in d.items():
            row[f"{split}_{k}"] = v
    rows.append(row)

rows.sort(key=lambda r: r["run"])
fields = sorted({k for r in rows for k in r.keys()})
with open(args.out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print("Wrote:", args.out_csv, "rows:", len(rows))
