"""Compare UE-AUROC across distillation conditions (runs on pt8).

Conditions (each = a set of ResNet-18 student checkpoints):
  hetero      : distilled from the 9-model heterogeneous teacher (baseline)
                runs/distill_resnet18_f{0..4}            (seed 42)
  denseteacher: distilled from a DenseNet-only (homogeneous) teacher
                runs/distill_r18_denseteacher_f{0..4}
  r18teacher  : distilled from a ResNet-18-only (homogeneous) teacher
                runs/distill_r18_r18teacher_f{0..4}
  multiseed   : hetero teacher, 5 student seeds x 5 folds  (CI on the 0.694)
                runs/distill_r18_s{100..104}_f{0..4}

For each student we compute a single-model uncertainty (predictive entropy)
and its UE-AUROC (uncertainty ranks errors above correct). Pooled across
folds/seeds with a 95% bootstrap CI. The mechanism prediction: homogeneous-
teacher students should have markedly lower UE-AUROC than the hetero baseline.

Run on pt8:
  conda activate iq
  cd /raid/candi/michal/miqa
  python repo/scripts/make_distill_ue_eval.py

Output: prints a comparison table + saves logs/distill_ue_comparison.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from miqa.datasets import UCLDistortionDataset
from miqa.models import build_model

# Runs root: on pt8 = /raid/candi/michal/miqa/runs; locally = ~/miqa_ckpts
RUNS = Path(os.environ.get("MIQA_CKPTS", "/raid/candi/michal/miqa/runs"))
# Index CSV: default to the local repo copy (NIfTI resolves locally)
INDEX_CSV = Path(os.environ.get("MIQA_INDEX", str(REPO / "dataset" / "index_full.csv")))
LOG_DIR = Path(os.environ.get("MIQA_LOGS", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# label maps to a list of (run_dir_name, fold)
def _folds(prefix):
    return [(f"{prefix}_f{f}", f) for f in range(5)]

GROUPS = {
    "hetero (baseline)":     _folds("distill_resnet18"),
    "denseteacher (homog.)": _folds("distill_r18_denseteacher"),
    "r18teacher (homog.)":   _folds("distill_r18_r18teacher"),
}
# multiseed: 5 seeds x 5 folds (only include folds that exist)
MULTISEED = [(f"distill_r18_s{s}_f{f}", f, s)
             for s in range(100, 105) for f in range(5)]


def load_fold_test_split(df, run_dir, fold):
    sp = json.loads((Path(run_dir) / "kfold_splits.json").read_text())
    test_ps = set(sp["folds"][str(fold)]["test"])
    ps = df["case_id"].str.rsplit("__", n=1).str[0]
    return df[ps.isin(test_ps)].reset_index(drop=True)


def bootstrap_ci(labels, scores, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(labels)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(labels[idx])) < 2:
            continue
        vals.append(roc_auc_score(labels[idx], scores[idx]))
    return (float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5))) if vals else (float("nan"),) * 2


def eval_checkpoint(df, run_dir, fold):
    """Return per-case (prob, label) for one student checkpoint."""
    ckpt_path = Path(run_dir) / "best_auroc.pt"
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ca = ckpt.get("args", {})
    model = build_model("resnet18", in_chans=ca.get("k_slices", 3),
                        pretrained=False).to(DEVICE).eval()
    model.load_state_dict(ckpt["model"], strict=True)
    te = load_fold_test_split(df, run_dir, fold)
    ds = UCLDistortionDataset(te, modality="dwi", use_mask_crop=True,
                              k_slices=ca.get("k_slices", 3), img_size=224,
                              augment=False, return_case_id=True)
    dl = DataLoader(ds, batch_size=32, shuffle=False)
    ys, ps = [], []
    with torch.no_grad():
        for x, y, _ in dl:
            p = torch.sigmoid(model(x.to(DEVICE)).squeeze(-1)).cpu().numpy()
            ps.append(p); ys.append(y.numpy().ravel())
    del model
    return np.concatenate(ps), np.concatenate(ys).astype(int)


def ue_auroc(prob, label):
    """UE-AUROC via predictive entropy (single-model uncertainty)."""
    eps = 1e-9
    ent = -prob * np.log(prob + eps) - (1 - prob) * np.log(1 - prob + eps)
    correct = ((prob >= 0.5).astype(int) == label).astype(int)
    err = 1 - correct
    if err.sum() == 0 or correct.sum() == 0:
        return np.nan, ent, err
    return roc_auc_score(err, ent), ent, err


def eval_group(df, runs):
    """runs: list of (run_dir_name, fold). Returns pooled arrays + per-run UE."""
    all_p, all_y, all_ent, all_err, per_run = [], [], [], [], []
    for name, fold in runs:
        rd = RUNS / name
        res = eval_checkpoint(df, rd, fold)
        if res is None:
            continue
        prob, label = res
        u, ent, err = ue_auroc(prob, label)
        auroc = roc_auc_score(label, prob) if len(np.unique(label)) > 1 else np.nan
        per_run.append({"run": name, "fold": fold, "auroc": float(auroc),
                        "ue_auroc": float(u)})
        all_p.append(prob); all_y.append(label)
        all_ent.append(ent); all_err.append(err)
    if not all_p:
        return None
    p = np.concatenate(all_p); y = np.concatenate(all_y)
    ent = np.concatenate(all_ent); err = np.concatenate(all_err)
    pooled_ue = roc_auc_score(err, ent)
    pooled_auroc = roc_auc_score(y, p)
    order = np.argsort(ent)
    k = int(round(0.70 * len(ent)))
    sel_acc70 = float(((p[order[:k]] >= 0.5).astype(int) == y[order[:k]]).mean())
    ue_ci = bootstrap_ci(err, ent)
    per_ue = np.array([r["ue_auroc"] for r in per_run])
    return {
        "n_runs": len(per_run),
        "auroc_pooled": float(pooled_auroc),
        "ue_pooled": float(pooled_ue),
        "sel_acc_70": sel_acc70,
        "ue_ci": list(ue_ci),
        "ue_perrun_mean": float(np.nanmean(per_ue)),
        "ue_perrun_std": float(np.nanstd(per_ue)),
        "per_run": per_run,
    }


def main():
    df = pd.read_csv(INDEX_CSV)
    print(f"Device: {DEVICE}   index: {len(df)} rows\n")
    out = {}

    for label, runs in GROUPS.items():
        print(f"=== {label} ===")
        r = eval_group(df, runs)
        if r is None:
            print("  (no completed runs)\n"); continue
        out[label] = r
        print(f"  n={r['n_runs']}  AUROC={r['auroc_pooled']:.3f}  "
              f"UE-AUROC={r['ue_pooled']:.3f} "
              f"[{r['ue_ci'][0]:.3f}, {r['ue_ci'][1]:.3f}]  "
              f"(per-run {r['ue_perrun_mean']:.3f}±{r['ue_perrun_std']:.3f})\n")

    # multiseed (only completed folds); also give per-seed UE for the CI
    print("=== multiseed (hetero teacher, 5 seeds) ===")
    ms = eval_group(df, [(n, f) for (n, f, s) in MULTISEED
                         if (RUNS / n / "best_auroc.pt").exists()])
    if ms is not None:
        out["multiseed"] = ms
        # per-seed mean UE
        by_seed = {}
        for (n, f, s) in MULTISEED:
            for pr in ms["per_run"]:
                if pr["run"] == n:
                    by_seed.setdefault(s, []).append(pr["ue_auroc"])
        seed_means = {s: float(np.nanmean(v)) for s, v in by_seed.items()}
        out["multiseed"]["per_seed_mean_ue"] = seed_means
        print(f"  n={ms['n_runs']}  AUROC={ms['auroc_pooled']:.3f}  "
              f"UE-AUROC={ms['ue_pooled']:.3f} "
              f"[{ms['ue_ci'][0]:.3f}, {ms['ue_ci'][1]:.3f}]")
        print(f"  per-seed mean UE: "
              + ", ".join(f"s{s}={v:.3f}" for s, v in sorted(seed_means.items())))
        if seed_means:
            vals = np.array(list(seed_means.values()))
            print(f"  across seeds: {vals.mean():.3f} ± {vals.std():.3f} "
                  f"(n={len(vals)} seeds)\n")
    else:
        print("  (no completed multiseed runs yet)\n")

    (LOG_DIR / "distill_ue_comparison.json").write_text(json.dumps(out, indent=2))
    print("Saved logs/distill_ue_comparison.json")
    print("\n--- INTERPRETATION ---")
    print("If denseteacher/r18teacher UE-AUROC << hetero (~0.69), the student's")
    print("uncertainty quality is inherited from TEACHER DIVERSITY, so the mechanism is")
    print("confirmed. multiseed CI shows the 0.69 is robust across student seeds.")


if __name__ == "__main__":
    main()
