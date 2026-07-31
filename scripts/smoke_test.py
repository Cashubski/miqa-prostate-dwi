#!/usr/bin/env python3
"""Run every training pipeline with minimal data to verify no crashes.

Run this before committing GPU time. If any pipeline fails here, it
will fail on the full data too.

Usage:
    python scripts/smoke_test.py --index_csv /path/to/index.csv

Runs (each with --max_cases 4 --epochs 2 --no_amp --device cpu):
  1. train.py                baseline (DWI, T2, fusion)
  2. train_multitask.py      multi-task with lambda=0.3
  3. train_fusion.py         mid, late, diff_map fusion
  4. pretrain_ssl.py         SimCLR and BYOL pretraining
  5. train_ssl_finetune      fine-tune from SSL checkpoint
  6. train_mc_dropout.py     MC Dropout training (dropout=0.3)
  7. evaluate_uncertainty    MC Dropout UQ inference (n_forward=5)
  8. evaluate_uncertainty    Deep Ensemble UQ (reuse baseline ckpt)

Total: ~18 mini-runs, takes ~3-5 minutes on CPU.
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run(cmd, name):
    """Run a command, return (name, success, elapsed_secs, error_msg)."""
    print(f"\n{'='*60}")
    print(f"SMOKE: {name}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = round(time.time() - t0, 1)
    ok = result.returncode == 0
    err = ""
    if not ok:
        # Last 20 lines of stderr
        lines = result.stderr.strip().split("\n")
        err = "\n".join(lines[-20:])
        print(f"  FAILED ({elapsed}s)")
        print(f"  STDERR (last 20 lines):\n{err}")
    else:
        print(f"  PASSED ({elapsed}s)")
    return name, ok, elapsed, err


def main():
    ap = argparse.ArgumentParser(description="Smoke test all pipelines")
    ap.add_argument("--index_csv", type=str, required=True)
    ap.add_argument("--tmp_dir", type=str, default="",
                    help="Temp dir for smoke outputs (default: /tmp/miqa_smoke)")
    ap.add_argument("--keep", action="store_true",
                    help="Keep temp outputs after test")
    args = ap.parse_args()

    tmp = Path(args.tmp_dir) if args.tmp_dir else Path("/tmp/miqa_smoke")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    scripts_dir = Path(__file__).parent
    py = sys.executable

    # Common args for minimal runs
    base = ["--max_cases", "4", "--epochs", "2", "--no_amp",
            "--device", "cpu", "--num_workers", "0",
            "--batch", "4", "--save_every", "0"]

    results = []

    # ----------------------------------------------------------------
    # 1. Baseline: DWI, T2, fusion (early)
    # ----------------------------------------------------------------
    for mod in ["dwi", "t2", "fusion"]:
        name = f"train.py --modality {mod}"
        out = tmp / f"baseline_{mod}"
        cmd = [py, str(scripts_dir / "train.py"),
               "--index_csv", args.index_csv,
               "--outdir", str(out),
               "--modality", mod,
               "--no_pretrained",
               *base]
        results.append(run(cmd, name))

    # ----------------------------------------------------------------
    # 2. Multi-task
    # ----------------------------------------------------------------
    name = "train_multitask.py"
    out = tmp / "multitask"
    cmd = [py, str(scripts_dir / "train_multitask.py"),
           "--index_csv", args.index_csv,
           "--outdir", str(out),
           "--aux_lambda", "0.3",
           "--no_pretrained",
           *base]
    results.append(run(cmd, name))

    # ----------------------------------------------------------------
    # 3. Fusion: mid, late, diff_map
    # ----------------------------------------------------------------
    for strat in ["mid", "late", "diff_map"]:
        name = f"train_fusion.py --fusion_strategy {strat}"
        out = tmp / f"fusion_{strat}"
        cmd = [py, str(scripts_dir / "train_fusion.py"),
               "--index_csv", args.index_csv,
               "--outdir", str(out),
               "--fusion_strategy", strat,
               "--no_pretrained",
               *base]
        results.append(run(cmd, name))

    # ----------------------------------------------------------------
    # 4. SSL pretraining: SimCLR + BYOL
    # ----------------------------------------------------------------
    ssl_ckpts = {}
    for method in ["simclr", "byol"]:
        name = f"pretrain_ssl.py --method {method}"
        out = tmp / f"ssl_{method}"
        cmd = [py, str(scripts_dir / "pretrain_ssl.py"),
               "--index_csv", args.index_csv,
               "--outdir", str(out),
               "--method", method,
               "--max_cases", "4", "--epochs", "2",
               "--no_amp", "--device", "cpu",
               "--num_workers", "0", "--batch", "4",
               "--slices_per_case", "2",
               "--save_every", "0"]
        r = run(cmd, name)
        results.append(r)
        if r[1]:  # success
            ssl_ckpts[method] = str(out / "last.pt")

    # ----------------------------------------------------------------
    # 5. SSL fine-tune (only if pretraining succeeded)
    # ----------------------------------------------------------------
    for method, ckpt in ssl_ckpts.items():
        name = f"train_ssl_finetune.py --ssl_method {method}"
        out = tmp / f"ft_{method}"
        cmd = [py, str(scripts_dir / "train_ssl_finetune.py"),
               "--index_csv", args.index_csv,
               "--outdir", str(out),
               "--ssl_ckpt", ckpt,
               "--ssl_method", method,
               "--no_pretrained" if "--no_pretrained" in base else "",
               *base]
        # Remove empty strings
        cmd = [c for c in cmd if c]
        results.append(run(cmd, name))

    # ----------------------------------------------------------------
    # 6a. K-fold baseline (fold 0 only, to test the plumbing)
    # ----------------------------------------------------------------
    name = "train.py --n_folds 3 --fold 0 (k-fold)"
    kf_out = tmp / "kfold_f0"
    cmd = [py, str(scripts_dir / "train.py"),
           "--index_csv", args.index_csv,
           "--outdir", str(kf_out),
           "--n_folds", "3", "--fold", "0",
           "--no_pretrained",
           *base]
    kf_result = run(cmd, name)
    results.append(kf_result)

    # Verify test_predictions.csv and kfold_splits.json were created
    if kf_result[1]:
        kf_pred = kf_out / "test_predictions.csv"
        kf_splits = kf_out / "kfold_splits.json"
        if kf_pred.exists():
            print(f"  test_predictions.csv: OK ({kf_pred})")
        else:
            print(f"  WARNING: test_predictions.csv not found")
        if kf_splits.exists():
            print(f"  kfold_splits.json: OK ({kf_splits})")
        else:
            print(f"  WARNING: kfold_splits.json not found")

    # ----------------------------------------------------------------
    # 6. MC Dropout training
    # ----------------------------------------------------------------
    name = "train_mc_dropout.py --dropout_rate 0.3"
    out = tmp / "mc_dropout"
    cmd = [py, str(scripts_dir / "train_mc_dropout.py"),
           "--index_csv", args.index_csv,
           "--outdir", str(out),
           "--dropout_rate", "0.3",
           "--no_pretrained",
           *base]
    mc_result = run(cmd, name)
    results.append(mc_result)

    # ----------------------------------------------------------------
    # 7. Evaluate uncertainty: MC Dropout mode
    # ----------------------------------------------------------------
    if mc_result[1]:  # MC dropout training succeeded
        mc_ckpt = str(tmp / "mc_dropout" / "best_auroc.pt")
        name = "evaluate_uncertainty.py --mc_dropout_ckpt"
        uq_out = tmp / "uq_mc"
        cmd = [py, str(scripts_dir / "evaluate_uncertainty.py"),
               "--index_csv", args.index_csv,
               "--outdir", str(uq_out),
               "--mc_dropout_ckpt", mc_ckpt,
               "--n_forward", "5",
               "--batch", "4"]
        results.append(run(cmd, name))

    # ----------------------------------------------------------------
    # 8. Evaluate uncertainty: Deep Ensemble mode (reuse baseline ckpt)
    # ----------------------------------------------------------------
    baseline_ckpt = str(tmp / "baseline_dwi" / "best_auroc.pt")
    if (tmp / "baseline_dwi" / "best_auroc.pt").exists():
        name = "evaluate_uncertainty.py --ensemble_ckpts"
        uq_out = tmp / "uq_ensemble"
        cmd = [py, str(scripts_dir / "evaluate_uncertainty.py"),
               "--index_csv", args.index_csv,
               "--outdir", str(uq_out),
               "--ensemble_ckpts", baseline_ckpt, baseline_ckpt,
               "--batch", "4"]
        results.append(run(cmd, name))

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print(f"\n\n{'='*60}")
    print("SMOKE TEST SUMMARY")
    print(f"{'='*60}")

    total = len(results)
    passed = sum(1 for _, ok, _, _ in results if ok)
    failed = total - passed

    for name, ok, elapsed, err in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name} ({elapsed}s)")

    print(f"\n{passed}/{total} passed, {failed} failed")

    if failed > 0:
        print("\nFAILED PIPELINES (DO NOT START GPU TRAINING):")
        for name, ok, _, err in results:
            if not ok:
                print(f"\n  {name}:")
                for line in err.split("\n")[-5:]:
                    print(f"    {line}")
        print("\nFix the failures above before running sweep.py!")
        sys.exit(1)
    else:
        print("\nAll pipelines passed. Safe to start GPU training.")

    if not args.keep:
        shutil.rmtree(tmp)
        print(f"Cleaned up {tmp}")


if __name__ == "__main__":
    main()
