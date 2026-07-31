"""Launch a grid sweep of UCL distortion classification experiments.

5-fold cross-validation with batch_size=32 and 100 epochs (cosine schedule).
~5,740 fold-runs across 14 tiers.

Dataset options:
  index.csv      : 245-case pilot subset (one acq/patient)
  index_full.csv : 1,027 acquisitions from 627 patients (full dataset)

GPU-hour estimates (GV100, ~50 min/fold-run at 1,027 cases, ~12 min at 245):
  Full sweep @ 1,027 cases: ~4,100 GV100-h
  Full sweep @ 245 cases:   ~1,080 GV100-h

Hardware: 2x Quadro GV100 (32 GB) + 2x Quadro P5000 (16 GB)
  P5000 throughput ≈ 0.65x GV100 (smaller batch due to 16 GB VRAM).
  Effective capacity (120 experiment days, Mar to Jun):
    GV100: 2 × 24h × 120d = 5,760 GV100-h
    P5000: 2 × 24h × 120d × 0.65 = 3,744 GV100-equiv-h
    Total: ~9,500 GV100-equiv-h

Recommended multi-cluster strategy (~2,400 GV100-equiv-h, ~30 days wall-clock):
  GV100 cluster (tiers on full 1,027-case data, headline results):
    0  Classical baselines (CPU)                            ~1h
    1  Core baselines (300 fold-runs)                     ~250h
    9  SSL pretrain + finetune (100 fold-runs)            ~200h
    10 Fusion (150 fold-runs)                             ~125h
    11 MC Dropout UQ (200 fold-runs)                      ~167h
    12 Learning curves (60 fold-runs)                      ~50h
    13 Leave-one-b-value-out (40 fold-runs)                ~33h
                                                   GV100: ~826h, ~17 days on 2x GV100

  P5000 cluster (tiers on 245-case subset, ablations):
    2  Slice ablation (300 fold-runs)                      ~60h
    3  Resolution + augmentation (150 fold-runs)           ~40h
    4  LR search (150 fold-runs)                           ~30h
    5  Extra architectures (150 fold-runs)                 ~30h
    6  Long cosine-LR (100 fold-runs)                      ~20h
    7  Full factorial (1200 fold-runs)                    ~240h
    8  Multi-task lambda sweep (2400 fold-runs)           ~480h
                                                   P5000: ~900h, ~28 days on 2x P5000

  Then validate: re-run top configs from tiers 7-8 on full 1,027 data (~80h on GV100).
  Both clusters finish within 30 days. Total: ~1,806 GV100-equiv-h with ~7,700h spare.

Tier presets (all DL tiers with 5-fold CV, 100 epochs, cosine schedule, batch=32):
  CHAPTER 4: BASELINES
  0    Classical baselines (LR/RF/SVM): CPU only, minutes           (5 fold-runs)
  1    Core baselines: arch x modality x crop x 5 seeds x 5 folds  (300 fold-runs)
  2    Slice ablation: k_slices x modality x 5 seeds x 5 folds     (300 fold-runs)
  3    Resolution + augmentation: size x aug x 5 seeds x 5 folds   (150 fold-runs)
  4    Learning rate search: lr x arch x 5 seeds x 5 folds         (150 fold-runs)
  5    Extra architectures: DenseNet121 + EfficientNet-B0 x 5 folds (150 fold-runs)
  6    Long cosine-LR: all 4 archs x 5 seeds x 5 folds             (100 fold-runs)
  7    Full factorial: 4 arch x 3 mod x 4 k x 5 seeds x 5 folds   (1200 fold-runs)
  CHAPTER 5: MULTI-TASK
  8    Lambda sweep: 8 lam x 4 arch x 3 mod x 5 seeds x 5 folds   (2400 fold-runs)
  CHAPTER 6: FUSION
  10   Fusion: 3 strats x 2 archs x 5 seeds x 5 folds             (150 fold-runs)
  CHAPTER 7: SSL
  9    SSL pretrain (once) + finetune x 5 folds                     (100 fold-runs)
  CHAPTER 8: UNCERTAINTY QUANTIFICATION
  11   MC Dropout: 4 rates x 2 archs x 5 seeds x 5 folds          (200 fold-runs)
  CHAPTER 9: CROSS-CUTTING ANALYSIS
  12   Learning curves: top configs x 4 fractions x 5 folds        (60 fold-runs)
  13   Leave-one-b-value-out: 2 archs x 4 b-vals x 5 folds        (40 fold-runs)
"""
import argparse, itertools, json, os, shutil, subprocess, sys
from datetime import datetime
from pathlib import Path


def _run_completed(run_dir: Path) -> bool:
    """Check whether a run with the same config has completed (any timestamp).

    Run names include a trailing timestamp (_YYYYMMDD_HHMMSS), so the same
    config gets a fresh dir on each launch. To avoid re-running completed
    work after a restart, we strip the trailing timestamp from `run_dir.name`
    and check siblings with the same stem for a valid summary.json (no error).

    A dir with no summary.json (or summary.json containing an "error" field)
    is a crashed/killed run and is NOT counted as completed.
    """
    import re
    import json
    # Direct check first (fast path)
    if (run_dir / "summary.json").exists():
        try:
            with open(run_dir / "summary.json") as f:
                s = json.load(f)
            if "error" not in s:
                return True
        except Exception:
            pass

    # Sibling check: strip trailing _YYYYMMDD_HHMMSS and look for any
    # other dir with the same stem that has a valid summary.json
    name = run_dir.name
    stem = re.sub(r'_\d{8}_\d{6}$', '', name)
    if stem == name:
        return False  # name didn't have a timestamp suffix
    for sibling in run_dir.parent.glob(f"{stem}_*"):
        if sibling == run_dir:
            continue
        sf = sibling / "summary.json"
        if sf.exists():
            try:
                with open(sf) as f:
                    s = json.load(f)
                if "error" not in s:
                    return True
            except Exception:
                pass
    return False


def _check_disk_space(path: Path, min_gb: float = 10.0):
    """Abort if fewer than min_gb are free on the partition containing path."""
    stat = os.statvfs(str(path))
    free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
    print(f"Disk space on {path}: {free_gb:.1f} GB free")
    if free_gb < min_gb:
        print(f"ERROR: Less than {min_gb} GB free. Aborting to prevent "
              f"disk-full failures.")
        sys.exit(1)


def make_run_name(arch, img_size, aug, seed, modality="dwi",
                  mask_crop=True, scheduler="none", k_slices=3, lr=3e-4,
                  fold=None):
    aug_tag = "aug" if aug else "noaug"
    crop_tag = "roi" if mask_crop else "global"
    sched_tag = f"_{scheduler}" if scheduler != "none" else ""
    lr_tag = f"_lr{lr:.0e}".replace("+", "") if lr != 3e-4 else ""
    k_tag = f"_k{k_slices}" if k_slices != 3 else ""
    fold_tag = f"_f{fold}" if fold is not None else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (f"ucl_{arch}_{modality}_{crop_tag}{k_tag}_s{img_size}"
            f"_{aug_tag}{lr_tag}{sched_tag}_seed{seed}{fold_tag}_{ts}")


def apply_tier_preset(args):
    """Override grid axes based on --tier value."""
    tier = args.tier
    if tier == 0:
        # Classical baselines (LR, RF, SVM), CPU only
        args.n_folds = 5
        args.classical = True
        return

    if tier == 1:
        # Core baselines: R18/R50 x DWI/T2/fusion x ROI/global x 5 seeds = 60
        args.archs = ["resnet18", "resnet50"]
        args.modalities = ["dwi", "t2", "fusion"]
        args.mask_crops = ["on", "off"]
        args.k_slices_list = [3]
        args.lrs = [3e-4]
        args.aug_modes = ["on"]
        args.sizes = [224]
        args.seeds = [1, 2, 3, 4, 5]
        args.scheduler = "cosine"
        args.warmup_epochs = 10
        args.epochs = 100
        args.batch = 32
        args.n_folds = 5

    elif tier == 2:
        # Slice ablation: k=1,3,5,7 x DWI/T2/fusion x 5 seeds = 60
        args.mask_crops = ["on"]
        args.k_slices_list = [1, 3, 5, 7]
        args.lrs = [3e-4]
        args.modalities = ["dwi", "t2", "fusion"]
        args.aug_modes = ["on"]
        args.sizes = [224]
        args.seeds = [1, 2, 3, 4, 5]
        args.scheduler = "cosine"
        args.warmup_epochs = 10
        args.epochs = 100
        args.batch = 32
        args.n_folds = 5

    elif tier == 3:
        # Resolution + augmentation: 128/224/384 x aug on/off x 5 seeds = 30
        args.mask_crops = ["on"]
        args.k_slices_list = [3]
        args.lrs = [3e-4]
        args.aug_modes = ["on", "off"]
        args.sizes = [128, 224, 384]
        args.seeds = [1, 2, 3, 4, 5]
        args.scheduler = "cosine"
        args.warmup_epochs = 10
        args.epochs = 100
        args.batch = 32
        args.n_folds = 5

    elif tier == 4:
        # LR search: 1e-4/3e-4/1e-3 x R18/R50 x 5 seeds = 30
        args.archs = ["resnet18", "resnet50"]
        args.mask_crops = ["on"]
        args.k_slices_list = [3]
        args.lrs = [1e-4, 3e-4, 1e-3]
        args.aug_modes = ["on"]
        args.sizes = [224]
        args.seeds = [1, 2, 3, 4, 5]
        args.scheduler = "cosine"
        args.warmup_epochs = 10
        args.epochs = 100
        args.batch = 32
        args.n_folds = 5

    elif tier == 5:
        # Extra architectures: DenseNet121/EfficientNet-B0 x modality x 5 seeds = 30
        args.archs = ["densenet121", "efficientnet_b0"]
        args.modalities = ["dwi", "t2", "fusion"]
        args.mask_crops = ["on"]
        args.k_slices_list = [3]
        args.lrs = [3e-4]
        args.aug_modes = ["on"]
        args.sizes = [224]
        args.seeds = [1, 2, 3, 4, 5]
        args.scheduler = "cosine"
        args.warmup_epochs = 10
        args.epochs = 100
        args.batch = 32
        args.n_folds = 5

    elif tier == 6:
        # Long cosine-LR: all 4 archs x best modality x 100ep x 5 seeds = 20
        args.archs = ["resnet18", "resnet50", "densenet121", "efficientnet_b0"]
        args.mask_crops = ["on"]
        args.k_slices_list = [3]
        args.lrs = [3e-4]
        args.aug_modes = ["on"]
        args.sizes = [224]
        args.seeds = [1, 2, 3, 4, 5]
        args.scheduler = "cosine"
        args.warmup_epochs = 10
        args.epochs = 100
        args.batch = 32
        args.n_folds = 5

    elif tier == 7:
        # Full factorial: 4 arch x 3 mod x 4 k x 5 seeds = 240
        args.archs = ["resnet18", "resnet50", "densenet121", "efficientnet_b0"]
        args.modalities = ["dwi", "t2", "fusion"]
        args.mask_crops = ["on"]
        args.k_slices_list = [1, 3, 5, 7]
        args.lrs = [3e-4]
        args.aug_modes = ["on"]
        args.sizes = [224]
        args.seeds = [1, 2, 3, 4, 5]
        args.scheduler = "cosine"
        args.warmup_epochs = 10
        args.epochs = 100
        args.batch = 32
        args.n_folds = 5

    elif tier == 8:
        # Multi-task: 8 lambda x 4 arch x 3 modality x 5 seeds
        args.archs = ["resnet18", "resnet50", "densenet121", "efficientnet_b0"]
        args.modalities = ["dwi", "t2", "fusion"]
        args.mask_crops = ["on"]
        args.k_slices_list = [3]
        args.lrs = [3e-4]
        args.aux_lambdas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        args.aug_modes = ["on"]
        args.sizes = [224]
        args.seeds = [1, 2, 3, 4, 5]
        args.scheduler = "cosine"
        args.warmup_epochs = 10
        args.epochs = 100
        args.batch = 32
        args.n_folds = 5

    elif tier == 9:
        # SSL: method x arch x init x 5 seeds (200ep pretrain)
        args.archs = ["resnet18", "resnet50"]
        args.ssl_methods = ["simclr", "byol"]
        args.ssl_inits = ["random", "imagenet"]
        args.ssl_epochs = 200
        args.modalities = ["dwi"]
        args.mask_crops = ["on"]
        args.k_slices_list = [3]
        args.lrs = [3e-4]
        args.aug_modes = ["on"]
        args.sizes = [224]
        args.seeds = [1, 2, 3, 4, 5]
        args.scheduler = "cosine"
        args.warmup_epochs = 10
        args.epochs = 100  # fine-tuning epochs
        args.batch = 32
        args.n_folds = 5

    elif tier == 10:
        # Fusion strategies: mid/late/diff_map x 2 archs x 5 seeds
        args.archs = ["resnet18", "resnet50"]
        args.fusion_strategies = ["mid", "late", "diff_map"]
        args.mask_crops = ["on"]
        args.k_slices_list = [3]
        args.lrs = [3e-4]
        args.aug_modes = ["on"]
        args.sizes = [224]
        args.seeds = [1, 2, 3, 4, 5]
        args.scheduler = "cosine"
        args.warmup_epochs = 10
        args.epochs = 100
        args.batch = 32
        args.n_folds = 5

    elif tier == 11:
        # MC Dropout UQ: 4 dropout rates x 2 archs x 5 seeds = 40 runs
        args.archs = ["resnet18", "resnet50"]
        args.dropout_rates = [0.1, 0.2, 0.3, 0.5]
        args.mask_crops = ["on"]
        args.k_slices_list = [3]
        args.lrs = [3e-4]
        args.aug_modes = ["on"]
        args.sizes = [224]
        args.seeds = [1, 2, 3, 4, 5]
        args.scheduler = "cosine"
        args.warmup_epochs = 10
        args.epochs = 100
        args.batch = 32
        args.n_folds = 5

    elif tier == 12:
        # Learning curves: top 3 configs x 4 fractions x 5 folds = 60
        args.archs = ["resnet18", "resnet50", "densenet121"]
        args.modalities = ["dwi"]
        args.mask_crops = ["on"]
        args.k_slices_list = [3]
        args.lrs = [3e-4]
        args.aug_modes = ["on"]
        args.sizes = [224]
        args.seeds = [42]
        args.train_fractions = [0.25, 0.5, 0.75, 1.0]
        args.scheduler = "cosine"
        args.warmup_epochs = 10
        args.epochs = 100
        args.batch = 32
        args.n_folds = 5

    elif tier == 13:
        # Leave-one-b-value-out: 2 archs x 4 b-values x 5 folds = 40
        args.archs = ["resnet18", "resnet50"]
        args.exclude_bvalues = ["1k4", "1k5", "1k6", "2k"]
        args.modalities = ["dwi"]
        args.mask_crops = ["on"]
        args.k_slices_list = [3]
        args.lrs = [3e-4]
        args.aug_modes = ["on"]
        args.sizes = [224]
        args.seeds = [42]
        args.scheduler = "cosine"
        args.warmup_epochs = 10
        args.epochs = 100
        args.batch = 32
        args.n_folds = 5


def main():
    ap = argparse.ArgumentParser(description="UCL distortion sweep runner")

    ap.add_argument("--index_csv", type=str, required=True)
    ap.add_argument("--runs_dir", type=str, required=True,
                    help="Root directory for run outputs")
    ap.add_argument("--script", type=str, default="",
                    help="Path to train.py "
                         "(default: same dir as this script)")

    # Tier preset
    ap.add_argument("--tier", type=int, default=0,
                    choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                    help="Tier preset (0=classical, 1-7=baseline, "
                         "8=multi-task, 9=SSL, 10=fusion, 11=UQ, "
                         "12=learning curves, 13=leave-one-bval-out)")

    # Cluster assignment hint
    ap.add_argument("--cluster", type=str, default="",
                    choices=["", "gv100", "p5000"],
                    help="Cluster hint for tier recommendations. "
                         "GV100 (full data): tiers 0,1,9-13. "
                         "P5000 (245-case ablations): tiers 2-8.")

    # Multi-task options (tier 8)
    ap.add_argument("--aux_lambdas", nargs="+", type=float,
                    default=[0.0, 0.1, 0.3, 0.5, 1.0],
                    help="Lambda values for multi-task sweep")

    # SSL options (tier 9)
    ap.add_argument("--ssl_methods", nargs="+", default=["simclr", "byol"],
                    help="SSL methods to sweep")
    ap.add_argument("--ssl_epochs", type=int, default=200,
                    help="Pretraining epochs for SSL")
    ap.add_argument("--ssl_inits", nargs="+",
                    default=["random", "imagenet"],
                    help="SSL backbone initialization: random / imagenet")

    # Fusion options (tier 10)
    ap.add_argument("--fusion_strategies", nargs="+",
                    default=["mid", "late", "diff_map"],
                    help="Fusion strategies to sweep")

    # UQ options (tier 11)
    ap.add_argument("--dropout_rates", nargs="+", type=float,
                    default=[0.1, 0.2, 0.3, 0.5],
                    help="Dropout rates for MC Dropout sweep")

    # Grid axes
    ap.add_argument("--archs", nargs="+", default=["resnet18", "resnet50"])
    ap.add_argument("--sizes", nargs="+", type=int, default=[224])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--aug_modes", nargs="+", default=["on", "off"],
                    help="'on' and/or 'off'")
    ap.add_argument("--modalities", nargs="+", default=["dwi"],
                    help="dwi, t2, and/or fusion")
    ap.add_argument("--mask_crops", nargs="+", default=["on"],
                    help="'on' (ROI) and/or 'off' (global)")
    ap.add_argument("--k_slices_list", nargs="+", type=int, default=[3],
                    help="k_slices values to sweep (e.g. 1 3 5 7)")
    ap.add_argument("--lrs", nargs="+", type=float, default=[3e-4],
                    help="Learning rates to sweep (e.g. 1e-4 3e-4 1e-3)")

    # Pass-through training args
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--scheduler", type=str, default="none",
                    choices=["none", "cosine"])
    ap.add_argument("--warmup_epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=0,
                    help="0=auto per arch")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--max_cases", type=int, default=0)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--n_folds", type=int, default=0,
                    help="K-fold CV (0=disabled)")

    # Safety
    ap.add_argument("--overwrite", action="store_true",
                    help="Allow overwriting existing run dirs")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print commands without executing")

    args = ap.parse_args()

    # Apply tier preset (overrides grid axes)
    args.classical = False
    args.train_fractions = [1.0]
    apply_tier_preset(args)

    # ---- Tier 0: classical baselines (CPU, no GPU needed) ----
    if getattr(args, "classical", False):
        classical_script = str(Path(__file__).parent / "train_classical.py")
        assert Path(classical_script).exists(), \
            f"Not found: {classical_script}"

        runs_root = Path(args.runs_dir)
        runs_root.mkdir(parents=True, exist_ok=True)

        folds = range(args.n_folds) if args.n_folds > 0 else [None]
        n_fold_runs = max(1, args.n_folds)
        print(f"Classical baseline sweep: {n_fold_runs} folds")

        for fold in folds:
            fold_tag = f"_f{fold}" if fold is not None else ""
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"classical{fold_tag}_{ts}"
            run_dir = runs_root / run_name

            cmd = [
                sys.executable, classical_script,
                "--index_csv", args.index_csv,
                "--outdir", str(run_dir),
                "--split_seed", str(args.split_seed),
            ]
            if fold is not None:
                cmd += ["--n_folds", str(args.n_folds),
                         "--fold", str(fold)]
            if args.max_cases > 0:
                cmd += ["--max_cases", str(args.max_cases)]

            print(f"  [{fold_tag or 'single'}] {run_name}")
            if args.dry_run:
                print(f"  CMD: {' '.join(cmd)}")
            else:
                result = subprocess.run(cmd)
                status = "DONE" if result.returncode == 0 else "FAILED"
                print(f"  {status}")
            print()

        print("Classical baseline sweep finished.")
        return

    # Locate training script
    if args.script:
        train_script = args.script
    else:
        train_script = str(Path(__file__).parent / "train.py")
    assert Path(train_script).exists(), f"Training script not found: {train_script}"

    runs_root = Path(args.runs_dir)
    runs_root.mkdir(parents=True, exist_ok=True)

    # Build grid (8 axes now)
    grid = list(itertools.product(
        args.archs, args.sizes, args.aug_modes, args.modalities,
        args.mask_crops, args.k_slices_list, args.lrs, args.seeds))

    # Estimate GPU hours; scale by dataset size
    # Base estimates are for 245 cases on GV100; scale linearly for larger datasets
    import pandas as _pd
    _n_cases = len(_pd.read_csv(args.index_csv))
    _scale = _n_cases / 245.0  # 1.0 for original 245-case dataset
    _base_time = {
        (224, "cosine", 100): 12, (224, "none", 30): 5,
        (384, "cosine", 100): 25, (128, "cosine", 100): 6,
        (224, "cosine", 30): 5,  (128, "none", 30): 3,
        (384, "none", 30): 12,
    }
    _time_per_run = {k: round(v * _scale) for k, v in _base_time.items()}
    n_fold_runs = max(1, args.n_folds) if args.n_folds > 0 else 1
    est_mins = sum(
        _time_per_run.get((size, args.scheduler, args.epochs), 12)
        for _, size, _, _, _, _, _, _ in grid) * n_fold_runs
    est_hours = est_mins / 60

    total_runs = len(grid) * n_fold_runs
    print(f"Sweep: {len(grid)} configs x {n_fold_runs} folds "
          f"= {total_runs} runs (tier={args.tier})")
    print(f"  dataset: {_n_cases} cases (scale={_scale:.1f}x vs 245-case baseline)")
    print(f"  archs={args.archs}, sizes={args.sizes}, "
          f"aug={args.aug_modes}")
    print(f"  modalities={args.modalities}, "
          f"mask_crops={args.mask_crops}")
    print(f"  k_slices={args.k_slices_list}, lrs={args.lrs}, "
          f"seeds={args.seeds}")
    print(f"  scheduler={args.scheduler}, epochs={args.epochs}, "
          f"batch={args.batch}, n_folds={args.n_folds}")
    print(f"  estimated GPU time: ~{est_hours:.0f}h")
    print(f"  runs_dir={runs_root}")
    print()

    if not args.dry_run:
        _check_disk_space(runs_root)

    folds = range(args.n_folds) if args.n_folds > 0 else [None]
    run_idx = 0
    for fold in folds:
        for i, (arch, size, aug_mode, modality, mask_crop, k_sl, lr, seed) in enumerate(grid):
            run_idx += 1
            aug_on = aug_mode == "on"
            crop_on = mask_crop == "on"
            run_name = make_run_name(arch, size, aug_on, seed,
                                     modality=modality, mask_crop=crop_on,
                                     scheduler=args.scheduler,
                                     k_slices=k_sl, lr=lr, fold=fold)
            run_dir = runs_root / run_name

            if _run_completed(run_dir) and not args.overwrite:
                print(f"[{run_idx}/{total_runs}] SKIP {run_name} (completed)")
                continue
            if run_dir.exists() and not _run_completed(run_dir):
                print(f"[{run_idx}/{total_runs}] RETRY {run_name} "
                      f"(dir exists but no summary.json, previous crash?)")
                shutil.rmtree(run_dir)

            cmd = [
                sys.executable, train_script,
                "--index_csv", args.index_csv,
                "--outdir", str(run_dir),
                "--model", arch,
                "--img_size", str(size),
                "--k_slices", str(k_sl),
                "--seed", str(seed),
                "--split_seed", str(args.split_seed),
                "--epochs", str(args.epochs),
                "--lr", str(lr),
                "--wd", str(args.wd),
                "--patience", str(args.patience),
                "--gpu", str(args.gpu),
                "--modality", modality,
                "--scheduler", args.scheduler,
                "--warmup_epochs", str(args.warmup_epochs),
            ]

            if args.batch > 0:
                cmd += ["--batch", str(args.batch)]

            if fold is not None:
                cmd += ["--n_folds", str(args.n_folds),
                         "--fold", str(fold)]

            if not aug_on:
                cmd.append("--no_aug")

            if not crop_on:
                cmd.append("--no_mask_crop")

            if args.no_amp:
                cmd.append("--no_amp")

            if args.max_cases > 0:
                cmd += ["--max_cases", str(args.max_cases)]

            print(f"[{run_idx}/{total_runs}] {run_name}")
            if args.dry_run:
                print(f"  CMD: {' '.join(cmd)}")
                continue

            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"  FAILED (exit {result.returncode})")
            else:
                print(f"  DONE")
            print()

    # ---- Tier 8: multi-task sweep (uses train_multitask.py) ----
    if args.tier == 8:
        mt_script = str(Path(__file__).parent / "train_multitask.py")
        assert Path(mt_script).exists(), f"Not found: {mt_script}"

        mt_grid = list(itertools.product(
            args.archs, args.modalities, args.aux_lambdas, args.seeds))

        mt_total = len(mt_grid) * n_fold_runs
        print(f"\n--- Multi-task sweep: {len(mt_grid)} configs x "
              f"{n_fold_runs} folds = {mt_total} runs ---")
        mt_idx = 0
        for fold in folds:
            for i, (arch, modality, aux_lam, seed) in enumerate(mt_grid):
                mt_idx += 1
                lam_tag = f"lam{aux_lam:.1f}".replace(".", "")
                fold_tag = f"_f{fold}" if fold is not None else ""
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_name = (f"mt_{arch}_{modality}_{lam_tag}"
                            f"_seed{seed}{fold_tag}_{ts}")
                run_dir = runs_root / run_name

                if _run_completed(run_dir) and not args.overwrite:
                    print(f"[{mt_idx}/{mt_total}] SKIP {run_name} (completed)")
                    continue
                if run_dir.exists() and not _run_completed(run_dir):
                    print(f"[{mt_idx}/{mt_total}] RETRY {run_name} (crashed)")
                    shutil.rmtree(run_dir)

                cmd = [
                    sys.executable, mt_script,
                    "--index_csv", args.index_csv,
                    "--outdir", str(run_dir),
                    "--model", arch,
                    "--modality", modality,
                    "--aux_lambda", str(aux_lam),
                    "--seed", str(seed),
                    "--split_seed", str(args.split_seed),
                    "--epochs", str(args.epochs),
                    "--lr", str(args.lrs[0]),
                    "--wd", str(args.wd),
                    "--patience", str(args.patience),
                    "--gpu", str(args.gpu),
                    "--scheduler", args.scheduler,
                    "--warmup_epochs", str(args.warmup_epochs),
                ]
                if args.batch > 0:
                    cmd += ["--batch", str(args.batch)]
                if fold is not None:
                    cmd += ["--n_folds", str(args.n_folds),
                             "--fold", str(fold)]
                if args.no_amp:
                    cmd.append("--no_amp")
                if args.max_cases > 0:
                    cmd += ["--max_cases", str(args.max_cases)]

                print(f"[{mt_idx}/{mt_total}] {run_name}")
                if args.dry_run:
                    print(f"  CMD: {' '.join(cmd)}")
                    continue

                result = subprocess.run(cmd)
                if result.returncode != 0:
                    print(f"  FAILED (exit {result.returncode})")
                else:
                    print(f"  DONE")
                print()

        print(f"Multi-task sweep finished. {mt_total} runs configured.")
        return

    # ---- Tier 9: SSL pretrain + fine-tune ----
    if args.tier == 9:
        pt_script = str(Path(__file__).parent / "pretrain_ssl.py")
        ft_script = str(Path(__file__).parent / "train_ssl_finetune.py")
        assert Path(pt_script).exists(), f"Not found: {pt_script}"
        assert Path(ft_script).exists(), f"Not found: {ft_script}"

        ssl_grid = list(itertools.product(
            args.ssl_methods, args.archs, args.ssl_inits, args.seeds))

        print(f"\n--- SSL sweep: {len(ssl_grid)} configs "
              f"(pretrain once + fine-tune x {n_fold_runs} folds each) ---")

        for i, (method, arch, init, seed) in enumerate(ssl_grid):
            init_tag = "imn" if init == "imagenet" else "rnd"
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Phase 1: pretrain (once per config, no fold loop)
            pt_name = f"ssl_{method}_{arch}_{init_tag}_seed{seed}_{ts}"
            pt_dir = runs_root / pt_name

            if _run_completed(pt_dir) and not args.overwrite:
                print(f"[{i+1}/{len(ssl_grid)}] SKIP pretrain {pt_name}")
            else:
                if pt_dir.exists() and not _run_completed(pt_dir):
                    print(f"[{i+1}/{len(ssl_grid)}] RETRY pretrain "
                          f"{pt_name} (crashed)")
                    shutil.rmtree(pt_dir)
                pt_cmd = [
                    sys.executable, pt_script,
                    "--index_csv", args.index_csv,
                    "--outdir", str(pt_dir),
                    "--method", method,
                    "--model", arch,
                    "--epochs", str(args.ssl_epochs),
                    "--seed", str(seed),
                    "--gpu", str(args.gpu),
                    "--scheduler", "cosine",
                    "--warmup_epochs", str(args.warmup_epochs),
                ]
                if init == "imagenet":
                    pt_cmd.append("--imagenet_init")
                if args.no_amp:
                    pt_cmd.append("--no_amp")
                if args.max_cases > 0:
                    pt_cmd += ["--max_cases", str(args.max_cases)]

                print(f"[{i+1}/{len(ssl_grid)}] PRETRAIN {pt_name}")
                if args.dry_run:
                    print(f"  CMD: {' '.join(pt_cmd)}")
                else:
                    result = subprocess.run(pt_cmd)
                    if result.returncode != 0:
                        print(f"  PRETRAIN FAILED (exit {result.returncode})")
                        continue
                    print(f"  PRETRAIN DONE")

            # Phase 2: fine-tune (with fold loop)
            ssl_ckpt = pt_dir / "last.pt"
            for fold in folds:
                fold_tag = f"_f{fold}" if fold is not None else ""
                ft_name = (f"ft_{method}_{arch}_{init_tag}"
                           f"_seed{seed}{fold_tag}_{ts}")
                ft_dir = runs_root / ft_name

                if _run_completed(ft_dir) and not args.overwrite:
                    print(f"  SKIP finetune {ft_name} (completed)")
                    continue
                if ft_dir.exists() and not _run_completed(ft_dir):
                    print(f"  RETRY finetune {ft_name} (crashed)")
                    shutil.rmtree(ft_dir)

                ft_cmd = [
                    sys.executable, ft_script,
                    "--ssl_ckpt", str(ssl_ckpt),
                    "--ssl_method", method,
                    "--index_csv", args.index_csv,
                    "--outdir", str(ft_dir),
                    "--model", arch,
                    "--epochs", str(args.epochs),
                    "--lr", str(args.lrs[0]),
                    "--wd", str(args.wd),
                    "--seed", str(seed),
                    "--split_seed", str(args.split_seed),
                    "--patience", str(args.patience),
                    "--gpu", str(args.gpu),
                    "--scheduler", args.scheduler,
                    "--warmup_epochs", str(args.warmup_epochs),
                ]
                if args.batch > 0:
                    ft_cmd += ["--batch", str(args.batch)]
                if fold is not None:
                    ft_cmd += ["--n_folds", str(args.n_folds),
                                "--fold", str(fold)]
                if args.no_amp:
                    ft_cmd.append("--no_amp")
                if args.max_cases > 0:
                    ft_cmd += ["--max_cases", str(args.max_cases)]

                print(f"  FINETUNE {ft_name}")
                if args.dry_run:
                    print(f"  CMD: {' '.join(ft_cmd)}")
                else:
                    result = subprocess.run(ft_cmd)
                    if result.returncode != 0:
                        print(f"  FINETUNE FAILED (exit {result.returncode})")
                    else:
                        print(f"  FINETUNE DONE")
                print()

        print(f"SSL sweep finished. {len(ssl_grid)} configs configured.")
        return

    # ---- Tier 10: fusion strategies ----
    if args.tier == 10:
        fusion_script = str(Path(__file__).parent / "train_fusion.py")
        assert Path(fusion_script).exists(), f"Not found: {fusion_script}"

        fusion_grid = list(itertools.product(
            args.fusion_strategies, args.archs, args.seeds))

        fu_total = len(fusion_grid) * n_fold_runs
        print(f"\n--- Fusion sweep: {len(fusion_grid)} configs x "
              f"{n_fold_runs} folds = {fu_total} runs ---")
        fu_idx = 0
        for fold in folds:
            for i, (strat, arch, seed) in enumerate(fusion_grid):
                fu_idx += 1
                fold_tag = f"_f{fold}" if fold is not None else ""
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_name = (f"fusion_{strat}_{arch}"
                            f"_seed{seed}{fold_tag}_{ts}")
                run_dir = runs_root / run_name

                if _run_completed(run_dir) and not args.overwrite:
                    print(f"[{fu_idx}/{fu_total}] SKIP {run_name} (completed)")
                    continue
                if run_dir.exists() and not _run_completed(run_dir):
                    print(f"[{fu_idx}/{fu_total}] RETRY {run_name} (crashed)")
                    shutil.rmtree(run_dir)

                cmd = [
                    sys.executable, fusion_script,
                    "--index_csv", args.index_csv,
                    "--outdir", str(run_dir),
                    "--model", arch,
                    "--fusion_strategy", strat,
                    "--seed", str(seed),
                    "--split_seed", str(args.split_seed),
                    "--epochs", str(args.epochs),
                    "--lr", str(args.lrs[0]),
                    "--wd", str(args.wd),
                    "--patience", str(args.patience),
                    "--gpu", str(args.gpu),
                    "--scheduler", args.scheduler,
                    "--warmup_epochs", str(args.warmup_epochs),
                ]
                if args.batch > 0:
                    cmd += ["--batch", str(args.batch)]
                if fold is not None:
                    cmd += ["--n_folds", str(args.n_folds),
                             "--fold", str(fold)]
                if args.no_amp:
                    cmd.append("--no_amp")
                if args.max_cases > 0:
                    cmd += ["--max_cases", str(args.max_cases)]

                print(f"[{fu_idx}/{fu_total}] {run_name}")
                if args.dry_run:
                    print(f"  CMD: {' '.join(cmd)}")
                    continue

                result = subprocess.run(cmd)
                if result.returncode != 0:
                    print(f"  FAILED (exit {result.returncode})")
                else:
                    print(f"  DONE")
                print()

        print(f"Fusion sweep finished. {fu_total} runs configured.")
        return

    # ---- Tier 11: MC Dropout UQ ----
    if args.tier == 11:
        mc_script = str(Path(__file__).parent / "train_mc_dropout.py")
        uq_script = str(Path(__file__).parent / "evaluate_uncertainty.py")
        assert Path(mc_script).exists(), f"Not found: {mc_script}"
        assert Path(uq_script).exists(), f"Not found: {uq_script}"

        uq_grid = list(itertools.product(
            args.archs, args.dropout_rates, args.seeds))

        uq_total = len(uq_grid) * n_fold_runs
        print(f"\n--- MC Dropout UQ sweep: {len(uq_grid)} configs x "
              f"{n_fold_runs} folds = {uq_total} runs "
              f"(train + UQ eval each) ---")

        uq_idx = 0
        for fold in folds:
            for i, (arch, dr, seed) in enumerate(uq_grid):
                uq_idx += 1
                dr_tag = f"d{dr:.0e}".replace(".", "").replace("+", "")
                fold_tag = f"_f{fold}" if fold is not None else ""
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_name = f"mc_{arch}_{dr_tag}_seed{seed}{fold_tag}_{ts}"
                run_dir = runs_root / run_name

                if _run_completed(run_dir) and not args.overwrite:
                    print(f"[{uq_idx}/{uq_total}] SKIP {run_name} (completed)")
                    continue
                if run_dir.exists() and not _run_completed(run_dir):
                    print(f"[{uq_idx}/{uq_total}] RETRY {run_name} (crashed)")
                    shutil.rmtree(run_dir)

                # Phase 1: train MC dropout model
                train_cmd = [
                    sys.executable, mc_script,
                    "--index_csv", args.index_csv,
                    "--outdir", str(run_dir),
                    "--model", arch,
                    "--dropout_rate", str(dr),
                    "--seed", str(seed),
                    "--split_seed", str(args.split_seed),
                    "--epochs", str(args.epochs),
                    "--lr", str(args.lrs[0]),
                    "--wd", str(args.wd),
                    "--patience", str(args.patience),
                    "--gpu", str(args.gpu),
                    "--scheduler", args.scheduler,
                    "--warmup_epochs", str(args.warmup_epochs),
                ]
                if args.batch > 0:
                    train_cmd += ["--batch", str(args.batch)]
                if fold is not None:
                    train_cmd += ["--n_folds", str(args.n_folds),
                                   "--fold", str(fold)]
                if args.no_amp:
                    train_cmd.append("--no_amp")
                if args.max_cases > 0:
                    train_cmd += ["--max_cases", str(args.max_cases)]

                print(f"[{uq_idx}/{uq_total}] TRAIN {run_name}")
                if args.dry_run:
                    print(f"  CMD: {' '.join(train_cmd)}")
                else:
                    result = subprocess.run(train_cmd)
                    if result.returncode != 0:
                        print(f"  TRAIN FAILED (exit {result.returncode})")
                        continue
                    print(f"  TRAIN DONE")

                # Phase 2: evaluate uncertainty
                uq_dir = runs_root / f"uq_{run_name}"
                ckpt_path = run_dir / "best_auroc.pt"
                uq_cmd = [
                    sys.executable, uq_script,
                    "--index_csv", args.index_csv,
                    "--outdir", str(uq_dir),
                    "--mc_dropout_ckpt", str(ckpt_path),
                    "--n_forward", "50",
                ]
                if fold is not None:
                    uq_cmd += ["--n_folds", str(args.n_folds),
                                "--fold", str(fold)]

                print(f"  UQ EVAL {run_name}")
                if args.dry_run:
                    print(f"  CMD: {' '.join(uq_cmd)}")
                else:
                    result = subprocess.run(uq_cmd)
                    if result.returncode != 0:
                        print(f"  UQ EVAL FAILED (exit {result.returncode})")
                    else:
                        print(f"  UQ EVAL DONE")
                print()

        print(f"MC Dropout UQ sweep finished. {uq_total} runs configured.")
        return

    # ---- Tier 12: Learning curves ----
    if args.tier == 12:
        lc_fractions = getattr(args, "train_fractions", [0.25, 0.5, 0.75, 1.0])
        lc_grid = list(itertools.product(
            args.archs, args.seeds, lc_fractions))
        lc_total = len(lc_grid) * n_fold_runs
        print(f"\n--- Learning curve sweep: {len(lc_grid)} configs x "
              f"{n_fold_runs} folds = {lc_total} runs ---")

        lc_idx = 0
        for fold in folds:
            for i, (arch, seed, frac) in enumerate(lc_grid):
                lc_idx += 1
                frac_tag = f"frac{int(frac * 100):03d}"
                fold_tag = f"_f{fold}" if fold is not None else ""
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_name = (f"lc_{arch}_{frac_tag}"
                            f"_seed{seed}{fold_tag}_{ts}")
                run_dir = runs_root / run_name

                if _run_completed(run_dir) and not args.overwrite:
                    print(f"[{lc_idx}/{lc_total}] SKIP {run_name} (completed)")
                    continue
                if run_dir.exists() and not _run_completed(run_dir):
                    print(f"[{lc_idx}/{lc_total}] RETRY {run_name} (crashed)")
                    shutil.rmtree(run_dir)

                cmd = [
                    sys.executable, train_script,
                    "--index_csv", args.index_csv,
                    "--outdir", str(run_dir),
                    "--model", arch,
                    "--seed", str(seed),
                    "--split_seed", str(args.split_seed),
                    "--epochs", str(args.epochs),
                    "--lr", str(args.lrs[0]),
                    "--wd", str(args.wd),
                    "--patience", str(args.patience),
                    "--gpu", str(args.gpu),
                    "--scheduler", args.scheduler,
                    "--warmup_epochs", str(args.warmup_epochs),
                    "--train_fraction", str(frac),
                ]
                if args.batch > 0:
                    cmd += ["--batch", str(args.batch)]
                if fold is not None:
                    cmd += ["--n_folds", str(args.n_folds),
                             "--fold", str(fold)]
                if args.no_amp:
                    cmd.append("--no_amp")
                if args.max_cases > 0:
                    cmd += ["--max_cases", str(args.max_cases)]

                print(f"[{lc_idx}/{lc_total}] {run_name}")
                if args.dry_run:
                    print(f"  CMD: {' '.join(cmd)}")
                    continue

                result = subprocess.run(cmd)
                if result.returncode != 0:
                    print(f"  FAILED (exit {result.returncode})")
                else:
                    print(f"  DONE")
                print()

        print(f"Learning curve sweep finished. {lc_total} runs configured.")
        return

    # ---- Tier 13: Leave-one-b-value-out generalization ----
    if args.tier == 13:
        exclude_bvals = getattr(args, "exclude_bvalues",
                                ["1k4", "1k5", "1k6", "2k"])
        lobo_grid = list(itertools.product(
            args.archs, exclude_bvals, args.seeds))
        lobo_total = len(lobo_grid) * n_fold_runs
        print(f"\n--- Leave-one-b-value-out sweep: {len(lobo_grid)} configs x "
              f"{n_fold_runs} folds = {lobo_total} runs ---")

        lobo_idx = 0
        for fold in folds:
            for i, (arch, bval, seed) in enumerate(lobo_grid):
                lobo_idx += 1
                fold_tag = f"_f{fold}" if fold is not None else ""
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_name = (f"lobo_{arch}_excl{bval}"
                            f"_seed{seed}{fold_tag}_{ts}")
                run_dir = runs_root / run_name

                if _run_completed(run_dir) and not args.overwrite:
                    print(f"[{lobo_idx}/{lobo_total}] SKIP {run_name} "
                          f"(completed)")
                    continue
                if run_dir.exists() and not _run_completed(run_dir):
                    print(f"[{lobo_idx}/{lobo_total}] RETRY {run_name} "
                          f"(crashed)")
                    shutil.rmtree(run_dir)

                cmd = [
                    sys.executable, train_script,
                    "--index_csv", args.index_csv,
                    "--outdir", str(run_dir),
                    "--model", arch,
                    "--seed", str(seed),
                    "--split_seed", str(args.split_seed),
                    "--epochs", str(args.epochs),
                    "--lr", str(args.lrs[0]),
                    "--wd", str(args.wd),
                    "--patience", str(args.patience),
                    "--gpu", str(args.gpu),
                    "--scheduler", args.scheduler,
                    "--warmup_epochs", str(args.warmup_epochs),
                    "--exclude_bvalue", bval,
                ]
                if args.batch > 0:
                    cmd += ["--batch", str(args.batch)]
                if fold is not None:
                    cmd += ["--n_folds", str(args.n_folds),
                             "--fold", str(fold)]
                if args.no_amp:
                    cmd.append("--no_amp")
                if args.max_cases > 0:
                    cmd += ["--max_cases", str(args.max_cases)]

                print(f"[{lobo_idx}/{lobo_total}] {run_name}")
                if args.dry_run:
                    print(f"  CMD: {' '.join(cmd)}")
                    continue

                result = subprocess.run(cmd)
                if result.returncode != 0:
                    print(f"  FAILED (exit {result.returncode})")
                else:
                    print(f"  DONE")
                print()

        print(f"Leave-one-b-value-out sweep finished. "
              f"{lobo_total} runs configured.")
        return

    print(f"Sweep finished. {len(grid)} runs configured.")


if __name__ == "__main__":
    main()
