# MIQA: Automated Detection of Geometric Distortion in Prostate DWI

Research code for the MSc thesis *"AI for Medical Image Quality Assessment:
Deep Learning for Automated Detection of Geometric Distortion in Prostate
Diffusion-Weighted MRI"* (UCL, MPHY0034).

The repository contains the full training, evaluation and analysis pipeline used
to benchmark convolutional architectures for binary distortion detection and to
study the reliability of their predictive uncertainty (the architecture-uncertainty
dissociation, its ensemble-homogeneity mechanism, and its resolution with a
heterogeneous ensemble and a distilled student).

## Data availability

The underlying data, a retrospective, anonymised single-centre cohort of prostate
DWI acquisitions from University College London Hospital, **cannot be shared** for
patient-confidentiality reasons and is therefore **not included** in this
repository. The code expects a CSV index of acquisitions (built by
`scripts/build_ucl_index_full.py`); paths to data and run directories are supplied
through the environment variables `MIQA_DATA_ROOT`, `MIQA_RUNS` and `MIQA_CKPTS`.

## Installation

```bash
python -m pip install -e .          # installs the `miqa` package + dependencies
python -m pip install -e ".[dev]"   # also installs pytest for the test suite
```

Requires Python >= 3.9 and PyTorch >= 2.0 (see `pyproject.toml` for the full list).

## Layout

```
src/miqa/            installable Python package
  models.py            build_model: ResNet-18/50, DenseNet-121, EfficientNet-B0
  datasets.py          UCLDistortionDataset (patient-level, 2.5D input)
  metrics.py           AUROC/AUPRC, bootstrap CIs, expected calibration error, UE-AUROC
  uncertainty.py       MC dropout, deep ensembles, selective prediction, UQ metrics
  multitask.py         auxiliary b-value head
  fusion.py            DWI + T2 fusion variants
  ssl.py               SimCLR / BYOL self-supervised pretraining
  schedulers.py        cosine schedule with warm-up
  calibration.py       reliability diagrams
  utils.py             seeding, patient splits, AMP compatibility shim
scripts/             command-line entry points (training, evaluation, analysis, figures)
tests/               CPU-only pytest suite on synthetic NIfTI data
pyproject.toml       package + dependency definition
```

## Reproducing the main experiments

All experiments use strict **patient-level five-fold cross-validation**; whole
patients are held out so no patient contributes to both training and test sets.

```bash
# Single-architecture baseline sweep (four backbones, patient-level 5-fold CV)
python scripts/sweep.py

# Uncertainty evaluation (MC dropout / deep ensemble) for a trained configuration
python scripts/evaluate_uncertainty.py

# Heterogeneous 9-model ensemble and selective-prediction analysis
python scripts/analyze_heterogeneous_ensemble.py

# Distilled ResNet-18 student from the ensemble teacher
python scripts/train_distill.py
```

Every training run writes a machine-readable `summary.json` on completion or
failure so that sweeps can resume idempotently and distinguish completed, failed
and crashed runs.

## Tests

```bash
pytest            # CPU-only; uses synthetic NIfTI fixtures, no real data required
```

The suite specifically guards the patient-level splitting logic and the metric
implementations on which the headline results depend.

## Citation

If you use this code, please cite the associated thesis (2026). A `CITATION.cff`
or archived-release DOI can be added on publication.

## AI use

As per Category 2, AI (Claude was used as a proofreader).

## License

Released under the MIT License (see `LICENSE`).
