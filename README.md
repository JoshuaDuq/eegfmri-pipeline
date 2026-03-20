# EEG–fMRI Analysis Pipeline

[![Python ≥ 3.11](https://img.shields.io/badge/python-≥3.11-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![BIDS](https://img.shields.io/badge/data-BIDS-orange.svg)](https://bids-specification.readthedocs.io/)
[![MNE-Python](https://img.shields.io/badge/MNE--Python-1.9.0-informational.svg)](https://mne.tools)

A modular EEG/fMRI analysis suite for multimodal neuroimaging research.
It covers EEG preprocessing, feature extraction, behavioral analysis,
machine learning, fMRI preprocessing, first-level and second-level GLM,
trial-wise beta/signature extraction — all available from a unified CLI and an interactive TUI. 

This pipeline is still in very early development and is subject to change. 
Please send any suggestions my way :)



<p align="center">
  <img src="docs/screenshots/tui_main_menu.png" width="800" alt="Main menu — All pipeline stages at a glance"/>
</p>



---

## Table of Contents

1. [Documentation](#1-documentation)
2. [Quick Start](#2-quick-start)
3. [Installation](#3-installation)
4. [Data Requirements](#4-data-requirements)
5. [Project Structure](#5-project-structure)
6. [CLI Reference](#6-cli-reference)
   - 6.1 [Preprocessing](#61-preprocessing)
   - 6.2 [Feature Extraction](#62-feature-extraction)
   - 6.3 [Behavioral Analysis](#63-behavioral-analysis)
   - 6.4 [Machine Learning](#64-machine-learning)
   - 6.5 [fMRI Preprocessing](#65-fmri-preprocessing)
   - 6.6 [fMRI Analysis](#66-fmri-analysis)
   - 6.7 [Plotting](#67-plotting)
   - 6.8 [Validation](#68-validation)
   - 6.9 [Stats and Info](#69-stats-and-info)
   - 6.10 [Coupling](#610-coupling)
7. [Interactive TUI](#7-interactive-tui)
8. [Configuration](#8-configuration)
9. [Subject Selection](#9-subject-selection)
10. [Output Formats](#10-output-formats)
11. [Advanced Topics](#11-advanced-topics)
12. [Dependencies](#12-dependencies)
13. [License](#13-license)
14. [Contributing](#14-contributing)

---

## 1. Documentation

Each major module has a dedicated README with methods, formulas, configuration
references, and output schemas. Start here for orientation, then follow the link
relevant to your workflow:

| Module | README |
|--------|--------|
| Architecture boundaries | [docs/architecture/README.md](docs/architecture/README.md) |
| EEG Preprocessing | [eeg_pipeline/preprocessing/README.md](eeg_pipeline/preprocessing/README.md) |
| Feature Extraction | [eeg_pipeline/analysis/features/README.md](eeg_pipeline/analysis/features/README.md) |
| Behavioral Analysis | [eeg_pipeline/analysis/behavior/README.md](eeg_pipeline/analysis/behavior/README.md) |
| Machine Learning | [eeg_pipeline/analysis/machine_learning/README.md](eeg_pipeline/analysis/machine_learning/README.md) |
| fMRI Analysis | [fmri_pipeline/README.md](fmri_pipeline/README.md) |
| EEG–BOLD coupling study | [studies/pain_study/scripts/README.md](studies/pain_study/scripts/README.md) |
| Interactive TUI | [eeg_pipeline/cli/tui/README.md](eeg_pipeline/cli/tui/README.md) |
| Tests | [tests/README.md](tests/README.md) |


---

## 2. Quick Start

```bash
# 1. Clone and install
git clone https://github.com/JoshuaDuq/eegfmri-pipeline.git && cd eegfmri-pipeline
python -m venv .venv311
source .venv311/bin/activate
pip install -e ".[dev,ml]"

# 2. Place your data (see §4 Data Requirements)

# 3. Inspect the CLI
eeg-pipeline --help

# 4. Launch the interactive TUI
cd eeg_pipeline/cli/tui && go build -o eeg-tui . && ./eeg-tui
```

---

## 3. Installation

**Requirements:** Python ≥ 3.11.

```bash
python -m venv .venv311
source .venv311/bin/activate
pip install -e ".[dev,ml]"
```

The repository also ships a `requirements.txt` shim for tooling that expects one:

```bash
pip install -r requirements.txt
```

After installation, the `eeg-pipeline` console script and `python -m eeg_pipeline`
entry point are both available in the active environment.

---

## 4. Data Requirements

The pipeline expects data organized under `data/` at the repository root.
All paths are configurable in `eeg_pipeline/utils/config/eeg_config.yaml`, with
fMRI-specific defaults in `fmri_pipeline/utils/config/fmri_config.yaml` where applicable.

### 4.1 Minimum Required Inputs by Workflow

| Workflow | Minimum required inputs |
|----------|-------------------------|
| EEG preprocessing / features from BIDS | `data/bids_output/eeg/sub-XXXX/eeg/*_eeg.<format>` plus matching BIDS sidecars |
| Behavioral analysis / ML | EEG `*_events.tsv` with at least `onset`, `duration`, `trial_type`, plus study-specific predictor/target columns |
| EEG raw input | `data/source_data/sub-XXXX/eeg/*.vhdr` with matching `.vmrk` and `.eeg` files |
| External event-log merge | Study logs such as PsychoPy `*TrialSummary.csv` plus BIDS `*_events.tsv` |
| fMRI raw input | `data/source_data/sub-XXXX/fmri/<dicom-series-dir>/` |
| fMRI first-level analysis | `data/bids_output/fmri/sub-XXXX/func/*_bold.nii.gz` plus `*_events.tsv` |

### 4.2 Starting from Raw Recordings

If you start from raw BrainVision EEG and fMRI DICOMs, perform raw-to-BIDS conversion
and event-log merging before running this pipeline (EEG → BIDS, fMRI → BIDS,
behavioral logs → `events.tsv`). These steps are intentionally external to this repo
and may differ by dataset.

### 4.3 Starting from BIDS-Compliant Data

If your data is already in [BIDS format](https://bids-specification.readthedocs.io/)
(e.g., converted with MNE-BIDS or BIDScoin), place it directly under `bids_output/`
and skip the raw-to-BIDS step:

```
data/bids_output/eeg/
├── dataset_description.json
├── participants.tsv
└── sub-XXXX/
    └── eeg/
        ├── sub-XXXX_task-YYY_run-01_eeg.vhdr   (or .set, .edf, .fif)
        ├── sub-XXXX_task-YYY_run-01_eeg.vmrk
        ├── sub-XXXX_task-YYY_run-01_eeg.eeg
        ├── sub-XXXX_task-YYY_run-01_events.tsv         # raw BIDS events; clean aligned events are written to derivatives
        ├── sub-XXXX_task-YYY_run-01_channels.tsv        # recommended
        └── sub-XXXX_task-YYY_run-01_electrodes.tsv      # recommended
```

### 4.4 Behavioral and Events Data

Behavior and ML workflows read trial-level predictors and outcomes from BIDS
`*_events.tsv` files. Required columns: `onset`, `duration`, `trial_type`.
After preprocessing, the pipeline writes `*_proc-clean_events.tsv` in derivatives
with a canonical `trial_id` column. Trialwise feature tables and behavior
analysis use `trial_id` as the only supported alignment contract.
Include any study-specific columns you plan to model (e.g., ratings, stimulus
parameters, binary outcome labels).

For fMRI contrast workflows, ensure events files include whichever columns you
reference via `--cond-a-column/--cond-a-value` and `--cond-b-column/--cond-b-value`.

### 4.5 fMRI Data (Optional)

```
data/source_data/sub-XXXX/fmri/   # Raw DICOMs (for fMRI conversion)
data/bids_output/fmri/             # BIDS-formatted fMRI (NIfTI + events)
data/fMRI_data/sub-XXXX/anat/      # T1w anatomical (for FreeSurfer / source localization)
```

Minimal BIDS-fMRI layout for `fmri-analysis`:

```
data/bids_output/fmri/
└── sub-XXXX/
    └── func/
        ├── sub-XXXX_task-task_run-01_bold.nii.gz
        └── sub-XXXX_task-task_run-01_events.tsv
```

### 4.6 Default Directory Layout

```
data/
├── source_data/                # Raw recordings (EEG .vhdr, fMRI DICOMs)
│   └── sub-XXXX/
│       ├── eeg/                # BrainVision triplets (.vhdr/.vmrk/.eeg)
│       └── fmri/               # DICOM series folders
├── bids_output/
│   ├── eeg/                    # BIDS-formatted EEG
│   └── fmri/                   # BIDS-formatted fMRI
├── fMRI_data/                  # Subject anatomicals (T1w)
│   └── sub-XXXX/anat/
└── derivatives/                # All pipeline outputs
    ├── preprocessed/
    │   ├── eeg/                # ICA components, bad channel logs
    │   └── fmri/               # fMRIPrep outputs
    ├── freesurfer/             # FreeSurfer reconstructions (source localization)
    ├── group/
    │   └── fmri/
    │       └── second_level/   # Group GLM inference maps
    └── sub-XXXX/
        ├── eeg/
        │   ├── sub-XXXX_task-*_proc-clean_epo.fif   # Cleaned epochs
        │   └── features/       # Extracted feature tables (.parquet; optionally .tsv/.csv)
        │       ├── power/
        │       ├── connectivity/
        │       ├── aperiodic/
        │       └── ...
        └── fmri/
            ├── first_level/    # GLM contrast maps
            ├── beta_series/    # Trial-wise beta estimates
            └── lss/            # Least-squares-separate betas
```

All paths are configurable via the `paths` section in `eeg_config.yaml`:

```yaml
paths:
  bids_root:       "../../../data/bids_output/eeg"
  bids_fmri_root:  "../../../data/bids_output/fmri"
  deriv_root:      "../../../data/derivatives"
  source_data:     "../../../data/source_data"
  freesurfer_dir:  "../../../data/derivatives/freesurfer"
```

---

## 5. Project Structure

```
eegfmri-pipeline/
├── eeg_pipeline/               # EEG analysis package and shared CLI
│   ├── analysis/               # EEG feature, behavior, and ML modules
│   ├── cli/                    # Command-line entry points and Go TUI
│   ├── context/                # Context builders for downstream pipelines
│   ├── domain/                 # Feature naming schema, registry, and constants
│   ├── infra/                  # Paths, logging, TSV/Parquet I/O helpers
│   ├── pipelines/              # Pipeline orchestration
│   ├── plotting/               # Visualization modules and plot catalog
│   ├── preprocessing/          # EEG preprocessing stages (bad channels, ICA, epochs)
│   └── utils/                  # Configuration, data discovery, validation
├── fmri_pipeline/              # fMRI preprocessing and analysis package
├── studies/                    # Study-specific workflows and utilities
├── scripts/                    # Standalone utility scripts
├── tests/                      # Test suite and repository guards
├── docs/                       # Architecture notes and guides
├── pyproject.toml              # Single source of truth for packaging
└── requirements.txt            # Editable-install shim for tooling compatibility
```

---

## 6. CLI Reference

Most commands follow this pattern:

```bash
eeg-pipeline <command> [mode] [--subject XXXX | --all-subjects] [options]
```

Use `--help` on any command for full option details:

```bash
eeg-pipeline <command> --help
```

**Command overview:**

| Command | Primary role |
|---------|-------------|
| `preprocessing` | EEG cleaning, bad channels, ICA, and epoching (`full`, `bad-channels`, `ica`, `epochs`) |
| `features` | EEG feature extraction and visualization (`compute`, `visualize`) |
| `behavior` | Behavioral statistics and plots (`compute`, `visualize`) |
| `ml` | Trial-level predictive modeling (`regression`, `timegen`, `classify`, `model_comparison`, `incremental_validity`, `uncertainty`, `shap`, `permutation`) |
| `fmri` | Containerized fMRIPrep preprocessing (`preprocess`) |
| `fmri-analysis` | First-level GLM, second-level inference, beta-series, LSS, and resting-state connectivity (`first-level`, `second-level`, `beta-series`, `lss`, `rest`) |
| `plotting` | Visualization suites and TFR plots (`visualize`, `tfr`) |
| `validate` | Data and derivative integrity checks |
| `stats` | Pipeline-wide coverage summaries |
| `info` | Read-only inspection of configuration and derived state |
| `coupling` | Study-specific EEG–BOLD coupling workflow |

---

### 6.1 Preprocessing

Automated EEG preprocessing: bad channel detection, ICA artifact removal, and epoching.

| Mode | Description |
|------|-------------|
| `full` | Run all preprocessing steps sequentially |
| `bad-channels` | Detect and interpolate bad channels only |
| `ica` | Fit and apply ICA only |
| `epochs` | Create epochs only |

```bash
# Full preprocessing pipeline
eeg-pipeline preprocessing full --subject 0001

# Bad channel detection (PyPREP + RANSAC)
eeg-pipeline preprocessing bad-channels --subject 0001 --ransac

# Custom epoch window with autoreject
eeg-pipeline preprocessing epochs --subject 0001 \
  --tmin -7.0 --tmax 15.0 --reject-method autoreject_local

# Without ICALabel (fall back to MNE-BIDS pipeline detection)
eeg-pipeline preprocessing full --subject 0001 --no-icalabel

# SSP instead of ICA for artifact removal
eeg-pipeline preprocessing full --subject 0001 --spatial-filter ssp

# Resting-state mode (fixed-length epochs; no event conditions required)
eeg-pipeline preprocessing full --subject 0001 --task-is-rest

# Write clean events.tsv aligned to kept epochs (for downstream alignment)
eeg-pipeline preprocessing full --subject 0001 --write-clean-events

# EEG–fMRI simultaneous acquisition: trim EEG to first fMRI volume
eeg-pipeline preprocessing epochs --subject 0001 --trim-to-first-volume

# Disable automatic break detection
eeg-pipeline preprocessing full --subject 0001 --no-find-breaks
```

Full pipeline steps, CLI options, and configuration details:
[eeg_pipeline/preprocessing/README.md](eeg_pipeline/preprocessing/README.md)

---

### 6.2 Feature Extraction

Extract trial-level EEG features from cleaned epochs.
Each feature family produces one row per trial with clearly documented columns.

| Mode | Description |
|------|-------------|
| `compute` | Extract features and write to derivatives (Parquet; optionally TSV/CSV via `--also-save-csv`) plus provenance metadata |
| `visualize` | Generate descriptive plots from already-computed tables |

**Feature families (16 categories):**

| Category | What is quantified |
|----------|--------------------|
| `power` | Band-limited oscillatory power (delta, theta, alpha, beta, gamma), baseline-normalized |
| `spectral` | Spectral summary measures (spectral edge, peak frequency, bandwidth) |
| `ratios` | Band power ratios (theta/beta, theta/alpha, alpha/beta, delta/alpha, delta/theta) |
| `aperiodic` | 1/f background (slope, offset) via iterative specparam-style fits |
| `connectivity` | Functional connectivity (wPLI, imcoh, AEC, PLV, PLI) with per-family spatial transforms |
| `directedconnectivity` | Directed connectivity (PSI, DTF, PDC) from MVAR models |
| `microstates` | Microstate sequence statistics (coverage, duration, occurrence, transitions) |
| `pac` | Phase–amplitude coupling (theta–gamma, alpha–gamma) with surrogate-based nulls |
| `itpc` | Inter-trial phase coherence with CV-safe aggregation modes |
| `erp` | ERP amplitudes within configurable component windows (N1, N2, P2, …) |
| `bursts` | Transient oscillatory bursts identified by envelope thresholding |
| `complexity` | Signal complexity (permutation entropy, sample entropy, MSE, LZC) |
| `asymmetry` | Hemispheric asymmetry indices for canonical electrode pairs |
| `erds` | Event-related desynchronization/synchronization relative to baseline |
| `quality` | Data quality indicators (SNR, muscle artifact burden, line noise) |
| `sourcelocalization` | Source-space features from LCMV beamformer or eLORETA solutions |

All feature computations are configurable through the TUI or YAML.
For exact formulas and configuration details, see
[eeg_pipeline/analysis/features/README.md](eeg_pipeline/analysis/features/README.md).

```bash
# All feature categories
eeg-pipeline features compute --subject 0001

# Specific categories with spatial modes
eeg-pipeline features compute --subject 0001 \
  --categories power connectivity aperiodic \
  --spatial roi global

# When --spatial is omitted, config's feature_engineering.spatial_modes is used
# (default: roi, global). Use --spatial to restrict to a subset.

# Custom frequency bands and ROIs
eeg-pipeline features compute --subject 0001 \
  --frequency-bands "mu:8.0:13.0" "high_beta:20.0:30.0" \
  --rois "Motor:C3,C4,Cz" "Occipital:O1,O2,Oz"

# ML-safe mode (prevents cross-trial leakage)
eeg-pipeline features compute --subject 0001 --analysis-mode trial_ml_safe

# CSD spatial transform for phase-based features
eeg-pipeline features compute --subject 0001 --spatial-transform csd

# Visualize extracted features
eeg-pipeline features visualize --subject 0001
```

---

### 6.3 Behavioral Analysis

Statistical analyses linking EEG features to behavior (pain ratings, temperature,
conditions). All stages operate on a trial table with explicit column semantics.
The trialwise join contract is canonical `trial_id`, not inferred paradigm
columns.

| Mode | Description |
|------|-------------|
| `compute` | Run behavioral analysis stages and save numerical outputs |
| `visualize` | Generate standardized plots from previously computed results |

For all 17 stage definitions, the pipeline DAG, statistical safeguards, and
configuration details, see
[eeg_pipeline/analysis/behavior/README.md](eeg_pipeline/analysis/behavior/README.md).

```bash
# All behavioral analyses
eeg-pipeline behavior compute --subject 0001

# Specific stages
eeg-pipeline behavior compute --subject 0001 \
  --computations correlations condition temporal

# Temperature-controlled with permutation testing
eeg-pipeline behavior compute --subject 0001 \
  --control-temperature --n-perm 1000

# Robust correlations with Bayes factors
eeg-pipeline behavior compute --subject 0001 \
  --robust-correlation percentage_bend --compute-bayes-factors

# Visualize
eeg-pipeline behavior visualize --subject 0001

# List available stages
eeg-pipeline behavior compute --list-stages
```

---

### 6.4 Machine Learning

Trial-level predictive modeling with leave-one-subject-out (LOSO) cross-validation.

| Mode | Description |
|------|-------------|
| `regression` | LOSO or within-subject regression for continuous outcomes (e.g., pain ratings) |
| `classify` | Binary classification (SVM, logistic regression, random forest, CNN) |
| `timegen` | Temporal generalization: train at one time window, evaluate across all windows |
| `model_comparison` | Compare ElasticNet vs Ridge vs RandomForest under a shared CV scheme |
| `incremental_validity` | Quantify ΔR² when adding EEG features over a baseline predictor |
| `uncertainty` | Conformal prediction intervals for calibrated uncertainty estimates |
| `shap` | SHAP-based feature importance |
| `permutation` | Permutation-based feature importance |

For the full architecture, CV schemes, evaluation metrics, and configuration details,
see [eeg_pipeline/analysis/machine_learning/README.md](eeg_pipeline/analysis/machine_learning/README.md).

```bash
# LOSO regression (requires ≥2 subjects)
eeg-pipeline ml regression --subject 0001 --subject 0002 --subject 0003

# Model family selection: elasticnet (default), ridge, or rf
eeg-pipeline ml regression --subject 0001 --subject 0002 --model ridge

# SVM classification with explicit binary threshold
eeg-pipeline ml classify --subject 0001 --subject 0002 \
  --classification-model svm --binary-threshold 30

# SHAP importance
eeg-pipeline ml shap --subject 0001 --subject 0002

# Within-subject CV
eeg-pipeline ml regression --subject 0001 --cv-scope subject

# Predict fMRI signature expression from EEG
eeg-pipeline ml regression --subject 0001 --subject 0002 \
  --target fmri_signature --fmri-signature-name SIGNATURE_A

# Model comparison with custom hyperparameters
eeg-pipeline ml model_comparison --subject 0001 --subject 0002 \
  --elasticnet-alpha-grid 0.01 0.1 1 10 \
  --rf-n-estimators 500

# Restrict to specific feature families and bands
eeg-pipeline ml regression --subject 0001 --subject 0002 \
  --feature-families power connectivity --feature-bands alpha beta

# Fine-grained feature filtering: scope, segment, stat
eeg-pipeline ml regression --subject 0001 --subject 0002 \
  --feature-scopes roi global --feature-segments active \
  --feature-stats wpli aec

# Feature harmonization across subjects (default: intersection)
eeg-pipeline ml regression --all-subjects \
  --feature-harmonization union_impute

# Append meta covariates to the feature matrix
eeg-pipeline ml regression --subject 0001 --subject 0002 \
  --covariates predictor trial_index

# Incremental validity: EEG over temperature baseline
eeg-pipeline ml incremental_validity --subject 0001 --subject 0002 \
  --baseline-predictors predictor

# Enforce ML-safe mode (prevents CV leakage from cross-trial features)
eeg-pipeline ml regression --subject 0001 --subject 0002 \
  --require-trial-ml-safe

# Pipeline preprocessing overrides
eeg-pipeline ml regression --subject 0001 --subject 0002 \
  --imputer mean --pca-enabled --pca-n-components 0.95 \
  --feature-selection-percentile 50
```

---

### 6.5 fMRI Preprocessing

Containerized fMRIPrep preprocessing via Docker or Apptainer.

```bash
# Docker
eeg-pipeline fmri preprocess --subject 0001 --engine docker

# Apptainer (HPC)
eeg-pipeline fmri preprocess --subject 0001 --engine apptainer

# Custom output spaces
eeg-pipeline fmri preprocess --subject 0001 \
  --output-spaces T1w MNI152NLin2009cAsym
```

**Key options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--engine` | `docker` or `apptainer` | `docker` |
| `--fmriprep-image` | Container image tag or URI | `nipreps/fmriprep:25.2.4` |
| `--output-spaces` | Output coordinate spaces | `MNI152NLin2009cAsym`, `T1w` |
| `--fs-license-file` | FreeSurfer license path | `paths.freesurfer_license`, else `EEG_PIPELINE_FREESURFER_LICENSE`, else `~/license.txt` |
| `--fs-subjects-dir` | FreeSurfer `SUBJECTS_DIR` | auto |
| `--ignore` | Skip steps (e.g., `fieldmaps slicetiming`) | none |
| `--bids-filter-file` | Optional BIDS filter JSON for subject/task/run selection | none |
| `--level` | Processing level: `full`, `resampling`, or `minimal` | `full` |
| `--cifti-output` | Output CIFTI dense timeseries: `91k` or `170k` | none |
| `--task-id` | Process only a specific task ID | all |
| `--nthreads` | Max threads across all processes (`0` = auto) | `0` |
| `--omp-nthreads` | Max threads per process (`0` = auto) | `0` |
| `--dummy-scans` | Non-steady-state volumes to discard | `0` |
| `--fd-spike-threshold` | Framewise displacement spike threshold (mm) | `0.5` |
| `--dvars-spike-threshold` | Standardized DVARS spike threshold | `1.5` |
| `--mem-mb` | Memory limit in MB | fMRIPrep default |
| `--fmriprep-extra-args` | Raw extra fMRIPrep CLI arguments (parsed with `shlex`) | none |
| `--use-aroma` / `--no-use-aroma` | Enable/disable ICA-AROMA | disabled |
| `--skip-bids-validation` | Skip bids-validator step | disabled |
| `--fs-no-reconall` | Disable FreeSurfer `recon-all` | enabled |
| `--longitudinal` | Create unbiased structural template (longitudinal mode) | disabled |

Note: for container runs, the pipeline automatically ignores macOS metadata files (`._*`, `.DS_Store`) by mounting a sanitized temporary BIDS view.

---

### 6.6 fMRI Analysis

Subject-level and group-level GLM analysis plus trial-wise beta estimation via nilearn.

| Mode | Description |
|------|--------------|
| `first-level` | First-level GLM with user-defined contrasts → contrast maps |
| `second-level` | Explicit group GLM from existing first-level MNI effect-size maps |
| `beta-series` | Trial-wise beta-series estimation (LSA method) |
| `lss` | Least-squares-separate (LSS) trial betas |
| `rest` | Resting-state ROI connectivity analysis (atlas-based, Fisher-z averaged across runs) |

For full methods, see [fmri_pipeline/README.md](fmri_pipeline/README.md).

```bash
# First-level GLM
eeg-pipeline fmri-analysis first-level --subject 0001 \
  --contrast-name contrast \
  --cond-a-value stimulation --cond-b-value fixation_rest

# With fMRIPrep preprocessed BOLD in MNI space
eeg-pipeline fmri-analysis first-level --subject 0001 \
  --input-source fmriprep --fmriprep-space MNI152NLin2009cAsym \
  --cond-a-value stimulation --cond-b-value fixation_rest

# Group mean from existing first-level MNI cope/effect-size maps
eeg-pipeline fmri-analysis second-level --subject 0001 --subject 0002 \
  --group-model one-sample \
  --group-contrast-names stimulation_vs_rest

# Beta-series for EEG–fMRI fusion
eeg-pipeline fmri-analysis beta-series --subject 0001 \
  --cond-a-value stimulation --cond-b-value fixation_rest

# LSS betas
eeg-pipeline fmri-analysis lss --subject 0001 \
  --cond-a-value stimulation --cond-b-value fixation_rest

# Resting-state ROI connectivity (atlas required)
eeg-pipeline fmri-analysis rest --subject 0001 \
  --atlas-labels-img /path/to/atlas_parc.nii.gz \
  --atlas-labels-tsv /path/to/atlas_labels.tsv

# Resting-state with custom bandpass and smoothing
eeg-pipeline fmri-analysis rest --subject 0001 \
  --atlas-labels-img /path/to/atlas_parc.nii.gz \
  --high-pass-hz 0.01 --low-pass-hz 0.08 --smoothing-fwhm 6.0

# With HTML report (task-based modes)
eeg-pipeline fmri-analysis first-level --subject 0001 \
  --cond-a-value stimulation --cond-b-value fixation_rest \
  --plots --plot-html-report
```

**Key options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--input-source` | `fmriprep` or `bids_raw` | `fmriprep` |
| `--hrf-model` | `spm`, `flobs`, `fir` | `spm` |
| `--confounds-strategy` | `auto`, `none`, `motion6`…`motion24+wmcsf+fd` | `auto` |
| `--smoothing-fwhm` | Spatial smoothing kernel (mm) | `5.0` |
| `--output-type` | `z-score`, `t-stat`, `cope`, `beta` | `z-score` |
| `--group-model` | `one-sample`, `two-sample`, `paired`, `repeated-measures` | `one-sample` |
| `--group-contrast-names` | First-level contrast names consumed by second-level mode | required for `second-level` |
| `--group-covariates-file` | Subject-level TSV/CSV for groups and covariates | none |
| `--group-permutation-inference` | Add max-T permutation inference to second-level mode | disabled |
| `--resample-to-freesurfer` | Resample to FreeSurfer space | disabled |
| `--plots` | Generate per-subject figures | disabled |
| `--plot-html-report` | Write self-contained HTML report | disabled |
| `--write-design-matrix` | Save first-level or second-level design matrices (TSV + PNG) | first-level: disabled; second-level: enabled |

---

### 6.7 Plotting

Curated visualization suites driven by a JSON plot catalog.

| Mode | Description |
|------|-------------|
| `visualize` | Render selected plot suites |
| `tfr` | Time-frequency representation plots |

```bash
# All available plots
eeg-pipeline plotting visualize --subject 0001 --all-plots

# Specific plot groups
eeg-pipeline plotting visualize --subject 0001 --groups power behavior

# TFR
eeg-pipeline plotting tfr --subject 0001

# Export as SVG and PDF
eeg-pipeline plotting visualize --subject 0001 --all-plots --formats svg pdf

# Group-level aggregate plots
eeg-pipeline plotting visualize --subject 0001 --subject 0002 --analysis-scope group
```

Available groups: `power`, `connectivity`, `aperiodic`, `phase`, `erds`, `complexity`,
`spectral`, `ratios`, `asymmetry`, `microstates`, `bursts`, `erp`, `tfr`, `behavior`.

---

### 6.8 Validation

Data integrity and schema checks across the pipeline.

| Mode | Scope |
|------|-------|
| `quick` | Fast, high-level check across key data products (default) |
| `all` | Exhaustive validation (all available checks) |
| `epochs` | Cleaned epochs (`.fif`) and metadata |
| `features` | Feature table schema, completeness, and basic ranges |
| `behavior` | Behavioral data tables and required columns |
| `bids` | BIDS layout and metadata for EEG/fMRI inputs |

```bash
# Quick validation
eeg-pipeline validate

# Full validation for specific subjects
eeg-pipeline validate all --subjects 0001 0002

# JSON output (for CI/scripting)
eeg-pipeline validate all --json
```

---

### 6.9 Stats and Info

Inspect pipeline state, subject availability, and current configuration.

**Info modes (read-only):**

| Mode | Reports |
|------|---------|
| `subjects` | Discovered subjects across BIDS, epochs, and feature derivatives |
| `features` | Feature availability per subject (which families/tables exist) |
| `config` | Current effective configuration (including runtime overrides) |
| `version` | Installed pipeline version and dependency snapshot |
| `plotters` | Available plot definitions and groups |
| `discover` | Discover available columns and values from events, trial tables, and condition-effects data |
| `rois` | Configured ROI definitions (channel groupings) |
| `fmri-conditions` | fMRI event conditions available for GLM specification |
| `fmri-columns` | Columns present in `events.tsv` files for fMRI analyses |
| `multigroup-stats` | Cross-subject summary statistics for selected features |
| `ml-feature-space` | Dimensions and structure of the ML feature matrix |

**Stats:** Pipeline-wide dashboard (default: `summary`). Modes: `summary`, `subjects`, `features`, `storage`, `timeline`.

```bash
eeg-pipeline info subjects
eeg-pipeline info features 0001
eeg-pipeline info config
eeg-pipeline info ml-feature-space
eeg-pipeline stats
```

---

### 6.10 Coupling

Study-specific EEG–BOLD coupling is exposed through the `coupling` command.
It is configured through the `studies/pain_study/` package and is meant for the
pain study workflow rather than the generic EEG/fMRI pipeline.

```bash
eeg-pipeline coupling compute --subject 0001 --subject 0002
```

---

## 7. Interactive TUI

A terminal UI built with Go and [Bubble Tea](https://github.com/charmbracelet/bubbletea)
providing menu-driven access to the full Python CLI.

**Prerequisites:** Go 1.21+, plus the Python environment used for the pipeline.

```bash
cd eeg_pipeline/cli/tui
go build -o eeg-tui .
./eeg-tui
```

**Capabilities:**

- Guided wizards for every pipeline stage.
- Subject selection with auto-discovery from BIDS and derivatives.
- Parameter configuration with validation and sensible defaults.
- Real-time execution with progress reporting and log streaming.
- Source localization and fMRI-aware workflows with guided configuration.

The TUI calls the Python CLI directly; all results are identical to CLI execution.

---

## 8. Configuration

Primary defaults live in `eeg_pipeline/utils/config/eeg_config.yaml`.
Behavioral analysis defaults live in `eeg_pipeline/utils/config/behavior_config.yaml`.
fMRI-specific defaults live in `fmri_pipeline/utils/config/fmri_config.yaml`.
CLI flags override config values at runtime.

### `eeg_config.yaml` sections

| Section | Controls |
|---------|---------|
| `project` | Task name and reproducibility settings |
| `paths` | BIDS root, derivatives root, source data, resting-state roots, FreeSurfer dirs, and signature maps |
| `analysis` | Strict mode and minimum subject count for group operations |
| `environment` | Per-library thread limits (`thread_limits`) |
| `event_columns` | Canonical column aliases for predictor, outcome, condition, and binary outcome |
| `alignment` | EEG↔fMRI trial alignment settings (trim-to-volume, onset reference, misalignment tolerance) |
| `eeg` | Montage, reference, and EOG/ECG channels |
| `preprocessing` | Filtering, resampling, break detection, resting-state mode, and clean events |
| `pyprep` | PyPREP bad-channel detection settings |
| `ica` | ICA method, component count, thresholds, and labels |
| `epochs` | Epoch window, baseline, and rejection policy |
| `frequency_bands` | Band definitions from delta through gamma |
| `time_windows` | Active and baseline windows |
| `rois` | ROI channel groupings |
| `feature_engineering` | Feature-category settings, spatial transforms, parallelization, and per-family options |
| `time_frequency_analysis` | TFR parameters and baseline normalization |
| `machine_learning` | Models, CV scheme, evaluation, importance, and preprocessing settings |
| `fmri_preprocessing` | fMRIPrep engine, image, spaces, and all runtime settings |
| `fmri_contrast` | Subject-level GLM specification, confound strategy, and output format |
| `fmri_group_level` | Group GLM model, covariates, permutation inference settings |
| `fmri_resting_state` | Resting-state ROI connectivity (atlas, bandpass, confound strategy, Fisher-z averaging) |
| `statistics` | Global alpha, permutation count, bootstrap, and cluster correction defaults |
| `visualization` | Band colors, robust limits, and footer templates |
| `plotting` | DPI, formats, figure sizes, styling, comparison windows, and per-plot settings |
| `system` | Global `n_jobs` |

### `behavior_config.yaml` sections

| Section | Controls |
|---------|---------|
| `behavior_analysis` | Predictor type, robust correlation, bootstrap, run adjustment, trial table export |
| `behavior_analysis.correlations` | Partial correlation targets, LOSO stability, Bayes factors |
| `behavior_analysis.condition` | Condition comparison columns, effect-size threshold, permutation |
| `behavior_analysis.temporal` | Time-resolved correlations, smoothing, and cluster correction |
| `behavior_analysis.regression` | Trialwise regression model, covariates, and permutation |
| `behavior_analysis.statistics` | Alpha, FDR, permutation scheme, and circular-shift settings |
| `behavior_analysis.icc` | Run-to-run reliability (ICC) configuration |
| `behavior_analysis.predictor_residual` | Spline/polynomial predictor residualization and cross-fit |

The source localization fMRI-constraint settings live under
`feature_engineering.sourcelocalization.fmri` in `eeg_config.yaml`.

### 8.1 Universal Runtime Overrides (`--set`)

For long-tail or rarely used parameters, use universal config overrides instead of
adding dedicated flags/widgets. This keeps the CLI and TUI maintainable while
preserving full configurability.

- CLI: repeat `--set KEY=VALUE`
- TUI: use `Config Overrides` in Advanced settings (`key=value;key2=value2`)

```bash
# Override behavior statistics at runtime
eeg-pipeline behavior compute --subject 0001 \
  --set behavior_analysis.statistics.fdr_alpha=0.01 \
  --set behavior_analysis.cluster.n_permutations=5000

# Override plotting style defaults
eeg-pipeline plotting visualize --subject 0001 --all-plots \
  --set plotting.defaults.dpi=400 \
  --set plotting.styling.colors.significant=\"#D62728\"

# Override ML data/feature filters
eeg-pipeline ml regression --all-subjects \
  --set machine_learning.data.feature_harmonization=union_impute \
  --set machine_learning.data.feature_bands='[\"alpha\",\"beta\"]'
```

Notes:

- `--set` values are type-coerced (`true/false`, `null`, ints, floats, JSON arrays/objects).
- `--set` is applied after command-specific overrides, so it has final precedence.
- Use dedicated flags/widgets for common workflows; use `--set`/`Config Overrides` for uncommon keys.

---

## 9. Subject Selection

Most commands accept these shared subject and runtime options:

| Option | Description |
|--------|-------------|
| `--subject XXXX` / `-s XXXX` | Single subject; repeat the flag for multiple subjects |
| `--all-subjects` | Process every discovered subject |
| `--group all` or `--group A,B,C` | Select a named group or comma-separated subject list |
| `--task` / `-t` | Override the task label from config |
| `--dry-run` | Preview work without executing |
| `--json` | Emit JSON output for scripting or the TUI |
| `--progress-json` | Emit progress events as JSON lines |
| `--set KEY=VALUE` | Override config values at runtime |
| `--bids-root` | Override `paths.bids_root` at runtime |
| `--bids-fmri-root` | Override `paths.bids_fmri_root` at runtime |
| `--bids-rest-root` | Override `paths.bids_rest_root` at runtime (resting-state EEG) |
| `--deriv-root` | Override `paths.deriv_root` at runtime |
| `--deriv-rest-root` | Override `paths.deriv_rest_root` at runtime (resting-state EEG) |

```bash
# Multiple subjects
eeg-pipeline features compute --subject 0001 --subject 0002 --subject 0003

# All subjects
eeg-pipeline features compute --all-subjects

# Dry run
eeg-pipeline ml regression --all-subjects --dry-run
```

`validate` uses `--subjects`; `info features` takes a positional subject ID.

---

## 10. Output Formats

Feature tables are saved as **Parquet** by default (recommended) with optional TSV/CSV:

```bash
eeg-pipeline features compute --subject 0001 --also-save-csv
```

Per-subject feature output structure:

```
derivatives/sub-XXXX/eeg/features/
├── power/
│   ├── features_power.parquet
│   └── metadata/
│       └── features_power.json        # Extraction config + column descriptions
├── connectivity/
│   ├── features_connectivity.parquet
│   └── metadata/
│       └── features_connectivity.json # Extraction config + column descriptions
├── aperiodic/
│   ├── features_aperiodic.parquet
│   ├── aperiodic_qc.tsv              # Per-segment/channel aperiodic fit QC
│   └── metadata/
│       └── features_aperiodic.json     # Extraction config + column descriptions
└── ...
```

Source localization outputs live under `sourcelocalization/<method>/source_estimates/`.
The final mode-specific subdirectory is `eeg_only/` or `fmri_informed/` depending on
`feature_engineering.sourcelocalization.mode`.

Plots are saved as PNG by default:

```bash
eeg-pipeline plotting visualize --subject 0001 --formats png svg pdf
```

---

## 11. Advanced Topics

### 11.1 Source Localization (EEG-only and fMRI-constrained)

Supports template-based (fsaverage) and subject-specific fMRI-constrained source
localization with LCMV beamformer or eLORETA.
The TUI includes a wizard for auto-generating BEM models and coregistration transforms
via Docker.

Full tutorial: [docs/eeg/source-localization.md](docs/eeg/source-localization.md)

### 11.2 fMRI Raw-to-BIDS Conversion

DICOM-to-BIDS conversion with optional event generation from behavioral logs
(for example, PsychoPy TrialSummary files).
Requires `dcm2niix` on `PATH`.

Guide: [docs/fmri/raw-to-bids.md](docs/fmri/raw-to-bids.md)

### 11.3 Docker Image for FreeSurfer + MNE

A Docker image is provided for BEM generation and coregistration:

```bash
docker build --platform linux/amd64 -t freesurfer-mne:7.4.1 \
  -f eeg_pipeline/docker_setup/Dockerfile.freesurfer-mne .
```

This image bundles FreeSurfer 7.4.1 + MNE-Python on Python 3.11 and is used for
`recon-all`, `watershed_bem`, BEM solution generation, and coregistration.

### 11.4 EEG–fMRI Fusion via ML

Predict trial-wise fMRI signature expression from EEG features:

```bash
eeg-pipeline ml regression --subject 0001 --subject 0002 \
  --target fmri_signature \
  --fmri-signature-name SIGNATURE_A \
  --fmri-signature-method beta-series \
  --fmri-signature-metric dot
```

Available signatures are whatever names were generated in the fMRI analysis output
tables from your configured `paths.signature_maps`. Methods: `beta-series`, `lss`.
Metrics: `dot`, `cosine`, `pearson_r`.

### 11.5 Spatial Transforms (CSD / Laplacian)

Phase-based feature families (connectivity, ITPC, PAC) have CSD applied by default
to reduce volume conduction. All other families default to `none`.
Override globally with `--spatial-transform csd` or configure per-family in YAML.
See the features README for the complete per-family defaults.

### 11.6 Individual Alpha Frequency (IAF)

Enable IAF-based adaptive frequency bands derived from posterior baseline PSD:

```bash
eeg-pipeline features compute --subject 0001 --iaf-enabled
```

### 11.7 Analysis Modes

| Mode | Intended use |
|------|-------------|
| `group_stats` *(default)* | Cross-trial estimators are permitted; outputs are typically one row per subject or condition. |
| `trial_ml_safe` | Estimators that pool across trials require an explicit `train_mask`; leakage-prone paths are disabled. Use whenever outputs feed into cross-validated models. |

```bash
eeg-pipeline features compute --subject 0001 --analysis-mode trial_ml_safe
```

For the full list of CV-safety guardrails, see the features README.

### 11.8 Resting-State Workflows

The pipeline supports resting-state EEG as a first-class workflow alongside task-based paradigms.
Enable resting-state mode via the CLI flag or in `eeg_config.yaml`:

```yaml
preprocessing:
  task_is_rest: true
  rest_epochs_duration: 10.0    # fixed-length epoch duration (seconds)
  rest_epochs_overlap: 0.0      # overlap between consecutive epochs
```

When `task_is_rest: true`:
- Preprocessing creates fixed-length epochs instead of event-locked epochs.
- No `events.tsv` conditions are required.
- Feature extraction runs in `group_stats` mode by default (no trial-level behavioral targets).
- Event-locked feature categories (`erp`, `erds`, `itpc`, `phase`) are blocked and raise a
  `ValueError` if requested.
- A separate BIDS root and derivatives root can be pointed to via `paths.bids_rest_root` and
  `paths.deriv_rest_root` so task and resting-state derivatives coexist without conflicts.

```bash
# Resting-state preprocessing
eeg-pipeline preprocessing full --subject 0001 --task-is-rest

# Override resting-state BIDS and derivatives roots at runtime
eeg-pipeline features compute --subject 0001 \
  --bids-rest-root ../data/bids_output/eeg_rest \
  --deriv-rest-root ../data/derivatives_rest

# Resting-state EEG feature extraction (spectral + connectivity; event-locked categories excluded)
eeg-pipeline features compute --subject 0001 \
  --categories power connectivity aperiodic spectral

# Resting-state fMRI connectivity (atlas-based ROI time series + Fisher-z matrix)
eeg-pipeline fmri-analysis rest --subject 0001 \
  --atlas-labels-img /path/to/atlas_parc.nii.gz \
  --atlas-labels-tsv /path/to/atlas_labels.tsv
```

For fMRI resting-state methods (atlas masking, confound scrubbing, Fisher-z run aggregation),
see [fmri_pipeline/README.md §9](fmri_pipeline/README.md).

---

## 12. Dependencies

| Package | Version | Role |
|---------|---------|------|
| **MNE-Python** | ≥ 1.9.0 | EEG processing, source localization |
| **MNE-BIDS** | ≥ 0.16.0 | BIDS I/O |
| **MNE-Connectivity** | ≥ 0.7.0 | Functional connectivity |
| **MNE-ICALabel** | ≥ 0.7.0 | Automatic ICA classification |
| **MNE-BIDS-Pipeline** | ≥ 1.9.0 | ICA detection fallback |
| **PyPREP** | ≥ 0.4.3 | Bad channel detection |
| **pybv** | ≥ 0.7.5 | BrainVision format I/O |
| **specparam** | ≥ 2.0.0rc3 | Aperiodic (1/f) fitting |
| **Nilearn** | ≥ 0.11.1 | fMRI GLM and neuroimaging |
| **NiBabel** | ≥ 3.2.0, < 6.0 | NIfTI/CIFTI I/O |
| **scikit-learn** | ≥ 1.0.0, < 2.0 | Machine learning models |
| **imbalanced-learn** | ≥ 0.12.0 | Class resampling (SMOTE, undersample) |
| **SHAP** | ≥ 0.40.0 | Feature importance |
| **PyTorch** | ≥ 2.7.1 *(optional, `ml` extra)* | Deep learning (EEGNet CNN) |
| **NetworkX** | ≥ 3.5 | Graph-theoretic connectivity metrics |
| **bctpy** | ≥ 0.6.1 | Brain Connectivity Toolbox |
| **statsmodels** | ≥ 0.13.0, < 1.0 | Statistical models, FDR correction |
| **sympy** | ≥ 1.14.0 | Symbolic math (spline knot computation) |
| **antropy** | ≥ 0.1.9 | Complexity measures |
| **NumPy** | ≥ 1.24, < 2.0 | Array computation |
| **SciPy** | ≥ 1.15.3 | Scientific computing |
| **pandas** | ≥ 2.3.0 | Data manipulation |
| **pyarrow** | ≥ 17.0.0 | Parquet I/O for feature tables |
| **h5io** | ≥ 0.1.0 | HDF5 I/O helpers |
| **joblib** | ≥ 1.5.1 | Parallel computation |
| **matplotlib** | ≥ 3.10.3 | Plotting backend |
| **seaborn** | ≥ 0.13.2 | Statistical visualizations |
| **PyYAML** | ≥ 6.0, < 7.0 | Configuration file parsing |

`pyproject.toml` is the single source of truth for all dependencies and version bounds.
`requirements.txt` installs the editable package with `dev` and `ml` extras.
PyTorch is only required for the CNN classifier (`ml classify --classification-model cnn`);
install with `pip install -e ".[ml]"` or omit for all other workflows.

---

## 13. License

This project is licensed under the [MIT License](LICENSE).

---

## 14. Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-feature`.
3. Make changes and add tests where applicable.
4. Run the test suite: `pytest`.
5. Submit a pull request.
