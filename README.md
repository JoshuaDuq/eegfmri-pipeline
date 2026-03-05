# EEG–fMRI Analysis Pipeline

[![Python ≥ 3.11](https://img.shields.io/badge/python-≥3.11-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![BIDS](https://img.shields.io/badge/data-BIDS-orange.svg)](https://bids-specification.readthedocs.io/)
[![MNE-Python](https://img.shields.io/badge/MNE--Python-1.9.0-informational.svg)](https://mne.tools)

A modular, end-to-end pipeline for simultaneous EEG–fMRI research.
Covers raw data conversion, preprocessing, feature extraction, behavioral analysis,
machine learning, fMRI first-level GLM, and publication-ready visualization —
all from a unified CLI and interactive TUI.
Under active development.



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

Each pipeline module has a dedicated README with full methods, formulas, configuration
references, and output schemas. Start here for orientation, then follow the link
relevant to your workflow:

| Module | README |
|--------|--------|
| EEG Preprocessing | [eeg_pipeline/preprocessing/README.md](eeg_pipeline/preprocessing/README.md) |
| Feature Extraction | [eeg_pipeline/analysis/features/README.md](eeg_pipeline/analysis/features/README.md) |
| Behavioral Analysis | [eeg_pipeline/analysis/behavior/README.md](eeg_pipeline/analysis/behavior/README.md) |
| Machine Learning | [eeg_pipeline/analysis/machine_learning/README.md](eeg_pipeline/analysis/machine_learning/README.md) |
| fMRI Analysis | [fmri_pipeline/README.md](fmri_pipeline/README.md) |
| Interactive TUI | [eeg_pipeline/cli/tui/README.md](eeg_pipeline/cli/tui/README.md) |


---

## 2. Quick Start

```bash
# 1. Clone and install
git clone https://github.com/JoshuaDuq/eegfmri-pipeline.git && cd eegfmri-pipeline
pip install -e .

# 2. Place your data (see §4 Data Requirements)

# 3. Launch the interactive TUI
cd eeg_pipeline/cli/tui && go build -o eeg-tui . && ./eeg-tui
```

---

## 3. Installation

**Requirements:** Python ≥ 3.11

```bash
python -m venv .venv311
source .venv/bin/activate
pip install -r requirements.txt
```

After installation, the `eeg-pipeline` command is available in the active environment.
`requirements.txt` installs with optional `[ml]` extras (PyTorch). To add ML after a base install:

```bash
pip install -e ".[ml]"
```

---

## 4. Data Requirements

The pipeline expects data organized under `data/` at the repository root.
All paths are configurable in `eeg_pipeline/utils/config/eeg_config.yaml`.

### 4.1 Minimum Required Inputs by Workflow

| Workflow | Minimum required inputs |
|----------|-------------------------|
| EEG preprocessing / features from BIDS | `data/bids_output/eeg/sub-XXXX/eeg/*_eeg.<format>` plus matching BIDS sidecars |
| Behavioral analysis / ML | EEG `*_events.tsv` with at least `onset`, `duration`, `trial_type`; plus study-specific predictor/target columns |
| EEG raw-to-BIDS | `data/source_data/sub-XXXX/eeg/*.vhdr` (+ `.vmrk` and `.eeg`) |
| External event-log merge into EEG events | Study event logs (e.g., PsychoPy `*TrialSummary.csv`) + BIDS `*_events.tsv` |
| fMRI raw-to-BIDS | `data/source_data/sub-XXXX/fmri/<dicom-series-dir>/` |
| fMRI first-level analysis | `data/bids_output/fmri/sub-XXXX/func/*_bold.nii.gz` + `*_events.tsv` |

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
        ├── sub-XXXX_task-YYY_run-01_events.tsv         # required for behavior/fMRI alignment
        ├── sub-XXXX_task-YYY_run-01_channels.tsv        # recommended
        └── sub-XXXX_task-YYY_run-01_electrodes.tsv      # recommended
```

### 4.4 Behavioral and Events Data

Behavior and ML workflows read trial-level predictors and outcomes from BIDS
`*_events.tsv` files. Required columns: `onset`, `duration`, `trial_type`.
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
├── eeg_pipeline/               # Core EEG pipeline package
│   ├── analysis/
│   │   ├── features/           # Feature extraction (16 categories)
│   │   ├── behavior/           # Behavioral correlation analysis
│   │   ├── machine_learning/   # ML models and evaluation
│   │   └── utilities/          # Data conversion helpers
│   ├── cli/
│   │   ├── commands/           # CLI command definitions
│   │   ├── tui/                # Go-based interactive TUI (Bubble Tea)
│   │   └── main.py             # CLI entry point
│   ├── pipelines/              # Pipeline orchestration and batch processing
│   ├── plotting/               # Visualization modules and plot catalog
│   ├── domain/                 # Domain types, constants, naming schema
│   ├── infra/                  # Path resolution, TSV/Parquet I/O
│   ├── utils/
│   │   ├── config/             # YAML configuration (eeg_config.yaml)
│   │   └── data/               # Subject discovery, feature I/O
│   └── docker_setup/           # Dockerfile for FreeSurfer + MNE
├── fmri_pipeline/              # fMRI preprocessing and analysis
│   ├── analysis/               # GLM, contrasts, beta-series, signatures
│   ├── cli/commands/           # fMRI CLI commands
│   └── pipelines/              # fMRI pipeline orchestration
├── scripts/                    # Standalone utility scripts
├── tests/                      # Test suite (pytest)
├── docs/                       # Extended tutorials and guides
├── pyproject.toml
└── requirements.txt
```

---

## 6. CLI Reference

All processing commands follow this pattern:

```bash
eeg-pipeline <command> <mode> [--subject XXXX | --all-subjects] [options]
```

Use `--help` on any command for full option details:

```bash
eeg-pipeline <command> --help
```

**Command overview:**

| Command | Primary role |
|---------|-------------|
| `preprocessing` | EEG cleaning: filtering, bad channels, ICA, epoching → cleaned `epo.fif` |
| `features` | Compute 16 EEG feature families → trial-wise feature tables |
| `behavior` | Relate features to behavior → effect sizes, statistical summaries |
| `ml` | Trial-level predictive models → cross-validated performance and diagnostics |
| `fmri` | fMRIPrep-based BOLD preprocessing → preprocessed derivatives |
| `fmri-analysis` | First-level GLM and trial-wise betas → contrast maps, beta-series |
| `plotting` | Visualization suites over existing derivatives → figures (PNG/SVG/PDF) |
| `validate` | Data and derivative integrity checks → structured validation reports |
| `stats` | Pipeline-wide coverage summary → aggregate statistics dashboard |
| `info` | Inspect configuration, subjects, ROIs, feature space → summaries |

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

# Without ICALabel
eeg-pipeline preprocessing full --subject 0001 --no-icalabel
```

Full pipeline steps, CLI options, and configuration reference:
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
For exact formulas and configuration reference, see
[eeg_pipeline/analysis/features/README.md](eeg_pipeline/analysis/features/README.md).

```bash
# All feature categories
eeg-pipeline features compute --subject 0001

# Specific categories with spatial modes
eeg-pipeline features compute --subject 0001 \
  --categories power connectivity aperiodic \
  --spatial roi global

# When --spatial is omitted, config's feature_engineering.spatial_modes is used
# (default: roi, channels, global). Use --spatial to restrict to a subset.

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

| Mode | Description |
|------|-------------|
| `compute` | Run behavioral analysis stages and save numerical outputs |
| `visualize` | Generate standardized plots from previously computed results |

For all 22 stage definitions, the pipeline DAG, statistical safeguards, and
configuration reference, see
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
| `classify` | Binary pain classification (SVM, logistic regression, random forest, CNN) |
| `timegen` | Temporal generalization: train at one time window, evaluate across all windows |
| `model_comparison` | Compare model families under a shared CV scheme |
| `incremental_validity` | Quantify change in explained variance (ΔR²) when adding EEG features over a baseline predictor |
| `uncertainty` | Conformal prediction intervals for calibrated uncertainty estimates |
| `shap` | SHAP-based feature importance |
| `permutation` | Permutation-based feature importance |

For the full architecture, CV schemes, evaluation metrics, and configuration reference,
see [eeg_pipeline/analysis/machine_learning/README.md](eeg_pipeline/analysis/machine_learning/README.md).

```bash
# LOSO regression (requires ≥2 subjects)
eeg-pipeline ml regression --subject 0001 --subject 0002 --subject 0003

# SVM classification
eeg-pipeline ml classify --subject 0001 --subject 0002 --classification-model svm

# SHAP importance
eeg-pipeline ml shap --subject 0001 --subject 0002

# Within-subject CV
eeg-pipeline ml regression --subject 0001 --cv-scope subject

# Predict fMRI signature expression from EEG
eeg-pipeline ml regression --subject 0001 --subject 0002 \
  --target fmri_signature --fmri-signature-name NPS

# Model comparison with custom hyperparameters
eeg-pipeline ml model_comparison --subject 0001 --subject 0002 \
  --elasticnet-alpha-grid 0.01 0.1 1 10 \
  --rf-n-estimators 500

# Restrict to specific feature families and bands
eeg-pipeline ml regression --subject 0001 --subject 0002 \
  --feature-families power connectivity --feature-bands alpha beta
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

Note: for container runs, the pipeline automatically ignores macOS metadata files (`._*`, `.DS_Store`) by mounting a sanitized temporary BIDS view.

---

### 6.6 fMRI Analysis

Subject-level GLM contrasts and trial-wise beta estimation via nilearn.

| Mode | Description |
|------|-------------|
| `first-level` | First-level GLM with user-defined contrasts → contrast maps |
| `beta-series` | Trial-wise beta-series estimation (LSA method) |
| `lss` | Least-squares-separate (LSS) trial betas |

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

# Beta-series for EEG–fMRI fusion
eeg-pipeline fmri-analysis beta-series --subject 0001 \
  --cond-a-value stimulation --cond-b-value fixation_rest

# LSS betas
eeg-pipeline fmri-analysis lss --subject 0001 \
  --cond-a-value stimulation --cond-b-value fixation_rest

# With HTML report
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
| `--resample-to-freesurfer` | Resample to FreeSurfer space | disabled |
| `--plots` | Generate per-subject figures | disabled |
| `--plot-html-report` | Write self-contained HTML report | disabled |
| `--write-design-matrix` | Save design matrices (TSV + PNG) | disabled |

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

## 7. Interactive TUI

A terminal UI built with Go and [Bubble Tea](https://github.com/charmbracelet/bubbletea)
providing menu-driven access to the full Python CLI.

**Prerequisites:** Go 1.21+, Python environment with pipeline dependencies installed.

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
- Source localization wizard with Docker-based BEM and coregistration auto-generation.

The TUI calls the Python CLI directly; all results are identical to CLI execution.

---

## 8. Configuration

All defaults live in `eeg_pipeline/utils/config/eeg_config.yaml`.
CLI flags override config values at runtime.

| Section | Controls |
|---------|---------|
| `project` | Task name, random seed |
| `paths` | BIDS root, derivatives root, source data, FreeSurfer directories |
| `eeg` | Montage, reference electrode, EOG/ECG channel names |
| `preprocessing` | Filter settings, resampling, break detection, clean events |
| `pyprep` | PyPREP bad channel detection (RANSAC, repeats) |
| `ica` | Method, component count, probability threshold, labels to retain |
| `epochs` | Time window, baseline, rejection method (autoreject) |
| `frequency_bands` | Band definitions (delta through gamma) |
| `time_windows` | Active and baseline window definitions |
| `rois` | ROI channel groupings |
| `feature_engineering` | Per-category settings, spatial transforms, parallelization |
| `time_frequency_analysis` | TFR parameters, baseline normalization mode |
| `behavior_analysis` | Correlation method, permutation count, temperature control, stage toggles |
| `machine_learning` | Models, CV scheme, evaluation, importance settings |
| `fmri_preprocessing` | fMRIPrep engine, image, output spaces |
| `fmri_contrast` | GLM specification, confound strategy, cluster correction |
| `fmri_constraint` | Source localization fMRI constraint threshold and mask |

### 8.1 Universal Runtime Overrides (`--set`)

For long-tail or rarely used parameters, use universal config overrides instead of
adding dedicated flags/widgets. This keeps CLI and TUI maintainable while preserving
full configurability.

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

Most commands accept these subject selection options:

| Option | Description |
|--------|-------------|
| `--subject XXXX` / `-s XXXX` | Single subject (repeatable for multiple) |
| `--all-subjects` | Process all discovered subjects |
| `--group all` or `--group A,B,C` | Named group or comma-separated list |
| `--task` / `-t` | Override task label (default from config) |
| `--dry-run` | Preview what would run without executing |
| `--json` | Output in JSON format (for TUI or scripting) |

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

Source localization outputs may appear under `sourcelocalization/eeg_only/` or
`sourcelocalization/fmri_informed/` depending on `feature_engineering.sourcelocalization.mode`.

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

Predict trial-wise fMRI signature expression (NPS, SIIPS1) from EEG features:

```bash
eeg-pipeline ml regression --subject 0001 --subject 0002 \
  --target fmri_signature \
  --fmri-signature-name NPS \
  --fmri-signature-method beta-series \
  --fmri-signature-metric dot
```

Available signatures: `NPS`, `SIIPS1`. Methods: `beta-series`, `lss`.
Metrics: `dot`, `cosine`, `pearson_r`.

**NPS (Neuroimaging Pain Signature):** The NPS weight map is not publicly distributed.
Obtain the signature weights from the original authors (Wager et al., 2013, *NEJM*)
and place them at `<weights_root>/NPS/weights_NSF_grouppred_cvpcr.nii.gz`.
Pass the root via `--pain-signature-weights` or `paths.signature_dir` in config.

**SIIPS1 (Stimulus Intensity Independent Pain Signature):** Weights are publicly
available from the
[CANlab Neuroimaging_Pattern_Masks](https://github.com/canlab/Neuroimaging_Pattern_Masks)
repository. Place `nonnoc_v11_4_137subjmap_weighted_mean.nii.gz` at
`<weights_root>/SIIPS1/`.

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

---

## 12. Dependencies

| Package | Version | Role |
|---------|---------|------|
| **MNE-Python** | 1.9.0 | EEG processing, source localization |
| **MNE-BIDS** | 0.16.0 | BIDS I/O |
| **MNE-Connectivity** | 0.7.0 | Functional connectivity |
| **MNE-ICALabel** | 0.7.0 | Automatic ICA classification |
| **PyPREP** | 0.4.3 | Bad channel detection |
| **specparam** | 2.0.0rc3 | Aperiodic (1/f) fitting |
| **Nilearn** | 0.11.1 | fMRI GLM and neuroimaging |
| **NiBabel** | ≥ 3.2.0 | NIfTI/CIFTI I/O |
| **scikit-learn** | ≥ 1.0.0 | Machine learning models |
| **SHAP** | ≥ 0.40.0 | Feature importance |
| **PyTorch** | 2.7.1 | Deep learning (EEGNet CNN) |
| **NetworkX** | 3.5 | Graph-theoretic connectivity metrics |
| **bctpy** | 0.6.1 | Brain Connectivity Toolbox |
| **statsmodels** | ≥ 0.13.0 | Statistical models, FDR correction |
| **antropy** | ≥ 0.1.9 | Complexity measures |
| **NumPy** | ≥ 1.24, < 2.0 | Array computation |
| **SciPy** | 1.15.3 | Scientific computing |
| **pandas** | 2.3.0 | Data manipulation |

`pyproject.toml` is the single source of truth for dependencies.
`requirements.txt` installs `-e ".[ml]"`.

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
