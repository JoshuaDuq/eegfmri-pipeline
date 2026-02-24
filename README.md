# EEG–fMRI Analysis Pipeline

> **Status:** Under active development.

[![Python ≥ 3.11](https://img.shields.io/badge/python-≥3.11-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![BIDS](https://img.shields.io/badge/data-BIDS-orange.svg)](https://bids-specification.readthedocs.io/)
[![MNE-Python](https://img.shields.io/badge/MNE--Python-1.9.0-informational.svg)](https://mne.tools)

A modular, end-to-end pipeline for simultaneous EEG–fMRI research. Covers raw data conversion, preprocessing, feature extraction, behavioral analysis, machine learning, fMRI first-level GLM, and publication-ready visualization — all from a unified CLI.

An **interactive TUI** provides guided wizards for every stage with no CLI flags to memorize. See [Interactive TUI](#interactive-tui).

### TUI Showcase

<p align="center">
  <img src="docs/screenshots/tui_main_menu.png" width="800" alt="Main menu — All pipeline stages at a glance"/>
</p>

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/JoshuaDuq/eegfmri-pipeline.git && cd eegfmri-pipeline
pip install -e .

# 2. Place your data (see "Data Requirements" below)

# 3. Launch
cd eeg_pipeline/cli/tui && go build -o eeg-tui . && ./eeg-tui
```

---

## Installation

**Requirements:** Python ≥ 3.11

```bash
# Create and activate a virtual environment
python -m venv .venv311
source .venv311/bin/activate

# Install dependencies
pip install -r requirements.txt

# Or install directly from pyproject metadata
# pip install -e ".[ml]"
```

After installation, the `eeg-pipeline` command is available globally in the environment.

---

## Data Requirements

### What you need to start

The pipeline expects data organized under `data/` at the repository root. All paths are configurable in `eeg_pipeline/utils/config/eeg_config.yaml`.

### Minimum required data by workflow

| Workflow | Minimum required inputs |
|----------|-------------------------|
| EEG preprocessing/features from BIDS | `data/bids_output/eeg/sub-XXXX/eeg/*_eeg.<format>` (plus matching sidecars expected by your BIDS converter) |
| Behavioral analysis / ML using trial metadata | EEG `*_events.tsv` with at least `onset`, `duration`, `trial_type`; include pain/temperature/rating columns (see below) |
| EEG raw to BIDS (`paradigm-specific-scripts`) | `data/source_data/sub-XXXX/eeg/*.vhdr` (+ matching `.vmrk` and `.eeg`) |
| Merge PsychoPy into EEG events (`paradigm-specific-scripts`) | `data/source_data/sub-XXXX/PsychoPy_Data/*TrialSummary.csv` + existing BIDS EEG `*_events.tsv` |
| fMRI raw to BIDS (`paradigm-specific-scripts`) | `data/source_data/sub-XXXX/fmri/<dicom-series-dir>/` and (if writing events) `PsychoPy_Data/*TrialSummary.csv` |
| fMRI first-level analysis (`fmri-analysis`) | `data/bids_output/fmri/sub-XXXX/func/*_bold.nii.gz` + matching `*_events.tsv` with `onset`, `duration`, `trial_type` |

#### Option A: Start from raw recordings (use paradigm-specific scripts)

Place raw BrainVision EEG files under `source_data/` (subject directory must be `sub-XXXX`):

```
data/source_data/
└── sub-XXXX/
    └── eeg/
        ├── sub-XXXX_task-task_run-01_eeg.vhdr
        ├── sub-XXXX_task-task_run-01_eeg.vmrk
        └── sub-XXXX_task-task_run-01_eeg.eeg
```

Then convert to BIDS:

```bash
python paradigm-specific-scripts/run_paradigm_specific.py eeg-raw-to-bids \
  --source-root data/source_data \
  --bids-root data/bids_output/eeg \
  --task task \
  --subject XXXX
```

> **Note:** Raw-to-BIDS and PsychoPy merge helpers are paradigm-specific and now live under `paradigm-specific-scripts/`. Adapt those scripts if your paradigm differs.

If you want behavioral columns merged into EEG events, also provide PsychoPy trial summaries:

```
data/source_data/
└── sub-XXXX/
    └── PsychoPy_Data/
        ├── ...run-1...TrialSummary.csv
        └── ...run-2...TrialSummary.csv
```

#### Option B: Start from BIDS-compliant data (recommended for other paradigms)

If your data is already in [BIDS format](https://bids-specification.readthedocs.io/) (e.g., converted with MNE-BIDS, BIDScoin, or another tool), place it directly under `bids_output/`:

```
data/bids_output/eeg/
├── dataset_description.json
├── participants.tsv
└── sub-XXXX/
    └── eeg/
        ├── sub-XXXX_task-YYY_run-01_eeg.vhdr   (or .set, .edf, .fif)
        ├── sub-XXXX_task-YYY_run-01_eeg.vmrk
        ├── sub-XXXX_task-YYY_run-01_eeg.eeg
        ├── sub-XXXX_task-YYY_run-01_events.tsv          # required for behavior/fMRI alignment workflows
        ├── sub-XXXX_task-YYY_run-01_channels.tsv        # recommended
        └── sub-XXXX_task-YYY_run-01_electrodes.tsv      # recommended
```

Skip the `raw-to-bids` step and proceed directly to preprocessing.

#### For behavioral data

Behavior and ML workflows read trial-level predictors/outcomes from EEG BIDS `*_events.tsv` files. Keep `onset`, `duration`, and `trial_type`, plus any paradigm-specific columns you plan to model (for example rating, temperature, and binary outcome labels).

#### For fMRI data (optional)

```
data/source_data/sub-XXXX/fmri/    # Raw DICOMs (for paradigm-specific fMRI conversion)
data/bids_output/fmri/              # BIDS-formatted fMRI (NIfTI + events)
data/fMRI_data/sub-XXXX/anat/       # T1w anatomical (for FreeSurfer/source localization)
```

Minimal BIDS-fMRI run example for `fmri-analysis`:

```
data/bids_output/fmri/
└── sub-XXXX/
    └── func/
        ├── sub-XXXX_task-task_run-01_bold.nii.gz
        └── sub-XXXX_task-task_run-01_events.tsv
```

For fMRI first-level GLM, each run-level `*_events.tsv` must include:

- `onset`
- `duration`
- `trial_type`

For fMRI contrast workflows, ensure your events file includes whichever columns/values you reference via `--cond-a-column/--cond-a-value` and `--cond-b-column/--cond-b-value`.

### Default directory layout

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
    ├── freesurfer/             # FreeSurfer reconstructions (for source localization)
    └── sub-XXXX/
        ├── eeg/
        │   ├── sub-XXXX_task-*_proc-clean_epo.fif   # Cleaned epochs
        │   └── features/       # Extracted feature tables (.parquet/.tsv)
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
  bids_root: "../../../data/bids_output/eeg"
  bids_fmri_root: "../../../data/bids_output/fmri"
  deriv_root: "../../../data/derivatives"
  source_data: "../../../data/source_data"
  freesurfer_dir: "../../../data/derivatives/freesurfer"
```

---

## Project Structure

```
eegfmri-pipeline/
├── eeg_pipeline/               # Core EEG pipeline package
│   ├── analysis/               # Analysis modules
│   │   ├── features/           # Feature extraction (16 categories)
│   │   ├── behavior/           # Behavioral correlation analysis
│   │   ├── machine_learning/   # ML models and evaluation
│   │   └── utilities/          # Data conversion helpers
│   ├── cli/
│   │   ├── commands/           # CLI command definitions (one file per command)
│   │   ├── tui/                # Go-based interactive terminal UI (Bubble Tea)
│   │   └── main.py             # CLI entry point
│   ├── pipelines/              # Pipeline orchestration and batch processing
│   ├── plotting/               # Visualization modules + plot catalog
│   ├── domain/                 # Domain types, constants, naming schema
│   ├── infra/                  # Path resolution, TSV/parquet I/O
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
├── README/                     # Extended tutorials
│   ├── FMRI_RAW_TO_BIDS.md
│   └── SOURCE_LOCALIZATION_TUTORIAL.md
├── pyproject.toml
└── requirements.txt
```

---

## Pipeline Overview

Most processing commands follow this pattern:

```bash
eeg-pipeline <command> <mode> [--subject XXXX | --group ... | --all-subjects] [options]
```

Utility commands use simpler forms:

```bash
eeg-pipeline info <mode> [target] [options]
eeg-pipeline stats [mode] [options]
eeg-pipeline validate [mode] [options]
```

Use `--help` on any command for full option details:

```bash
eeg-pipeline <command> --help
```

### Available commands at a glance

| Command | Description |
|---------|-------------|
| `preprocessing` | Bad channels, ICA, epoching |
| `features` | Extract or visualize 16 EEG feature categories |
| `behavior` | Behavioral correlations, condition comparisons, mediation |
| `ml` | Machine learning (regression, classification, SHAP, etc.) |
| `fmri` | Containerized fMRIPrep preprocessing |
| `fmri-analysis` | First-level GLM, beta-series, LSS |
| `plotting` | Curated visualization suites |
| `validate` | Data integrity checks |
| `stats` | Pipeline-wide statistics dashboard |
| `info` | Discover subjects, features, config, ROIs |

---

### 1. Preprocessing

Automated EEG preprocessing — bad channel detection, ICA artifact removal, and epoching.

| Mode | Description |
|------|-------------|
| `full` | Run all preprocessing steps sequentially |
| `bad-channels` | Detect and interpolate bad channels only |
| `ica` | Fit and apply ICA only |
| `epochs` | Create epochs only |

```bash
# Full preprocessing pipeline
eeg-pipeline preprocessing full --subject 0001

# Just bad channel detection with PyPREP + RANSAC
eeg-pipeline preprocessing bad-channels --subject 0001 --use-pyprep --ransac

# Custom epoch window with autoreject
eeg-pipeline preprocessing epochs --subject 0001 \
  --tmin -7.0 --tmax 15.0 --reject-method autoreject_local

# Disable ICALabel, use MNE-BIDS pipeline detection
eeg-pipeline preprocessing full --subject 0001 --no-icalabel
```

**Key options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--ica-method` | `extended_infomax`, `fastica`, `picard` | `extended_infomax` |
| `--use-icalabel / --no-icalabel` | Automatic ICA component classification | enabled |
| `--use-pyprep / --no-pyprep` | PyPREP bad channel detection | enabled |
| `--ransac / --no-ransac` | RANSAC for bad channel detection | enabled |
| `--reject-method` | `none`, `autoreject_local`, `autoreject_global` | `autoreject_local` |
| `--l-freq` | High-pass filter (Hz) | `0.1` |
| `--h-freq` | Low-pass filter (Hz) | `100` |
| `--resample` | Resampling frequency (Hz) | `500` |
| `--notch` | Notch filter frequency (Hz) | `60` |
| `--conditions` | Epoching conditions (comma-separated) | from config |
| `--tmin`, `--tmax` | Epoch time window (seconds) | `-7.0`, `15.0` |
| `--baseline` | Baseline window, e.g., `-0.2 0` | `[-0.2, 0.0]` |
| `--write-clean-events` | Write post-rejection events.tsv | enabled |
| `--n-jobs` | Parallel jobs for bad channel detection | `1` |

---

### 2. Feature Extraction

Extract trial-level EEG features from cleaned epochs across 16 categories.

| Mode | Description |
|------|-------------|
| `compute` | Extract features and save to derivatives |
| `visualize` | Generate descriptive feature plots |

**16 feature categories:**

| Category | Description |
|----------|-------------|
| `power` | Band power (delta, theta, alpha, beta, gamma) with baseline normalization |
| `spectral` | Spectral edge frequency, peak frequency, bandwidth |
| `ratios` | Band power ratios (theta/beta, theta/alpha, alpha/beta, delta/alpha, delta/theta) |
| `aperiodic` | 1/f slope and offset via specparam-style fitting (iterative peak rejection, QC per channel, duration/frequency-resolution validity gates) |
| `connectivity` | Functional connectivity (wPLI, AEC, PLV) with per-family spatial transforms (default CSD for phase-based metrics) |
| `directedconnectivity` | Directed connectivity (PSI, DTF, PDC) via MVAR models with automatic order downshift on short segments |
| `microstates` | Microstate dynamics from fixed templates or subject-fitted clustering: coverage, duration, occurrence, transitions. Default assignment uses GFP-peak labeling with sample-wise backfitting (subject-fitted templates are flagged as non-i.i.d. in provenance). |
| `pac` | Phase-amplitude coupling (theta–gamma, alpha–gamma) with harmonic filtering and configurable surrogate nulls (`trial_shuffle` default, `circular_shift` fallback). |
| `itpc` | Inter-trial phase coherence (fold-safe for ML, condition-aware) |
| `erp` | Event-related potential components (N1, N2, P2 — configurable windows) |
| `bursts` | Oscillatory burst detection (beta, gamma) with threshold methods |
| `complexity` | Permutation entropy, sample entropy, multiscale entropy, LZC |
| `asymmetry` | Hemispheric asymmetry indices (F3/F4, C3/C4, P3/P4, O1/O2) |
| `erds` | Event-related desynchronization/synchronization with pain markers |
| `quality` | Data quality metrics (SNR, muscle artifact, line noise) |
| `sourcelocalization` | Source-space features via LCMV beamformer or eLORETA |

> **Note:** All the feature computations are highly configurable through the TUI.
>
> For detailed documentation of every feature's exact computation method, see [eeg_pipeline/analysis/features/README.md](eeg_pipeline/analysis/features/README.md).

```bash
# Extract all feature categories
eeg-pipeline features compute --subject 0001

# Extract specific categories with spatial modes
eeg-pipeline features compute --subject 0001 \
  --categories power connectivity aperiodic \
  --spatial roi global

# Custom frequency bands and ROIs
eeg-pipeline features compute --subject 0001 \
  --frequency-bands "mu:8.0:13.0" "high_beta:20.0:30.0" \
  --rois "Motor:C3,C4,Cz" "Occipital:O1,O2,Oz"

# ML-safe mode (prevents cross-trial leakage)
eeg-pipeline features compute --subject 0001 --analysis-mode trial_ml_safe

# Apply CSD spatial transform for phase-based features
eeg-pipeline features compute --subject 0001 --spatial-transform csd

# Visualize extracted features
eeg-pipeline features visualize --subject 0001
```

For up-to-date flags and defaults, run `eeg-pipeline features --help`.



For fMRI-constrained source localization, see [docs/eeg/source-localization.md](docs/eeg/source-localization.md).

---

### 3. Behavioral Analysis

Statistical analyses relating EEG features to pain ratings, temperature, and experimental conditions.

| Mode | Description |
|------|-------------|
| `compute` | Run statistical analyses |
| `visualize` | Generate behavioral plots |

**19 computation stages:**

| Stage | Description |
|-------|-------------|
| `trial_table` | Build trial-level feature table (events + features merged) |
| `lag_features` | Temporal dynamics (prev_*, delta_*) for habituation |
| `predictor_residual` | Rating − f(temperature): pain beyond stimulus intensity |
| `temperature_models` | Temperature→rating model comparison + breakpoint detection |
| `correlations` | Feature–pain rating correlations (partial, permutation-tested) |
| `predictor_sensitivity` | Pain sensitivity profiling |
| `condition` | Condition comparison (high vs. low pain, effect sizes) |
| `temporal` | Temporal dynamics across trial phases |
| `regression` | Trialwise regression models |
| `models` | Sensitivity model families (OLS, robust, quantile, logistic) |
| `stability` | Within-subject cross-run stability |
| `icc` | Run-level reliability (intraclass correlation) |
| `consistency` | Effect direction consistency across outcomes/methods |
| `influence` | Influential observation detection (Cook's D, leverage) |
| `cluster` | Cluster permutation tests |
| `mediation` | Mediation analysis |
| `moderation` | Moderation analysis |
| `mixed_effects` | Mixed-effects models |
| `report` | Single-subject summary report |

> **Note:** All the behavior computations are highly configurable through the TUI.
>
> For detailed documentation of every stage's exact computation method, statistical safeguards, and configuration, see [eeg_pipeline/analysis/behavior/README.md](eeg_pipeline/analysis/behavior/README.md).

```bash
# Run all behavioral analyses
eeg-pipeline behavior compute --subject 0001

# Run specific computations
eeg-pipeline behavior compute --subject 0001 \
  --computations correlations condition temporal

# Control for temperature with permutation testing
eeg-pipeline behavior compute --subject 0001 \
  --control-temperature --n-perm 1000

# Robust correlations with Bayes factors
eeg-pipeline behavior compute --subject 0001 \
  --robust-correlation percentage_bend --compute-bayes-factors

# Visualize behavioral plots
eeg-pipeline behavior visualize --subject 0001

# List all available stages
eeg-pipeline behavior compute --list-stages
```

For up-to-date flags and defaults, run `eeg-pipeline behavior --help`.


### 4. Machine Learning

Trial-level predictive modeling with leave-one-subject-out (LOSO) cross-validation.

| Mode | Description |
|------|-------------|
| `regression` | LOSO regression predicting pain intensity |
| `classify` | Binary pain classification (SVM, LR, RF, CNN) |
| `timegen` | Time-generalization analysis |
| `model_comparison` | Compare ElasticNet vs Ridge vs RandomForest |
| `incremental_validity` | Quantify Δ performance when adding EEG over baseline |
| `uncertainty` | Conformal prediction intervals |
| `shap` | SHAP-based feature importance |
| `permutation` | Permutation-based feature importance |

```bash
# LOSO regression (requires ≥2 subjects)
eeg-pipeline ml regression --subject 0001 --subject 0002 --subject 0003

# Classification with SVM
eeg-pipeline ml classify --subject 0001 --subject 0002 --classification-model svm

# SHAP feature importance
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

# List available stages
eeg-pipeline ml --list-stages
```

For up-to-date flags and defaults, run `eeg-pipeline ml --help`.



### 5. fMRI Preprocessing

Containerized fMRIPrep preprocessing via Docker or Apptainer.

```bash
# Docker-based fMRIPrep
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
| `--fmriprep-image` | Docker image tag or apptainer URI | `nipreps/fmriprep:25.2.4` |
| `--output-spaces` | Output spaces | `MNI152NLin2009cAsym`, `T1w` |
| `--fs-license-file` | FreeSurfer license path | `paths.freesurfer_license` or `EEG_PIPELINE_FREESURFER_LICENSE` |
| `--fs-subjects-dir` | FreeSurfer SUBJECTS_DIR | auto |
| `--ignore` | Skip steps (e.g., `fieldmaps slicetiming`) | none |

---

### 6. fMRI Analysis (under development)

Subject-level GLM contrasts and trial-wise beta estimation via nilearn.

| Mode | Description |
|------|-------------|
| `first-level` | First-level GLM + contrast maps |
| `beta-series` | Trial-wise beta-series estimation (for EEG–fMRI fusion) |
| `lss` | Least-squares-separate trial betas |

```bash
# First-level GLM
eeg-pipeline fmri-analysis first-level --subject 0001 \
  --contrast-name contrast \
  --cond-a-value stimulation --cond-b-value fixation_rest

# With fMRIPrep preprocessed BOLD
eeg-pipeline fmri-analysis first-level --subject 0001 \
  --input-source fmriprep --fmriprep-space MNI152NLin2009cAsym \
  --cond-a-value stimulation --cond-b-value fixation_rest

# Beta-series for EEG–fMRI fusion
eeg-pipeline fmri-analysis beta-series --subject 0001 \
  --cond-a-value stimulation --cond-b-value fixation_rest

# LSS estimation
eeg-pipeline fmri-analysis lss --subject 0001 \
  --cond-a-value stimulation --cond-b-value fixation_rest

# Generate plots and HTML report
eeg-pipeline fmri-analysis first-level --subject 0001 \
  --cond-a-value stimulation --cond-b-value fixation_rest \
  --plots --plot-html-report
```

**Key options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--input-source` | `fmriprep` or `bids_raw` | `fmriprep` |
| `--hrf-model` | `spm`, `flobs`, `fir` | `spm` |
| `--confounds-strategy` | `auto`, `none`, `motion6`...`motion24+wmcsf+fd` | `auto` |
| `--smoothing-fwhm` | Spatial smoothing kernel (mm) | `5.0` |
| `--output-type` | `z-score`, `t-stat`, `cope`, `beta` | `z-score` |
| `--resample-to-freesurfer` | Resample to FreeSurfer space | disabled |
| `--plots` | Generate per-subject figures | disabled |
| `--plot-html-report` | Write HTML report | disabled |
| `--write-design-matrix` | Save design matrices (TSV + PNG) | disabled |

---

### 7. Plotting (under development)

Curated visualization suites driven by a JSON plot catalog.

| Mode | Description |
|------|-------------|
| `visualize` | Render selected plot suites |
| `tfr` | Time-frequency representation plots |

```bash
# Render all available plots
eeg-pipeline plotting visualize --subject 0001 --all-plots

# Render specific plot groups
eeg-pipeline plotting visualize --subject 0001 --groups power behavior

# TFR visualization
eeg-pipeline plotting tfr --subject 0001

# Export as SVG and PDF
eeg-pipeline plotting visualize --subject 0001 --all-plots --formats svg pdf

# Group-level aggregate plots
eeg-pipeline plotting visualize --subject 0001 --subject 0002 --analysis-scope group
```

**Key options:** `--plots <PLOT_ID>`, `--groups <power|connectivity|aperiodic|phase|erds|complexity|spectral|ratios|asymmetry|microstates|erp|tfr|behavior>`, `--all-plots`, `--formats png|svg|pdf`, `--analysis-scope subject|group`

---

### 8. Validation

Data integrity and schema checks across the pipeline.

| Mode | Description |
|------|-------------|
| `quick` | Fast integrity check (default) |
| `all` | Validate everything |
| `epochs` | Validate cleaned epochs (.fif files) |
| `features` | Validate feature tables (schema, completeness) |
| `behavior` | Validate behavioral data |
| `bids` | Validate BIDS compliance |

```bash
# Quick validation
eeg-pipeline validate

# Full validation for specific subjects
eeg-pipeline validate all --subjects 0001 0002

# JSON output (for CI/scripting)
eeg-pipeline validate all --json
```

---

### 9. Stats & Info

Inspect pipeline state, discover subjects, and review current configuration.

**Info modes:**

| Mode | Description |
|------|-------------|
| `subjects` | List discovered subjects across BIDS, epochs, features |
| `features` | Show feature availability per subject |
| `config` | Print current configuration |
| `version` | Show pipeline version |
| `plotters` | List available plot definitions |
| `discover` | Auto-discover subjects from all data sources |
| `rois` | Show configured ROI definitions |
| `fmri-conditions` | List fMRI event conditions |
| `fmri-columns` | List available fMRI events.tsv columns |
| `multigroup-stats` | Cross-subject feature statistics |
| `ml-feature-space` | Preview ML feature matrix dimensions |

**Stats:** Pipeline-wide dashboard showing subject counts, feature coverage, storage usage, and processing status.

```bash
# List discovered subjects
eeg-pipeline info subjects

# Show feature availability for a subject
eeg-pipeline info features 0001

# Show current configuration
eeg-pipeline info config

# Preview ML feature space
eeg-pipeline info ml-feature-space

# Pipeline statistics dashboard
eeg-pipeline stats
```

---

## Interactive TUI

A terminal-based UI built with Go and [Bubble Tea](https://github.com/charmbracelet/bubbletea) that provides an interactive, menu-driven interface over the Python CLI. Under active development.

**Prerequisites:** Go 1.21+, Python environment with pipeline dependencies installed.

```bash
# Build and run (from repository root)
cd eeg_pipeline/cli/tui
go build -o eeg-tui .
./eeg-tui
```

Or build and run in one line:

```bash
cd eeg_pipeline/cli/tui && go build -o eeg-tui . && ./eeg-tui
```

**What the TUI provides:**

- **Guided wizards** for every pipeline stage (preprocessing, features, behavior, ML, fMRI, etc.)
- **Subject selection** with auto-discovery from BIDS/derivatives
- **Parameter configuration** with sensible defaults and validation
- **Real-time execution** with progress reporting and log streaming
- **Source localization wizard** with Docker-based BEM/trans auto-generation

The TUI calls the same Python CLI under the hood, so all results are identical.

---

## Configuration

All defaults live in `eeg_pipeline/utils/config/eeg_config.yaml`. CLI flags override config values at runtime.

**Key configuration sections:**

| Section | Controls |
|---------|----------|
| `project` | Task name (`task`), random seed (`42`) |
| `paths` | BIDS root, derivatives root, source data, FreeSurfer dirs |
| `eeg` | Montage (`easycap-M1`), reference (`average`), EOG/ECG channels |
| `preprocessing` | Filter settings, resampling, break detection, clean events |
| `pyprep` | PyPREP bad channel detection settings (RANSAC, repeats) |
| `ica` | Method, components, probability threshold, labels to keep |
| `epochs` | Time window, baseline, rejection method (autoreject) |
| `frequency_bands` | Band definitions (delta through gamma) |
| `time_windows` | Active and baseline windows |
| `rois` | ROI channel groupings |
| `feature_engineering` | Per-category settings, spatial transforms, parallelization |
| `time_frequency_analysis` | TFR parameters, baseline mode |
| `behavior_analysis` | Correlation method, permutation, temperature control, stage toggles |
| `fmri_preprocessing` | fMRIPrep engine, image, output spaces |

---

## Subject Selection

Most subject-processing commands (`preprocessing`, `features`, `behavior`, `ml`, `plotting`, `fmri`, `fmri-analysis`) accept these options:

| Option | Description |
|--------|-------------|
| `--subject XXXX` / `-s XXXX` | Single subject (repeatable) |
| `--all-subjects` | Process all discovered subjects |
| `--group all` or `--group A,B,C` | Process a named group or comma-separated list |
| `--task` / `-t` | Override task label (default from config: `task`) |
| `--dry-run` | Preview what would run without executing |
| `--json` | Output in JSON format (for TUI/scripting) |

`validate` uses `--subjects`, and `info features` takes a positional subject ID (e.g., `eeg-pipeline info features 0001`).

```bash
# Multiple subjects
eeg-pipeline features compute --subject 0001 --subject 0002 --subject 0003

# All subjects
eeg-pipeline features compute --all-subjects

# Dry run
eeg-pipeline ml regression --all-subjects --dry-run
```

---

## Output Formats

Feature tables are saved as **Parquet** (default, recommended) with optional TSV/CSV export:

```bash
# Also save CSV alongside Parquet
eeg-pipeline features compute --subject 0001 --also-save-csv
```

Feature outputs follow a consistent structure per subject:

```
derivatives/sub-XXXX/eeg/features/
├── power/
│   ├── features_power.parquet          # Feature table
│   └── metadata/
│       └── features_power.json         # Extraction config + column descriptions
├── connectivity/
│   ├── features_connectivity.parquet
│   └── metadata/
├── aperiodic/
│   ├── features_aperiodic.parquet
│   └── metadata/
│       └── qc/                         # Per-channel QC tables
└── ...
```

Plots are saved as PNG by default, with optional SVG and PDF:

```bash
eeg-pipeline plotting visualize --subject 0001 --formats png svg pdf
```

---

## Advanced Topics

### Source Localization (EEG-only and fMRI-constrained)

Supports both template-based (fsaverage) and subject-specific fMRI-constrained source localization with LCMV beamformer or eLORETA. The TUI includes a dedicated wizard for auto-generating BEM models and coregistration transforms via Docker.

See the full tutorial: [docs/eeg/source-localization.md](docs/eeg/source-localization.md)

### fMRI Raw-to-BIDS Conversion

DICOM-to-BIDS conversion with event generation from PsychoPy logs. Requires `dcm2niix` on PATH.

See: [docs/fmri/raw-to-bids.md](docs/fmri/raw-to-bids.md)

### Docker-based FreeSurfer + MNE

A Docker image is provided for BEM generation and coregistration:

```bash
docker build --platform linux/amd64 -t freesurfer-mne:7.4.1 \
  -f eeg_pipeline/docker_setup/Dockerfile.freesurfer-mne .
```

This image bundles FreeSurfer 7.4.1 + MNE-Python on Python 3.11 and is used for `recon-all`, `watershed_bem`, BEM solution generation, and coregistration.

### EEG–fMRI Fusion via ML

Predict trial-wise fMRI signature expression (NPS, SIIPS1) from EEG features:

```bash
eeg-pipeline ml regression --subject 0001 --subject 0002 \
  --target fmri_signature \
  --fmri-signature-name NPS \
  --fmri-signature-method beta-series \
  --fmri-signature-metric dot
```

Available signatures: `NPS`, `SIIPS1`. Methods: `beta-series`, `lss`. Metrics: `dot`, `cosine`, `pearson_r`.

**NPS (Neuroimaging Pain Signature):** The NPS weight map is not publicly distributed. To compute NPS expression, obtain the signature weights from the original authors and add them under a weights root (e.g. `path/to/weights/NPS/weights_NSF_grouppred_cvpcr.nii.gz`), then pass that root via `--pain-signature-weights` or your config.

**SIIPS1 (Stimulus Intensity Independent Pain Signature-1):** SIIPS1 weights are publicly available from the [CANlab Neuroimaging_Pattern_Masks](https://github.com/canlab/Neuroimaging_Pattern_Masks) repository. Clone or download the repo, then copy `Multivariate_signature_patterns/2017_Woo_SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz` into your weights root as `SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz`, and pass that root via `--pain-signature-weights` or your config.

### Spatial Transforms (CSD/Laplacian)

Phase-based features (connectivity, ITPC, PAC) benefit from current source density (CSD) to reduce volume conduction. CSD is applied **per feature family** by default:

| Family | Default transform |
|--------|-------------------|
| Connectivity, ITPC, PAC | `csd` |
| Power, aperiodic, bursts, ERDS, complexity, ratios, asymmetry, spectral, ERP, quality, microstates | `none` |

Override globally: `--spatial-transform csd` or per-family in the YAML config.

### Individualized Alpha Frequency (IAF)

Enable IAF-based adaptive frequency bands derived from posterior baseline PSD:

```bash
eeg-pipeline features compute --subject 0001 --iaf-enabled
```

### Analysis Modes

| Mode | Description |
|------|-------------|
| `group_stats` | Default. Cross-trial estimates allowed (one row per subject/condition). |
| `trial_ml_safe` | ML/CV-safe. Cross-trial estimators require `train_mask`; leakage-prone paths are blocked or reduced (e.g., microstate template fitting uses training trials only when `train_mask` is provided). |

```bash
# For ML pipelines, enforce safety
eeg-pipeline features compute --subject 0001 --analysis-mode trial_ml_safe
```

**Leakage guardrails in `trial_ml_safe`:**

- Evoked-subtracted aperiodic features require `train_mask` (including precomputed extraction paths).
- Dynamic connectivity state-transition metrics (state clustering across trials/windows) are disabled.
- For connectivity `condition`/`subject` granularity, phase estimation auto-switches to `across_epochs` outside CV to avoid biased per-epoch phase averaging.
- In CV (`train_mask` present), `phase_estimator=across_epochs` is blocked by default to prevent leakage; prefer `within_epoch` unless you intentionally override.
- Source-space `wpli`/`plv` connectivity is cross-epoch by definition; outputs are marked as broadcast/non-i.i.d., and in `trial_ml_safe` mode the estimate is fit on `train_mask` trials.

---

## Dependencies

Core scientific dependencies:

| Package | Version | Purpose |
|---------|---------|---------|
| **MNE-Python** | 1.9.0 | EEG processing, source localization |
| **MNE-BIDS** | 0.16.0 | BIDS I/O |
| **MNE-Connectivity** | 0.7.0 | Functional connectivity |
| **MNE-ICALabel** | 0.7.0 | Automatic ICA classification |
| **PyPREP** | 0.4.3 | Bad channel detection |
| **specparam** | 2.0.0rc3 | Aperiodic (1/f) fitting |
| **Nilearn** | 0.11.1 | fMRI GLM and neuroimaging |
| **NiBabel** | ≥3.2.0 | NIfTI/CIFTI I/O |
| **scikit-learn** | ≥1.0.0 | Machine learning models |
| **SHAP** | ≥0.40.0 | Feature importance |
| **PyTorch** | 2.7.1 | Deep learning models |
| **NetworkX** | 3.5 | Graph-theoretic metrics |
| **bctpy** | 0.6.1 | Brain Connectivity Toolbox |
| **statsmodels** | ≥0.13.0 | Statistical models |
| **antropy** | ≥0.1.9 | Complexity measures |
| **NumPy** | ≥1.24, <2.0 | Array computation |
| **SciPy** | 1.15.3 | Scientific computing |
| **pandas** | 2.3.0 | Data manipulation |

`pyproject.toml` is the single source of truth for dependencies.
`requirements.txt` is a thin compatibility shim that installs `-e ".[ml]"`.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contributing

Contributions are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes and add tests where applicable
4. Run the test suite: `pytest`
5. Submit a pull request
