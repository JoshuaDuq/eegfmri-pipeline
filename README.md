# EEG–fMRI Analysis Pipeline

[![Python ≥ 3.11](https://img.shields.io/badge/python-≥3.11-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![BIDS](https://img.shields.io/badge/data-BIDS-orange.svg)](https://bids-specification.readthedocs.io/)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://joshuaduq.github.io/eegfmri-pipeline/)

Modular neuroimaging analysis suite for simultaneous or separate EEG and fMRI data. Runs from BIDS-formatted raw data through preprocessing, 16 EEG feature families, behavioral statistics, machine learning, and fMRI GLM — all from a single CLI or interactive TUI.

<p align="center">
  <img src="docs/screenshots/tui_main_menu.png" width="800" alt="Interactive TUI"/>
</p>

Full documentation: **[joshuaduq.github.io/eegfmri-pipeline](https://joshuaduq.github.io/eegfmri-pipeline/)**

---

## Pipeline Overview

Four sequential stages. Each stage writes BIDS derivatives consumed by the next. `trial_id` in `proc-clean_events.tsv` is the join key across all stages.

**Stage 1 — Preprocessing**
Bad-channel detection (deviation + correlation + RANSAC, 3×), ICA fitting (extended Infomax, 99% variance) and labeling (ICLabel p > 0.8), epoch creation (tmin = −7 s / tmax = 15 s), and Autoreject. Outputs `proc-clean_epo.fif` and `proc-clean_events.tsv`.

**Stage 2 — Feature Extraction**
16 feature families per trial: power, spectral, aperiodic, ERP, ERDS, ratios, asymmetry, microstates, connectivity, directed connectivity, ITPC, PAC, source localization, complexity, bursts, and quality. Outputs one Parquet file per family under `features/<family>/`.

**Stage 3a — Behavioral Statistics**
Partial Spearman correlations with permutation p-values, predictor residualization, OLS regression (HC3), ICC(3,1), Welch t-tests, temporal cluster permutation, and BH + Simes FDR correction. Outputs to `stats/`.

**Stage 3b — Machine Learning**
Nested LOSO cross-validation with regression (ElasticNet / Ridge / RF, Yeo-Johnson target) and classification (SVM / LR / RF / EEGNet), temporal generalization, SHAP attribution, conformal intervals, and permutation tests. Outputs to `ml/`.

**Stage 4 — fMRI** *(under active development)*
fMRIPrep preprocessing, first-level Nilearn GLM (SPM HRF, cosine drift, 0.008 Hz HP), trial-wise beta estimation (beta-series / LSS), group one-sample GLM with max-T permutation, and resting-state connectivity. Outputs to `sub-*/fmri/` and `group/fmri/`.

> `fmri`, `fmri-analysis`, and `plotting` are still evolving — interfaces, defaults, and outputs may change between releases.

---

## Installation

```bash
git clone https://github.com/JoshuaDuq/eegfmri-pipeline.git && cd eegfmri-pipeline
python3.11 -m venv .venv311 && source .venv311/bin/activate
pip install -e ".[dev,ml]"
```

---

## Quick Start

Place BIDS EEG data under `data/bids_output/eeg/`, then:

```bash
eeg-pipeline validate quick
eeg-pipeline info subjects
eeg-pipeline preprocessing full --subject 0001
eeg-pipeline features compute --subject 0001 --analysis-mode trial_ml_safe
eeg-pipeline ml regression --all-subjects
```

Full walkthrough: [Quick Start guide](https://joshuaduq.github.io/eegfmri-pipeline/user_guide/quickstart.html).

---

## CLI Reference

Use `--help` on any command to inspect its options:

```bash
eeg-pipeline <command> --help
```

Available commands and their modes:

- `validate` — `quick`, `all`, `epochs`, `features`, `behavior`, `bids`
- `info` — `subjects`, `features`, `config`, `version`, `plotters`, `discover`, `rois`, `fmri-conditions`, `fmri-columns`, `multigroup-stats`, `ml-feature-space`
- `preprocessing` — `full`, `bad-channels`, `ica`, `epochs`
- `features` — `compute`, `visualize`
- `behavior` — `compute`, `visualize`
- `ml` — `regression`, `classify`, `timegen`, `model_comparison`, `incremental_validity`, `uncertainty`, `shap`, `permutation`
- `fmri` — `preprocess`
- `fmri-analysis` — `first-level`, `second-level`, `beta-series`, `lss`, `rest`
- `plotting` — `visualize`, `tfr`
- `stats` — `summary`, `subjects`, `features`, `storage`, `timeline`

**Optional TUI** (requires Go 1.21+):

```bash
cd eeg_pipeline/cli/tui && go build -o eeg-tui . && ./eeg-tui
```

---

## Contributing & License

Contributions welcome — see the [contributing guide](https://joshuaduq.github.io/eegfmri-pipeline/contributing.html).
MIT — see [LICENSE](LICENSE).
