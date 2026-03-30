# EEG–fMRI Analysis Pipeline

[![Python ≥ 3.11](https://img.shields.io/badge/python-≥3.11-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![BIDS](https://img.shields.io/badge/data-BIDS-orange.svg)](https://bids-specification.readthedocs.io/)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://joshuaduq.github.io/eegfmri-pipeline/)

Modular neuroimaging analysis suite for simultaneous or separate EEG and fMRI
data. The pipeline runs from BIDS-formatted raw data through preprocessing, 16
EEG feature families, behavioral statistics, machine learning, and fMRI GLM —
all from a single CLI or interactive TUI.

<p align="center">
  <img src="docs/screenshots/tui_main_menu.png" width="800" alt="Interactive TUI"/>
</p>

---

## Documentation

**[joshuaduq.github.io/eegfmri-pipeline](https://joshuaduq.github.io/eegfmri-pipeline/)**

| | |
|:--|:--|
| [Installation](https://joshuaduq.github.io/eegfmri-pipeline/install.html) | Python setup, TUI build, Docker image |
| [Quick Start](https://joshuaduq.github.io/eegfmri-pipeline/user_guide/quickstart.html) | Full walkthrough from install to outputs, with pipeline overview |
| [User Guide](https://joshuaduq.github.io/eegfmri-pipeline/user_guide/index.html) | Data layout, configuration, CLI, TUI |
| [Methods](https://joshuaduq.github.io/eegfmri-pipeline/methods/index.html) | Formulas, CV schemes, output schemas |
| [API](https://joshuaduq.github.io/eegfmri-pipeline/api/index.html) | Public Python API reference |

---

## What It Does

Four sequential stages. Each stage writes BIDS derivatives consumed by the
next. `trial_id` in `proc-clean_events.tsv` is the join key across all stages.

| Stage | What it does | Key tools | Output |
|:---:|:---|:---|:---|
| **1 — Preprocessing** | Bad-channel detection (deviation + correlation + RANSAC, 3×); ICA fitting (extended Infomax, 99% variance) and labeling (ICLabel p > 0.8); epoch creation tmin = −7 s / tmax = 15 s, autoreject | PyPREP · MNE-BIDS-Pipeline · ICLabel | `proc-clean_epo.fif` `proc-clean_events.tsv` |
| **2 — Feature Extraction** | 16 families: power, spectral, aperiodic, ERP, ERDS, ratios, asymmetry, microstates, connectivity, directed connectivity, ITPC, PAC, source localization, complexity, bursts, quality. One row per trial. | MNE · MNE-Connectivity · specparam · scikit-learn | `features/<family>/features_<family>.parquet` |
| **3a — Behavioral Stats** | Partial Spearman correlations + permutation p-values; predictor residualization; OLS regression (HC3); ICC(3,1); condition Welch t-test; temporal cluster permutation; BH + Simes FDR | SciPy · statsmodels | `stats/` |
| **3b — Machine Learning** | Nested LOSO CV; regression (ElasticNet / Ridge / RF, Yeo-Johnson target); classification (SVM / LR / RF / EEGNet); temporal generalization; SHAP; conformal intervals; permutation test | scikit-learn · SHAP · PyTorch | `ml/` |
| **4 — fMRI** *(optional)* | fMRIPrep preprocessing; first-level Nilearn GLM (HRF spm, cosine drift, 0.008 Hz HP); trial-wise betas (beta-series / LSS); group one-sample GLM + max-T permutation; resting-state connectivity | fMRIPrep · Nilearn · NiBabel | `sub-*/fmri/` `group/fmri/` |

The **fMRI pipeline** (`fmri`, `fmri-analysis`) and the **plotting pipeline** (`plotting`) are still under active development: interfaces, defaults, and outputs may change between releases; report issues if something does not match the docs.

---

## Quick Start

Install:

```bash
git clone https://github.com/JoshuaDuq/eegfmri-pipeline.git && cd eegfmri-pipeline
python3.11 -m venv .venv311 && source .venv311/bin/activate
pip install -e ".[dev,ml]"
```

Place BIDS EEG data under `data/bids_output/eeg/`, validate, and run:

```bash
eeg-pipeline validate quick
eeg-pipeline info subjects
eeg-pipeline preprocessing full --subject 0001
eeg-pipeline features compute --subject 0001 --analysis-mode trial_ml_safe
eeg-pipeline ml regression --all-subjects
```

Full walkthrough with all stages: [Quick Start](https://joshuaduq.github.io/eegfmri-pipeline/user_guide/quickstart.html).

---

## Command Reference

| Command | Modes |
|:--|:--|
| `validate` | `quick`, `all`, `epochs`, `features`, `behavior`, `bids` |
| `info` | `subjects`, `features`, `config`, `version`, `plotters`, `discover`, `rois`, `fmri-conditions`, `fmri-columns`, `multigroup-stats`, `ml-feature-space` |
| `preprocessing` | `full`, `bad-channels`, `ica`, `epochs` |
| `features` | `compute`, `visualize` |
| `behavior` | `compute`, `visualize` |
| `ml` | `regression`, `classify`, `timegen`, `model_comparison`, `incremental_validity`, `uncertainty`, `shap`, `permutation` |
| `fmri` | `preprocess` |
| `fmri-analysis` | `first-level`, `second-level`, `beta-series`, `lss`, `rest` |
| `plotting` | `visualize`, `tfr` *(under development)* |
| `stats` | `summary`, `subjects`, `features`, `storage`, `timeline` |

Commands `fmri`, `fmri-analysis`, and `plotting` are still evolving; see the note above.

Use `--help` on any command to inspect its options:

```bash
eeg-pipeline preprocessing --help
eeg-pipeline features --help
eeg-pipeline ml --help
eeg-pipeline fmri-analysis --help
```

**Optional TUI** (requires Go 1.21+):

```bash
cd eeg_pipeline/cli/tui && go build -o eeg-tui . && ./eeg-tui
```

---

## Contributing & License

Contributions welcome — see the [contributing guide](https://joshuaduq.github.io/eegfmri-pipeline/contributing.html).
MIT — see [LICENSE](LICENSE).
