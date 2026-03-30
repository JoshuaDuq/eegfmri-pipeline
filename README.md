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

The pipeline processes EEG and fMRI data through four consecutive stages.
Each stage writes BIDS-derivative outputs that the next stage consumes. The
``trial_id`` column in ``proc-clean_events.tsv`` is the join key across all
stages: EEG features, fMRI betas, and behavioral targets align exclusively
on this identifier.

```text
BIDS EEG / fMRI Data
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  1. EEG Preprocessing                                   │
│     Bad-channel detection: PyPREP (deviation +          │
│     correlation + optional RANSAC), 3 iterations        │
│     ICA: extended Infomax (99% variance) via            │
│     MNE-BIDS-Pipeline; artifact labeling with ICLabel   │
│     (threshold p > 0.8; keeps brain and other)          │
│     Epoching: tmin = -7 s, tmax = 15 s, autoreject     │
│     → proc-clean_epo.fif + proc-clean_events.tsv        │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  2. Feature Extraction (16 families)                    │
│     power · spectral · aperiodic · erp · erds           │
│     ratios · asymmetry · microstates · connectivity      │
│     directedconnectivity · itpc · pac                   │
│     sourcelocalization · complexity · bursts · quality  │
│     One row per trial; spatial modes: ROI, channel,     │
│     global, channel-pair                                │
│     → features/<family>/features_<family>.parquet       │
└──────────────┬─────────────────────┬────────────────────┘
               │                     │
               ▼                     ▼
┌─────────────────────┐  ┌──────────────────────────────┐
│  3a. Behavioral     │  │  3b. Machine Learning         │
│      Statistics     │  │                               │
│  Partial Spearman   │  │  LOSO nested CV: ElasticNet,  │
│  correlations,      │  │  Ridge, Random Forest         │
│  predictor          │  │  Classification: SVM, LR,     │
│  residualization,   │  │  RF, EEGNet CNN               │
│  OLS regression,    │  │  Temporal generalization,     │
│  ICC(3,1),          │  │  SHAP importance, conformal   │
│  condition Welch t, │  │  intervals, permutation test  │
│  cluster permutation│  │  Primary metric: subject-     │
│  BH FDR + Simes     │  │  level Fisher-z Pearson r     │
│  → stats/           │  │  → ml/                        │
└─────────────────────┘  └──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────┐
│  4. fMRI Pipeline (optional)                            │
│     Preprocessing: fMRIPrep (Docker / Apptainer)        │
│     First-level GLM: Nilearn FirstLevelModel            │
│     HRF: spm; drift: cosine; high-pass: 0.008 Hz        │
│     Trial-wise betas: beta-series (LSA) or LSS          │
│     Group inference: one-sample GLM + permutation       │
│     Resting-state: ROI timeseries + connectivity        │
│     EEG–fMRI fusion: predict signature from EEG         │
│     → sub-*/fmri/, group/fmri/                          │
└─────────────────────────────────────────────────────────┘
```

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
| `plotting` | `visualize`, `tfr` |
| `stats` | `summary`, `subjects`, `features`, `storage`, `timeline` |

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
