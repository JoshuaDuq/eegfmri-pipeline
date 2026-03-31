# EEG–fMRI Analysis Pipeline

[![Python ≥ 3.11](https://img.shields.io/badge/python-≥3.11-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![BIDS](https://img.shields.io/badge/data-BIDS-orange.svg)](https://bids-specification.readthedocs.io/)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://joshuaduq.github.io/eegfmri-pipeline/)

A modular neuroimaging analysis suite for EEG, fMRI, and multimodal
EEG–fMRI research. The pipeline runs from BIDS-formatted raw data through
preprocessing, feature extraction, behavioral statistics, machine learning,
source localization, and fMRI analysis from a single CLI or interactive TUI.

The Sphinx documentation is the canonical source for all detailed methods,
configuration, output formats, and command references.

<p align="center">
  <img src="docs/screenshots/tui_main_menu.png" width="800" alt="Interactive TUI"/>
</p>

## What It Does

- EEG preprocessing with artifact detection, ICA, epoching, and cleaning
- Trial-level and **resting-state** EEG feature extraction across 16 feature families
- Behavioral statistics with robust inference and multiple-comparison control
- Nested machine-learning workflows for regression and classification
- Source localization workflows for EEG analyses
- fMRI preprocessing and GLM-based analysis
- Interactive TUI and CLI interfaces for configuration and batch execution

## Documentation

The project documentation lives in Sphinx and is the source of truth for
implementation details.

- [Documentation home](https://joshuaduq.github.io/eegfmri-pipeline/)
- [Installation guide](https://joshuaduq.github.io/eegfmri-pipeline/install.html)
- [Quick start](https://joshuaduq.github.io/eegfmri-pipeline/user_guide/quickstart.html)
- [User guide](https://joshuaduq.github.io/eegfmri-pipeline/user_guide/index.html)
- [Methods reference](https://joshuaduq.github.io/eegfmri-pipeline/methods/index.html)
- [API reference](https://joshuaduq.github.io/eegfmri-pipeline/api/index.html)
- [FAQ](https://joshuaduq.github.io/eegfmri-pipeline/faq.html)
- [Contributing](https://joshuaduq.github.io/eegfmri-pipeline/contributing.html)

## Requirements

- Python 3.11 or later
- Git for cloning the repository
- Go 1.21+ only if you want to build the optional TUI
- Docker plus a FreeSurfer license only for source localization workflows

See the [installation guide](https://joshuaduq.github.io/eegfmri-pipeline/install.html)
for environment variables, optional components, and the Docker image used by
the source-localization path.

## Install

```bash
git clone https://github.com/JoshuaDuq/eegfmri-pipeline.git
cd eegfmri-pipeline
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -e ".[dev,ml]"
```

The `ml` extra installs PyTorch and is only required for the CNN classifier.
If you do not need that model, `pip install -e ".[dev]"` is sufficient.

## Quick Start

Before running any pipeline, configuration must be set: paths to your BIDS
root, derivatives directory, task type, number of parallel jobs, and—for
feature extraction—which feature families, frequency bands, and analysis mode
to use. **The recommended way to handle all of this is the TUI.**

### Using the TUI (recommended)

Build and launch the TUI once after installation:

```bash
cd eeg_pipeline/cli/tui && go build -o eeg-tui . && cd -
./eeg_pipeline/cli/tui/eeg-tui
```

The TUI walks you through every configuration step interactively—pipeline
selection, subject selection, feature families, frequency bands, spatial
options, time ranges, preprocessing stages, and advanced options—before
assembling and running the underlying CLI command. No flags to memorize; the
wizard validates your choices at each step and shows a live summary on the
home screen.

### Using the CLI directly

If you prefer scripting or headless execution, configure
`eeg_pipeline/utils/config/` first, then:

```bash
eeg-pipeline validate quick
eeg-pipeline info subjects
eeg-pipeline preprocessing full --subject 0001
eeg-pipeline features compute --subject 0001 --analysis-mode trial_ml_safe
eeg-pipeline ml regression --all-subjects
```

CLI flags for feature extraction (`--features`, `--bands`, `--rois`, etc.) map
directly to the wizard steps in the TUI. See the
[Quick start guide](https://joshuaduq.github.io/eegfmri-pipeline/user_guide/quickstart.html)
for the full walkthrough.

## Feature Extraction

Feature extraction supports both **task-based** (event-related, trial-level)
and **resting-state** paradigms. The `--analysis-mode` flag controls the
output format:

- `trial_ml_safe` — per-trial features suitable for ML pipelines
- `rest` — segment-averaged features for resting-state analyses

Feature families (power spectra, connectivity, complexity, etc.) and frequency
bands are selected per-run via the TUI wizard or the corresponding CLI flags.

## Current Scope

The EEG pipeline, feature extraction, behavioral statistics, machine learning,
and source-localization workflows are documented and maintained. The fMRI
pipeline and plotting commands are still evolving, so verify critical workflows
after upgrading.

## Contributing

Use short imperative commit subjects and keep changes focused. See the
[contributing guide](https://joshuaduq.github.io/eegfmri-pipeline/contributing.html)
for branch naming, testing, and pull-request expectations.

## License

MIT. See [LICENSE](LICENSE).
