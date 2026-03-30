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
- Trial-level EEG feature extraction across 16 feature families
- Behavioral statistics with robust inference and multiple-comparison control
- Nested machine-learning workflows for regression and classification
- Source localization workflows for EEG analyses
- fMRI preprocessing and GLM-based analysis
- CLI and TUI interfaces for interactive use and batch execution

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

Place BIDS-formatted EEG data under `data/bids_output/eeg/`, then run:

```bash
eeg-pipeline validate quick
eeg-pipeline info subjects
eeg-pipeline preprocessing full --subject 0001
eeg-pipeline features compute --subject 0001 --analysis-mode trial_ml_safe
eeg-pipeline ml regression --all-subjects
```

For the full walkthrough from data layout to results, see the
[Quick start guide](https://joshuaduq.github.io/eegfmri-pipeline/user_guide/quickstart.html).

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
