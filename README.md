# EEG–fMRI Analysis Pipeline

[![Python ≥ 3.11](https://img.shields.io/badge/python-≥3.11-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![BIDS](https://img.shields.io/badge/data-BIDS-orange.svg)](https://bids-specification.readthedocs.io/)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://joshuaduq.github.io/eegfmri-pipeline/)

Modular neuroimaging analysis suite — EEG preprocessing, 16 feature families,
behavioural statistics, machine learning, fMRI GLM, and source localisation,
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
| [Quick Start](https://joshuaduq.github.io/eegfmri-pipeline/user_guide/quickstart.html) | End-to-end walkthrough |
| [User Guide](https://joshuaduq.github.io/eegfmri-pipeline/user_guide/index.html) | Data layout, configuration, CLI, TUI |
| [Methods](https://joshuaduq.github.io/eegfmri-pipeline/methods/index.html) | Notation, formulas, output schemas |
| [API](https://joshuaduq.github.io/eegfmri-pipeline/api/index.html) | Public Python API reference |

---

## Quick Start

```bash
git clone https://github.com/JoshuaDuq/eegfmri-pipeline.git && cd eegfmri-pipeline
python3.11 -m venv .venv311 && source .venv311/bin/activate
pip install -e ".[dev,ml]"
```

Place BIDS data under `data/bids_output/eeg/`, then:

```bash
eeg-pipeline preprocessing full --subject 0001
eeg-pipeline features compute --subject 0001
eeg-pipeline ml regression --all-subjects
```

**Optional TUI** (requires Go 1.21+):

```bash
cd eeg_pipeline/cli/tui && go build -o eeg-tui . && ./eeg-tui
```

---

## Pipeline

| Command | Description |
|:--------|:------------|
| `preprocessing` | Bad channel detection · ICA · epoch creation |
| `features` | 16 trial-level EEG feature families |
| `behavior` | Correlations · regression · ICC · condition comparisons |
| `ml` | LOSO regression · classification · SHAP · permutation tests |
| `fmri` | fMRIPrep preprocessing (Docker / Apptainer) |
| `fmri-analysis` | First-level GLM · group inference · beta-series · resting-state |
| `plotting` | 40+ plot types across all domains |

---

## Contributing & License

Contributions welcome — see the [contributing guide](https://joshuaduq.github.io/eegfmri-pipeline/contributing.html).
MIT — see [LICENSE](LICENSE).
