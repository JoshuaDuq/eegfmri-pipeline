# EEG–fMRI Analysis Pipeline

[![Python ≥ 3.11](https://img.shields.io/badge/python-≥3.11-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![BIDS](https://img.shields.io/badge/data-BIDS-orange.svg)](https://bids-specification.readthedocs.io/)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://joshuaduq.github.io/eegfmri-pipeline/)

A modular, reproducible analysis suite for multimodal neuroimaging research —
EEG preprocessing, 16 feature families, behavioural statistics, machine learning,
fMRI GLM, and source localisation, all from a single CLI or interactive TUI.

> This pipeline is in active development. Suggestions welcome.

<p align="center">
  <img src="docs/screenshots/tui_main_menu.png" width="800" alt="Interactive TUI — pipeline stages at a glance"/>
</p>

---

## Documentation

📖 **[joshuaduq.github.io/eegfmri-pipeline](https://joshuaduq.github.io/eegfmri-pipeline/)**

| | |
|---|---|
| [Installation](https://joshuaduq.github.io/eegfmri-pipeline/install.html) | Python setup, TUI build, Docker image, environment variables |
| [Quick Start](https://joshuaduq.github.io/eegfmri-pipeline/user_guide/quickstart.html) | 9-step walkthrough from clone to results |
| [User Guide](https://joshuaduq.github.io/eegfmri-pipeline/user_guide/index.html) | Data layout, configuration, CLI reference, TUI |
| [Methods Reference](https://joshuaduq.github.io/eegfmri-pipeline/methods/index.html) | Scientific documentation — notation, formulas, output schemas |
| [API Reference](https://joshuaduq.github.io/eegfmri-pipeline/api/index.html) | Public symbols for `eeg_pipeline` and `fmri_pipeline` |

---

## Quick Start

```bash
git clone https://github.com/JoshuaDuq/eegfmri-pipeline.git
cd eegfmri-pipeline
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -e ".[dev,ml]"
eeg-pipeline --help
```

Place BIDS-formatted EEG data under `data/bids_output/eeg/`, then run:

```bash
eeg-pipeline preprocessing full --subject 0001
eeg-pipeline features compute --subject 0001
eeg-pipeline ml regression --all-subjects
```

See the [Quick Start guide](https://joshuaduq.github.io/eegfmri-pipeline/user_guide/quickstart.html) for the full walkthrough.

---

## Pipeline Stages

| Stage | Command | Description |
|-------|---------|-------------|
| Preprocessing | `preprocessing` | Bad channel detection, ICA, epoch creation |
| Feature Extraction | `features` | 16 trial-level EEG feature families |
| Behavioural Statistics | `behavior` | Correlations, regression, ICC, condition comparisons |
| Machine Learning | `ml` | LOSO regression, classification, SHAP, permutation tests |
| fMRI Preprocessing | `fmri` | Containerised fMRIPrep (Docker / Apptainer) |
| fMRI Analysis | `fmri-analysis` | First-level GLM, group inference, beta-series, resting-state |
| Source Localisation | `features sourcelocalization` | LCMV / eLORETA, template or fMRI-constrained |
| Plotting | `plotting` | 40+ plot types across all domains |

---

## Interactive TUI

An optional terminal UI (Go + Bubble Tea) wraps the full CLI in guided wizards.

```bash
cd eeg_pipeline/cli/tui
go build -o eeg-tui .
./eeg-tui
```

Requires Go 1.21+. See the [TUI reference](https://joshuaduq.github.io/eegfmri-pipeline/user_guide/tui.html).

---

## Contributing

See [CONTRIBUTING](https://joshuaduq.github.io/eegfmri-pipeline/contributing.html) for branch conventions, code style, and testing requirements.

```bash
make test               # full test suite
make verify-structure   # repo layout guard
make docs               # build documentation
```

---

## License

MIT — see [LICENSE](LICENSE).
