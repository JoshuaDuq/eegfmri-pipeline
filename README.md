# EEG–fMRI Analysis Pipeline

[![Python ≥ 3.11](https://img.shields.io/badge/python-≥3.11-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![BIDS](https://img.shields.io/badge/data-BIDS-orange.svg)](https://bids-specification.readthedocs.io/)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://joshuaduq.github.io/eegfmri-pipeline/)

Modular neuroimaging analysis suite — EEG preprocessing, 16 feature families,
behavioral statistics, machine learning, fMRI GLM, and source localization,
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
| [Quick Start](https://joshuaduq.github.io/eegfmri-pipeline/user_guide/quickstart.html) | Clean walkthrough from install to outputs |
| [User Guide](https://joshuaduq.github.io/eegfmri-pipeline/user_guide/index.html) | Data layout, configuration, CLI, TUI |
| [Methods](https://joshuaduq.github.io/eegfmri-pipeline/methods/index.html) | Notation, formulas, output schemas |
| [API](https://joshuaduq.github.io/eegfmri-pipeline/api/index.html) | Public Python API reference |

---

## Quick Start

Install the package:

```bash
git clone https://github.com/JoshuaDuq/eegfmri-pipeline.git && cd eegfmri-pipeline
python3.11 -m venv .venv311 && source .venv311/bin/activate
pip install -e ".[dev,ml]"
```

Place BIDS EEG data under `data/bids_output/eeg/`, then validate the dataset and
run the first pipeline stage:

```bash
eeg-pipeline validate quick
eeg-pipeline info subjects
eeg-pipeline preprocessing full --subject 0001
eeg-pipeline features compute --subject 0001
eeg-pipeline ml regression --all-subjects
```

Available command families and modes:

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

The full, command-by-command walkthrough lives in
[Quick Start](https://joshuaduq.github.io/eegfmri-pipeline/user_guide/quickstart.html),
and the complete CLI reference is in
[User Guide](https://joshuaduq.github.io/eegfmri-pipeline/user_guide/index.html).

**Optional TUI** requires Go 1.21+:

```bash
cd eeg_pipeline/cli/tui && go build -o eeg-tui . && ./eeg-tui
```

---

## Pipeline

The pipeline is organized as a single CLI with one command family per analysis
stage. Use `--help` on any command to inspect its options:

```bash
eeg-pipeline preprocessing --help
eeg-pipeline features --help
eeg-pipeline behavior --help
eeg-pipeline ml --help
eeg-pipeline fmri --help
eeg-pipeline fmri-analysis --help
eeg-pipeline plotting --help
eeg-pipeline validate --help
eeg-pipeline info --help
eeg-pipeline stats --help
```

---

## Contributing & License

Contributions welcome — see the [contributing guide](https://joshuaduq.github.io/eegfmri-pipeline/contributing.html).
MIT — see [LICENSE](LICENSE).
