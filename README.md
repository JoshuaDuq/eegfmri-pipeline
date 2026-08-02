# EEG–fMRI Analysis Pipeline

[![Python ≥ 3.11](https://img.shields.io/badge/python-≥3.11-blue.svg)](https://www.python.org)
[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)
[![BIDS](https://img.shields.io/badge/data-BIDS-orange.svg)](https://bids-specification.readthedocs.io/)
[![Docs](https://img.shields.io/badge/docs-Sphinx-blue.svg)](https://joshuaduq.github.io/eegfmri-pipeline/)

A research pipeline for reproducible EEG, fMRI, and multimodal EEG–fMRI
analysis on BIDS-organized data.

The project brings common neuroimaging workflows into one documented command
surface: preprocessing, feature extraction, behavioral statistics, machine
learning, source localization, and fMRI analysis. It is built for research code
that needs to remain inspectable: methods are explicit, configuration is
centralized, and derivatives are organized around reproducible analysis stages.

The [Sphinx documentation](https://joshuaduq.github.io/eegfmri-pipeline/) is the
canonical guide for installation, configuration, methods, command references,
and output formats.

<p align="center">
  <img src="docs/screenshots/tui_main_menu.png" width="800" alt="Interactive TUI"/>
</p>

<p align="center"><em>Guided terminal workflows expose the same analysis stages
available through the CLI.</em></p>

## Why This Project Exists

EEG and fMRI projects often accumulate one-off scripts, implicit assumptions,
and fragile handoffs between preprocessing, statistics, and modeling. This
pipeline aims to make those handoffs visible. The same workflows can be run from
a Python CLI or explored through an optional terminal UI, with detailed methods
and output contracts maintained in the documentation.

The implementation follows the scientific Python ecosystem, including
MNE-Python, Nilearn, NumPy, SciPy, pandas, and scikit-learn.

## Workflows

- EEG preprocessing with artifact handling, ICA, epoching, and cleaned derivatives
- EEG feature extraction for task-based and resting-state analyses
- Behavioral statistics and machine-learning workflows
- EEG source localization with template and subject-specific paths
- fMRI preprocessing, GLM analysis, and EEG–fMRI integration workflows
- Native simultaneous EEG-fMRI gradient and pulse-artifact correction from original 5 kHz recordings
- Scriptable CLI commands and guided terminal UI workflows

## Typical EEG Preprocessing

The maintained MNE workflow separates fitting, manual review, and cleaning so
ICA decisions remain inspectable:

1. Convert EEG to BIDS and merge behavioral events.
2. Detect bad channels with PyPREP. For a shared cross-run ICA, set
   `pyprep.bad_channel_sync_policy=subject_union`.
3. Run `eeg-pipeline preprocessing ica`, inspect the MNE HTML report and
   `*_proc-ica_components.tsv`, and record any manual exclusions.
4. Run `eeg-pipeline preprocessing epochs` with
   `ica.manual_review_complete=true` to apply the reviewed ICA and reject bad
   epochs.

For BrainVision Analyzer-corrected 1 kHz EEG, preserve the volume and R-peak
annotations during BIDS conversion and set
`preprocessing.brainvision_analyzer.enabled=true`. This enables strict
marker-dependent cardiac and scanner-artifact quality checks. Data entering
from the original 5 kHz recording instead uses the separate native EEG-fMRI
artifact-correction workflow before the stages above.

The optional `ica.band_specific_report.enabled=true` setting adds independent
delta/theta, alpha, beta, gamma, and 1–30 Hz ICA diagnostics to the HTML report.
These decompositions are exploratory: the standard 1–100 Hz ICA and its
reviewed component table remain authoritative for cleaning.

Set `ica.cardiac_review.enabled=true` to add a manual ECG review before the ICA
component dossiers. It detects R peaks directly from the ECG channel, plots
run-level signals, and shows every standard ICA component with its R-locked
waveform and MNE ECG-correlation and CTPS outputs. It adds no pipeline-specific
score, quality label, ranking, recommendation, or automatic exclusion and does
not depend on Analyzer R annotations.

See the [EEG preprocessing quick start](docs/user_guide/quickstart.rst),
[CLI reference](docs/user_guide/cli/preprocessing.rst), and
[scientific methods](docs/methods/eeg/preprocessing.rst) for commands, outputs,
and interpretation guidance.

## Study-Specific Guides

- [Study 1 run guide](studies/pain_study/study1/RUN_GUIDE.md) for rerunning the
  trial-wise EEG-to-fMRI pain-signature workflow and locating article-ready outputs
- [Study 1 protocol README](studies/pain_study/study1/README.md) for the scientific
  rationale, estimands, preprocessing assumptions, and interpretation framework

## Explore the Documentation

The Sphinx site is where the project opens up:

- [Documentation home](https://joshuaduq.github.io/eegfmri-pipeline/) for the full map
- [Installation](https://joshuaduq.github.io/eegfmri-pipeline/install.html) for environment setup
- [Quick start](https://joshuaduq.github.io/eegfmri-pipeline/user_guide/quickstart.html) for a first workflow
- [User guide](https://joshuaduq.github.io/eegfmri-pipeline/user_guide/index.html) for data layout, configuration, CLI, and TUI usage
- [Methods reference](https://joshuaduq.github.io/eegfmri-pipeline/methods/index.html) for algorithms, assumptions, and output schemas
- [FAQ](https://joshuaduq.github.io/eegfmri-pipeline/faq.html) for common setup and workflow questions

If you are new to the project, start with the quick start. If you are reviewing
the scientific assumptions behind a workflow, start with the methods reference.

## Minimal Install

Requires Python 3.11 or later. Some workflows also require external neuroimaging
tools, containers, or a FreeSurfer license; see the installation guide for the
complete setup.

```bash
git clone https://github.com/JoshuaDuq/eegfmri-pipeline.git
cd eegfmri-pipeline
python -m venv .venv
# Activate .venv using the command for your shell, then:
python -m pip install --upgrade pip
python -m pip install -e ".[dev,ml]"
eeg-pipeline --help
```

The `ml` extra installs PyTorch-backed models. If you do not need those models,
install `".[dev]"` instead.

## Scope and Status

The EEG, feature extraction, behavioral statistics, machine learning, and source
localization workflows are documented and maintained. fMRI and plotting
workflows are active areas of development, so validate critical analyses after
upgrades and confirm derivative paths before downstream use.

## Contributing

Development guidance, testing expectations, and branch conventions are covered
in the [contributing guide](https://joshuaduq.github.io/eegfmri-pipeline/contributing.html).

## License

GPL-3.0-only. See [LICENSE](LICENSE).

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party provenance.
