# EEG–fMRI Analysis Pipeline

[Python ≥ 3.11](https://www.python.org)
[License: MIT](LICENSE)
[BIDS](https://bids-specification.readthedocs.io/)
[Docs](https://joshuaduq.github.io/eegfmri-pipeline/)

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
- Scriptable CLI commands and guided terminal UI workflows

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

MIT. See [LICENSE](LICENSE).