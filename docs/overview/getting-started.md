# Getting Started

This page is the shortest path from a fresh clone to a working public pipeline install.

## Installation

Requirements:

- Python 3.11 or newer
- A virtual environment
- Go 1.21+ only if you want the optional TUI

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,ml]"
```

For tooling that expects a requirements file:

```bash
python -m pip install -r requirements.txt
```

## First Commands

```bash
eeg-pipeline --help
eeg-pipeline preprocessing --help
eeg-pipeline features --help
eeg-pipeline behavior --help
eeg-pipeline ml --help
eeg-pipeline fmri --help
eeg-pipeline fmri-analysis --help
```

## CLI-Only Workflow

If you do not need the terminal UI, the Python environment is enough.
Typical first-run sequence:

1. Validate the dataset layout with `eeg-pipeline validate`.
2. Run EEG preprocessing for a small subject set.
3. Extract features from cleaned epochs.
4. Run behavior, machine learning, plotting, or fMRI analysis against the derivatives.

## Optional TUI

The Go TUI wraps the same CLI and executes commands from the repository root.

```bash
cd eeg_pipeline/cli/tui
go build -o eeg-tui .
./eeg-tui
```

## Documentation Build

```bash
python -m pip install -e ".[docs]"
make docs
make docs-check
```
