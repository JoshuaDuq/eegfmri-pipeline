# Getting Started

This page is the shortest path from a fresh clone to a working install.
For the complete copy-paste workflow and the full command matrix, see the
[Quick Start](../user_guide/quickstart) page.

## What the Pipeline Does

The pipeline processes BIDS-formatted EEG and fMRI data through four stages:

1. **Preprocessing** — bad-channel detection (PyPREP), ICA fitting and
   artifact labeling (ICLabel), epoch creation and autoreject.
   Writes `proc-clean_epo.fif` and `proc-clean_events.tsv`.

2. **Feature extraction** — 16 EEG feature families (power, connectivity,
   aperiodic, ERP, ERDS, PAC, ITPC, microstates, complexity, and more).
   One row per trial; written as Parquet tables.

3. **Statistics and ML** — behavioral correlations with permutation FDR,
   and nested LOSO cross-validated regression / classification with SHAP.

4. **fMRI** (optional) — containerized fMRIPrep preprocessing, Nilearn
   first-level GLM, trial-wise beta estimation, and group inference.

The `trial_id` column in `proc-clean_events.tsv` is the join key that
aligns EEG features, fMRI betas, and behavioral targets across all stages.

## Installation

Requirements:

- Python 3.11 or newer
- A virtual environment
- Go 1.21+ only if you want the optional TUI

```bash
git clone https://github.com/JoshuaDuq/eegfmri-pipeline.git
cd eegfmri-pipeline
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -e ".[dev,ml]"
```

## Smoke Test

Validate inputs and confirm subject discovery before running any analysis:

```bash
eeg-pipeline validate quick    # BIDS structure and config consistency
eeg-pipeline info subjects     # List all discoverable subjects and run counts
eeg-pipeline info config       # Show the fully resolved active configuration
```

If `validate quick` exits cleanly and `info subjects` lists your subjects,
the environment is set up correctly.

## Next Steps

- Full copy-paste workflow: [Quick Start](../user_guide/quickstart)
- Data and events requirements: [Data Layout](../user_guide/data_layout)
- Configuration reference: [Configuration](../user_guide/configuration)
- Command flags: [CLI Reference](../user_guide/cli/index)
- Guided terminal interface: [TUI](../user_guide/tui)
- Install details and environment variables: [Install](../install)

## Optional TUI

The Go TUI wraps the same CLI and runs commands from the repository root.

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
