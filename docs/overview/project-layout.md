# Project Layout

The repository is organized by domain rather than by a single monolithic pipeline.

| Path | Purpose |
| --- | --- |
| `eeg_pipeline/` | EEG preprocessing, feature extraction, behavior, ML, plotting, and CLI code |
| `fmri_pipeline/` | fMRI preprocessing, analysis, and utilities |
| `docs/` | Sphinx/MyST documentation source |
| `tests/` | Repository-wide test suite, including architecture and docs guards |
| `scripts/` | Public repo-level helper utilities |

## Main user-facing entrypoints

- `eeg-pipeline` for the Python CLI
- `eeg_pipeline/cli/tui/` for the optional Bubble Tea interface
- `eeg_pipeline/utils/config/eeg_config.yaml` for the main YAML configuration root
- `fmri_pipeline/utils/config/fmri_config.yaml` for fMRI-specific defaults

## Documentation ownership

Long-form guidance lives in `docs/`.
Package `README.md` files point back to these pages so there is only one canonical place
to maintain public user-facing guidance.
