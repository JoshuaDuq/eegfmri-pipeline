# CLI Reference

The public CLI is exposed through the `eeg-pipeline` console script.

## Top-Level Commands

| Command | Purpose |
| --- | --- |
| `preprocessing` | EEG cleaning, ICA, epoching, and preprocessing statistics |
| `features` | Trialwise EEG feature extraction |
| `behavior` | Trial-table construction and statistical inference |
| `ml` | Predictive modeling on feature tables |
| `plotting` | Figure generation from existing derivatives |
| `fmri` | fMRIPrep-style preprocessing orchestration |
| `fmri-analysis` | First-level, second-level, trialwise, and resting-state analysis |
| `validate` | Dataset and configuration validation |
| `stats` | Repository and derivative summary commands |
| `info` | Subject and dataset discovery helpers |

## Common Usage Pattern

```bash
eeg-pipeline <command> --help
```

Examples:

```bash
eeg-pipeline preprocessing --help
eeg-pipeline features compute --help
eeg-pipeline behavior --help
eeg-pipeline ml --help
eeg-pipeline fmri-analysis --help
```

## Shared Conventions

- subject selection is explicit or config-driven
- configuration is loaded from YAML rather than scattered flags
- pipeline outputs are written under the configured derivatives root
- validation errors are expected to surface early instead of being hidden
