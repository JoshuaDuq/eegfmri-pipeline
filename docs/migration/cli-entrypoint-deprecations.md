# CLI Entrypoint Removal

As of February 22, 2026, the deprecated standalone utility wrappers have been
removed:

- `scripts/eeg_raw_to_bids.py`
- `scripts/merge_psychopy.py`

Current approach:

- Run raw-to-BIDS conversion and event merging in external dataset-specific tooling.
- Run this repository's CLI from BIDS onward (`preprocessing`, `features`, `behavior`,
  `ml`, `plotting`, `fmri preprocess`, `fmri-analysis`).

Rationale:

- single command surface for TUI/CLI consistency
- less duplicate argument wiring
- lower risk of behavior drift between entrypoints
