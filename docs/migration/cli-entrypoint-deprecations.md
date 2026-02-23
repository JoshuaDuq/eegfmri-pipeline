# CLI Entrypoint Removal

As of February 22, 2026, the deprecated standalone utility wrappers have been
removed:

- `scripts/eeg_raw_to_bids.py`
- `scripts/merge_psychopy.py`

Use the paradigm-specific CLI script:

- `python paradigm-specific-scripts/run_paradigm_specific.py eeg-raw-to-bids ...`
- `python paradigm-specific-scripts/run_paradigm_specific.py merge-psychopy ...`

Rationale:

- single command surface for TUI/CLI consistency
- less duplicate argument wiring
- lower risk of behavior drift between entrypoints
