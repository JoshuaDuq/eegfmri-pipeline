# CLI Entrypoint Removal

As of February 22, 2026, the deprecated standalone utility wrappers have been
removed:

- `scripts/eeg_raw_to_bids.py`
- `scripts/merge_psychopy.py`

Use the primary CLI:

- `eeg-pipeline utilities raw-to-bids ...`
- `eeg-pipeline utilities merge-psychopy ...`

Rationale:

- single command surface for TUI/CLI consistency
- less duplicate argument wiring
- lower risk of behavior drift between entrypoints
