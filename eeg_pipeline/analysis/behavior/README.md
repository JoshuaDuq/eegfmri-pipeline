# Behavioral Statistics Pipeline

`eeg_pipeline.analysis.behavior` contains the subject-level EEG/behavior analysis
pipeline and its group-level correlation aggregation.

## Active stage surface

The current behavior DAG includes:

- `load`
- `metadata`
- `trial_table`
- `lag_features`
- `predictor_residual`
- `correlate_design`
- `correlate_effect_sizes`
- `correlate_pvalues`
- `correlate_primary_selection`
- `correlate_fdr`
- `correlations`
- `regression`
- `condition_column`
- `condition_multigroup`
- `temporal_tfr`
- `temporal_stats`
- `cluster`
- `icc`
- `hierarchical_fdr_summary`
- `report`
- `export`

Removed computations such as `feature_qc`, `condition_window`, `predictor_models`,
`models`, `stability`, `consistency`, `influence`, `predictor_sensitivity`,
`mediation`, `moderation`, and subject-level `mixed_effects` are not part of the
pipeline anymore.

## Main modules

- `orchestration.py`: stage wrappers and shared helpers
- `stage_catalog.py`: canonical stage definitions
- `stage_registry.py`: dependency registry and dry-run support
- `stage_execution.py`: stage execution engine
- `stage_runners.py`: stage dispatch wiring
- `group_level.py`: multilevel correlation aggregation across subjects
- `stages/trial_table.py`: trial-table construction and residualization
- `stages/correlate.py`: correlation stages
- `stages/models.py`: regression stage
- `stages/condition.py`: condition comparisons
- `stages/temporal.py`: temporal and cluster analyses
- `stages/diagnostics.py`: ICC reliability
- `stages/fdr.py`: hierarchical FDR summary
- `stages/report.py`: reporting
- `stages/export.py`: export/materialization

## Notes

- Trial-level inference is guarded against invalid i.i.d. assumptions.
- Group-level analysis currently supports multilevel correlations only.
- The CLI and TUI should only expose the active stage surface above.
