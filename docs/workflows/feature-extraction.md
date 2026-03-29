# Feature Extraction Workflow

Use this workflow after EEG preprocessing has produced clean epochs and clean event tables.

## Typical Run Order

1. Confirm cleaned epochs exist for the target subjects.
2. Choose feature families, bands, time windows, and spatial aggregation.
3. Run feature extraction.
4. Inspect feature tables and QC metadata before launching behavior or ML analyses.

## Main Command Surface

```bash
eeg-pipeline features --help
eeg-pipeline features compute --help
```

## Inputs

- cleaned epochs
- clean events with `trial_id`
- configuration for feature categories, bands, windows, and transforms

## Outputs

- parquet feature tables under subject derivatives
- metadata describing selected bands, windows, and QC summaries
- optional exports for downstream plotting or modeling

## Reference

Full methods, feature families, normalization rules, PAC handling, and output contracts:
[feature reference](../reference/features.md)
