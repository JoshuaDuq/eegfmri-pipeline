# Behavioral Analysis Workflow

Use this workflow when you need trial-level statistics that combine behavioral variables
with EEG-derived feature tables.

## Typical Run Order

1. Verify clean events and feature tables share the same `trial_id` space.
2. Build the canonical trial table.
3. Run only the required computations: correlations, regression, condition tests, temporal inference, or exports.
4. Review multiple-comparison summaries and group-level outputs before interpretation.

## Main Command Surface

```bash
eeg-pipeline behavior --help
```

## Inputs

- aligned events tables
- trialwise EEG feature tables
- predictor, target, condition, and optional covariate columns

## Outputs

- trial tables
- effect-size and p-value tables
- regression and condition-comparison results
- temporal inference outputs and reports

## Reference

Full DAG, safeguards, correction strategy, and stage details:
[behavior reference](../reference/behavior.md)
