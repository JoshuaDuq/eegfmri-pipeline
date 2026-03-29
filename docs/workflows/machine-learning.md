# Machine Learning Workflow

Use this workflow for predictive modeling on trialwise feature tables after preprocessing
and feature extraction are complete.

## Typical Run Order

1. Confirm feature tables and targets are aligned.
2. Choose regression or classification mode.
3. Select the model family and cross-validation strategy.
4. Review subject-level metrics, importances, and uncertainty outputs.

## Main Command Surface

```bash
eeg-pipeline ml --help
```

## Inputs

- trialwise EEG feature tables
- target columns derived from events
- optional covariates
- public ML configuration sections

## Outputs

- fold metrics and subject-level aggregates
- model artifacts and selected hyperparameters
- importance outputs when enabled
- optional time-generalization results

## Reference

Full preprocessing chain, model families, metrics, permutation testing, and uncertainty:
[machine-learning reference](../reference/machine-learning.md)
