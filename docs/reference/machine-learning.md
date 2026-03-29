# Machine-Learning Reference

This is the technical reference for predictive modeling on public trialwise feature tables.
The primary statistical unit is the subject, and subject-level aggregation is the default
for headline metrics.

## Data Flow

1. load feature tables into a design matrix
2. align targets and optional covariates
3. fit fold-local preprocessing steps
4. train the selected estimator
5. aggregate metrics at the subject and run level

## Preprocessing Chain

Shared public preprocessing components include:

- replacement of non-finite values
- dropping unusable columns
- imputation with train-only statistics
- variance filtering
- optional feature selection
- scaling where model families require it
- optional PCA

## Supported Model Families

- ElasticNet
- Ridge
- Random Forest
- public classification pipelines in `classification.py`

## Evaluation

The public docs assume cross-validation-safe training with:

- subject-aware splits
- permutation testing where configured
- optional feature-importance outputs
- optional uncertainty estimation

## Outputs

Expected derivative families:

- fold metrics and summary tables
- fitted-model artifacts
- feature-importance outputs
- time-generalization outputs when enabled
