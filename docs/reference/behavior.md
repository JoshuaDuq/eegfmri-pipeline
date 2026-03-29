# Behavioral Analysis Reference

This is the technical reference for the public behavioral statistics pipeline.
It runs as a dependency-resolved DAG over trial tables aligned with EEG features.

## Pipeline Shape

Canonical stage flow:

1. load aligned behavior and feature inputs
2. build the trial table
3. optionally residualize the predictor
4. run correlations, regression, or condition comparisons
5. run temporal inference when requested
6. summarize multiple-comparison control and exports

## Statistical Safeguards

The public behavior pipeline treats the subject as the scientific unit and respects
within-subject clustering.

- grouped permutation is preferred when trials are not exchangeable
- run adjustment is treated as part of the design, not an optional heuristic
- predictor-type validation happens at entrypoints
- invalid analysis combinations raise instead of silently degrading

## Core Contracts

- clean events must contain `trial_id`
- trialwise feature tables must contain `trial_id`
- outcome, predictor, and condition columns must exist before analysis begins

## Outputs

Expected derivative families:

- canonical trial tables
- correlation, regression, and condition-comparison outputs
- temporal statistics and cluster outputs
- summary tables and reports for downstream interpretation
