# fMRI Reference

This is the technical reference for the public fMRI pipeline.
It covers preprocessing, first-level GLM analysis, second-level inference, trialwise beta
estimation, resting-state analysis, and reporting.

## Public Pipeline Stages

1. BIDS validation and run discovery
2. fMRIPrep preprocessing
3. first-level GLM fitting and contrast construction
4. second-level group analysis
5. trialwise beta estimation and signature readout
6. resting-state ROI analysis
7. QC and reporting

## Inputs

Minimum public inputs:

- BOLD images in BIDS layout
- matching `events.tsv`
- derivative confounds and masks for modeled runs
- anatomy and FreeSurfer resources when source-space or surface steps require them

## GLM Assumptions

The public GLM layer expects valid event timing, consistent BOLD-events pairing,
and configuration-driven contrast definitions. Run exclusion and mask construction
are explicit and logged.

## Resting-State Analysis

Resting-state paths build ROI time series, compute connectivity summaries, and aggregate
run-level results with deterministic transforms rather than best-effort heuristics.

## Related Pages

- operational guide: [fMRI workflow](../workflows/fmri.md)
- raw-data preparation: [raw-to-BIDS](../fmri/raw-to-bids.md)
