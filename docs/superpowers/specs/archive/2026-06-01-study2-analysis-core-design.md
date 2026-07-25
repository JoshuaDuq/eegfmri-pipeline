# Study 2 Analysis Core Design

## Scope

This pass finishes the Study 2 analysis-core modules that can be validated from
arrays and tables without requiring the KINGSTON drive data layout. The work does
not claim that end-to-end sLORETA extraction, FreeSurfer/BEM/trans discovery, or
Study 1 target-retrained model refits are complete.

## Current State

Study 2 already contains tested modules for:

- Study 1 entry gates
- frozen linear contribution scores
- source-model quality control
- source-stage subject and cohort quality control
- subject-level source-power association maps

The explicit implementation-status module still marks several protocol pieces as
unimplemented. Most of those are statistical interpretation layers that can be
implemented independently of the heavy MNE/MRI source-extraction stage.

## Architecture

Add focused modules under `studies/pain_study/study2/`, each with one public
calculation boundary and strict input validation:

- `source_inference.py` computes group one-sample source inference from
  subject-level maps, explicit vertex adjacency, and empirical null maps.
- `directional_consistency.py` evaluates spatial agreement between
  prediction-associated and true-target maps.
- `artifact_controls.py` evaluates artifact-template thresholds,
  artifact-expression p-values, robustness summaries, and gamma interpretation
  status.
- `spatial_comparison.py` compares EEG source maps with fMRI covariance maps
  against supplied spatial surrogate maps.
- `behavioral_convergence.py` computes nuisance-adjusted within-subject
  behavioral convergence slopes and circular-shift permutation summaries.
- `reporting.py` provides deterministic bootstrap interval summaries for Study 2
  tabular outputs.

No module will silently impute missing values. Shape mismatches, non-finite
values, rank-deficient designs, invalid p-values, empty masks, and impossible
permutation structures raise explicit errors.

## Data Flow

1. Existing Study 1 and source-stage code produces subject-level maps and QC.
2. `source_inference.py` aggregates eligible subject maps per band and tests
   observed cluster mass against supplied null maps.
3. `directional_consistency.py` and `artifact_controls.py` decide whether maps
   satisfy interpretation gates.
4. `spatial_comparison.py` evaluates coarse EEG-fMRI spatial correspondence
   using externally supplied surrogate maps.
5. `behavioral_convergence.py` tests whether source-pattern expression tracks
   behavioral ratings after nuisance residualization.
6. `reporting.py` produces bootstrap intervals for reported scalar summaries.
7. `implementation_status.py` is updated only for components genuinely covered
   by these modules and tests.

## Deliberate Non-Scope

This pass does not implement:

- reading KINGSTON drive data
- MNE sLORETA source-power extraction
- FreeSurfer/BEM/trans discovery
- fsaverage morphing
- source point-spread estimation from inverse operators
- refitting the full Study 1 ElasticNet pipeline for each target-retrained draw

Those remain an end-to-end execution layer that should be implemented after the
analysis core is stable and after the external data layout is inspected.

## Testing

Each new module gets focused pytest coverage using synthetic arrays/tables:

- valid calculations return expected statistics and statuses
- invalid inputs fail fast
- permutation p-values use the configured empirical plus-one correction
- Holm correction is applied where protocol-defined
- gamma artifact survival changes interpretation status without hiding results
- implementation status no longer lists the completed analysis-core components

The final verification target is the Study 2 pipeline/config test subset plus
Ruff on the touched Study 2 files.
