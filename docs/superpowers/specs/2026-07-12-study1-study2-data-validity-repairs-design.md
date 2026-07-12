# Study 1 and Study 2 Data-Validity Repairs

## Objective

Correct implementation defects that can change trial alignment, feature values, model predictions,
permutation nulls, source estimates, or reported inferential statistics. The work must not turn
confirmatory thresholds, cohort targets, or interpretive diagnostics into hard runtime gates.

## Scope

The repair covers confirmed code defects from the Study 1 and Study 2 audit:

1. use the shortest circular displacement when enforcing the minimum within-run shift;
2. sample one coherent within-run permutation assignment per draw and reuse it across outer folds;
3. standardize the Study 1 event contract on `run_id` at the event boundary and `run` in output
   tables, with the duplicated tests and documentation matching that contract;
4. expose the held-out EEG residual prediction separately from the nuisance-added raw-target
   prediction and use the residual prediction as the Study 2 score;
5. compute and standardize band contribution scores in the Study 2 input builder, aggregating the
   three scanner-clean gamma sub-bands into one gamma contribution;
6. compute point-spread summaries from resolution-matrix columns under the MNE convention;
7. validate spatial-surrogate counts and provenance, generate BrainSMASH surrogates through an
   explicit stage, and Holm-correct the three spatial-comparison p-values;
8. compute non-blocking within-subject prediction and matched temporal-specificity diagnostics;
9. correct behavioral-convergence permutation eligibility and trial ordering;
10. make incomplete artifact-control families evaluate as unmet advisory criteria instead of
    silently passing;
11. label below-target source cohorts as feasibility analyses in outputs without preventing
    computation;
12. reconcile the full test suite and protocol documentation with the corrected behavior.

The repair does not rerun or write production data on `/Volumes/KINGSTON`. It does not add hard
Study 1-to-Study 2 entry gates, abort source inference below 30 participants, or require optional
interpretive diagnostics before ordinary pipeline execution.

## Architecture

### Permutation assignments

Introduce a small immutable representation of subject-run circular-shift assignments. A draw is
sampled once from the original group, run, and trial-index arrays. Fold-specific nuisance residuals
remain fold-contained, but each fold applies the same preselected assignment to any rows it uses.
This preserves reduced-model residual learning while defining one coherent dataset relabeling per
null draw.

The admissible-shift helper measures `min(forward_distance, block_length - forward_distance)`.
Censored runs continue to use original trial labels and retain the existing minimum-trial and
minimum-admissible-shift rules.

### Prediction decomposition

Model-comparison prediction returns a structured result containing the raw evaluation target, full
prediction, nuisance prediction, and EEG residual prediction. Existing Study 1 statistics continue
to use the full prediction. Study 2 consumes only the held-out EEG residual prediction, eliminating
the current second residualization approximation.

### Spectral contributions

Band membership is defined by configuration rather than exact string equality. Alpha and beta map
to their corresponding feature labels; gamma maps to `gamma_low_clean`, `gamma_mid_clean`, and
`gamma_high_clean`. The Study 2 builder writes the combined and per-band standardized scores needed
by both primary and secondary source stages.

The existing broad beta feature remains available. Scanner-harmonic-safe beta is added as a
non-blocking sensitivity preset so existing analyses remain runnable while the corrected sensitivity
can quantify dependence on the approximately 20 Hz scanner component.

### Source and spatial inference

Point-spread FWHM iterates over resolution-matrix columns and records the matrix convention in the
output metadata. Spatial correspondence validates the configured surrogate count and hemisphere
metadata. A dedicated BrainSMASH preparation stage produces deterministic, seeded surrogate arrays;
missing BrainSMASH support raises a direct dependency error when that stage is invoked. The summary
contains raw and Holm-adjusted p-values.

Source inference remains executable for statistically valid small cohorts. Its summary records the
configured confirmatory and feasibility thresholds and labels the result `confirmatory`,
`feasibility`, or `below_feasibility`; these labels do not block computation.

### Advisory diagnostics

Within-subject-centered incremental prediction is calculated from held-out predictions after
centering observed, nuisance, and full predictions within subject. Temporal specificity is reported
as a matched primary-minus-control effect rather than inferred solely from non-significance of a
control. Artifact-control summaries enumerate required artifact metrics and mark missing metrics as
unmet advisory criteria without raising during ordinary analysis.

Behavioral convergence orders trials by the configured within-run trial index and counts only runs
that satisfy the same circular-shift eligibility rules as the prediction analyses.

## Error Handling

Malformed identifiers, non-finite inputs, impossible shift assignments, mismatched feature names,
incorrect matrix orientation metadata, and invalid surrogate shapes raise explicit errors at their
entry points. Scientific thresholds produce status fields, not exceptions, unless the underlying
statistic is mathematically undefined.

## Testing

Every behavior change follows red-green-refactor:

- unit tests for circular distance and coherent cross-fold assignments;
- model tests that distinguish residual from nuisance-added predictions;
- contribution tests for clean-gamma aggregation and builder output columns;
- asymmetric resolution-matrix tests that distinguish PSF columns from CTF rows;
- spatial tests for surrogate validation and Holm adjustment;
- reporting tests for centered and matched temporal diagnostics;
- behavioral tests for ordering and eligible-run counts;
- artifact tests for incomplete metric families;
- integration tests for the `run_id` event contract and the Study 2 source-stage input;
- the complete Study 1/2 test suite, architecture checks, and Ruff on modified modules.

## Success Criteria

The repaired code passes the complete repository test suite relevant to Studies 1 and 2, the new
tests fail against the original implementation and pass against the corrected implementation, and
no production datasets or existing production result roots are modified.
