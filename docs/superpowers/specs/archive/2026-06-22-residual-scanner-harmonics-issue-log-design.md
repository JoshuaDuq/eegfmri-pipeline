# Residual Scanner Harmonics Issue Log Design

## Purpose

Add one cohort-wide entry to `studies/pain_study/STUDY_ISSUES_README.md` that records
residual scanner-frequency structure in simultaneous EEG-fMRI data without making an
unvalidated preprocessing or participant-exclusion decision.

## Scope

The entry covers numbered participants `sub-0000`, `sub-0001`, `sub-0003`, `sub-0004`,
`sub-0005`, `sub-0007`, `sub-0008`, and `sub-0009`. Pilots and excluded `sub-0002` are
outside this entry. Task data are assessed for every available final-clean continuous
run: five runs for `sub-0003` and six runs for each other participant.

## Evidence

Document two processing boundaries:

1. BrainVision Analyzer `_scannerpulse_corrected` exports, establishing that residual
   structure is present before the Python preprocessing pipeline.
2. Final `*_proc-clean_raw.fif` derivatives, establishing that it remains in the data
   entering Study 1 and Study 2 feature generation.

For each run, estimate Welch PSD with 16.384-second segments and 50% segment overlap.
For final-clean derivatives, summarize the median spectrum across all EEG channels. For
Analyzer exports, use the fixed representative channel set Fp1, F3, C3, O1, Fz, Cz, Pz,
Oz, POz, FC3, PO3, and PO7. Within the fixed frequency regions 18-23, 38-43, 56-66, and
77-85 Hz, record the most prominent local peak. Participant summaries report the median
and maximum peak prominence across runs in decibels. The issue entry must define this
calculation so the table is reproducible and must not describe these values as neural
oscillations.

## Entry Structure

Use the repository's existing issue fields:

- `Study/stage`: simultaneous EEG-fMRI preprocessing and Study 1/Study 2 spectral inputs.
- `Issue`: shared acquisition/correction context and participant-level evidence table.
- `Decision`: unresolved cohort-wide issue; preserve current raw data, template, and
  derivatives; do not treat current beta/gamma features as fully scanner-controlled.
- `Follow-up`: perform the outcome-blind correction benchmark, define estimator-specific
  scanner-residual handling, establish QC acceptance criteria, and reprocess the cohort
  only if a canonical method is validated.

The table contains one row per participant with available run count and final-clean
median/maximum prominence for each fixed frequency region. A short boundary summary
states that the same pattern is already present in the Analyzer exports.

## Scientific Interpretation

The issue is not an automatic subject exclusion. It is a hold on confirmatory
interpretation of scanner-sensitive beta/gamma measures until correction and residual-QC
criteria are validated without reference to pain, fMRI targets, behavior, or model
performance. Alpha and other unaffected analyses remain subject to their existing QC
rules.

## Verification

After editing, verify that:

- all eight numbered participants appear exactly once in the table;
- run counts match available final-clean derivatives;
- reported values match the measured participant summaries;
- existing issue entries and uncommitted user edits are unchanged;
- no language claims that the proposed Analyzer resampling experiment is already a fix.
