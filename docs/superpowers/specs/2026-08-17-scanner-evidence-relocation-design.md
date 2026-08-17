# Relocating the scanner evidence out of core

**Date:** 2026-08-17
**Status:** design, awaiting implementation plan

## Purpose

`eeg_pipeline/` is the shared MNE preprocessing pipeline. Today it renders residual
gradient artifact, Analyzer correction quality and volume-locked residual as first-class
report sections, and carries scanner columns in the cohort sidecar. All of it is true only
of EEG recorded inside an MR scanner.

This moves that evidence to `studies/pain_study/`, where it survives as analysis producing
plots and TSVs rather than as MNE report sections. Nothing is thrown away; the pipeline
stops knowing about scanners.

### The rule change

[`tests/architecture/test_paradigm_code_stays_in_studies.py`](../../../tests/architecture/test_paradigm_code_stays_in_studies.py)
opens with:

> `eeg_pipeline/` holds what is true of any EEG-fMRI study. This checks it stayed that way.

That becomes **"any EEG study."** Every change below follows from that one edit, and that
test file is where the new contract is written down.

The precedent is the same file's existing `RELOCATED` list: `eeg_pipeline/preprocessing/bcg`
and `eeg_pipeline/preprocessing/residual_gradient.py` already made this trip, for the same
reason. This extends the move rather than inventing a boundary.

## Scope decisions

Four questions were settled before this was written. Recorded here because each one
changes what an implementer should do when they hit an ambiguous case.

| Decision | Answer |
|---|---|
| How far does "no fMRI references" reach? | Everything scanner-derived — not only the gradient tables, but the Analyzer section, the in/out-of-scanner strata, and the scanner metric family. |
| Delete, or relocate? | **Relocate.** Every measurement survives under `studies/pain_study/`, rendered as plots and TSVs instead of report sections. |
| Do the volume-rate-reading measurement helpers go too? | **Yes, all three** — `gradient_windows`, `gradient_marks_hz`, continuity volume gaps. Accepted consequence: aperiodic exponents change. |
| Does ICA cardiac review go? | **No, it stays in core.** ECG artifact is in every EEG recording; ECG-based component review is ordinary practice, not a scanner capability. |

## What moves

The target shape is the one `line_comb` and `cardiac_gaps` already use: `analysis/` holds
the measurement, `scripts/` decides which files to read and draws the output, `cli/`
registers the command.

### To `studies/pain_study/analysis/gradient/` (new)

The measurement halves, with all MNE-report rendering stripped:

| Source | What survives the move |
|---|---|
| `eeg_pipeline/preprocessing/report/scanner.py` (1,156 ln) | `measure_volume_timing`, `compute_comb_residual`, `compute_volume_locked_average`, `_locked_average_rms_uv`, the `CombResidual` / `VolumeLockedAverage` / `VolumeTiming` / `CombNotMeasured` dataclasses |
| `eeg_pipeline/preprocessing/report/cohort/gradient.py` (501 ln) | `participant_comb`, `cohort_comb`, `comb_attenuation`, `comb_audit`, the `CohortComb` dataclass |

Dropped in transit, because the study side draws plots: `scanner_residual_html`,
`_comb_table`, `_locked_table`, `_declined_table`, `_COMB_INTRO`, `_COMB_NOTE`,
`volume_locked_note_html`, `timing_table`, `attenuation_table`,
`add_scanner_residual_section`, `add_gradient_section`. The prose in those functions is
worth preserving as the new README's explanatory text rather than discarding — it is the
only written account of what the measurements mean.

### To `studies/pain_study/analysis/bcg/` (exists)

| Source | Note |
|---|---|
| `eeg_pipeline/preprocessing/report/analyzer_qc.py` (1,423 ln) | Marker QC, R-locked amplitude, cardiac attenuation, marker agreement, RR intervals |
| `eeg_pipeline/preprocessing/report/cohort/analyzer.py` (693 ln) | Cohort roll-up: Analyzer markers vs. ECG-detected R peaks |
| `eeg_pipeline/preprocessing/report/cohort_qc.py` (243 ln) | Cohort roll-up of Analyzer correction quality |
| `eeg_pipeline/preprocessing/pulse_artifact_qc.py` | Analyzer pulse markers; owns `PULSE_MARKER_DESCRIPTION` |
| `eeg_pipeline/preprocessing/cardiac_artifact_qc.py` | BCG attenuation |

**Judgement call to confirm at implementation:** `add_rr_interval_section` lives in
`analyzer_qc.py` and draws beat-to-beat intervals, which is ECG physiology rather than
scanner physics. It moves with its module because its purpose in the report is to say
whether Analyzer's marker train can be believed — it exists to validate a correction that
is leaving. If a reviewer wants RR plausibility retained for non-scanner datasets, it
should be re-derived inside the ICA cardiac review that stays, not held back here.

### To `studies/pain_study/scripts/gradient/` (new)

| Source | Note |
|---|---|
| `eeg_pipeline/preprocessing/pipeline/scanner_harmonic_qc.py` | Already an I/O-and-outputs stage |
| `eeg_pipeline/plotting/scanner_harmonic_comb.py` | Already a plot |

New files, following `scripts/line_comb/`:

- `plot.py` — the three figures: `plot_comb_residual`, `plot_volume_locked_average`,
  `plot_cohort_comb`, written as PNG at the module's existing DPI convention.
- `config.yaml` — the four settings leaving core (below), plus `volume_marker_description`.
- `README.md` — what the measurements are and what they are not, carrying over the prose
  from the deleted table notes.
- `__init__.py`.

### CLI

`studies/pain_study/cli/gradient.py` with `setup_gradient` / `run_gradient`, registered in
`studies/pain_study/cli/command_registry.py`, giving `eeg-pipeline gradient <mode>`
alongside `line-comb` and `cardiac-gaps`.

## What is deleted outright in core

These are report plumbing with no study equivalent — the study side draws plots, so the
`grid_table` callers, `add_*_section` functions and MNE wiring go rather than move.

**Report structure**

- `report/organize.py` — `"Residual scanner gradient"` and
  `"Scanner artifact correction (Analyzer)"` out of `SECTION_ORDER`.
- `report/run_evidence.py` — `has_scanner_evidence`, `gradient_marks_hz`,
  `MARKED_GRADIENT_HARMONICS`, the `add_scanner_residual_section` /
  `add_marker_agreement_section` / `add_rr_interval_section` calls, and the
  `gradient_fundamental_hz` argument passed to `compute_run_spectra`.
- `report/cohort/report.py` — the `add_gradient_section` and `add_analyzer_section` calls,
  and the `comb` entry in `_audit_tables` (which writes `*_comb_qc.tsv`).

**Spectra**

- `report/spectra.py` — `gradient_windows`, `_HARMONIC_SKIRT_BINS`, the
  `gradient_fundamental_hz` parameter of `compute_run_spectra`, and the gradient clause in
  the figure caption at line 444.
- `report/cohort/spectra.py` — `MARKED_GRADIENT_HARMONICS`, the in-scanner/out-of-scanner
  linestyle split and legend, and the two-branch aperiodic interpretation prose.

**Cohort record and sidecar** *(schema change)*

- `report/cohort/sidecar.py` — `SCANNER_RUN_COLUMNS` (11 columns) and `AcquisitionContext`.
- `report/cohort/record.py` — `_acquisition_context` and the in-scanner row block.
- `report/cohort/multiplicity.py` — `SCANNER_FAMILY` and its two `MetricSource` entries;
  `FAMILY_ORDER` drops from four families to three.
- `report/cohort/composition.py` — the In scanner / Outside scanner strata.

**Summary and configuration**

- `report/at_a_glance.py` — the three headline rows pointing at the Analyzer section
  (`worst_marker_agreement`, `worst_marker_agreement_lag_ms`,
  `worst_marker_agreement_lag_iqr_ms`).
- `report/provenance.py` — five config rows: volume marker annotation, pulse marker
  annotation, gradient comb range, comb Welch window, TR tolerance.
- `report/settings.py` — `volume_marker_description`, `comb_frequency_range_hz`,
  `comb_welch_seconds`, `repetition_time_tolerance_s`, and the `VOLUME_MARKER_DESCRIPTION`
  import. `DEFAULT_REPETITION_TIME_TOLERANCE_S` goes with them.
- `report/continuity.py` — `_volume_gaps`, `VOLUME_GAP_FACTOR`, `has_volume_markers`,
  `volume_gaps`, `volume_description`, and the volume-marker rug.
- `report/filtering.py` — the gradient-comb clauses in the notch prose (lines 50, 274).

**Pipeline**

- `pipelines/preprocessing.py` — `STEP_SCANNER_HARMONIC_QC`, `STEP_PULSE_MARKER_QC`,
  `STEP_CARDIAC_ATTENUATION_QC`, their insertion in `_get_steps_for_mode`,
  `_run_scanner_harmonic_qc`, and `_validate_eeg_fmri_declaration`.

## What is left inert

Per the settled blast radius, the config machinery stays and simply stops being read:

- `preprocessing.eeg_fmri` in `eeg_config.yaml`
- `eeg_pipeline/utils/config/acquisition.py` (`is_eeg_fmri`)
- the coherence rules in `utils/config/coherence.py` that reference it

Removing these touches config loading and every shipped preset, and would break study
configs that set the key. It is a separate change. The one exception is
`_validate_eeg_fmri_declaration`, deleted above because it validates against an ECG channel
for stages that no longer exist.

## Two rewires

Both are consequences of `ica_cardiac_review.py` staying in core while the module it
imports from leaves.

**1. `PULSE_MARKER_DESCRIPTION`.** `ica_cardiac_review.py:12` imports it from
`pulse_artifact_qc.py`, which moves to the study — and core must not import from
`studies/` (enforced by `test_core_does_not_import_the_study`). The key already exists as
config: `report.acquisition.pulse_marker_description`, registered in
`utils/config/loader.py:56` and `report/settings.py:325`. The rewire is to read it from
config, with the literal default `"Pulse Artifact/R"` given a home in core — the natural
one is `report/settings.py`, which already holds the field.

**2. The `eeg_fmri` gate.** `pipelines/preprocessing.py:1140` skips the ICA cardiac review
unless `_is_eeg_fmri()`. With scanner acquisition no longer a concept in core, that gate
goes; the review runs whenever `ica.cardiac_review.enabled` is set and an ECG channel is
present. The absence of an ECG channel becomes the condition it skips on, logged in the
same way for the same reason.

## Consequences accepted

1. **Aperiodic exponents change.** Deleting `gradient_windows` means the comb harmonics
   are no longer withheld from the aperiodic fit, so the fitted line is tilted by the comb.
   Every exponent in the subject and cohort reports, and every `aperiodic_*` sidecar value,
   shifts — and will not be comparable to anything already computed. This was chosen
   deliberately over keeping an internal volume-rate read.
2. **The cohort sidecar loses 11 columns**, requiring a schema version bump. Cohort reports
   cannot read sidecars written before this change, and vice versa.
3. **The study reads plots, not report sections.** Every measurement survives, as PNG and
   TSV under the study tree. BCG, decomb and marker-recovery work continues; it stops
   arriving inside the MNE HTML report.

## Testing

The architecture test is the specification. `RELOCATED` gains the moved core paths,
`RELOCATED_TO` gains `studies/pain_study/analysis/gradient` and
`studies/pain_study/scripts/gradient`, and the docstring records the new rule.

A new assertion should check that core Python names no scanner concept, in the same spirit
as `PARADIGM_MARKERS`. **The markers must be compound terms, not the bare word
`gradient`:** `volume_locked`, `repetition_time_s`, `volume_marker`, `Pulse Artifact/R`,
`scanner gradient`, `brainvision_analyzer`. Core legitimately contains `np.gradient`
(`analysis/features/quality.py`, `analysis/features/spectral.py`),
`GradientBoostingRegressor` (`analysis/machine_learning/uncertainty.py`) and
`cnn_gradient_clip_norm` (`analysis/machine_learning/cnn.py`) — five files that a naive
substring check would fail on, none of which has anything to do with a scanner.

Two test files move with their modules and lose their report-rendering cases:
`tests/preprocessing/test_report_scanner.py` (736 ln) and
`tests/preprocessing/report/test_cohort_gradient.py` (429 ln). Alongside them,
`test_report_analyzer_qc.py`, `test_cohort_analyzer.py`, `test_pulse_artifact_qc.py`,
`test_cardiac_artifact_qc.py`, `test_report_marker_agreement.py`,
`test_report_rr_intervals.py` and `test_scanner_harmonic_qc.py` follow their subjects to
`studies/tests/`.

Edited in place: `test_cohort_record.py`, `test_cohort_sidecar.py`, `test_cohort_report.py`,
`test_cohort_multiplicity.py`, `test_cohort_composition.py`, `test_cohort_spectra.py`,
`test_cohort_at_a_glance.py`, `test_cohort_end_to_end.py`, `test_report_modes.py`,
`test_report_organize.py`, `test_report_spectra.py`, `test_report_continuity.py`,
`test_report_run_evidence.py`, `test_report_at_a_glance.py`, `test_report_settings_*.py`,
`test_ica_cardiac_promotion_wiring.py`.

Per the project's standing constraint, verification runs targeted subsets rather than the
full suite.

## Sequencing

One rule change, but roughly 4,000 lines relocating across ~20 core modules and ~45 test
files. It is one plan, in four phases, each leaving the suite green:

1. **Gradient.** `report/scanner.py` and `report/cohort/gradient.py` out; the new
   `analysis/gradient/`, `scripts/gradient/` and CLI command in. Report structure, spectra
   and continuity edits land here.
2. **Analyzer and cardiac.** `analyzer_qc.py`, `cohort/analyzer.py`, `cohort_qc.py`,
   `pulse_artifact_qc.py`, `cardiac_artifact_qc.py` out; the two rewires land here, and the
   ICA cardiac review must still run at the end of it.
3. **Cohort record and sidecar.** The schema change, in its own phase because it is the one
   step that invalidates existing derivatives.
4. **Architecture test and pipeline steps.** The new contract written down, and the STEP
   constants removed once nothing dispatches to them.

Phase 1 alone changes aperiodic exponents. That is worth confirming against a real subject
before phase 2 starts, rather than discovering it at the end.

## Out of scope

- Removing the `eeg_fmri` config key and its coherence rules (left inert, above).
- `studies/pain_study/analysis/line_comb`, `scripts/line_comb`, `gradient_trough_ica`,
  `scanner_contamination.py` and the decomb manifest reader — already study-side, or
  already correctly placed.
- Regenerating derivatives. Existing reports and sidecars are not migrated; the schema bump
  means they are read by the version that wrote them.
