# Relocating the scanner evidence out of core

**Date:** 2026-08-17
**Status:** design, awaiting implementation plan

## Purpose

`eeg_pipeline/` is the shared MNE preprocessing pipeline. Today it renders residual
gradient artifact, Analyzer correction quality and volume-locked residual as first-class
report sections, carries scanner columns in the cohort sidecar, and ships a config whose
scanner keys only mean anything inside a bore. All of it is true only of EEG recorded
inside an MR scanner.

This moves that evidence, and its configuration, to `studies/pain_study/`, where it
survives as analysis producing plots and TSVs rather than as MNE report sections. Nothing
is thrown away; the pipeline stops knowing about scanners.

### The rule

Everything relating to fMRI analysis — ballistocardiogram, gradient artifact, Analyzer
correction, volume timing — moves to the paradigm-specific folder. **ECG analysis stays**:
an ECG lead is ordinary EEG equipment, cardiac artifact is in every recording, and
ECG-based ICA component review is standard practice. The line is drawn at what the scanner
causes, not at what the heart does.

[`tests/architecture/test_paradigm_code_stays_in_studies.py`](../../../tests/architecture/test_paradigm_code_stays_in_studies.py)
opens with:

> `eeg_pipeline/` holds what is true of any EEG-fMRI study. This checks it stayed that way.

That becomes **"any EEG study."** Every change below follows from that one edit, and that
test file is where the new contract is written down.

The precedent is the same file's existing `RELOCATED` list: `eeg_pipeline/preprocessing/bcg`
and `eeg_pipeline/preprocessing/residual_gradient.py` already made this trip, for the same
reason. This extends the move rather than inventing a boundary.

## Scope decisions

Settled before this was written. Recorded because each one changes what an implementer
should do on hitting an ambiguous case.

| Decision | Answer |
|---|---|
| How far does the rule reach? | Everything scanner-derived — the gradient sections, the Analyzer section, the in/out-of-scanner strata, the scanner metric family. |
| Delete, or relocate? | **Relocate.** Every measurement survives under `studies/pain_study/`, rendered as plots and TSVs instead of report sections. |
| Do the volume-rate-reading measurement helpers go? | **Yes, all three** — `gradient_windows`, `gradient_marks_hz`, continuity volume gaps. Accepted consequence: aperiodic exponents change. |
| Does ECG analysis go? | **No.** ICA cardiac review, RR interval evidence and the ECG coupling QC all stay in core and stay in the report. |
| Does the ECG *marker* path go? | **No.** Any EEG study may carry beat markers. The path stays, genericized off Analyzer's marker name via new config. |
| Configuration? | **In scope.** The scanner keys leave the core config for study workflow configs, and the config machinery that reads them goes with them. |

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
| `eeg_pipeline/preprocessing/report/analyzer_qc.py` (1,423 ln) | Marker QC, R-locked amplitude, cardiac attenuation, marker agreement — **less** `add_rr_interval_section`, see below |
| `eeg_pipeline/preprocessing/report/cohort/analyzer.py` (693 ln) | Cohort roll-up: Analyzer markers vs. ECG-detected R peaks |
| `eeg_pipeline/preprocessing/report/cohort_qc.py` (243 ln) | Cohort roll-up of Analyzer correction quality |
| `eeg_pipeline/preprocessing/pulse_artifact_qc.py` | Analyzer pulse markers; owns `PULSE_MARKER_DESCRIPTION` |
| `eeg_pipeline/preprocessing/cardiac_artifact_qc.py` | BCG attenuation |

### To `studies/pain_study/scripts/gradient/` (new)

| Source | Note |
|---|---|
| `eeg_pipeline/preprocessing/pipeline/scanner_harmonic_qc.py` | Already an I/O-and-outputs stage |
| `eeg_pipeline/plotting/scanner_harmonic_comb.py` | Already a plot |

New files, following `scripts/line_comb/`: `plot.py` (the three figures —
`plot_comb_residual`, `plot_volume_locked_average`, `plot_cohort_comb` — written as PNG),
`config.yaml`, `README.md`, `__init__.py`.

### CLI

`studies/pain_study/cli/gradient.py` with `setup_gradient` / `run_gradient`, registered in
`studies/pain_study/cli/command_registry.py`, giving `eeg-pipeline gradient <mode>`
alongside `line-comb` and `cardiac-gaps`.

## What stays in core

Named explicitly, because the earlier draft had two of these leaving:

- **`ica_cardiac_review.py` and `ica_cardiac_report.py`** — ECG-based ICA component review,
  and the "ICA cardiac artifact review" report section.
- **`detect_ecg_events` and `_marker_beats`** — including the marker path. Any EEG study may
  carry beat markers; what is Analyzer-specific is the hardcoded marker *name*, not the
  ability to read one. Genericized rather than moved — see rewire 1.
- **`add_rr_interval_section`** — beat-to-beat interval evidence. It currently lives inside
  `analyzer_qc.py`, so it must be **extracted to its own module before that file moves**.
  A tachogram is ECG physiology; it reads no scanner quantity once it takes its beats from
  `detect_ecg_events` rather than from `analyzer_qc`'s own marker reading.
- **`preprocessing.clean_events_qc.ecg_coupling`** — correlates EEG against the lead.
  Depends on an ECG channel, never on a scanner.

## What is deleted outright in core

Report plumbing with no study equivalent — the study side draws plots, so the `grid_table`
callers, `add_*_section` functions and MNE wiring go rather than move.

**Report structure**

- `report/organize.py` — `"Residual scanner gradient"` and
  `"Scanner artifact correction (Analyzer)"` out of `SECTION_ORDER`.
- `report/run_evidence.py` — `has_scanner_evidence`, `gradient_marks_hz`,
  `MARKED_GRADIENT_HARMONICS`, the `add_scanner_residual_section` and
  `add_marker_agreement_section` calls, and the `gradient_fundamental_hz` argument passed
  to `compute_run_spectra`. The `add_rr_interval_section` call stays, re-pointed at the
  extracted module.
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

**Summary and report settings**

- `report/at_a_glance.py` — the three headline rows pointing at the Analyzer section.
- `report/provenance.py` — the five scanner config rows.
- `report/settings.py` — `volume_marker_description`, `pulse_marker_description`,
  `comb_frequency_range_hz`, `comb_welch_seconds`, `repetition_time_tolerance_s`,
  `bcg_residual_window_s`, `bcg_residual_baseline_s`, `bcg_residual_measurement_s`,
  `DEFAULT_REPETITION_TIME_TOLERANCE_S`, and the two marker-description imports.
- `report/continuity.py` — `_volume_gaps`, `VOLUME_GAP_FACTOR`, `has_volume_markers`,
  `volume_gaps`, `volume_description`, and the volume-marker rug.
- `report/filtering.py` — the gradient-comb clauses in the notch prose (lines 50, 274).

**Pipeline**

- `pipelines/preprocessing.py` — `STEP_SCANNER_HARMONIC_QC`, `STEP_PULSE_MARKER_QC`,
  `STEP_CARDIAC_ATTENUATION_QC`, their insertion in `_get_steps_for_mode`,
  `_run_scanner_harmonic_qc`, `_is_eeg_fmri`, `_validate_eeg_fmri_declaration`, and the
  `_is_eeg_fmri()` gate on the ICA cardiac review at line 1140.

## Configuration changes

The keys leave core for the workflow config of whichever study folder now owns the code
that reads them. `studies/pain_study/scripts/workflow_config.py` already defines the
resolution order: paths come from core, workflow settings sit next to their code.

### Out of `eeg_pipeline/utils/config/eeg_config.yaml`

| Key | Goes to |
|---|---|
| `paths.decomb_manifest` | `scripts/line_comb/config.yaml` |
| `preprocessing.eeg_fmri` | deleted — nothing gates on it once the stages are gone |
| `preprocessing.brainvision_analyzer.*` (10 keys, incl. `pulse_artifact_qc` and `cardiac_artifact_qc` blocks) | `scripts/bcg/config.yaml` (new) |
| `preprocessing.scanner_harmonic_qc.*` (5 keys) | `scripts/gradient/config.yaml` |
| `alignment.trim_to_volume_bounds` | `scripts/conversion/` — already exposed there as `--trim-to-volume-bounds` |
| `report.thresholds.min_r_markers_per_volume` | `scripts/bcg/config.yaml` |
| `report.thresholds.comb_frequency_range_hz`, `comb_welch_seconds`, `repetition_time_tolerance_s` | `scripts/gradient/config.yaml` |
| `report.analysis.bcg_residual_window_s`, `bcg_residual_baseline_s`, `bcg_residual_measurement_s` | `scripts/bcg/config.yaml` |
| `report.acquisition.volume_marker_description` | `scripts/gradient/config.yaml` |
| `report.acquisition.pulse_marker_description` | replaced by `ica.cardiac_review.marker_description` below; the study sets `"Pulse Artifact/R"` as an override |

### Into `eeg_config.yaml`: the ECG beat source

Two core consumers now need to know where beat times come from — the ICA cardiac review and
the extracted RR interval section — and today neither the marker name nor the preference
order is configurable. Added under `ica.cardiac_review`, beside the `ecg_channel` key that
already lives there:

```yaml
    # Where beat times come from.
    #   markers  read the annotation named below; fail if it is absent
    #   detect   run find_ecg_events on the ECG channel, ignoring any markers
    #   auto     prefer markers where present, fall back to detection
    beat_source: auto
    # Annotation carrying one mark per heartbeat, exactly as the recording spells it.
    # Null means the recording carries none, which is the ordinary case: a montage with
    # an ECG lead and no marker train detects from the channel.
    marker_description: null
```

`auto` reproduces today's behaviour, so nothing changes for a study that sets
`marker_description`. The pain study sets `"Pulse Artifact/R"` in its own override and keeps
the marker preference it depends on.

The block also corrects a comment that is already wrong: `eeg_config.yaml:505` says R peaks
"are detected from the ECG signal and do not require Analyzer R annotations," while
`detect_ecg_events` in fact prefers the marker train where one exists.

**Not consolidated here, but worth noting:** `eeg.ecg_channels` and
`ica.cardiac_review.ecg_channel` already name the same lead in two places. That predates this
change and merging them is its own piece of work.

Prose edits where a surviving key is justified by a scanner fact: the `notch_freq: null`
comment (line 311) and the line-comb note (line 327) both explain themselves by reference
to the decomb chain, and `report.display.spectra_marked_frequencies` (line 754) describes
gradient harmonics that are no longer added automatically.

### Out of the config machinery

- `eeg_pipeline/utils/config/acquisition.py` — deleted entirely (`is_eeg_fmri`).
- `utils/config/coherence.py` — `_check_scanner_settings` deleted, the `is_eeg_fmri` import
  and its call site with it. `_check_decomb_notch` moves to the line-comb workflow.
  **`_check_ecg_settings` stays** — it checks for a named ECG channel, which is the right
  requirement whether or not there was a scanner.
- `utils/config/loader.py` — `volume_marker_description` and `pulse_marker_description` out
  of `_NON_PATH_KEYS` (lines 55-56), and **`marker_description` added in their place**. That
  set exists to stop path-resolution from claiming values that merely look path-like, and
  `"Pulse Artifact/R"` contains a slash: renaming the key without moving its entry would
  send the loader looking for a file called `Pulse Artifact/R` under the project root.
- `utils/data/preprocessing.py` — `trim_to_volume_bounds`, and its `is_eeg_fmri` import.
- `cli/commands/preprocessing_overrides.py` — the `--trim-to-volume-bounds` override.

### `presets/eeg_only.yaml` is retired

This preset exists to switch scanner stages off: `eeg_fmri: false`,
`brainvision_analyzer.enabled: false`, `trim_to_volume_bounds: false`,
`decomb_manifest: null`. Every one of those keys is being deleted, because core no longer
has the stages they disable.

**Core becomes EEG-only by construction, so "EEG only" stops being a preset.** What is left
of the file is genuinely useful and should not be lost: the `task: null` deliberate-unset,
and the long explanatory block on naming an ECG channel and switching the two ECG stages
on. That prose moves into `eeg_config.yaml` beside the keys it describes. `presets/rest.yaml`
keeps its role — resting-state is still a real variant — with its `eeg_fmri` sentence
dropped.

### Study configs gained

- `studies/pain_study/scripts/gradient/config.yaml` — comb range, Welch window, TR
  tolerance, volume marker description, the `scanner_harmonic_qc` block.
- `studies/pain_study/scripts/bcg/config.yaml` (new folder) — the `brainvision_analyzer`
  block, `bcg_residual_*` windows, pulse marker description, `min_r_markers_per_volume`.
  Deliberately not folded into the existing `scripts/cardiac_gaps/`: that workflow *repairs*
  the beats Analyzer never marked and writes a corrected tree, while this one *measures*
  what the correction achieved. Sharing a config would tie a remediation step's settings to
  a QC step's, and the pairing with the existing `analysis/bcg/` is what the layout expects.
- `studies/pain_study/scripts/line_comb/config.yaml` — gains `decomb_manifest`.

## Two rewires

Both follow from ICA cardiac review staying while the modules it leans on leave.

**1. `ica_cardiac_review.py` reads its marker name from config.** It currently imports
`PULSE_MARKER_DESCRIPTION` from `pulse_artifact_qc.py` (line 12), which moves to the study —
and core must not import from `studies/`, enforced by `test_core_does_not_import_the_study`.

The fix is to **keep the marker path and genericize it**, not to delete it. Reading beat
markers is a general capability: any EEG study may record a beat marker train, from a pulse
oximeter trigger or the amplifier itself. What is Analyzer-specific is the hardcoded string,
so `_marker_beats` takes the description from `ica.cardiac_review.marker_description` and
`detect_ecg_events` honours `beat_source` instead of a fixed preference order.

The default `beat_source: auto` preserves current behaviour exactly. Note that the reason
core hardcodes marker-preference today is an in-scanner one — the channel detector locks
onto the magnetohydrodynamic deflection and reports 8 and 2 bpm where the markers report 61
and 60. Outside a bore that reason does not apply, which is why the order becomes a choice
rather than a rule.

**2. `add_rr_interval_section` is extracted before `analyzer_qc.py` moves.** It goes to its
own core module — `report/rr_intervals.py` — and takes its beats from `detect_ecg_events`,
so it inherits the configured beat source rather than reading Analyzer's markers itself.

## Consequences accepted

1. **Aperiodic exponents change.** Deleting `gradient_windows` means the comb harmonics are
   no longer withheld from the aperiodic fit, so the fitted line is tilted by the comb.
   Every exponent in the subject and cohort reports, and every `aperiodic_*` sidecar value,
   shifts — and will not be comparable to anything already computed.
2. **The cohort sidecar loses 11 columns**, requiring a schema version bump. Cohort reports
   cannot read sidecars written before this change, and vice versa.
3. **Existing study configs break.** Any config setting `preprocessing.eeg_fmri`,
   `brainvision_analyzer.*` or the moved report keys now names a key core does not define.
   `studies/pain_study/scripts/config/thermal_pain_eeg_overrides.yaml` and the study1/2/3
   configs must be checked and updated in the same change — including setting
   `ica.cardiac_review.marker_description: "Pulse Artifact/R"`, without which the review
   silently switches to channel detection and meets the MHD lock-on.

## Sequencing

One rule change, but roughly 4,000 lines relocating across ~20 core modules, the config
tree, and ~45 test files. One plan, five phases, each leaving the suite green:

1. **Gradient.** `report/scanner.py` and `report/cohort/gradient.py` out; the new
   `analysis/gradient/`, `scripts/gradient/` and CLI command in. Report structure, spectra
   and continuity edits land here.
2. **Extract and genericize what stays.** `add_rr_interval_section` to
   `report/rr_intervals.py`; the `ecg` beat-source config and the config-driven marker name
   in `ica_cardiac_review.py`. Both before anything cardiac moves, so the suite never passes
   through a state with no ECG review. This phase must be behaviour-preserving — `auto` with
   the study's marker description set has to reproduce today's detections exactly.
3. **Analyzer and BCG.** `analyzer_qc.py`, `cohort/analyzer.py`, `cohort_qc.py`,
   `pulse_artifact_qc.py`, `cardiac_artifact_qc.py` out to `analysis/bcg/`.
4. **Config.** Keys out of `eeg_config.yaml` into the workflow configs, `acquisition.py`
   deleted, `coherence.py` and `loader.py` trimmed, `eeg_only.yaml` retired, study configs
   updated.
5. **Cohort sidecar, architecture test, pipeline steps.** The schema change, the new
   contract written down, and the STEP constants removed once nothing dispatches to them.

Phase 1 alone changes aperiodic exponents. Worth confirming against a real subject at the
end of it rather than discovering it at the end of phase 5.

## Testing

The architecture test is the specification. `RELOCATED` gains the moved core paths,
`RELOCATED_TO` gains `studies/pain_study/analysis/gradient`,
`studies/pain_study/scripts/gradient` and `studies/pain_study/scripts/bcg`, and the
docstring records the new rule.

A new assertion should check that core Python names no scanner concept, in the same spirit
as `PARADIGM_MARKERS`. **The markers must be compound terms, not the bare word
`gradient`:** `volume_locked`, `repetition_time_s`, `volume_marker`, `Pulse Artifact/R`,
`scanner gradient`, `brainvision_analyzer`. Core legitimately contains `np.gradient`
(`analysis/features/quality.py`, `analysis/features/spectral.py`),
`GradientBoostingRegressor` (`analysis/machine_learning/uncertainty.py`) and
`cnn_gradient_clip_norm` (`analysis/machine_learning/cnn.py`) — five files a naive
substring check would fail on, none of which has anything to do with a scanner. A companion
assertion should check `eeg_config.yaml` for the same terms.

Moving to `studies/tests/` with their subjects: `test_report_scanner.py` (736 ln),
`test_cohort_gradient.py` (429 ln), `test_report_analyzer_qc.py`, `test_cohort_analyzer.py`,
`test_pulse_artifact_qc.py`, `test_cardiac_artifact_qc.py`,
`test_report_marker_agreement.py`, `test_scanner_harmonic_qc.py`.

Staying in core, but re-pointed at the extracted modules:
`test_report_rr_intervals.py`, `test_ica_cardiac_report_figures.py`,
`test_ica_cardiac_promotion_wiring.py`.

New coverage for the beat source, since it is the one behavioural seam in this change:
each of `markers` / `detect` / `auto` resolving as documented; `markers` failing loudly
rather than silently detecting when the named annotation is absent; `auto` with a marker
description set reproducing today's detections exactly; and `marker_description` surviving
config load with its slash intact.

Edited in place: `test_cohort_record.py`, `test_cohort_sidecar.py`, `test_cohort_report.py`,
`test_cohort_multiplicity.py`, `test_cohort_composition.py`, `test_cohort_spectra.py`,
`test_cohort_at_a_glance.py`, `test_cohort_end_to_end.py`, `test_report_modes.py`,
`test_report_organize.py`, `test_report_spectra.py`, `test_report_continuity.py`,
`test_report_run_evidence.py`, `test_report_at_a_glance.py`, `test_report_settings_*.py`,
and the config tests covering `coherence.py`, `loader.py` and the shipped presets.

Per the project's standing constraint, verification runs targeted subsets rather than the
full suite.

## Out of scope

- `studies/pain_study/analysis/line_comb`, `scripts/line_comb`, `gradient_trough_ica`,
  `scanner_contamination.py` — already study-side and correctly placed. They gain config
  keys but no code moves.
- Regenerating derivatives. Existing reports and sidecars are not migrated; the schema bump
  means they are read by the version that wrote them.
- **Beat-train QC, deferred to its own spec.** `drop_double_marks`, `find_gaps` and
  `physiological_floor` in `studies/pain_study/analysis/bcg/detect.py` are general — they
  answer "can this beat train be trusted", which any study with markers should ask, and the
  percentile-based gap test catches the single missed beat a 2x-median threshold is blind
  to. They are held back because this change is a relocation: folding a new capability in
  alongside shifting aperiodic exponents and a sidecar schema bump would make a regression
  impossible to attribute. Their thresholds would also need promoting from cohort-tuned
  constants to documented config defaults, which is its own work.
- **`recover_beats` stays in the study, permanently.** It is gap repair rather than
  detection — it needs `MINIMUM_SEED_BEATS = 8` existing marks, builds its template from
  them, and searches only inside gaps, so it cannot detect from scratch and is not an
  alternative beat source. It is also calibrated on one cohort and one vendor's failure
  mode, and its own A/B was net -36% residual BCG with 4 runs regressing. That is a valid
  result for this dataset and not a general capability.
