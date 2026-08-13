# Preprocessing report: move setup constants out of the code and into the config

Date: 2026-08-12

## Problem

The report modules describe an acquisition in twelve places where the description is a
module constant rather than a setting, and expose a thirteenth parameter only as a CLI flag.
A lab whose recording differs from this project's cannot change them without editing the
pipeline, and the report gives no sign that a number was assumed rather than measured.

[`settings.py:13`](../../../eeg_pipeline/preprocessing/report/settings.py) already states
the line this spec applies:

> Values that define a *method* rather than a setup stay as module constants where they
> are used [...] Exposing those would invite tuning an estimator per subject.

The constants below are on the setup side of that line and were never moved. Estimator
constants — `HARMONIC_PEAK_FRACTION`, `BACKGROUND_FRACTION_RANGE`,
`RESOLVABLE_PEAK_FALSE_POSITIVE_RATE`, `_EXTRAPOLATION_INFLATION`, `PEAK_RESIDUAL_QUANTILE`,
`CUTOFF_DB`, the `MINIMUM_*` counts — stay exactly where they are. Several carry docstrings
recording the simulations that calibrated them; a config key would invite retuning them per
study and quietly break comparability between two labs' reports.

### Two constants make the same claim and disagree

| | | |
|---|---|---|
| `analyzer_qc.py:62` | `PLAUSIBLE_RR_RANGE_S = (0.3, 2.0)` | 30–**200** bpm |
| `cohort/analyzer.py:50` | `PLAUSIBLE_BPM = (30.0, 220.0)` | 30–**220** bpm |

Both feed counts of implausible intervals printed to the reader, and the cohort text prints
its bound as "possible heart rate". A participant at 210 bpm is implausible in the subject
report and ordinary in the cohort report. Both ranges are adult; a developmental cohort sits
outside them.

### One constant suppresses events it should not

`continuity.py:195`:

```python
_NON_EVENT_PREFIXES = ("BAD", "EDGE", "NEW SEGMENT", "VOLUME/", "R  ", "R/", "RESPONSE/")
```

The docstring above it says event names "belong to whatever paradigm produced the data and
cannot be enumerated here", then enumerates this acquisition's. `volume_marker_description`
and `pulse_marker_description` are already passed in and merged into the exclusion set at
`continuity.py:205`, so `VOLUME/`, `R  ` and `R/` are residue matching only this site.
`RESPONSE/` suppresses button presses — in a response-locked or self-paced paradigm those
are the events, and the "Data quality over time" figure would report `0 event(s)` on a run
full of them.

### One parameter is tunable but unrecorded

`BandGates` (participants required before a median or decile band is drawn) is reachable
only through CLI flags on the cohort command
([`cohort_report_parser.py:61`](../../../eeg_pipeline/cli/commands/cohort_report_parser.py)).
It therefore never reaches `PROVENANCE_KEYS`, and a lab cannot ship it in a config file.

### Out of scope, on inspection

`PULSE_MARKER_QC_SUFFIX` and `CARDIAC_ATTENUATION_QC_SUFFIX` (`analyzer_qc.py:48-49`) look
like vendor settings but are not. Both files are written by this pipeline
(`pipelines/preprocessing.py:842`, `preprocessing/cardiac_artifact_qc.py:427`) and read back
by the report. That is an internal writer/reader contract; a config key would only let a lab
break its own discovery.

`DRAWN_RR_RANGE_S` (`analyzer_qc.py:77`) stays hardcoded. Its docstring explains it is
deliberately fixed so the same interval occupies the same height in every report; making it
settable defeats the property it exists for.

## Design

### Threading

No new plumbing style. `run_evidence.py` is already the hub that reads `ReportSettings` and
passes explicit keyword arguments to pure measurement functions
(`window_seconds=settings.continuity_window_seconds`,
`volume_description=settings.volume_marker_description`). Measurement modules stay free of
settings coupling and testable without a config.

Each new parameter follows that: a field on `ReportSettings`, a keyword argument with the
current constant as its default, resolution at the existing call site. Three sites resolve
settings today and gain only argument passing:

- `report/run_evidence.py` — continuity, preservation, spectra
- `pipelines/preprocessing.py:1914` — `add_analyzer_correction_review`
- `cli/commands/cohort_report_orchestrator.py:28` — cohort gates and panels

### Keys

Placed in the existing `report` blocks; no new block.

| Key | Replaces | Default |
|---|---|---|
| `acquisition.non_event_prefixes` | `continuity.py:195` | `["BAD", "EDGE", "NEW SEGMENT"]` |
| `acquisition.component_label_patterns` | `cohort/record.py:308` | current 9 pairs |
| `analysis.alpha_reference_band_hz` | `preservation.py:59` | `[3.0, 25.0]` |
| `analysis.bcg_residual_window_s` | `analyzer_qc.py:403` | `[-0.2, 0.6]` |
| `analysis.bcg_residual_baseline_s` | `analyzer_qc.py:404` | `[-0.2, -0.1]` |
| `analysis.bcg_residual_measurement_s` | `analyzer_qc.py:407` | `[0.0, 0.5]` |
| `thresholds.plausible_heart_rate_bpm` | `analyzer_qc.py:62` + `cohort/analyzer.py:50` | `[30.0, 220.0]` |
| `thresholds.marker_agreement_tolerance_s` | `analyzer_qc.py:543` | `0.1` |
| `thresholds.notch_exclusion_half_width_hz` | `filtering.py:44` | `2.0` |
| `thresholds.repetition_time_tolerance_s` | `cohort/gradient.py:65` | `1e-3` |
| `thresholds.channel_position_tolerance_m` | `cohort/coverage.py:44` | `5e-3` |
| `thresholds.min_subjects_for_median` | `cohort/aggregate.py:116` | `5` |
| `thresholds.min_subjects_for_outer_band` | `cohort/aggregate.py:116` | `10` |

The heart-rate key is stored in bpm because that is the unit a reader thinks in, and the
subject panel derives its RR-seconds window from it. One source, two consumers, so the two
reports cannot drift apart again.

`component_label_patterns` is an ordered list of `[substring, class]` pairs, matched
lowercase against the prose a detector writes into `status_description`. Order is
load-bearing and must be preserved from YAML: `"channel noise"` and `"line noise"` both
contain `"noise"`, and the more specific reading has to win. `class` must be one of
`COMPONENT_LABEL_CLASSES` minus `other` and `unrecorded`, which are outcomes rather than
patterns; an unmatched description still lands in `other` and is still counted.

The two `min_subjects_*` keys feed `BandGates`. The existing CLI flags stay and become
overrides of the config value rather than the only way to set it; `BandGates.__post_init__`
already rejects a gate that would extrapolate a quantile, so no config value can produce a
band the sample cannot support. These are distinct from the existing
`thresholds.min_runs_for_quantile_band`, which gates a within-participant band over runs
rather than a cohort band over participants; both stay.

### Validation

Follows the existing `validate()` idiom in `settings.py`: ordered pairs where a pair is a
range, positive tolerances, non-empty strings, `re.compile` on anything used as a pattern.
`bcg_residual_baseline_s` and `bcg_residual_measurement_s` must additionally fall inside
`bcg_residual_window_s`, since a baseline outside the epoch is silently no baseline.

### Follow-through the keys imply

Every new key gets a row in `PROVENANCE_KEYS` (`report/provenance.py:21`), so a report
records the setup it assumed. `provenance_html` already omits keys a config does not carry,
so an EEG-only dataset gains no empty rows.

Keys that change a pooled number go into `COMPARED_SETTINGS`
(`report/cohort/homogeneity.py:34`): `alpha_reference_band_hz`,
`plausible_heart_rate_bpm`, the three BCG windows, and `marker_agreement_tolerance_s`. Added
at the same time: `posterior_channel_pattern`, which was already configurable and already
missing from that list — it selects the sensors alpha prominence is measured over, so it
changes the pooled number as much as `alpha_band_hz` does.

## Behaviour changes

One, by decision: the subject report stops flagging 200–220 bpm intervals as implausible.
Nothing that currently counts as plausible becomes implausible; cohort output is unchanged.

`non_event_prefixes` requires a companion edit rather than a behaviour change. Trimming the
shipped default to pipeline-and-MNE conventions moves the site spellings to
`studies/pain_study/scripts/config/thermal_pain_eeg_overrides.yaml`, so this study's
continuity figure keeps excluding what it excludes today. This study's own events are
`Trig_*` (`condition_preferred_prefixes` in that file), so nothing it counts today changes.

Every other key defaults to its current value, so an unmodified config produces an
unmodified report.

## Testing

- Each key: default reproduces the current constant; an override reaches the measurement.
- Rejected values raise from `validate()` with the config key named — the existing pattern
  in `tests/preprocessing/test_report_settings_new_keys.py`.
- Heart rate: one range drives both the subject RR window and the cohort bpm bound, so a
  single override moves both.
- BCG baseline outside its window is rejected.
- `non_event_prefixes`: an event named `Response/R  1` is counted with the shipped default
  and excluded with the pain-study override.
- Provenance: every new key appears in a rendered report when the config carries it.
- Homogeneity: two participants differing on `posterior_channel_pattern` are reported as a
  disagreement.

Targeted subsets only — the full suite takes ~9 minutes.

## Documentation

The "what this preset does not decide for you" list in
`eeg_pipeline/utils/config/presets/eeg_only.yaml` currently names epochs, montage, EOG,
resample and notch, and no `report` key at all. It gains the settings a new lab must review
before trusting a panel: `report.analysis.response_window_s` (a paradigm whose response
falls outside 0–1 s reports near-zero split-half reliability for a sound dataset),
`alpha_band_hz` and `alpha_reference_band_hz`, `acquisition.posterior_channel_pattern`,
`acquisition.non_event_prefixes`, and the top-level `rois` block, whose region definitions
are this study's easycap-M1 names and whose hemisphere labels the config's own comment says
are valid only for consistent left-side stimulation.
