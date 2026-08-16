# Trimming the subject preprocessing report

## Problem

A subject report is 79 MB across 31 sections (measured on `sub-0011`, 59 ICA components).
Three blocks are 89% of it, and all three are per-ICA-component:

| Block | MB | % |
|---|---|---|
| Five `ICA component review: <band>` sections | 49.4 | 62% |
| `ICA cardiac artifact review` | 13.2 | 17% |
| `ICA decomposition quality` | 7.7 | 10% |
| All 23 other sections combined | 6.9 | 9% |

Each component is drawn as seven full-page figures plus a contact-sheet tile. The 23
substantive QC sections — filter response, coverage, continuity, preservation, gradient
residual, Analyzer QC, rejection, spectra — cost 6.9 MB between them and are not the
problem. Byte-level work is already done: figures are WebP, and the exploratory
band-fitted ICAs already live in a separate linked file.

The lever is how many times each component is drawn, not encoding and not which sections
exist.

## Governing constraint

ICLabel is wrong in both directions. A component marked bad may be good; a component
marked good may be an artifact. The report's job is to let a reviewer audit that verdict,
which means **per-component detail may be reduced by redundancy, never by the detector's
verdict**. Filtering slides to "excluded or flagged" would make false negatives
structurally undiscoverable — an artifact labelled `brain 0.87` and kept would have no
slide at all. Every component keeps its evidence. What goes is the same evidence drawn
repeatedly.

## Changes

### 1. Collapse the authoritative review to one band

`BAND_ICA_DEFINITIONS` currently serves two different loops:

- `_add_standard_component_review` (`band_ica_report.py:1839`) — five sections of the
  *same* broadband ICA. The topomap is bit-identical across all five, the spectrum is one
  curve on five x-limits, and only the TFR row differs. Delta+theta, alpha and beta are
  strict subsets of Broadband 1–30. This is the 49.4 MB.
- The exploratory band-fitted ICAs (`band_ica_report.py:2088`) — five *separate* ICA
  fits, already written to their own linked file. Narrow-band decomposition is the point
  there.

Collapsing the shared constant would silently destroy the second. So the review gets its
own list: `BandIcaReportSettings.review_bands`, default a single
`BandIcaDefinition("broadband1to100", "Broadband 1–100 Hz", 1.0, 100.0)`, settable via
`ica.band_specific_report.review_bands`. `BAND_ICA_DEFINITIONS` is left untouched for the
exploratory fits.

`_tfr_parameter_groups` already walks the four DPSS parameter sets across an arbitrary
wide band, so smoothing stays band-appropriate with no new machinery. Compute drops too:
one TFR per component instead of five.

Old reports rebuild cleanly — `remove_tagged_content(report, tag="ica-component-review")`
is keyed to the tag, not the section name, and `_remove_legacy_condition_tfr_entries`
already exists for exactly this migration.

Result: 295 slides → 59, no component dropped.

### 2. Log frequency axis on the dossier TFR

On a linear 1–100 Hz axis, 1–8 Hz occupies 7% of the panel height — and blink/eye
discrimination lives there, which is precisely the ICLabel judgement being audited. A log
y-scale gives 1–8 Hz roughly 45% of the axis.

`pcolormesh(shading="auto")` handles a log scale, and `_annotate_spectral_resolution`
works in data coordinates so its boundaries and `±N Hz` labels still land correctly. With
one wide band it now has four groups to annotate, where today each narrow band has
constant smoothing and no annotation — so the figure states a resolution it actually has.

### 3. Fold `plot_properties` into the dossier

MNE's `Component properties` slider is 59 more raster figures inside `ICA decomposition
quality`. Of its five panels, the topomap and spectrum are already on the dossier. The
two that are not — the epochs image and per-epoch variance — are exactly
ICLabel-validation evidence: a `brain 0.9` component driven by three epochs is not brain,
and nothing else in the report shows that.

Those two panels move onto the dossier and the separate slider is removed. Every
component then carries one complete slide: topomap, full ICLabel class distribution,
spectrum, TFR, epochs image, per-epoch variance. This is more validation evidence per
component than today, in one place instead of two.

### 4. Cardiac review keeps all 59 slides, gains a screening view

A cardiac component ICLabel missed is the false negative most worth catching, so the
per-component R-locked slides stay for every component. What is added is an
all-components cardiac-score-by-run figure, mirroring the ocular section's existing `ICA
components: EOG correlation by run`, so a suspicion can be raised in screening and then
interrogated in the slides.

This section stays around 13 MB. Review validity is not traded for bytes here.

### 5. Relocate the MNE panel deduplication

`drop_superseded_mne_ica_panels` is currently called from
`_add_standard_component_review`, and the last thing to call it after MNE-BIDS-Pipeline's
`_08a_apply_ica` re-writes those panels is `_append_band_ica_condition_tfrs` — which is
gated on `ica.band_specific_report.comparisons` being non-empty
(`pipelines/preprocessing.py:1650`).

Consequences today:

- A dataset with no configured condition contrasts (including this repo's own `eeg_only`
  preset, which sets `comparisons: []`) keeps MNE's `ICA component topographies` and `ICA
  component properties` permanently duplicated in `ICA: removals`.
- Setting `band_specific_report.enabled: false` — the obvious move for a lighter report —
  means the drop never runs at all, so the cheap configuration produces the *more*
  duplicated document.

The call moves to `_append_report_review_sections` (`pipelines/preprocessing.py:1905`),
which already runs last and unconditionally.

### 6. Drop the remaining superseded MNE panels

In `report/organize.py`, beside the existing drops and following the module's rule that a
drop lives with the section rendering its replacement:

- The two EOG panels in `ICA: components` (`Scores for matching EOG patterns`, `Original
  and cleaned EOG epochs`). The matching ECG pair is already dropped, and the ocular
  review renders both equivalents; the asymmetry is unintentional.
- `Raw (clean)`'s `Time series` and `PSD`, which the continuity and sensor-spectra
  sections replace.

### 7. Runner-up class on the triage sheet

`plot_component_overview` (`report/summary.py:814`) is the screening view and is now doing
more work. It shows topomap, short label, probability and excluded status. It gains the
runner-up class when the winning class is under-confident, so `brain 0.51 / muscle 0.44`
is visible where a reviewer decides what to interrogate, rather than only in the component
TSV.

## Expected result

Roughly 79 MB → 33–38 MB; 31 sections → 27; ~415 component figures → ~120 (59 dossiers +
59 cardiac slides + overviews). Every one of the 59 components retains complete
ICLabel-validation evidence.

## Out of scope

- Restricting any evidence by detector verdict. Rejected on the grounds above.
- A `report.detail` coarse dial. Worth considering later; it is not needed for this change
  and would add a tunable whose settings all have to be tested.
- Any change to the 23 non-component QC sections.
- Any change to the exploratory band-fitted ICAs, which already live in their own file.

## Testing

Extends the existing suites rather than adding parallel ones:

- `test_report_modes.py` — a case asserting no duplicate MNE ICA panels survive with
  `comparisons: []`. This fails against current `main`.
- `test_band_ica_report.py` — `review_bands` default and parsing; the dossier carrying the
  epochs-image and variance panels; the log frequency axis.
- `test_report_organize.py` — the two new drops.
- `test_ica_cardiac_report_figures.py` — the screening figure, and that all components
  still get a slide.
- `test_report_summary.py` — runner-up annotation.

Per the project's standing rule, verification runs these files only; the full suite takes
about nine minutes.

## Verification

Rebuild one subject report and measure section count, figure count and file size against
the 79 MB / 31 section / 415 figure baseline recorded above.
