# fMRI subject report: run-level inference, MNI referability, print output

Date: 2026-08-01

Successor to `2026-07-28-fmri-report-scientific-robustness-design.md`, which built the
panels this spec reframes. Everything below is verified against
`outputs/fmri_report_redesign/` (sub-0001, task-thermalactive) and the derivatives tree
at `outputs/fmri_complete_report_20260801/derivatives/sub-0001/`.

## Problem

The redesigned report computes the right quantities and then subordinates the decisive
ones. Four defects, in descending order of consequence.

### 1. The report's own numbers say the contrast is near-null; the report reads as a finding

`thresholds.tsv` records 8,463 voxels above the applied height and **8,001 expected
under the map's own fitted null** — a 5.8% excess. "At a glance" promotes
`8,463 of 50,626 (16.72%)`, which reads as a result. The count that survives FDR
against the fitted null is 82.

No familywise correction respecting the actual between-run variance is offered at all.
`Methods` states this plainly, but the only corrections on the page (Bonferroni, FDR vs
N(0,1)) both assume a null the same page shows to be wrong.

### 2. The stated cause of over-dispersion is contradicted by the report's own data

The empirical null is N(-0.61, 1.51²). The calibration caption attributes this to
"unmodelled autocorrelation", but `model_fit_measurements.tsv` gives median residual
ACF(1) of 0.050–0.073 across the six runs. Autocorrelation is not what inflates σ here.

Measured instead, from the per-run effect maps (whole-mask mean β, % signal change):

| run | mean β | r with combined | median SE |
|---|---|---|---|
| run-01 | -0.0834 | 0.402 | 0.1016 |
| run-02 | -0.0124 | 0.466 | 0.0956 |
| run-03 | **+0.0570** | **0.314** | 0.0958 |
| run-04 | -0.0422 | 0.518 | 0.1105 |
| run-05 | -0.0676 | 0.471 | 0.1094 |
| run-06 | -0.0367 | 0.554 | 0.1049 |

Combined mean β = -0.0271 %SC, mean z = -0.547 — which *is* the fitted null's -0.61.
The shift is a whole-brain per-run offset, not anatomy. It is what manufactures the
138,213 mm³ negative cluster (peak z = -8.81) and the CSF-over-GM survival ordering
(CSF 21.6% > GM 20.0% > WM 11.3%).

Leave-one-run-out on the same maps, inverse-variance recombination:

| dropped | survivors \|z\|>2.3 | Δ | max\|z\| | r(z_loro, z_all) |
|---|---|---|---|---|
| none | 8,203 | — | 9.14 | — |
| run-01 | 7,325 | -878 | 8.37 | 0.943 |
| run-02 | 7,600 | -603 | 8.39 | 0.880 |
| **run-03** | **10,465** | **+2,262** | 9.01 | 0.937 |
| run-04 | 7,018 | -1,185 | 8.58 | 0.959 |
| run-05 | 7,391 | -812 | 8.78 | 0.961 |
| run-06 | 6,325 | -1,878 | 8.21 | 0.949 |

run-03 is the influential run at map level — the only run with a positive whole-mask
mean, so it partly cancels the other five. `run_consistency.png` identifies run-02
instead, because run-02 carries the largest *peak* estimates (peak 2: 1.2 %SC against a
combined 0.42 %SC). Both are true; they measure different things, and the existing
forest panel cannot produce the second.

A run-level sign-flip null over the same maps (32 distinct sign patterns; the ±global
pair is redundant for a two-sided max statistic). **These are prototype values computed
under plain inverse-variance recombination**, which is not the pooling rule the pipeline
uses — see the r = 0.988 / max\|diff\| = 1.73 discrepancy noted in Phase 1A. The shipped
implementation pools exactly, so the heights and counts below will move. They are given
to establish the method's discriminating power, not as the values to expect:

| threshold | height | surviving |
|---|---|---|
| applied, uncorrected | 2.30 | 8,463 |
| Bonferroni 0.05 | 4.89 | 369 |
| FDR q=0.05 vs fitted null | -6.57 / +5.35 | 82 |
| **sign-flip FWE 5%** | **7.58** | **26** |

Global max-statistic p = 0.061, against a resolution floor of 1/33 = 0.030 with six
runs. The null's median max\|z\| is 5.99; under N(0,1) over 50,626 voxels it would be
about 4.4. The over-dispersion is between-run structure, and spatial smoothing (6.6 mm
FWHM estimated) would if anything *lower* a max statistic, not raise it.

### 3. MNI derivatives exist and go unused

At the same config hash `1990c9ee`, alongside the native maps:

```
..._space-MNI152NLin2009cAsym_stat-effect_size_1990c9ee.nii.gz    (53,65,56) 75,788 voxels
..._space-MNI152NLin2009cAsym_stat-effect_variance_1990c9ee.nii.gz
..._space-MNI152NLin2009cAsym_stat-z_score_1990c9ee.nii.gz
```

The report nonetheless prints "coordinates: scanner-native (mm), not MNI; not
atlas-referable", "no anatomical labels", and "No glass brain for this contrast". A
results table of native-space peak coordinates is not publishable, and the disclaimers
are stale rather than wrong-in-principle.

### 4. Figures are not submittable, and several fight their own data

- Raster-heavy panels emit **PNG only** at `HTML_FIGURE_DPI = 150`: every brain mosaic,
  both carpets, `effect_versus_evidence`, all six design matrices,
  `design_correlation`. `PRINT_FIGURE_DPI = 300` exists and is unused by these outputs.
  `_ALLOWED_FORMATS` is `{"png", "svg"}`; there is no PDF.
- `cut_coords_for` spaces cuts evenly across slices holding ≥ `MIN_SLICE_AREA_FRACTION`
  (0.25) of the fullest slice's area. At the extremes this admits tiles that are mostly
  neck (`z = -50`), eyeball (`y = +71`), and brain edge (`x = ±55`), and leaves a wide
  dead band before the colorbar.
- Both carpets z-score per voxel and clip at ±2.5, which renders structure as grey
  static.
- DVARS is standardized in `motion_coupling.png` (~1.0) and raw in `carpet.png`
  (~25–30) — same report, two units.
- The two regressors carrying the contrast sit at VIF ≈ 8–9, indistinguishable among 46
  other bars in `design_vif.png`.
- Peaks for `peak_response` and `run_consistency` are drawn from positive clusters, so
  the largest effect in the map is never shown.
- Per-run event counts are 6/5, 5/6, 5/6, 7/4, **8/3**, **3/8**. `event_raster.png`
  additionally shows the two conditions time-segregated within several runs, confounding
  the contrast with time-on-task. Neither is quantified anywhere.
- "no criterion is applied" and near-variants appear about twelve times. The principle
  is right; at that density it is boilerplate that crowds out the observed-vs-expected
  comparisons, which are measurements and belong in the headline.

## Design

Three phases. Phase 1 is the scientific core and lands first.

### Phase 1 — A: new statistics, computed where the model lives

`analysis/run_level.py` already writes 4D per-run effect and variance volumes
(`(49,58,48,6)`, fully populated) and holds the fitted `flm`. Both new statistics are
computed there, during analysis, and written to derivatives. The report package stays
fit-free and only reads.

**Sign-flip null.** For sign pattern `s`, pass the per-run vectors `sᵢ·cᵢ` through the
same `compute_contrast` pooling the pipeline already uses — `_contrast_vectors` accepts
a per-run list of definitions today, so this is the exact pooling rule, not a
re-derivation. Iterate the 32 distinct patterns, record max\|z\| per pattern.

Writes `..._desc-signflip_null.tsv` (one row per pattern) and, into the manifest: FWE
height at 5%, voxels surviving it, global max-statistic p, the resolution floor
`1/(n_patterns+1)`, and `n_runs`.

The resolution floor is recorded and reported, not hidden: with six runs the smallest
attainable p is 0.030, and a report that prints `p = 0.061` without it invites the
reader to treat the value as continuous.

**Leave-one-run-out.** `compute_fixed_effects` over the run-level maps for all seven
combinations under one rule. Reported as **deltas**, with the all-runs baseline from
that same rule printed beside the stored value.

This last point is a hard requirement. Plain inverse-variance recombination reproduces
the stored z at r = 0.988 but with max\|diff\| = 1.73, and yields 8,203 survivors where
the report's cluster table says 8,463. The spec does not resolve that discrepancy by
choosing a favourite; it displays both numbers so the recombination rule's cost is
visible.

Writes `..._desc-runinfluence.tsv`: dropped run, survivors, delta, max\|z\|,
r(z_loro, z_all).

### Phase 1 — B: reframing, no new computation

1. **"At a glance"** leads with observed against expected under the map's own fitted
   null, and with the sign-flip FWE count. The raw percentage stays, demoted.
2. **Thresholds panel** gains the sign-flip row, and its caption stops describing the
   page as offering no familywise correction.
3. **Calibration caption** drops the autocorrelation attribution. It states the measured
   ACF(1) range and the per-run offset table instead, since those are what the data
   support.
4. **Peak selection by \|z\|** in `timeseries.py` and `run_consistency.py`, so the
   largest cluster appears in both. Both modules cap at `max_peaks`; only the ordering
   changes.
5. **Contrast regressors split out** of `design_vif.png` into their own labelled row.
6. **Cluster table sourced from the MNI map**: coordinates in MNI, atlas labels via the
   existing `report/atlas.py`, with the native maps still drawn. `atlas_applies_to`
   already gates on space, so the gate moves rather than being removed. The native table
   remains available as a download.

### Phase 2 — C: new panels

1. **Run influence** — Δ survivors and r(z_loro, z_all) per dropped run.
2. **Sign-flip null** — the null max-\|z\| distribution, observed value marked, FWE
   height drawn, resolution floor annotated.
3. **Global offset by run** — whole-mask mean β per run against the combined value,
   making the -0.61 shift a measurement rather than an inference. Report-side; the maps
   are on disk.
4. **Design confound** — per-run correlation of the contrast regressor with time-on-task
   and with the drift basis, plus the per-run event counts already in
   `design_summary.tsv`. Report-side; design matrices are recorded.

### Phase 3 — D: plot craft

- `cut_coords_for` / `mask_cut_coords` place cuts at equal quantiles of **cumulative**
  in-plane mask area rather than evenly across slices passing a 25% floor. Tiles then
  concentrate where there is brain and none land on neck or eyeball. `_mosaic.py`
  closes the colorbar gap.
- Carpets draw windowed % signal change with contiguous voxel ordering, replacing
  per-voxel z at ±2.5.
- DVARS standardized in every panel that shows it.
- Tissue panel gains an expected-under-null reference so CSF > GM is legible as an
  anomaly.
- Caption boilerplate reduced to one statement of the no-criterion principle per
  section, not per panel.

### Phase 3 — E: print profile

- `_ALLOWED_FORMATS` gains `pdf`.
- A `print` render profile re-renders figures at `PRINT_FIGURE_DPI` with type scaled to
  final physical size. Each figure declares its own target width; the default is 170 mm
  (double column), with 85 mm (single column) declared explicitly by the panels that
  read at that size — the motion, calibration, tissue, run-influence and sign-flip
  panels. Multi-tile mosaics and carpets are always 170 mm.
- `set_rasterized(True)` on **only** the brain/carpet image artists, so the raster layer
  is embedded at print resolution while text, axes and annotation stay vector.
- The provenance strip is suppressed in the print profile (it is a working-artifact
  device, and `savefig_kwargs` already depends on the tight bbox that includes it — the
  print path must set both together or silently crop).
- Output to `plots/print/`. The screen report is untouched, per the decision to keep one
  figure spec with two render profiles.

### F: cohort-ready outputs

Each new measurement lands in a per-subject TSV/JSON with a fixed schema, so the planned
cohort report stacks rows without re-deriving anything:

| field | source |
|---|---|
| `signflip_fwe_height`, `signflip_survivors`, `signflip_global_p`, `signflip_resolution` | Phase 1A |
| `loro_max_abs_delta`, `loro_most_influential_run` | Phase 1A |
| `global_offset_mean_beta`, `global_offset_mean_z` | Phase 2 |
| `contrast_vif_max`, `event_count_min_per_run` | Phase 1B / 2 |
| `fitted_null_mu`, `fitted_null_sigma`, `expected_under_fitted_null` | already computed |

## Testing

- Sign-flip: the identity pattern must reproduce the stored z map exactly (same pooling
  path); the null must contain exactly 32 patterns for 6 runs and `2^(n-1)` in general;
  a single-run model must yield `None` rather than a degenerate null.
- LORO: dropping a run must change the survivor count; the all-runs baseline under the
  recombination rule must be recorded alongside the stored value, and a test asserts
  both are surfaced rather than reconciled.
- Peak ordering: a map whose strongest peak is negative must place that peak first in
  both `timeseries` and `run_consistency`.
- Cut placement: a mask tapering at one end must yield no cut in the taper.
- Print profile: a rasterized-artist figure must still contain selectable text in the
  PDF; the provenance strip must be absent and the bbox correspondingly adjusted.
- Report-package invariant: the existing test asserting the fitting modules stay absent
  from the report import path must still pass.

## Non-goals

- No cluster-extent familywise correction. The sign-flip gives a height, not an extent
  distribution; adding extent inference is a separate piece of work.
- No change to what the GLM fits. The global per-run offset is *reported*, not
  corrected — whether to add a global-signal or grand-mean term is an analysis decision,
  not a reporting one.
- No cohort report. Phase F only fixes the schemas it will consume.
