# Subject Report: Table Layer and Figure Corrections

## Objective

Bring the subject HTML report into line with the conventions it already declares, in
two independent passes:

1. **Tables** — one builder and one stylesheet, replacing five coexisting table
   stylings, three run-naming conventions, and two tables carrying raw dataframe
   column names.
2. **Figures** — twelve corrections, two of which change measurements that currently
   report the filter or a slow drift rather than the artifact they name.

Out of scope, deferred to a separate spec: new panels (processing-steps table, line
noise, electrode bridging, rest-mode spectral split-half, per-channel summary matrix)
and the exclusion-class summary above the component ledger.

## Motivation

`report/style.py` states the document's conventions: hue encodes a measured quantity,
a neutral grey ramp encodes a pipeline decision, zero-centred power uses a diverging
colour-vision-safe map on a symmetric scale. Most of the report honours them. The
defects below are places where it does not, plus two measurements that answer a
different question from the one their label asks.

Findings were taken from `sub-0015_report.html` (70.9 MB, 34 tables, 468 embedded
figures) built 2026-07-26, read alongside the modules that produced it.

---

## Part 1 — The table layer

### Current state

Twenty-two hand-built tables in two shapes, plus tables MNE and MNE-ICALabel
contribute:

| Shape | Sites |
| --- | --- |
| Key–value (`<table><tbody>`, 2 columns) | `filtering`, `preservation` ×2, `rejection` ×2, `summary` ×2, `coverage`, `cohort_qc` ×2 |
| Headed grid (`<thead>` + rows) | `provenance`, `build_record`, `at_a_glance`, `continuity`, `scanner` ×2, `analyzer_qc` ×2, `coverage`, `summary` (ledger), `ica_ocular_report`, `band_ica_report` |
| Headed grid, two-level header | `spectra` (hand-rolled `rowspan='2'`) |

Five stylings render in one document:

- bare `<table>` — 24 tables, no Bootstrap class, so no padding, no row rule, no
  numeric alignment
- `class='table table-sm'` — one table, `coverage.py:216`
- `class="dataframe table table-striped table-sm"` — two tables from
  `DataFrame.to_html`
- `class="table mne-repr-table"` — MNE's own info tables
- `border="1" cellspacing="0" cellpadding="5" style="…font-size:13px…"` —
  MNE-ICALabel's report table

Three run-naming conventions:

- full BIDS id (`sub-0015_task-thermalactive_run-1`) — `spectra.py:402`,
  `scanner.py:438`, `scanner.py:460`, `continuity.py:269`, `analyzer_qc.py:815`
- `run-1` — marker agreement (`analyzer_qc.py:423`), coverage, blink detection
- bare `1` — the Analyzer QC table

`run_label()` already exists in `style.py:135` with a `bare` option and is used in
figure titles. The five sites above bypass it.

Two tables leak raw dataframe column names into a document whose other tables carry
prose headers with units: `run / status / marker_count / median_bpm /
marker_fraction / recording_coverage` (`analyzer_qc.py:153`) and `recording_id /
r_locked_epoch_count / mne_average_pulse_bpm`. A shared `float_format` in the first
prints a beat count as `520.00`.

### Design

**New module `eeg_pipeline/preprocessing/report/tables.py`.** After this change, no
other report module contains table markup.

```python
class Align(Enum):
    TEXT = "text"
    NUM = "num"

@dataclass(frozen=True)
class Column:
    header: str                 # prose, with units: "Median (bpm)"
    align: Align = Align.NUM
    group: str | None = None    # populates a two-level header

def metric_table(rows: Sequence[tuple[str, str]]) -> str: ...
def grid_table(columns: Sequence[Column], rows: Sequence[Sequence[str]]) -> str: ...
```

Both escape every cell, emit `class="report-table"`, and mark numeric cells
`class="num"`. `grid_table` derives the `rowspan`/`colspan` header from `Column.group`,
which retires the hand-rolled `rowspan='2'` in `spectra.py`.

Alignment is **declared per column, never inferred.** A "right-align everything but
the first column" heuristic gets the ledger's prose `Marked by` column wrong and
`Largest surviving line` (`12.5 at 87.8 Hz (TP9)`) wrong.

**Stylesheet.** Rules go into the existing `REPORT_CSS` in `style.py`, which is
already idempotent across repeated report opens via its sentinel:

- compact cell padding and a hairline row rule
- `.num { text-align: right; font-variant-numeric: tabular-nums; }`

Bootstrap's `table-striped` is deliberately **not** used. Zebra striping is a
light-to-dark neutral ramp, and `style.py` reserves that ramp for pipeline decisions
(`EXCLUDED_COLOR` / `RETAINED_COLOR`). Striping every table would spend the
document's decision ink on rows that decide nothing.

**Run labels.** `grid_table` does not rewrite recording ids; the five offending call
sites pass `run_label(...)` explicitly, as `analyzer_qc.py:423` already does. Putting
the rewrite inside the builder would hide a naming convention in a rendering helper.

**The two `to_html` tables** get explicit `Column` specs with prose headers and
per-column formatters, so a count renders as `520` and a rate as `68.3`.

**MNE-ICALabel's table is dropped** via `organize.drop_replaced_panels`, called from
the module that renders the exclusion ledger. This follows the rule already
established for MNE's ECG panels: a replacement is removed by whoever supplies the
substitute, so running one stage alone never leaves the report with neither. Nothing
is lost — the winning class and its probability are in the ledger, and the full
per-class distribution is drawn as a stacked bar under every component dossier.

### Testing

`tables.py` gets direct unit tests: escaping, alignment classes, grouped headers,
empty-row handling. The 22 call sites are covered by existing per-module report tests
that assert on rendered HTML; their expected strings change, which is also the
mechanism that catches a mis-converted site.

---

## Part 2 — Figure corrections

Twelve edits across nine modules. Only one helper is shared.

### Measurement changes

**1. Volume-locked residual envelope (`scanner.py`).** Baseline-correct each volume
epoch before averaging. The quantity is currently the peak-to-peak range of the
across-channel RMS of an uncorrected volume-locked average, so a slow drift enters it
as directly as gradient residual does. This changes the "Locked residual before/after
ICA (µV p-p)" column.

*Verification, not assumption:* on `sub-0015`, runs 2, 4 and 6 currently show a
monotone decay across the volume and read *worse* after ICA. If that survives
baseline correction, the drift explanation is wrong, those runs carry real residual
that ICA increased, and the panel must say so rather than the number quietly
improving. The outcome is reported either way.

**2. Gradient comb (`scanner.py`).** Harmonics falling inside a configured stopband
are excluded from "Median excess" and "Largest surviving line," and drawn hollow with
a note. At present the deepest excursion in the figure — roughly −25 dB at 60 Hz — is
the notch filter, which reads as the correction having succeeded. A shared
`stopband_harmonics()` predicate serves both the statistic and the marker style.

### Axis honesty

**3. Continuity, lower panel (`continuity.py`).** Clip the y-axis to the
post-settling range; draw the high-pass startup transient as the hatched stub it
already carries. The prose already excludes the first 4.8 s from the largest-excursion
column while the axis still spans it, so the figure and its caption disagree: real
excursions of ±10 dB are compressed into the lower fifth of a 0–25 dB axis.

**4. Component dossier (`band_ica_report.py`).** One colourbar per distinct scale.
The grand average and the condition pair share ±7.7 and currently draw two identical
colourbars.

**5. Cardiac component figure (`ica_cardiac_report.py`).** The ECG median moves off
the component's z axis onto its own strip. It is currently drawn at −4 to −6 in units
of "baseline-standardized amplitude (z)" of a different signal, so its amplitude
carries no meaning.

**6. Sensor spectra (`spectra.py`).** The notch trough falls off the bottom of the
axis without the limit being declared. State it, as `robust_symmetric_limit`'s own
docstring asks callers to.

### Encoding and legibility

**7. Variance stem plot (`summary.py`).** Adopt the excluded/retained status strip
already used by the EOG correlation figure, extracted as `status_strip(ax, statuses)`
in `style.py` — the only shared helper in Part 2. Status is currently encoded as
light-grey versus dark markers at small size, which is the least legible instance of
the grey-ramp convention in the document. The x-label's claim that the order is "not
sorted by variance" is also corrected: the plotted values are monotonically decreasing.

**8. Continuity heatmap (`continuity.py`).** ROI-grouped channel labels with
brackets, replacing 63 individual labels at roughly 4 pt.

**9. ECG heart-rate scatter (`ica_cardiac_report.py`).** Add 0.5× and 2× median
guides, matching the 1.5× guide the RR-interval panel already carries. Run 1's scatter
is visibly bimodal at ~68 and ~35 bpm — the half-rate detection signature — and
nothing currently helps a reader see it.

**10. Beat-train comparison (`analyzer_qc.py`).** A matched/unmatched raster becomes
the primary panel, with bpm traces kept only for runs where both trains exist. Runs
with zero Analyzer markers currently spend a full panel each drawing one lone trace:
two of six panels on this subject.

**11. Split-half figure (`preservation.py`).** Move the caveat annotation ("r is over
all channels × times, not over these two traces") from on top of the traces into the
caption. Mark the alpha peak on the curve, not only in a corner text block.

**12. Continuity event rug (`continuity.py`).** Event onsets, `BAD_*` spans and break
annotations under the time axis, which is what turns "there is a bad stretch at
5.5 min" into "it covers trials 12–18". Absent rather than blank for resting-state
recordings, which have no events to draw.

### Testing

Figure tests assert on artist state rather than pixels, following the existing report
tests. The two measurement changes additionally get value-level tests: a synthetic
epoch set with known drift for item 1, and a comb with a known notched harmonic for
item 2. Item 1's verification against `sub-0015` is a pipeline run, not a unit test.

---

## Sequencing

Part 1 lands first and alone. It touches 22 call sites and rewrites their
expected-HTML assertions; interleaving figure changes into the same diff would make
both hard to review. Part 2's twelve items are independent of one another and can land
in any order.

## Compatibility

Reports built before this change carry different values in two columns of the scanner
section (Part 2, items 1 and 2). Cohort comparisons that span the change must be
rebuilt rather than mixed. No other measurement moves; the table work is presentational
throughout.
