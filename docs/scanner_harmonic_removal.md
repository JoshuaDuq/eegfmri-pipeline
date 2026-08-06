# Removing the room's line comb

**Built:** 2026-07-29. **Companion to:** [`scanner_harmonic_diagnosis.md`](scanner_harmonic_diagnosis.md).
**Code:** `studies/pain_study/analysis/line_comb/removal.py`,
`studies/pain_study/scripts/line_comb/remove.py`.
**Run:** `eeg-pipeline line-comb {benchmark,apply,verify,report}`.
**Configuration:** `line_comb_removal` in `studies/pain_study/scripts/line_comb/config.yaml`.

> **Current status (2026-08-05).** The session-pooled implementation and all earlier
> benchmark tables are superseded. The current implementation is block-adaptive, reads
> BIDS channel types, and gives every -5 to +15 second `Trig_therm/T  1` study interval
> its own immutable, data-local transform and audit. `sub-0000` run 1 was repaired by cropping its second
> scanner-acquisition block and remains eligible; all six correctly trimmed `sub-0001`
> recordings also remain eligible. A fresh 90-recording benchmark is required before
> `apply` will run.

---

## What this removes and why it is worth removing

The diagnosis established that the narrowband contamination surviving in the final EEG is
not scanner-gradient residual but a mains-synchronous comb at multiples of 1.19999 Hz,
plus four lines that drift on their own. Converted to amplitude, across 15 participants:

| | |
|---|---|
| All 50 artifact lines together | **1.52 µV RMS** (0.93–2.12 across participants) |
| Largest single line, 57.22 Hz | 1.45 µV (up to 2.38) |
| Share of broadband 1–95 Hz power | 2.4% |
| Share of **gamma** 30.1–80 Hz power | **34.9%** (14.7–54.7 across participants) |

So this matters for gamma and essentially nothing else: beta carries 0.6%, and delta
through alpha carry none. A third of the gamma power in this dataset is a machine in the
scanner room, and because the coupling is a participant-level property (ICC 0.75) it
enters between-participant comparisons as a systematic offset, not as noise.

> **Correction.** These shares were first reported about a third too high — gamma as 47.4%
> rather than 34.9% — because the estimator dropped the line bins and compared band powers.
> Dropping a bin removes its background along with its line, so ordinary spectrum was being
> counted as contamination: on a flat spectrum, masking a fifth of the bins reads as a fifth
> of the power being artifact when none of it is. The share is now the excess over the local
> background at the line bins, in `harmonic_diagnosis.line_excess_fraction`. Every share in
> this document and in the diagnosis uses the corrected estimator. The conclusions do not
> change — gamma is dominated by the artifact either way — but the numbers do.

---

## Why sinusoidal regression rather than a notch

The lines are narrow enough for sinusoidal regression, but their frequencies are not
stationary over a run. Fitting and subtracting local sinusoids remains preferable to a
bank of broad FIR notches, provided frequency is estimated and applied locally.

The implementation is MNE's `notch_filter(method="spectrum_fit")`, which is Thomson's
multitaper line-removal (the method behind CleanLine): overlap-add windows, a multitaper
estimate of each sinusoid's amplitude and phase, subtraction in the time domain.

It cannot be a stage of MNE-BIDS-Pipeline, whose `notch_freq` is FIR only. The removal
therefore runs on the BIDS runs beforehand and writes a cleaned copy of the dataset;
`paths.bids_root` is then pointed at that copy.

---

## Measuring the frequencies rather than assuming them

Each harmonic contributes its refined peak weighted by its own prominence, and the
fundamental is the weighted least-squares slope through the origin over harmonics 24–79.
Harmonic *k* carries the fundamental's error multiplied by *k*, so fitting all of them at
once determines the fundamental far more sharply than its own bin could.

Each run is estimated in 54-second windows (60 scanner TRs) with 50% overlap. Every window
must independently support at least the configured number of harmonics and a finite,
positive delete-one-harmonic standard error; there is no stationary or session-pooled
fallback. Window-specific targets are filtered separately and reconstructed with
normalized squared-sine overlap-add. The whole-run estimate is retained for provenance,
not used as a substitute for a failed local estimate.

The 54-second duration resolves the 1.2 Hz spacing while the 27-second hop follows the
observed within-run movement. On the eligible validation run, 17 windows spanned 1.11 mHz
in fundamental frequency, large enough to invalidate one constant run or session value.

Recordings are read with `mne_bids.read_raw_bids`, with sidecar/header disagreement set to
raise. This makes `channels.tsv` authoritative: ECG and EOG remain in the written dataset
but neither enters an EEG artifact endpoint nor receives the EEG line-comb transform. Two
apparent focal benchmark failures were ECG lines misclassified by the bare BrainVision
reader; changing the reader fixed the measurement rather than suppressing the auxiliary
signal.

The same raw recording also supplies one 20-second spectrum for every exact study interval
(-5 to +15 seconds around each of the 11 `Trig_therm/T  1` events). The complete recording
is first cleaned with the overlapping 54-second plan. The samples in each exact study
interval are then replaced by an independently fitted 20-second transform; the exact
samples are unblended, and their correction is tapered only into samples outside the
analysis interval to avoid a step discontinuity. A line observed
only outside an interval cannot authorize an isolated or comb-adjacent target inside it.
The validated arithmetic comb remains the physical prior in every interval, with its local
position inherited from the best-overlapping supported 54-second estimate; this is the only
deliberate use of evidence extending beyond the 20-second interval.

Residual refinement is performed once per statistical family. What licenses a second
subtraction is a Thomson multitaper F-test—the same sinusoid test underlying MNE's
spectrum-fit detector—applied to the first pass's own output, which is the raw data with
the already-modelled component accounted for and is how a line hidden under a stronger
neighbour's skirt becomes visible. A frequency the whole array evidences is fitted jointly;
one that fewer than half the channels evidence is subtracted only where it was found. The
search is restricted to ±0.15 Hz of an already authorized artifact target, so it cannot
discover and remove an unrelated oscillation elsewhere in the spectrum.

No detection threshold here is derived from the acceptance tolerances the benchmark
applies. An earlier version selected residual targets by the preservation gate's own
maximum permitted excess, which removed precisely what the gate would flag: the suppression
gates could then only fail where the second subtraction itself fell short, never because a
line had been missed, so they measured the search's stopping rule rather than the method.
Detection and acceptance are now different statistics with separately declared thresholds,
and `tests/analysis/line_comb/test_removal.py` fails if the two objects ever share a field.
The consequence is deliberate: a residual carrying power without being a resolvable
sinusoid—a drifting or nonstationary one—is left in place and reported as a gate failure
instead of being subtracted.

Whether any sinusoid survived is then decided over the cohort, not inside each recording,
for the same reason the seam criterion is. Requiring zero significant residuals within
every recording rejects a clean cohort at the test's own error rate: at a 5% family-wise
rate over ninety recordings the expected number of false failures is four or five, which
is what was measured when it was tried. Each recording instead contributes one probability
—its smallest, corrected for the size of the family it searched—and Benjamini–Hochberg over
those decides whether any recording is genuinely unclean. `apply` refuses if it is, and
refuses outright if the benchmark predates the column.

### What the removal costs at its own targets

Every probe described above sits where nothing is removed, so it measures the transform
away from its targets and cannot report a loss. One further probe sits *on* the targets,
at four positions taken from each recording's own fitted plan, and reports the fraction of
its power that survives. This is a measurement and never a criterion: a narrowband signal
at an artifact frequency is not separable from the artifact, so what it quantifies is the
method's unavoidable cost rather than a defect. Interpreting it requires knowing whether
any neural activity is expected at those frequencies—see the mains-notch collateral
analysis for the equivalent argument about the gamma band.

Continuous refinement is scored on the reconstructed overlap-add output, not on each
window in isolation. A detected residual is routed to every overlapping synthesis window
that contributes to the evidenced samples. Those already-authorized residual frequencies
are removed by a joint sliding sinusoid regression with sub-bin frequency refinement, at
half the main spectrum-fit duration so amplitude and phase can follow the non-stationarity
that survived the first pass. This is the regression principle used by CleanLine, without
letting a blind search remove frequencies outside an authorized artifact neighborhood.

---

## Choosing the parameters by measurement

Three settings decide the outcome, and each was swept rather than defaulted.

**Adaptive estimation window.** The fundamental and target positions are estimated in
54-second windows with 50% overlap. The duration was chosen to resolve the 1.2 Hz spacing
and require at least 20 independently supported harmonics even in the cohort's weakest
window. Short windows do not supply enough frequency information; whole-run or
session-pooled estimates cannot follow the measured within-run motion.

**Sinusoid-fit window.** The subtraction uses a 27-second `spectrum_fit` window. It is
shorter than the estimation window so amplitude and phase can adapt, while its nominal
37 mHz frequency resolution separates the narrow shoulders observed in the real data.

**Multitaper bandwidth**, which sets how many tapers estimate each amplitude. At 0.6 Hz
the estimation band reaches ±0.3 Hz, half the distance to the neighbouring comb line, so
no line's amplitude is estimated from a band containing another. 0.3 and 1.0 performed
comparably; 0.6 was taken because it is the one that provably cannot reach a neighbour.

**Notch width**, which turned out to be the parameter that matters most, and the one that
nearly went wrong. `spectrum_fit` subtracts a sinusoid at *every bin within
`notch_widths`* of a target, not only at the target. Left at MNE's default of `freq/200`
it removes 0.14–0.47 Hz around each of 55 lines — a quarter of 28–95 Hz. The width is
therefore set explicitly, and scaled with frequency because the uncertainty is: a
mains-locked comb moves harmonic *k* by *k* times the fundamental's wander, so the top of
the comb strays about a bin within a single run while the bottom barely moves.

Sweeping the constant on sub-0009 run-1, reading the narrowest setting that still pushes
every line below its own local background:

| Width | Band touched | Median suppression | Worst line remaining | Amplitude removed |
|---|---:|---:|---:|---:|
| `freq/200` (MNE default) | 25.3% | 29.8 dB | −6.6 dB | 2.20 µV |
| `freq/300` | 17.1% | 28.7 dB | −6.1 dB | 2.10 µV |
| **`freq/450`** | **12.1%** | **26.0 dB** | **−4.3 dB** | **2.00 µV** |
| `freq/600` | 8.4% | 20.4 dB | **+4.0 dB** | 1.90 µV |
| fixed 0.10 Hz | 8.8% | 19.0 dB | **+4.1 dB** | 1.92 µV |
| fixed 0.05 Hz | 4.4% | 16.0 dB | **+9.1 dB** | 1.55 µV |

Below `freq/450` the high harmonics escape the window their own wander needs. Note the
last row: a single-bin notch removes 1.55 µV, almost exactly the 1.52 µV the comb carries,
and still leaves the worst line 9 dB up — the artifact is not where a single bin says it
is. `freq/450` is the setting in the configuration, with a 0.05 Hz floor. The current
adaptive plan then expands each comb width by two propagated standard errors. Isolated
lines start at the larger of `freq/450`, 0.05 Hz, and one 27-second spectrum-fit bin;
same-window observed support and statistically evidenced residuals expand only the source
that needs it. Across the earlier 90 fitted continuous plans, base widths
cost at most 13.591% of 28--95 Hz and the uncertainty-expanded widths at most 17.017%.
The base number is retained as a descriptive decomposition, not a gate: a separate 15%
ceiling had no scientific meaning. The sole 18% opportunity-cost criterion now measures
the largest transform that is actually applied, including common and channel-local
exact-study refinements.

## Detecting non-comb lines automatically

There is no static isolated-frequency list. A narrow peak must first clear the line-shape,
prominence, and comb-separation checks. A session target then needs replication in at least
three runs, or strong block evidence in at least two runs. A source confined to one
recording can still be removed, but only in that recording and only in the adaptive windows
that support it: at least one interval must exceed 15 dB and the source must recur in three
non-overlapping evidence intervals. Those intervals may be 54-second continuous windows or
exact thermal-study intervals; overlapping intervals do not count as independent evidence.

A separate evidence path handles a distinct line beside a validated comb harmonic. Such a
candidate must be a local summit in the channel-median EEG spectrum, reach the existing
10 dB prominence floor, be no wider than 0.25 Hz at 3 dB down, fall inside the already
declared ±0.15 Hz residual-responsibility region, and sit outside the parent target's
actual removal support. Proximity to a proven electrical comb, line shape, and spatial
replication jointly identify it as artifact. It receives its own narrow `freq/450` target;
the parent notch is not widened.

Evidence may come from the whole run, a 54-second adaptive window, or a study interval. An
accepted source is routed to every continuous overlap-add window contributing samples to
its evidence interval. Exact study plans are stricter: only evidence from that same exact
interval may add an isolated or comb-adjacent exact-study target. This prevents an untreated
continuous neighbor from reintroducing a line while preventing outside-interval evidence
from changing the frequencies removed from the study samples.

Source clustering also retains a cannot-link constraint: two resolvable summits observed in
the same spectrum cannot be the same drifting source, even when they lie within the measured
0.109 Hz line-claim distance. This prevents two simultaneous electrical lines from being
collapsed into one target. Ordinary and comb-adjacent evidence are reconciled only after the
exact per-window targets and widths are known; a narrow target is omitted only when an
already fitted target's removal support physically covers it.

This recording-local path matters scientifically. It detects strong but intermittent
machine sidebands that a session-recurrence rule misses without granting permission to
remove every isolated spectral maximum. There is no count budget: every source satisfying
the predeclared evidence rules is retained in the immutable plan. Source count remains an
audit quantity, while the removed-band, residual, injected-signal, transient, and
study-window gates decide whether the resulting transform is scientifically acceptable.
The fitted target follows the measured peak position in each supporting window.

---

## The preservation gate

Each benchmarked run receives four off-target sinusoids and a 50 ms 40 Hz burst. The
current decision rules are maxima or explicit preservation quantities; median suppression
is reported but cannot gate because a line beginning below 10 dB cannot lose 10 dB without
mandating a trough.

| Criterion | Current threshold |
|---|---:|
| Worst channel-median residual above a complete target-free matched search | ≤ 1 dB |
| Worst channel × window residual above its matched multiple-search control | ≤ 1 dB |
| Same two residual endpoints in the exact -5/+15 s study windows | ≤ 1 dB each |
| Significant Thomson-F residuals in exact authorized regions | descriptive provenance |
| Seams, cohort randomization test over matched shifted maxima | familywise *p* ≤ 0.05 |
| Injected sinusoids | within ±0.5 dB |
| Off-target spectrum | within ±0.2 dB |
| Study-window injected sinusoids / off-target spectrum | ±0.5 / ±0.2 dB |
| Intrinsic burst-energy retention | 0.85–1.05 |
| Burst shape correlation | ≥ 0.99 |
| Largest continuous or exact channel-level 28–95 Hz cost | ≤ 18%, with half-bin tolerance |

Targeted regression checks cover the four recordings that exposed the superseded method:
`sub-0007` run 5, `sub-0009` run 5, `sub-0010` run 6, and `sub-0015` run 2. All four now
pass the aggregate and focal full-run/study endpoints, signal-preservation checks, and 18%
total-cost criterion. The two runs needing aggregate exact-epoch refinement also have zero
post-clean Thomson-F residuals. These are implementation checks, not a substitute for the
required cohort benchmark.

Two things about this gate are worth stating plainly, because both were mistakes caught
during the work rather than foresight.

*The transient metric was wrong at first.* It compared the burst's window energy before
and after removal — but that window also holds comb lines, and taking those out is the
point of the exercise. On synthetic data where the comb is a large share of the window,
that read as a 38% signal loss that never happened. The metric now compares the recovered
probe against the same probe put through the same removal by itself, which isolates
collateral damage from the loss that removing a frequency must cause. The unavoidable part
is reported separately: a 50 ms burst at 40 Hz spans about nine comb lines and genuinely
loses about 18% of its energy. A longer, more physiological gamma burst loses far less.

*The band-fraction criterion did not exist at first, and its absence hid a real defect.*
The gate measured spectral change only at bins more than 0.4 Hz from any target — which
excluded exactly the bins MNE's default width was emptying. Every other criterion passed
while a quarter of the band was being removed. The gate now measures how much of the band
the removal touches at all. An early 8% limit was incompatible with the measured 12.1%
minimum for suppression; a later 15% base-width limit was redundant and arbitrary. The
retained 18% ceiling is applied to the total actual transform and still rejects MNE's
roughly 25% default-width removal.

---

## Superseded historical cohort result

The numbers in this section describe the old session-pooled transform before the current
adaptive implementation and are retained only as an audit trail. They are not evidence
that the current adaptive transform passes the cohort gate. A fresh benchmark must cover
exactly the 90 current recordings and bind their content digests and fitted-plan digests
before apply is authorized.

All 90 runs, session-pooled frequencies, harmonics 22–79 plus the four isolated lines
(61 targets per run), `freq/450` widths.

| | |
|---|---|
| Median suppression | **23.7 dB** (20.0–26.6 across runs) |
| Median residual prominence, over all lines | **−17.0 dB** |
| Session fundamental | 1.1999827 Hz, SD **115 µHz** across 15 participants |
| Per-run estimates before pooling | SD 180 µHz |
| Band touched | 12.1% of 28–95 Hz |
| Binary round-trip | ≤ 5.7 × 10⁻⁸ relative, every run |

### What each band gained

Measured on the continuous BIDS runs, one per participant, as excess above the local
background. "Bins removed" is the share of the band the removal touches at all.

| Band | Lines | Artifact before | Artifact after | Bins removed |
|---|---:|---:|---:|---:|
| delta 1–4 | 0 | 0.00% | 0.00% | 0% |
| theta 4–8 | 0 | 0.00% | 0.00% | 0% |
| alpha 8–13 | 0 | 0.00% | 0.00% | 0% |
| beta 13–30 | 2 | 1.18% | **0.20%** | 0.7% |
| gamma 30–45 | 10 | 9.19% | **2.57%** | 5.6% |
| gamma 45–58 | 10 | 43.35% | **4.28%** | 8.8% |
| gamma 62–95 | 25 | 30.49% | **2.11%** | 13.8% |
| gamma 30–95 | 48 | 26.59% | **3.11%** | 10.7% |

Gamma is where the work was needed and where it paid: 45–58 Hz falls from 43% artifact to
4%, and 62–95 Hz from 30% to 2%. Below 30 Hz there was almost nothing to remove and almost
nothing was touched.

### The blind test

Re-running the diagnosis's own detector — an FDR-controlled sweep of 3–95 Hz that knows
nothing about where the lines are — on one run per participant:

| Stage | Lines detected at *q* < 0.05 | On the comb | Max prominence |
|---|---:|---:|---:|
| Original | **82** | 44 | 16.8 dB |
| Cleaned | **0** | 0 | — |

Nothing is detectable anywhere in the band afterwards. That is the level at which the
contamination mattered: it was a confound because it was consistent across participants,
and no line is now consistent enough to be found.

### What is left, and where

Four of the 50 lines still exceed 1 dB in at least one participant:

| Hz | Median residual | Worst participant | Participants above 1 dB |
|---:|---:|---:|---:|
| **57.2247** | +2.6 dB | **+18.3 dB** | 8 / 15 |
| 43.2029 | −11.6 dB | +14.0 dB | 1 / 15 |
| 86.3970 | −18.7 dB | +5.6 dB | 1 / 15 |
| 47.0362 | −7.8 dB | +2.0 dB | 4 / 15 |

Per run, 25 of 90 keep at least one line above their own background and 11 keep one above
6 dB. **57.2 Hz is the recurring one**, and the reason is that it is the largest line in
the dataset to begin with — up to 26 dB in a single run — so a uniform ~24 dB of suppression
still leaves several decibels standing.

It is not the pooling. The isolated lines do drift within a session, and monotonically:
sub-0000's 47.04 Hz climbs from 46.9656 to 46.9839 Hz across runs 1 to 6, and its second
harmonic at 94.07 Hz climbs exactly twice as far (35.0 against 18.3 mHz), which is what a
single source warming up over an hour looks like. But that drift is at most 46 mHz, well
inside the ±64 mHz the notch spans at 57 Hz, so the target was never off the line.

The practical consequence is confined to 55–59 Hz: a residual there is participant-specific
and can reach 18 dB in one run. Anyone analysing that neighbourhood specifically should read
`per_line_residual.tsv` rather than assume it is clean.

**Part of that 55–59 Hz residual turned out to be two lines nobody was aiming at.** Read on
sub-0012's cleaned data on 2026-07-31 — the first participant to be measured after the
removal rather than before it — two peaks stood at 57.40 and 58.33 Hz, 9.8 and 10.0 dB above
background and carrying 0.31 and 0.25 µV, the two largest narrowband residuals left anywhere
in the analysis band. Both are 0.05–0.07 Hz wide, which is the spectral resolution, so
neither is a rhythm. They were invisible before removal because each sits about 0.2 Hz from
a much stronger targeted line whose skirt raised the local background over it; taking the
strong line out is what exposed them. They are now targeted as 57.3485 and 58.3442 Hz.

Reading them **before** removal is what the cohort counts rest on, so those counts are
weaker than they look: 12 of 15 participants for 57.3485 and 7 of 15 for 58.3442, measured
in data where the neighbouring line is still present and could account for part of a
detection. Only sub-0012 has been measured with the neighbour gone. The lines are not free
to add, either — 57.2247 and 57.3485 are 0.124 Hz apart, closer than the search half-width,
so their windows overlap and the stronger line was being handed to both nominals. Each line
is now claimed once, strongest first, which is what lets the pair be separated at all.

## Running it

```bash
eeg-pipeline line-comb benchmark
```

```bash
eeg-pipeline line-comb apply
```

Benchmark first. A failure means the settings are wrong for the data in front of you, not
that the criteria should move.

`apply` writes `bids_output/eeg_linecleaned`, mirroring every sidecar byte-for-byte and
rewriting only the `.eeg` binaries. Sampling rate, channel set, length and annotations are
untouched, and each written binary is read back and compared against what was intended
before the run is accepted. Verified through `read_raw_bids`, the cleaned dataset returns
identical `Volume/V 1`, `Pulse Artifact/R` and `Trig_therm/T 1` annotations. Expect about
two minutes per run.

To use it, point the pipeline at the cleaned root:

```yaml
paths:
  bids_root: "…/bids_output/eeg_linecleaned"
```

Outputs land in `outputs/line_comb_removal/`: `benchmark.tsv` with every gate metric per
run, and `removal_manifest.tsv` with every run's window-specific estimates, automatic
targets, suppression achieved, content identity, fitted-plan identity, and round-trip
deviation. Benchmark and apply are bound to those identities; a changed input, setting, or
implementation invalidates authorization instead of silently reusing it.

---

## Which frequency bands are usable

Three hard edges bound any band definition on this dataset, independently of the removal:

- **59.5–60.5 Hz** is a −54.9 dB mains notch applied by the pipeline. Any band spanning it
  contains a hole.
- **Above 99 Hz the comb is still there.** ~~The removal ceiling is 95 Hz~~ — that ceiling
  was raised to 99 Hz on 2026-07-31, after sub-0012's cleaned data showed harmonics 80–82
  (96.0, 97.2, 98.4 Hz) standing 7.0–9.2 dB above background having never been targeted.
  97.2 Hz carries the comb in all 15 participants with no measurable frequency scatter.
  These lie above every band this study analyses, so removing them is hygiene rather than
  a result. Harmonic 83 at 99.6 Hz is still untouched: the background estimator needs
  ±4.6 Hz of valid support and there it runs into the 100 Hz low-pass.
- **Below 26 Hz nothing was ever contaminated**, and the removal does not reach there.

On that basis, after the cleaned dataset has been through MNE-BIDS-Pipeline:

| Band | Recommended range | Artifact after removal | Note |
|---|---|---:|---|
| delta | 1–4 Hz | 0.00% | untouched by the removal |
| theta | 4–8 Hz | 0.00% | untouched |
| alpha | 8–13 Hz | 0.00% | untouched |
| **beta** | **13–30 Hz** | 0.20% | now continuous; see below |
| gamma low | 30–45 Hz | 2.6% | |
| gamma mid | 45–58 Hz | 4.3% | stops below the mains notch |
| gamma high | 62–95 Hz | 2.1% | starts above the notch, ends at the ceiling |

**Beta should become one continuous 13–30 Hz band.** The present configuration splits it
into 13–17.9 and 23.1–30 with a hole cut around 20 Hz, on the assumption of a scanner line
there. The diagnosis found no line anywhere in 13–26 Hz, and the two that do exist — 26.4
and 27.6 Hz — are now removed. The 20 Hz slice-excitation rate is the one place gradient
left any trace at all (within-run phase resultant 0.377 against a 0.27 null) but it
produced no detectable spectral line, so it does not warrant carving the band. Dropping the
exclusion recovers 5 Hz of bandwidth that was being discarded for a reason that no longer
holds.

**The three "clean" gamma windows should be retired.** `gamma_low_clean` (30.1–38),
`gamma_mid_clean` (43–56) and `gamma_high_clean` (67–77) were drawn to dodge a 20 Hz-spaced
scanner comb that does not exist, and together they discard about 25 Hz of usable spectrum
while still admitting 4, 10 and 8 comb lines respectively. The ranges above replace them.

Two caveats that belong in a methods section rather than in a band definition:

- 12.1% of the bins in 28–95 Hz were removed with the lines, so gamma band power is an
  average over the surviving 88%. It is applied identically to every participant, so it
  does not create a between-participant confound, but the absolute value is biased low and
  any genuine narrowband activity at a comb frequency is gone with the artifact.
- A residual near 57.2 Hz remains in some participants, up to 18 dB in the worst run. It
  sits inside the 45–58 Hz window. If that neighbourhood carries the hypothesis, check
  `per_line_residual.tsv` per participant before relying on it.

**The task samples are now measured directly, but they are not the final downstream
epochs.** The benchmark audits the exact -5/+15 second raw intervals that enter the
studies, before MNE-BIDS-Pipeline band-pass, mains notch, resampling, ICA, and epoch
rejection. Re-run downstream preprocessing and its scanner-harmonic QC after applying the
validated continuous transform; do not treat this pre-preprocessing audit as permission to
reuse old derivatives.

## What has to happen afterwards

**The pipeline has to be re-run.** The cleaned dataset only reaches the analyses through
MNE-BIDS-Pipeline, and that means PyPREP, ICA and epoching again. ICA is fitted on data
that no longer contains the comb, so the decomposition will differ and any recorded manual
component decisions must be reviewed against the new components rather than carried over.

**Existing gamma results have to be re-derived, not adjusted.** Roughly half of the
measured 30–80 Hz power in the current derivatives is this artifact, and it is
participant-specific, so it is correlated with whatever else varies between participants —
cap fit, head geometry. There is no correction factor; the analyses have to be run again on
the cleaned data.

**All downstream derivatives still need regeneration.** Most contamination is in gamma,
but the adaptive transform changes the continuous input and includes documented isolated
targets below 30 Hz. Do not carry old ICA, epoch, or feature outputs forward selectively.

---

## Limitations

- The current adaptive gate has four targeted full-channel real-data regression checks. The
  required 90-recording benchmark has not yet been run, and apply will refuse to proceed
  until every eligible recording passes with the same settings, source hash, input digest,
  and fitted-plan digest.
- The total 28–95 Hz cost varies with target evidence and estimator uncertainty. The gate
  permits at most 18% for the worst continuous or exact channel-level transform. This is not free: genuine
  activity at an artifact frequency is not identifiable from the artifact and is removed
  with it.
- Frequencies are locally constant within each 54-second estimation window and blended
  across 27-second hops. Faster movement than those windows can resolve remains a limit;
  every window must nevertheless supply its own supported estimate, with no pooled value
  substituted on failure.
- Exact study intervals are temporally isolated for isolated and adjacent-line evidence,
  but not for comb identity: the arithmetic family and local fundamental come from the
  best-overlapping supported 54-second estimate. This improves 20-second frequency
  localization but assumes that the already validated room comb remains the same physical
  source across the overlap. Exact residual searches and signal-preservation gates test the
  consequence on the study samples themselves.
- Short broadband transients overlapping the comb lose real energy — about 18% for a 50 ms
  burst at 40 Hz. Longer events lose proportionally less.
- Isolated-line origins remain unidentified. Session-level lines require run replication;
  strong recording-specific lines require independent temporal replication and are removed
  only where supported. A new line family outside those physical constraints correctly
  fails the evidence or residual gate instead of being silently added to a manual list.
- Automatic detection does not make artifact identity omniscient. A genuine neural
  oscillation exactly coincident with a narrow electrical line is not identifiable from
  EEG alone and will be subtracted with that line. Conversely, diffuse muscle, ocular,
  pulse, gradient, and broadband transient artifacts are outside this tool's narrow-line
  model and require their own automated stages and QC.
- The software is reusable, but the supplied defaults are not site-universal. The 1.2 Hz
  nominal, harmonic ranges, and evidence calibration describe this scanner room. A public
  deployment at another site must diagnose and configure its own physical comb; failure to
  support at least 20 harmonics surfaces as an error rather than silently applying this one.
- Nothing here addresses the source. Pausing the cold head during acquisition, and lead
  management to reduce the pickup loop, remain the only fixes that would stop the artifact
  reaching the amplifier at all.
