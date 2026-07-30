# Removing the room's line comb

**Built:** 2026-07-29. **Companion to:** [`scanner_harmonic_diagnosis.md`](scanner_harmonic_diagnosis.md).
**Code:** `studies/pain_study/analysis/line_removal.py`,
`studies/pain_study/scripts/remove_line_comb.py`.
**Configuration:** `preprocessing.line_comb_removal` in `eeg_pipeline/utils/config/eeg_config.yaml`.

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

The lines are monochromatic (0.109 Hz half-power width, at the measurement floor),
stationary in frequency, and stable in amplitude within a session. That is exactly the
case where fitting and subtracting a sinusoid beats filtering it out: the fit removes the
line and returns the rest of the band untouched, where fifty narrow notches would take the
band with them and ring.

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

**Estimates are pooled over a session, not used per run.** This was measured, not assumed:

| | |
|---|---|
| Per-run estimate scatter, within one session | 124 µHz (sub-0009), 277 µHz (sub-0000) |
| Per-run scatter across the benchmark sample | 399 µHz |
| True between-session variability (from the diagnosis) | **61 µHz** |

The per-run estimate is therefore mostly measurement error — it scatters four to six times
more than the quantity it is tracking. Taking the median over a session's six runs removes
most of that while still allowing one session to differ from the next. It also fixes a
concrete failure: sub-0000 run-1 estimated 1.199340 Hz, 660 µHz below its own session
median of 1.199982, and left a line 1.3 dB above background; with the session estimate the
same run leaves it at −1.1 dB.

---

## Choosing the parameters by measurement

Three settings decide the outcome, and each was swept rather than defaulted.

**Window length.** Four-second windows fail outright — they cannot resolve a 1.2 Hz
spacing, and leave lines up to 11 dB above background. Ten and twenty seconds both work;
20 s was taken for the smaller collateral change.

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
is. `freq/450` is the setting in the configuration, with a floor of 0.05 Hz (one bin at
20 s) for the lowest harmonics.

---

## The preservation gate

Criteria fixed before the measurement, mirroring the ones the earlier residual-OBS
benchmark used so the two decisions stay comparable. Each benchmarked run has probes
injected — four sinusoids clear of every target, and a 50 ms 40 Hz burst — then the lines
are removed and the probes are measured.

| Criterion | Threshold | Result across 5 runs |
|---|---|---|
| Median residual prominence | ≤ 1 dB | −20.9 to −16.4 dB |
| Median suppression | ≥ 10 dB | 19.9 to 25.7 dB |
| Injected sinusoids | within ±0.5 dB | 0.000 dB |
| Untouched spectrum | within ±0.2 dB | ≤ 0.001 dB |
| Transient energy | within ±5% | ratio 1.000 |
| Transient shape | *r* ≥ 0.99 | 1.000 |
| Band touched | ≤ 15% | 12.1% |

**5 of 5 runs passed every criterion.**

One gap in this gate is worth naming: it tests the *median* residual across a run's lines,
not the worst one. Applied to the cohort that turned out to matter — see below.

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
the removal touches at all, and the threshold was revised from 8% to 15% once the wander
physics showed that 12.1% is the floor for full suppression. The revision is recorded in
the code rather than quietly applied.

---

## Cohort result

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

## Running it

```bash
python -m studies.pain_study.scripts.remove_line_comb --stage benchmark --limit 5
```

```bash
python -m studies.pain_study.scripts.remove_line_comb --stage apply
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
run, and `removal_manifest.tsv` with each run's estimated frequencies, session-pooled
fundamental, suppression achieved and round-trip deviation.

---

## Which frequency bands are usable

Three hard edges bound any band definition on this dataset, independently of the removal:

- **59.5–60.5 Hz** is a −54.9 dB mains notch applied by the pipeline. Any band spanning it
  contains a hole.
- **Above 95 Hz the comb is still there.** The removal ceiling is 95 Hz because the
  background estimator needs ±4.6 Hz of valid support and beyond that it runs into the
  100 Hz low-pass. Harmonics 80–83 sit at 96.0, 97.2, 98.4 and 99.6 Hz with 5.3–9.1 dB
  prominence, untouched.
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

**This is a projection, not a measurement of the final epochs.** The numbers above come
from the cleaned continuous BIDS runs; MNE-BIDS-Pipeline has not yet been re-run on them.
Its remaining operations — band-pass, mains notch, resampling, ICA, epoching — do not
reintroduce narrowband lines, and the blind detector finding nothing in the cleaned
continuous data is the strongest available evidence short of producing the epochs. Confirm
by re-running `diagnose_scanner_harmonics --stage all` against the new derivatives once
they exist.

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

**Nothing below 30 Hz needs revisiting.** Delta through beta carry 0–2.2% of this
contamination and the removal touches nothing below 28.8 Hz.

---

## Limitations

- The injected-probe gate was run on five runs spanning the cohort, not on all ninety.
  Per-run suppression is recorded for every run in the manifest, but signal preservation
  was verified on the sample.
- The gate tests the median residual across a run's lines, not the worst. Twenty-nine of
  90 runs keep at least one line above their own background; no cohort-level line survives
  detection, but a single-run analysis should consult the manifest.
- 12.1% of 28–95 Hz is removed. That is the price of taking out 55 wandering lines, and it
  is lower than masking them in analysis would cost (about 22%), but it is not free: any
  genuine narrowband activity at a comb frequency goes with the artifact, and there is no
  way to tell the two apart at the same frequency.
- Frequencies are pooled per session. A source that drifted materially within a single run
  would be tracked less well than one that does not; the measured within-session scatter
  says that is not happening here, but it is an assumption the design makes.
- Short broadband transients overlapping the comb lose real energy — about 18% for a 50 ms
  burst at 40 Hz. Longer events lose proportionally less.
- The four isolated lines are removed at their measured positions, but their origin is
  still unidentified, and a source that can drift 190 mHz between sessions could in
  principle move further in a session not yet recorded.
- Nothing here addresses the source. Pausing the cold head during acquisition, and lead
  management to reduce the pickup loop, remain the only fixes that would stop the artifact
  reaching the amplifier at all.
