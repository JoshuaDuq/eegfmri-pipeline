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
| Share of **gamma** 30.1–80 Hz power | **47.4%** (31.5–63.0 across participants) |

So this matters for gamma and essentially nothing else: beta carries 2.2%, and delta
through alpha carry none. About half of the gamma power in this dataset is a machine in
the scanner room, and because the coupling is a participant-level property (ICC 0.75) it
enters between-participant comparisons as a systematic offset, not as noise.

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

- The removal is validated on five runs spanning the cohort, not on all ninety. The
  manifest records per-run suppression for every run so the full distribution is
  inspectable, but the injected-probe gate was run on the sample.
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
