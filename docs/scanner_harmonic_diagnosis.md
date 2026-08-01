# Scanner harmonics across participants: the residual comb is not the scanner

**Investigated:** 2026-07-29.
**Dataset:** `task-thermalactive`, 15 participants, 90 runs, plus 14 baseline recordings.
**Stage examined:** final cleaned epochs, `derivatives/preprocessed/eeg/sub-*/eeg/sub-*_task-thermalactive_epo.fif`.
**Control:** gradient-free EEG at the head of all 104 source recordings.
**Reproduce:** `eeg-pipeline line-comb diagnose`, then `eeg-pipeline line-comb plot`
(`studies/pain_study/scripts/line_comb/`).
**Removal:** [`scanner_harmonic_removal.md`](scanner_harmonic_removal.md) acts on what this
document measures.

---

## Conclusion

**The narrowband contamination left in the final EEG is not scanner-gradient harmonics.**
It is a comb of lines at integer multiples of **1.19999 Hz** — 72.0 cycles per minute,
exactly one fiftieth of the 60 Hz mains — spanning harmonics 24 to 79, from 28.8 to
94.8 Hz. Four further narrow lines sit off that comb, the largest and strongest line in
the entire spectrum being 57.22 Hz.

Three independent measurements agree:

```
the volume comb k/TR is gone
  -> at 66 volume-harmonic positions, cohort prominence never exceeds -0.48 dB
  -> no epoch phase-locks to the volume grid, not even at 20.0, 41.1, 61.1, 82.2 Hz
the surviving lines are present when the gradients are off
  -> 43 of 50 keep a prominence whose CI excludes zero in 731 s of EEG recorded
     before the scanner played a single gradient pulse
the surviving lines are a machine, not a person
  -> the comb fundamental is 1.199998 Hz +/- 61 uHz across 15 sessions and 5 months
  -> relative scatter 5.1e-05; individual lines stable to 5-6 mHz
```

The gradient correction is therefore working. What earlier reviews recorded as "residual
scanner harmonics" is an environmental line comb that the gradient correction could never
have touched, because it was never gradient-locked. Suppressing it needs a different
mechanism, and the numbers below say which analyses it actually reaches — essentially only
gamma.

This supersedes the attribution in
[`studies/pain_study/SCANNER_HARMONICS_QC_README.md`](../studies/pain_study/SCANNER_HARMONICS_QC_README.md).
That review measured the residual lines correctly but assumed they were gradient residual;
it had no gradients-off condition to test the assumption against. Its frequency list
(37.17, 38.39, 51.57, 52.80, 57.19, 61.10, 83.98 Hz) is reproduced here to within tens of
millihertz — and all of it except 57.19 Hz consists of multiples of 1.2 Hz.

---

## 1. Data and provenance

The chain that produced the analysed epochs, verified byte-for-byte at the Analyzer step:

| Stage | Artifact |
|---|---|
| Acquisition | BrainVision 5 kHz, 64 channels (63 EEG + ECG), SyncBox locked (`SyncStatus, Sync On` every 2 s) |
| Trim | to first and last `Volume` marker (`original_trimmed_5khz`) |
| Analyzer | MR Correction + Cardioballistic Correction, pulse template 0–60 s, 30–115 bpm, output 1 kHz |
| BIDS | `pybv`, 1 kHz, volume and R annotations preserved |
| MNE-BIDS-Pipeline | 0.1–100 Hz band-pass, 60 Hz notch, resample to 500 Hz, PyPREP bad channels, ICA |
| Analysed here | 66 epochs per participant (11 trials × 6 runs), −7 to +15 s, 500 Hz |

The `…run1_sub0005….eeg` this diagnosis read was byte-identical to the same file under the
batch now named `reference_bcg_pre_recovery/` (called
`processed_trimmed_0-60s_30-115bpm_marker_template/` when this was written), and the BIDS
`acq_time` (2026-05-13T11:01:58.685017Z) matches that file's `New Segment` timestamp. The
epochs therefore descend from the re-exported *trimmed* Analyzer output described in
[`pulse_artifact_correction_recovery.md`](pulse_artifact_correction_recovery.md), not from
the native 5 kHz correction workflow.

That is the *pre-recovery* correction. The source tree has since moved on to
`step3_bcg_corrected/`, which corrects the BCG at the recovered beats as well; the comb
measurements were re-run against it on 2026-07-31 and did not move (fundamental unchanged,
line prominence +0.01 dB per run), which is what a room artifact should do.

Acquisition is identical across all 90 runs: TR = 0.900 s (exactly 4500 samples at 5 kHz),
54 slices, multiband 3, 545 volumes per run. Sessions span 2026-02-09 to 2026-07-13, one
session per participant.

Run identity for the concatenated epochs is reconstructed from the cumulative lengths of
the per-run `*_proc-filt_raw.fif` files. The partition is verified, not assumed: every
participant yields exactly 11 epochs in each of 6 runs, and the analysis aborts rather
than guessing if it ever does not.

---

## 2. Method

### 2.1 TR-commensurate frequency grids

Every segment holds a whole number of TRs. This puts each volume-comb line *k*/TR exactly
on a DFT bin centre, so "on the scanner comb" becomes an exact statement rather than a
nearest-bin approximation, and scanner lines suffer no scalloping loss. Two grids are used:

| Grid | Segment | Δf | Source |
|---|---|---|---|
| Catalogue | 24 TR = 21.6 s | 0.046296 Hz | final epochs, 66 segments per participant |
| Matched | 4 TR = 3.6 s | 0.185185 Hz | final epochs **and** gradient-free control |

The matched grid is an exact sub-grid of the catalogue grid (every 4th bin), so the two
compare bin for bin. It exists because line prominence grows with segment length:
comparing a 21.6 s catalogue against a 3.6 s control would report a difference that is a
property of the window, not of the gradients.

Hann window; power averaged across segments; median across good EEG channels (the
pipeline's own bad-channel list is excluded); participant is the unit of inference.

### 2.2 Prominence, not absolute power

A line means nothing except against the background beside it. Local background is a running
median over ±4.6296 Hz (±100 catalogue bins, ±25 matched bins) with the central 3 bins
excluded; prominence is the bin minus that background. The window is wide enough that the
few line bins inside it cannot move the median. Under a convex 1/f background a symmetric
median sits slightly *above* the centre bin, so the estimator is conservative.

### 2.3 Detection

Prominence is computed per participant, then averaged across the 15. The null is fitted to
the lower tail of that cohort statistic across all bins in 3–95 Hz (excluding 59.5–60.5 Hz,
the notch): for a Gaussian the gap between the median and the 15.87th percentile is one
sigma, and reading the scale from the lower half stops the lines themselves inflating it.
One-sided p-values then go through Benjamini–Hochberg at *q* < 0.05. Runs of significant
bins are collapsed to their largest bin and the centre frequency is refined by quadratic
interpolation. Confidence intervals are percentile bootstraps over participants
(10 000 resamples, seed 42).

No dB threshold is invented anywhere. **72 features** survive FDR control.

### 2.4 Classifying the 72 detections

Comb membership is decided first, and arithmetically: a detection within 0.06 Hz of an
integer multiple of the fitted fundamental is a member. **52 of 72** are, against a chance
rate of 10% — binomial *p* = 4 × 10⁻³⁶.

Half-power linewidth then separates instrument lines from brain rhythms among the rest.
Width cannot be the *primary* criterion, because it is measured against the peak's own
height: a weak line reaches the half-power point further out and measures wider than a
strong line from the same source. Among comb members the two are correlated at Spearman
ρ = −0.87:

| Prominence | n | Median width (window widths) |
|---|---:|---:|
| 8–20 dB | 13 | 1.53 |
| 5–8 dB | 15 | 1.70 |
| 3–5 dB | 12 | 1.96 |
| 0–3 dB | 12 | 10.37 |

A Hann-windowed 21.6 s segment cannot render a pure tone narrower than
1.4382 / 21.6 = 0.0666 Hz, so anything near 1.0–2.0 window widths is monochromatic to the
limit of measurement. The resulting four classes:

| Class | n | Frequency span | Prominence | Width | Detected in | ICC | Topography *r* |
|---|---:|---|---:|---:|---:|---:|---:|
| `comb` | 46 | 27.6–94.8 Hz | 6.28 dB | 1.69× | 14/15 | 0.76 | 0.25 |
| `comb_wide` | 6 | 13.2–67.2 Hz | 1.73 dB | 14.95× | 4.5/15 | 0.72 | 0.12 |
| `isolated` | 4 | 47.0–94.1 Hz | 8.12 dB | 1.86× | 14/15 | 0.92 | 0.25 |
| `other` | 16 | 9.0–90.1 Hz | 1.68 dB | 15.71× | 3.5/15 | 0.41 | 0.10 |

`other` is 16 broad, weak, low-prevalence features concentrated in the alpha range with
near-zero topographic agreement between participants: brain rhythms, and excluded from
every artifact statistic below.

`comb_wide` is the one genuinely ambiguous group — six weak detections that land on comb
positions but measure a rhythm's width. Five of the six sit within 9 mHz of a harmonic,
which is far too precise to be coincidence, so they are counted as members; but because a
rhythm coinciding with a comb position cannot be excluded for any one of them individually,
they are left out of the band-power masking, where wrongly masking a rhythm would overstate
contamination. The **artifact set used for band impact is therefore 50 lines**
(`comb` + `isolated`).

### 2.5 The gradients-off control

The scanner plays roughly ten dummy volumes before it emits the first `Volume` marker: the
median gap between measured gradient onset and the first marker is **8.75 s**. The marker
is therefore useless for locating gradient-free data, and onset is found from the signal
instead. Per-block (0.25 s) standard deviation is computed on mean-removed data — these
recordings are DC-coupled with channel offsets of hundreds of microvolts, so a raw RMS
measures the offset and misses the gradient entirely — and the threshold sits at the
geometric mean of the profile's smallest and largest block. The transition is a factor of
~40 (14 µV to 600–750 µV), so the threshold choice does not matter.

Each recording contributes the span from 0.5 s after start to 0.25 s before onset, tiled
with 3.6 s blocks at 50% overlap.

| | |
|---|---|
| Recordings probed | 104 (90 task, 14 baseline) |
| Contributing ≥1 segment | 85, covering all 15 participants |
| Segments | 251 |
| Gradient-free EEG total | 731 s |
| Window length | median 6.25 s, range 3.75–44.0 s |
| Amplitude in window | median 14.1 µV SD — ordinary EEG |

This control is uncorrected 5 kHz data. It establishes **whether a line exists without
gradients**, not what its amplitude would be after processing.

### 2.6 Attribution tests

- **Volume-phase locking.** Each epoch's metadata carries the latency of the first `Volume`
  marker inside it, which fixes the epoch's phase in the scanner's own frame. Rotating each
  epoch's DFT coefficient by that offset and testing phase concentration (Rayleigh)
  separates a deterministic volume-locked residual from a narrowband oscillation that
  merely sits at the same frequency. Reported both pooled over 66 epochs and averaged
  within run, since each run has its own sequence start and a per-run phase offset would
  dilute the pooled figure. The test is interpretable only at frequencies completing a
  whole number of cycles per TR; off-comb rows are the negative control.
- **Cross-session frequency stability.** Per-participant centre frequency by quadratic
  interpolation, then the spread across 15 sessions spanning five months.
- **Topography.** Per-channel prominence per participant, and the mean pairwise spatial
  correlation between participants.
- **Cardiac.** R peaks from the retained ECG lead, per run.
- **Variance components.** Balanced one-way random effects on run-level prominence,
  15 participants × 6 runs, giving the between-participant share (ICC).

---

## 3. The scanner comb is absent

At the 66 volume-comb harmonic positions between 3.3 and 94.4 Hz that do **not** coincide
with a 1.2 Hz line:

| | |
|---|---|
| Cohort mean prominence | median **−5.90 dB** |
| Largest at any position | **−0.48 dB** (82.22 Hz) |
| Positions above +1 dB | **0 / 66** |

The values are negative because the local background is lifted by the 1.2 Hz comb around
them while the volume-comb positions themselves are empty. At the four references the
earlier review named:

| Harmonic *k* | Hz | Cohort median prominence | 95% CI | Participants > 1 dB |
|---:|---:|---:|---|---:|
| 18 | 20.0000 | −3.75 dB | [−4.33, −1.87] | 0/15 |
| 37 | 41.1111 | −3.55 dB | [−4.23, −2.98] | 0/15 |
| 55 | 61.1111 | +0.37 dB | [−0.49, +2.23] | 5/15 |
| 74 | 82.2222 | −0.68 dB | [−1.56, +0.73] | 4/15 |
| 54 | 60.0000 | −54.90 dB | [−56.53, −53.00] | 0/15 |

The small positive value at *k* = 55 is the shoulder of the 1.2 Hz line at 61.196 Hz,
85 mHz away, not a volume-comb residual. Note also that with TR = 0.9 s the volume comb
contains 60.000 Hz exactly (*k* = 54), so scanner harmonic 54 and the mains line are
inseparable by frequency; that position is the −54.9 dB notch.

**Phase locking confirms it.** Within-run resultant length, where 11 epochs put the null
expectation near 0.27:

| Probe | Pooled *R* | Within-run *R* |
|---|---:|---:|
| 20.0000 Hz (*k* = 18) | 0.186 | 0.377 |
| 41.1111 Hz (*k* = 37) | 0.110 | 0.275 |
| 61.1111 Hz (*k* = 55) | 0.123 | 0.280 |
| 82.2222 Hz (*k* = 74) | 0.122 | 0.274 |
| All 72 detected lines | 0.104 (median) | 0.270 (median) |

Nothing is locked to the volume grid. The mild elevation at 20 Hz — the slice-excitation
rate, 18 excitations per 0.9 s TR — is the only trace of the gradient anywhere in the final
data, and it is small.

---

## 4. What is actually there: a 1.2 Hz comb

### 4.1 The comb

Pairwise gaps between the narrow lines pile up at exact multiples of 1.2 Hz. Fitting
integer multiples of a single fundamental through the origin:

| | |
|---|---|
| Fundamental | **1.19999 Hz** |
| Members | 52 of 72 detections (chance rate 10%, binomial *p* = 4 × 10⁻³⁶) |
| Lines used for the fit | 41 |
| Harmonics spanned | 24 – 79 (28.8 – 94.8 Hz) |
| RMS residual | **6.6 mHz** (0.14 catalogue bins) |
| Largest residual | 26 mHz |
| Rayleigh *p* for volume-comb membership | 0.78 (i.e. unrelated to *k*/TR) |

Estimated independently in each participant, from the 46 well-resolved members:

| | |
|---|---|
| Mean over 15 sessions | 1.199998 Hz |
| SD | **61 µHz** (relative 5.1 × 10⁻⁵) |
| Range | 1.199910 – 1.200110 Hz |
| In cycles per minute | 71.9999 |
| 60 Hz ÷ fundamental | **50.0001** |

Fifteen participants measured over five months give the same number to five significant
figures, and that number is the mains frequency divided by exactly 50.

### 4.2 Line-by-line properties

Strongest twelve comb members:

| Hz | Harmonic | Cohort median prominence | 95% CI | n detected | Gradients-off | Freq. SD | ICC |
|---:|---:|---:|---|---:|---:|---:|---:|
| 54.0005 | 45 | 14.12 dB | [11.48, 16.68] | 15/15 | 5.42 dB | 2.9 mHz | 0.68 |
| 58.8004 | 49 | 12.12 | [10.71, 12.82] | 15/15 | 3.64 | 1.9 | 0.68 |
| 51.6015 | 43 | 11.32 | [8.86, 13.55] | 15/15 | 4.30 | 1.6 | 0.76 |
| 73.2015 | 61 | 11.18 | [9.30, 13.07] | 15/15 | 5.45 | 6.2 | 0.82 |
| 61.1958 | 51 | 10.17 | [8.52, 11.41] | 15/15 | 3.62 | 2.7 | 0.71 |
| 87.5971 | 73 | 9.80 | [5.16, 11.64] | 15/15 | 5.18 | 4.3 | 0.77 |
| 56.3981 | 47 | 9.45 | [7.47, 10.50] | 15/15 | 2.58 | 2.6 | 0.69 |
| 85.1987 | 71 | 8.84 | [5.92, 11.24] | 15/15 | 4.89 | 6.2 | 0.77 |
| 82.8027 | 69 | 8.76 | [7.82, 10.93] | 15/15 | 4.77 | 5.8 | 0.68 |
| 92.4022 | 77 | 8.47 | [5.47, 10.68] | 14/15 | 3.18 | 3.7 | 0.72 |
| 77.9969 | 65 | 8.32 | [5.31, 8.95] | 15/15 | 3.01 | 3.3 | 0.78 |
| 80.4008 | 67 | 8.28 | [5.60, 10.34] | 15/15 | 3.92 | 5.1 | 0.76 |

Across the 46 `comb` lines: prominence median 6.28 dB (1.87–14.12); detected in a median of
14 of 15 participants; **39 of 46** keep a gradients-off prominence whose bootstrap CI
excludes zero; per-participant frequency SD median 5.6 mHz.

### 4.3 Four lines that do not join the comb

| Hz | Prominence | 95% CI | n detected | Gradients-off | Freq. SD | Drift over study | ICC |
|---:|---:|---|---:|---:|---:|---:|---:|
| **57.2247** | **18.18 dB** | [14.42, 22.90] | 15/15 | 15.00 dB [8.61, …] | **56.3 mHz** | −0.14 Hz | 0.92 |
| 58.1807 | 9.73 | [9.35, 11.40] | 15/15 | 4.38 [2.79, …] | 17.2 mHz | −0.02 Hz | 0.78 |
| 47.0362 | 6.51 | [0.10, 11.91] | 13/15 | 4.05 [3.36, …] | 57.3 mHz | +0.12 Hz | 0.93 |
| 94.0748 | 5.98 | [0.93, 11.18] | 10/15 | 6.62 [4.18, …] | 59.4 mHz | +0.11 Hz | 0.95 |

94.0748 = 2 × 47.0374, so those two are one source and its second harmonic. All four are
narrow (0.09–0.15 Hz), all four survive with the gradients off, and 57.2247 Hz is the single
largest line in the whole spectrum — larger than anything on the 1.2 Hz comb.

Their frequency behaviour separates them from the comb decisively:

| Participant | Session | 57.22 Hz | 47.04 Hz | 54.00 Hz | 73.20 Hz |
|---|---|---:|---:|---:|---:|
| sub-0000 | 2026-02-09 | 57.3025 | 46.9763 | 53.9993 | 73.1994 |
| sub-0001 | 2026-03-02 | 57.3201 | 46.9444 | 54.0017 | 73.2022 |
| sub-0005 | 2026-05-13 | 57.2405 | 47.0468 | 54.0053 | 73.2120 |
| sub-0009 | 2026-06-22 | 57.1793 | 47.0466 | 53.9975 | 73.1955 |
| sub-0013 | 2026-07-02 | 57.1283 | 47.1157 | 54.0019 | 73.2087 |
| sub-0015 | 2026-07-13 | 57.1822 | 47.0529 | 54.0039 | 73.2100 |
| **SD (n = 15)** | | **56.3 mHz** | **57.3 mHz** | **2.9 mHz** | **6.2 mHz** |
| **Range** | | 192 mHz | 196 mHz | 11 mHz | 20 mHz |

The comb is mains-disciplined; these four are not. 57.22 Hz falls by about 0.14 Hz over the
five months of acquisition, drifting like a free-running motor.

### 4.4 Between-participant structure

Median ICC across the 72 detections is **0.746**: three quarters of the variance in line
prominence is between participants, not between runs within a participant. For 57.2247 Hz
it is 0.92, and across the four isolated lines a median of 0.92.

The lines are at the same frequencies in everyone, but their *amplitude* is a
participant-level property, stable across the six runs of a session. That is what a fixed
external field coupled through a per-session geometry looks like: the source does not
change, the pickup loop does — cap placement, lead dress, head position in the bore.

Topographic agreement between participants is modest (median pairwise *r* = 0.25 for comb
lines and for the isolated lines, against 0.10 for the broad rhythms). Higher than the
rhythms but far from identical, again consistent with a shared source picked up through a
geometry re-established at every setup.

### 4.5 It is not cardiac

| | |
|---|---|
| Heart rate, 90 runs | median 67.0 bpm (1.117 Hz), range 52.8 – 95.1 bpm (0.88 – 1.59 Hz) |
| Comb fundamental | 1.199998 Hz, SD 61 µHz |

The cardiac rate is not 1.2 Hz, and it varies by 80% across runs while the comb fundamental
holds to five significant figures. Ballistocardiogram sidebands cannot produce this comb.

---

## 5. What it is

**Ruled out.** *Gradient artifact*: absent from the volume comb, no volume phase locking,
and the lines are fully present before the scanner starts. *Mains*: 60 Hz is notched to
−54.9 dB, and a comb spaced 1.2 Hz is not mains harmonics. *Cardiac*: rate mismatch and
rate variability, above. *Correction failure*: the control is uncorrected raw data and
shows the same comb, so no step of the Analyzer or MNE chain created it.

**Supported: a mains-synchronous mechanical oscillator in the scanner room at 72.0 rpm,
coupled into the EEG leads by vibration in the static field.** A conductive loop moving in
B₀ generates an EMF proportional to its rate of change of enclosed flux, so a periodic
vibration appears as a harmonic comb of exactly this kind — narrow, rich in high harmonics,
present whenever the magnet is up, and scaled by loop geometry rather than by anything
physiological. 1.199998 Hz is 60 Hz / 50.0001, which is what a mains-driven synchronous
motor turning at 72 rpm gives. The cold head of the cryocooler is the standard occupant of
that description on an MR system, and the helium-pump artifact is a documented EEG-fMRI
contaminant (Nierhaus et al., 2013; Rothlübbers et al., 2015). What is established here is
the *signature* — mains-locked, 1.2 Hz, gradients-independent, vibration-shaped — not the
physical unit. Confirming the cold head specifically would take one recording with the cold
head cycled, which is a facility operation, not an analysis.

The four isolated lines are a second, separate source: equally narrow, equally
gradients-independent, but **not** mains-disciplined — 57.2 Hz wanders by 192 mHz across
sessions and drifts downward over five months. That is a free-running rotating machine (a
fan, pump or chiller) rather than a synchronous one. It is also the largest single line in
the data, at 18.2 dB.

---

## 6. What this reaches

Power that the 50 artifact lines add to each analysis band, integrated in the power domain,
per participant. Rhythms and the six ambiguous wide members are not masked, so this is
contamination only.

| Band | Hz | Artifact lines inside | Median artifact share | Range over 15 participants |
|---|---|---:|---:|---|
| delta | 1.0–3.9 | 0 | 0.00% | — |
| theta | 4.0–7.9 | 0 | 0.00% | — |
| alpha | 8.0–12.9 | 0 | 0.00% | — |
| beta | 13.0–30.0 | 2 | 0.62% | [0.15, 1.77] |
| beta_low_clean | 13.0–17.9 | 0 | 0.00% | — |
| beta_high_clean | 23.1–30.0 | 2 | 2.79% | [0.92, 8.62] |
| **gamma** | 30.1–80.0 | **35** | **34.86%** | [14.67, 54.68] |
| gamma_low_clean | 30.1–38.0 | 4 | 4.51% | [2.05, 7.40] |
| gamma_mid_clean | 43.0–56.0 | 10 | 29.17% | [16.39, 47.15] |
| gamma_high_clean | 67.0–77.0 | 8 | 32.18% | [3.64, 42.12] |

The share is the power sitting **above the local background** at the line bins, divided by
total band power. An earlier version of this table dropped the line bins and compared band
powers, which also drops their background and so charged ordinary spectrum to the artifact;
it read gamma as 47.4% against the 34.9% here. The correction lowers every figure and
changes no conclusion.

Delta through alpha are untouched: the comb's lowest detected member is harmonic 22
(26.4 Hz) and its lowest confirmed one harmonic 23 (27.6 Hz). Beta is nearly untouched at
0.62%, and `beta_low_clean` contains no artifact line at all.
The existing beta restriction around 20 Hz was designed against a gradient comb that is not
there; beta is clean, but for a different reason than the one on record.

Every gamma window carries lines, including all three named "clean". Those three were
chosen to dodge a 20 Hz-spaced scanner comb. Against a 1.2 Hz comb running continuously
from 28.8 to 94.8 Hz, no 10 Hz-wide window in the gamma range can avoid contamination:
`gamma_low_clean` admits 4 lines, `gamma_mid_clean` 10, `gamma_high_clean` 8. The share is
4.5% to 32% at the median and reaches 47% in the worst participant, and because it is a
participant-level property (median ICC 0.746) it enters between-participant comparisons as
a systematic offset rather than as noise.

These are measurements, not a verdict on whether any given analysis is admissible. What
they establish is the size and the structure of the term.

**Two notes for anyone reusing the older numbers.** First, the standard
end-of-preprocessing QC plot uses 4 s segments (0.25 Hz bins), which averages a 0.11 Hz
line across bins four times its width and makes the comb nearly invisible; it must not be
read as evidence the spectrum is clean. Second, the `gamma_masked_minus_full_db` column in
the existing cohort QC averages decibels, which is a geometric mean over power and
understates narrow high-amplitude bins — its own docstring puts the discrepancy near 11 dB
on synthetic data. The table above measures excess over the local background instead, which
neither understates the lines nor charges their background to them.

---

## 7. Limitations

- The control is uncorrected 5 kHz data recorded before the sequence starts. It establishes
  that a line exists without gradients; it does not establish what that line's amplitude
  would be after the full processing chain, and the two columns are not comparable in level.
- Control windows are short (median 6.25 s) and 19 of 104 recordings gave none. The control
  spectrum is computed on the 3.6 s matched grid, where a 0.11 Hz line is unresolved; it can
  say a line is present, not measure its width.
- Seven of the 50 artifact lines have a gradients-off CI that includes zero. With windows
  of a few seconds, absence of evidence at those seven is weak evidence of absence.
- The six `comb_wide` detections are genuinely ambiguous between weak comb lines and
  rhythms coinciding with comb positions. They are counted as comb members in the structural
  statistics and excluded from the band-power masking; both choices are stated rather than
  resolved.
- Physical attribution of the 1.2 Hz source to the cold head specifically is inference from
  the signature, not direct evidence. Nothing here identifies the machine by observation.
- The 57.2 Hz source is uncharacterised beyond its frequency behaviour.
- Volume-phase locking is interpretable only at frequencies periodic in the TR; it is a test
  for the scanner comb, not a general test of determinism.
- Participants and sessions are confounded — one session each — so between-participant and
  between-session scatter cannot be separated.

---

## 8. Reproducing

```bash
eeg-pipeline line-comb diagnose --stage all
```

```bash
eeg-pipeline line-comb plot
```

The first stage reads the drive once (~15 min) and caches per-participant spectra under
`outputs/scanner_harmonic_diagnosis/cache/`; `--stage analyse` then reruns every statistic
from the cache in about a minute. Outputs in `outputs/scanner_harmonic_diagnosis/`:

| File | Contents |
|---|---|
| `cohort_line_catalog.tsv` | 72 detections: frequency, prominence, CI, *q*, linewidth, class, harmonic, prevalence |
| `comb_structure.tsv` | fitted fundamental, harmonic range, residuals, enrichment test |
| `per_subject_fundamental.tsv` | the fundamental estimated in each participant |
| `control_persistence.tsv` | matched final vs gradients-off prominence, per line |
| `phase_locking.tsv` | Rayleigh resultants, pooled and within-run |
| `frequency_stability.tsv` | per-line frequency spread and drift across sessions |
| `topography_reproducibility.tsv` | between-participant spatial correlation |
| `variance_components.tsv` | between- and within-participant variance, ICC |
| `band_impact.tsv` | band power with and without the artifact lines |
| `control_windows.tsv` | gradient onset and usable window for all 104 recordings |
| `cardiac_rate.tsv` | heart rate per run |
| `per_run_line_prominence.tsv`, `per_subject_line_prominence.tsv` | the underlying values |
| `figures/fig1…fig6` | cohort spectrum, final vs control, comb structure, per-participant heatmap, topographies, band impact |

Estimators live in `studies/pain_study/analysis/line_comb/diagnosis.py` and cohort assembly
in `cohort.py` alongside it, covered by `tests/analysis/line_comb/test_diagnosis.py`,
`tests/analysis/line_comb/test_cohort.py` and
`tests/scripts/line_comb/test_diagnose.py` (142 tests).

---

## References

Allen, P. J., Josephs, O., & Turner, R. (2000). A method for removing imaging artifact from
continuous EEG recorded during functional MRI. *NeuroImage, 12*, 230–239.
https://doi.org/10.1006/nimg.2000.0599

Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. *Journal of the
Royal Statistical Society B, 57*, 289–300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x

Mullinger, K. J., Castellone, P., & Bowtell, R. (2013). Best current practice for obtaining
high quality EEG data during simultaneous fMRI. *Journal of Visualized Experiments, 76*,
50283. https://doi.org/10.3791/50283

Nierhaus, T., Gundlach, C., Goltz, D., Thiel, S. D., Pleger, B., & Villringer, A. (2013).
Internal ventilation system of MR scanners induces specific EEG artifact during
simultaneous EEG-fMRI. *NeuroImage, 74*, 70–76.
https://doi.org/10.1016/j.neuroimage.2013.02.016

Rothlübbers, S., Relvas, V., Leal, A., Murta, T., Lemieux, L., & Figueiredo, P. (2015).
Characterisation and reduction of the EEG artefact caused by the helium cooling pump in the
MR environment. *Magnetic Resonance Materials in Physics, Biology and Medicine, 28*,
195–208. https://doi.org/10.1007/s10334-014-0463-2

Zar, J. H. (2010). *Biostatistical Analysis* (5th ed.). Pearson. [Rayleigh test
approximation, §27.1]
