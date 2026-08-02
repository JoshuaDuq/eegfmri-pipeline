# Making 1-100 Hz usable without removing neural signal

Date: 2026-08-02
Revised: 2026-08-02, after the goal widened from gamma to the whole 1-100 Hz range.

## Why

An audit of the delivered epochs (`derivatives/preprocessed/eeg`, built 2026-08-01 from
`bids_output/eeg_linecleaned` by MNE-BIDS-Pipeline 1.10.1) measured what is still in them
for the 14 analysed participants, sub-0008 excluded.

Four costs were measured per band, on the channel-median spectrum of 21.6 s
TR-commensurate segments. They are kept apart because they need different fixes.

| band | independent lines | comb sidebands | holes | BCG excess | BCG subjects |
|---|---|---|---|---|---|
| delta 1-4 | 0.00% | 4.18% | 4.67% | 2.41x | 9/14 |
| theta 4-8 | 0.00% | 3.39% | 3.13% | 5.22x | 14/14 |
| alpha 8-13 | 0.00% | 4.15% | 3.90% | 7.54x | 14/14 |
| beta 13-30 | 0.73% | 4.18% | 5.23% | 7.57x | 14/14 |
| gamma 30.1-45 | 0.00% | 3.57% | 8.56% | 3.22x | 13/14 |
| gamma 45-58 | 4.78% | 3.30% | 15.61% | 1.19x | 4/14 |
| notch gap 58-62 | 11.75% | -- | 34.50% | -- | -- |
| gamma 62-95 | 0.00% | 4.06% | 16.59% | 1.05x | 1/14 |
| top 95-100 | 7.44% | 3.17% | 17.78% | -- | -- |

- **Independent lines** are narrowband features belonging to neither comb: the only thing
  `line-comb apply` can remove. Worst case across participants is 23.4% (45-58 Hz,
  sub-0009).
- **Comb sidebands** are excess in the shoulders of each gradient harmonic k/TR. Residual
  volume-to-volume gradient variability, near-uniform across the spectrum.
- **Holes** are power *missing* where the spectrum sits below its own background, from
  removal already applied -- the gradient volume-average subtraction and the 60 Hz notch.
  Removing artifact and removing signal both show as deviation from background; only the
  sign differs, and a full-band ambition has to count both.
- **BCG excess** is R-locked modulation of that band's analytic envelope against a
  circular-shift null, as a multiple of the null. 1.0 means nothing above chance.

### The band splits in two at about 45 Hz

Below 45 Hz there are essentially **no removable lines** -- 0.00% in delta, theta, alpha
and gamma_low, 0.73% in beta -- and BCG runs at 2.4-7.6x chance in 13-14 of 14
participants on 539-858 of 866 channels. Above 45 Hz the reverse holds: BCG falls to
1.19x and then 1.05x, while lines and notch holes take over.

**This means Stage 1 does nothing for 1-45 Hz and Stage 2 does nothing for 45-100 Hz.**
Which stage is load-bearing depends entirely on which end of the range is wanted, and for
the full range both are.

### A correction carried forward

Two things the audit got wrong on the way, recorded so the design is not read as resting
on them.

**Gamma is not free of BCG.** The first pass measured the R-locked envelope on 62-95 Hz
only (0.39 median z, 1 of 14 participants) and generalised that to "gamma". Measured per
band, **30.1-45 Hz carries R-locked modulation at 3.22x the null in 13 of 14 participants
on 701 of 866 channels.** It has no lines, which is what was measured; it is not clean.

**The peaks near 81.2 and 82.3 Hz are not lines.** The bin at every gradient-comb
harmonic is nulled -- median prominence -5.59 dB against +0.17 dB off-comb, in 81 of 81
harmonics across 3-95 Hz -- which is the signature of volume-average subtraction having
already removed the comb. Those peaks are the shoulders flanking harmonics 73 and 74. TR
is exactly 0.900000 s in all 90 runs by the Volume markers, so the existing `isolated_hz`
entry `82.2228` is correctly centred and the `freq/450` width spans its sidebands.

### Estimator note

The production 4.63 Hz background half-width returns NaN within 4.63 Hz of DC and so
cannot see delta at all; this is the trap that once read the 1/f slope as a 4 dB line
below 3 Hz. Below 13 Hz the half-width drops to 1.0 Hz, still 15x a line's width, and a
running median over a monotone background is unbiased for the 1/f slope.

## Shape of the work

Four stages. The cheap ones each produce a measurement that decides whether they ship, and
both run before the expensive step, so **MNE-BIDS-Pipeline is re-run exactly once**
regardless of how they turn out.

1. **Line and notch corrections**, then `line-comb apply`. Serves 45-100 Hz.
2. **ECG-driven BCG correction**, gated. Serves 1-45 Hz. Now critical path.
3. One **MBP re-run** carrying whatever cleared stages 1-2, plus ICA re-review.
4. **CSD gamma variant**, at feature level; no pipeline re-run.

Stages 1, 3 and 4 are configuration changes and a pipeline run, and belong to one
implementation plan. **Stage 2 gets its own plan.** It is method development with an
empirical gate: it can fail, its failure must not block the rest, and it is comparable in
size to `2026-07-30-cardiac-gap-fill-design.md`. Splitting it also keeps the gate honest,
since a plan that bundles it with the certain wins invites shipping it on their strength.

## Stage 1 -- lines and notches (serves 45-100 Hz)

### 1a. Four additions to `isolated_hz`

In `studies/pain_study/scripts/line_comb/config.yaml`:

| line | disposition |
|---|---|
| 57.19 Hz | none needed -- seed `57.2247` is 0.03 Hz away, inside the 0.15 Hz search |
| 99.60 Hz | none needed -- covered by `removal_harmonic_range [22, 83]` and `high_hz 99.8` |
| 82.2222 Hz (gradient h74) | none needed -- `82.2228` is on the comb, notch spans the sidebands |
| **81.1111 Hz** (gradient h73) | **add** -- same sideband structure, +2.4 dB, inside gamma_high |
| **29.6854 Hz** | **add** -- narrow, off both combs, 9/14, 0.41 Hz below the gamma_low edge |
| **23.7776 Hz** | **add** -- narrow, off both combs, 7/14, beta |
| **61.0353 Hz** | **add** -- mains +1.02 Hz sideband, 11/14 |
| 93.76 Hz | leave -- 16.6 dB but in one participant; a cohort notch is the wrong trade |
| 59.02 Hz | leave -- 4/14, and inside the 60 Hz notch skirt |

Add 81.1111 Hz with a comment recording that it is the imaging gradient's 73rd harmonic at
TR = 0.9 s and must be re-derived if the sequence TR changes, matching the note on
`82.2228`.

### 1b. Narrow the mains notch -- the largest recoverable piece of spectrum

58-62 Hz is the worst region in the range and most of it is self-inflicted. The MBP FIR
notch occupies **0.97 Hz** (59.537-60.463, -54.6 dB at centre). The `spectrum_fit` method
the comb removal already uses would take **0.133 Hz** at `freq/450`.

Move 60 Hz mains out of `preprocessing.notch_freq` and into the line-comb `isolated_hz`
list, so it is removed by the same narrow spectrum_fit pass as everything else. This
recovers about 0.84 Hz and, more usefully, **makes 45-95 Hz continuous** rather than
45-58 plus 62-95.

Risk to manage: mains is far stronger than any comb line, so it must be resolved first in
`estimate_comb`'s strongest-first ordering, and the benchmark must confirm the 59.02 and
61.04 Hz sidebands are still reachable once the 60 Hz peak beside them is gone.

### Signal-loss budget

Masked bandwidth today is 1.11 Hz of 45-58 (8.5%) and 1.50 Hz of 62-95 (4.5%). Adding
81.1111 Hz costs 62-95 a further ~0.18 Hz, to 5.1%. The two beta additions cost ~0.13 Hz
of 13-30 Hz, 0.9% of that band. Narrowing the mains notch **returns** 0.84 Hz. Nothing is
added inside 30.1-45 Hz.

### Verification

`eeg-pipeline line-comb benchmark` must stay at 13/13 runs on all 7 gates before `apply`.
Afterwards, re-run the audit on the cleaned BIDS and confirm the 45-58 share falls toward
zero, that 30.1-45 gains no new trough, and that the 58-62 hole drops from 34.5% to
roughly the comb-null level seen in neighbouring bands.

## Stage 2 -- BCG, behind a held-out gate (serves 1-45 Hz)

This stage now decides whether 1-45 Hz is usable at all. Nothing in Stage 1 touches it.

R peaks come from the **ECG channel carried in the epochs**, not Analyzer markers: the
audit recovered 1523-2056 beats per participant that way, the complete train where the
Analyzer's had gaps, and the residual is known to sit in the beats it never marked.
Correction is an optimal-basis-set pass over the continuous runs, before MBP.

The gate matters more than the method, because a previous attempt failed by staying local
to the beats its basis was built from.

- **Split by beat parity.** Build the basis on odd beats; measure reduction on even beats
  only. A basis that merely fits its own beats shows nothing here.
- **Per-band targets, not broadband.** The pass must reduce R-locked excess in each of
  delta, theta, alpha, beta and gamma_low. Baselines to beat are the table above:
  2.41 / 5.22 / 7.54 / 7.57 / 3.22x. A pass that fixes beta and leaves gamma_low at 3.2x
  has not delivered the range.
- **Neural-signal guard 1.** Band power *outside* the R-locked window must not fall, in
  every band. A drop there means the pass is subtracting broadband rather than artifact.
- **Neural-signal guard 2.** Individual alpha peak frequency and amplitude unchanged.
  Alpha carries the heaviest BCG load (7.54x) and is also the most identifiable neural
  feature in the range, which is exactly what makes it the right sentinel: a pass that
  cannot tell them apart will move it.
- **Neural-signal guard 3.** The muscle-index topography must not change. In 30-45 Hz,
  cardiac-locked and muscle activity overlap, and a pass that removes scalp EMG along with
  pulse artifact would look like success on guard 1.

Failing any guard means the pass does not ship and the existing annotation of uncorrected
intervals stands. Stage 3 proceeds either way.

## Stage 3 -- the MBP re-run

Ordering, because two of these are easy to get wrong:

- **Events are not rebuilt.** `line-comb apply` runs `bids_output/eeg` ->
  `bids_output/eeg_linecleaned`. BIDS is not regenerated from the Analyzer export, so the
  events chain (`fix_restart_trial_triggers.py`, then `merge-psychopy`) is not redone.
- **`bad-channels` runs before ICA.** Detection is pyprep-only and `preprocessing ica`
  never runs it; on freshly cleaned data it must run first or the run inherits a stale set.
- **`ica.manual_review_complete` returns to `false`** and all 15 participants are
  re-reviewed. The pipeline runs on everyone; sub-0008 is excluded from analysis, not from
  preprocessing.
- **`preprocessing.notch_freq` becomes null** if Stage 1b ships, since mains then moves to
  the line-comb pass. The two must not both run.

Bad-channel policy is unchanged -- uninterpolated and logged -- because the config's
reasoning holds: an interpolated channel is a weighted sum of its neighbours, so
connectivity between it and those neighbours is partly set by the interpolation. Today
that is 16 channels across 9 of the 14 analysed participants. **Cz in sub-0011** gets a
specific look, being midline and reference-adjacent. pyprep re-detecting on cleaned data
may change the set anyway.

## Stage 4 -- CSD gamma variant

`feature_engineering.spatial_transform_per_family.power` stays `"none"`. CSD is added as a
**parallel gamma power variant**, not a replacement: the config is right that CSD changes
units, topography and interpretation of amplitude features. A dB contrast is a ratio so
the unit change cancels, but the topography change does not. Parameters already exist
(`lambda2: 1.0e-5`, `stiffness: 4.0`) and CapTrak positions are present.

**CSD must earn its place.** Recompute the across-channel correlation between the muscle
index and the pain-minus-warm contrast on CSD output. Pre-CSD values are median
r = +0.34 / +0.28 / +0.19 for gamma_low / gamma_mid / gamma_high, 12 of 14 participants
positive. If that correlation does not fall, CSD bought nothing and the finding is
reported rather than the transform kept for appearances.

**Interaction with Stage 3.** CSD fits a spherical spline over the montage, so a missing
channel distorts its neighbourhood, degrading exactly the participants that already have
problems. Resolution: interpolate bad channels **inside the CSD branch only**, leaving the
voltage-space data uninterpolated.

## Deliberately left alone

- **The gradient comb itself.** Nulled at all 81 harmonics in 3-95 Hz. Correctly handled.
- **The comb-null collateral.** The holes column, 4.7% at delta rising to 17.8% at
  95-100 Hz, is the price of the gradient correction and matches the ~15% uniform gamma
  loss recorded previously. It is near-uniform, so it cancels in any contrast. Reducing it
  means a better gradient correction, which is a far larger project than this spec.
  Documented as a known constant, explicitly not targeted.
- **The comb sidebands**, 3.2-4.2% in every band. Already centred on by the existing
  notches; what remains is volume-to-volume variability that a fixed-frequency notch
  cannot reach.
- **93.76 Hz** (one participant), **59.02 Hz** (4/14, inside the notch skirt), the
  **ICLabel 0.8 threshold** (trained outside a scanner; loosening it takes brain
  components), and the 5-11% of channel-trial cells AutoReject interpolates, already
  persisted as a derivative.

## What is achievable

If both gated stages ship: continuous usable spectrum from about 1 Hz to about 97 Hz, with
0.13 Hz removed at mains, ~1.5 Hz removed across the independent lines, the comb-null
collateral as a known near-uniform constant, and R-locked residual reduced to a level the
Stage 2 gate has to state in advance.

If Stage 2 fails: 45-97 Hz is clean and continuous, and 1-45 Hz remains usable only for
contrasts where cardiac-locked modulation is plausibly balanced across conditions, with
the uncorrected intervals annotated. That is the realistic downside, and it is why Stage 2
is the one that gets its own plan and its own gate.

## What this does not fix

Muscle contamination is reduced by CSD, not eliminated, and the pain-minus-warm gamma
effect will still need defending against it in the analysis design. The audit's positive
finding -- the effect survives on the lowest-muscle quartile of channels at +0.28 dB
(45-58) and +0.33 dB (62-95), 13 of 14 participants positive -- came from a coarse
contrast over the whole 21.6 s epoch, mixing baseline, stimulus and rating periods. It is
evidence that something survives, not an effect size.
