# Cleaning the delivered epochs without removing neural signal

Date: 2026-08-02

## Why

An audit of the delivered epochs (`derivatives/preprocessed/eeg`, built 2026-08-01 from
`bids_output/eeg_linecleaned` by MNE-BIDS-Pipeline 1.10.1) measured what is still in them
for the 14 analysed participants, sub-0008 excluded. Three things came out of it.

- Narrowband lines survive in the two upper gamma windows: **4.8% of 45-58 Hz** band power
  sits above background at a line (median across participants; 23.4% at worst, sub-0009),
  and **1.7% of 62-95 Hz**. **30.1-45 Hz measures 0.00% in every participant and all 866
  good channels.**
- Residual ballistocardiogram is the largest artifact present: R-locked peak-to-peak of
  3.0-24.0 uV in 1-20 Hz, 3.1-16.0x a circular-shift null, in all 14 participants and
  855 of 866 channels. It barely reaches gamma -- median z on the 62-95 Hz envelope is
  0.40, with 1 of 14 participants above z = 3.
- Muscle is the confound that can manufacture a gamma result. Each channel's
  pain-minus-warm gamma change correlates with its 62-95/8-30 Hz muscle index across
  channels in 12 of 14 participants (median r = +0.34 / +0.28 / +0.19 for gamma_low /
  gamma_mid / gamma_high), and the
  contrast is 1.7-3.5x larger on the highest-muscle quartile of channels than the lowest.

The constraint on every fix below is that it must not remove neural signal of interest.
Each stage therefore carries an explicit signal-loss budget or a guard that fails the
stage.

## Attribution correction

The audit first reported lines at 81.195 and 82.294 Hz and inferred from them that the
true TR was 0.8992 s rather than the nominal 0.9. Both conclusions were wrong and the
design does not rest on them.

The Volume markers give TR = 0.900000 s in all 90 runs. Checking the raw bins showed that
**the bin exactly at each gradient-comb position k/TR is nulled** -- median prominence
-5.59 dB against +0.17 dB at off-comb bins, in 81 of 81 harmonics across 3-95 Hz. That is
the signature of volume-average subtraction having already removed the comb. What the
peak-picker found at 81.195 and 82.294 Hz were the *shoulders* either side of the nulled
centres of harmonics 73 and 74, which carry more sideband energy than the rest of the comb
(+2.4 and +4.3 dB against a +0.88 dB typical shoulder). That is volume-to-volume gradient
variability, not an independent line.

Consequences: the existing `isolated_hz` entry `82.2228` is **correctly centred** and the
`freq/450` notch width (+/-0.09 Hz) spans its sidebands, so it needs no change; and the
right target for harmonic 73 is the comb centre 81.1111 Hz, not the shoulder.

Re-running the band shares with the corrected attribution moved them very little:
gamma_mid 4.78% median (from 5.25%), gamma_high 1.67% (from 1.41%), gamma_low 0.00%
either way. The whole of gamma_high's contamination is the h73+h74 sidebands.

## Shape of the work

Four stages. The two cheap ones each produce a measurement that decides whether they
ship, and both run before the expensive step, so **MNE-BIDS-Pipeline is re-run exactly
once regardless of how they turn out**.

1. Line-config corrections, then `line-comb apply`; verified on the cleaned BIDS.
2. ECG-driven BCG correction, developed and gated against the existing derivatives.
3. One MBP re-run carrying whatever cleared stages 1-2, plus ICA re-review.
4. CSD gamma variant, at feature level; needs no pipeline re-run.

Stages 1, 3 and 4 are configuration changes and a pipeline run, and belong to one
implementation plan. **Stage 2 gets its own plan.** It is method development with an
empirical gate, not a config change: it can fail, its failure must not block the rest, and
it is comparable in size to the work already specified in
`2026-07-30-cardiac-gap-fill-design.md`. Splitting it also keeps the gate honest, since a
plan that bundles it with the certain wins invites shipping it on their strength.

## Stage 1 -- lines

Four additions to `line_comb_removal.isolated_hz` in
`studies/pain_study/scripts/line_comb/config.yaml`. Nothing else changes.

| line | disposition |
|---|---|
| 57.19 Hz | none needed -- seed `57.2247` is 0.03 Hz away, inside the 0.15 Hz search |
| 99.60 Hz | none needed -- covered by `removal_harmonic_range [22, 83]` and `high_hz 99.8` |
| 82.2222 Hz (gradient h74) | none needed -- `82.2228` is on the comb and the notch spans the sidebands |
| **81.1111 Hz** (gradient h73) | **add** -- same sideband structure, +2.4 dB, inside gamma_high, currently absent |
| **29.6854 Hz** | **add** -- narrow, off both combs, 9/14 participants, 0.41 Hz below the gamma_low edge |
| **23.7776 Hz** | **add** -- narrow, off both combs, 7/14 participants, beta |
| **61.0353 Hz** | **add** -- mains +1.02 Hz sideband, 11/14, in the unanalysed 58-62 gap; hygiene |
| 93.76 Hz | leave -- 16.6 dB but in one participant; a cohort notch is the wrong trade |
| 59.02 Hz | leave -- 4/14, and inside the 60 Hz notch skirt |

Add 81.1111 Hz with a comment recording that it is the imaging gradient's 73rd harmonic at
TR = 0.9 s and must be re-derived if the sequence TR changes, matching the note already on
`82.2228`.

### Signal-loss budget

Masked bandwidth today is 1.11 Hz of gamma_mid (8.5% of the window) and 1.50 Hz of
gamma_high (4.5%). Adding 81.1111 Hz costs gamma_high a further ~0.18 Hz, taking it to
5.1%. The two beta additions cost ~0.13 Hz of 13-30 Hz, 0.9% of that band. **Nothing is
added inside 30.1-45 Hz, which measures 0.00% contaminated and stays untouched.**

### Verification

`eeg-pipeline line-comb benchmark` must stay at 13/13 runs on all 7 gates before `apply`.
Afterwards, re-run the audit scripts on the cleaned BIDS and confirm that the gamma_mid
share falls toward zero, that gamma_low is still 0.00%, and that no new trough appears at
any added frequency.

## Stage 2 -- BCG, behind a held-out gate

R peaks come from the **ECG channel carried in the epochs**, not from Analyzer markers.
The audit recovered 1523-2056 beats per participant this way, which is the complete train
where the Analyzer's had gaps -- and the residual is known to sit in the beats the
Analyzer never marked. Correction is an optimal-basis-set pass over the continuous runs,
before MBP.

The gate matters more than the method, because a previous attempt at a basis from
recovered beats failed by staying local to the beats it was built from.

- **Split by beat parity.** Build the basis on odd-numbered beats; measure R-locked
  residual reduction on even-numbered beats only. A basis that merely fits its own beats
  shows nothing here.
- **Baseline to beat:** the per-participant audit numbers -- 3.0-24.0 uV R-locked
  peak-to-peak, 3.1-16.0x the circular-shift null.
- **Neural-signal guard 1.** 1-20 Hz band power *outside* the R-locked window must not
  fall. A drop there means the pass is subtracting broadband rather than artifact.
- **Neural-signal guard 2.** Individual alpha peak frequency and amplitude unchanged.
  Alpha is the most identifiable neural feature in the contaminated band; if it moves,
  the pass is eating signal.

Failing any guard means the pass does not ship and the existing annotation of uncorrected
intervals stands. Stage 3 proceeds either way, so a failure costs nothing but the
development time.

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

Bad-channel policy is unchanged -- uninterpolated and logged -- because the config's
reasoning holds: an interpolated channel is a weighted sum of its neighbours, so
connectivity between it and those neighbours is partly set by the interpolation. Today
that is 16 channels across 9 of the 14 analysed participants. **Cz in sub-0011** gets a
specific look, being midline and reference-adjacent. pyprep re-detecting on cleaned data
may change the set anyway.

## Stage 4 -- CSD gamma variant

`feature_engineering.spatial_transform_per_family.power` stays `"none"`. CSD is added as a
**parallel gamma power variant**, not a replacement: the config's reasoning is right that
CSD changes the units, topography and interpretation of amplitude features. A dB contrast
is a ratio, so the unit change cancels, but the topography change does not. The
confirmatory analysis is specified on the CSD variant; the voltage-space variant remains
for continuity. Parameters already exist (`lambda2: 1.0e-5`, `stiffness: 4.0`) and CapTrak
electrode positions are present.

**CSD must earn its place.** Recompute the across-channel correlation between the muscle
index and the pain-minus-warm contrast on CSD output. The pre-CSD values are median
r = +0.34 / +0.28 / +0.19 for gamma_low / mid / high, 12 of 14 participants positive. If
that correlation does not fall, CSD has bought nothing here and the finding is reported
rather than the transform being kept for appearances.

**Interaction with Stage 3.** CSD fits a spherical spline over the montage, so a missing
channel distorts its neighbourhood -- degrading exactly the participants that already have
problems. Resolution: interpolate bad channels **inside the CSD branch only**, leaving the
voltage-space data uninterpolated. This is the one place the two stages pull against each
other and it is settled here rather than left to the implementer.

## Deliberately left alone

- **The gradient comb.** Nulled at all 81 harmonics in 3-95 Hz. Correctly handled.
- **The -5.59 dB holes at the comb bins.** Inherent to volume-average subtraction. No
  neural signal is phase-locked to the scanner clock, so they cost essentially nothing.
- **93.76 Hz** (one participant), **59.02 Hz** (4/14, inside the notch skirt), the
  **ICLabel 0.8 threshold** (it was trained outside a scanner; loosening it starts taking
  brain components), and the 5-11% of channel-trial cells AutoReject interpolates, which
  is already persisted as a derivative.
- **Channel exclusion and muscle-index covariates as EMG remedies.** Considered and not
  taken; CSD is the chosen handling.

## What this does not fix

Muscle contamination is reduced by CSD, not eliminated, and the pain-minus-warm gamma
effect will still need to be defended against it in the analysis design. The audit's
positive finding -- the effect survives on the lowest-muscle quartile of channels at
+0.28 dB (gamma_mid) and +0.33 dB (gamma_high), 13 of 14 participants positive -- came
from a coarse contrast over the whole 21.6 s epoch, which mixes baseline, stimulus and
rating periods. It is evidence that something survives, not an effect size.
