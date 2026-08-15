# Line comb: diagnosis and removal

**Command:** `eeg-pipeline line-comb <mode>` · **Settings:** [`config.yaml`](config.yaml) ·
**Engine:** [`studies/pain_study/analysis/line_comb/`](../../analysis/line_comb/) ·
**Full write-ups:** [`docs/scanner_harmonic_diagnosis.md`](../../../../docs/scanner_harmonic_diagnosis.md),
[`docs/scanner_harmonic_removal.md`](../../../../docs/scanner_harmonic_removal.md)

## Why this folder exists

Earlier reviews of this dataset recorded narrowband contamination in the final cleaned EEG
as "residual scanner harmonics" — meaning the gradient correction had failed. It had not.

The diagnosis measured three things that agree: the volume comb at `k/TR` is gone, the
surviving lines are still there in the 731 s of EEG recorded *before* the scanner played a
single gradient pulse, and the fundamental is 1.199998 Hz ± 61 µHz across 15 sessions and
5 months. That is 72.0 cycles per minute, exactly one fiftieth of the 60 Hz mains. A person
does not hold a frequency to five decimal places for five months, and a gradient artifact
does not appear when the gradients are off.

So the contamination is the **room**, not the scanner, and the gradient correction could
never have touched it because it was never gradient-locked. Removing it needs a different
mechanism, which is the rest of this folder.

It is worth removing for exactly one reason: the 50 artifact lines carry **34.9% of gamma
power** (1.52 µV RMS in total). Beta carries 0.6%; delta through alpha carry none. And
because the coupling is a participant-level property (ICC 0.75), it enters
between-participant comparisons as a systematic offset rather than as noise.

## Why it is not a stage of the main pipeline

MNE-BIDS-Pipeline's notch is FIR. Fifty narrow FIR notches would take the surrounding band
with them. The removal runs before the pipeline reads the data and writes its own BIDS
root, leaving every sidecar byte-identical and rewriting only the `.eeg` binaries — so
sampling rate, channel set, length and annotations cannot drift, and the `Volume` and `R`
marker names downstream code reads from `events.tsv` survive.

## The files

| File | What it contributes |
|---|---|
| `diagnose.py` | Measures which narrowband lines exist and which belong to the comb. Two stages: `cache` reads the recordings once to disk, `analyse` reworks every statistic from that cache in about a minute without touching the drive again. |
| `plot.py` | Draws the diagnosis. Reads only what `diagnose` wrote, so figures can be restyled without recomputing. |
| `remove.py` | The removal itself, plus the benchmark that must pass before it is trusted and the verification that re-measures what was written. |
| `notch.py` | The optional wide notch, for contamination that is a cluster rather than a set of resolvable lines. Reads what `remove.py` wrote. |
| `report.py` | Resolves each participant's actual manifest targets against the verification spectra and writes participant-specific residual and band tables. |
| `psd.py` | Welch spectra of the source and of every derivative that exists, as an overview, a tiled figure at roughly one bin per pixel, and one panel per recording. Needs only `apply` to have run, and is the one stage that accepts `--subjects`. |
| `config.yaml` | Every number above, and where the inputs and outputs live. |

## Stage order

```bash
eeg-pipeline line-comb diagnose     # measure the lines (cache + analyse)
eeg-pipeline line-comb plot         # figures from that measurement
eeg-pipeline line-comb benchmark    # does the removal preserve signal? run this first
eeg-pipeline line-comb apply        # write the cleaned BIDS copy
eeg-pipeline line-comb verify       # re-measure what was written
eeg-pipeline line-comb report       # band-by-band outcome tables
eeg-pipeline line-comb notch        # optional: wide notch over cluster bands
eeg-pipeline line-comb psd          # before-and-after spectra of whatever exists
```

Every stage reads its settings from `config.yaml`. To point one somewhere else, use
`--workflow-config`, **not** `--config` — the latter is the core pipeline's config and the
top-level CLI takes it out of the command line before this subcommand ever sees it.

`notch` is the counterpart to `apply`, not a variant of it. `apply` subtracts a sinusoid
wherever a line is resolvable; `notch` removes a band where the contamination is a cluster
and no sinusoid subtraction can reach it. It reads what `apply` wrote and writes its own
root (`paths.notched_root`), so the two transforms stay separable and either can be
inspected alone. Point the core `paths.bids_root` at whichever one you intend to analyse.

Because they are counterparts they must not both aim at the same spectrum, so **the
removal excludes every band listed in `notch_bands`**, exactly as `exclude_mains` excludes
mains from it. Leaving that out deadlocked the workflow rather than merely duplicating
work: subtracting a sinusoid from a cluster promotes the neighbouring peak, the survivor
fails the residual criterion, `apply` refuses — and `notch`, which reads what `apply`
wrote, can never run. On sub-0008 that failed 5 of 6 recordings at p=0.0244, the floor with
40 controls, each with its worst residual inside 57.15–57.35 Hz.

It is not gated, because a fixed-width FIR notch has no estimator that could be wrong.
What it produces instead is `notch_manifest.tsv`: per recording and per band, the measured
in-band attenuation and the measured change in every analysed band — not the requested
width, which understates the real footprint because of the filter transitions.

**Run `benchmark` before `apply`.** Its criteria are stated before the measurement is
taken. A failure starts a root-cause investigation: it may reveal a defective estimator,
an unsupported artifact, excessive collateral cost, or a poorly calibrated decision rule.
A narrow miss is not evidence that the entire method is invalid, but the threshold is not
moved after seeing the result without an independently justified recalibration.

**Detection and the transform both work on the continuous run alone** — the whole-run
spectrum plus overlapping 54-second spectra — and one transform is applied, the continuous
overlap-add.

There used to be a second scope: the exact analysis interval around each task event, with
its own 20-second transform spliced into the reconstruction, its own detection evidence,
and four of its own gate criteria. All of it is gone. The transform went first, because
anchoring it left lines standing in the very samples the studies read. The rest followed on
measurement: a 20-second epoch spectrum resolves 50 mHz where a 54-second window resolves
18.5 mHz, and 37 mHz is the separation this code uses to call two peaks distinct sources —
so an epoch cannot resolve anything a window covering the same samples cannot. Measured on
sub-0008, the epochs authorized no distinct source at all: every difference they made was a
position shift of 2–26 mHz, each inside the notch applied anyway, plus one duplicate target
per run. Their four criteria held back 2 recordings of 90.

The practical consequence is that **this workflow needs no events**. A resting or baseline
acquisition, or any continuous recording, is a valid input.

Every narrow summit in the band is observed exactly once: those clearing the comb are
isolated lines, those beside a validated harmonic but outside its notch are comb-adjacent.
Both take the same route to becoming a target — cross-run replication first, then the
single-recording route — because adjacency to a harmonic constrains where a false positive
can land without replicating one.

A Thomson F-test on the first pass's own output resolves residuals once inside already
authorized artifact regions, using thresholds declared independently of the acceptance
gate, so the gate can still fail on a residual the detector does not reach. Continuous
residuals are measured after overlap-add and routed across the synthesis windows that
create the evidenced samples; each is located once over the window that evidenced it, and
only its amplitude and phase follow the shorter regression sub-windows. Simultaneous
resolvable summits remain distinct sources.

## What the criteria are, and what kind of thing each one is

They are not all the same kind of claim, and conflating them was a real defect: a
criterion that had a stated error rate was given a decibel cushion on top, which took the
error rate away.

**Calibrated tests — decided over the recordings.** The residual questions, whole-run and
focal, and the seam. Each measures an observation against controls that repeat the same
search where no target is, so under the null the observation is exchangeable with them and
an exact probability follows by counting. `residual_null_p` and `focal_residual_null_p` are
the decision; the decibel figures they replace are still reported because decibels are
readable in a way probabilities are not.

**Preservation is reported, not decided — and that is a finding, not an omission.** Two
questions ask whether the transform disturbed spectrum it never targeted: the injected
tones, and the band outside the removed bins. Both were answered by a constant with no
derivation, at margins of 3452× and 8×, and neither could be given a null.

The tones cannot support a test at all: four tones on one channel is four observations, and
the best a sign test can return is 2⁻⁴ = 0.0625.

The band question looks like it can, and does not. Displacing the whole transform by a
quarter of the comb spacing gives a control matched in size, width and window geometry —
but its targets land between harmonics where there is no line, so it subtracts almost
nothing and leaks almost nothing, while leakage from the real transform scales with the
line power it removed. Counted against that control the real transform "fails" at p≈2e-16
on every recording, which says only that it removed something. No offset repairs it: a
control that removes comparable power *away* from the lines cannot exist, because the power
is only at the lines.

So both are reported beside their controls and nothing is decided from either. What the
pairs show is the scale: probes disturbed by ~1.5e-5 dB against a control's ~8e-4, and the
off-target band by ~0.015 dB against ~0.01 — four orders below the 0.2 dB that used to be
the criterion.

None of them belongs in the per-run gate. Each fails about one recording in twenty by
construction, so an all-runs rule rejects a faultless cohort roughly ninety-nine times in a
hundred. Benjamini-Hochberg over the recordings controls the false discovery rate instead —
and with a single recording it reduces to `p <= 0.05`, so a lone continuous acquisition is
decided by its own exact test rather than by a cohort statistic it cannot supply. `apply`
consults all three and refuses on any.

`residual_excess_db` is still reported: it is the same observation in decibels above the
controls' 95th percentile, which is readable in a way a probability is not. It is no longer
what anything is decided by. The gate that used to read it sat at 1.0 dB while 0.0 dB was
already the calibrated boundary, and that cushion rescued 6 of 90 recordings.

**A derived bound.** `transient_preserved`. A 50 ms burst at 40 Hz has a spectral width of
3.2 Hz, so it spans about 13 Hz and crosses roughly eleven comb lines at 1.2 Hz spacing,
each subtracted over freq/450 = 0.089 Hz — an expected loss near 7.5%. The floor sits at
twice that. It comes from the instrument and the transform, not from the data.

**An invariant.** `transient_undistorted`. `spectrum_fit` is linear, so this reads 1.0 on
any data with any settings. Kept because a genuinely non-linear failure would break it, but
it is not evidence that anything worked.

**No ceiling on spectral cost.** There used to be one, at 18% of 28–95 Hz, and it was
retired rather than recalibrated. Nothing is left for it to constrain: the cost is the
notch width times the number of targets, the width ratio was fixed by a documented sweep to
the narrowest setting that still pushes every line below its local background, and a target
only exists once the replication rules admit it. So the cost is already whatever the
evidence justifies, and a number on top of that could only have been chosen after seeing
the answer — which is exactly what 0.18 was, set because the cohort had reached 0.170.

What replaces it is measurement. A broadband probe goes through the identical transform, so
`measured_band_attenuated_1db` reports what a signal occupying the band actually loses
rather than what the plan asked for — on the delivered cohort those differ by about three
points, 0.204 against 0.161. It is printed per cohort, stored per recording, and written
into the delivered dataset's `GeneratedBy` provenance, so the cost travels with the data
and can be weighed against the artifact removed.

A study that does want a stated budget sets `line_comb_removal.max_band_cost`, which is
null by default. `apply` then refuses against it, and the declaration is recorded — an
explicit scientific decision rather than a constant shipped as if it were a finding.

The benchmark also reports `min_in_band_probe_survival` and
`median_in_band_probe_survival` — the fraction of a narrowband probe placed *on* four of
the plan's own targets that survives removal. Every other probe sits where nothing is
removed and so cannot report a loss; this one measures the cost at the frequencies the
method does act on. It is reported, never gated, because signal at an artifact frequency is
not separable from the artifact.

MNE-BIDS loading makes `channels.tsv` authoritative, so ECG/EOG remain byte-for-byte data
channels outside the EEG transform and its gates.

To make the rest of the pipeline read the cleaned data, point `paths.bids_root` in the core
`eeg_config.yaml` at the directory `apply` wrote (`paths.output_root` here).
