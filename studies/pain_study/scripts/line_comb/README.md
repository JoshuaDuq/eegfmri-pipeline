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
| `report.py` | Resolves each participant's actual manifest targets against the verification spectra and writes participant-specific residual and band tables. |
| `config.yaml` | Every number above, and where the inputs and outputs live. |

## Stage order

```bash
eeg-pipeline line-comb diagnose     # measure the lines (cache + analyse)
eeg-pipeline line-comb plot         # figures from that measurement
eeg-pipeline line-comb benchmark    # does the removal preserve signal? run this first
eeg-pipeline line-comb apply        # write the cleaned BIDS copy
eeg-pipeline line-comb verify       # re-measure what was written
eeg-pipeline line-comb report       # band-by-band outcome tables
```

**Run `benchmark` before `apply`.** Its criteria are stated before the measurement is
taken. A failure starts a root-cause investigation: it may reveal a defective estimator,
an unsupported artifact, excessive collateral cost, or a poorly calibrated decision rule.
A narrow miss is not evidence that the entire method is invalid, but the threshold is not
moved after seeing the result without an independently justified recalibration.

Detection covers the complete continuous run (whole-run plus overlapping 54-second
spectra) and every exact -5 to +15 second interval around `Trig_therm/T  1`. Evidence from
those intervals contributes to the continuous plan only where the sample ranges overlap.
Each exact interval also has its own immutable 20-second transform, which replaces the
continuous result on every exact sample and tapers only outside it. Outside-interval
evidence cannot authorize an isolated or comb-adjacent exact target. A Thomson F-test on
the first pass's own output resolves residuals once inside already authorized artifact
regions, using thresholds declared independently of the acceptance gate, so the gate can
still fail on a residual the detector does not reach. Continuous residuals are measured
after overlap-add and routed across the synthesis windows that create the evidenced
samples. Simultaneous resolvable summits remain distinct sources. The benchmark reports
continuous and exact study-window residual, focal, injected-sinusoid, and
off-target-spectrum endpoints separately.

Two criteria are decided over the cohort rather than inside each run, because at their
per-run error rates neither survives being applied ninety times: the seam criterion, and
whether any sinusoid remains in an authorized region. The second gives each recording one
Bonferroni-corrected probability and applies Benjamini-Hochberg across recordings. `apply`
consults both and refuses on either.

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
