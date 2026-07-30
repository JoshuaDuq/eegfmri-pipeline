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
| `report.py` | Turns the diagnosis catalogue, removal manifest and verification spectra into the band-by-band outcome tables behind the removal doc. |
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
taken. A failure means the settings are wrong, not that the criteria should move.

To make the rest of the pipeline read the cleaned data, point `paths.bids_root` in the core
`eeg_config.yaml` at the directory `apply` wrote (`paths.output_root` here).
