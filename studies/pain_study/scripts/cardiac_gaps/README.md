# Cardiac gaps: recovering the heartbeats Analyzer never marked

**Command:** `eeg-pipeline cardiac-gaps <mode>` · **Settings:** [`config.yaml`](config.yaml) ·
**Engine:** [`studies/pain_study/analysis/bcg/`](../../analysis/bcg/) ·
**Full write-up:** [`docs/pulse_artifact_correction_recovery.md`](../../../../docs/pulse_artifact_correction_recovery.md)

## Why this folder exists

The EEG amplifier started before the MRI and stopped after it. Those leading and trailing
stretches carry no `Volume/V` markers, so Analyzer's MR Correction had no gradient template
for them and left the gradient artifact standing. Analyzer's Cardioballistic correction
then built its pulse template from `Start = 0, Length = 15 s` — exactly that uncorrected
region. A template built on gradient artifact matches no heartbeat, so R detection failed,
the R-to-artifact delay could not be computed, and Analyzer fell back to its
`"default value of 0.21 s?"` prompt: no markers written, no template subtracted, and the
ballistocardiogram left in the EEG at 20–36 µV.

## Why the fix is *not* "correct the whole recording"

Two measurements decided the shape of this workflow:

- **Analyzer wins where it marked a beat.** Its correction there measures ~20× below the
  circular-shift null. Ours does not beat it. So its work is kept wherever a marker exists.
- **The residual sits entirely in the beats it never marked.** That is untouched artifact,
  and it is all this workflow changes.

Correction is therefore *confined to the gaps*. Everything Analyzer marked is left alone.

**ICA is not the alternative.** 176 cardiac components are already excluded cohort-wide;
the residual survives that. This is not a configuration gap.

## The files

| File | What it contributes |
|---|---|
| `correct.py` | The whole workflow: find Analyzer's marker gaps from its own RR intervals, recover beats there by QRS template matching, correct only those stretches, and score the result. |
| `config.yaml` | The two Analyzer exports that get paired, where output goes, and the correction method and rank. |

## Stage order

```bash
eeg-pipeline cardiac-gaps report      # measure the gaps, change nothing
eeg-pipeline cardiac-gaps markers     # write recordings carrying recovered R markers
eeg-pipeline cardiac-gaps benchmark   # score methods and ranks before writing
eeg-pipeline cardiac-gaps apply       # write the corrected recordings
eeg-pipeline cardiac-gaps verify      # re-read the written binaries and score them
```

**Choose on both benchmark arms together** — artifact removed *and* signal preserved.
Removal alone improves monotonically with rank while the signal is being destroyed, so a
method picked on `removal_max` will look best exactly when it is worst. OBS with 4
components is what the two arms together selected.

`verify` re-reads the binaries rather than trusting the in-memory result, and output
provenance is recorded so it cannot score a stale file.

## Reading the output

A run that resolves nothing is a measurement, not a fault — it is reported as a row, and it
does not abort the cohort. Rows whose recovered rate is not a physiological heart rate are
flagged rather than silently kept.
