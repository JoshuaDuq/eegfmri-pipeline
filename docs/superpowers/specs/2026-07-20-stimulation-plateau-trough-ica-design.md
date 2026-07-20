# Stimulation-Plateau-Restricted Gradient-Trough ICA

## Objective

Repeat the participant-level gradient-trough ICA analysis for `sub-0014` and
`sub-0015`, but train ICA only on artifact-trough plateau samples that occur during
the constant-temperature plateau of thermal stimulation. Preserve the existing
whole-run trough ICA results.

## Scientific definition

Each thermal trial is anchored to its `Trig_therm/T  1` marker. The stimulation
plateau is the half-open interval from +3.0 seconds through +10.5 seconds relative
to that marker: the 7.5-second period after ramp-up and before ramp-down.

A gradient-trough plateau window is eligible only when its complete half-open source
sample interval is contained within one thermal plateau interval. Selection by trough
center and boundary truncation are prohibited. This guarantees that every ICA sample
belongs simultaneously to a stable gradient-trough bottom and the constant-temperature
stimulation plateau.

## Data flow

1. Load the same six BrainVision Analyzer-corrected 1 kHz runs per participant.
2. Align event-table thermal trials to BrainVision annotations using the first volume
   marker as the run clock anchor.
3. Construct each trial's +3.0 to +10.5-second stimulation-plateau source interval.
4. Reuse the existing run-specific ECG procedure: average rectified 0–900 ms volume
   epochs, refine each participant trough within ±5 ms, and identify the contiguous
   20%-depth trough bottom constrained to 10–20 ms.
5. Generate all source trough windows, then retain only windows fully contained in a
   stimulation-plateau interval.
6. Apply the same 40 Hz zero-phase FIR high-pass filter to each continuous EEG run
   before extracting the eligible windows. Concatenate selected samples into one ICA
   trial per run; do not filter after concatenation.
7. Export the same complete, additionally unfiltered -7 to +15-second thermal epochs
   for later unmixing-matrix application.
8. Fit deterministic participant-level extended Infomax ICA at numerical rank and
   apply the exact unmixing matrix to the complete unfiltered epochs.
9. Recompute the unchanged low/high component time courses and DPSS multitaper TFRs:
   10–100 Hz in 1 Hz steps, -5 to +14.4 seconds in 100 ms steps, fixed 800 ms window,
   ±5 Hz smoothing, 32-second padding, and -5 to -0.01-second dB baseline.

## Conditions and outputs

Low remains 44.3/45.3 °C and high remains 48.3/49.3 °C; 46.3/47.3 °C trials remain
excluded from condition comparisons. No ICA components are automatically rejected.

All new exports, derivatives, and MATLAB figures are written beneath a separate
`outputs/gradient_trough_ica_stimulation_plateau` root. Existing files beneath
`outputs/gradient_trough_ica` are never modified.

The selection audit records participant, run, thermal trial, temperature, volume,
trough, source samples, stimulation-relative latency, and selected-run samples. The
selection-QC plot distinguishes all run-level trough plateaus from the subset retained
during stimulation plateaus.

## Validation and failure behavior

The workflow fails rather than silently dropping or repairing data when:

- thermal trial/event alignment exceeds 2 ms;
- stimulation-plateau intervals overlap or extend outside a recording;
- a retained trough window crosses a stimulation-plateau boundary;
- any thermal trial contributes no eligible trough window;
- selected intervals overlap or lose chronological ordering;
- run channels, sample rate, trial count, or metadata differ from the established
  participant contract; or
- output files already exist.

Tests cover exact containment, boundary rejection, trial attribution, chronological
ordering, configuration, MATLAB ICA/TFR settings, and separation from the existing
output root. Real exports are read back to verify six ICA trials, 66 unfiltered epochs,
63 aligned EEG channels, finite values, and complete audit provenance.
