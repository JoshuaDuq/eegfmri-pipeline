# Gradient-trough ICA

This workflow fits participant-level ICA decompositions to the stable bottoms of the
residual scanner-gradient troughs in the BrainVision Analyzer-corrected 1 kHz data. It
then applies the fitted unmixing matrix to the complete, unfiltered corrected data and
computes component time-frequency representations for low versus high heat.

The Python stage performs no component rejection. It:

1. averages rectified ECG in 0–900 ms volume-marker epochs separately for every run;
2. refines the participant reference troughs within ±5 ms;
3. selects the contiguous 20%-depth trough bottom, requiring 10–20 ms duration;
4. high-pass filters each continuous EEG run at 40 Hz before extracting those samples;
5. exports six plateau-selected ICA trials and 66 unfiltered -7 to +15 s thermal trials.

Run the exporter from the repository root:

```bash
.venv/bin/python -m studies.pain_study.gradient_trough_ica.exporter
```

The exporter refuses to overwrite existing files. Its runtime manifest is written to
`outputs/gradient_trough_ica/gradient_trough_ica_runtime.json`.

In MATLAB, add this package's `matlab` directory to the path and pass explicit FieldTrip
and runtime locations:

```matlab
addpath("/absolute/path/to/studies/pain_study/gradient_trough_ica/matlab");
runGradientTroughIca( ...
    "/absolute/path/to/fieldtrip", ...
    "/absolute/path/to/outputs/gradient_trough_ica/gradient_trough_ica_runtime.json");
```

The MATLAB stage uses deterministic extended Infomax (`runica`) at the numerical data
rank. It transfers the exact unmixing matrix to the unfiltered trials. FieldTrip
`mtmconvol` uses DPSS tapers, a fixed 800 ms window, ±5 Hz smoothing, 10–100 Hz, and a
-5 to -0.01 s dB baseline. Low is 44.3/45.3 °C, high is 48.3/49.3 °C, and middle
temperatures are excluded. All QC, component, and TFR figures are generated in MATLAB.

