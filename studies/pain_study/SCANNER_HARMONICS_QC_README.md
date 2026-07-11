# Residual Scanner-Harmonics QC

## Decision

For simultaneous EEG-fMRI analyses, define gamma from the retained intervals
30.1–38.0, 43.0–56.0, and 67.0–77.0 Hz. Exclude 38.0–43.0, 56.0–67.0, and
77.0–85.0 Hz from confirmatory gamma estimands. This restriction removes
empirically demonstrated residual scanner-harmonic structure; it does not
imply that gamma is absent outside the excluded windows.

## Scope

This QC was rerun on 2026-07-09 at the final-clean processing boundary:

```text
/Volumes/KINGSTON/EEG_fMRI_data/derivatives/preprocessed/eeg/
```

The analytic set comprised all numbered participants with final-clean thermal
task EEG, excluding the prespecified pilot `sub-0006`. `sub-0002` has no
final-clean recordings because its MRI experiment was incomplete. The result
therefore covers 77 runs from 13 participants:

```text
sub-0000: 6    sub-0001: 6    sub-0003: 5    sub-0004: 6
sub-0005: 6    sub-0007: 6    sub-0008: 6    sub-0009: 6
sub-0010: 6    sub-0011: 6    sub-0012: 6    sub-0013: 6
sub-0014: 6
```

## Test

For every `*_proc-clean_raw.fif` file, the test:

1. verified the 500 Hz sampling rate;
2. estimated Welch PSDs from 15 to 90 Hz over all EEG channels with
   `n_fft=n_per_seg=8192` and `n_overlap=4096`;
3. converted the across-channel median PSD to dB;
4. detected peaks with `scipy.signal.find_peaks(prominence=1.0, distance=4)`;
5. selected the most prominent peak in each fixed window: 38–43, 56–67, and
   77–85 Hz.

The test fails if any run has no peak with at least 1 dB prominence in any
window. It completed without such a failure. Values below are participant
medians across runs, followed by the minimum run-level prominence. The latter
shows that the residual peak was present in every tested run, not only in a
participant average.

| Participant | Runs | 38–43 Hz: peak; median/min prominence | 56–67 Hz: peak; median/min prominence | 77–85 Hz: peak; median/min prominence |
|---|---:|---|---|---|
| `sub-0000` | 6 | 41.14 Hz; 23.6/21.4 dB | 61.10 Hz; 29.9/26.2 dB | 82.21 Hz; 25.9/21.0 dB |
| `sub-0001` | 6 | 41.14 Hz; 17.9/17.3 dB | 61.10 Hz; 23.5/18.2 dB | 82.21 Hz; 22.1/19.5 dB |
| `sub-0003` | 5 | 41.14 Hz; 24.9/22.5 dB | 61.10 Hz; 31.0/25.4 dB | 82.21 Hz; 24.7/22.4 dB |
| `sub-0004` | 6 | 41.14 Hz; 23.2/22.3 dB | 61.10 Hz; 28.2/25.1 dB | 82.21 Hz; 23.4/21.0 dB |
| `sub-0005` | 6 | 41.14 Hz; 23.5/21.7 dB | 61.10 Hz; 26.9/23.6 dB | 82.21 Hz; 24.3/20.4 dB |
| `sub-0007` | 6 | 41.14 Hz; 22.3/21.8 dB | 61.10 Hz; 26.8/23.4 dB | 82.21 Hz; 25.3/21.5 dB |
| `sub-0008` | 6 | 41.14 Hz; 25.5/22.7 dB | 61.10 Hz; 25.7/21.9 dB | 82.21 Hz; 25.0/23.3 dB |
| `sub-0009` | 6 | 41.14 Hz; 15.9/13.3 dB | 57.19 Hz; 24.6/23.5 dB | 82.21 Hz; 19.0/17.0 dB |
| `sub-0010` | 6 | 41.14 Hz; 25.1/20.5 dB | 61.10 Hz; 31.8/30.0 dB | 82.21 Hz; 24.2/21.1 dB |
| `sub-0011` | 6 | 41.14 Hz; 24.9/22.3 dB | 61.10 Hz; 29.7/27.7 dB | 82.21 Hz; 25.0/23.7 dB |
| `sub-0012` | 6 | 41.14 Hz; 18.6/17.1 dB | 61.10 Hz; 26.8/26.1 dB | 82.21 Hz; 23.4/21.1 dB |
| `sub-0013` | 6 | 41.14 Hz; 21.2/18.4 dB | 61.10 Hz; 24.3/20.2 dB | 82.21 Hz; 23.7/20.2 dB |
| `sub-0014` | 6 | 41.14 Hz; 23.4/19.6 dB | 61.10 Hz; 28.2/25.9 dB | 82.21 Hz; 25.9/21.6 dB |

## Scanner-Timing Consistency Check

The fMRI volume repetition time was 0.9 s, so the volume frequency is
1.111 Hz. The cohort peak centres closely match integer multiples of this
frequency:

| Observed centre | Harmonic | Predicted centre | Difference |
|---:|---:|---:|---:|
| 41.138 Hz | 37 × 1.111 Hz | 41.111 Hz | +0.027 Hz |
| 61.096 Hz | 55 × 1.111 Hz | 61.111 Hz | −0.015 Hz |
| 82.214 Hz | 74 × 1.111 Hz | 82.222 Hz | −0.008 Hz |

The Welch-bin width was 500 / 8192 = 0.061 Hz. Each mismatch is smaller than
one bin, which is strong sequence-specific evidence for scanner-locked
residuals. The 82 Hz peak lies above the nominal 80 Hz gamma upper bound, but
the contaminated 77–85 Hz window overlaps the 77–80 Hz edge of broad gamma;
the exclusion removes that edge.

## Interpretation

Gradient artifacts in simultaneous EEG-fMRI are deterministic signals driven
by MRI gradient switching. They are expected at harmonics of slice and volume
acquisition timing and can remain after average-artifact subtraction (Allen,
Josephs, & Turner, 2000; Mullinger, Yan, & Bowtell, 2011). The observed
cohort-wide, narrow-band, TR-harmonic peaks therefore justify excluding these
windows from confirmatory gamma measures. This test does not prove that all
pain-related high-frequency activity is absent from the excluded windows; it
establishes that those windows cannot be interpreted as clean neural gamma.

## References

Allen, P. J., Josephs, O., & Turner, R. (2000). A method for removing imaging
artifact from continuous EEG recorded during functional MRI. *NeuroImage, 12*,
230–239. https://doi.org/10.1006/nimg.2000.0599

Mullinger, K. J., Yan, W. X., & Bowtell, R. (2011). Reducing the gradient
artefact in simultaneous EEG-fMRI by adjusting the subject's axial position.
*NeuroImage, 54*, 1942–1950. https://doi.org/10.1016/j.neuroimage.2010.09.079
