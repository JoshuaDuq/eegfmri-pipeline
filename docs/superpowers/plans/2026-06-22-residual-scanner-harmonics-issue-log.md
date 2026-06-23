# Residual Scanner Harmonics Issue Log Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a precise, reproducible cohort-wide residual scanner-harmonics entry with participant-level QC to the pain-study issue log.

**Architecture:** Use read-only Welch PSD measurements at the BrainVision-corrected and final-clean processing boundaries. Append one issue entry to the existing log, preserve all pre-existing edits, and verify every reported participant value against the measured summaries.

**Tech Stack:** Python 3.14, MNE-Python, NumPy, SciPy, Markdown, Git

---

## File Structure

- Modify: `studies/pain_study/STUDY_ISSUES_README.md`
  - Append one cohort-wide issue entry above the existing dated entries.
  - Do not alter the pre-existing `sub-0008` baseline or `sub-0009` behavioral entries.
- Reference: `docs/superpowers/specs/2026-06-22-residual-scanner-harmonics-issue-log-design.md`
  - Defines scope, measurement boundaries, interpretation, and validation requirements.

### Task 1: Validate Participant-Level QC Inputs

**Files:**
- Read: `/Volumes/KINGSTON/EEG_fMRI_data/source_data/sub-*/eeg/*_scannerpulse_corrected.vhdr`
- Read: `/Volumes/KINGSTON/EEG_fMRI_data/derivatives/preprocessed/eeg/sub-*/eeg/*_proc-clean_raw.fif`

- [ ] **Step 1: Confirm the included participants and final-clean run counts**

Run:

```bash
find /Volumes/KINGSTON/EEG_fMRI_data/derivatives/preprocessed/eeg \
  -type f -name 'sub-000*_task-thermalactive_run-*_proc-clean_raw.fif' \
  | sed -E 's#.*(sub-[0-9]+).*#\1#' | sort | uniq -c
```

Expected numbered-participant counts:

```text
6 sub-0000
6 sub-0001
5 sub-0003
6 sub-0004
6 sub-0005
6 sub-0007
6 sub-0008
6 sub-0009
```

- [ ] **Step 2: Recompute final-clean participant summaries**

For each final-clean run, read the FIF without preloading, compute Welch PSD over all EEG
channels from 15 to 90 Hz with `n_fft=8192`, `n_per_seg=8192`, and
`n_overlap=4096` at 500 Hz, and convert the across-channel median spectrum to dB.
Use `scipy.signal.find_peaks(prominence=1.0, distance=4)` and select the greatest
prominence within each fixed region: 18-23, 38-43, 56-66, and 77-85 Hz.

Run this extraction logic for each included subject:

```python
from pathlib import Path

import mne
import numpy as np
from scipy.signal import find_peaks


DERIVATIVE_ROOT = Path(
    "/Volumes/KINGSTON/EEG_fMRI_data/derivatives/preprocessed/eeg"
)
FREQUENCY_WINDOWS = ((18.0, 23.0), (38.0, 43.0), (56.0, 66.0), (77.0, 85.0))


def strongest_window_peaks(frequencies, median_db):
    peaks, properties = find_peaks(median_db, prominence=1.0, distance=4)
    prominences = properties["prominences"]
    selected = []
    for lower, upper in FREQUENCY_WINDOWS:
        eligible = np.flatnonzero(
            (frequencies[peaks] >= lower) & (frequencies[peaks] <= upper)
        )
        if eligible.size == 0:
            raise ValueError(f"No spectral peak found in {lower}-{upper} Hz.")
        peak_index = eligible[np.argmax(prominences[eligible])]
        selected.append(
            (float(frequencies[peaks[peak_index]]), float(prominences[peak_index]))
        )
    return selected


def final_clean_summary(subject):
    paths = sorted(
        (DERIVATIVE_ROOT / subject / "eeg").glob(
            f"{subject}_task-thermalactive_run-*_proc-clean_raw.fif"
        )
    )
    run_peaks = [[] for _ in FREQUENCY_WINDOWS]
    for path in paths:
        raw = mne.io.read_raw_fif(path, preload=False, verbose=False)
        if raw.info["sfreq"] != 500.0:
            raise ValueError(f"Unexpected sampling rate in {path}: {raw.info['sfreq']}")
        spectrum = raw.compute_psd(
            method="welch",
            fmin=15.0,
            fmax=90.0,
            n_fft=8192,
            n_per_seg=8192,
            n_overlap=4096,
            picks="eeg",
            verbose=False,
        )
        median_db = 10.0 * np.log10(np.median(spectrum.get_data(), axis=0))
        for index, peak in enumerate(
            strongest_window_peaks(spectrum.freqs, median_db)
        ):
            run_peaks[index].append(peak)

    summary = []
    for peaks in run_peaks:
        peak_array = np.asarray(peaks, dtype=float)
        summary.append(
            (
                float(np.median(peak_array[:, 0])),
                float(np.median(peak_array[:, 1])),
                float(np.max(peak_array[:, 1])),
            )
        )
    return len(paths), summary
```

Expected participant summaries, formatted as `median peak frequency; median/maximum
prominence across runs`:

| Participant | Runs | 18-23 Hz | 38-43 Hz | 56-66 Hz | 77-85 Hz |
|---|---:|---|---|---|---|
| `sub-0000` | 6 | 20.02; 10.8/12.9 dB | 41.14; 23.6/25.2 dB | 61.10; 29.9/33.1 dB | 82.21; 25.9/27.7 dB |
| `sub-0001` | 6 | 20.02; 8.3/10.6 dB | 41.14; 17.9/20.5 dB | 61.10; 23.5/24.5 dB | 82.21; 22.1/24.0 dB |
| `sub-0003` | 5 | 20.02; 12.2/14.5 dB | 41.14; 24.9/27.3 dB | 61.10; 31.0/32.3 dB | 82.21; 24.7/30.3 dB |
| `sub-0004` | 6 | 21.12; 10.8/12.3 dB | 41.14; 23.2/25.1 dB | 61.10; 28.2/31.6 dB | 82.21; 23.4/26.3 dB |
| `sub-0005` | 6 | 20.02; 10.5/13.0 dB | 41.14; 23.5/24.8 dB | 61.10; 26.9/28.3 dB | 82.21; 24.3/26.8 dB |
| `sub-0007` | 6 | 20.02; 13.5/15.4 dB | 41.14; 22.3/24.5 dB | 61.10; 26.8/30.4 dB | 82.21; 25.3/27.7 dB |
| `sub-0008` | 6 | 20.02; 11.7/13.7 dB | 41.14; 25.5/26.9 dB | 61.10; 25.7/27.2 dB | 82.21; 25.0/26.7 dB |
| `sub-0009` | 6 | 20.02; 8.5/11.7 dB | 41.14; 15.9/17.3 dB | 57.19; 24.6/25.3 dB | 82.21; 19.0/21.3 dB |

- [ ] **Step 3: Validate the pre-Python boundary**

For Analyzer exports, compute the same 16.384-second, 50%-overlap Welch summary at
1,000 Hz using `n_fft=16384`, `n_per_seg=16384`, and `n_overlap=8192`. Use the fixed
representative EEG channels Fp1, F3, C3, O1, Fz, Cz, Pz, Oz, POz, FC3, PO3, and PO7.

Use the same `strongest_window_peaks` function with the following boundary reader:

```python
import re

from scipy.signal import welch


SOURCE_ROOT = Path("/Volumes/KINGSTON/EEG_fMRI_data/source_data")
REPRESENTATIVE_INDICES = np.asarray([0, 2, 4, 8, 16, 17, 18, 19, 30, 40, 44, 58])


def analyzer_summary(subject):
    paths = sorted(
        path
        for path in (SOURCE_ROOT / subject / "eeg").glob(
            "*run*_scannerpulse_corrected.vhdr"
        )
        if not path.name.startswith("._")
    )
    run_peaks = [[] for _ in FREQUENCY_WINDOWS]
    for path in paths:
        header = path.read_text(encoding="utf-8-sig")
        points = int(re.search(r"^DataPoints=(\d+)", header, re.M).group(1))
        channels = int(
            re.search(r"^NumberOfChannels=(\d+)", header, re.M).group(1)
        )
        interval_us = float(
            re.search(r"^SamplingInterval=(\d+(?:\.\d+)?)", header, re.M).group(1)
        )
        sampling_rate = 1_000_000.0 / interval_us
        if sampling_rate != 1000.0:
            raise ValueError(f"Unexpected sampling rate in {path}: {sampling_rate}")

        data = np.memmap(
            path.with_suffix(".eeg"),
            dtype="<f4",
            mode="r",
            shape=(channels, points),
        )
        frequencies, psd = welch(
            np.asarray(data[REPRESENTATIVE_INDICES]),
            fs=sampling_rate,
            nperseg=16384,
            noverlap=8192,
            axis=-1,
        )
        median_db = 10.0 * np.log10(np.median(psd, axis=0))
        for index, peak in enumerate(
            strongest_window_peaks(frequencies, median_db)
        ):
            run_peaks[index].append(peak)

    summary = []
    for peaks in run_peaks:
        peak_array = np.asarray(peaks, dtype=float)
        summary.append(
            (
                float(np.median(peak_array[:, 0])),
                float(np.median(peak_array[:, 1])),
                float(np.max(peak_array[:, 1])),
            )
        )
    return len(paths), summary
```

Expected ranges of participant median run-level prominence:

```text
18-23 Hz: 13.8-18.3 dB
38-43 Hz: 16.6-28.0 dB
56-66 Hz: 29.9-33.6 dB
77-85 Hz: 21.5-26.7 dB
```

These ranges establish that residual structure is present before MNE preprocessing. Do
not compare their absolute magnitudes directly with the final-clean table because the
channel aggregation differs.

### Task 2: Add the Cohort-Wide Issue Entry

**Files:**
- Modify: `studies/pain_study/STUDY_ISSUES_README.md`

- [ ] **Step 1: Append the approved entry above existing issues**

Add a `2026-06-22` issue titled `Cohort - Residual Scanner Harmonics In EEG` containing:

1. The shared acquisition and correction context: SyncBox, 5 kHz recording, TR 900 ms,
   `V 1` volume markers, and 21-volume sliding-average BrainVision correction.
2. The exact final-clean QC method from Task 1.
3. The complete eight-row final-clean table from Task 1.
4. The Analyzer-boundary prominence ranges from Task 1.
5. A decision that this is an unresolved cohort-wide issue, not an automatic subject
   exclusion, and that current scanner-sensitive beta/gamma measures are not yet cleared
   for confirmatory interpretation.
6. A follow-up requiring an outcome-blind correction benchmark, estimator-specific
   residual handling, prespecified run/subject acceptance criteria, and cohort-wide
   reprocessing only after one canonical method is validated.

- [ ] **Step 2: Preserve existing issue entries**

Run:

```bash
git diff -- studies/pain_study/STUDY_ISSUES_README.md
```

Expected: the new cohort entry is added; the existing uncommitted `sub-0008` and
`sub-0009` entries remain byte-for-byte unchanged.

### Task 3: Verify Accuracy And Markdown Hygiene

**Files:**
- Verify: `studies/pain_study/STUDY_ISSUES_README.md`

- [ ] **Step 1: Verify participant coverage**

Run a bounded check over the new section and confirm that each of `sub-0000`,
`sub-0001`, `sub-0003`, `sub-0004`, `sub-0005`, `sub-0007`, `sub-0008`, and
`sub-0009` appears exactly once in the QC table.

- [ ] **Step 2: Verify values against Task 1 output**

Check all 32 frequency/prominence cells and eight run counts against the measured table.
Expected: exact agreement after one-decimal dB rounding and two-decimal frequency
rounding.

- [ ] **Step 3: Run repository formatting checks**

Run:

```bash
git diff --check -- studies/pain_study/STUDY_ISSUES_README.md
```

Expected: no output and exit code 0.

- [ ] **Step 4: Leave the issue-log edit uncommitted**

Do not commit `studies/pain_study/STUDY_ISSUES_README.md` because it contains pre-existing
user edits in the same file. Report the verification results and leave the combined file
available for user review.
