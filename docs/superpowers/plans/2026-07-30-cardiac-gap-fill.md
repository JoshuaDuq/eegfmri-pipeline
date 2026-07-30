# Cardiac Gap-Fill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover the heartbeats BrainVision Analyzer never marked and remove the ballistocardiogram only in those stretches, leaving Analyzer's correction intact everywhere else.

**Architecture:** A new `eeg_pipeline/preprocessing/bcg/` package with four focused modules — `metrics` (the referee: held-out R-locked reduction against a circular-shift null, plus preservation measures), `sources` (pairing and validating the two Analyzer exports), `detect` (gap finding and QRS template matching), and `correct` (OBS and AAS, array in / array out). A study script drives them with `benchmark` / `apply` / `verify` subcommands, mirroring `remove_line_comb.py`.

**Tech Stack:** Python 3.14, MNE 1.12.1 (`mne.preprocessing.apply_pca_obs`), NumPy 1.26.4, SciPy 1.17.1, NeuroKit2 0.2.12, pytest.

## Global Constraints

- numpy stays pinned `>=1.24.0,<2.0.0` (`pyproject.toml:32`). `neurokit2==0.2.12` is the only version that resolves under it; 0.2.13 pulls numpy 2.5.1.
- Never run the full pytest suite (~9 min). Verify with targeted node IDs only.
- No unit test may read `/Volumes/KINGSTON`. All pytest fixtures are synthetic; real-data checks live in the CLI's `benchmark` and `verify` subcommands.
- The referee reports measurements only. No pass/fail verdict is computed or stored in library code; thresholds live in the caller's config.
- Uncorrected export root: `/Volumes/KINGSTON/EEG_fMRI_data/source_data/processed_scanner_artifact_with_pulse_markers_no_bcg_correction` (BINARY/IEEE_FLOAT_32, 104 recordings).
- Corrected export root: `data/source_data/processed_trimmed_0-60s_30-115bpm_marker_template`.
- Analyzer R markers appear in MNE annotations with description `Pulse Artifact/R`; match on the final `/`-separated segment equal to `R`.
- Work directly in the main checkout. Do not create git worktrees.

---

### Task 1: Package scaffold and export pairing

**Files:**
- Create: `eeg_pipeline/preprocessing/bcg/__init__.py`
- Create: `eeg_pipeline/preprocessing/bcg/sources.py`
- Test: `tests/preprocessing/bcg/__init__.py`
- Test: `tests/preprocessing/bcg/test_sources.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `RunPair(subject: str, run: str, uncorrected_vhdr: Path, corrected_vhdr: Path)`; `discover_run_pairs(uncorrected_root: Path, corrected_root: Path) -> list[RunPair]`; `PairValidation(subject, run, n_times_uncorrected, n_times_corrected, aligned, ecg_max_abs_diff_uv, status)`; `validate_pair(pair: RunPair) -> PairValidation`.

- [ ] **Step 1: Write the failing test**

```python
# tests/preprocessing/bcg/test_sources.py
from pathlib import Path

from eeg_pipeline.preprocessing.bcg.sources import discover_run_pairs


def _touch(root: Path, name: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text("")


def test_discover_pairs_matches_subject_and_run(tmp_path):
    unc, cor = tmp_path / "unc", tmp_path / "cor"
    _touch(unc, "ThermalPainEEGFMRI_run1_sub0009_2026-06-22_a_pulse_markers.vhdr")
    _touch(unc, "ThermalPainEEGFMRI_run2_sub0009_2026-06-22_b_pulse_markers.vhdr")
    _touch(cor, "ThermalPainEEGFMRI_run1_sub0009_2026-06-22_a_corrected.vhdr")

    pairs = discover_run_pairs(unc, cor)

    assert [(p.subject, p.run) for p in pairs] == [("sub0009", "1")]


def test_discover_pairs_ignores_appledouble_files(tmp_path):
    unc, cor = tmp_path / "unc", tmp_path / "cor"
    _touch(unc, "ThermalPainEEGFMRI_run1_sub0005_x_pulse_markers.vhdr")
    _touch(unc, "._ThermalPainEEGFMRI_run1_sub0005_x_pulse_markers.vhdr")
    _touch(cor, "ThermalPainEEGFMRI_run1_sub0005_x_corrected.vhdr")

    pairs = discover_run_pairs(unc, cor)

    assert len(pairs) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/preprocessing/bcg/test_sources.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eeg_pipeline.preprocessing.bcg'`

- [ ] **Step 3: Write minimal implementation**

```python
# eeg_pipeline/preprocessing/bcg/__init__.py
"""Ballistocardiogram gap-fill: referee, beat recovery, and confined correction."""
```

```python
# eeg_pipeline/preprocessing/bcg/sources.py
"""Pair the two Analyzer exports and prove they describe the same samples.

The pulse-markers-only export carries the uncorrected ballistocardiogram and Analyzer's
R marks; the corrected export is what currently feeds BIDS. Substituting gap stretches
from one into the other is only valid while they stay sample-aligned, so that identity is
measured per run rather than assumed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

RUN_PATTERN = re.compile(r"_run(?P<run>\d+)_(?P<subject>sub\d+)_")
BASELINE_PATTERN = re.compile(r"^BaselineEEG_(?P<subject>sub\d+)_")


@dataclass(frozen=True)
class RunPair:
    subject: str
    run: str
    uncorrected_vhdr: Path
    corrected_vhdr: Path


def _key(path: Path) -> tuple[str, str] | None:
    name = path.name
    match = RUN_PATTERN.search(name)
    if match:
        return match.group("subject"), match.group("run")
    baseline = BASELINE_PATTERN.match(name)
    if baseline:
        return baseline.group("subject"), "baseline"
    return None


def _index(root: Path) -> dict[tuple[str, str], Path]:
    found: dict[tuple[str, str], Path] = {}
    for path in sorted(root.glob("*.vhdr")):
        if path.name.startswith("._"):
            continue
        key = _key(path)
        if key is not None:
            found.setdefault(key, path)
    return found


def discover_run_pairs(uncorrected_root: Path, corrected_root: Path) -> list[RunPair]:
    """Every recording present in both exports, keyed by subject and run."""
    uncorrected = _index(Path(uncorrected_root))
    corrected = _index(Path(corrected_root))
    return [
        RunPair(subject, run, uncorrected[(subject, run)], corrected[(subject, run)])
        for subject, run in sorted(uncorrected.keys() & corrected.keys())
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/preprocessing/bcg/test_sources.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Add the validation test**

```python
# append to tests/preprocessing/bcg/test_sources.py
import numpy as np
import mne

from eeg_pipeline.preprocessing.bcg.sources import RunPair, validate_pair


def _write_brainvision(root: Path, stem: str, data_uv, sfreq=1000.0):
    root.mkdir(parents=True, exist_ok=True)
    names = [f"EEG{i:02d}" for i in range(data_uv.shape[0] - 1)] + ["ECG"]
    types = ["eeg"] * (data_uv.shape[0] - 1) + ["ecg"]
    info = mne.create_info(names, sfreq, ch_types=types)
    raw = mne.io.RawArray(data_uv * 1e-6, info, verbose="ERROR")
    mne.export.export_raw(root / f"{stem}.vhdr", raw, fmt="brainvision",
                          overwrite=True, verbose="ERROR")
    return root / f"{stem}.vhdr"


def test_validate_pair_reports_alignment_and_ecg_identity(tmp_path):
    rng = np.random.default_rng(0)
    shared_ecg = rng.normal(0, 50, 4000)
    base = rng.normal(0, 10, (3, 4000))

    unc = np.vstack([base, shared_ecg])
    cor = np.vstack([base + 5.0, shared_ecg])   # EEG differs, ECG identical

    a = _write_brainvision(tmp_path / "unc", "ThermalPainEEGFMRI_run1_sub0009_a", unc)
    b = _write_brainvision(tmp_path / "cor", "ThermalPainEEGFMRI_run1_sub0009_b", cor)

    result = validate_pair(RunPair("sub0009", "1", a, b))

    assert result.aligned is True
    assert result.ecg_max_abs_diff_uv < 1e-3
    assert result.status == "ok"


def test_validate_pair_flags_length_mismatch(tmp_path):
    rng = np.random.default_rng(1)
    a = _write_brainvision(tmp_path / "unc", "ThermalPainEEGFMRI_run1_sub0009_a",
                           rng.normal(0, 10, (2, 4000)))
    b = _write_brainvision(tmp_path / "cor", "ThermalPainEEGFMRI_run1_sub0009_b",
                           rng.normal(0, 10, (2, 3000)))

    result = validate_pair(RunPair("sub0009", "1", a, b))

    assert result.aligned is False
    assert result.status == "length_mismatch"
```

- [ ] **Step 6: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/preprocessing/bcg/test_sources.py -v`
Expected: FAIL with `ImportError: cannot import name 'validate_pair'`

- [ ] **Step 7: Implement validation**

```python
# append to eeg_pipeline/preprocessing/bcg/sources.py
import numpy as np

ECG_IDENTITY_TOLERANCE_UV = 1e-3


@dataclass(frozen=True)
class PairValidation:
    subject: str
    run: str
    n_times_uncorrected: int
    n_times_corrected: int
    aligned: bool
    ecg_max_abs_diff_uv: float
    status: str


def validate_pair(pair: RunPair, ecg_channel: str = "ECG") -> PairValidation:
    """Measure sample alignment and ECG identity for one paired recording.

    Analyzer's pulse correction modifies EEG only, so a non-zero ECG difference means the
    two files are not the same recording and the pair must not be used.
    """
    import mne

    mne.set_log_level("ERROR")
    left = mne.io.read_raw_brainvision(pair.uncorrected_vhdr, preload=True, verbose="ERROR")
    right = mne.io.read_raw_brainvision(pair.corrected_vhdr, preload=True, verbose="ERROR")

    aligned = left.n_times == right.n_times
    difference = float("nan")
    status = "ok"
    if not aligned:
        status = "length_mismatch"
    elif ecg_channel not in left.ch_names or ecg_channel not in right.ch_names:
        status = "missing_ecg"
    else:
        a = left.copy().pick([ecg_channel]).get_data()[0] * 1e6
        b = right.copy().pick([ecg_channel]).get_data()[0] * 1e6
        difference = float(np.abs(a - b).max())
        if difference > ECG_IDENTITY_TOLERANCE_UV:
            status = "ecg_mismatch"

    return PairValidation(
        subject=pair.subject,
        run=pair.run,
        n_times_uncorrected=int(left.n_times),
        n_times_corrected=int(right.n_times),
        aligned=aligned,
        ecg_max_abs_diff_uv=difference,
        status=status,
    )
```

- [ ] **Step 8: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/preprocessing/bcg/test_sources.py -v`
Expected: PASS (4 passed)

- [ ] **Step 9: Commit**

```bash
git add eeg_pipeline/preprocessing/bcg/ tests/preprocessing/bcg/
git commit -m "feat(bcg): pair and validate the two Analyzer exports"
```

---

### Task 2: Referee — held-out R-locked reduction

**Files:**
- Create: `eeg_pipeline/preprocessing/bcg/metrics.py`
- Test: `tests/preprocessing/bcg/test_metrics.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `epoch_stack(data_uv: np.ndarray, onsets_samples: np.ndarray, sfreq: float, window: tuple[float, float]) -> np.ndarray` shaped `(n_channels, n_epochs, n_times)`; `held_out_reduction(data_uv, onsets_samples, sfreq, window) -> tuple[np.ndarray, np.ndarray]` returning per-channel reduction fraction and per-channel template peak-to-peak.

The reduction must be cross-validated — template from even-indexed epochs, scored on odd-indexed epochs — so that averaging noise contributes approximately zero. A within-sample statistic would report several percent on artifact-free data.

- [ ] **Step 1: Write the failing test**

```python
# tests/preprocessing/bcg/test_metrics.py
import numpy as np

from eeg_pipeline.preprocessing.bcg.metrics import epoch_stack, held_out_reduction

SFREQ = 1000.0
WINDOW = (-0.3, 0.7)


def _beats(n=200, rr=0.9, jitter=0.02, seed=0, start=5.0):
    rng = np.random.default_rng(seed)
    intervals = rr + rng.normal(0, jitter, n)
    return start + np.cumsum(intervals)


def _noise(n_channels, duration_s, seed=0, sd=10.0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, sd, (n_channels, int(duration_s * SFREQ)))


def _inject(data, beats, sfreq, amplitude_uv):
    out = data.copy()
    t = np.arange(int(0.4 * sfreq)) / sfreq
    shape = amplitude_uv * np.sin(2 * np.pi * 5.0 * t) * np.exp(-t / 0.1)
    for beat in beats:
        start = int(round(beat * sfreq))
        stop = start + shape.size
        if stop <= out.shape[1]:
            out[:, start:stop] += shape
    return out


def test_epoch_stack_shape_and_mean_removal():
    data = _noise(4, 60)
    onsets = np.round(_beats(50) * SFREQ).astype(int)

    stack = epoch_stack(data, onsets, SFREQ, WINDOW)

    assert stack.shape[0] == 4
    assert stack.shape[2] == int(round((WINDOW[1] - WINDOW[0]) * SFREQ))
    assert np.allclose(stack.mean(axis=2), 0.0, atol=1e-9)


def test_held_out_reduction_is_near_zero_without_artifact():
    data = _noise(8, 240, seed=1)
    onsets = np.round(_beats(250, seed=2) * SFREQ).astype(int)

    reduction, _ = held_out_reduction(data, onsets, SFREQ, WINDOW)

    assert np.nanmax(reduction) < 0.02


def test_held_out_reduction_recovers_injected_artifact():
    beats = _beats(250, seed=3)
    clean = _noise(8, 240, seed=4)
    contaminated = _inject(clean, beats, SFREQ, amplitude_uv=30.0)
    onsets = np.round(beats * SFREQ).astype(int)

    reduction, template_pp = held_out_reduction(contaminated, onsets, SFREQ, WINDOW)

    assert np.nanmax(reduction) > 0.20
    assert np.nanmax(template_pp) > 15.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/preprocessing/bcg/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eeg_pipeline.preprocessing.bcg.metrics'`

- [ ] **Step 3: Write minimal implementation**

```python
# eeg_pipeline/preprocessing/bcg/metrics.py
"""Measurements for event-locked artifact, and the nulls that make them interpretable.

Every statistic here is held out or controlled. Raw event-locked amplitude is not a
measurement of artifact: averaging 500 epochs of ordinary EEG produces several microvolts
of peak-to-peak by itself, and on this cohort a naive peak-to-peak read 5.71 uV against
its own null of 6.99 uV.

No function returns a verdict. Thresholds belong to the caller.
"""

from __future__ import annotations

import numpy as np


def epoch_stack(
    data_uv: np.ndarray,
    onsets_samples: np.ndarray,
    sfreq: float,
    window: tuple[float, float],
) -> np.ndarray:
    """Mean-removed epochs shaped (n_channels, n_epochs, n_times).

    Epochs running off either end of the recording are dropped rather than padded.
    """
    pre = int(round(window[0] * sfreq))
    length = int(round((window[1] - window[0]) * sfreq))
    starts = np.asarray(onsets_samples, dtype=int) + pre
    keep = (starts >= 0) & (starts + length <= data_uv.shape[1])
    starts = starts[keep]
    if starts.size == 0:
        return np.empty((data_uv.shape[0], 0, length))
    index = starts[:, None] + np.arange(length)[None, :]
    epochs = data_uv[:, index]
    return epochs - epochs.mean(axis=2, keepdims=True)


def held_out_reduction(
    data_uv: np.ndarray,
    onsets_samples: np.ndarray,
    sfreq: float,
    window: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Variance removed from held-out epochs by an event-locked template.

    The template is built from even-indexed epochs and scored on odd-indexed ones, so
    averaging noise cannot inflate the result: with no event-locked structure the
    expectation is approximately zero.

    Returns (per-channel reduction fraction, per-channel template peak-to-peak in uV).
    """
    epochs = epoch_stack(data_uv, onsets_samples, sfreq, window)
    n_channels = data_uv.shape[0]
    if epochs.shape[1] < 4:
        nan = np.full(n_channels, np.nan)
        return nan, nan

    template = epochs[:, 0::2, :].mean(axis=1, keepdims=True)
    test = epochs[:, 1::2, :]
    before = test.var(axis=(1, 2))
    after = (test - template).var(axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        reduction = 1.0 - (after / before)
    flat = template[:, 0, :]
    return reduction, flat.max(axis=1) - flat.min(axis=1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/preprocessing/bcg/test_metrics.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add eeg_pipeline/preprocessing/bcg/metrics.py tests/preprocessing/bcg/test_metrics.py
git commit -m "feat(bcg): held-out event-locked variance reduction"
```

---

### Task 3: Referee — circular-shift null and the naive-statistic regression

**Files:**
- Modify: `eeg_pipeline/preprocessing/bcg/metrics.py`
- Test: `tests/preprocessing/bcg/test_metrics.py`

**Interfaces:**
- Consumes: `epoch_stack`, `held_out_reduction` from Task 2.
- Produces: `ReductionResult(per_channel: np.ndarray, template_pp_uv: np.ndarray, null_max: float, max_value: float, max_channel: int, channels_above_null: int, n_epochs: int, n_surrogate: int)`; `rlocked_reduction(data_uv, onset_seconds: np.ndarray, sfreq, *, window, n_surrogate: int = 20, seed: int = 0) -> ReductionResult`; `naive_peak_to_peak(data_uv, onsets_samples, sfreq, window, measure: tuple[float, float]) -> np.ndarray`.

The null circularly shifts the whole event train by one random constant. Drawing fresh random times instead destroys the train's periodicity as well as its phase, which makes it anticonservative: any quasi-periodic train scores against it.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/preprocessing/bcg/test_metrics.py
from eeg_pipeline.preprocessing.bcg.metrics import (
    ReductionResult,
    naive_peak_to_peak,
    rlocked_reduction,
)


def test_rlocked_reduction_reports_nothing_above_null_on_clean_data():
    data = _noise(8, 240, seed=5)
    beats = _beats(250, seed=6)

    result = rlocked_reduction(data, beats, SFREQ, window=WINDOW, n_surrogate=15, seed=0)

    assert isinstance(result, ReductionResult)
    assert result.channels_above_null == 0


def test_rlocked_reduction_flags_every_channel_when_artifact_present():
    beats = _beats(250, seed=7)
    data = _inject(_noise(8, 240, seed=8), beats, SFREQ, amplitude_uv=30.0)

    result = rlocked_reduction(data, beats, SFREQ, window=WINDOW, n_surrogate=15, seed=0)

    assert result.channels_above_null == 8
    assert result.max_value > 0.20


def test_naive_peak_to_peak_fails_where_held_out_statistic_does_not():
    """The naive statistic must stay in the codebase only as a documented failure.

    On artifact-free data it returns several microvolts, which is what made an earlier
    reported BCG amplitude meaningless.
    """
    data = _noise(32, 240, seed=9)
    onsets = np.round(_beats(250, seed=10) * SFREQ).astype(int)

    naive = naive_peak_to_peak(data, onsets, SFREQ, WINDOW, measure=(0.0, 0.6))
    reduction, _ = held_out_reduction(data, onsets, SFREQ, WINDOW)

    assert np.nanmax(naive) > 1.0
    assert np.nanmax(reduction) < 0.02
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/preprocessing/bcg/test_metrics.py -v`
Expected: FAIL with `ImportError: cannot import name 'ReductionResult'`

- [ ] **Step 3: Write the implementation**

```python
# append to eeg_pipeline/preprocessing/bcg/metrics.py
from dataclasses import dataclass


@dataclass(frozen=True)
class ReductionResult:
    per_channel: np.ndarray
    template_pp_uv: np.ndarray
    null_max: float
    max_value: float
    max_channel: int
    channels_above_null: int
    n_epochs: int
    n_surrogate: int


def circular_shift_null(
    data_uv: np.ndarray,
    onset_seconds: np.ndarray,
    sfreq: float,
    window: tuple[float, float],
    n_surrogate: int,
    seed: int,
) -> np.ndarray:
    """Maximum held-out reduction under random circular shifts of the whole event train.

    Shifting preserves the inter-event structure exactly and changes only the train's
    alignment to the data, which is what isolates phase-locking from mere periodicity.
    """
    rng = np.random.default_rng(seed)
    duration = data_uv.shape[1] / sfreq
    out = np.empty(n_surrogate)
    for index in range(n_surrogate):
        shifted = np.sort((np.asarray(onset_seconds) + rng.uniform(2.0, duration - 2.0)) % duration)
        reduction, _ = held_out_reduction(
            data_uv, np.round(shifted * sfreq).astype(int), sfreq, window
        )
        out[index] = np.nanmax(reduction)
    return out


def rlocked_reduction(
    data_uv: np.ndarray,
    onset_seconds: np.ndarray,
    sfreq: float,
    *,
    window: tuple[float, float] = (-0.3, 0.7),
    n_surrogate: int = 20,
    seed: int = 0,
) -> ReductionResult:
    """Held-out event-locked reduction with its circular-shift null."""
    onsets = np.asarray(onset_seconds, dtype=float)
    samples = np.round(onsets * sfreq).astype(int)
    reduction, template_pp = held_out_reduction(data_uv, samples, sfreq, window)
    null = circular_shift_null(data_uv, onsets, sfreq, window, n_surrogate, seed)
    null_max = float(np.max(null)) if null.size else float("nan")
    finite = np.nan_to_num(reduction, nan=-np.inf)
    return ReductionResult(
        per_channel=reduction,
        template_pp_uv=template_pp,
        null_max=null_max,
        max_value=float(np.nanmax(reduction)),
        max_channel=int(np.argmax(finite)),
        channels_above_null=int(np.sum(finite > null_max)),
        n_epochs=int(epoch_stack(data_uv, samples, sfreq, window).shape[1]),
        n_surrogate=int(n_surrogate),
    )


def naive_peak_to_peak(
    data_uv: np.ndarray,
    onsets_samples: np.ndarray,
    sfreq: float,
    window: tuple[float, float],
    measure: tuple[float, float],
) -> np.ndarray:
    """Peak-to-peak of the event-locked average. Retained only as a documented failure.

    This is not a measurement of artifact. It is kept so the regression test can assert it
    still misbehaves, and so nobody reintroduces it believing it is safe.
    """
    epochs = epoch_stack(data_uv, onsets_samples, sfreq, window)
    if epochs.shape[1] == 0:
        return np.full(data_uv.shape[0], np.nan)
    evoked = epochs.mean(axis=1)
    lo = int(round((measure[0] - window[0]) * sfreq))
    hi = int(round((measure[1] - window[0]) * sfreq))
    segment = evoked[:, lo:hi]
    return segment.max(axis=1) - segment.min(axis=1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/preprocessing/bcg/test_metrics.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add eeg_pipeline/preprocessing/bcg/metrics.py tests/preprocessing/bcg/test_metrics.py
git commit -m "feat(bcg): circular-shift null and naive-statistic regression"
```

---

### Task 4: Referee — preservation measures

**Files:**
- Modify: `eeg_pipeline/preprocessing/bcg/metrics.py`
- Test: `tests/preprocessing/bcg/test_metrics.py`

**Interfaces:**
- Consumes: `epoch_stack` from Task 2.
- Produces: `band_power(data_uv, sfreq, band: tuple[float, float], picks: np.ndarray | None = None) -> float`; `band_retention(before_uv, after_uv, sfreq, band, picks=None) -> float`; `lock_ratio(signal_uv: np.ndarray, onset_seconds, sfreq, window) -> float`; `EvokedPreservation(correlation: np.ndarray, amplitude_ratio: np.ndarray)`; `evoked_preservation(before_uv, after_uv, onset_seconds, sfreq, window) -> EvokedPreservation`.

`band_retention` alone is **not** a preservation measure: the artifact contributes power inside every band of interest, so removing it necessarily lowers band power. It becomes interpretable only when the same correction is also run at shifted event times (the sham, wired up in Task 8).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/preprocessing/bcg/test_metrics.py
from eeg_pipeline.preprocessing.bcg.metrics import (
    band_retention,
    evoked_preservation,
    lock_ratio,
)


def test_band_retention_is_one_when_nothing_changes():
    data = _noise(4, 60, seed=11)

    assert band_retention(data, data.copy(), SFREQ, (8.0, 13.0)) == 1.0


def test_band_retention_detects_attenuation():
    data = _noise(4, 60, seed=12)

    retained = band_retention(data, data * 0.5, SFREQ, (8.0, 13.0))

    assert 0.2 < retained < 0.3


def test_lock_ratio_is_high_for_events_on_a_real_deflection():
    beats = _beats(200, seed=13)
    signal = _inject(_noise(1, 200, seed=14, sd=5.0), beats, SFREQ, amplitude_uv=200.0)

    on_beat = lock_ratio(signal[0], beats, SFREQ, (-0.2, 0.4))
    off_beat = lock_ratio(signal[0], beats + 0.45, SFREQ, (-0.2, 0.4))

    assert on_beat > off_beat


def test_evoked_preservation_is_perfect_for_untouched_data():
    data = _noise(4, 120, seed=15)
    events = np.arange(5.0, 110.0, 4.0)

    result = evoked_preservation(data, data.copy(), events, SFREQ, (-0.1, 0.5))

    assert np.allclose(result.correlation, 1.0, atol=1e-6)
    assert np.allclose(result.amplitude_ratio, 1.0, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/preprocessing/bcg/test_metrics.py -v`
Expected: FAIL with `ImportError: cannot import name 'band_retention'`

- [ ] **Step 3: Write the implementation**

```python
# append to eeg_pipeline/preprocessing/bcg/metrics.py
from scipy.signal import welch


def band_power(
    data_uv: np.ndarray,
    sfreq: float,
    band: tuple[float, float],
    picks: np.ndarray | None = None,
) -> float:
    """Mean power in a band, averaged over the selected channels."""
    selected = data_uv if picks is None else data_uv[picks]
    nperseg = min(int(4 * sfreq), selected.shape[1])
    freqs, power = welch(selected, fs=sfreq, nperseg=nperseg, axis=1)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return float(np.mean(power[:, mask]))


def band_retention(
    before_uv: np.ndarray,
    after_uv: np.ndarray,
    sfreq: float,
    band: tuple[float, float],
    picks: np.ndarray | None = None,
) -> float:
    """Fraction of band power surviving a correction.

    Not a preservation measure on its own -- the artifact contributes power inside the
    band, so a value below 1 is expected even for a perfect correction. Pair it with the
    same quantity computed under a sham correction.
    """
    baseline = band_power(before_uv, sfreq, band, picks)
    if baseline == 0.0:
        return float("nan")
    return band_power(after_uv, sfreq, band, picks) / baseline


def lock_ratio(
    signal_uv: np.ndarray,
    onset_seconds: np.ndarray,
    sfreq: float,
    window: tuple[float, float],
) -> float:
    """Event-locked average peak-to-peak over mean single-trial SD, for one channel.

    Used on the ECG channel to judge whether a marker set sits on the QRS complex rather
    than on a T-wave or on noise.
    """
    samples = np.round(np.asarray(onset_seconds) * sfreq).astype(int)
    stack = epoch_stack(signal_uv[None, :], samples, sfreq, window)
    if stack.shape[1] == 0:
        return float("nan")
    evoked = stack.mean(axis=1)[0]
    single = float(np.sqrt(stack[0].var(axis=1)).mean())
    if single == 0.0:
        return float("nan")
    return float((evoked.max() - evoked.min()) / single)


@dataclass(frozen=True)
class EvokedPreservation:
    correlation: np.ndarray
    amplitude_ratio: np.ndarray


def evoked_preservation(
    before_uv: np.ndarray,
    after_uv: np.ndarray,
    onset_seconds: np.ndarray,
    sfreq: float,
    window: tuple[float, float],
) -> EvokedPreservation:
    """Per-channel survival of the stimulus-locked average.

    Stimulus events are not cardiac-locked, so the ballistocardiogram averages out of this
    waveform. What remains is brain response, which a correction must not distort.
    """
    samples = np.round(np.asarray(onset_seconds) * sfreq).astype(int)
    left = epoch_stack(before_uv, samples, sfreq, window).mean(axis=1)
    right = epoch_stack(after_uv, samples, sfreq, window).mean(axis=1)

    correlation = np.empty(left.shape[0])
    ratio = np.empty(left.shape[0])
    for channel in range(left.shape[0]):
        a, b = left[channel], right[channel]
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        correlation[channel] = float(np.dot(a, b) / denominator) if denominator else np.nan
        span = a.max() - a.min()
        ratio[channel] = float((b.max() - b.min()) / span) if span else np.nan
    return EvokedPreservation(correlation=correlation, amplitude_ratio=ratio)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/preprocessing/bcg/test_metrics.py -v`
Expected: PASS (10 passed)

- [ ] **Step 5: Commit**

```bash
git add eeg_pipeline/preprocessing/bcg/metrics.py tests/preprocessing/bcg/test_metrics.py
git commit -m "feat(bcg): preservation measures for the second arm"
```

---

### Task 5: Gap finding from Analyzer's own markers

**Files:**
- Create: `eeg_pipeline/preprocessing/bcg/detect.py`
- Test: `tests/preprocessing/bcg/test_detect.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Gap(start_s: float, end_s: float, preceding_beat_s: float, following_beat_s: float, duration_s: float)`; `find_gaps(beat_seconds: np.ndarray, *, minimum_seconds: float = 2.0, factor: float = 2.0) -> list[Gap]`; `gap_summary(beat_seconds, duration_s, **kwargs) -> dict[str, float]`; `read_analyzer_beats(vhdr_path) -> np.ndarray`.

A gap needs both tests: longer than `minimum_seconds` in absolute terms, and longer than `factor` times the run's own median RR. The absolute floor alone would flag ordinary bradycardia; the relative one alone would flag a run whose median RR is already inflated by widespread detection failure.

- [ ] **Step 1: Write the failing test**

```python
# tests/preprocessing/bcg/test_detect.py
import numpy as np

from eeg_pipeline.preprocessing.bcg.detect import find_gaps, gap_summary


def test_no_gaps_in_a_regular_train():
    beats = np.arange(5.0, 100.0, 0.9)

    assert find_gaps(beats) == []


def test_finds_a_single_deleted_stretch():
    beats = np.arange(5.0, 100.0, 0.9)
    kept = beats[(beats < 40.0) | (beats > 52.0)]

    gaps = find_gaps(kept)

    assert len(gaps) == 1
    assert gaps[0].preceding_beat_s < 40.0
    assert gaps[0].following_beat_s > 52.0
    assert 11.0 < gaps[0].duration_s < 13.5


def test_ordinary_slow_rate_is_not_a_gap():
    """A steady 40 bpm run has RR of 1.5 s throughout and contains no missing beats."""
    beats = np.arange(5.0, 200.0, 1.5)

    assert find_gaps(beats) == []


def test_gap_summary_reports_time_and_implied_missing_beats():
    beats = np.arange(5.0, 100.0, 0.9)
    kept = beats[(beats < 40.0) | (beats > 52.0)]

    summary = gap_summary(kept, duration_s=100.0)

    assert summary["n_gaps"] == 1
    assert summary["gap_seconds"] > 11.0
    assert summary["implied_missing_beats"] > 11
    assert 0.0 < summary["gap_fraction"] < 0.2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/preprocessing/bcg/test_detect.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eeg_pipeline.preprocessing.bcg.detect'`

- [ ] **Step 3: Write the implementation**

```python
# eeg_pipeline/preprocessing/bcg/detect.py
"""Find the stretches Analyzer left unmarked, and recover the beats inside them.

Analyzer detects conservatively: what it marks sits on the QRS complex more reliably than
any general-purpose detector measured on this cohort, but it marks too little. Across the
104 exports, 85 recordings contain at least one RR interval above 2 s, totalling 6,954 s
with a largest single gap of 53.9 s. An 11 s interval is not a heartbeat rate, so the
gaps are provable from Analyzer's own markers with no detector involved.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Gap:
    start_s: float
    end_s: float
    preceding_beat_s: float
    following_beat_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def read_analyzer_beats(vhdr_path: Path | str) -> np.ndarray:
    """Onsets in seconds of Analyzer's R markers."""
    import mne

    mne.set_log_level("ERROR")
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=False, verbose="ERROR")
    onsets = [
        onset
        for onset, description in zip(raw.annotations.onset, raw.annotations.description)
        if description.split("/")[-1].strip() == "R"
    ]
    return np.asarray(sorted(onsets), dtype=float)


def find_gaps(
    beat_seconds: np.ndarray,
    *,
    minimum_seconds: float = 2.0,
    factor: float = 2.0,
) -> list[Gap]:
    """Intervals that are both absolutely long and long for this run.

    Both tests are required. The absolute floor alone flags ordinary bradycardia; the
    relative test alone flags a run whose median RR is already inflated because detection
    failed nearly everywhere.
    """
    beats = np.sort(np.asarray(beat_seconds, dtype=float))
    if beats.size < 3:
        return []
    intervals = np.diff(beats)
    threshold = max(minimum_seconds, factor * float(np.median(intervals)))
    return [
        Gap(
            start_s=float(beats[index]),
            end_s=float(beats[index + 1]),
            preceding_beat_s=float(beats[index]),
            following_beat_s=float(beats[index + 1]),
        )
        for index in np.flatnonzero(intervals > threshold)
    ]


def gap_summary(
    beat_seconds: np.ndarray,
    duration_s: float,
    *,
    minimum_seconds: float = 2.0,
    factor: float = 2.0,
) -> dict[str, float]:
    """Per-run gap totals, including how many beats the gaps imply are missing."""
    beats = np.sort(np.asarray(beat_seconds, dtype=float))
    gaps = find_gaps(beats, minimum_seconds=minimum_seconds, factor=factor)
    intervals = np.diff(beats) if beats.size > 1 else np.array([np.nan])
    median_rr = float(np.median(intervals)) if beats.size > 1 else float("nan")
    total = float(sum(gap.duration_s for gap in gaps))
    return {
        "n_beats": float(beats.size),
        "n_gaps": float(len(gaps)),
        "gap_seconds": total,
        "gap_fraction": total / duration_s if duration_s else float("nan"),
        "median_rr_s": median_rr,
        "max_rr_s": float(intervals.max()) if intervals.size else float("nan"),
        "implied_missing_beats": total / median_rr if median_rr else float("nan"),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/preprocessing/bcg/test_detect.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add eeg_pipeline/preprocessing/bcg/detect.py tests/preprocessing/bcg/test_detect.py
git commit -m "feat(bcg): find Analyzer's marker gaps from its own RR intervals"
```

---

### Task 6: Beat recovery by QRS template matching

**Files:**
- Modify: `eeg_pipeline/preprocessing/bcg/detect.py`
- Test: `tests/preprocessing/bcg/test_detect.py`

**Interfaces:**
- Consumes: `Gap`, `find_gaps` from Task 5; `lock_ratio` from Task 4.
- Produces: `RecoverySettings(template_window=(-0.2, 0.4), correlation_threshold=0.5, refractory_fraction=0.5, iterations=2)`; `BeatQuality(analyzer_lock_ratio, recovered_lock_ratio, combined_lock_ratio, rr_median_s, rr_min_s, rr_max_s, implied_bpm, refractory_violations, recovered_beats, gap_seconds_before, gap_seconds_after, status)`; `BeatRecovery(analyzer_beats, recovered_beats, combined_beats, quality: BeatQuality)`; `recover_beats(ecg_uv: np.ndarray, analyzer_beats: np.ndarray, sfreq: float, *, settings: RecoverySettings = RecoverySettings()) -> BeatRecovery`.

Matching is on QRS **shape**, by normalised cross-correlation against a template built from Analyzer's own beats. Amplitude thresholding is what fails here: the magnetohydrodynamic effect inflates the T-wave in the bore, and on sub-0000 run 1 that drove MNE to 1073 beats and NeuroKit2's Pan-Tompkins to 1566 against Analyzer's 689, with lock ratios of 1.58 and 0.71 against Analyzer's 4.39.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/preprocessing/bcg/test_detect.py
from eeg_pipeline.preprocessing.bcg.detect import RecoverySettings, recover_beats

SFREQ = 1000.0


def _synthetic_ecg(beats, duration_s, sfreq=SFREQ, t_wave_uv=0.0, seed=0):
    """ECG with a sharp QRS and an optionally inflated T-wave.

    The T-wave models the magnetohydrodynamic effect, which is what defeats
    amplitude-threshold detectors inside the bore.
    """
    rng = np.random.default_rng(seed)
    signal = rng.normal(0, 5.0, int(duration_s * sfreq))
    qrs_t = np.arange(int(0.05 * sfreq)) / sfreq
    qrs = 600.0 * np.exp(-((qrs_t - 0.025) ** 2) / (2 * 0.006 ** 2))
    t_t = np.arange(int(0.16 * sfreq)) / sfreq
    t_wave = t_wave_uv * np.exp(-((t_t - 0.08) ** 2) / (2 * 0.035 ** 2))
    for beat in beats:
        start = int(round(beat * sfreq))
        if start + qrs.size < signal.size:
            signal[start:start + qrs.size] += qrs
        offset = start + int(0.30 * sfreq)
        if t_wave_uv and offset + t_wave.size < signal.size:
            signal[offset:offset + t_wave.size] += t_wave
    return signal


def test_recovers_beats_deleted_from_a_known_stretch():
    beats = np.arange(5.0, 190.0, 0.9)
    ecg = _synthetic_ecg(beats, 200.0)
    kept = beats[(beats < 80.0) | (beats > 95.0)]
    deleted = beats[(beats >= 80.0) & (beats <= 95.0)]

    result = recover_beats(ecg, kept, SFREQ)

    assert result.recovered_beats.size >= deleted.size - 2
    matched = [np.min(np.abs(result.recovered_beats - d)) < 0.05 for d in deleted]
    assert sum(matched) >= deleted.size - 2


def test_inflated_t_wave_does_not_create_extra_beats():
    """Regression for the sub-0000 failure: T-waves must not be recovered as beats."""
    beats = np.arange(5.0, 190.0, 0.9)
    ecg = _synthetic_ecg(beats, 200.0, t_wave_uv=900.0)
    kept = beats[(beats < 80.0) | (beats > 95.0)]

    result = recover_beats(ecg, kept, SFREQ)

    assert result.quality.refractory_violations == 0
    assert result.combined_beats.size < beats.size * 1.15
    assert 55.0 < result.quality.implied_bpm < 75.0


def test_recovery_leaves_analyzer_beats_untouched():
    beats = np.arange(5.0, 190.0, 0.9)
    ecg = _synthetic_ecg(beats, 200.0)
    kept = beats[(beats < 80.0) | (beats > 95.0)]

    result = recover_beats(ecg, kept, SFREQ)

    assert np.all(np.isin(kept, result.combined_beats))


def test_too_few_seed_beats_reports_status_rather_than_raising():
    ecg = _synthetic_ecg(np.array([10.0, 11.0]), 60.0)

    result = recover_beats(ecg, np.array([10.0, 11.0]), SFREQ)

    assert result.quality.status == "insufficient_seed_beats"
    assert result.recovered_beats.size == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/preprocessing/bcg/test_detect.py -v`
Expected: FAIL with `ImportError: cannot import name 'RecoverySettings'`

- [ ] **Step 3: Write the implementation**

```python
# append to eeg_pipeline/preprocessing/bcg/detect.py
from eeg_pipeline.preprocessing.bcg.metrics import lock_ratio

MINIMUM_SEED_BEATS = 8


@dataclass(frozen=True)
class RecoverySettings:
    template_window: tuple[float, float] = (-0.2, 0.4)
    correlation_threshold: float = 0.5
    refractory_fraction: float = 0.5
    iterations: int = 2
    minimum_seconds: float = 2.0
    factor: float = 2.0


@dataclass(frozen=True)
class BeatQuality:
    analyzer_lock_ratio: float
    recovered_lock_ratio: float
    combined_lock_ratio: float
    rr_median_s: float
    rr_min_s: float
    rr_max_s: float
    implied_bpm: float
    refractory_violations: int
    recovered_beats: int
    gap_seconds_before: float
    gap_seconds_after: float
    status: str


@dataclass(frozen=True)
class BeatRecovery:
    analyzer_beats: np.ndarray
    recovered_beats: np.ndarray
    combined_beats: np.ndarray
    quality: BeatQuality


def qrs_template(
    ecg_uv: np.ndarray, beat_seconds: np.ndarray, sfreq: float, window: tuple[float, float]
) -> np.ndarray:
    """Average ECG waveform around a set of beats, mean-removed."""
    from eeg_pipeline.preprocessing.bcg.metrics import epoch_stack

    samples = np.round(np.asarray(beat_seconds) * sfreq).astype(int)
    stack = epoch_stack(ecg_uv[None, :], samples, sfreq, window)
    template = stack.mean(axis=1)[0]
    return template - template.mean()


def _normalised_correlation(signal: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Sliding Pearson correlation of template against signal, aligned to window starts."""
    length = template.size
    centred = template - template.mean()
    norm = np.linalg.norm(centred)
    if norm == 0:
        return np.zeros(max(signal.size - length + 1, 0))
    windows = np.lib.stride_tricks.sliding_window_view(signal, length)
    windows = windows - windows.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(windows, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(norms > 0, (windows @ centred) / (norms * norm), 0.0)


def _pick_peaks(scores: np.ndarray, threshold: float, refractory_samples: int) -> np.ndarray:
    """Greedy highest-first selection honouring a refractory period."""
    candidates = np.flatnonzero(scores >= threshold)
    if candidates.size == 0:
        return np.empty(0, dtype=int)
    order = candidates[np.argsort(scores[candidates])[::-1]]
    chosen: list[int] = []
    for index in order:
        if all(abs(index - taken) >= refractory_samples for taken in chosen):
            chosen.append(int(index))
    return np.array(sorted(chosen), dtype=int)


def recover_beats(
    ecg_uv: np.ndarray,
    analyzer_beats: np.ndarray,
    sfreq: float,
    *,
    settings: RecoverySettings = RecoverySettings(),
) -> BeatRecovery:
    """Recover beats inside Analyzer's gaps by matching its own QRS shape.

    Analyzer's marks are never revisited: the search runs only inside the gaps, and the
    template is re-estimated once from the combined set so a run with few seed beats is
    not permanently limited by a thin initial template.
    """
    analyzer = np.sort(np.asarray(analyzer_beats, dtype=float))
    duration = ecg_uv.size / sfreq
    window = settings.template_window
    before = gap_summary(analyzer, duration,
                         minimum_seconds=settings.minimum_seconds, factor=settings.factor)

    if analyzer.size < MINIMUM_SEED_BEATS:
        return BeatRecovery(
            analyzer_beats=analyzer,
            recovered_beats=np.empty(0),
            combined_beats=analyzer,
            quality=_quality(ecg_uv, analyzer, np.empty(0), analyzer, sfreq, window,
                             duration, before, before, "insufficient_seed_beats"),
        )

    median_rr = float(np.median(np.diff(analyzer)))
    refractory = max(int(round(settings.refractory_fraction * median_rr * sfreq)), 1)
    offset = int(round(window[0] * sfreq))

    recovered = np.empty(0)
    seed = analyzer
    for _ in range(max(settings.iterations, 1)):
        template = qrs_template(ecg_uv, seed, sfreq, window)
        found: list[float] = []
        for gap in find_gaps(analyzer, minimum_seconds=settings.minimum_seconds,
                             factor=settings.factor):
            lo = int(round((gap.start_s + median_rr * 0.5) * sfreq))
            hi = int(round((gap.end_s - median_rr * 0.5) * sfreq))
            lo, hi = max(lo, 0), min(hi, ecg_uv.size)
            if hi - lo <= template.size:
                continue
            scores = _normalised_correlation(ecg_uv[lo:hi], template)
            picks = _pick_peaks(scores, settings.correlation_threshold, refractory)
            found.extend(((lo + picks - offset) / sfreq).tolist())
        recovered = np.sort(np.asarray(found, dtype=float))
        seed = np.sort(np.concatenate([analyzer, recovered])) if recovered.size else analyzer

    combined = np.sort(np.concatenate([analyzer, recovered])) if recovered.size else analyzer
    after = gap_summary(combined, duration,
                        minimum_seconds=settings.minimum_seconds, factor=settings.factor)
    return BeatRecovery(
        analyzer_beats=analyzer,
        recovered_beats=recovered,
        combined_beats=combined,
        quality=_quality(ecg_uv, analyzer, recovered, combined, sfreq, window,
                         duration, before, after, "ok"),
    )


def _quality(ecg_uv, analyzer, recovered, combined, sfreq, window,
             duration, before, after, status) -> BeatQuality:
    intervals = np.diff(combined) if combined.size > 1 else np.array([np.nan])
    median_rr = float(np.median(intervals)) if combined.size > 1 else float("nan")
    refractory_floor = 0.5 * median_rr if combined.size > 1 else np.inf
    return BeatQuality(
        analyzer_lock_ratio=lock_ratio(ecg_uv, analyzer, sfreq, window) if analyzer.size else float("nan"),
        recovered_lock_ratio=lock_ratio(ecg_uv, recovered, sfreq, window) if recovered.size else float("nan"),
        combined_lock_ratio=lock_ratio(ecg_uv, combined, sfreq, window) if combined.size else float("nan"),
        rr_median_s=median_rr,
        rr_min_s=float(np.min(intervals)) if intervals.size else float("nan"),
        rr_max_s=float(np.max(intervals)) if intervals.size else float("nan"),
        implied_bpm=60.0 * combined.size / duration if duration else float("nan"),
        refractory_violations=int(np.sum(intervals < refractory_floor)) if combined.size > 1 else 0,
        recovered_beats=int(recovered.size),
        gap_seconds_before=float(before["gap_seconds"]),
        gap_seconds_after=float(after["gap_seconds"]),
        status=status,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/preprocessing/bcg/test_detect.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add eeg_pipeline/preprocessing/bcg/detect.py tests/preprocessing/bcg/test_detect.py
git commit -m "feat(bcg): recover gap beats by QRS template matching"
```

---

### Task 7: Correction methods with confinement guarantee

**Files:**
- Create: `eeg_pipeline/preprocessing/bcg/correct.py`
- Test: `tests/preprocessing/bcg/test_correct.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `correct_beats(data_uv: np.ndarray, beat_seconds: np.ndarray, sfreq: float, *, method: str = "obs", n_components: int = 4, window: tuple[float, float] = (-0.3, 0.7), n_neighbours: int = 21, ch_names: list[str] | None = None) -> np.ndarray`; `substitute_stretches(base_uv, replacement_uv, stretches: list[tuple[float, float]], sfreq) -> np.ndarray`.

`correct_beats` takes an array and returns a corrected copy, so the same entry point serves the real correction and the sham (same call, shifted beat times). `method="obs"` wraps `mne.preprocessing.apply_pca_obs`; `method="aas"` is sliding-window average subtraction, the family Analyzer's own correction belongs to.

- [ ] **Step 1: Write the failing test**

```python
# tests/preprocessing/bcg/test_correct.py
import numpy as np
import pytest

from eeg_pipeline.preprocessing.bcg.correct import correct_beats, substitute_stretches

SFREQ = 1000.0


def _beats(n=120, rr=0.9, start=5.0):
    return start + np.arange(n) * rr


def _inject(data, beats, sfreq, amplitude_uv):
    out = data.copy()
    t = np.arange(int(0.4 * sfreq)) / sfreq
    shape = amplitude_uv * np.sin(2 * np.pi * 5.0 * t) * np.exp(-t / 0.1)
    for beat in beats:
        start = int(round(beat * sfreq))
        if start + shape.size <= out.shape[1]:
            out[:, start:start + shape.size] += shape
    return out


@pytest.mark.parametrize("method", ["obs", "aas"])
def test_correction_reduces_injected_artifact(method):
    rng = np.random.default_rng(0)
    beats = _beats()
    clean = rng.normal(0, 10.0, (4, int(130 * SFREQ)))
    dirty = _inject(clean, beats, SFREQ, amplitude_uv=60.0)

    corrected = correct_beats(dirty, beats, SFREQ, method=method)

    before = np.var(dirty - clean)
    after = np.var(corrected - clean)
    assert after < before


@pytest.mark.parametrize("method", ["obs", "aas"])
def test_correction_is_confined_to_the_supplied_beats(method):
    rng = np.random.default_rng(1)
    data = rng.normal(0, 10.0, (3, int(120 * SFREQ)))
    beats = np.array([20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0])

    corrected = correct_beats(data, beats, SFREQ, method=method, window=(-0.3, 0.7))

    untouched = slice(int(60 * SFREQ), int(100 * SFREQ))
    assert np.array_equal(corrected[:, untouched], data[:, untouched])


def test_substitute_stretches_replaces_only_named_windows():
    base = np.zeros((2, 10000))
    replacement = np.ones((2, 10000))

    out = substitute_stretches(base, replacement, [(2.0, 4.0)], SFREQ)

    assert np.all(out[:, 2000:4000] == 1.0)
    assert np.all(out[:, :2000] == 0.0)
    assert np.all(out[:, 4000:] == 0.0)


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown method"):
        correct_beats(np.zeros((2, 5000)), np.array([1.0, 2.0]), SFREQ, method="nope")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/preprocessing/bcg/test_correct.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eeg_pipeline.preprocessing.bcg.correct'`

- [ ] **Step 3: Write the implementation**

```python
# eeg_pipeline/preprocessing/bcg/correct.py
"""Remove the ballistocardiogram at a given set of beats, and nowhere else.

Both methods take an array and return a corrected copy, so the identical call serves the
real correction and the sham control that measures what the procedure destroys when run at
beat times where no artifact sits.

Confinement matters as much as removal here: Analyzer's correction is already excellent at
the beats it marked (0.16% residual variance, 0 of 63 channels above null), so touching
those stretches can only make them worse.
"""

from __future__ import annotations

import numpy as np


def _epoch_bounds(beat_seconds, sfreq, window, n_times):
    pre = int(round(window[0] * sfreq))
    length = int(round((window[1] - window[0]) * sfreq))
    starts = np.round(np.asarray(beat_seconds, dtype=float) * sfreq).astype(int) + pre
    keep = (starts >= 0) & (starts + length <= n_times)
    return starts[keep], length


def _correct_aas(data_uv, beat_seconds, sfreq, window, n_neighbours):
    """Sliding-window average artifact subtraction (Allen et al. 1998).

    Each beat's template is the mean of its `n_neighbours` nearest epochs, which lets the
    template follow slow changes in the artifact rather than assuming one fixed shape.
    """
    starts, length = _epoch_bounds(beat_seconds, sfreq, window, data_uv.shape[1])
    if starts.size == 0:
        return data_uv.copy()
    index = starts[:, None] + np.arange(length)[None, :]
    epochs = data_uv[:, index]

    out = data_uv.copy()
    half = max(n_neighbours // 2, 1)
    for position in range(starts.size):
        lo = max(position - half, 0)
        hi = min(position + half + 1, starts.size)
        neighbours = np.delete(np.arange(lo, hi), np.where(np.arange(lo, hi) == position))
        if neighbours.size == 0:
            continue
        template = epochs[:, neighbours, :].mean(axis=1)
        out[:, index[position]] -= template
    return out


def _correct_obs(data_uv, beat_seconds, sfreq, n_components, ch_names):
    """PCA optimal basis set (Niazy et al. 2005) via MNE."""
    import mne

    mne.set_log_level("ERROR")
    names = ch_names or [f"CH{i:03d}" for i in range(data_uv.shape[0])]
    info = mne.create_info(names, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data_uv * 1e-6, info, verbose="ERROR")
    mne.preprocessing.apply_pca_obs(
        raw,
        picks=names,
        qrs_times=np.asarray(beat_seconds, dtype=float),
        n_components=n_components,
        copy=False,
        verbose="ERROR",
    )
    return raw.get_data() * 1e6


def correct_beats(
    data_uv: np.ndarray,
    beat_seconds: np.ndarray,
    sfreq: float,
    *,
    method: str = "obs",
    n_components: int = 4,
    window: tuple[float, float] = (-0.3, 0.7),
    n_neighbours: int = 21,
    ch_names: list[str] | None = None,
) -> np.ndarray:
    """Corrected copy of `data_uv`, with the artifact removed at `beat_seconds`."""
    if method == "aas":
        return _correct_aas(data_uv, beat_seconds, sfreq, window, n_neighbours)
    if method == "obs":
        return _correct_obs(data_uv, beat_seconds, sfreq, n_components, ch_names)
    raise ValueError(f"unknown method {method!r}; expected 'obs' or 'aas'")


def substitute_stretches(
    base_uv: np.ndarray,
    replacement_uv: np.ndarray,
    stretches: list[tuple[float, float]],
    sfreq: float,
) -> np.ndarray:
    """Copy named time ranges out of `replacement_uv` into `base_uv`.

    This is how Analyzer's correction is kept everywhere except the gaps.
    """
    if base_uv.shape != replacement_uv.shape:
        raise ValueError(
            f"shape mismatch: base {base_uv.shape} vs replacement {replacement_uv.shape}"
        )
    out = base_uv.copy()
    for start_s, end_s in stretches:
        lo = max(int(round(start_s * sfreq)), 0)
        hi = min(int(round(end_s * sfreq)), base_uv.shape[1])
        if hi > lo:
            out[:, lo:hi] = replacement_uv[:, lo:hi]
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/preprocessing/bcg/test_correct.py -v`
Expected: PASS (6 passed)

If `test_correction_is_confined_to_the_supplied_beats` fails for `obs`, that is a real finding about `apply_pca_obs`, not a test bug: it means the MNE implementation touches samples outside the supplied epochs. Record the measured extent in the test as a documented limitation and confine by explicit splicing in `correct_beats` instead of relying on MNE.

- [ ] **Step 5: Commit**

```bash
git add eeg_pipeline/preprocessing/bcg/correct.py tests/preprocessing/bcg/test_correct.py
git commit -m "feat(bcg): confined OBS and AAS correction"
```

---

### Task 8: Benchmark subcommand — both arms, both methods

**Files:**
- Create: `studies/pain_study/scripts/correct_cardiac_gaps.py`
- Test: `tests/scripts/test_correct_cardiac_gaps.py`

**Interfaces:**
- Consumes: everything from Tasks 1-7.
- Produces: `benchmark_run(pair: RunPair, settings: BenchmarkSettings) -> list[dict]`; `BenchmarkSettings(methods=("obs", "aas"), n_components=(4, 8), band=(1.0, 20.0), alpha_band=(8.0, 13.0), n_surrogate=20, seed=0)`; CLI entry `main(argv)` with subcommands `benchmark`, `apply`, `verify`.

Each row carries **both arms**: `removal_*` from `rlocked_reduction`, and `sham_*` from the identical correction run at circularly-shifted beats. A row without its sham is not interpretable, because the artifact contributes power inside every band of interest.

- [ ] **Step 1: Write the failing test**

```python
# tests/scripts/test_correct_cardiac_gaps.py
import numpy as np
import pytest

correct_cardiac_gaps = pytest.importorskip(
    "studies.pain_study.scripts.correct_cardiac_gaps"
)


def test_benchmark_row_carries_both_arms():
    rng = np.random.default_rng(0)
    sfreq = 1000.0
    beats = 5.0 + np.arange(120) * 0.9
    data = rng.normal(0, 10.0, (4, int(130 * sfreq)))

    rows = correct_cardiac_gaps.benchmark_arrays(
        data, beats, sfreq, methods=("aas",), n_components=(4,), n_surrogate=5
    )

    assert len(rows) == 1
    row = rows[0]
    for field in ("method", "removal_max", "removal_null_max",
                  "removal_channels_above_null", "sham_alpha_retained",
                  "real_alpha_retained"):
        assert field in row


def test_sham_retention_is_higher_than_real_when_artifact_present():
    """The sham must remove less than the real correction, or it is not a control."""
    rng = np.random.default_rng(1)
    sfreq = 1000.0
    beats = 5.0 + np.arange(120) * 0.9
    clean = rng.normal(0, 10.0, (4, int(130 * sfreq)))
    t = np.arange(int(0.4 * sfreq)) / sfreq
    shape = 60.0 * np.sin(2 * np.pi * 5.0 * t) * np.exp(-t / 0.1)
    dirty = clean.copy()
    for beat in beats:
        start = int(round(beat * sfreq))
        dirty[:, start:start + shape.size] += shape

    rows = correct_cardiac_gaps.benchmark_arrays(
        dirty, beats, sfreq, methods=("aas",), n_components=(4,), n_surrogate=5
    )

    assert rows[0]["sham_band_retained"] > rows[0]["real_band_retained"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/scripts/test_correct_cardiac_gaps.py -v`
Expected: FAIL — the module does not exist, so `importorskip` skips; treat a SKIP here as failure and continue.

- [ ] **Step 3: Write the implementation**

```python
# studies/pain_study/scripts/correct_cardiac_gaps.py
"""Fill Analyzer's pulse-marker gaps: recover beats, correct only there, and score it.

Analyzer's correction is kept wherever it marked a beat, because it measurably beats ours
on both arms there (0.16% residual against our 2.03%, alpha retained 0.54 against 0.34).
What it never marked is untouched artifact -- 6,954 s across the cohort, up to 77% of a
single run -- and that is all this stage changes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from eeg_pipeline.preprocessing.bcg import correct as bcg_correct
from eeg_pipeline.preprocessing.bcg import detect as bcg_detect
from eeg_pipeline.preprocessing.bcg import metrics as bcg_metrics
from eeg_pipeline.preprocessing.bcg.sources import discover_run_pairs, validate_pair

DEFAULT_UNCORRECTED = Path(
    "/Volumes/KINGSTON/EEG_fMRI_data/source_data/"
    "processed_scanner_artifact_with_pulse_markers_no_bcg_correction"
)
DEFAULT_CORRECTED = Path(
    "data/source_data/processed_trimmed_0-60s_30-115bpm_marker_template"
)
OCCIPITAL = ("O1", "O2", "Oz", "PO3", "PO4", "POz")


@dataclass(frozen=True)
class BenchmarkSettings:
    methods: tuple[str, ...] = ("obs", "aas")
    n_components: tuple[int, ...] = (4, 8)
    band: tuple[float, float] = (1.0, 20.0)
    alpha_band: tuple[float, float] = (8.0, 13.0)
    window: tuple[float, float] = (-0.3, 0.7)
    n_surrogate: int = 20
    seed: int = 0
    picks: tuple[str, ...] = field(default=OCCIPITAL)


def benchmark_arrays(
    data_uv: np.ndarray,
    beats: np.ndarray,
    sfreq: float,
    *,
    methods=("obs", "aas"),
    n_components=(4,),
    band=(1.0, 20.0),
    alpha_band=(8.0, 13.0),
    window=(-0.3, 0.7),
    n_surrogate: int = 20,
    seed: int = 0,
    picks: np.ndarray | None = None,
    ch_names: list[str] | None = None,
    stim_onsets: np.ndarray | None = None,
) -> list[dict]:
    """Score each method on both arms, with a sham control for the preservation arm.

    The sham applies the identical correction at a circularly-shifted beat train, where no
    artifact sits, so everything it removes is signal loss.
    """
    duration = data_uv.shape[1] / sfreq
    rng = np.random.default_rng(seed)
    sham_beats = np.sort((beats + rng.uniform(2.0, duration - 2.0)) % duration)

    rows: list[dict] = []
    for method in methods:
        ranks = n_components if method == "obs" else (0,)
        for rank in ranks:
            kwargs = dict(method=method, window=window, ch_names=ch_names)
            if method == "obs":
                kwargs["n_components"] = rank

            real = bcg_correct.correct_beats(data_uv, beats, sfreq, **kwargs)
            sham = bcg_correct.correct_beats(data_uv, sham_beats, sfreq, **kwargs)
            result = bcg_metrics.rlocked_reduction(
                real, beats, sfreq, window=window, n_surrogate=n_surrogate, seed=seed
            )
            rows.append({
                "method": method,
                "n_components": rank,
                "removal_max": result.max_value,
                "removal_null_max": result.null_max,
                "removal_channels_above_null": result.channels_above_null,
                "real_alpha_retained": bcg_metrics.band_retention(
                    data_uv, real, sfreq, alpha_band, picks),
                "sham_alpha_retained": bcg_metrics.band_retention(
                    data_uv, sham, sfreq, alpha_band, picks),
                "real_band_retained": bcg_metrics.band_retention(
                    data_uv, real, sfreq, band, picks),
                "sham_band_retained": bcg_metrics.band_retention(
                    data_uv, sham, sfreq, band, picks),
            })
            if stim_onsets is not None and stim_onsets.size >= 8:
                evoked = bcg_metrics.evoked_preservation(
                    data_uv, real, stim_onsets, sfreq, (-0.1, 0.5)
                )
                rows[-1]["evoked_correlation_median"] = float(
                    np.nanmedian(evoked.correlation))
                rows[-1]["evoked_amplitude_ratio_median"] = float(
                    np.nanmedian(evoked.amplitude_ratio))
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/scripts/test_correct_cardiac_gaps.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Add the run-level benchmark and CLI**

```python
# append to studies/pain_study/scripts/correct_cardiac_gaps.py
def _load_pair(pair):
    import mne

    mne.set_log_level("ERROR")
    uncorrected = mne.io.read_raw_brainvision(pair.uncorrected_vhdr, preload=True, verbose="ERROR")
    corrected = mne.io.read_raw_brainvision(pair.corrected_vhdr, preload=True, verbose="ERROR")
    return uncorrected, corrected


def benchmark_run(pair, settings: BenchmarkSettings) -> list[dict]:
    """Benchmark one recording, scoring only the stretches this stage would change."""
    validation = validate_pair(pair)
    if validation.status != "ok":
        return [{"subject": pair.subject, "run": pair.run, "status": validation.status}]

    uncorrected, _ = _load_pair(pair)
    sfreq = uncorrected.info["sfreq"]
    eeg_names = uncorrected.copy().pick("eeg").ch_names
    data = uncorrected.copy().pick(eeg_names).get_data() * 1e6
    ecg = uncorrected.copy().pick(["ECG"]).get_data()[0] * 1e6

    analyzer = bcg_detect.read_analyzer_beats(pair.uncorrected_vhdr)
    recovery = bcg_detect.recover_beats(ecg, analyzer, sfreq)
    if recovery.recovered_beats.size < 8:
        return [{"subject": pair.subject, "run": pair.run,
                 "status": f"too_few_recovered ({recovery.recovered_beats.size})"}]

    # `np.array(...) or None` raises on a multi-element array; test emptiness explicitly.
    selected = [i for i, n in enumerate(eeg_names) if n in settings.picks]
    picks = np.array(selected, dtype=int) if selected else None

    stim = np.asarray([
        onset
        for onset, description in zip(
            uncorrected.annotations.onset, uncorrected.annotations.description
        )
        if description.split("/")[-1].strip().startswith("S")
    ], dtype=float)

    rows = benchmark_arrays(
        data, recovery.recovered_beats, sfreq,
        methods=settings.methods, n_components=settings.n_components,
        band=settings.band, alpha_band=settings.alpha_band, window=settings.window,
        n_surrogate=settings.n_surrogate, seed=settings.seed,
        picks=picks, ch_names=eeg_names, stim_onsets=stim,
    )
    for row in rows:
        row.update({
            "subject": pair.subject, "run": pair.run, "status": "ok",
            "analyzer_beats": int(analyzer.size),
            "recovered_beats": int(recovery.recovered_beats.size),
            "analyzer_lock_ratio": recovery.quality.analyzer_lock_ratio,
            "recovered_lock_ratio": recovery.quality.recovered_lock_ratio,
            "gap_seconds_before": recovery.quality.gap_seconds_before,
            "gap_seconds_after": recovery.quality.gap_seconds_after,
        })
    return rows


def _write_tsv(rows: list[dict], destination: Path) -> None:
    import csv

    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["benchmark"])
    parser.add_argument("--uncorrected-root", type=Path, default=DEFAULT_UNCORRECTED)
    parser.add_argument("--corrected-root", type=Path, default=DEFAULT_CORRECTED)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/cardiac_gap_fill/benchmark.tsv"))
    args = parser.parse_args(argv)

    pairs = discover_run_pairs(args.uncorrected_root, args.corrected_root)
    if args.subjects:
        pairs = [p for p in pairs if p.subject in set(args.subjects)]
    if args.limit:
        pairs = pairs[: args.limit]

    settings = BenchmarkSettings()
    rows: list[dict] = []
    for pair in pairs:
        try:
            rows.extend(benchmark_run(pair, settings))
        except Exception as error:  # a failing run is a measurement, not a fault
            rows.append({"subject": pair.subject, "run": pair.run,
                         "status": f"error: {type(error).__name__}: {error}"})
        print(json.dumps(rows[-1]), flush=True)
    _write_tsv(rows, args.output)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the targeted tests again**

Run: `.venv/bin/python -m pytest tests/scripts/test_correct_cardiac_gaps.py tests/preprocessing/bcg/ -v`
Expected: PASS (all)

- [ ] **Step 7: Run the real benchmark on three subjects**

Run: `.venv/bin/python studies/pain_study/scripts/correct_cardiac_gaps.py benchmark --subjects sub0009 sub0005 sub0000 --limit 6`
Expected: rows written to `outputs/cardiac_gap_fill/benchmark.tsv`. Inspect `removal_*` against `sham_*` and pick the method and rank on both arms together. Do not choose on `removal_max` alone — it improves monotonically while signal is destroyed.

- [ ] **Step 8: Commit**

```bash
git add studies/pain_study/scripts/correct_cardiac_gaps.py tests/scripts/test_correct_cardiac_gaps.py
git commit -m "feat(bcg): benchmark gap correction on removal and preservation"
```

---

### Task 9: Apply and verify — write the corrected tree

**Files:**
- Modify: `studies/pain_study/scripts/correct_cardiac_gaps.py`
- Test: `tests/scripts/test_correct_cardiac_gaps.py`

**Interfaces:**
- Consumes: `benchmark_run`, `substitute_stretches`, `recover_beats`, `validate_pair`.
- Produces: `apply_run(pair, output_root: Path, settings: ApplySettings) -> dict`; `ApplySettings(method: str, n_components: int, window, pad_seconds: float = 0.5)`.

The written binary is read back and compared to what was intended, exactly as `remove_line_comb.py` does. Float32 storage loses about 2^-24 of full scale, so anything a decade above that is corruption rather than quantisation.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/scripts/test_correct_cardiac_gaps.py
def test_apply_only_changes_gap_stretches(tmp_path):
    """Everything outside a gap must survive byte-for-byte from Analyzer's output."""
    rng = np.random.default_rng(2)
    sfreq = 1000.0
    n = int(200 * sfreq)
    analyzer_corrected = rng.normal(0, 10.0, (4, n))
    uncorrected = analyzer_corrected + rng.normal(0, 1.0, (4, n))

    out = correct_cardiac_gaps.substitute_gap_stretches(
        analyzer_corrected, uncorrected, [(80.0, 95.0)], sfreq, pad_seconds=0.5
    )

    lo, hi = int(79.5 * sfreq), int(95.5 * sfreq)
    assert np.array_equal(out[:, :lo], analyzer_corrected[:, :lo])
    assert np.array_equal(out[:, hi:], analyzer_corrected[:, hi:])
    assert not np.array_equal(out[:, lo:hi], analyzer_corrected[:, lo:hi])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/scripts/test_correct_cardiac_gaps.py::test_apply_only_changes_gap_stretches -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'substitute_gap_stretches'`

- [ ] **Step 3: Write the implementation**

```python
# append to studies/pain_study/scripts/correct_cardiac_gaps.py
ROUNDTRIP_RELATIVE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ApplySettings:
    method: str = "obs"
    n_components: int = 4
    window: tuple[float, float] = (-0.3, 0.7)
    pad_seconds: float = 0.5


def substitute_gap_stretches(base_uv, replacement_uv, gaps, sfreq, pad_seconds=0.5):
    """Splice corrected gap stretches into Analyzer's output, with a small pad.

    The pad covers epochs of beats sitting just inside a gap edge, whose correction window
    extends slightly beyond the gap itself.
    """
    padded = [(start - pad_seconds, end + pad_seconds) for start, end in gaps]
    return bcg_correct.substitute_stretches(base_uv, replacement_uv, padded, sfreq)


def apply_run(pair, output_root: Path, settings: ApplySettings) -> dict:
    """Correct one recording's gap stretches and write the result beside its sidecars."""
    import mne

    validation = validate_pair(pair)
    if validation.status != "ok":
        return {"subject": pair.subject, "run": pair.run, "status": validation.status}

    uncorrected, corrected = _load_pair(pair)
    sfreq = uncorrected.info["sfreq"]
    eeg_names = uncorrected.copy().pick("eeg").ch_names
    unc = uncorrected.copy().pick(eeg_names).get_data() * 1e6
    cor = corrected.copy().pick(eeg_names).get_data() * 1e6

    ecg = uncorrected.copy().pick(["ECG"]).get_data()[0] * 1e6
    analyzer = bcg_detect.read_analyzer_beats(pair.uncorrected_vhdr)
    recovery = bcg_detect.recover_beats(ecg, analyzer, sfreq)
    if recovery.recovered_beats.size == 0:
        return {"subject": pair.subject, "run": pair.run, "status": "no_recovered_beats",
                "gap_seconds_before": recovery.quality.gap_seconds_before}

    kwargs = dict(method=settings.method, window=settings.window, ch_names=eeg_names)
    if settings.method == "obs":
        kwargs["n_components"] = settings.n_components
    repaired = bcg_correct.correct_beats(unc, recovery.recovered_beats, sfreq, **kwargs)

    gaps = [(g.start_s, g.end_s) for g in bcg_detect.find_gaps(analyzer)]
    merged = substitute_gap_stretches(cor, repaired, gaps, sfreq, settings.pad_seconds)

    info = mne.create_info(eeg_names, sfreq, ch_types="eeg")
    out_raw = mne.io.RawArray(merged * 1e-6, info, verbose="ERROR")
    destination = output_root / pair.corrected_vhdr.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    mne.export.export_raw(destination, out_raw, fmt="brainvision",
                          overwrite=True, verbose="ERROR")

    check = mne.io.read_raw_brainvision(destination, preload=True, verbose="ERROR")
    deviation = float(np.max(np.abs(check.get_data() * 1e6 - merged)))
    scale = float(np.max(np.abs(merged)))
    if deviation > ROUNDTRIP_RELATIVE_TOLERANCE * scale:
        raise RuntimeError(
            f"{destination.name}: written data differs by {deviation:.3e} uV, "
            f"above the {ROUNDTRIP_RELATIVE_TOLERANCE * scale:.3e} uV round-trip tolerance."
        )

    return {
        "subject": pair.subject, "run": pair.run, "status": "ok",
        "method": settings.method, "n_components": settings.n_components,
        "recovered_beats": int(recovery.recovered_beats.size),
        "gap_seconds_before": recovery.quality.gap_seconds_before,
        "gap_seconds_after": recovery.quality.gap_seconds_after,
        "gap_fraction_replaced": sum(e - s for s, e in gaps) / (unc.shape[1] / sfreq),
        "roundtrip_max_deviation_uv": deviation,
        "output": str(destination),
    }
```

Replace `main` with the three-subcommand version:

```python
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["benchmark", "apply", "verify"])
    parser.add_argument("--uncorrected-root", type=Path, default=DEFAULT_UNCORRECTED)
    parser.add_argument("--corrected-root", type=Path, default=DEFAULT_CORRECTED)
    parser.add_argument("--output-root", type=Path,
                        default=Path("outputs/cardiac_gap_fill/corrected"))
    parser.add_argument("--method", default="obs", choices=["obs", "aas"])
    parser.add_argument("--n-components", type=int, default=4)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    pairs = discover_run_pairs(args.uncorrected_root, args.corrected_root)
    if args.subjects:
        pairs = [p for p in pairs if p.subject in set(args.subjects)]
    if args.limit:
        pairs = pairs[: args.limit]

    default_names = {
        "benchmark": "benchmark.tsv",
        "apply": "apply.tsv",
        "verify": "verify.tsv",
    }
    destination = args.output or Path("outputs/cardiac_gap_fill") / default_names[args.command]
    apply_settings = ApplySettings(method=args.method, n_components=args.n_components)

    rows: list[dict] = []
    for pair in pairs:
        try:
            if args.command == "benchmark":
                rows.extend(benchmark_run(pair, BenchmarkSettings()))
            elif args.command == "apply":
                rows.append(apply_run(pair, args.output_root, apply_settings))
            else:
                rows.append(verify_run(pair, args.output_root))
        except Exception as error:  # a failing run is a measurement, not a fault
            rows.append({"subject": pair.subject, "run": pair.run,
                         "status": f"error: {type(error).__name__}: {error}"})
        print(json.dumps(rows[-1]), flush=True)
    _write_tsv(rows, destination)
    print(f"wrote {len(rows)} rows to {destination}")
```

And add `verify_run`, which re-scores what `apply` actually wrote:

```python
def verify_run(pair, output_root: Path) -> dict:
    """Re-score a written recording with the referee, inside the gaps and outside them.

    Scoring the two separately is what shows whether the stage introduced a time-varying
    difference within the run, which is the risk of correcting only part of it.
    """
    import mne

    mne.set_log_level("ERROR")
    destination = output_root / pair.corrected_vhdr.name
    if not destination.exists():
        return {"subject": pair.subject, "run": pair.run, "status": "not_written"}

    written = mne.io.read_raw_brainvision(destination, preload=True, verbose="ERROR")
    uncorrected, _ = _load_pair(pair)
    sfreq = written.info["sfreq"]
    data = written.get_data() * 1e6

    ecg = uncorrected.copy().pick(["ECG"]).get_data()[0] * 1e6
    analyzer = bcg_detect.read_analyzer_beats(pair.uncorrected_vhdr)
    recovery = bcg_detect.recover_beats(ecg, analyzer, sfreq)

    result = bcg_metrics.rlocked_reduction(
        data, recovery.recovered_beats, sfreq, n_surrogate=20, seed=0
    )
    analyzer_result = bcg_metrics.rlocked_reduction(
        data, analyzer, sfreq, n_surrogate=20, seed=0
    )
    return {
        "subject": pair.subject, "run": pair.run, "status": "ok",
        "recovered_removal_max": result.max_value,
        "recovered_null_max": result.null_max,
        "recovered_channels_above_null": result.channels_above_null,
        "analyzer_removal_max": analyzer_result.max_value,
        "analyzer_channels_above_null": analyzer_result.channels_above_null,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/scripts/test_correct_cardiac_gaps.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Run the full targeted suite**

Run: `.venv/bin/python -m pytest tests/preprocessing/bcg/ tests/scripts/test_correct_cardiac_gaps.py -v`
Expected: PASS (all). Do not run the whole suite.

- [ ] **Step 6: Commit**

```bash
git add studies/pain_study/scripts/correct_cardiac_gaps.py tests/scripts/test_correct_cardiac_gaps.py
git commit -m "feat(bcg): apply gap correction and verify the written binary"
```

---

### Task 10: NeuroKit2 cross-check and the dependency pin

**Files:**
- Modify: `pyproject.toml:24-32` (dependency list)
- Modify: `eeg_pipeline/preprocessing/bcg/detect.py`
- Test: `tests/preprocessing/bcg/test_detect.py`

**Interfaces:**
- Consumes: `BeatRecovery` from Task 6.
- Produces: `crosscheck_agreement(recovered_beats: np.ndarray, ecg_uv: np.ndarray, sfreq: float, *, tolerance: float = 0.05) -> dict[str, float]`.

The cross-check never rejects a beat. On this cohort it is the cross-check that is more often wrong: on sub-0000 run 1, NeuroKit2's default method returned 1043 beats against Analyzer's 689, at a QRS lock ratio of 1.44 against 4.39. It is recorded as a measurement so disagreement is visible, not acted on.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/preprocessing/bcg/test_detect.py
from eeg_pipeline.preprocessing.bcg.detect import crosscheck_agreement


def test_crosscheck_reports_agreement_without_changing_beats():
    beats = np.arange(5.0, 190.0, 0.9)
    ecg = _synthetic_ecg(beats, 200.0)

    report = crosscheck_agreement(beats, ecg, SFREQ)

    assert 0.0 <= report["agreement_fraction"] <= 1.0
    assert report["crosscheck_beats"] > 0
    assert "crosscheck_lock_ratio" in report


def test_crosscheck_degrades_gracefully_when_neurokit_fails():
    """A cross-check that cannot run is a missing measurement, not an error."""
    report = crosscheck_agreement(np.array([1.0, 2.0]), np.zeros(100), SFREQ)

    assert report["status"] != "ok"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/preprocessing/bcg/test_detect.py -k crosscheck -v`
Expected: FAIL with `ImportError: cannot import name 'crosscheck_agreement'`

- [ ] **Step 3: Write the implementation**

```python
# append to eeg_pipeline/preprocessing/bcg/detect.py
def crosscheck_agreement(
    recovered_beats: np.ndarray,
    ecg_uv: np.ndarray,
    sfreq: float,
    *,
    tolerance: float = 0.05,
) -> dict[str, float]:
    """Compare our beat set against NeuroKit2, as a measurement only.

    Never used to accept or reject a beat. NeuroKit2 reads the same magnetohydrodynamically
    distorted ECG and inflates counts on the affected subjects, so its disagreement is
    evidence about the run, not about our beats.
    """
    try:
        import neurokit2 as nk

        _, info = nk.ecg_peaks(ecg_uv, sampling_rate=int(sfreq), correct_artifacts=True)
        other = np.asarray(info["ECG_R_Peaks"], dtype=float) / sfreq
    except Exception as error:
        return {"status": f"unavailable: {type(error).__name__}", "agreement_fraction": float("nan"),
                "crosscheck_beats": 0.0, "crosscheck_lock_ratio": float("nan")}

    beats = np.asarray(recovered_beats, dtype=float)
    if beats.size == 0 or other.size == 0:
        return {"status": "no_beats", "agreement_fraction": float("nan"),
                "crosscheck_beats": float(other.size), "crosscheck_lock_ratio": float("nan")}

    nearest = np.abs(beats[:, None] - other[None, :]).min(axis=1)
    return {
        "status": "ok",
        "agreement_fraction": float(np.mean(nearest <= tolerance)),
        "crosscheck_beats": float(other.size),
        "crosscheck_lock_ratio": lock_ratio(ecg_uv, other, sfreq, (-0.2, 0.4)),
    }
```

- [ ] **Step 4: Pin the dependency**

Add to the `dependencies` list in `pyproject.toml`, keeping `numpy>=1.24.0,<2.0.0` unchanged:

```toml
    "neurokit2==0.2.12",
```

Version 0.2.13 resolves numpy to 2.5.1 and breaks the pin. Confirm the environment is unchanged afterwards:

Run: `.venv/bin/python -c "import numpy, neurokit2, mne; print(numpy.__version__, neurokit2.__version__, mne.__version__)"`
Expected: `1.26.4 0.2.12 1.12.1`

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/preprocessing/bcg/test_detect.py -v`
Expected: PASS (10 passed)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml eeg_pipeline/preprocessing/bcg/detect.py tests/preprocessing/bcg/test_detect.py
git commit -m "feat(bcg): record NeuroKit2 cross-check agreement as a measurement"
```

---

## Notes for the implementer

**Read the spec first:** `docs/superpowers/specs/2026-07-30-cardiac-gap-fill-design.md`. It carries the measurements that justify each decision, including why Analyzer's correction is kept and why off-the-shelf QRS detectors are not used.

**The one trap to avoid.** Choosing a correction method or rank on removal alone is wrong and the data shows it: rank 4 to 24 leaves removal flat at ~2% while alpha retention falls from 0.34 to 0.18. Every method decision needs its sham.

**Known-good reference values** for `sub0009` run 1, useful for sanity-checking the pipeline end to end:

| quantity | value |
|---|---|
| Analyzer beats | 482 |
| n_times | 512,100 at 1000 Hz |
| ECG difference between exports | 0.0 uV |
| R-locked variance, uncorrected | 0.579, 63/63 channels above null |
| R-locked variance, Analyzer | 0.0016, 0/63 |
| R-locked variance, our OBS rank 4 | 0.021, 11/63 |
| alpha retained, Analyzer | 0.543 |
| alpha retained, our OBS rank 4 | 0.343 |
| sham alpha retained, OBS rank 4 | 0.707 / 0.741 |
