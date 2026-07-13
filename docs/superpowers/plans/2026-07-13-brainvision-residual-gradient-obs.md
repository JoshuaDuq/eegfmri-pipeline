# BrainVision Residual Gradient OBS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and pilot an outcome-blind native-Python residual OBS benchmark for 1 kHz
BrainVision files that have already undergone scanner AAS, low-pass/downsampling, and pulse
correction.

**Architecture:** A reusable preprocessing module validates the BrainVision-derived MNE input,
extracts complete 900-sample volume epochs, and applies deterministic channel-wise cross-fitted
temporal PCA. A separate QC module evaluates scanner-line attenuation and injected-signal
preservation. A pain-study command discovers only `*_scannerpulse_corrected.vhdr` files, runs the
fixed 0–4 component pilot grid, writes isolated benchmark derivatives, and makes a fail-fast
selection decision without changing canonical preprocessing.

**Tech Stack:** Python 3.11, MNE-Python, NumPy, SciPy, pandas, PyYAML, pytest, Ruff, Black

---

## File structure

- Create `eeg_pipeline/preprocessing/residual_gradient.py`: strict input validation, volume layout,
  cross-fitted OBS, and correction audit records.
- Create `eeg_pipeline/analysis/qc/residual_gradient.py`: injected-signal construction, recovery
  metrics, raw-object harmonic summaries, and component selection.
- Create `studies/pain_study/scripts/benchmark_residual_gradient.py`: study discovery,
  orchestration, derivative/provenance writing, and CLI.
- Create `studies/pain_study/scripts/config/residual_gradient_benchmark.yaml`: fixed acquisition,
  component grid, pilot, PSD, and preservation settings.
- Create `tests/preprocessing/test_residual_gradient.py`: core numerical and failure contracts.
- Create `tests/analysis/test_residual_gradient_qc.py`: QC and decision-rule tests.
- Create `tests/scripts/test_benchmark_residual_gradient.py`: discovery, configuration, routing,
  atomic output, and CLI tests.
- Modify `studies/pain_study/SCANNER_HARMONICS_QC_README.md`: benchmark purpose, command, outputs,
  interpretation, and production prohibition.

Do not modify the existing MNE preprocessing pipeline or Study 1/Study 2 production configuration
in this implementation. Promotion is a later design decision based on the pilot report.

### Task 1: Strict BrainVision source and volume-layout contract

**Files:**
- Create: `eeg_pipeline/preprocessing/residual_gradient.py`
- Create: `tests/preprocessing/test_residual_gradient.py`

- [ ] **Step 1: Write failing tests for the accepted input and volume layout**

```python
from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pytest

from eeg_pipeline.preprocessing.residual_gradient import (
    ResidualObsSettings,
    build_volume_layout,
    validate_brainvision_source,
    validate_residual_obs_raw,
)


def make_raw(
    *,
    sfreq: float = 1_000.0,
    n_times: int = 54_000,
    marker_samples: np.ndarray | None = None,
) -> mne.io.RawArray:
    info = mne.create_info(["Fz", "Cz", "ECG"], sfreq, ["eeg", "eeg", "ecg"])
    raw = mne.io.RawArray(np.zeros((3, n_times)), info, verbose="ERROR")
    samples = marker_samples if marker_samples is not None else np.arange(0, 45_000, 900)
    raw.set_annotations(
        mne.Annotations(
            onset=samples / sfreq,
            duration=np.zeros(samples.size),
            description=["Volume/V  1"] * samples.size,
        )
    )
    return raw


def test_build_volume_layout_accepts_complete_epochs_and_long_gap() -> None:
    samples = np.concatenate([np.arange(0, 27_000, 900), np.arange(30_000, 48_000, 900)])
    raw = make_raw(n_times=49_000, marker_samples=samples)
    settings = ResidualObsSettings()

    layout = build_volume_layout(raw, settings)

    assert layout.epoch_samples == 900
    assert layout.starts.tolist() == samples.tolist()
    assert layout.block_ids.tolist() == [0] * 30 + [1] * 20


def test_validate_raw_rejects_unexpected_sampling_frequency() -> None:
    with pytest.raises(ValueError, match="Expected sampling frequency 1000.0 Hz"):
        validate_residual_obs_raw(make_raw(sfreq=500.0), ResidualObsSettings())


def test_build_volume_layout_rejects_short_interval() -> None:
    raw = make_raw(marker_samples=np.array([0, 900, 1_799, 2_700]))

    with pytest.raises(ValueError, match="shorter than the 900-sample TR"):
        build_volume_layout(raw, ResidualObsSettings(min_complete_epochs=1))


def test_validate_brainvision_source_requires_referenced_files(tmp_path: Path) -> None:
    header = tmp_path / "run_scannerpulse_corrected.vhdr"
    header.write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n"
        "DataFile=run_scannerpulse_corrected.eeg\n"
        "MarkerFile=run_scannerpulse_corrected.vmrk\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="referenced data file"):
        validate_brainvision_source(header)
```

- [ ] **Step 2: Run the tests and verify the new module is absent**

Run:

```bash
uv run --python 3.11 --extra dev python -m pytest \
  tests/preprocessing/test_residual_gradient.py -q
```

Expected: collection fails with `ModuleNotFoundError: eeg_pipeline.preprocessing.residual_gradient`.

- [ ] **Step 3: Implement immutable settings, source validation, and volume layout**

```python
"""Residual scanner-gradient modeling after BrainVision AAS and pulse correction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ResidualObsSettings:
    expected_sfreq_hz: float = 1_000.0
    volume_marker: str = "Volume/V  1"
    tr_s: float = 0.9
    min_complete_epochs: int = 50
    n_folds: int = 5

    def __post_init__(self) -> None:
        if self.expected_sfreq_hz <= 0:
            raise ValueError("expected_sfreq_hz must be positive.")
        if not self.volume_marker:
            raise ValueError("volume_marker must be non-empty.")
        if self.tr_s <= 0:
            raise ValueError("tr_s must be positive.")
        if self.min_complete_epochs < 2:
            raise ValueError("min_complete_epochs must be at least 2.")
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least 2.")


@dataclass(frozen=True)
class VolumeLayout:
    starts: NDArray[np.int64]
    block_ids: NDArray[np.int64]
    epoch_samples: int

    @property
    def n_epochs(self) -> int:
        return int(self.starts.size)


def validate_brainvision_source(vhdr_path: str | Path) -> Path:
    path = Path(vhdr_path)
    if path.suffix.lower() != ".vhdr":
        raise ValueError(f"Expected a .vhdr BrainVision header, got: {path}")
    if not path.name.endswith("_scannerpulse_corrected.vhdr"):
        raise ValueError(f"Expected a _scannerpulse_corrected.vhdr input, got: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"BrainVision header does not exist: {path}")

    entries = _read_header_entries(path)
    _require_referenced_file(path, entries, "DataFile", "data")
    _require_referenced_file(path, entries, "MarkerFile", "marker")
    return path


def _read_header_entries(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        entries[key.strip()] = value.strip()
    return entries


def _require_referenced_file(
    header_path: Path,
    entries: dict[str, str],
    key: str,
    label: str,
) -> None:
    if key not in entries:
        raise ValueError(f"BrainVision header has no {key} entry: {header_path}")
    referenced_path = header_path.parent / entries[key]
    if not referenced_path.is_file():
        raise FileNotFoundError(
            f"BrainVision referenced {label} file does not exist: {referenced_path}"
        )


def validate_residual_obs_raw(raw: mne.io.BaseRaw, settings: ResidualObsSettings) -> None:
    sfreq = float(raw.info["sfreq"])
    if sfreq != settings.expected_sfreq_hz:
        raise ValueError(
            f"Expected sampling frequency {settings.expected_sfreq_hz} Hz, got {sfreq} Hz."
        )
    if settings.volume_marker not in set(map(str, raw.annotations.description)):
        raise ValueError(f"Required annotation is absent: {settings.volume_marker!r}")
    if not mne.pick_types(raw.info, eeg=True, exclude=[]).size:
        raise ValueError("Residual OBS requires at least one EEG channel.")


def build_volume_layout(
    raw: mne.io.BaseRaw,
    settings: ResidualObsSettings,
) -> VolumeLayout:
    validate_residual_obs_raw(raw, settings)
    epoch_samples = int(round(settings.tr_s * settings.expected_sfreq_hz))
    descriptions = np.asarray(raw.annotations.description, dtype=str)
    onsets = raw.annotations.onset[descriptions == settings.volume_marker]
    starts = raw.time_as_index(onsets, use_rounding=True).astype(np.int64)
    intervals = np.diff(starts)
    if np.any(intervals <= 0):
        raise ValueError("Volume marker samples must be strictly increasing.")
    if np.any(intervals < epoch_samples):
        value = int(intervals[intervals < epoch_samples][0])
        raise ValueError(
            f"Volume marker interval {value} is shorter than the {epoch_samples}-sample TR."
        )

    complete = starts + epoch_samples <= raw.n_times
    starts = starts[complete]
    if starts.size < settings.min_complete_epochs:
        raise ValueError(
            f"Expected at least {settings.min_complete_epochs} complete volume epochs, "
            f"got {starts.size}."
        )
    retained_intervals = np.diff(starts)
    block_ids = np.concatenate(
        [np.array([0]), np.cumsum(retained_intervals > epoch_samples)]
    ).astype(np.int64)
    return VolumeLayout(starts=starts, block_ids=block_ids, epoch_samples=epoch_samples)
```

- [ ] **Step 4: Run the volume-layout tests**

Run the Task 1 pytest command again.

Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit the input contract**

```bash
git add eeg_pipeline/preprocessing/residual_gradient.py \
  tests/preprocessing/test_residual_gradient.py
git commit -m "feat: validate residual gradient OBS inputs"
```

### Task 2: Deterministic cross-fitted residual OBS

**Files:**
- Modify: `eeg_pipeline/preprocessing/residual_gradient.py`
- Modify: `tests/preprocessing/test_residual_gradient.py`

- [ ] **Step 1: Add failing identity, low-rank, and preservation tests**

```python
from eeg_pipeline.preprocessing.residual_gradient import apply_residual_obs


def make_artifact_raw(*, extra_samples: int = 0) -> tuple[mne.io.RawArray, np.ndarray]:
    sfreq = 1_000.0
    epoch_samples = 900
    n_epochs = 50
    time = np.arange(epoch_samples) / sfreq
    artifact = np.sin(2 * np.pi * 20 * time) + 0.7 * np.sin(2 * np.pi * 41 * time)
    artifact -= artifact.mean()
    amplitudes = np.linspace(0.8, 1.2, n_epochs)
    eeg = np.concatenate([amplitude * artifact for amplitude in amplitudes])
    if extra_samples:
        eeg = np.pad(eeg, (0, extra_samples))
    data = np.vstack([eeg, 0.5 * eeg, np.arange(eeg.size, dtype=float) * 1e-9]) * 1e-6
    info = mne.create_info(["Fz", "Cz", "ECG"], sfreq, ["eeg", "eeg", "ecg"])
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    samples = np.arange(n_epochs) * epoch_samples
    raw.set_annotations(
        mne.Annotations(
            onset=samples / sfreq,
            duration=np.zeros(n_epochs),
            description=["Volume/V  1"] * n_epochs,
        )
    )
    return raw, samples


def test_zero_components_is_exact_identity() -> None:
    raw, _ = make_artifact_raw()
    layout = build_volume_layout(raw, ResidualObsSettings())

    result = apply_residual_obs(raw, layout, n_components=0, n_folds=5)

    np.testing.assert_array_equal(result.raw.get_data(), raw.get_data())
    assert result.component_rows == ()


def test_one_component_removes_known_rank_one_residual() -> None:
    raw, _ = make_artifact_raw()
    layout = build_volume_layout(raw, ResidualObsSettings())

    result = apply_residual_obs(raw, layout, n_components=1, n_folds=5)

    assert np.sqrt(np.mean(result.raw.get_data(picks="eeg") ** 2)) < 1e-10


def test_obs_preserves_non_eeg_and_samples_outside_epochs() -> None:
    raw, _ = make_artifact_raw(extra_samples=100)
    layout = build_volume_layout(raw, ResidualObsSettings())

    result = apply_residual_obs(raw, layout, n_components=1, n_folds=5)

    ecg_pick = raw.ch_names.index("ECG")
    np.testing.assert_array_equal(result.raw.get_data([ecg_pick]), raw.get_data([ecg_pick]))
    np.testing.assert_array_equal(result.raw.get_data()[:, -100:], raw.get_data()[:, -100:])
    np.testing.assert_array_equal(result.raw.annotations.onset, raw.annotations.onset)
    np.testing.assert_array_equal(result.raw.annotations.duration, raw.annotations.duration)
    np.testing.assert_array_equal(result.raw.annotations.description, raw.annotations.description)
    assert result.raw.annotations.orig_time == raw.annotations.orig_time
```

- [ ] **Step 2: Run the three tests and verify the missing symbol failure**

Run:

```bash
uv run --python 3.11 --extra dev python -m pytest \
  tests/preprocessing/test_residual_gradient.py -k 'components or rank_one or non_eeg' -q
```

Expected: collection fails because `apply_residual_obs` is not defined.

- [ ] **Step 3: Implement cross-fitted projection and audit rows**

Add this implementation to `eeg_pipeline/preprocessing/residual_gradient.py`:

```python
@dataclass(frozen=True)
class ResidualObsResult:
    raw: mne.io.BaseRaw
    component_rows: tuple[dict[str, float | int | str], ...]


def apply_residual_obs(
    raw: mne.io.BaseRaw,
    layout: VolumeLayout,
    *,
    n_components: int,
    n_folds: int,
) -> ResidualObsResult:
    if n_components < 0:
        raise ValueError("n_components must be non-negative.")
    if n_folds < 2 or n_folds > layout.n_epochs:
        raise ValueError("n_folds must be between 2 and the number of volume epochs.")

    corrected = raw.copy().load_data()
    if n_components == 0:
        return ResidualObsResult(raw=corrected, component_rows=())

    picks = mne.pick_types(corrected.info, eeg=True, exclude=[])
    offsets = np.arange(layout.epoch_samples, dtype=np.int64)
    sample_matrix = layout.starts[:, np.newaxis] + offsets[np.newaxis, :]
    fold_ids = np.arange(layout.n_epochs, dtype=np.int64) % n_folds
    rows: list[dict[str, float | int | str]] = []

    for pick in picks:
        channel_epochs = corrected._data[pick, sample_matrix].copy()
        centered_epochs = channel_epochs - channel_epochs.mean(axis=1, keepdims=True)
        corrected_epochs = channel_epochs.copy()

        for fold in range(n_folds):
            held_mask = fold_ids == fold
            train = centered_epochs[~held_mask]
            if n_components > min(train.shape):
                raise ValueError(
                    f"n_components={n_components} exceeds basis rank {min(train.shape)}."
                )
            _, singular_values, right_vectors = np.linalg.svd(train, full_matrices=False)
            basis = _normalize_component_signs(right_vectors[:n_components])
            held = centered_epochs[held_mask]
            fitted = (held @ basis.T) @ basis
            corrected_epochs[held_mask] -= fitted
            explained = singular_values[:n_components] ** 2
            total = float(np.sum(singular_values**2))
            rows.append(
                {
                    "channel": corrected.ch_names[pick],
                    "fold": fold,
                    "n_components": n_components,
                    "training_variance_explained": float(np.sum(explained) / total),
                    "held_removed_rms_v": float(np.sqrt(np.mean(fitted**2))),
                }
            )

        corrected._data[pick, sample_matrix] = corrected_epochs

    return ResidualObsResult(raw=corrected, component_rows=tuple(rows))


def _normalize_component_signs(components: NDArray[np.float64]) -> NDArray[np.float64]:
    normalized = components.copy()
    maxima = np.argmax(np.abs(normalized), axis=1)
    signs = np.sign(normalized[np.arange(normalized.shape[0]), maxima])
    signs[signs == 0] = 1
    return normalized * signs[:, np.newaxis]
```

Direct `_data` access is acceptable only inside this focused preprocessing kernel after
`raw.copy().load_data()`. It avoids repeated public setter calls while preserving the caller's Raw.

- [ ] **Step 4: Run all preprocessing tests**

Run:

```bash
uv run --python 3.11 --extra dev python -m pytest tests/preprocessing -q
```

Expected: all preprocessing tests pass.

- [ ] **Step 5: Commit the OBS kernel**

```bash
git add eeg_pipeline/preprocessing/residual_gradient.py \
  tests/preprocessing/test_residual_gradient.py
git commit -m "feat: add cross-fitted residual OBS"
```

### Task 3: Harmonic and injected-signal QC

**Files:**
- Create: `eeg_pipeline/analysis/qc/residual_gradient.py`
- Create: `tests/analysis/test_residual_gradient_qc.py`

- [ ] **Step 1: Write failing tests for injection recovery and selection**

```python
from __future__ import annotations

import mne
import numpy as np

from eeg_pipeline.analysis.qc.residual_gradient import (
    InjectionSettings,
    SelectionThresholds,
    build_validation_injection,
    evaluate_injection_recovery,
    select_component_count,
)


def test_validation_injection_is_deterministic_and_not_volume_locked() -> None:
    settings = InjectionSettings(
        frequencies_hz=(15.5, 26.0, 34.0, 49.5, 72.0),
        sinusoid_amplitude_uv=1.0,
        transient_amplitude_uv=2.0,
        transient_spacing_s=13.7,
    )

    first = build_validation_injection(60_000, 1_000.0, settings)
    second = build_validation_injection(60_000, 1_000.0, settings)

    np.testing.assert_array_equal(first.signal_v, second.signal_v)
    assert all(sample % 900 != 0 for sample in first.transient_samples)


def test_recovery_reports_exact_identity() -> None:
    settings = InjectionSettings()
    injection = build_validation_injection(60_000, 1_000.0, settings)

    metrics = evaluate_injection_recovery(
        expected=injection,
        recovered_v=injection.signal_v,
        sfreq=1_000.0,
    )

    assert metrics.minimum_sinusoid_amplitude_ratio == 1.0
    assert metrics.maximum_phase_error_deg == 0.0
    assert metrics.transient_peak_ratio == 1.0


def test_selection_returns_smallest_fully_eligible_order() -> None:
    run_rows = []
    for run in range(1, 7):
        for count, reduction in ((0, 0.0), (1, 2.0), (2, 4.0), (3, 5.0)):
            row = {"run": run, "n_components": count, "volume_locked_rms": 10.0 - reduction}
            for label in ("18_23", "38_43", "56_67", "77_85"):
                row[f"harmonic_{label}_peak_power_db"] = 20.0 - reduction
                row[f"harmonic_{label}_prominence_db"] = 10.0 - reduction
            run_rows.append(row)
    preservation_rows = [
        {
            "n_components": count,
            "minimum_sinusoid_amplitude_ratio": 0.98 if count <= 2 else 0.90,
            "maximum_phase_error_deg": 2.0,
            "transient_peak_ratio": 0.98,
            "outside_harmonic_psd_change_db": 0.2,
        }
        for count in (1, 2, 3)
    ]

    decision = select_component_count(
        run_rows,
        preservation_rows,
        (0, 1, 2, 3),
        SelectionThresholds(),
    )

    assert decision.selected_components == 1
    assert decision.status == "accepted"
```

- [ ] **Step 2: Run the QC tests and verify the module is absent**

Run:

```bash
uv run --python 3.11 --extra dev python -m pytest \
  tests/analysis/test_residual_gradient_qc.py -q
```

Expected: collection fails with `ModuleNotFoundError`.

- [ ] **Step 3: Implement deterministic injections and explicit decision gates**

Create `eeg_pipeline/analysis/qc/residual_gradient.py` with these public contracts:

```python
"""Outcome-blind QC for residual scanner-gradient correction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import mne
import numpy as np
from numpy.typing import NDArray

from eeg_pipeline.preprocessing.residual_gradient import VolumeLayout


HARMONIC_LABELS = ("18_23", "38_43", "56_67", "77_85")


@dataclass(frozen=True)
class InjectionSettings:
    frequencies_hz: tuple[float, ...] = (15.5, 26.0, 34.0, 49.5, 72.0)
    sinusoid_amplitude_uv: float = 1.0
    transient_amplitude_uv: float = 2.0
    transient_spacing_s: float = 13.7
    transient_width_s: float = 0.08


@dataclass(frozen=True)
class ValidationInjection:
    signal_v: NDArray[np.float64]
    sinusoid_v: NDArray[np.float64]
    transient_v: NDArray[np.float64]
    transient_samples: tuple[int, ...]
    frequencies_hz: tuple[float, ...]


@dataclass(frozen=True)
class RecoveryMetrics:
    minimum_sinusoid_amplitude_ratio: float
    maximum_phase_error_deg: float
    transient_peak_ratio: float


@dataclass(frozen=True)
class PsdChangeMetrics:
    median_change_db: float
    maximum_channel_absolute_change_db: float


@dataclass(frozen=True)
class SelectionThresholds:
    minimum_sinusoid_amplitude_ratio: float = 0.95
    maximum_phase_error_deg: float = 5.0
    minimum_transient_peak_ratio: float = 0.95
    maximum_outside_harmonic_psd_change_db: float = 0.5
    maximum_run_prominence_increase_db: float = 1.0


@dataclass(frozen=True)
class ComponentDecision:
    status: str
    selected_components: int | None
    evaluated_components: tuple[int, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_validation_injection(
    n_times: int,
    sfreq: float,
    settings: InjectionSettings,
) -> ValidationInjection:
    if n_times <= 0 or sfreq <= 0:
        raise ValueError("n_times and sfreq must be positive.")
    time = np.arange(n_times, dtype=float) / sfreq
    phases = np.linspace(0.17, 1.31, len(settings.frequencies_hz))
    components = [
        np.sin(2 * np.pi * frequency * time + phase)
        for frequency, phase in zip(settings.frequencies_hz, phases, strict=True)
    ]
    sinusoid = settings.sinusoid_amplitude_uv * 1e-6 * np.mean(components, axis=0)

    first_s = 5.35
    event_times = np.arange(first_s, time[-1] - 1.0, settings.transient_spacing_s)
    transient_samples = tuple(np.rint(event_times * sfreq).astype(int))
    transient = np.zeros(n_times, dtype=float)
    width_samples = settings.transient_width_s * sfreq
    support = np.arange(n_times)
    for sample in transient_samples:
        transient += np.exp(-0.5 * ((support - sample) / width_samples) ** 2)
    transient *= settings.transient_amplitude_uv * 1e-6
    return ValidationInjection(
        signal_v=sinusoid + transient,
        sinusoid_v=sinusoid,
        transient_v=transient,
        transient_samples=transient_samples,
        frequencies_hz=settings.frequencies_hz,
    )


def evaluate_injection_recovery(
    *,
    expected: ValidationInjection,
    recovered_v: NDArray[np.float64],
    sfreq: float,
) -> RecoveryMetrics:
    if recovered_v.shape != expected.signal_v.shape:
        raise ValueError("recovered_v shape must match the validation injection.")
    time = np.arange(recovered_v.size, dtype=float) / sfreq
    design_columns: list[NDArray[np.float64]] = []
    amplitude_ratios: list[float] = []
    phase_errors: list[float] = []
    for frequency in expected.frequencies_hz:
        angle = 2 * np.pi * frequency * time
        design_columns.extend([np.sin(angle), np.cos(angle)])
    design = np.column_stack(design_columns)
    expected_coefficients = np.linalg.lstsq(design, expected.signal_v, rcond=None)[0]
    recovered_coefficients = np.linalg.lstsq(design, recovered_v, rcond=None)[0]
    for index in range(len(expected.frequencies_hz)):
        pair = slice(2 * index, 2 * index + 2)
        expected_pair = expected_coefficients[pair]
        recovered_pair = recovered_coefficients[pair]
        amplitude_ratios.append(np.linalg.norm(recovered_pair) / np.linalg.norm(expected_pair))
        expected_phase = np.arctan2(expected_pair[1], expected_pair[0])
        recovered_phase = np.arctan2(recovered_pair[1], recovered_pair[0])
        phase = np.angle(np.exp(1j * (recovered_phase - expected_phase)), deg=True)
        phase_errors.append(abs(float(phase)))
    expected_residual = expected.signal_v - design @ expected_coefficients
    recovered_residual = recovered_v - design @ recovered_coefficients
    expected_peak = max(expected_residual[sample] for sample in expected.transient_samples)
    recovered_peak = max(recovered_residual[sample] for sample in expected.transient_samples)
    return RecoveryMetrics(
        minimum_sinusoid_amplitude_ratio=float(min(amplitude_ratios)),
        maximum_phase_error_deg=float(max(phase_errors)),
        transient_peak_ratio=float(recovered_peak / expected_peak),
    )


def select_component_count(
    run_rows: Sequence[dict[str, Any]],
    preservation_rows: Sequence[dict[str, Any]],
    component_counts: Sequence[int],
    thresholds: SelectionThresholds,
) -> ComponentDecision:
    references = [row for row in run_rows if row["n_components"] == 0]
    reference_by_run = {int(row["run"]): row for row in references}
    reasons: list[str] = []

    for count in sorted(value for value in component_counts if value > 0):
        candidates = [row for row in run_rows if row["n_components"] == count]
        preservation = [
            row for row in preservation_rows if row["n_components"] == count
        ]
        if not preservation:
            raise ValueError(f"No signal-preservation rows for components={count}.")
        eligible = (
            min(row["minimum_sinusoid_amplitude_ratio"] for row in preservation)
            >= thresholds.minimum_sinusoid_amplitude_ratio
            and max(row["maximum_phase_error_deg"] for row in preservation)
            <= thresholds.maximum_phase_error_deg
            and min(row["transient_peak_ratio"] for row in preservation)
            >= thresholds.minimum_transient_peak_ratio
            and max(
                abs(row["outside_harmonic_psd_change_db"])
                for row in preservation
            )
            <= thresholds.maximum_outside_harmonic_psd_change_db
        )
        for label in HARMONIC_LABELS:
            candidate_power = np.median(
                [row[f"harmonic_{label}_peak_power_db"] for row in candidates]
            )
            reference_power = np.median(
                [row[f"harmonic_{label}_peak_power_db"] for row in references]
            )
            candidate_prominence = np.median(
                [row[f"harmonic_{label}_prominence_db"] for row in candidates]
            )
            reference_prominence = np.median(
                [row[f"harmonic_{label}_prominence_db"] for row in references]
            )
            eligible &= candidate_power < reference_power
            eligible &= candidate_prominence < reference_prominence
            eligible &= all(
                row[f"harmonic_{label}_prominence_db"]
                <= reference_by_run[int(row["run"])][f"harmonic_{label}_prominence_db"]
                + thresholds.maximum_run_prominence_increase_db
                for row in candidates
            )
        eligible &= np.median([row["volume_locked_rms"] for row in candidates]) < np.median(
            [row["volume_locked_rms"] for row in references]
        )
        if eligible:
            return ComponentDecision("accepted", count, tuple(component_counts), ())
        reasons.append(f"components={count} failed one or more fixed gates")

    return ComponentDecision("rejected", None, tuple(component_counts), tuple(reasons))
```

Add these array-only helpers in the same module:

```python
def compute_volume_locked_rms(
    data_v: NDArray[np.float64],
    starts: NDArray[np.int64],
    epoch_samples: int,
) -> float:
    if data_v.ndim != 2:
        raise ValueError("data_v must have shape (channels, samples).")
    offsets = np.arange(epoch_samples, dtype=np.int64)
    epochs = data_v[:, starts[:, np.newaxis] + offsets[np.newaxis, :]]
    phase_locked_mean = epochs.mean(axis=1)
    return float(np.sqrt(np.mean(phase_locked_mean**2)))


def compute_outside_harmonic_psd_change(
    before_v: NDArray[np.float64],
    after_v: NDArray[np.float64],
    *,
    sfreq: float,
    nperseg: int,
    harmonic_windows_hz: Sequence[tuple[float, float]],
) -> PsdChangeMetrics:
    from scipy.signal import welch

    frequencies, before_psd = welch(before_v, fs=sfreq, nperseg=nperseg, axis=-1)
    _, after_psd = welch(after_v, fs=sfreq, nperseg=nperseg, axis=-1)
    retained = (frequencies >= 13.0) & (frequencies <= 95.0)
    for low_hz, high_hz in harmonic_windows_hz:
        retained &= ~((frequencies >= low_hz) & (frequencies <= high_hz))
    floor = np.finfo(float).tiny
    before_db = 10 * np.log10(np.maximum(before_psd, floor))
    after_db = 10 * np.log10(np.maximum(after_psd, floor))
    channel_changes = np.median(after_db[:, retained] - before_db[:, retained], axis=1)
    return PsdChangeMetrics(
        median_change_db=float(np.median(channel_changes)),
        maximum_channel_absolute_change_db=float(np.max(np.abs(channel_changes))),
    )


def summarize_candidate(
    raw: mne.io.BaseRaw,
    *,
    layout: VolumeLayout,
    source_file: str | Path,
    run: int,
    n_components: int,
    nperseg: int,
    harmonic_windows_hz: Sequence[tuple[float, float]],
) -> dict[str, Any]:
    from eeg_pipeline.analysis.qc.scanner_harmonics import (
        FrequencyWindow,
        summarize_scanner_harmonics,
    )
    from scipy.signal import welch

    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    data = raw.get_data(picks)
    frequencies, psd = welch(data, fs=raw.info["sfreq"], nperseg=nperseg, axis=-1)
    windows = tuple(
        FrequencyWindow(f"scanner_{low_hz:g}_{high_hz:g}", low_hz, high_hz)
        for low_hz, high_hz in harmonic_windows_hz
    )
    row = summarize_scanner_harmonics(
        freqs=frequencies,
        psd=psd,
        source_file=source_file,
        sfreq=float(raw.info["sfreq"]),
        n_samples=raw.n_times,
        channel_names=[raw.ch_names[pick] for pick in picks],
        harmonic_windows=windows,
    )
    row["run"] = run
    row["n_components"] = n_components
    row["volume_locked_rms"] = compute_volume_locked_rms(
        data,
        layout.starts,
        layout.epoch_samples,
    )
    return row
```

- [ ] **Step 4: Run QC tests and the existing scanner-harmonic tests**

Run:

```bash
uv run --python 3.11 --extra dev python -m pytest \
  tests/analysis/test_residual_gradient_qc.py \
  tests/analysis/test_scanner_harmonics.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit QC and decision rules**

```bash
git add eeg_pipeline/analysis/qc/residual_gradient.py \
  tests/analysis/test_residual_gradient_qc.py
git commit -m "feat: add residual OBS preservation gates"
```

### Task 4: Strict benchmark configuration and pilot discovery

**Files:**
- Create: `studies/pain_study/scripts/config/residual_gradient_benchmark.yaml`
- Create: `studies/pain_study/scripts/benchmark_residual_gradient.py`
- Create: `tests/scripts/test_benchmark_residual_gradient.py`

- [ ] **Step 1: Write failing configuration and discovery tests**

```python
from __future__ import annotations

from pathlib import Path

import pytest

from studies.pain_study.scripts.benchmark_residual_gradient import (
    discover_pilot_files,
    load_benchmark_config,
)


def test_default_config_has_fixed_pilot_and_component_grid() -> None:
    path = Path(
        "studies/pain_study/scripts/config/residual_gradient_benchmark.yaml"
    )

    config = load_benchmark_config(path)

    assert config.pilot_subject == "0006"
    assert config.component_counts == (0, 1, 2, 3, 4)
    assert config.obs.volume_marker == "Volume/V  1"


def test_discovery_returns_only_corrected_pilot_headers(tmp_path: Path) -> None:
    eeg_dir = tmp_path / "sub-0006" / "eeg"
    eeg_dir.mkdir(parents=True)
    accepted = eeg_dir / "run1_sub0006_scannerpulse_corrected.vhdr"
    accepted.touch()
    (eeg_dir / "run1_sub0006.vhdr").touch()

    assert discover_pilot_files(tmp_path, "0006") == [accepted]


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("pilot_subject: '0006'\nunknown: true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown benchmark configuration keys"):
        load_benchmark_config(path)
```

- [ ] **Step 2: Run the script tests and verify missing interfaces**

Run:

```bash
uv run --python 3.11 --extra dev python -m pytest \
  tests/scripts/test_benchmark_residual_gradient.py -q
```

Expected: collection fails because the benchmark module is absent.

- [ ] **Step 3: Add the fixed YAML configuration**

```yaml
pilot_subject: "0006"
expected_runs: 6
component_counts: [0, 1, 2, 3, 4]
obs:
  expected_sfreq_hz: 1000.0
  volume_marker: "Volume/V  1"
  tr_s: 0.9
  min_complete_epochs: 50
  n_folds: 5
harmonic_windows_hz:
  - [18.0, 23.0]
  - [38.0, 43.0]
  - [56.0, 67.0]
  - [77.0, 85.0]
welch:
  nperseg: 16384
  overlap_fraction: 0.5
injection:
  frequencies_hz: [15.5, 26.0, 34.0, 49.5, 72.0]
  sinusoid_amplitude_uv: 1.0
  transient_amplitude_uv: 2.0
  transient_spacing_s: 13.7
  transient_width_s: 0.08
acceptance:
  minimum_sinusoid_amplitude_ratio: 0.95
  maximum_phase_error_deg: 5.0
  minimum_transient_peak_ratio: 0.95
  maximum_outside_harmonic_psd_change_db: 0.5
  maximum_run_prominence_increase_db: 1.0
```

- [ ] **Step 4: Implement frozen configuration objects and exact discovery**

Add these frozen configuration objects and strict loader:

```python
@dataclass(frozen=True)
class WelchSettings:
    nperseg: int
    overlap_fraction: float


@dataclass(frozen=True)
class BenchmarkConfig:
    pilot_subject: str
    expected_runs: int
    component_counts: tuple[int, ...]
    obs: ResidualObsSettings
    harmonic_windows_hz: tuple[tuple[float, float], ...]
    welch: WelchSettings
    injection: InjectionSettings
    acceptance: SelectionThresholds


def _require_exact_keys(
    mapping: dict[str, Any],
    expected: set[str],
    label: str,
) -> None:
    extra = sorted(set(mapping) - expected)
    missing = sorted(expected - set(mapping))
    if extra:
        raise ValueError(f"Unknown {label} configuration keys: {extra}")
    if missing:
        raise ValueError(f"Missing {label} configuration keys: {missing}")


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Benchmark configuration does not exist: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Benchmark configuration root must be a mapping.")
    _require_exact_keys(
        payload,
        {
            "pilot_subject",
            "expected_runs",
            "component_counts",
            "obs",
            "harmonic_windows_hz",
            "welch",
            "injection",
            "acceptance",
        },
        "benchmark",
    )
    _require_exact_keys(
        payload["obs"],
        {
            "expected_sfreq_hz",
            "volume_marker",
            "tr_s",
            "min_complete_epochs",
            "n_folds",
        },
        "obs",
    )
    _require_exact_keys(payload["welch"], {"nperseg", "overlap_fraction"}, "welch")
    _require_exact_keys(
        payload["injection"],
        {
            "frequencies_hz",
            "sinusoid_amplitude_uv",
            "transient_amplitude_uv",
            "transient_spacing_s",
            "transient_width_s",
        },
        "injection",
    )
    _require_exact_keys(
        payload["acceptance"],
        {
            "minimum_sinusoid_amplitude_ratio",
            "maximum_phase_error_deg",
            "minimum_transient_peak_ratio",
            "maximum_outside_harmonic_psd_change_db",
            "maximum_run_prominence_increase_db",
        },
        "acceptance",
    )
    injection_values = dict(payload["injection"])
    injection_values["frequencies_hz"] = tuple(injection_values["frequencies_hz"])
    config = BenchmarkConfig(
        pilot_subject=str(payload["pilot_subject"]),
        expected_runs=int(payload["expected_runs"]),
        component_counts=tuple(map(int, payload["component_counts"])),
        obs=ResidualObsSettings(**payload["obs"]),
        harmonic_windows_hz=tuple(
            tuple(map(float, window)) for window in payload["harmonic_windows_hz"]
        ),
        welch=WelchSettings(**payload["welch"]),
        injection=InjectionSettings(**injection_values),
        acceptance=SelectionThresholds(**payload["acceptance"]),
    )
    if config.component_counts != (0, 1, 2, 3, 4):
        raise ValueError("component_counts must be exactly [0, 1, 2, 3, 4].")
    if config.expected_runs != 6:
        raise ValueError("expected_runs must be exactly 6 for the prespecified pilot.")
    return config
```

Implement discovery exactly as:

```python
def discover_pilot_files(source_root: str | Path, subject: str) -> list[Path]:
    root = Path(source_root)
    if not root.is_dir():
        raise NotADirectoryError(f"Source root is not a directory: {root}")
    eeg_dir = root / f"sub-{subject}" / "eeg"
    paths = sorted(eeg_dir.glob("*_scannerpulse_corrected.vhdr"))
    if not paths:
        raise FileNotFoundError(
            f"No _scannerpulse_corrected.vhdr files found for sub-{subject}: {eeg_dir}"
        )
    return paths
```

At the start of `run_pilot_benchmark()`, enforce the run count before loading data:

```python
paths = discover_pilot_files(source_root, config.pilot_subject)
if len(paths) != config.expected_runs:
    raise ValueError(
        f"Expected {config.expected_runs} pilot runs for sub-{config.pilot_subject}, "
        f"found {len(paths)}."
    )
```

- [ ] **Step 5: Run the configuration/discovery tests**

Run the Task 4 pytest command again.

Expected: all Task 4 tests pass.

- [ ] **Step 6: Commit configuration and discovery**

```bash
git add studies/pain_study/scripts/config/residual_gradient_benchmark.yaml \
  studies/pain_study/scripts/benchmark_residual_gradient.py \
  tests/scripts/test_benchmark_residual_gradient.py
git commit -m "feat: configure residual OBS pilot benchmark"
```

### Task 5: Atomic candidate outputs, audits, and provenance

**Files:**
- Modify: `studies/pain_study/scripts/benchmark_residual_gradient.py`
- Modify: `tests/scripts/test_benchmark_residual_gradient.py`

- [ ] **Step 1: Add failing report-writer tests**

```python
import json

import pandas as pd

from studies.pain_study.scripts.benchmark_residual_gradient import (
    OutputPolicy,
    write_benchmark_reports,
)


def test_reports_are_complete_and_provenance_is_machine_readable(tmp_path: Path) -> None:
    run_rows = [{"run": 1, "n_components": count} for count in range(5)]
    component_rows = [{"run": 1, "channel": "Fz", "fold": 0, "n_components": 1}]
    preservation_rows = [{"n_components": 1, "minimum_sinusoid_amplitude_ratio": 0.99}]
    provenance = {"subject": "0006", "inputs": ["run1.vhdr"], "config_sha256": "abc"}

    paths = write_benchmark_reports(
        output_root=tmp_path,
        run_rows=run_rows,
        component_rows=component_rows,
        preservation_rows=preservation_rows,
        provenance=provenance,
        decision={"status": "accepted", "selected_components": 1},
        policy=OutputPolicy.ERROR,
    )

    assert pd.read_csv(paths.run_audit, sep="\t").shape[0] == 5
    assert json.loads(paths.provenance.read_text())["subject"] == "0006"


def test_report_writer_refuses_existing_files(tmp_path: Path) -> None:
    (tmp_path / "residual_obs_run_audit.tsv").write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_benchmark_reports(
            output_root=tmp_path,
            run_rows=[],
            component_rows=[],
            preservation_rows=[],
            provenance={},
            decision={},
            policy=OutputPolicy.ERROR,
        )
```

- [ ] **Step 2: Run tests and verify missing writer symbols**

Run the Task 4 pytest command.

Expected: collection fails because `OutputPolicy` and `write_benchmark_reports` are absent.

- [ ] **Step 3: Implement explicit output policy and atomic report writes**

Add:

```python
class OutputPolicy(str, Enum):
    ERROR = "error"
    OVERWRITE = "overwrite"


@dataclass(frozen=True)
class ReportPaths:
    run_audit: Path
    component_audit: Path
    preservation_audit: Path
    provenance: Path
    decision: Path


def write_benchmark_reports(
    *,
    output_root: str | Path,
    run_rows: Sequence[dict[str, Any]],
    component_rows: Sequence[dict[str, Any]],
    preservation_rows: Sequence[dict[str, Any]],
    provenance: dict[str, Any],
    decision: dict[str, Any],
    policy: OutputPolicy,
) -> ReportPaths:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = ReportPaths(
        run_audit=root / "residual_obs_run_audit.tsv",
        component_audit=root / "residual_obs_component_variance.tsv",
        preservation_audit=root / "residual_obs_signal_preservation.tsv",
        provenance=root / "residual_obs_provenance.json",
        decision=root / "residual_obs_decision.json",
    )
    for path in paths.__dict__.values():
        if path.exists() and policy is OutputPolicy.ERROR:
            raise FileExistsError(f"Refusing to overwrite existing benchmark output: {path}")

    _atomic_tsv(paths.run_audit, run_rows)
    _atomic_tsv(paths.component_audit, component_rows)
    _atomic_tsv(paths.preservation_audit, preservation_rows)
    _atomic_json(paths.provenance, provenance)
    _atomic_json(paths.decision, decision)
    return paths
```

Use these atomic writers and candidate saver:

```python
def _temporary_path(destination: Path) -> Path:
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{destination.stem}.",
        suffix=destination.suffix,
        dir=destination.parent,
        delete=False,
    )
    handle.close()
    return Path(handle.name)


def _atomic_tsv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty benchmark table: {path}")
    temporary = _temporary_path(path)
    try:
        pd.DataFrame(rows).to_csv(temporary, sep="\t", index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = _temporary_path(path)
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_candidate(raw: mne.io.BaseRaw, path: Path, policy: OutputPolicy) -> None:
    if path.exists() and policy is OutputPolicy.ERROR:
        raise FileExistsError(f"Refusing to overwrite candidate derivative: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name.removesuffix('_raw.fif')}_tmp_raw.fif")
    temporary.unlink(missing_ok=True)
    try:
        raw.save(temporary, overwrite=False, verbose="ERROR")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
```

- [ ] **Step 4: Implement one-run and pilot orchestration**

Add the following orchestration. `summarize_candidate()` is a focused helper in the QC module that
runs Welch on all EEG channels, calls `summarize_scanner_harmonics()`, adds the explicit run and
component count, and adds `compute_volume_locked_rms()`.

```python
@dataclass(frozen=True)
class RunBenchmark:
    run_rows: tuple[dict[str, Any], ...]
    component_rows: tuple[dict[str, Any], ...]
    preservation_rows: tuple[dict[str, Any], ...]
    candidate_paths: tuple[Path, ...]
    provenance: dict[str, Any]


def _add_injection(
    raw: mne.io.BaseRaw,
    injection: ValidationInjection,
) -> mne.io.BaseRaw:
    injected = raw.copy().load_data()
    picks = mne.pick_types(injected.info, eeg=True, exclude=[])
    injected._data[picks] += injection.signal_v[np.newaxis, :]
    return injected


def _assert_candidate_contract(
    source: mne.io.BaseRaw,
    candidate: mne.io.BaseRaw,
    layout: VolumeLayout,
) -> None:
    if candidate.n_times != source.n_times:
        raise RuntimeError("Residual OBS changed the sample count.")
    if candidate.ch_names != source.ch_names:
        raise RuntimeError("Residual OBS changed channel order or names.")
    if candidate.info["meas_date"] != source.info["meas_date"]:
        raise RuntimeError("Residual OBS changed the measurement date.")
    annotations_match = (
        np.array_equal(candidate.annotations.onset, source.annotations.onset)
        and np.array_equal(candidate.annotations.duration, source.annotations.duration)
        and np.array_equal(candidate.annotations.description, source.annotations.description)
        and candidate.annotations.orig_time == source.annotations.orig_time
    )
    if not annotations_match:
        raise RuntimeError("Residual OBS changed annotations.")
    if not np.isfinite(candidate.get_data()).all():
        raise RuntimeError("Residual OBS produced non-finite samples.")
    eeg_picks = mne.pick_types(source.info, eeg=True, exclude=[])
    non_eeg = np.setdiff1d(np.arange(len(source.ch_names)), eeg_picks)
    np.testing.assert_array_equal(candidate.get_data(non_eeg), source.get_data(non_eeg))
    eligible = np.zeros(source.n_times, dtype=bool)
    for start in layout.starts:
        eligible[start : start + layout.epoch_samples] = True
    np.testing.assert_array_equal(
        candidate.get_data()[:, ~eligible],
        source.get_data()[:, ~eligible],
    )


def benchmark_run(
    vhdr_path: Path,
    *,
    output_root: Path,
    config: BenchmarkConfig,
    output_policy: OutputPolicy,
) -> RunBenchmark:
    source_path = validate_brainvision_source(vhdr_path)
    raw = mne.io.read_raw_brainvision(source_path, preload=True, verbose="ERROR")
    layout = build_volume_layout(raw, config.obs)
    run = int(get_run_index(source_path))
    injection = build_validation_injection(raw.n_times, raw.info["sfreq"], config.injection)
    injected_raw = _add_injection(raw, injection)
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])

    run_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    preservation_rows: list[dict[str, Any]] = []
    candidate_paths: list[Path] = []
    for count in config.component_counts:
        baseline = apply_residual_obs(
            raw,
            layout,
            n_components=count,
            n_folds=config.obs.n_folds,
        )
        injected = apply_residual_obs(
            injected_raw,
            layout,
            n_components=count,
            n_folds=config.obs.n_folds,
        )
        _assert_candidate_contract(raw, baseline.raw, layout)
        recovered = np.median(
            injected.raw.get_data(eeg_picks) - baseline.raw.get_data(eeg_picks),
            axis=0,
        )
        recovery = evaluate_injection_recovery(
            expected=injection,
            recovered_v=recovered,
            sfreq=float(raw.info["sfreq"]),
        )
        outside_change = compute_outside_harmonic_psd_change(
            raw.get_data(eeg_picks),
            baseline.raw.get_data(eeg_picks),
            sfreq=float(raw.info["sfreq"]),
            nperseg=config.welch.nperseg,
            harmonic_windows_hz=config.harmonic_windows_hz,
        )
        run_rows.append(
            summarize_candidate(
                baseline.raw,
                layout=layout,
                source_file=source_path,
                run=run,
                n_components=count,
                nperseg=config.welch.nperseg,
                harmonic_windows_hz=config.harmonic_windows_hz,
            )
        )
        component_rows.extend({"run": run, **row} for row in baseline.component_rows)
        preservation_rows.append(
            {
                "run": run,
                "n_components": count,
                **asdict(recovery),
                "outside_harmonic_psd_change_db": outside_change.median_change_db,
                "maximum_channel_outside_harmonic_psd_change_db": (
                    outside_change.maximum_channel_absolute_change_db
                ),
            }
        )
        if count > 0:
            candidate_path = (
                output_root
                / f"sub-{config.pilot_subject}"
                / "eeg"
                / f"sub-{config.pilot_subject}_run-{run:02d}_desc-resobs{count:02d}_raw.fif"
            )
            _save_candidate(baseline.raw, candidate_path, output_policy)
            candidate_paths.append(candidate_path)

    provenance = {
        "input": str(source_path),
        "input_sha256": _sha256(source_path),
        "run": run,
        "eligible_epochs": layout.n_epochs,
        "blocks": int(layout.block_ids.max()) + 1,
        "excluded_samples": raw.n_times - layout.n_epochs * layout.epoch_samples,
        "candidate_paths": list(map(str, candidate_paths)),
        "mne_version": mne.__version__,
        "numpy_version": np.__version__,
    }
    return RunBenchmark(
        tuple(run_rows),
        tuple(component_rows),
        tuple(preservation_rows),
        tuple(candidate_paths),
        provenance,
    )


def run_pilot_benchmark(
    *,
    source_root: Path,
    output_root: Path,
    config: BenchmarkConfig,
    output_policy: OutputPolicy,
) -> ComponentDecision:
    paths = discover_pilot_files(source_root, config.pilot_subject)
    if len(paths) != config.expected_runs:
        raise ValueError(
            f"Expected {config.expected_runs} pilot runs, found {len(paths)}."
        )
    results = [
        benchmark_run(
            path,
            output_root=output_root,
            config=config,
            output_policy=output_policy,
        )
        for path in paths
    ]
    run_rows = [row for result in results for row in result.run_rows]
    component_rows = [row for result in results for row in result.component_rows]
    preservation_rows = [row for result in results for row in result.preservation_rows]
    decision = select_component_count(
        run_rows,
        preservation_rows,
        config.component_counts,
        config.acceptance,
    )
    provenance = {
        "subject": config.pilot_subject,
        "config": asdict(config),
        "runs": [result.provenance for result in results],
    }
    write_benchmark_reports(
        output_root=output_root,
        run_rows=run_rows,
        component_rows=component_rows,
        preservation_rows=preservation_rows,
        provenance=provenance,
        decision=decision.to_dict(),
        policy=output_policy,
    )
    return decision
```

Use a streaming fingerprint helper; do not catch exceptions inside either orchestration function:

```python
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
```

- [ ] **Step 5: Run script, preprocessing, and QC tests**

Run:

```bash
uv run --python 3.11 --extra dev python -m pytest \
  tests/scripts/test_benchmark_residual_gradient.py \
  tests/preprocessing/test_residual_gradient.py \
  tests/analysis/test_residual_gradient_qc.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit orchestration and reports**

```bash
git add studies/pain_study/scripts/benchmark_residual_gradient.py \
  tests/scripts/test_benchmark_residual_gradient.py
git commit -m "feat: write residual OBS benchmark derivatives"
```

### Task 6: CLI and user documentation

**Files:**
- Modify: `studies/pain_study/scripts/benchmark_residual_gradient.py`
- Modify: `tests/scripts/test_benchmark_residual_gradient.py`
- Modify: `studies/pain_study/SCANNER_HARMONICS_QC_README.md`

- [ ] **Step 1: Add a failing CLI parser test**

```python
from studies.pain_study.scripts.benchmark_residual_gradient import build_parser


def test_cli_requires_source_output_and_config() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--source-root",
            "/data/source_data",
            "--output-root",
            "/data/derivatives/qc/residual_gradient_obs_benchmark",
            "--config",
            "config.yaml",
            "--output-policy",
            "error",
        ]
    )

    assert args.output_policy == "error"
```

- [ ] **Step 2: Implement the CLI without algorithm-selection flags**

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark residual OBS on BrainVision scanner- and pulse-corrected EEG."
        )
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-policy",
        choices=[policy.value for policy in OutputPolicy],
        default=OutputPolicy.ERROR.value,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_benchmark_config(args.config)
    decision = run_pilot_benchmark(
        source_root=args.source_root,
        output_root=args.output_root,
        config=config,
        output_policy=OutputPolicy(args.output_policy),
    )
    print(json.dumps(decision.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Document the exact benchmark command and interpretation boundary**

Add to `studies/pain_study/SCANNER_HARMONICS_QC_README.md`:

~~~~markdown
## Residual OBS pilot benchmark

The BrainVision `*_scannerpulse_corrected` files have already undergone 21-volume scanner AAS,
100 Hz low-pass filtering, downsampling to 1 kHz, R-peak detection, and 21-beat pulse correction.
The residual OBS benchmark therefore does not run full FASTR or repeat AAS.

```bash
uv run --python 3.11 --extra dev python \
  studies/pain_study/scripts/benchmark_residual_gradient.py \
  --source-root /Volumes/KINGSTON/EEG_fMRI_data/source_data \
  --output-root \
    /Volumes/KINGSTON/EEG_fMRI_data/derivatives/qc/residual_gradient_obs_benchmark \
  --config \
    studies/pain_study/scripts/config/residual_gradient_benchmark.yaml \
  --output-policy error
```

The command evaluates the prespecified `sub-0006` pilot only. A nonzero component count is accepted
only if every scanner-harmonic and injected-signal preservation gate passes across all six runs.
Benchmark outputs are not production preprocessing derivatives and must not replace current MNE
inputs without a separate approved production-integration design.
~~~~

- [ ] **Step 4: Run CLI tests and documentation checks**

Run:

```bash
uv run --python 3.11 --extra dev python -m pytest \
  tests/scripts/test_benchmark_residual_gradient.py -q
uv run --python 3.11 --extra dev ruff check \
  eeg_pipeline/analysis/qc/residual_gradient.py \
  eeg_pipeline/preprocessing/residual_gradient.py \
  studies/pain_study/scripts/benchmark_residual_gradient.py \
  tests/analysis/test_residual_gradient_qc.py \
  tests/preprocessing/test_residual_gradient.py \
  tests/scripts/test_benchmark_residual_gradient.py
```

Expected: tests pass and Ruff reports no errors.

- [ ] **Step 5: Commit CLI and documentation**

```bash
git add studies/pain_study/scripts/benchmark_residual_gradient.py \
  tests/scripts/test_benchmark_residual_gradient.py \
  studies/pain_study/SCANNER_HARMONICS_QC_README.md
git commit -m "docs: add residual OBS benchmark workflow"
```

### Task 7: Run and inspect the six-run pilot

**Files:**
- External output only:
  `/Volumes/KINGSTON/EEG_fMRI_data/derivatives/qc/residual_gradient_obs_benchmark/`

- [ ] **Step 1: Confirm the external output directory does not contain prior benchmark files**

Run:

```bash
find /Volumes/KINGSTON/EEG_fMRI_data/derivatives/qc/residual_gradient_obs_benchmark \
  -maxdepth 1 -type f 2>/dev/null
```

Expected: no output. If files exist, stop and obtain explicit authorization before using
`--output-policy overwrite`.

- [ ] **Step 2: Run the benchmark with the frozen configuration**

Run the exact documented command from Task 6.

Expected: exit code 0 and a JSON decision with either:

```json
{"status": "accepted", "selected_components": 1}
```

where the selected value may be 1–4, or:

```json
{"status": "rejected", "selected_components": null}
```

No other status is valid.

- [ ] **Step 3: Verify complete run/component coverage**

Run:

```bash
uv run --python 3.11 python - <<'PY'
from pathlib import Path
import json
import pandas as pd

root = Path(
    "/Volumes/KINGSTON/EEG_fMRI_data/derivatives/qc/"
    "residual_gradient_obs_benchmark"
)
run_audit = pd.read_csv(root / "residual_obs_run_audit.tsv", sep="\t")
assert run_audit.shape[0] == 30
assert sorted(run_audit["run"].unique()) == [1, 2, 3, 4, 5, 6]
assert sorted(run_audit["n_components"].unique()) == [0, 1, 2, 3, 4]
decision = json.loads((root / "residual_obs_decision.json").read_text())
assert decision["status"] in {"accepted", "rejected"}
print(run_audit.groupby("n_components")["volume_locked_rms"].median())
print(decision)
PY
```

Expected: assertions pass and five component-count RMS medians plus one decision are printed.

- [ ] **Step 4: Inspect every acceptance gate, not only the selected aggregate**

```bash
uv run --python 3.11 python - <<'PY'
from pathlib import Path
import json
import numpy as np
import pandas as pd

from eeg_pipeline.analysis.qc.residual_gradient import select_component_count
from studies.pain_study.scripts.benchmark_residual_gradient import load_benchmark_config

root = Path(
    "/Volumes/KINGSTON/EEG_fMRI_data/derivatives/qc/"
    "residual_gradient_obs_benchmark"
)
config = load_benchmark_config(
    "studies/pain_study/scripts/config/residual_gradient_benchmark.yaml"
)
runs = pd.read_csv(root / "residual_obs_run_audit.tsv", sep="\t")
preservation = pd.read_csv(root / "residual_obs_signal_preservation.tsv", sep="\t")
numeric = runs.select_dtypes(include=["number"]).to_numpy()
assert np.isfinite(numeric).all()
assert np.isfinite(preservation.select_dtypes(include=["number"]).to_numpy()).all()
reference = runs[runs.n_components == 0].set_index("run")
for count in (1, 2, 3, 4):
    candidate = runs[runs.n_components == count].set_index("run")
    print(f"components={count}")
    for label in ("18_23", "38_43", "56_67", "77_85"):
        power = f"harmonic_{label}_peak_power_db"
        prominence = f"harmonic_{label}_prominence_db"
        power_change = candidate[power] - reference[power]
        prominence_change = candidate[prominence] - reference[prominence]
        print(
            label,
            "median_power_change_db=", float(power_change.median()),
            "median_prominence_change_db=", float(prominence_change.median()),
            "maximum_prominence_increase_db=", float(prominence_change.max()),
        )
    rows = preservation[preservation.n_components == count]
    print(
        "minimum_amplitude_ratio=", rows.minimum_sinusoid_amplitude_ratio.min(),
        "maximum_phase_error_deg=", rows.maximum_phase_error_deg.max(),
        "minimum_transient_ratio=", rows.transient_peak_ratio.min(),
        "maximum_outside_change_db=", rows.outside_harmonic_psd_change_db.abs().max(),
        "maximum_channel_outside_change_db=",
        rows.maximum_channel_outside_harmonic_psd_change_db.max(),
    )

computed = select_component_count(
    runs.to_dict("records"),
    preservation.to_dict("records"),
    config.component_counts,
    config.acceptance,
).to_dict()
written = json.loads((root / "residual_obs_decision.json").read_text())
assert computed == written
print(written)
PY
```

Expected: all values are finite, every gate is printed for counts 1–4, and the recomputed decision
exactly equals `residual_obs_decision.json`. Whether accepted or rejected, do not modify Study 1,
Study 2, or production preprocessing in this task.

### Task 8: Final verification and review preparation

**Files:**
- Verify all files changed by Tasks 1–7.

- [ ] **Step 1: Format the implementation mechanically**

```bash
uv run --python 3.11 --extra dev black \
  eeg_pipeline/analysis/qc/residual_gradient.py \
  eeg_pipeline/preprocessing/residual_gradient.py \
  studies/pain_study/scripts/benchmark_residual_gradient.py \
  tests/analysis/test_residual_gradient_qc.py \
  tests/preprocessing/test_residual_gradient.py \
  tests/scripts/test_benchmark_residual_gradient.py
```

Expected: Black exits 0.

- [ ] **Step 2: Run targeted lint and tests**

```bash
uv run --python 3.11 --extra dev ruff check \
  eeg_pipeline/analysis/qc/residual_gradient.py \
  eeg_pipeline/preprocessing/residual_gradient.py \
  studies/pain_study/scripts/benchmark_residual_gradient.py \
  tests/analysis/test_residual_gradient_qc.py \
  tests/preprocessing/test_residual_gradient.py \
  tests/scripts/test_benchmark_residual_gradient.py
uv run --python 3.11 --extra dev python -m pytest \
  tests/analysis/test_residual_gradient_qc.py \
  tests/analysis/test_scanner_harmonics.py \
  tests/preprocessing \
  tests/scripts/test_benchmark_residual_gradient.py -q
```

Expected: Ruff exits 0 and all targeted tests pass.

- [ ] **Step 3: Run repository architecture and maintainability gates**

```bash
make verify-architecture
make verify-maintainability
```

Expected: both commands exit 0.

- [ ] **Step 4: Verify the branch diff and external-source immutability**

```bash
git diff --check main...HEAD
git status --short
git diff --stat main...HEAD
```

Expected: no whitespace errors, a clean worktree, and changes limited to the planned code, tests,
configuration, and documentation. Confirm no files beneath
`/Volumes/KINGSTON/EEG_fMRI_data/source_data` changed during the benchmark.

- [ ] **Step 5: Request code review**

Use the `requesting-code-review` skill. The review must compare the implementation with
`docs/superpowers/specs/2026-07-13-brainvision-residual-gradient-obs-design.md`, inspect the
cross-fitting mathematics and preservation gates, and verify that no production pipeline path was
changed.

- [ ] **Step 6: Commit any review fixes and rerun the affected verification commands**

Use one focused commit per corrected issue. Do not amend already reviewed commits. The final branch
must be clean before invoking `finishing-a-development-branch`.
