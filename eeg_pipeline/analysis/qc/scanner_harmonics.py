"""Scanner-harmonic QC for gamma-band EEG analyses.

This module reports narrowband scanner-harmonic contamination without changing
the underlying EEG. Gamma summaries use a fixed frequency mask that excludes
scanner-locked peaks while preserving adjacent gamma bins.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.signal import find_peaks, welch


DEFAULT_QC_CHANNELS = (
    "Fp1",
    "F3",
    "C3",
    "O1",
    "Fz",
    "Cz",
    "Pz",
    "Oz",
    "POz",
    "FC3",
    "PO3",
    "PO7",
)


@dataclass(frozen=True)
class FrequencyWindow:
    """Named inclusive frequency window in Hz."""

    name: str
    low_hz: float
    high_hz: float

    def __post_init__(self) -> None:
        low_hz = float(self.low_hz)
        high_hz = float(self.high_hz)
        if not self.name:
            raise ValueError("FrequencyWindow.name must be non-empty.")
        if not np.isfinite(low_hz) or not np.isfinite(high_hz):
            raise ValueError("FrequencyWindow bounds must be finite.")
        if low_hz <= 0 or high_hz <= 0:
            raise ValueError("FrequencyWindow bounds must be positive.")
        if low_hz >= high_hz:
            raise ValueError("FrequencyWindow.low_hz must be less than high_hz.")

    @property
    def label(self) -> str:
        return _window_label(self.low_hz, self.high_hz)

    @property
    def column_prefix(self) -> str:
        return f"harmonic_{_window_column_label(self.low_hz, self.high_hz)}"


DEFAULT_GAMMA_WINDOW = FrequencyWindow("masked_gamma", 30.1, 80.0)
DEFAULT_GAMMA_EXCLUSIONS = (
    FrequencyWindow("scanner_38_43", 38.0, 43.0),
    FrequencyWindow("scanner_56_67", 56.0, 67.0),
    FrequencyWindow("scanner_77_85", 77.0, 85.0),
)
DEFAULT_HARMONIC_WINDOWS = (
    FrequencyWindow("scanner_18_23", 18.0, 23.0),
    *DEFAULT_GAMMA_EXCLUSIONS,
)


def build_frequency_mask(
    freqs: Sequence[float],
    *,
    include: FrequencyWindow,
    exclusions: Sequence[FrequencyWindow] = (),
) -> np.ndarray:
    """Return a boolean frequency mask for one include window minus exclusions."""
    freq_array = _validate_freqs(freqs)
    mask = (freq_array >= include.low_hz) & (freq_array <= include.high_hz)
    for exclusion in exclusions:
        excluded = (freq_array >= exclusion.low_hz) & (freq_array <= exclusion.high_hz)
        mask &= ~excluded
    if not np.any(mask):
        raise ValueError(
            f"No frequencies remain after applying exclusions to {include.label} Hz."
        )
    return mask


def summarize_scanner_harmonics(
    *,
    freqs: Sequence[float],
    psd: np.ndarray,
    source_file: str | Path,
    sfreq: float,
    n_samples: int,
    channel_names: Sequence[str],
    gamma_window: FrequencyWindow = DEFAULT_GAMMA_WINDOW,
    gamma_exclusions: Sequence[FrequencyWindow] = DEFAULT_GAMMA_EXCLUSIONS,
    harmonic_windows: Sequence[FrequencyWindow] = DEFAULT_HARMONIC_WINDOWS,
) -> dict[str, Any]:
    """Build a flat scanner-harmonic QC summary from channel-wise PSD values."""
    freq_array = _validate_freqs(freqs)
    psd_array = _validate_psd(psd, n_freqs=freq_array.size)
    _validate_sampling(freq_array, sfreq, n_samples)
    if len(channel_names) != psd_array.shape[0]:
        raise ValueError("channel_names length must match PSD channel count.")

    source_path = Path(source_file)
    gamma_mask = build_frequency_mask(
        freq_array,
        include=gamma_window,
        exclusions=gamma_exclusions,
    )
    full_gamma_mask = build_frequency_mask(freq_array, include=gamma_window)
    median_psd_db = _median_psd_db(psd_array)
    entities = _extract_bids_entities(source_path.name)

    summary: dict[str, Any] = {
        "source_file": str(source_path),
        "subject": entities.get("subject", ""),
        "task": entities.get("task", ""),
        "run": entities.get("run", ""),
        "sampling_frequency_hz": float(sfreq),
        "n_samples": int(n_samples),
        "duration_s": float(n_samples) / float(sfreq),
        "n_channels": int(psd_array.shape[0]),
        "channel_names": ",".join(channel_names),
        "gamma_full_range_hz": gamma_window.label,
        "gamma_excluded_ranges_hz": _format_windows(gamma_exclusions),
        "gamma_masked_ranges_hz": _format_retained_ranges(gamma_window, gamma_exclusions),
        "gamma_retained_fraction": float(np.count_nonzero(gamma_mask))
        / float(np.count_nonzero(full_gamma_mask)),
        "gamma_full_power_db": _mean_db(median_psd_db, full_gamma_mask),
        "gamma_masked_power_db": _mean_db(median_psd_db, gamma_mask),
    }
    summary["gamma_masked_minus_full_db"] = (
        summary["gamma_masked_power_db"] - summary["gamma_full_power_db"]
    )

    for window in harmonic_windows:
        summary.update(_summarize_harmonic_window(freq_array, median_psd_db, window))

    return summary


def analyze_brainvision_file(
    vhdr_path: str | Path,
    *,
    channels: Sequence[str] | None = DEFAULT_QC_CHANNELS,
    nperseg: int = 16_384,
    min_samples: int = 4_096,
    gamma_window: FrequencyWindow = DEFAULT_GAMMA_WINDOW,
    gamma_exclusions: Sequence[FrequencyWindow] = DEFAULT_GAMMA_EXCLUSIONS,
    harmonic_windows: Sequence[FrequencyWindow] = DEFAULT_HARMONIC_WINDOWS,
) -> dict[str, Any]:
    """Read one BrainVision file and return its scanner-harmonic QC summary."""
    freqs, psd, sfreq, n_samples, channel_names = _read_brainvision_psd(
        Path(vhdr_path),
        channels=channels,
        nperseg=nperseg,
        min_samples=min_samples,
    )
    return summarize_scanner_harmonics(
        freqs=freqs,
        psd=psd,
        source_file=vhdr_path,
        sfreq=sfreq,
        n_samples=n_samples,
        channel_names=channel_names,
        gamma_window=gamma_window,
        gamma_exclusions=gamma_exclusions,
        harmonic_windows=harmonic_windows,
    )


def discover_brainvision_files(
    input_root: str | Path,
    *,
    subjects: Sequence[str] | None = None,
    pattern: str = "*.vhdr",
) -> list[Path]:
    """Find BrainVision header files, optionally restricted to subject labels."""
    root = Path(input_root)
    if not root.exists():
        raise FileNotFoundError(f"Input root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Input root is not a directory: {root}")

    subject_filter = _normalize_subject_filter(subjects)
    candidates = sorted(
        path for path in root.rglob(pattern) if path.is_file() and not path.name.startswith("._")
    )
    if subject_filter is None:
        return candidates

    selected = [
        path
        for path in candidates
        if _extract_bids_entities(path.name).get("subject") in subject_filter
    ]
    return selected


def write_scanner_harmonic_reports(
    rows: Sequence[dict[str, Any]],
    output_dir: str | Path,
    *,
    stem: str = "scanner_harmonic_gamma_qc",
) -> tuple[Path, Path]:
    """Write TSV and JSON scanner-harmonic reports."""
    if not rows:
        raise ValueError("Cannot write scanner-harmonic report with no rows.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tsv_path = output_path / f"{stem}.tsv"
    json_path = output_path / f"{stem}.json"

    fieldnames = _ordered_report_columns(rows)
    with tsv_path.open("w", encoding="utf-8", newline="") as fid:
        writer = csv.DictWriter(fid, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps(list(rows), indent=2), encoding="utf-8")
    return tsv_path, json_path


def _read_brainvision_psd(
    vhdr_path: Path,
    *,
    channels: Sequence[str] | None,
    nperseg: int,
    min_samples: int,
) -> tuple[np.ndarray, np.ndarray, float, int, list[str]]:
    if not vhdr_path.exists():
        raise FileNotFoundError(f"BrainVision header does not exist: {vhdr_path}")
    if vhdr_path.suffix.lower() != ".vhdr":
        raise ValueError(f"Expected a .vhdr BrainVision header, got: {vhdr_path}")
    if nperseg <= 0:
        raise ValueError("nperseg must be positive.")
    if min_samples <= 0:
        raise ValueError("min_samples must be positive.")

    import mne

    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    n_samples = int(raw.n_times)
    if n_samples < min_samples:
        raise ValueError(
            f"PSD requires at least {min_samples} samples, got {n_samples}: {vhdr_path}"
        )

    picks = _resolve_channel_picks(raw, channels)
    channel_names = [raw.ch_names[pick] for pick in picks]
    data = raw.get_data(picks=picks)
    segment_length = min(int(nperseg), n_samples)
    freqs, psd = welch(
        data,
        fs=sfreq,
        nperseg=segment_length,
        noverlap=segment_length // 2,
        axis=-1,
    )
    return freqs, psd, sfreq, n_samples, channel_names


def _resolve_channel_picks(raw: Any, channels: Sequence[str] | None) -> list[int]:
    if channels is not None:
        missing = [channel for channel in channels if channel not in raw.ch_names]
        if missing:
            missing_list = ", ".join(missing)
            raise ValueError(f"Requested QC channels are missing: {missing_list}")
        return [raw.ch_names.index(channel) for channel in channels]

    import mne

    picks = mne.pick_types(raw.info, eeg=True, ecg=False, stim=False, misc=False, exclude=())
    if len(picks) == 0:
        raise ValueError("No EEG channels found in BrainVision file.")
    return list(map(int, picks))


def _summarize_harmonic_window(
    freqs: np.ndarray,
    median_psd_db: np.ndarray,
    window: FrequencyWindow,
) -> dict[str, float]:
    peak_hz, peak_power_db, prominence_db = select_harmonic_peak(
        freqs,
        median_psd_db,
        window,
    )
    prefix = window.column_prefix
    return {
        f"{prefix}_peak_hz": peak_hz,
        f"{prefix}_peak_power_db": peak_power_db,
        f"{prefix}_prominence_db": prominence_db,
    }


def select_harmonic_peak(
    freqs: Sequence[float],
    spectrum_db: Sequence[float],
    window: FrequencyWindow,
) -> tuple[float, float, float]:
    """Select the strongest prominent peak inside one harmonic window."""
    frequency_array = _validate_freqs(freqs)
    spectrum_array = np.asarray(spectrum_db, dtype=float)
    if spectrum_array.shape != frequency_array.shape:
        raise ValueError("spectrum_db must match the one-dimensional frequency grid.")
    if not np.all(np.isfinite(spectrum_array)):
        raise ValueError("spectrum_db must contain only finite values.")

    mask = (frequency_array >= window.low_hz) & (frequency_array <= window.high_hz)
    if not np.any(mask):
        raise ValueError(f"No PSD frequencies fall inside harmonic window {window.label} Hz.")

    window_freqs = frequency_array[mask]
    window_db = spectrum_array[mask]
    max_index = int(np.argmax(window_db))
    peaks, properties = find_peaks(window_db, prominence=0)
    if len(peaks):
        prominence_index = int(np.argmax(properties["prominences"]))
        peak_index = int(peaks[prominence_index])
        prominence_db = float(properties["prominences"][prominence_index])
    else:
        peak_index = max_index
        prominence_db = 0.0
    return (
        float(window_freqs[peak_index]),
        float(window_db[peak_index]),
        prominence_db,
    )


def _validate_freqs(freqs: Sequence[float]) -> np.ndarray:
    freq_array = np.asarray(freqs, dtype=float)
    if freq_array.ndim != 1:
        raise ValueError("freqs must be a 1D array.")
    if freq_array.size < 2:
        raise ValueError("freqs must contain at least two values.")
    if not np.all(np.isfinite(freq_array)):
        raise ValueError("freqs must contain only finite values.")
    if not np.all(np.diff(freq_array) > 0):
        raise ValueError("freqs must be strictly increasing.")
    return freq_array


def _validate_psd(psd: np.ndarray, *, n_freqs: int) -> np.ndarray:
    psd_array = np.asarray(psd, dtype=float)
    if psd_array.ndim != 2:
        raise ValueError("psd must have shape (n_channels, n_freqs).")
    if psd_array.shape[0] == 0:
        raise ValueError("psd must contain at least one channel.")
    if psd_array.shape[1] != n_freqs:
        raise ValueError("psd frequency dimension must match freqs length.")
    if not np.all(np.isfinite(psd_array)):
        raise ValueError("psd must contain only finite values.")
    if np.any(psd_array < 0):
        raise ValueError("psd must be non-negative.")
    return psd_array


def _validate_sampling(freqs: np.ndarray, sfreq: float, n_samples: int) -> None:
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("sfreq must be a finite positive number.")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    nyquist = sfreq / 2.0
    if freqs[-1] > nyquist + np.finfo(float).eps:
        raise ValueError("freqs cannot exceed the Nyquist frequency.")


def _median_psd_db(psd: np.ndarray) -> np.ndarray:
    median_psd = np.median(psd, axis=0)
    return 10.0 * np.log10(np.maximum(median_psd, np.finfo(float).tiny))


def _mean_db(values_db: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(values_db[mask]))


def _extract_bids_entities(filename: str) -> dict[str, str]:
    entities: dict[str, str] = {}
    patterns = {
        "subject": r"(?:^|_)sub-?([^_]+)",
        "task": r"(?:^|_)task-([^_]+)",
        "run": r"(?:^|_)run-?([^_]+)",
    }
    for name, pattern in patterns.items():
        match = re.search(pattern, filename)
        if match:
            entities[name] = match.group(1)
    return entities


def _normalize_subject_filter(subjects: Sequence[str] | None) -> set[str] | None:
    if subjects is None:
        return None
    normalized = {subject.replace("sub-", "", 1) for subject in subjects}
    if not normalized:
        raise ValueError("subjects filter cannot be empty.")
    return normalized


def _format_windows(windows: Sequence[FrequencyWindow]) -> str:
    return ";".join(window.label for window in windows)


def _format_retained_ranges(
    include: FrequencyWindow,
    exclusions: Sequence[FrequencyWindow],
) -> str:
    retained = _subtract_windows(include, exclusions)
    return ";".join(_window_label(low, high) for low, high in retained)


def _subtract_windows(
    include: FrequencyWindow,
    exclusions: Sequence[FrequencyWindow],
) -> list[tuple[float, float]]:
    retained = [(include.low_hz, include.high_hz)]
    relevant_exclusions = sorted(
        (
            (max(include.low_hz, exclusion.low_hz), min(include.high_hz, exclusion.high_hz))
            for exclusion in exclusions
            if exclusion.high_hz > include.low_hz and exclusion.low_hz < include.high_hz
        ),
        key=lambda bounds: bounds[0],
    )

    for exclusion_low, exclusion_high in relevant_exclusions:
        next_retained: list[tuple[float, float]] = []
        for low, high in retained:
            if exclusion_high <= low or exclusion_low >= high:
                next_retained.append((low, high))
                continue
            if low < exclusion_low:
                next_retained.append((low, exclusion_low))
            if exclusion_high < high:
                next_retained.append((exclusion_high, high))
        retained = next_retained

    if not retained:
        raise ValueError(f"Exclusions fully cover include window {include.label} Hz.")
    return retained


def _window_label(low_hz: float, high_hz: float) -> str:
    return f"{_format_hz(low_hz)}-{_format_hz(high_hz)}"


def _window_column_label(low_hz: float, high_hz: float) -> str:
    return f"{_format_hz(low_hz)}_{_format_hz(high_hz)}".replace(".", "p")


def _format_hz(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def _ordered_report_columns(rows: Sequence[dict[str, Any]]) -> list[str]:
    preferred = [
        "source_file",
        "subject",
        "task",
        "run",
        "sampling_frequency_hz",
        "n_samples",
        "duration_s",
        "n_channels",
        "channel_names",
        "gamma_full_range_hz",
        "gamma_excluded_ranges_hz",
        "gamma_masked_ranges_hz",
        "gamma_retained_fraction",
        "gamma_full_power_db",
        "gamma_masked_power_db",
        "gamma_masked_minus_full_db",
    ]
    keys = set().union(*(row.keys() for row in rows))
    harmonic_keys = sorted(key for key in keys if key.startswith("harmonic_"))
    remaining = sorted(keys - set(preferred) - set(harmonic_keys))
    return [key for key in preferred if key in keys] + harmonic_keys + remaining
