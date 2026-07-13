"""Shared continuous-run spectral estimation for Study 1 figures."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

NUMBERED_SUBJECT_PATTERN = re.compile(r"^sub-\d+$")
FINAL_CLEAN_FILENAME_PATTERN = re.compile(
    r"^(?P<subject>sub-\d+)_task-[^_]+_run-(?P<run>[^_]+)_proc-clean_raw\.fif$"
)


@dataclass(frozen=True)
class ContinuousSpectrumSpecification:
    """Welch settings shared by continuous-run spectral figures."""

    frequency_range_hz: tuple[float, float]
    n_fft: int
    n_overlap: int
    sampling_frequency_hz: float

    def __post_init__(self) -> None:
        lower_frequency, upper_frequency = self.frequency_range_hz
        if lower_frequency <= 0.0 or upper_frequency <= lower_frequency:
            raise ValueError("frequency_range_hz must contain increasing positive values.")
        if upper_frequency > self.sampling_frequency_hz / 2.0:
            raise ValueError("frequency_range_hz must not exceed the Nyquist frequency.")
        if self.n_fft <= 0:
            raise ValueError("n_fft must be positive.")
        if self.n_overlap < 0 or self.n_overlap >= self.n_fft:
            raise ValueError("n_overlap must be smaller than n_fft and nonnegative.")
        if self.sampling_frequency_hz <= 0.0:
            raise ValueError("sampling_frequency_hz must be positive.")


@dataclass(frozen=True)
class ContinuousRunSpectrum:
    """One final-clean run reduced to a channel-median linear PSD."""

    subject_id: str
    run_id: str
    source_file: Path | str
    frequencies_hz: np.ndarray
    median_psd_v2_hz: np.ndarray
    n_channels: int
    sampling_frequency_hz: float
    n_samples: int
    recording_duration_s: float
    bad_annotation_duration_s: float
    analyzed_duration_s: float


def parse_final_clean_filename(path: Path) -> tuple[str, str]:
    """Return the participant and run identifiers from a final-clean filename."""
    entities = FINAL_CLEAN_FILENAME_PATTERN.fullmatch(Path(path).name)
    if entities is None:
        raise ValueError(f"Invalid final-clean EEG filename: {Path(path).name}")
    return entities.group("subject"), entities.group("run")


def discover_final_clean_runs(
    derivative_root: Path,
    *,
    task: str,
    excluded_subjects: Sequence[str],
    requested_subjects: Sequence[str] = (),
) -> tuple[Path, ...]:
    """Discover final-clean FIF runs for numbered, eligible participants."""
    excluded = set(excluded_subjects)
    requested = set(requested_subjects)
    candidates = Path(derivative_root).glob(f"sub-*/eeg/sub-*_task-{task}_run-*_proc-clean_raw.fif")
    selected = tuple(
        sorted(
            path
            for path in candidates
            if NUMBERED_SUBJECT_PATTERN.fullmatch(path.parts[-3]) is not None
            and path.parts[-3] not in excluded
            and (not requested or path.parts[-3] in requested)
        )
    )
    if not selected:
        raise FileNotFoundError(
            "No numbered-participant final-clean FIF files found for task "
            f"{task!r} in {derivative_root}."
        )
    return selected


def bad_annotation_duration_s(
    *,
    onsets_s: np.ndarray,
    durations_s: np.ndarray,
    descriptions: Sequence[str],
    recording_start_s: float,
    recording_duration_s: float,
) -> float:
    """Return the clipped union duration of annotations beginning with ``BAD``."""
    onsets = np.asarray(onsets_s, dtype=float)
    durations = np.asarray(durations_s, dtype=float)
    labels = tuple(str(description) for description in descriptions)
    if onsets.ndim != 1 or durations.shape != onsets.shape or len(labels) != onsets.size:
        raise ValueError("Annotation onsets, durations, and descriptions must align.")
    if not np.isfinite(onsets).all() or not np.isfinite(durations).all():
        raise ValueError("Annotation onsets and durations must be finite.")
    if np.any(durations < 0.0):
        raise ValueError("Annotation durations must be nonnegative.")
    if not np.isfinite(recording_start_s) or recording_duration_s <= 0.0:
        raise ValueError("Recording bounds must be finite with positive duration.")

    recording_end_s = recording_start_s + recording_duration_s
    intervals = sorted(
        (
            max(float(onset), recording_start_s),
            min(float(onset + duration), recording_end_s),
        )
        for onset, duration, label in zip(onsets, durations, labels, strict=True)
        if label.lower().startswith("bad")
        and onset + duration > recording_start_s
        and onset < recording_end_s
    )
    if not intervals:
        return 0.0

    union_duration_s = 0.0
    current_start, current_end = intervals[0]
    for interval_start, interval_end in intervals[1:]:
        if interval_start <= current_end:
            current_end = max(current_end, interval_end)
            continue
        union_duration_s += current_end - current_start
        current_start, current_end = interval_start, interval_end
    return union_duration_s + current_end - current_start


def estimate_raw_continuous_run_spectrum(
    raw,
    *,
    subject_id: str,
    run_id: str,
    source_file: Path | str,
    specification: ContinuousSpectrumSpecification,
) -> ContinuousRunSpectrum:
    """Estimate a validated channel-median Welch PSD from one loaded run."""
    sampling_frequency_hz = float(raw.info["sfreq"])
    if sampling_frequency_hz != specification.sampling_frequency_hz:
        raise ValueError(
            f"Unexpected sampling frequency in {source_file}: {sampling_frequency_hz} Hz."
        )
    n_samples = int(raw.n_times)
    if n_samples < specification.n_fft:
        raise ValueError(f"Run has fewer samples than n_fft: {source_file}")

    recording_duration_s = n_samples / sampling_frequency_hz
    rejected_duration_s = bad_annotation_duration_s(
        onsets_s=raw.annotations.onset,
        durations_s=raw.annotations.duration,
        descriptions=raw.annotations.description,
        recording_start_s=float(raw.first_time),
        recording_duration_s=recording_duration_s,
    )
    lower_frequency, upper_frequency = specification.frequency_range_hz
    spectrum = raw.compute_psd(
        method="welch",
        fmin=lower_frequency,
        fmax=upper_frequency,
        n_fft=specification.n_fft,
        n_per_seg=specification.n_fft,
        n_overlap=specification.n_overlap,
        picks="eeg",
        reject_by_annotation=True,
        verbose=False,
    )
    frequencies = np.asarray(spectrum.freqs, dtype=float)
    channel_psd = np.asarray(spectrum.get_data(), dtype=float)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError(f"Unexpected PSD frequency shape for {source_file}: {frequencies.shape}.")
    if channel_psd.ndim != 2 or channel_psd.shape[0] == 0:
        raise ValueError(f"Unexpected PSD channel shape for {source_file}: {channel_psd.shape}.")
    if channel_psd.shape[1] != frequencies.size:
        raise ValueError(f"PSD values and frequencies are misaligned: {source_file}")
    if not np.isfinite(channel_psd).all() or np.any(channel_psd <= 0.0):
        raise ValueError(f"PSD contains nonpositive or non-finite values: {source_file}")

    return ContinuousRunSpectrum(
        subject_id=subject_id,
        run_id=run_id,
        source_file=source_file,
        frequencies_hz=frequencies,
        median_psd_v2_hz=np.median(channel_psd, axis=0),
        n_channels=int(channel_psd.shape[0]),
        sampling_frequency_hz=sampling_frequency_hz,
        n_samples=n_samples,
        recording_duration_s=recording_duration_s,
        bad_annotation_duration_s=rejected_duration_s,
        analyzed_duration_s=recording_duration_s - rejected_duration_s,
    )


def estimate_continuous_run_spectrum(
    path: Path,
    specification: ContinuousSpectrumSpecification,
) -> ContinuousRunSpectrum:
    """Estimate a validated channel-median Welch PSD for one final-clean run."""
    import mne

    source_path = Path(path)
    subject_id, run_id = parse_final_clean_filename(source_path)
    raw = mne.io.read_raw_fif(source_path, preload=False, verbose="ERROR")
    return estimate_raw_continuous_run_spectrum(
        raw,
        subject_id=subject_id,
        run_id=run_id,
        source_file=source_path,
        specification=specification,
    )


__all__ = [
    "ContinuousRunSpectrum",
    "ContinuousSpectrumSpecification",
    "bad_annotation_duration_s",
    "discover_final_clean_runs",
    "estimate_continuous_run_spectrum",
    "estimate_raw_continuous_run_spectrum",
    "parse_final_clean_filename",
]
