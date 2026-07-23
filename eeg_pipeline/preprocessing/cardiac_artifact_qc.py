"""Cardiac artifact QC driven by preserved Analyzer R markers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mne
import numpy as np
import pandas as pd

from eeg_pipeline.preprocessing.pulse_artifact_qc import PULSE_MARKER_DESCRIPTION

PULSE_EVENT_ID = 999


@dataclass(frozen=True)
class CardiacAttenuationMetrics:
    """Run-level marker-locked attenuation after ICA application."""

    recording_id: str
    marker_count: int
    before_rms_uv: float
    after_rms_uv: float
    attenuation_percent: float


def pulse_marker_events(raw: mne.io.BaseRaw) -> np.ndarray:
    """Create MNE events from preserved BrainVision Analyzer R annotations."""
    events, _ = mne.events_from_annotations(
        raw,
        event_id={PULSE_MARKER_DESCRIPTION: PULSE_EVENT_ID},
        use_rounding=True,
        verbose="ERROR",
    )
    if len(events) == 0:
        raise ValueError(f"Raw recording contains no {PULSE_MARKER_DESCRIPTION!r} markers.")
    return events


def _marker_locked_rms(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    *,
    baseline: tuple[float, float],
    measurement_window: tuple[float, float],
) -> float:
    epochs = mne.Epochs(
        raw,
        events,
        event_id=PULSE_EVENT_ID,
        tmin=baseline[0],
        tmax=measurement_window[1],
        baseline=baseline,
        picks="eeg",
        preload=True,
        reject_by_annotation=True,
        verbose="ERROR",
    )
    if len(epochs) == 0:
        raise ValueError("No valid marker-locked EEG epochs remain for cardiac QC.")
    evoked = epochs.average()
    window = (evoked.times >= measurement_window[0]) & (evoked.times <= measurement_window[1])
    if not np.any(window):
        raise ValueError("Cardiac QC measurement window contains no samples.")
    return float(np.sqrt(np.mean(evoked.data[:, window] ** 2)))


def compute_cardiac_attenuation(
    before: mne.io.BaseRaw,
    after: mne.io.BaseRaw,
    *,
    recording_id: str,
    baseline: tuple[float, float],
    measurement_window: tuple[float, float],
) -> CardiacAttenuationMetrics:
    """Measure cardiac-locked EEG RMS before and after ICA."""
    recordings_align = (
        before.ch_names == after.ch_names
        and before.n_times == after.n_times
        and before.first_samp == after.first_samp
        and before.info["sfreq"] == after.info["sfreq"]
    )
    if not recordings_align:
        raise ValueError(f"{recording_id}: before/after raw recordings do not align.")

    before_referenced = before.copy().set_eeg_reference(
        "average",
        projection=False,
        verbose=False,
    )
    after_referenced = after.copy().set_eeg_reference(
        "average",
        projection=False,
        verbose=False,
    )
    events = pulse_marker_events(before_referenced)
    before_rms = _marker_locked_rms(
        before_referenced,
        events,
        baseline=baseline,
        measurement_window=measurement_window,
    )
    after_rms = _marker_locked_rms(
        after_referenced,
        events,
        baseline=baseline,
        measurement_window=measurement_window,
    )
    if before_rms == 0:
        raise ValueError(f"{recording_id}: pre-ICA marker-locked EEG RMS is zero.")
    attenuation_percent = 100.0 * (1.0 - after_rms / before_rms)
    return CardiacAttenuationMetrics(
        recording_id=recording_id,
        marker_count=len(events),
        before_rms_uv=before_rms * 1e6,
        after_rms_uv=after_rms * 1e6,
        attenuation_percent=attenuation_percent,
    )


def add_marker_ctps_columns(
    components: pd.DataFrame,
    scores: np.ndarray,
    *,
    threshold: float,
) -> pd.DataFrame:
    """Add marker-based CTPS evidence without changing exclusion statuses."""
    expected_components = np.arange(len(scores))
    if "component" not in components or not np.array_equal(
        components["component"].to_numpy(), expected_components
    ):
        raise ValueError("ICA component table does not match the marker CTPS scores.")
    if not 0 < threshold <= 1:
        raise ValueError(f"CTPS threshold must be in (0, 1], got {threshold}.")

    result = components.copy()
    result["analyzer_marker_ctps_score"] = np.asarray(scores, dtype=float)
    result["analyzer_marker_ctps_flag"] = np.asarray(scores) >= threshold
    return result


def compute_marker_ctps_scores(
    raws: Iterable[mne.io.BaseRaw],
    ica: mne.preprocessing.ICA,
    *,
    threshold: float,
    epoch_window: tuple[float, float],
) -> np.ndarray:
    """Score ICA components using CTPS epochs anchored to Analyzer markers."""
    marker_epochs = []
    for raw in raws:
        marker_epochs.append(
            mne.Epochs(
                raw,
                pulse_marker_events(raw),
                event_id=PULSE_EVENT_ID,
                tmin=epoch_window[0],
                tmax=epoch_window[1],
                baseline=None,
                picks="eeg",
                preload=True,
                reject_by_annotation=True,
                verbose="ERROR",
            )
        )
    if not marker_epochs:
        raise ValueError("No filtered raw recordings were provided for marker CTPS QC.")
    epochs = mne.concatenate_epochs(marker_epochs, verbose="ERROR")
    if len(epochs) == 0:
        raise ValueError("No valid marker-locked epochs remain for CTPS QC.")
    _, scores = ica.find_bads_ecg(
        epochs,
        method="ctps",
        threshold=threshold,
        verbose="ERROR",
    )
    return np.asarray(scores, dtype=float)


def write_cardiac_attenuation_qc(
    recordings: Iterable[tuple[str, mne.io.BaseRaw, mne.io.BaseRaw]],
    *,
    output_path: Path,
    baseline: tuple[float, float],
    measurement_window: tuple[float, float],
) -> Path:
    """Write run-level marker-locked EEG attenuation before versus after ICA."""
    rows = []
    for recording_id, before, after in recordings:
        metrics = compute_cardiac_attenuation(
            before,
            after,
            recording_id=recording_id,
            baseline=baseline,
            measurement_window=measurement_window,
        )
        rows.append(
            {
                "recording_id": metrics.recording_id,
                "marker_count": metrics.marker_count,
                "before_rms_uv": metrics.before_rms_uv,
                "after_rms_uv": metrics.after_rms_uv,
                "attenuation_percent": metrics.attenuation_percent,
            }
        )
    if not rows:
        raise ValueError("No before/after recordings were provided for cardiac QC.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(output_path, sep="\t", index=False)
    _write_cardiac_attenuation_figure(table, output_path.with_suffix(".png"))
    return output_path


def _write_cardiac_attenuation_figure(table: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    positions = np.arange(len(table))
    figure_height = max(3.0, 0.45 * len(table) + 1.5)
    figure, axis = plt.subplots(figsize=(8.0, figure_height), constrained_layout=True)
    for position, before, after in zip(
        positions,
        table["before_rms_uv"],
        table["after_rms_uv"],
        strict=True,
    ):
        axis.plot([before, after], [position, position], color="0.65", linewidth=1.5)
    axis.scatter(table["before_rms_uv"], positions, label="Before ICA", color="#B24C3B")
    axis.scatter(table["after_rms_uv"], positions, label="After ICA", color="#276B8A")
    axis.set_yticks(positions, table["recording_id"])
    axis.set_xlabel("R-marker-locked EEG RMS (µV)")
    axis.set_title("Cardiac artifact attenuation after ICA")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _resolve_subjects(pipeline_root: Path, subjects: list[str]) -> list[str]:
    if subjects == ["all"]:
        resolved = sorted(
            path.name.removeprefix("sub-") for path in pipeline_root.glob("sub-*") if path.is_dir()
        )
    else:
        resolved = [subject.removeprefix("sub-") for subject in subjects]
    if not resolved:
        raise FileNotFoundError(f"No subject derivatives found under {pipeline_root}.")
    return resolved


def _visible_matches(directory: Path, pattern: str) -> list[Path]:
    return sorted(
        path
        for path in directory.glob(pattern)
        if path.is_file() and not path.name.startswith("._")
    )


def _require_single_path(paths: list[Path], description: str) -> Path:
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected one {description}, found {len(paths)}: {paths}")
    return paths[0]


def run_marker_ctps_qc(
    *,
    pipeline_root: Path,
    subjects: list[str],
    task: str | None,
    threshold: float,
    epoch_window: tuple[float, float],
) -> Path:
    """Add Analyzer-marker CTPS flags to native MNE-BIDS component tables."""
    task_selector = f"_task-{task}_" if task is not None else "_task-"
    summary_frames = []
    for subject in _resolve_subjects(pipeline_root, subjects):
        eeg_directory = pipeline_root / f"sub-{subject}" / "eeg"
        ica_path = _require_single_path(
            _visible_matches(eeg_directory, f"sub-{subject}_proc-icafit_ica.fif"),
            f"sub-{subject} ICA fit",
        )
        components_path = _require_single_path(
            _visible_matches(eeg_directory, f"sub-{subject}_proc-ica_components.tsv"),
            f"sub-{subject} ICA component table",
        )
        filtered_paths = [
            path
            for path in _visible_matches(
                eeg_directory,
                f"sub-{subject}_task-*_run-*_proc-filt_raw.fif",
            )
            if task_selector in path.name
        ]
        if not filtered_paths:
            raise FileNotFoundError(f"No filtered raw runs found for sub-{subject}, task={task!r}.")

        ica = mne.preprocessing.read_ica(ica_path, verbose="ERROR")
        raws = [mne.io.read_raw_fif(path, preload=True, verbose="ERROR") for path in filtered_paths]
        scores = compute_marker_ctps_scores(
            raws,
            ica,
            threshold=threshold,
            epoch_window=epoch_window,
        )
        components = pd.read_csv(components_path, sep="\t")
        updated = add_marker_ctps_columns(
            components,
            scores,
            threshold=threshold,
        )
        updated.to_csv(components_path, sep="\t", index=False)

        summary = updated[
            [
                "component",
                "status",
                "status_description",
                "analyzer_marker_ctps_score",
                "analyzer_marker_ctps_flag",
            ]
        ].copy()
        summary.insert(0, "participant_id", f"sub-{subject}")
        summary_frames.append(summary)

    output_path = (
        pipeline_root
        / "qc"
        / f"{'task-' + task + '_' if task is not None else ''}desc-markerctps_components.tsv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(summary_frames, ignore_index=True).to_csv(
        output_path,
        sep="\t",
        index=False,
    )
    return output_path


def run_cardiac_attenuation_qc(
    *,
    pipeline_root: Path,
    subjects: list[str],
    task: str | None,
    baseline: tuple[float, float],
    measurement_window: tuple[float, float],
) -> Path:
    """Pair filtered and clean runs and write marker-locked attenuation QC."""
    task_selector = f"_task-{task}_" if task is not None else "_task-"
    recordings = []
    for subject in _resolve_subjects(pipeline_root, subjects):
        eeg_directory = pipeline_root / f"sub-{subject}" / "eeg"
        filtered_paths = [
            path
            for path in _visible_matches(
                eeg_directory,
                f"sub-{subject}_task-*_run-*_proc-filt_raw.fif",
            )
            if task_selector in path.name
        ]
        if not filtered_paths:
            raise FileNotFoundError(f"No filtered raw runs found for sub-{subject}, task={task!r}.")
        for filtered_path in filtered_paths:
            clean_path = filtered_path.with_name(
                filtered_path.name.replace("_proc-filt_raw.fif", "_proc-clean_raw.fif")
            )
            if not clean_path.is_file():
                raise FileNotFoundError(f"Missing ICA-cleaned raw file: {clean_path}")
            recording_id = filtered_path.name.removesuffix("_proc-filt_raw.fif")
            recordings.append(
                (
                    recording_id,
                    mne.io.read_raw_fif(filtered_path, preload=True, verbose="ERROR"),
                    mne.io.read_raw_fif(clean_path, preload=True, verbose="ERROR"),
                )
            )

    output_path = (
        pipeline_root
        / "qc"
        / f"{'task-' + task + '_' if task is not None else ''}desc-cardiacattenuation_qc.tsv"
    )
    return write_cardiac_attenuation_qc(
        recordings,
        output_path=output_path,
        baseline=baseline,
        measurement_window=measurement_window,
    )


__all__ = [
    "CardiacAttenuationMetrics",
    "add_marker_ctps_columns",
    "compute_cardiac_attenuation",
    "compute_marker_ctps_scores",
    "pulse_marker_events",
    "run_cardiac_attenuation_qc",
    "run_marker_ctps_qc",
    "write_cardiac_attenuation_qc",
]
