"""Cardiac artifact QC driven by the recording's beat-marker train."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import mne
import numpy as np
import pandas as pd

from eeg_pipeline.preprocessing.derivatives import (
    FILTERED_RAW_SUFFIX,
    ICA_SUFFIX,
    clean_raw_for_filtered,
    entity_prefix,
    find_filtered_raw_runs,
    find_ica_solutions,
    resolve_subjects,
    runs_for_prefix,
)
from eeg_pipeline.preprocessing.ica_exclusions import components_path_for_ica

#: No default beat annotation. A label is a search instruction, not evidence a train
#: exists, so a caller that names none goes straight to detecting R peaks from the ECG
#: channel rather than searching for somebody else's spelling. Studies supply the label
#: through ``ica.cardiac_review.marker_description``.
DEFAULT_BEAT_MARKER_DESCRIPTION: str | None = None

PULSE_EVENT_ID = 999

#: Channel the R-peak fallback and CTPS scoring read. Callers pass the configured name;
#: this default only keeps the low-level helpers usable on their own.
DEFAULT_ECG_CHANNEL = "ECG"


@dataclass(frozen=True)
class CardiacAttenuationMetrics:
    """Run-level marker-locked attenuation after ICA application."""

    recording_id: str
    marker_count: int
    before_rms_uv: float
    after_rms_uv: float
    attenuation_percent: float
    #: Same quantity in decibels, using 20*log10 because the ratio is one of amplitudes
    #: rather than powers. Kept alongside the percentage so existing QC tables stay
    #: readable.
    attenuation_db: float
    is_fallback: bool = False


def pulse_marker_events(
    raw: mne.io.BaseRaw,
    *,
    ecg_channel: str = DEFAULT_ECG_CHANNEL,
    marker_description: str | None = DEFAULT_BEAT_MARKER_DESCRIPTION,
) -> tuple[np.ndarray, bool]:
    """Create MNE events from the recording's beat annotations, or from the channel.

    The second return value says which of the two produced them. It travels with every
    measurement built on these events, because a marker train and a detection over the
    same recording are not the same evidence and must not be read as though they were.
    """
    events = []
    if marker_description:
        try:
            events, _ = mne.events_from_annotations(
                raw,
                event_id={marker_description: PULSE_EVENT_ID},
                use_rounding=True,
                verbose="ERROR",
            )
        except ValueError as exc:
            if "Could not find any of the events" in str(exc):
                events = []
            else:
                raise
    if len(events) > 0:
        return events, False

    # An upstream detector that fails to find R peaks may fall back to a fixed delay,
    # and doesn't export them. We use MNE's ecg detector to approximate them for QC.
    events, _, _, _ = mne.preprocessing.find_ecg_events(
        raw, ch_name=ecg_channel, event_id=PULSE_EVENT_ID, return_ecg=True, verbose="ERROR"
    )
    if len(events) == 0:
        searched = f"no {marker_description!r} markers" if marker_description else "no beat markers"
        raise ValueError(f"Raw recording contains {searched} and fallback failed.")
    return events, True


def _marker_locked_rms(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    *,
    baseline: tuple[float, float],
    measurement_window: tuple[float, float],
) -> float:
    """Measure the R-locked EEG RMS of the marker-locked average.

    Restricted to EEG. ICA is fitted and applied to EEG alone, so the ECG channel is
    bit-identical before and after — and it is both the largest-amplitude channel and the
    one perfectly phase-locked to the marker, so including it would dominate the average
    and drive the measured attenuation toward zero on a recording that was cleaned well.
    """
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
    ecg_channel: str = DEFAULT_ECG_CHANNEL,
    marker_description: str | None = DEFAULT_BEAT_MARKER_DESCRIPTION,
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
    events, is_fallback = pulse_marker_events(
        before_referenced, ecg_channel=ecg_channel, marker_description=marker_description
    )
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
    if after_rms <= 0:
        raise ValueError(f"{recording_id}: post-ICA marker-locked EEG RMS is zero.")
    return CardiacAttenuationMetrics(
        recording_id=recording_id,
        marker_count=len(events),
        before_rms_uv=before_rms * 1e6,
        after_rms_uv=after_rms * 1e6,
        attenuation_percent=attenuation_percent,
        attenuation_db=float(20.0 * np.log10(before_rms / after_rms)),
        is_fallback=is_fallback,
    )


def add_marker_ctps_columns(
    components: pd.DataFrame,
    scores: np.ndarray,
    *,
    threshold: float,
    is_fallback: bool = False,
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
    result["analyzer_marker_ctps_fallback"] = is_fallback
    return result


def compute_marker_ctps_scores(
    raws: Iterable[mne.io.BaseRaw],
    ica: mne.preprocessing.ICA,
    *,
    threshold: float,
    epoch_window: tuple[float, float],
    ecg_channel: str = DEFAULT_ECG_CHANNEL,
    marker_description: str | None = DEFAULT_BEAT_MARKER_DESCRIPTION,
) -> tuple[np.ndarray, bool]:
    """Score ICA components using CTPS epochs anchored to the beat markers."""
    marker_epochs = []
    any_fallback = False
    for raw in raws:
        events, is_fallback = pulse_marker_events(
            raw, ecg_channel=ecg_channel, marker_description=marker_description
        )
        if is_fallback:
            any_fallback = True
        marker_epochs.append(
            mne.Epochs(
                raw,
                events,
                event_id=PULSE_EVENT_ID,
                tmin=epoch_window[0],
                tmax=epoch_window[1],
                baseline=None,
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
        ch_name=ecg_channel,
        method="ctps",
        threshold=threshold,
        verbose="ERROR",
    )
    return np.asarray(scores, dtype=float), any_fallback


def write_cardiac_attenuation_qc(
    recordings: Iterable[tuple[str, mne.io.BaseRaw, mne.io.BaseRaw]],
    *,
    output_path: Path,
    baseline: tuple[float, float],
    measurement_window: tuple[float, float],
    ecg_channel: str = DEFAULT_ECG_CHANNEL,
    marker_description: str | None = DEFAULT_BEAT_MARKER_DESCRIPTION,
) -> Path:
    """Write run-level marker-locked EEG attenuation before versus after ICA.

    ``recordings`` is consumed lazily so a cohort run holds one before/after pair in
    memory at a time rather than every preloaded run of every subject at once.
    """
    rows = []
    for recording_id, before, after in recordings:
        metrics = compute_cardiac_attenuation(
            before,
            after,
            recording_id=recording_id,
            baseline=baseline,
            measurement_window=measurement_window,
            ecg_channel=ecg_channel,
            marker_description=marker_description,
        )
        rows.append(
            {
                "recording_id": metrics.recording_id,
                "marker_count": metrics.marker_count,
                "before_rms_uv": metrics.before_rms_uv,
                "after_rms_uv": metrics.after_rms_uv,
                "attenuation_percent": metrics.attenuation_percent,
                "attenuation_db": metrics.attenuation_db,
                "is_fallback": metrics.is_fallback,
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
    median_db = float(table["attenuation_db"].median())
    axis.text(
        0.98,
        0.02,
        f"median attenuation {median_db:.1f} dB",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    axis.set_title("Cardiac artifact attenuation after ICA")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _qc_output_path(pipeline_root: Path, task: str | None, description: str) -> Path:
    task_entity = f"task-{task}_" if task is not None else ""
    return pipeline_root / "qc" / f"{task_entity}{description}"


def run_marker_ctps_qc(
    *,
    pipeline_root: Path,
    subjects: list[str],
    task: str | None,
    threshold: float,
    epoch_window: tuple[float, float],
    ecg_channel: str = DEFAULT_ECG_CHANNEL,
    marker_description: str | None = DEFAULT_BEAT_MARKER_DESCRIPTION,
) -> Path:
    """Add beat-marker CTPS flags to native MNE-BIDS component tables.

    One decomposition per session, scored against the runs that fed it.
    """
    summary_frames = []
    for subject in resolve_subjects(pipeline_root, subjects):
        run_paths = find_filtered_raw_runs(pipeline_root, subject=subject, task=task)
        if not run_paths:
            raise FileNotFoundError(f"No filtered raw runs found for sub-{subject}, task={task!r}.")
        for ica_path in find_ica_solutions(pipeline_root, subject=subject):
            prefix = entity_prefix(ica_path, ICA_SUFFIX)
            session_runs = runs_for_prefix(run_paths, prefix)
            if not session_runs:
                continue
            components_path = components_path_for_ica(ica_path)
            if not components_path.is_file():
                raise FileNotFoundError(f"ICA component table does not exist: {components_path}")

            ica = mne.preprocessing.read_ica(ica_path, verbose="ERROR")
            raws = [
                mne.io.read_raw_fif(path, preload=True, verbose="ERROR") for path in session_runs
            ]
            scores, any_fallback = compute_marker_ctps_scores(
                raws,
                ica,
                threshold=threshold,
                epoch_window=epoch_window,
                ecg_channel=ecg_channel,
                marker_description=marker_description,
            )
            components = pd.read_csv(components_path, sep="\t")
            updated = add_marker_ctps_columns(
                components,
                scores,
                threshold=threshold,
                is_fallback=any_fallback,
            )
            updated.to_csv(components_path, sep="\t", index=False)

            summary = updated[
                [
                    "component",
                    "status",
                    "status_description",
                    "analyzer_marker_ctps_score",
                    "analyzer_marker_ctps_flag",
                    "analyzer_marker_ctps_fallback",
                ]
            ].copy()
            summary.insert(0, "participant_id", f"sub-{subject}")
            summary.insert(1, "decomposition_id", prefix)
            summary_frames.append(summary)

    if not summary_frames:
        raise FileNotFoundError(f"No ICA decomposition matched any filtered run for task={task!r}.")
    output_path = _qc_output_path(pipeline_root, task, "desc-markerctps_components.tsv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(summary_frames, ignore_index=True).to_csv(
        output_path,
        sep="\t",
        index=False,
    )
    return output_path


def _iter_before_after_runs(
    pipeline_root: Path,
    subjects: list[str],
    task: str | None,
) -> Iterator[tuple[str, mne.io.BaseRaw, mne.io.BaseRaw]]:
    """Yield one filtered/clean pair at a time so the cohort is never all in memory."""
    for subject in resolve_subjects(pipeline_root, subjects):
        filtered_paths = find_filtered_raw_runs(pipeline_root, subject=subject, task=task)
        if not filtered_paths:
            raise FileNotFoundError(f"No filtered raw runs found for sub-{subject}, task={task!r}.")
        for filtered_path in filtered_paths:
            clean_path = clean_raw_for_filtered(filtered_path)
            yield (
                entity_prefix(filtered_path, FILTERED_RAW_SUFFIX),
                mne.io.read_raw_fif(filtered_path, preload=True, verbose="ERROR"),
                mne.io.read_raw_fif(clean_path, preload=True, verbose="ERROR"),
            )


def run_cardiac_attenuation_qc(
    *,
    pipeline_root: Path,
    subjects: list[str],
    task: str | None,
    baseline: tuple[float, float],
    measurement_window: tuple[float, float],
    ecg_channel: str = DEFAULT_ECG_CHANNEL,
    marker_description: str | None = DEFAULT_BEAT_MARKER_DESCRIPTION,
) -> Path:
    """Pair filtered and clean runs and write marker-locked attenuation QC."""
    return write_cardiac_attenuation_qc(
        _iter_before_after_runs(pipeline_root, subjects, task),
        output_path=_qc_output_path(pipeline_root, task, "desc-cardiacattenuation_qc.tsv"),
        baseline=baseline,
        measurement_window=measurement_window,
        ecg_channel=ecg_channel,
        marker_description=marker_description,
    )


__all__ = [
    "DEFAULT_BEAT_MARKER_DESCRIPTION",
    "DEFAULT_ECG_CHANNEL",
    "CardiacAttenuationMetrics",
    "add_marker_ctps_columns",
    "compute_cardiac_attenuation",
    "compute_marker_ctps_scores",
    "pulse_marker_events",
    "run_cardiac_attenuation_qc",
    "run_marker_ctps_qc",
    "write_cardiac_attenuation_qc",
]
