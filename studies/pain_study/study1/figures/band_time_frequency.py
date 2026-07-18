"""Nuisance-adjusted temperature-slope TFRs from final-clean Study 1 epochs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
import re
from typing import Any

import mne
import numpy as np
import pandas as pd
from scipy import signal

from eeg_pipeline.utils.config.loader import require_config_value
from studies.pain_study.study1.feature_benchmark import PRIMARY_BAND_PRESETS
from studies.pain_study.study1.figures.band_time_frequency_model import (
    TemperatureModel,
    TemperatureSlopeBatch,
    build_temperature_model,
    compute_variable_cycle_hanning_power,
    summarize_temperature_slope_batch,
)

FIGURE_CONFIG_KEY = "study1.figures.band_time_frequency"
BAND_CONFIG_KEY = "study1.figures.band_power_epoch_evolution.bands"

SUBJECT_MAP_COLUMNS = (
    "subject_id",
    "band",
    "frequency_hz",
    "time_s",
    "temperature_slope_db_per_c",
)
COHORT_MAP_COLUMNS = (
    "band",
    "frequency_hz",
    "time_s",
    "mean_temperature_slope_db_per_c",
    "n_subjects",
)
SOURCE_AUDIT_COLUMNS = (
    "subject_id",
    "source_file",
    "event_file",
    "modified_time_ns",
    "modified_time_utc",
    "n_clean_trials",
    "n_model_trials",
    "n_excluded_missing_metadata",
    "design_rank",
    "design_columns",
    "design_condition_number",
    "n_eeg_channels",
    "included_channels",
    "source_sampling_frequency_hz",
    "analysis_sampling_frequency_hz",
)


@dataclass(frozen=True)
class CleanEpochSource:
    """Newest final-clean epoch artifact selected for one participant."""

    subject_id: str
    path: Path
    modified_time_ns: int


@dataclass(frozen=True)
class SubjectSlopeMap:
    """One participant's global temperature-slope map and model audit."""

    slope: np.ndarray
    times: np.ndarray
    frequencies: np.ndarray
    channel_names: tuple[str, ...]
    analysis_sampling_frequency_hz: float
    model: TemperatureModel
    event_path: Path


@dataclass(frozen=True)
class BandTfrSummary:
    """Participant temperature slopes, cohort means, and exact provenance."""

    subject_maps: pd.DataFrame
    cohort_maps: pd.DataFrame
    source_audit: pd.DataFrame
    subject_ids: tuple[str, ...]
    bands: tuple[str, ...]


def load_band_tfr_summary(
    *,
    task: str,
    derivative_root: Path,
    config: Any,
) -> BandTfrSummary:
    """Build temperature-slope TFRs from each participant's newest clean epochs."""

    figure_config = _mapping(
        require_config_value(config, FIGURE_CONFIG_KEY),
        FIGURE_CONFIG_KEY,
    )
    band_specs = _band_specs(config)
    frequency_step_hz = _positive_float(
        figure_config.get("frequency_step_hz"),
        "frequency_step_hz",
    )
    frequencies = _configured_frequencies(band_specs, step_hz=frequency_step_hz)
    n_cycles = _positive_float(figure_config.get("n_cycles"), "n_cycles")
    time_step_s = _positive_float(figure_config.get("time_step_s"), "time_step_s")
    analysis_sampling_frequency_hz = _positive_float(
        figure_config.get("analysis_sampling_frequency_hz"),
        "analysis_sampling_frequency_hz",
    )
    trial_batch_size = _positive_int(
        figure_config.get("trial_batch_size"),
        "trial_batch_size",
    )
    baseline_window = _window(
        require_config_value(config, "time_frequency_analysis.baseline_window"),
        label="baseline",
    )
    display_window = _window(figure_config.get("display_window_s"), label="display")
    temperatures = _temperatures(config)
    sources = discover_latest_clean_epochs(
        derivative_root,
        task=task,
        excluded_subjects=_string_sequence(
            figure_config.get("excluded_subjects"),
            label="excluded_subjects",
        ),
    )

    subject_frames = []
    audit_rows = []
    for source in sources:
        epochs = mne.read_epochs(source.path, preload=False, verbose=False)
        if len(epochs) < 1:
            raise ValueError(f"Study 1 temperature TFR source contains no epochs: {source.path}")
        event_path, events = _load_matching_events(source.path, n_epochs=len(epochs))
        model = build_temperature_model(
            events,
            n_epochs=len(epochs),
            temperatures=temperatures,
        )
        result = _compute_subject_slope(
            subject_id=source.subject_id,
            epochs=epochs,
            event_path=event_path,
            model=model,
            frequencies=frequencies,
            n_cycles=n_cycles,
            time_step_s=time_step_s,
            analysis_sampling_frequency_hz=analysis_sampling_frequency_hz,
            trial_batch_size=trial_batch_size,
            baseline_window=baseline_window,
            display_window=display_window,
        )
        subject_frames.append(
            _subject_map_frame(
                source.subject_id,
                result,
                band_specs=band_specs,
            )
        )
        audit_rows.append(_audit_row(source, epochs, result))
    return build_band_tfr_summary(
        pd.concat(subject_frames, ignore_index=True),
        pd.DataFrame(audit_rows, columns=SOURCE_AUDIT_COLUMNS),
        band_names=tuple(str(spec["name"]) for spec in band_specs),
    )


def _compute_subject_slope(
    *,
    subject_id: str,
    epochs: Any,
    event_path: Path,
    model: TemperatureModel,
    frequencies: np.ndarray,
    n_cycles: float,
    time_step_s: float,
    analysis_sampling_frequency_hz: float,
    trial_batch_size: int,
    baseline_window: tuple[float, float],
    display_window: tuple[float, float],
) -> SubjectSlopeMap:
    eeg_indices = mne.pick_types(epochs.info, meg=False, eeg=True, exclude=[])
    if len(eeg_indices) < 1:
        raise ValueError(f"Study 1 temperature TFR source has no EEG channels: {subject_id}.")
    channel_names = tuple(str(epochs.ch_names[index]) for index in eeg_indices)
    selected_epochs = epochs[model.epoch_indices]
    source_sampling_frequency_hz = float(epochs.info["sfreq"])

    slope: np.ndarray | None = None
    reference: TemperatureSlopeBatch | None = None
    for start in range(0, len(selected_epochs), trial_batch_size):
        stop = min(start + trial_batch_size, len(selected_epochs))
        data = selected_epochs[start:stop].get_data(picks=eeg_indices, copy=True)
        resampled = _resample(
            data,
            source_sampling_frequency_hz=source_sampling_frequency_hz,
            target_sampling_frequency_hz=analysis_sampling_frequency_hz,
        )
        power = compute_variable_cycle_hanning_power(
            resampled,
            sampling_frequency_hz=analysis_sampling_frequency_hz,
            epoch_start_s=float(epochs.times[0]),
            frequencies=frequencies,
            n_cycles=n_cycles,
            time_step_s=time_step_s,
            time_window=display_window,
        )
        batch = summarize_temperature_slope_batch(
            power=power.power,
            times=power.times,
            frequencies=power.frequencies,
            window_durations_s=power.window_durations_s,
            channel_names=channel_names,
            baseline_window=baseline_window,
            display_window=display_window,
            temperature_weights=model.temperature_weights[start:stop],
        )
        if reference is None:
            reference = batch
            slope = np.zeros_like(batch.slope_sum)
        else:
            _validate_matching_batch(reference, batch, subject_id=subject_id)
        slope += batch.slope_sum
    if reference is None or slope is None:
        raise RuntimeError(f"Study 1 temperature TFR batching failed for {subject_id}.")
    return SubjectSlopeMap(
        slope=slope,
        times=reference.times,
        frequencies=reference.frequencies,
        channel_names=reference.channel_names,
        analysis_sampling_frequency_hz=analysis_sampling_frequency_hz,
        model=model,
        event_path=event_path,
    )


def _resample(
    data: np.ndarray,
    *,
    source_sampling_frequency_hz: float,
    target_sampling_frequency_hz: float,
) -> np.ndarray:
    ratio = Fraction(target_sampling_frequency_hz / source_sampling_frequency_hz).limit_denominator(
        1000
    )
    achieved = source_sampling_frequency_hz * ratio.numerator / ratio.denominator
    if not np.isclose(achieved, target_sampling_frequency_hz, rtol=0.0, atol=1e-9):
        raise ValueError("Study 1 temperature TFR sampling-rate ratio is not exact.")
    return signal.resample_poly(data, ratio.numerator, ratio.denominator, axis=-1)


def _subject_map_frame(
    subject_id: str,
    result: SubjectSlopeMap,
    *,
    band_specs: Sequence[Mapping[str, object]],
) -> pd.DataFrame:
    frames = []
    for band_spec in band_specs:
        band = str(band_spec["name"])
        indices = _frequency_indices(
            result.frequencies,
            band_spec["frequency_hz"],
            label=band,
        )
        frequency_grid, time_grid = np.meshgrid(
            result.frequencies[indices],
            result.times,
            indexing="ij",
        )
        frames.append(
            pd.DataFrame(
                {
                    "subject_id": subject_id,
                    "band": band,
                    "frequency_hz": frequency_grid.ravel(),
                    "time_s": time_grid.ravel(),
                    "temperature_slope_db_per_c": result.slope[indices].ravel(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True).loc[:, SUBJECT_MAP_COLUMNS]


def _audit_row(source: CleanEpochSource, epochs: Any, result: SubjectSlopeMap) -> dict[str, object]:
    model = result.model
    return {
        "subject_id": source.subject_id,
        "source_file": str(source.path),
        "event_file": str(result.event_path),
        "modified_time_ns": source.modified_time_ns,
        "modified_time_utc": datetime.fromtimestamp(
            source.modified_time_ns / 1e9,
            tz=UTC,
        ).isoformat(),
        "n_clean_trials": model.n_clean_epochs,
        "n_model_trials": model.n_model_trials,
        "n_excluded_missing_metadata": len(model.excluded_epoch_indices),
        "design_rank": model.design_rank,
        "design_columns": model.design_columns,
        "design_condition_number": model.condition_number,
        "n_eeg_channels": len(result.channel_names),
        "included_channels": ",".join(sorted(result.channel_names)),
        "source_sampling_frequency_hz": float(epochs.info["sfreq"]),
        "analysis_sampling_frequency_hz": result.analysis_sampling_frequency_hz,
    }


def build_band_tfr_summary(
    subject_maps: pd.DataFrame,
    source_audit: pd.DataFrame,
    *,
    band_names: Sequence[str],
) -> BandTfrSummary:
    """Validate complete participant slope maps and compute equal-weight means."""

    _require_columns(subject_maps, SUBJECT_MAP_COLUMNS, label="participant maps")
    _require_columns(source_audit, SOURCE_AUDIT_COLUMNS, label="source audit")
    maps = subject_maps.loc[:, SUBJECT_MAP_COLUMNS].copy()
    audit = source_audit.loc[:, SOURCE_AUDIT_COLUMNS].copy()
    maps["subject_id"] = maps["subject_id"].astype(str)
    maps["band"] = maps["band"].astype(str)
    key = ["subject_id", "band", "frequency_hz", "time_s"]
    if maps.duplicated(key).any():
        raise ValueError("Study 1 temperature TFR maps contain duplicate coordinates.")
    numeric_columns = ("frequency_hz", "time_s", "temperature_slope_db_per_c")
    numeric = maps.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Study 1 temperature TFR participant maps must be finite.")
    maps.loc[:, numeric_columns] = numeric

    subjects = tuple(sorted(maps["subject_id"].unique()))
    bands = tuple(str(band).strip() for band in band_names)
    if len(subjects) < 2 or not bands or set(maps["band"].unique()) != set(bands):
        raise ValueError("Study 1 temperature TFR requires complete subjects and bands.")
    audit["subject_id"] = audit["subject_id"].astype(str)
    if audit["subject_id"].duplicated().any() or set(audit["subject_id"]) != set(subjects):
        raise ValueError(
            "Study 1 temperature TFR source audit must contain every participant once."
        )
    _validate_audit(audit)

    cohort_frames = []
    for band in bands:
        matrix = (
            maps.loc[maps["band"].eq(band)]
            .pivot(
                index="subject_id",
                columns=["frequency_hz", "time_s"],
                values="temperature_slope_db_per_c",
            )
            .sort_index(axis=1)
            .reindex(subjects)
        )
        if matrix.isna().any().any():
            raise ValueError(f"Study 1 temperature TFR requires a complete grid for {band}.")
        cohort_frames.append(
            pd.DataFrame(
                {
                    "band": band,
                    "frequency_hz": matrix.columns.get_level_values("frequency_hz"),
                    "time_s": matrix.columns.get_level_values("time_s"),
                    "mean_temperature_slope_db_per_c": matrix.mean(axis=0).to_numpy(),
                    "n_subjects": len(subjects),
                }
            )
        )
    return BandTfrSummary(
        subject_maps=maps.sort_values(key, kind="stable").reset_index(drop=True),
        cohort_maps=pd.concat(cohort_frames, ignore_index=True).loc[:, COHORT_MAP_COLUMNS],
        source_audit=audit.reset_index(drop=True),
        subject_ids=subjects,
        bands=bands,
    )


def discover_latest_clean_epochs(
    derivative_root: Path,
    *,
    task: str,
    excluded_subjects: Sequence[str] = (),
) -> tuple[CleanEpochSource, ...]:
    """Select the uniquely newest final-clean epoch file for each participant."""

    root = Path(derivative_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Study 1 EEG derivative root does not exist: {root}")
    task_name = str(task).strip()
    if not task_name:
        raise ValueError("Study 1 temperature TFR discovery requires a task name.")
    excluded = {str(subject).strip() for subject in excluded_subjects}
    candidates_by_subject: dict[str, list[Path]] = {}
    for candidate in root.rglob(f"sub-*_task-{task_name}*_epo.fif"):
        if (
            not candidate.is_file()
            or _is_hidden(candidate, root)
            or not _is_final_clean_epoch(candidate)
        ):
            continue
        subject_id = _subject_id(candidate)
        if subject_id is not None and subject_id not in excluded:
            candidates_by_subject.setdefault(subject_id, []).append(candidate)
    if not candidates_by_subject:
        raise FileNotFoundError(
            f"No final-clean Study 1 epochs found for task {task_name!r} under {root}."
        )

    sources = []
    for subject_id, candidates in sorted(candidates_by_subject.items()):
        modified_times = {candidate: candidate.stat().st_mtime_ns for candidate in candidates}
        latest_time = max(modified_times.values())
        latest = [path for path, modified in modified_times.items() if modified == latest_time]
        if len(latest) != 1:
            paths = ", ".join(str(path) for path in sorted(latest))
            raise ValueError(f"Ambiguous newest final-clean epochs for {subject_id}: {paths}.")
        sources.append(CleanEpochSource(subject_id, latest[0], latest_time))
    return tuple(sources)


def _load_matching_events(epoch_path: Path, *, n_epochs: int) -> tuple[Path, pd.DataFrame]:
    event_path = epoch_path.with_name(epoch_path.name.replace("_epo.fif", "_events.tsv"))
    if event_path == epoch_path or not event_path.is_file():
        raise FileNotFoundError(f"Study 1 temperature TFR event table is missing: {event_path}")
    events = pd.read_csv(event_path, sep="\t")
    if len(events) != n_epochs:
        raise ValueError("Study 1 temperature TFR event table does not match its epoch file.")
    return event_path, events


def _validate_matching_batch(
    reference: TemperatureSlopeBatch,
    candidate: TemperatureSlopeBatch,
    *,
    subject_id: str,
) -> None:
    if (
        not np.array_equal(reference.times, candidate.times)
        or not np.array_equal(reference.frequencies, candidate.frequencies)
        or reference.channel_names != candidate.channel_names
    ):
        raise ValueError(
            f"Study 1 temperature TFR batches use inconsistent coordinates for {subject_id}."
        )


def _validate_audit(audit: pd.DataFrame) -> None:
    numeric_columns = (
        "n_clean_trials",
        "n_model_trials",
        "n_excluded_missing_metadata",
        "design_rank",
        "design_columns",
        "design_condition_number",
        "n_eeg_channels",
        "source_sampling_frequency_hz",
        "analysis_sampling_frequency_hz",
    )
    numeric = audit.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Study 1 temperature TFR source audit must be finite.")
    if not np.array_equal(
        numeric["n_clean_trials"].to_numpy(dtype=int),
        numeric["n_model_trials"].to_numpy(dtype=int)
        + numeric["n_excluded_missing_metadata"].to_numpy(dtype=int),
    ):
        raise ValueError("Study 1 temperature TFR trial audit does not reconcile.")
    channel_counts = audit["included_channels"].astype(str).map(_channel_count).to_numpy()
    if not np.array_equal(channel_counts, numeric["n_eeg_channels"].to_numpy(dtype=int)):
        raise ValueError("Study 1 temperature TFR channel audit is inconsistent.")


def _configured_frequencies(
    band_specs: Sequence[Mapping[str, object]],
    *,
    step_hz: float,
) -> np.ndarray:
    frequencies = []
    for specification in band_specs:
        lower, upper = _window(specification["frequency_hz"], label="frequency")
        first = np.ceil(lower / step_hz) * step_hz
        values = np.arange(first, upper + step_hz / 2.0, step_hz)
        if len(values) < 2 or values[-1] > upper + 1e-9:
            values = values[values <= upper + 1e-9]
        if len(values) < 2:
            raise ValueError("Study 1 temperature TFR bands require at least two frequencies.")
        frequencies.extend(values.tolist())
    result = np.unique(np.asarray(frequencies, dtype=float))
    if np.any(np.diff(result) <= 0.0):
        raise ValueError("Study 1 temperature TFR frequencies must be increasing.")
    return result


def _frequency_indices(
    frequencies: np.ndarray,
    bounds: object,
    *,
    label: str,
) -> np.ndarray:
    lower, upper = _window(bounds, label=f"{label} frequency")
    indices = np.flatnonzero((frequencies >= lower) & (frequencies <= upper))
    if len(indices) < 2:
        raise ValueError(f"Study 1 temperature TFR has fewer than two bins in {label}.")
    return indices


def _band_specs(config: Any) -> tuple[Mapping[str, object], ...]:
    configured = require_config_value(config, BAND_CONFIG_KEY)
    if not isinstance(configured, list) or not all(
        isinstance(spec, Mapping) for spec in configured
    ):
        raise ValueError(f"{BAND_CONFIG_KEY} must be a list of mappings.")
    names = [str(spec.get("name", "")).strip() for spec in configured]
    if names != PRIMARY_BAND_PRESETS["alpha_beta_gamma"]:
        raise ValueError("Study 1 temperature TFR bands must match alpha_beta_gamma exactly.")
    return tuple(configured)


def _temperatures(config: Any) -> tuple[float, ...]:
    values = require_config_value(config, "study1.figures.validity.temperatures")
    temperatures = tuple(float(value) for value in values)
    if len(temperatures) < 2 or not np.isfinite(temperatures).all():
        raise ValueError("Study 1 temperature TFR temperatures must be finite.")
    return temperatures


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return value


def _window(value: object, *, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Study 1 temperature TFR {label} window requires two values.")
    start, end = (float(item) for item in value)
    if not np.isfinite((start, end)).all() or start >= end:
        raise ValueError(f"Study 1 temperature TFR {label} window must be finite and increasing.")
    return start, end


def _positive_float(value: object, label: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"Study 1 temperature TFR {label} must be positive and finite.")
    return parsed


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Study 1 temperature TFR {label} must be a positive integer.")
    parsed = int(value)
    if parsed != value or parsed < 1:
        raise ValueError(f"Study 1 temperature TFR {label} must be a positive integer.")
    return parsed


def _string_sequence(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Study 1 temperature TFR {label} must be a sequence.")
    values = tuple(str(item).strip() for item in value)
    if any(not item for item in values) or len(set(values)) != len(values):
        raise ValueError(f"Study 1 temperature TFR {label} requires unique non-empty values.")
    return values


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, label: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"Study 1 temperature TFR {label} missing columns: {missing}.")


def _channel_count(value: str) -> int:
    channels = tuple(channel.strip() for channel in value.split(","))
    return len(channels) if all(channels) and len(set(channels)) == len(channels) else 0


def _is_hidden(path: Path, root: Path) -> bool:
    return any(part.startswith(".") for part in path.relative_to(root).parts)


def _is_final_clean_epoch(path: Path) -> bool:
    return path.name.endswith(("_proc-clean_epo.fif", "_proc-cleaned_epo.fif", "_clean_epo.fif"))


def _subject_id(path: Path) -> str | None:
    match = re.match(r"(sub-\d{4})_", path.name)
    return match.group(1) if match is not None else None


__all__ = [
    "BandTfrSummary",
    "CleanEpochSource",
    "SubjectSlopeMap",
    "build_band_tfr_summary",
    "discover_latest_clean_epochs",
    "load_band_tfr_summary",
]
