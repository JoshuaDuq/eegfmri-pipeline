"""Validated time-resolved band-power summaries for Study 1."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from eeg_pipeline.utils.analysis.spectral import compute_frequency_weights
from eeg_pipeline.utils.analysis.tfr import compute_tfr_morlet
from eeg_pipeline.utils.config.loader import require_config_value
from eeg_pipeline.utils.data.epochs import load_epochs_for_analysis
from studies.pain_study.study1.cohort import load_primary_target_table
from studies.pain_study.study1.feature_benchmark import PRIMARY_BAND_PRESETS

SUBJECT_COLUMNS = (
    "subject_id",
    "band",
    "time_s",
    "power_db",
    "n_retained_trials",
    "n_channels",
    "included_channels",
)
COHORT_COLUMNS = (
    "band",
    "time_s",
    "mean_power_db",
    "ci_low",
    "ci_high",
    "n_subjects",
)
BOOTSTRAP_BATCH_SIZE = 256
FIGURE_CONFIG_KEY = "study1.figures.band_power_epoch_evolution"


@dataclass(frozen=True)
class BandPowerEpochSummary:
    """Participant trajectories and equally weighted cohort summaries."""

    subject_timecourses: pd.DataFrame
    cohort_timecourses: pd.DataFrame
    subject_ids: tuple[str, ...]
    bands: tuple[str, ...]
    times: np.ndarray


def load_band_power_epoch_summary(
    *,
    task: str,
    config: Any,
) -> BandPowerEpochSummary:
    """Load retained Study 1 epochs and compute publication timecourses."""

    task_name = str(task).strip()
    if not task_name:
        raise ValueError("Study 1 band-power timecourses require a task name.")
    figure_config = _figure_config(config)
    band_specs = _band_specs(figure_config)
    band_names = [_band_name(spec) for spec in band_specs]
    if band_names != PRIMARY_BAND_PRESETS["alpha_beta_gamma"]:
        raise ValueError("Study 1 band-power epoch bands must match alpha_beta_gamma exactly.")

    targets = load_primary_target_table(config).copy()
    _require_columns(
        targets,
        ("subject_id", "task", "run", "within_run_trial"),
    )
    targets = targets.loc[targets["task"].astype(str).eq(task_name)].copy()
    if targets.empty:
        raise ValueError(f"Study 1 primary target table contains no rows for task '{task_name}'.")
    subject_ids = tuple(sorted(targets["subject_id"].astype(str).unique().tolist()))
    logger = logging.getLogger(__name__)
    subject_frames = []
    stride = _positive_int(figure_config.get("temporal_stride"), "temporal_stride")
    baseline_window = _window(
        require_config_value(config, "time_frequency_analysis.baseline_window"),
        label="baseline",
    )
    display_window = _window(figure_config.get("display_window_s"), label="display")
    excluded_channels = _string_sequence(
        figure_config.get("excluded_channels"),
        label="excluded_channels",
    )

    for subject_id in subject_ids:
        epochs, events = load_epochs_for_analysis(
            subject=subject_id,
            task=task_name,
            align="strict",
            preload=False,
            config=config,
            logger=logger,
        )
        if epochs is None or events is None:
            raise FileNotFoundError(
                f"Study 1 band-power epochs or clean events are missing for {subject_id}."
            )
        subject_targets = targets.loc[targets["subject_id"].astype(str).eq(subject_id)]
        epoch_indices, retained_trial_ids = _retained_epoch_selection(subject_targets, events)
        retained_epochs = epochs[epoch_indices]
        tfr = compute_tfr_morlet(retained_epochs, config, logger=logger)
        if tfr is None:
            raise ValueError(f"Study 1 TFR computation returned no data for {subject_id}.")
        tfr_data = np.asarray(tfr.data)
        times = np.asarray(tfr.times, dtype=float)
        full_resolution = compute_subject_band_timecourses(
            subject_id=subject_id,
            tfr_data=tfr_data,
            times=times,
            frequencies=np.asarray(tfr.freqs, dtype=float),
            channel_names=tuple(str(channel) for channel in tfr.ch_names),
            trial_ids=retained_trial_ids,
            retained_trial_ids=retained_trial_ids,
            band_specs=band_specs,
            baseline_window=baseline_window,
            display_window=display_window,
            excluded_channels=excluded_channels,
        )
        subject_frames.append(_downsample_timecourses(full_resolution, stride=stride))

    bootstrap = require_config_value(config, "study1.figures.validity.bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise ValueError("study1.figures.validity.bootstrap must be a mapping.")
    return build_band_power_epoch_summary(
        pd.concat(subject_frames, ignore_index=True),
        band_specs=band_specs,
        bootstrap_iterations=_positive_int(bootstrap.get("iterations"), "bootstrap iterations"),
        confidence_level=float(bootstrap.get("confidence_level")),
        seed=int(bootstrap.get("seed")),
    )


def compute_subject_band_timecourses(
    *,
    subject_id: str,
    tfr_data: np.ndarray,
    times: np.ndarray,
    frequencies: np.ndarray,
    channel_names: Sequence[str],
    trial_ids: np.ndarray,
    retained_trial_ids: np.ndarray,
    band_specs: Sequence[Mapping[str, object]],
    baseline_window: tuple[float, float],
    display_window: tuple[float, float],
    excluded_channels: Sequence[str],
) -> pd.DataFrame:
    """Compute one retained-trial mean dB trajectory per participant and band."""

    subject = str(subject_id).strip()
    if not subject:
        raise ValueError("Study 1 band-power timecourses require a subject identifier.")
    power = _validated_tfr_data(tfr_data, times, frequencies, channel_names, trial_ids)
    retained_indices = _retained_trial_indices(trial_ids, retained_trial_ids)
    included_indices, included_names = _included_channel_indices(
        channel_names,
        excluded_channels,
    )
    baseline_mask = _time_mask(times, baseline_window, label="baseline")
    display_mask = _time_mask(times, display_window, label="display")
    display_times = np.asarray(times, dtype=float)[display_mask]

    records: list[pd.DataFrame] = []
    retained_power = power[retained_indices][:, included_indices, :, :]
    for band_spec in band_specs:
        band_name, frequency_mask = _band_frequency_mask(frequencies, band_spec)
        frequency_weights = compute_frequency_weights(
            np.asarray(frequencies, dtype=float)[frequency_mask]
        )
        band_power = np.average(
            retained_power[:, :, frequency_mask, :],
            axis=2,
            weights=frequency_weights,
        ).mean(axis=1)
        baseline_power = band_power[:, baseline_mask].mean(axis=1)
        if not np.isfinite(baseline_power).all() or np.any(baseline_power <= 0.0):
            raise ValueError(
                f"Study 1 baseline band power must be finite and positive for {subject}/{band_name}."
            )
        with np.errstate(divide="raise", invalid="raise"):
            trial_power_db = 10.0 * np.log10(band_power[:, display_mask] / baseline_power[:, None])
        participant_power_db = trial_power_db.mean(axis=0)
        if not np.isfinite(participant_power_db).all():
            raise ValueError(
                f"Study 1 band-power timecourse is non-finite for {subject}/{band_name}."
            )
        records.append(
            pd.DataFrame(
                {
                    "subject_id": subject,
                    "band": band_name,
                    "time_s": display_times,
                    "power_db": participant_power_db,
                    "n_retained_trials": len(retained_indices),
                    "n_channels": len(included_indices),
                    "included_channels": ",".join(included_names),
                }
            )
        )
    if not records:
        raise ValueError("Study 1 band-power timecourses require at least one configured band.")
    return pd.concat(records, ignore_index=True).loc[:, SUBJECT_COLUMNS]


def build_band_power_epoch_summary(
    subject_timecourses: pd.DataFrame,
    *,
    band_specs: Sequence[Mapping[str, object]],
    bootstrap_iterations: int,
    confidence_level: float,
    seed: int,
) -> BandPowerEpochSummary:
    """Aggregate complete participant trajectories with a paired bootstrap."""

    _require_columns(subject_timecourses, SUBJECT_COLUMNS)
    frame = subject_timecourses.loc[:, SUBJECT_COLUMNS].copy()
    frame["subject_id"] = frame["subject_id"].astype(str)
    frame["band"] = frame["band"].astype(str)
    if (frame["subject_id"].str.strip() == "").any():
        raise ValueError("Study 1 band-power subject identifiers must be non-empty.")
    for column in ("time_s", "power_db"):
        numeric = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError(f"Study 1 band-power column '{column}' must be finite.")
        frame[column] = numeric
    if frame.duplicated(["subject_id", "band", "time_s"]).any():
        raise ValueError(
            "Study 1 band-power trajectories contain duplicate subject/band/time rows."
        )

    bands = tuple(_band_name(spec) for spec in band_specs)
    observed_bands = tuple(frame["band"].drop_duplicates().tolist())
    if set(observed_bands) != set(bands):
        raise ValueError(
            "Study 1 band-power trajectories do not match configured bands: "
            f"observed={observed_bands}, configured={bands}."
        )
    subject_ids = tuple(sorted(frame["subject_id"].unique().tolist()))
    if len(subject_ids) < 2:
        raise ValueError("Study 1 cohort band-power summaries require at least two participants.")
    times = np.sort(frame["time_s"].unique())

    cohort_frames = []
    for band_index, band in enumerate(bands):
        band_frame = frame.loc[frame["band"].eq(band)]
        matrix = band_frame.pivot(
            index="subject_id",
            columns="time_s",
            values="power_db",
        ).reindex(index=subject_ids, columns=times)
        if matrix.isna().any().any():
            raise ValueError(
                f"Study 1 band-power trajectories must be complete for every participant: {band}."
            )
        mean, ci_low, ci_high = _paired_mean_bootstrap(
            matrix.to_numpy(dtype=float),
            iterations=bootstrap_iterations,
            confidence_level=confidence_level,
            seed=seed + band_index,
        )
        cohort_frames.append(
            pd.DataFrame(
                {
                    "band": band,
                    "time_s": times,
                    "mean_power_db": mean,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n_subjects": len(subject_ids),
                }
            )
        )
    cohort = pd.concat(cohort_frames, ignore_index=True).loc[:, COHORT_COLUMNS]
    return BandPowerEpochSummary(
        subject_timecourses=frame.sort_values(
            ["subject_id", "band", "time_s"],
            kind="stable",
        ).reset_index(drop=True),
        cohort_timecourses=cohort,
        subject_ids=subject_ids,
        bands=bands,
        times=times,
    )


def _paired_mean_bootstrap(
    values: np.ndarray,
    *,
    iterations: int,
    confidence_level: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or not np.isfinite(matrix).all():
        raise ValueError("Study 1 paired mean bootstrap requires a finite participant matrix.")
    if isinstance(iterations, bool) or int(iterations) != iterations or iterations < 1:
        raise ValueError("Study 1 band-power bootstrap iterations must be a positive integer.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("Study 1 band-power confidence level must be between zero and one.")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, matrix.shape[0], size=(int(iterations), matrix.shape[0]))
    estimates = np.empty((int(iterations), matrix.shape[1]), dtype=float)
    for start in range(0, int(iterations), BOOTSTRAP_BATCH_SIZE):
        stop = min(start + BOOTSTRAP_BATCH_SIZE, int(iterations))
        estimates[start:stop] = matrix[indices[start:stop]].mean(axis=1)
    tail = (1.0 - confidence_level) / 2.0
    return (
        matrix.mean(axis=0),
        np.quantile(estimates, tail, axis=0),
        np.quantile(estimates, 1.0 - tail, axis=0),
    )


def _retained_epoch_selection(
    targets: pd.DataFrame,
    events: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    _require_columns(targets, ("run", "within_run_trial"))
    _require_columns(events, ("run_id", "trial_number", "trial_id"))
    target_keys = pd.DataFrame(
        {
            "run": _integer_ids(targets["run"].to_numpy(), "target run"),
            "within_run_trial": _integer_ids(
                targets["within_run_trial"].to_numpy(),
                "target within-run trial",
            ),
            "_target_order": np.arange(len(targets)),
        }
    )
    event_keys = pd.DataFrame(
        {
            "run": _integer_ids(events["run_id"].to_numpy(), "clean-event run"),
            "within_run_trial": _integer_ids(
                events["trial_number"].to_numpy(),
                "clean-event within-run trial",
            ),
            "trial_id": _integer_ids(events["trial_id"].to_numpy(), "clean-event trial IDs"),
            "_epoch_index": np.arange(len(events)),
        }
    )
    key_columns = ["run", "within_run_trial"]
    if target_keys.duplicated(key_columns).any():
        raise ValueError("Study 1 target trials contain duplicate run/trial keys.")
    if event_keys.duplicated(key_columns).any():
        raise ValueError("Study 1 clean events contain duplicate run/trial keys.")
    merged = target_keys.merge(
        event_keys,
        on=key_columns,
        how="left",
        validate="one_to_one",
        indicator=True,
    ).sort_values("_target_order", kind="stable")
    if not merged["_merge"].eq("both").all():
        missing = merged.loc[merged["_merge"].ne("both"), key_columns]
        raise ValueError(
            "Study 1 retained target trials are absent from clean epochs:\n"
            f"{missing.to_string(index=False)}"
        )
    return (
        merged["_epoch_index"].to_numpy(dtype=int),
        merged["trial_id"].to_numpy(dtype=int),
    )


def _downsample_timecourses(frame: pd.DataFrame, *, stride: int) -> pd.DataFrame:
    times = np.sort(frame["time_s"].unique())
    retained_times = times[::stride]
    if retained_times[-1] != times[-1]:
        retained_times = np.append(retained_times, times[-1])
    return frame.loc[frame["time_s"].isin(retained_times)].reset_index(drop=True)


def _validated_tfr_data(
    tfr_data: np.ndarray,
    times: np.ndarray,
    frequencies: np.ndarray,
    channel_names: Sequence[str],
    trial_ids: np.ndarray,
) -> np.ndarray:
    power = np.asarray(tfr_data)
    if power.ndim != 4:
        raise ValueError("Study 1 band-power TFR data must have trial/channel/frequency/time axes.")
    if np.iscomplexobj(power) or not np.issubdtype(power.dtype, np.number):
        raise ValueError("Study 1 band-power TFR data must be real-valued.")
    power = power.astype(float, copy=False)
    expected_shape = (len(trial_ids), len(channel_names), len(frequencies), len(times))
    if power.shape != expected_shape:
        raise ValueError(
            f"Study 1 band-power TFR shape must be {expected_shape}; observed={power.shape}."
        )
    if not np.isfinite(power).all() or np.any(power <= 0.0):
        raise ValueError("Study 1 band-power TFR values must be finite and positive.")
    _finite_increasing_axis(times, "time")
    _finite_increasing_axis(frequencies, "frequency")
    if len(set(str(channel) for channel in channel_names)) != len(channel_names):
        raise ValueError("Study 1 band-power channel names must be unique.")
    _integer_ids(trial_ids, "TFR trial IDs")
    return power


def _retained_trial_indices(trial_ids: np.ndarray, retained_trial_ids: np.ndarray) -> np.ndarray:
    available = _integer_ids(trial_ids, "TFR trial IDs")
    retained = _integer_ids(retained_trial_ids, "retained trial IDs")
    if len(np.unique(available)) != len(available) or len(np.unique(retained)) != len(retained):
        raise ValueError("Study 1 band-power trial IDs must be unique.")
    lookup = {trial_id: index for index, trial_id in enumerate(available.tolist())}
    missing = [trial_id for trial_id in retained.tolist() if trial_id not in lookup]
    if missing:
        raise ValueError(f"Retained Study 1 trial IDs are absent from the TFR: {missing}.")
    if len(retained) == 0:
        raise ValueError("Study 1 band-power timecourses require retained trials.")
    return np.asarray([lookup[trial_id] for trial_id in retained.tolist()], dtype=int)


def _included_channel_indices(
    channel_names: Sequence[str],
    excluded_channels: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...]]:
    excluded = {str(channel) for channel in excluded_channels}
    included = tuple(str(channel) for channel in channel_names if str(channel) not in excluded)
    if not included:
        raise ValueError("Study 1 band-power channel exclusion removed every channel.")
    indices = np.asarray(
        [index for index, channel in enumerate(channel_names) if str(channel) in included],
        dtype=int,
    )
    return indices, included


def _band_frequency_mask(
    frequencies: np.ndarray,
    band_spec: Mapping[str, object],
) -> tuple[str, np.ndarray]:
    band = _band_name(band_spec)
    bounds = band_spec.get("frequency_hz")
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValueError(f"Study 1 band '{band}' requires two frequency bounds.")
    lower, upper = float(bounds[0]), float(bounds[1])
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError(f"Study 1 band '{band}' has invalid frequency bounds.")
    mask = (np.asarray(frequencies, dtype=float) >= lower) & (
        np.asarray(frequencies, dtype=float) <= upper
    )
    if not np.any(mask):
        raise ValueError(f"Study 1 TFR contains no frequencies for configured band '{band}'.")
    return band, mask


def _band_name(band_spec: Mapping[str, object]) -> str:
    name = str(band_spec.get("name", "")).strip()
    if not name:
        raise ValueError("Study 1 band specifications require non-empty names.")
    return name


def _figure_config(config: Any) -> Mapping[str, object]:
    value = require_config_value(config, FIGURE_CONFIG_KEY)
    if not isinstance(value, Mapping):
        raise ValueError(f"{FIGURE_CONFIG_KEY} must be a mapping.")
    return value


def _band_specs(figure_config: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    raw_specs = figure_config.get("bands")
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError(f"{FIGURE_CONFIG_KEY}.bands must be a non-empty list.")
    if not all(isinstance(spec, Mapping) for spec in raw_specs):
        raise ValueError(f"{FIGURE_CONFIG_KEY}.bands entries must be mappings.")
    specs = tuple(raw_specs)
    names = tuple(_band_name(spec) for spec in specs)
    if len(set(names)) != len(names):
        raise ValueError("Study 1 band-power band names must be unique.")
    return specs


def _window(value: object, *, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Study 1 band-power {label} window requires two values.")
    window = (float(value[0]), float(value[1]))
    if not np.isfinite(window).all() or window[0] >= window[1]:
        raise ValueError(f"Study 1 band-power {label} window must satisfy start < end.")
    return window


def _string_sequence(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"Study 1 band-power {label} must be a list.")
    parsed = tuple(str(item).strip() for item in value)
    if any(not item for item in parsed) or len(set(parsed)) != len(parsed):
        raise ValueError(f"Study 1 band-power {label} must contain unique names.")
    return parsed


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"Study 1 band-power {label} must be an integer.")
    numeric = float(value)
    if not np.isfinite(numeric) or not numeric.is_integer() or numeric < 1:
        raise ValueError(f"Study 1 band-power {label} must be a positive integer.")
    return int(numeric)


def _time_mask(
    times: np.ndarray,
    window: tuple[float, float],
    *,
    label: str,
) -> np.ndarray:
    start, end = (float(window[0]), float(window[1]))
    if not np.isfinite([start, end]).all() or start >= end:
        raise ValueError(f"Study 1 band-power {label} window must satisfy start < end.")
    mask = (np.asarray(times, dtype=float) >= start) & (np.asarray(times, dtype=float) <= end)
    if not np.any(mask):
        raise ValueError(f"Study 1 band-power {label} window contains no TFR samples.")
    return mask


def _finite_increasing_axis(values: np.ndarray, label: str) -> None:
    axis = np.asarray(values, dtype=float)
    if axis.ndim != 1 or len(axis) == 0 or not np.isfinite(axis).all():
        raise ValueError(f"Study 1 band-power {label} axis must be finite and one-dimensional.")
    if np.any(np.diff(axis) <= 0.0):
        raise ValueError(f"Study 1 band-power {label} axis must be strictly increasing.")


def _integer_ids(values: np.ndarray, label: str) -> np.ndarray:
    numeric = np.asarray(values, dtype=float)
    if (
        numeric.ndim != 1
        or not np.isfinite(numeric).all()
        or not np.allclose(
            numeric,
            np.round(numeric),
        )
    ):
        raise ValueError(f"Study 1 {label} must be finite integers.")
    return np.round(numeric).astype(int)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Study 1 band-power timecourses are missing columns: {missing}.")


__all__ = [
    "BandPowerEpochSummary",
    "build_band_power_epoch_summary",
    "compute_subject_band_timecourses",
    "load_band_power_epoch_summary",
]
