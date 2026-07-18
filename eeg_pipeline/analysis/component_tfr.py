"""ICA-component time-frequency analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mne
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ComponentTFRParameters:
    """Validated parameters for component time-frequency analysis."""

    condition_column: str
    ica_band: tuple[float, float]
    ica_fit_window: tuple[float, float]
    ica_max_iterations: int
    random_state: int
    frequency_range: tuple[float, float]
    frequency_step: float
    window_duration: float
    frequency_smoothing: float
    time_step: float
    baseline_window: tuple[float, float]
    minimum_epochs_per_condition: int
    n_jobs: int
    components_per_page: int

    @classmethod
    def from_config(cls, config: Any) -> "ComponentTFRParameters":
        """Build parameters from the required YAML configuration."""
        prefix = "component_time_frequency"
        condition_column = _required_string(config, f"{prefix}.condition_column")
        ica_band = _required_interval(config, f"{prefix}.ica.band", positive=True)
        ica_fit_window = _required_interval(config, f"{prefix}.ica.fit_window")
        frequency_range = _required_interval(
            config,
            f"{prefix}.tfr.frequency_range",
            positive=True,
        )
        baseline_window = _required_interval(config, f"{prefix}.tfr.baseline_window")

        baseline_mode = _required_string(config, f"{prefix}.tfr.baseline_mode")
        if baseline_mode != "logratio":
            raise ValueError(
                f"{prefix}.tfr.baseline_mode must be 'logratio', got {baseline_mode!r}."
            )

        return cls(
            condition_column=condition_column,
            ica_band=ica_band,
            ica_fit_window=ica_fit_window,
            ica_max_iterations=_required_positive_int(
                config,
                f"{prefix}.ica.max_iterations",
            ),
            random_state=_required_int(config, f"{prefix}.ica.random_state"),
            frequency_range=frequency_range,
            frequency_step=_required_positive_float(
                config,
                f"{prefix}.tfr.frequency_step",
            ),
            window_duration=_required_positive_float(
                config,
                f"{prefix}.tfr.window_duration",
            ),
            frequency_smoothing=_required_positive_float(
                config,
                f"{prefix}.tfr.frequency_smoothing",
            ),
            time_step=_required_positive_float(config, f"{prefix}.tfr.time_step"),
            baseline_window=baseline_window,
            minimum_epochs_per_condition=_required_positive_int(
                config,
                f"{prefix}.minimum_epochs_per_condition",
            ),
            n_jobs=_required_nonzero_int(config, f"{prefix}.n_jobs"),
            components_per_page=_required_positive_int(
                config,
                f"{prefix}.plotting.components_per_page",
            ),
        )

    @property
    def frequencies(self) -> np.ndarray:
        """Return the inclusive TFR frequency grid."""
        lower, upper = self.frequency_range
        frequencies = np.arange(
            lower,
            upper + self.frequency_step / 2.0,
            self.frequency_step,
            dtype=float,
        )
        if not np.isclose(frequencies[-1], upper):
            raise ValueError(
                "component_time_frequency.tfr.frequency_step must divide the configured "
                "frequency range exactly."
            )
        return frequencies

    @property
    def time_bandwidth(self) -> float:
        """Return the DPSS time-bandwidth product matching the requested smoothing."""
        return 2.0 * self.window_duration * self.frequency_smoothing


@dataclass(frozen=True)
class ConditionTFR:
    """Raw and baseline-normalized component power for one condition."""

    value: Any
    label: str
    file_label: str
    epoch_count: int
    raw_power: mne.time_frequency.AverageTFR
    baseline_power: mne.time_frequency.AverageTFR


def prepare_component_epochs_input(
    epochs: mne.Epochs,
    events: pd.DataFrame,
    parameters: ComponentTFRParameters,
) -> mne.Epochs:
    """Validate and align broadband epochs with their condition metadata."""
    if len(epochs) == 0:
        raise ValueError("Component TFR analysis requires at least one retained epoch.")
    if len(events) != len(epochs):
        raise ValueError(
            "Clean events must align one-to-one with clean epochs: "
            f"events={len(events)}, epochs={len(epochs)}."
        )
    if parameters.condition_column not in events.columns:
        raise ValueError(
            f"Clean events are missing condition column {parameters.condition_column!r}. "
            f"Available columns: {list(events.columns)}"
        )

    condition_values = events[parameters.condition_column]
    missing_rows = condition_values.index[condition_values.isna()].tolist()
    if missing_rows:
        raise ValueError(
            f"Condition column {parameters.condition_column!r} contains missing values "
            f"at clean-event rows {missing_rows}."
        )
    _validate_condition_values(condition_values, parameters.condition_column)

    if epochs.tmin > parameters.baseline_window[0] or epochs.tmax < parameters.baseline_window[1]:
        raise ValueError(
            f"Epoch interval [{epochs.tmin}, {epochs.tmax}] does not contain baseline "
            f"window {parameters.baseline_window}."
        )
    if epochs.tmin > parameters.ica_fit_window[0] or epochs.tmax < parameters.ica_fit_window[1]:
        raise ValueError(
            f"Epoch interval [{epochs.tmin}, {epochs.tmax}] does not contain ICA fit "
            f"window {parameters.ica_fit_window}."
        )

    eeg_picks = mne.pick_types(epochs.info, eeg=True, exclude="bads")
    if len(eeg_picks) < 2:
        raise ValueError(
            "Component TFR analysis requires at least two non-bad EEG channels, "
            f"found {len(eeg_picks)}."
        )

    broadband_epochs = epochs.copy().pick(eeg_picks).load_data()
    broadband_epochs.metadata = pd.DataFrame(
        {parameters.condition_column: condition_values.to_numpy(copy=True)}
    )
    _require_finite_data(broadband_epochs, "Broadband clean epochs")
    _validate_sampling_rate(broadband_epochs.info["sfreq"], parameters)
    return broadband_epochs


def fit_band_limited_ica(
    broadband_epochs: mne.Epochs,
    parameters: ComponentTFRParameters,
) -> mne.preprocessing.ICA:
    """Fit rank-aware extended Infomax on the pooled 6-14 Hz active window."""
    fit_epochs = broadband_epochs.copy()
    fit_epochs.filter(
        l_freq=parameters.ica_band[0],
        h_freq=parameters.ica_band[1],
        picks="eeg",
        n_jobs=parameters.n_jobs,
        verbose=False,
    )
    fit_epochs.baseline = None
    fit_epochs.crop(
        tmin=parameters.ica_fit_window[0],
        tmax=parameters.ica_fit_window[1],
        include_tmax=True,
    )

    ica = mne.preprocessing.ICA(
        n_components=None,
        method="infomax",
        fit_params={"extended": True},
        max_iter=parameters.ica_max_iterations,
        random_state=parameters.random_state,
    )
    ica.fit(fit_epochs, picks="eeg", reject_by_annotation=True, verbose=False)
    if ica.n_components_ < 2:
        raise ValueError(f"ICA produced fewer than two components: {ica.n_components_}.")
    if not np.isfinite(ica.unmixing_matrix_).all() or not np.isfinite(ica.mixing_matrix_).all():
        raise ValueError("ICA produced non-finite mixing or unmixing weights.")
    return ica


def apply_unmixing_to_broadband_epochs(
    ica: mne.preprocessing.ICA,
    broadband_epochs: mne.Epochs,
) -> mne.Epochs:
    """Apply fitted unmixing weights to the unfiltered analysis epochs."""
    component_epochs = ica.get_sources(broadband_epochs)
    component_epochs.metadata = broadband_epochs.metadata.copy()
    _require_finite_data(component_epochs, "Broadband ICA component epochs")
    return component_epochs


def compute_condition_tfrs(
    component_epochs: mne.Epochs,
    parameters: ComponentTFRParameters,
) -> list[ConditionTFR]:
    """Compute raw and log-ratio component power separately for every condition."""
    if component_epochs.metadata is None:
        raise ValueError("Component epochs are missing condition metadata.")

    condition_series = component_epochs.metadata[parameters.condition_column]
    conditions = _sorted_conditions(condition_series)
    file_labels = [_condition_file_label(value) for value in conditions]
    if len(file_labels) != len(set(file_labels)):
        raise ValueError(
            f"Condition values in {parameters.condition_column!r} collide after filename "
            f"sanitization: {conditions}."
        )

    decimation = _time_decimation(component_epochs.info["sfreq"], parameters.time_step)
    frequencies = parameters.frequencies
    n_cycles = frequencies * parameters.window_duration
    results = []
    for value, file_label in zip(conditions, file_labels):
        indices = np.flatnonzero(condition_series.to_numpy() == value)
        epoch_count = len(indices)
        if epoch_count < parameters.minimum_epochs_per_condition:
            raise ValueError(
                f"Condition {parameters.condition_column}={value!r} has {epoch_count} epochs; "
                "component_time_frequency.minimum_epochs_per_condition requires at least "
                f"{parameters.minimum_epochs_per_condition}."
            )

        raw_power = component_epochs[indices].compute_tfr(
            method="multitaper",
            freqs=frequencies,
            picks="all",
            output="power",
            average=True,
            return_itc=False,
            decim=decimation,
            n_jobs=parameters.n_jobs,
            n_cycles=n_cycles,
            time_bandwidth=parameters.time_bandwidth,
            use_fft=True,
            zero_mean=True,
            verbose=False,
        )
        baseline_power = raw_power.copy().apply_baseline(
            baseline=parameters.baseline_window,
            mode="logratio",
            verbose=False,
        )
        if not np.isfinite(raw_power.get_data()).all():
            raise ValueError(
                f"Raw component TFR contains non-finite values for condition {value!r}."
            )
        if not np.isfinite(baseline_power.get_data()).all():
            raise ValueError(
                f"Baseline-normalized component TFR contains non-finite values for "
                f"condition {value!r}."
            )
        results.append(
            ConditionTFR(
                value=value,
                label=str(value),
                file_label=file_label,
                epoch_count=epoch_count,
                raw_power=raw_power,
                baseline_power=baseline_power,
            )
        )
    return results


def _require_finite_data(epochs: mne.Epochs, description: str) -> None:
    if not np.isfinite(epochs.get_data(copy=False)).all():
        raise ValueError(f"{description} contain non-finite values.")


def _validate_sampling_rate(sampling_rate: float, parameters: ComponentTFRParameters) -> None:
    nyquist = float(sampling_rate) / 2.0
    required_frequency = max(parameters.ica_band[1], parameters.frequency_range[1])
    if required_frequency >= nyquist:
        raise ValueError(
            f"Highest requested frequency {required_frequency} Hz must be below Nyquist "
            f"({nyquist} Hz)."
        )
    _time_decimation(sampling_rate, parameters.time_step)


def _time_decimation(sampling_rate: float, time_step: float) -> int:
    samples_per_step = float(sampling_rate) * time_step
    decimation = int(round(samples_per_step))
    if decimation < 1 or not np.isclose(samples_per_step, decimation, rtol=0.0, atol=1e-9):
        raise ValueError(
            "component_time_frequency.tfr.time_step must correspond to an integer number "
            f"of samples; sfreq={sampling_rate}, time_step={time_step}."
        )
    return decimation


def _sorted_conditions(series: pd.Series) -> list[Any]:
    values = series.drop_duplicates().tolist()
    if pd.api.types.is_numeric_dtype(series.dtype):
        return sorted(values, key=float)
    return sorted(values, key=str)


def _validate_condition_values(series: pd.Series, column: str) -> None:
    if pd.api.types.is_numeric_dtype(series.dtype):
        numeric = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError(f"Condition column {column!r} contains non-finite numeric values.")
        return

    invalid_types = sorted({type(value).__name__ for value in series if not isinstance(value, str)})
    if invalid_types:
        raise TypeError(
            f"Condition column {column!r} must contain only numeric values or only strings; "
            f"found non-string types {invalid_types}."
        )


def _condition_file_label(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        text = str(int(value))
    elif isinstance(value, (float, np.floating)):
        text = np.format_float_positional(float(value), trim="-")
    else:
        text = str(value).strip()

    sanitized = "".join(character if character.isalnum() else "-" for character in text)
    sanitized = "-".join(part for part in sanitized.split("-") if part)
    if not sanitized:
        raise ValueError(f"Condition value {value!r} cannot be converted to a filename label.")
    return sanitized


def _required_value(config: Any, key: str) -> Any:
    sentinel = object()
    value = config.get(key, sentinel)
    if value is sentinel:
        raise ValueError(f"Missing required config value: {key}")
    return value


def _required_string(config: Any, key: str) -> str:
    value = _required_value(config, key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string.")
    return value.strip()


def _required_float(config: Any, key: str) -> float:
    value = _required_value(config, key)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a finite number.") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{key} must be finite.")
    return parsed


def _required_positive_float(config: Any, key: str) -> float:
    value = _required_float(config, key)
    if value <= 0:
        raise ValueError(f"{key} must be > 0.")
    return value


def _required_int(config: Any, key: str) -> int:
    value = _required_value(config, key)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{key} must be an integer.")
    return parsed


def _required_positive_int(config: Any, key: str) -> int:
    value = _required_int(config, key)
    if value < 1:
        raise ValueError(f"{key} must be >= 1.")
    return value


def _required_nonzero_int(config: Any, key: str) -> int:
    value = _required_int(config, key)
    if value == 0 or value < -1:
        raise ValueError(f"{key} must be -1 or a positive integer.")
    return value


def _required_interval(
    config: Any,
    key: str,
    *,
    positive: bool = False,
) -> tuple[float, float]:
    value = _required_value(config, key)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{key} must contain exactly two numbers.")
    lower = _required_interval_number(value[0], key)
    upper = _required_interval_number(value[1], key)
    if lower >= upper:
        raise ValueError(f"{key} must be strictly increasing.")
    if positive and lower <= 0:
        raise ValueError(f"{key} values must be > 0.")
    return lower, upper


def _required_interval_number(value: Any, key: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must contain finite numbers.") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{key} must contain finite numbers.")
    return parsed
