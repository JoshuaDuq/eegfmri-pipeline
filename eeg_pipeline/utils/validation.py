"""
Data Validation Module
======================

Centralized validation for EEG data and analysis inputs.
Validates data quality before expensive computations.

Usage
-----
```python
from eeg_pipeline.utils.validation import (
    validate_epochs,
    validate_features,
    validate_targets,
    ValidationResult,
)

result = validate_epochs(epochs, config)
if not result.valid:
    logger.warning(f"Validation failed: {result.issues}")
```
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
import mne


###################################################################
# Constants
###################################################################

DEFAULT_MIN_EPOCHS = 20
DEFAULT_MIN_CHANNELS = 10
DEFAULT_SAMPLING_FREQ = 500
DEFAULT_MAX_AMPLITUDE_UV = 500
DEFAULT_MIN_ROWS = 10
DEFAULT_MAX_NAN_FRACTION = 0.1
DEFAULT_MIN_SAMPLES = 10
DEFAULT_PERCENT_THRESHOLD = 5.0
CRITICAL_NAN_FRACTION = 0.01
WARNING_EXTREME_FRACTION = 0.1
CRITICAL_NAN_FRACTION_CONNECTIVITY = 0.1
ZERO_VARIANCE_THRESHOLD = 1e-12
ZERO_VARIANCE_TARGET_THRESHOLD = 1e-10
SAMPLING_FREQ_TOLERANCE = 1.0
SMALL_EPOCH_COUNT = 50
MIN_UNIQUE_VALUES_WARNING = 3
VOLTS_TO_MICROVOLTS = 1e6


###################################################################
# Validation Result
###################################################################


@dataclass
class ValidationResult:
    """Result of a validation check."""

    valid: bool
    issues: list[str] = field(default_factory=list)
    critical: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.valid

    def log_issues(self, logger: logging.Logger) -> None:
        """Log all issues and warnings."""
        for issue in self.critical:
            logger.critical(f"Validation critical: {issue}")
        for issue in self.issues:
            logger.error(f"Validation error: {issue}")
        for warning in self.warnings:
            logger.warning(f"Validation warning: {warning}")

    @classmethod
    def success(cls, **metadata) -> ValidationResult:
        """Create a successful validation result."""
        return cls(valid=True, metadata=metadata)

    @classmethod
    def failure(cls, issues: list[str], **metadata) -> ValidationResult:
        """Create a failed validation result."""
        return cls(valid=False, issues=issues, metadata=metadata)


###################################################################
# Helper Functions
###################################################################


def _get_logger(logger: Optional[logging.Logger]) -> logging.Logger:
    """Get logger instance, creating one if needed."""
    return logger if logger is not None else logging.getLogger(__name__)


def _get_config_value(config: Any, key: str, default: Any) -> Any:
    """Extract value from config object with fallback."""
    if hasattr(config, "get"):
        return config.get(key, default)
    return default


def _check_nan_inf_in_data(data: np.ndarray) -> tuple[int, float]:
    """Count NaN/Inf values and return count and fraction."""
    nan_count = int(np.sum(~np.isfinite(data)))
    nan_fraction = nan_count / data.size if data.size > 0 else 0.0
    return nan_count, nan_fraction


def _check_extreme_amplitudes(data: np.ndarray, max_uv: float) -> float:
    """Calculate fraction of data exceeding amplitude threshold."""
    data_microvolts = data * VOLTS_TO_MICROVOLTS
    extreme_fraction = float(np.mean(np.abs(data_microvolts) > max_uv))
    return extreme_fraction


def _check_flat_channels(data: np.ndarray) -> int:
    """Count channels with zero variance."""
    channel_variances = np.var(data, axis=(0, 2))
    flat_count = int(np.sum(channel_variances < ZERO_VARIANCE_THRESHOLD))
    return flat_count


def _validate_epoch_count(n_epochs: int, min_epochs: int) -> Optional[str]:
    """Validate epoch count meets minimum requirement."""
    if n_epochs < min_epochs:
        return f"Insufficient epochs: {n_epochs} < {min_epochs}"
    return None


def _validate_sampling_rate(sfreq: float, expected_sfreq: float) -> Optional[str]:
    """Validate sampling rate matches expected value."""
    if abs(sfreq - expected_sfreq) > SAMPLING_FREQ_TOLERANCE:
        return f"Unexpected sampling rate: {sfreq} Hz (expected {expected_sfreq})"
    return None


def _validate_channel_count(n_channels: int, min_channels: int) -> Optional[str]:
    """Validate channel count meets minimum requirement."""
    if n_channels < min_channels:
        return f"Insufficient channels: {n_channels} < {min_channels}"
    return None


def _validate_epochs_data_quality(
    epochs: mne.Epochs, config: Any
) -> tuple[list[str], list[str]]:
    """Validate data quality for epochs (NaN, extremes, flat channels)."""
    issues = []
    warnings = []

    try:
        data = epochs.get_data()

        nan_count, nan_fraction = _check_nan_inf_in_data(data)
        if nan_count > 0:
            if nan_fraction > CRITICAL_NAN_FRACTION:
                issues.append(f"Data contains {nan_fraction:.1%} NaN/Inf values")
            else:
                warnings.append(
                    f"Data contains {nan_count} NaN/Inf values ({nan_fraction:.2%})"
                )

        max_uv = _get_config_value(
            config, "validation.max_amplitude_uv", DEFAULT_MAX_AMPLITUDE_UV
        )
        extreme_fraction = _check_extreme_amplitudes(data, max_uv)
        if extreme_fraction > WARNING_EXTREME_FRACTION:
            warnings.append(
                f"{extreme_fraction:.1%} of data exceeds {max_uv} µV"
            )

        flat_count = _check_flat_channels(data)
        if flat_count > 0:
            warnings.append(f"{flat_count} channels appear flat (zero variance)")

    except Exception as e:
        warnings.append(f"Could not check data quality: {e}")

    return issues, warnings


###################################################################
# Epochs Validation
###################################################################


def validate_epochs(
    epochs: mne.Epochs,
    config: Any,
    logger: Optional[logging.Logger] = None,
) -> ValidationResult:
    """
    Validate epochs before processing.

    Checks:
    - Minimum epoch count
    - Data contains no NaN/Inf
    - Sampling rate matches expected
    - Channel count is reasonable
    - Data range is reasonable (no extreme values)

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to validate
    config : Any
        Configuration object
    logger : logging.Logger, optional
        Logger for warnings

    Returns
    -------
    ValidationResult
        Validation outcome with issues if any
    """
    logger = _get_logger(logger)
    issues = []
    warnings = []

    if epochs is None:
        return ValidationResult.failure(["Epochs object is None"])

    n_epochs = len(epochs)
    n_channels = len(epochs.ch_names)
    sfreq = epochs.info["sfreq"]

    min_epochs = _get_config_value(config, "validation.min_epochs", DEFAULT_MIN_EPOCHS)
    epoch_issue = _validate_epoch_count(n_epochs, min_epochs)
    if epoch_issue:
        issues.append(epoch_issue)

    expected_sfreq = _get_config_value(
        config, "preprocessing.resample_freq", DEFAULT_SAMPLING_FREQ
    )
    sfreq_warning = _validate_sampling_rate(sfreq, expected_sfreq)
    if sfreq_warning:
        warnings.append(sfreq_warning)

    min_channels = _get_config_value(
        config, "validation.min_channels", DEFAULT_MIN_CHANNELS
    )
    channel_issue = _validate_channel_count(n_channels, min_channels)
    if channel_issue:
        issues.append(channel_issue)

    if epochs.preload or n_epochs <= SMALL_EPOCH_COUNT:
        data_issues, data_warnings = _validate_epochs_data_quality(epochs, config)
        issues.extend(data_issues)
        warnings.extend(data_warnings)

    valid = len(issues) == 0
    metadata = {
        "n_epochs": n_epochs,
        "n_channels": n_channels,
        "sfreq": sfreq,
        "duration_s": epochs.times[-1] - epochs.times[0],
    }

    result = ValidationResult(
        valid=valid,
        issues=issues,
        warnings=warnings,
        metadata=metadata,
    )

    if not valid or warnings:
        result.log_issues(logger)

    return result


###################################################################
# Feature Validation
###################################################################


def _check_dataframe_empty(features_df: pd.DataFrame) -> Optional[str]:
    """Check if DataFrame is None or empty."""
    if features_df is None:
        return "Features DataFrame is None"
    if features_df.empty:
        return "Features DataFrame is empty"
    return None


def _check_minimum_rows(n_rows: int, min_rows: int) -> Optional[str]:
    """Check if DataFrame has minimum required rows."""
    if n_rows < min_rows:
        return f"Insufficient rows: {n_rows} < {min_rows}"
    return None


def _check_expected_columns(
    features_df: pd.DataFrame, expected_columns: Optional[list[str]]
) -> Optional[str]:
    """Check if expected columns are present."""
    if not expected_columns:
        return None
    missing = set(expected_columns) - set(features_df.columns)
    if missing:
        return f"Missing expected columns: {missing}"
    return None


def _check_nan_fractions(
    features_df: pd.DataFrame, max_nan_fraction: float
) -> Optional[str]:
    """Check NaN fractions per column."""
    nan_fractions = features_df.isna().mean()
    high_nan_cols = nan_fractions[nan_fractions > max_nan_fraction]
    if len(high_nan_cols) > 0:
        return f"{len(high_nan_cols)} columns have >{max_nan_fraction:.0%} NaN values"
    return None


def _check_constant_columns(features_df: pd.DataFrame) -> Optional[str]:
    """Check for constant columns."""
    constant_cols = [
        col for col in features_df.columns if features_df[col].nunique() <= 1
    ]
    if constant_cols:
        return f"{len(constant_cols)} constant columns detected"
    return None


def _check_infinite_values(features_df: pd.DataFrame) -> Optional[str]:
    """Check for infinite values in numeric columns."""
    numeric_df = features_df.select_dtypes(include=[np.number])
    inf_count = int(np.sum(np.isinf(numeric_df.values)))
    if inf_count > 0:
        return f"Data contains {inf_count} infinite values"
    return None


def validate_features(
    features_df: pd.DataFrame,
    expected_columns: Optional[list[str]] = None,
    min_rows: int = DEFAULT_MIN_ROWS,
    max_nan_fraction: float = DEFAULT_MAX_NAN_FRACTION,
    logger: Optional[logging.Logger] = None,
) -> ValidationResult:
    """
    Validate a feature DataFrame.

    Checks:
    - DataFrame is not empty
    - Minimum row count
    - NaN fraction within limits
    - Expected columns present (if specified)
    - No constant columns

    Parameters
    ----------
    features_df : pd.DataFrame
        Feature DataFrame to validate
    expected_columns : list[str], optional
        Columns that must be present
    min_rows : int
        Minimum required rows
    max_nan_fraction : float
        Maximum allowed NaN fraction per column
    logger : logging.Logger, optional
        Logger for warnings

    Returns
    -------
    ValidationResult
        Validation outcome
    """
    logger = _get_logger(logger)
    issues = []
    warnings = []

    empty_issue = _check_dataframe_empty(features_df)
    if empty_issue:
        return ValidationResult.failure([empty_issue])

    n_rows, n_cols = features_df.shape

    row_issue = _check_minimum_rows(n_rows, min_rows)
    if row_issue:
        issues.append(row_issue)

    column_issue = _check_expected_columns(features_df, expected_columns)
    if column_issue:
        issues.append(column_issue)

    nan_warning = _check_nan_fractions(features_df, max_nan_fraction)
    if nan_warning:
        warnings.append(nan_warning)

    constant_warning = _check_constant_columns(features_df)
    if constant_warning:
        warnings.append(constant_warning)

    inf_issue = _check_infinite_values(features_df)
    if inf_issue:
        issues.append(inf_issue)

    numeric_df = features_df.select_dtypes(include=[np.number])
    valid = len(issues) == 0
    metadata = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_numeric_cols": len(numeric_df.columns),
        "overall_nan_fraction": float(features_df.isna().mean().mean()),
    }

    result = ValidationResult(
        valid=valid,
        issues=issues,
        warnings=warnings,
        metadata=metadata,
    )

    if not valid or warnings:
        result.log_issues(logger)

    return result


###################################################################
# Target Validation
###################################################################


def _convert_to_array(targets: Any) -> np.ndarray:
    """Convert targets to numpy array."""
    if isinstance(targets, pd.Series):
        return targets.values
    return np.asarray(targets)


def _check_targets_empty(values: np.ndarray) -> Optional[str]:
    """Check if targets array is empty."""
    if len(values) == 0:
        return "Targets is empty"
    return None


def _check_valid_sample_count(n_valid: int, min_samples: int) -> Optional[str]:
    """Check if valid sample count meets minimum."""
    if n_valid < min_samples:
        return f"Insufficient valid samples: {n_valid} < {min_samples}"
    return None


def _check_all_nan(n_valid: int) -> Optional[str]:
    """Check if all values are NaN."""
    if n_valid == 0:
        return "All target values are NaN"
    return None


def _check_value_range(
    valid_values: np.ndarray, expected_range: Optional[tuple[float, float]]
) -> Optional[str]:
    """Check if values are within expected range."""
    if expected_range is None:
        return None
    min_val, max_val = expected_range
    out_of_range = int(np.sum((valid_values < min_val) | (valid_values > max_val)))
    if out_of_range > 0:
        return f"{out_of_range} values outside expected range [{min_val}, {max_val}]"
    return None


def _check_target_variance(valid_values: np.ndarray) -> Optional[str]:
    """Check if target has sufficient variance."""
    variance = float(np.var(valid_values))
    if variance < ZERO_VARIANCE_TARGET_THRESHOLD:
        return "Target has zero variance (all identical values)"
    return None


def _check_unique_value_count(valid_values: np.ndarray) -> Optional[str]:
    """Check for suspiciously few unique values."""
    unique_count = len(np.unique(valid_values))
    if unique_count <= MIN_UNIQUE_VALUES_WARNING:
        return f"Target has only {unique_count} unique values"
    return None


def validate_targets(
    targets: pd.Series | np.ndarray,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    expected_range: Optional[tuple[float, float]] = None,
    logger: Optional[logging.Logger] = None,
) -> ValidationResult:
    """
    Validate target variable (e.g., VAS ratings).

    Checks:
    - Not None/empty
    - Minimum sample count
    - Values within expected range
    - Sufficient variance
    - Not all NaN

    Parameters
    ----------
    targets : pd.Series or np.ndarray
        Target variable
    min_samples : int
        Minimum required samples
    expected_range : tuple, optional
        (min_value, max_value) expected range
    logger : logging.Logger, optional
        Logger for warnings

    Returns
    -------
    ValidationResult
        Validation outcome
    """
    logger = _get_logger(logger)
    issues = []
    warnings = []

    if targets is None:
        return ValidationResult.failure(["Targets is None"])

    values = _convert_to_array(targets)

    empty_issue = _check_targets_empty(values)
    if empty_issue:
        return ValidationResult.failure([empty_issue])

    valid_mask = np.isfinite(values)
    n_valid = int(np.sum(valid_mask))

    sample_issue = _check_valid_sample_count(n_valid, min_samples)
    if sample_issue:
        issues.append(sample_issue)

    all_nan_issue = _check_all_nan(n_valid)
    if all_nan_issue:
        return ValidationResult.failure([all_nan_issue])

    valid_values = values[valid_mask]

    range_warning = _check_value_range(valid_values, expected_range)
    if range_warning:
        warnings.append(range_warning)

    variance_issue = _check_target_variance(valid_values)
    if variance_issue:
        issues.append(variance_issue)

    unique_warning = _check_unique_value_count(valid_values)
    if unique_warning:
        warnings.append(unique_warning)

    valid = len(issues) == 0
    metadata = {
        "n_total": len(values),
        "n_valid": n_valid,
        "nan_fraction": float(1 - n_valid / len(values)),
        "mean": float(np.mean(valid_values)),
        "std": float(np.std(valid_values)),
        "min": float(np.min(valid_values)),
        "max": float(np.max(valid_values)),
    }

    result = ValidationResult(
        valid=valid,
        issues=issues,
        warnings=warnings,
        metadata=metadata,
    )

    if not valid or warnings:
        result.log_issues(logger)

    return result


###################################################################
# Connectivity Validation
###################################################################


def _check_connectivity_shape(
    data: np.ndarray, n_epochs: int, n_channels: int
) -> list[str]:
    """Check connectivity data shape matches expected dimensions."""
    issues = []
    if data.ndim != 3:
        issues.append(
            f"Expected 3D array (epochs, channels, times), got {data.ndim}D"
        )
    else:
        if data.shape[0] != n_epochs:
            issues.append(f"Epoch count mismatch: {data.shape[0]} != {n_epochs}")
        if data.shape[1] != n_channels:
            issues.append(
                f"Channel count mismatch: {data.shape[1]} != {n_channels}"
            )
    return issues


def _check_connectivity_nan(data: np.ndarray) -> tuple[list[str], list[str]]:
    """Check for NaN/Inf in connectivity data."""
    issues = []
    warnings = []
    nan_fraction = float(np.mean(~np.isfinite(data)))
    if nan_fraction > 0:
        if nan_fraction > CRITICAL_NAN_FRACTION_CONNECTIVITY:
            issues.append(f"Data contains {nan_fraction:.1%} NaN/Inf values")
        else:
            warnings.append(f"Data contains {nan_fraction:.2%} NaN/Inf values")
    return issues, warnings


def validate_connectivity_input(
    data: np.ndarray,
    n_epochs: int,
    n_channels: int,
    logger: Optional[logging.Logger] = None,
) -> ValidationResult:
    """
    Validate input for connectivity computation.

    Parameters
    ----------
    data : np.ndarray
        Data array (epochs, channels, times)
    n_epochs : int
        Expected number of epochs
    n_channels : int
        Expected number of channels
    logger : logging.Logger, optional
        Logger

    Returns
    -------
    ValidationResult
        Validation outcome
    """
    _get_logger(logger)
    issues = []
    warnings = []

    if data is None:
        return ValidationResult.failure(["Data is None"])

    shape_issues = _check_connectivity_shape(data, n_epochs, n_channels)
    issues.extend(shape_issues)

    nan_issues, nan_warnings = _check_connectivity_nan(data)
    issues.extend(nan_issues)
    warnings.extend(nan_warnings)

    valid = len(issues) == 0
    metadata = {
        "shape": data.shape,
        "nan_fraction": float(np.mean(~np.isfinite(data))),
    }

    return ValidationResult(
        valid=valid,
        issues=issues,
        warnings=warnings,
        metadata=metadata,
    )


###################################################################
# Data Format Helpers
###################################################################


def _get_percent_threshold(
    percent_threshold: Optional[float], config: Optional[Any]
) -> float:
    """Extract percent threshold from config or use default."""
    if percent_threshold is not None:
        return percent_threshold

    if config and hasattr(config, "get"):
        return float(
            _get_config_value(
                config, "io.constants.percent_threshold", DEFAULT_PERCENT_THRESHOLD
            )
        )

    if config and hasattr(config, "io") and hasattr(config.io, "constants"):
        return config.io.constants.percent_threshold

    return DEFAULT_PERCENT_THRESHOLD


def detect_data_format(
    data: np.ndarray,
    data_format: Optional[str] = None,
    percent_threshold: Optional[float] = None,
    config: Optional[Any] = None,
) -> bool:
    """
    Detect if data is in percent change or absolute units.

    Parameters
    ----------
    data : np.ndarray
        Data to check
    data_format : str, optional
        Force format ("percent" or "abs")
    percent_threshold : float, optional
        Threshold for percent detection
    config : Any, optional
        Config object to look up default threshold

    Returns
    -------
    bool
        True if data appears to be in percent change
    """
    if data_format is not None:
        return data_format == "percent"

    threshold = _get_percent_threshold(percent_threshold, config)
    data_finite = data[np.isfinite(data)]

    if data_finite.size == 0:
        return False

    data_abs_max = float(np.nanmax(np.abs(data_finite)))
    return data_abs_max > threshold


###################################################################
# Alignment Validation
###################################################################


def _handle_alignment_error(
    msg: str, strict: bool, logger: Optional[logging.Logger]
) -> None:
    """Handle alignment error by raising or logging."""
    if strict:
        raise ValueError(msg)
    if logger:
        logger.warning(msg)


def ensure_aligned_lengths(
    *arrays_or_series,
    context: str = "",
    strict: Optional[bool] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Ensure multiple arrays or series have the same length and index.

    Parameters
    ----------
    *arrays_or_series : Array-like
        Objects to check
    context : str
        Error message context
    strict : bool, optional
        If True, raise ValueError on mismatch
    logger : logging.Logger, optional
        Logger for warnings if not strict
    """
    strict_mode = strict if strict is not None else True

    non_null = [obj for obj in arrays_or_series if obj is not None]
    if len(non_null) < 2:
        return

    lengths = {len(obj) for obj in non_null}
    if len(lengths) > 1:
        length_list = [len(obj) for obj in non_null]
        msg = f"{context}: Length mismatch detected: {length_list}"
        _handle_alignment_error(msg, strict_mode, logger)
        return

    ref_index = None
    for obj in non_null:
        index = getattr(obj, "index", None)
        if index is None:
            continue
        if ref_index is None:
            ref_index = index
        elif len(index) != len(ref_index) or not index.equals(ref_index):
            msg = f"{context}: Index misalignment detected; align before analysis"
            _handle_alignment_error(msg, strict_mode, logger)
            return


###################################################################
# Pipeline Specific Validation
###################################################################


def validate_epochs_for_plotting(
    epochs: mne.Epochs, logger: Optional[logging.Logger] = None
) -> bool:
    """Check if epochs are valid for plotting."""
    if epochs is None:
        if logger:
            logger.warning("Epochs object is None")
        return False
    if len(epochs) == 0:
        if logger:
            logger.warning("Epochs object is empty")
        return False
    return True


def require_epochs_tfr(
    tfr: Any, context_msg: str, logger: Optional[logging.Logger] = None
) -> bool:
    """Check if object is EpochsTFR."""
    if not isinstance(tfr, mne.time_frequency.EpochsTFR):
        if logger:
            logger.warning(f"{context_msg} requires EpochsTFR; skipping.")
        return False
    return True


def validate_predictor_file(
    df: pd.DataFrame, predictor_type: str, target: str, logger: logging.Logger
) -> bool:
    """Validate predictor file columns."""
    if predictor_type == "Channel":
        required_columns = {"channel", "band"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            logger.debug(
                f"Skipping combined file for target '{target}' - missing required columns "
                f"(expected 'channel' and 'band')"
            )
            return False
    return True
