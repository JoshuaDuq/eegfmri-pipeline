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

from eeg_pipeline.utils.config.loader import get_config_value


###################################################################
# Constants
###################################################################

DEFAULT_MIN_EPOCHS = 20
DEFAULT_MIN_CHANNELS = 10
DEFAULT_SAMPLING_FREQ = 500
DEFAULT_MAX_AMPLITUDE_UV = 500
DEFAULT_MIN_ROWS = 10
DEFAULT_MAX_NAN_FRACTION = 0.1
DEFAULT_PERCENT_THRESHOLD = 5.0
CRITICAL_NAN_FRACTION = 0.01
WARNING_EXTREME_FRACTION = 0.1
ZERO_VARIANCE_THRESHOLD = 1e-12
SAMPLING_FREQ_TOLERANCE = 1.0
SMALL_EPOCH_COUNT = 50
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

        max_uv = get_config_value(
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

    min_epochs = get_config_value(config, "validation.min_epochs", DEFAULT_MIN_EPOCHS)
    epoch_issue = _validate_epoch_count(n_epochs, min_epochs)
    if epoch_issue:
        issues.append(epoch_issue)

    expected_sfreq = get_config_value(
        config, "preprocessing.resample_freq", DEFAULT_SAMPLING_FREQ
    )
    sfreq_warning = _validate_sampling_rate(sfreq, expected_sfreq)
    if sfreq_warning:
        warnings.append(sfreq_warning)

    min_channels = get_config_value(
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
# Data Format Helpers
###################################################################


def _get_percent_threshold(
    percent_threshold: Optional[float], config: Optional[Any]
) -> float:
    """Extract percent threshold from config or use default."""
    if percent_threshold is not None:
        return percent_threshold
    return float(
        get_config_value(
            config, "io.constants.percent_threshold", DEFAULT_PERCENT_THRESHOLD
        )
    )


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


def require_epochs_tfr(
    tfr: Any, context_msg: str, logger: Optional[logging.Logger] = None
) -> bool:
    """Check if object is EpochsTFR."""
    if not isinstance(tfr, mne.time_frequency.EpochsTFR):
        if logger:
            logger.warning(f"{context_msg} requires EpochsTFR; skipping.")
        return False
    return True


###################################################################
# Data Value Validation
###################################################################

def check_pyriemann() -> bool:
    """Check if pyriemann package is available."""
    try:
        import pyriemann  # type: ignore[reportMissingImports]
        return True
    except ImportError:
        return False
