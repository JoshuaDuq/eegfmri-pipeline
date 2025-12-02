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
from typing import Any, List, Optional, Dict, Union

import numpy as np
import pandas as pd
import mne


###################################################################
# Validation Result
###################################################################


@dataclass
class ValidationResult:
    """Result of a validation check."""
    
    valid: bool
    issues: List[str] = field(default_factory=list)
    critical: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
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
    def success(cls, **metadata) -> "ValidationResult":
        """Create a successful validation result."""
        return cls(valid=True, metadata=metadata)
    
    @classmethod
    def failure(cls, issues: List[str], **metadata) -> "ValidationResult":
        """Create a failed validation result."""
        return cls(valid=False, issues=issues, metadata=metadata)


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
    if logger is None:
        logger = logging.getLogger(__name__)
    
    issues = []
    warnings = []
    
    # Check epochs exist
    if epochs is None:
        return ValidationResult.failure(["Epochs object is None"])
    
    n_epochs = len(epochs)
    n_channels = len(epochs.ch_names)
    sfreq = epochs.info["sfreq"]
    
    # Check minimum epoch count
    min_epochs = config.get("validation.min_epochs", 20)
    if n_epochs < min_epochs:
        issues.append(f"Insufficient epochs: {n_epochs} < {min_epochs}")
    
    # Check sampling rate
    expected_sfreq = config.get("preprocessing.resample_freq", 500)
    if abs(sfreq - expected_sfreq) > 1:
        warnings.append(f"Unexpected sampling rate: {sfreq} Hz (expected {expected_sfreq})")
    
    # Check channel count
    min_channels = config.get("validation.min_channels", 10)
    if n_channels < min_channels:
        issues.append(f"Insufficient channels: {n_channels} < {min_channels}")
    
    # Check data quality (only if preloaded or small)
    if epochs.preload or n_epochs <= 50:
        try:
            data = epochs.get_data()
            
            # Check for NaN/Inf
            nan_count = np.sum(~np.isfinite(data))
            if nan_count > 0:
                nan_fraction = nan_count / data.size
                if nan_fraction > 0.01:
                    issues.append(f"Data contains {nan_fraction:.1%} NaN/Inf values")
                else:
                    warnings.append(f"Data contains {nan_count} NaN/Inf values ({nan_fraction:.2%})")
            
            # Check for extreme values
            max_uv = config.get("validation.max_amplitude_uv", 500)
            data_uv = data * 1e6  # Convert to microvolts
            extreme_fraction = np.mean(np.abs(data_uv) > max_uv)
            if extreme_fraction > 0.1:
                warnings.append(f"{extreme_fraction:.1%} of data exceeds {max_uv} µV")
            
            # Check for flat channels
            channel_vars = np.var(data, axis=(0, 2))
            flat_channels = np.sum(channel_vars < 1e-12)
            if flat_channels > 0:
                warnings.append(f"{flat_channels} channels appear flat (zero variance)")
                
        except Exception as e:
            warnings.append(f"Could not check data quality: {e}")
    
    # Build result
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


def validate_features(
    features_df: pd.DataFrame,
    expected_columns: Optional[List[str]] = None,
    min_rows: int = 10,
    max_nan_fraction: float = 0.1,
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
    expected_columns : List[str], optional
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
    if logger is None:
        logger = logging.getLogger(__name__)
    
    issues = []
    warnings = []
    
    # Check existence
    if features_df is None:
        return ValidationResult.failure(["Features DataFrame is None"])
    
    if features_df.empty:
        return ValidationResult.failure(["Features DataFrame is empty"])
    
    n_rows, n_cols = features_df.shape
    
    # Check minimum rows
    if n_rows < min_rows:
        issues.append(f"Insufficient rows: {n_rows} < {min_rows}")
    
    # Check expected columns
    if expected_columns:
        missing = set(expected_columns) - set(features_df.columns)
        if missing:
            issues.append(f"Missing expected columns: {missing}")
    
    # Check NaN fraction per column
    nan_fractions = features_df.isna().mean()
    high_nan_cols = nan_fractions[nan_fractions > max_nan_fraction]
    if len(high_nan_cols) > 0:
        warnings.append(f"{len(high_nan_cols)} columns have >{max_nan_fraction:.0%} NaN values")
    
    # Check for constant columns
    constant_cols = [col for col in features_df.columns if features_df[col].nunique() <= 1]
    if constant_cols:
        warnings.append(f"{len(constant_cols)} constant columns detected")
    
    # Check for infinite values
    numeric_df = features_df.select_dtypes(include=[np.number])
    inf_count = np.sum(np.isinf(numeric_df.values))
    if inf_count > 0:
        issues.append(f"Data contains {inf_count} infinite values")
    
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


def validate_targets(
    targets: Union[pd.Series, np.ndarray],
    min_samples: int = 10,
    expected_range: Optional[tuple] = None,
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
    if logger is None:
        logger = logging.getLogger(__name__)
    
    issues = []
    warnings = []
    
    # Check existence
    if targets is None:
        return ValidationResult.failure(["Targets is None"])
    
    # Convert to numpy
    if isinstance(targets, pd.Series):
        values = targets.values
    else:
        values = np.asarray(targets)
    
    # Check not empty
    if len(values) == 0:
        return ValidationResult.failure(["Targets is empty"])
    
    # Get valid (non-NaN) values
    valid_mask = np.isfinite(values)
    n_valid = np.sum(valid_mask)
    
    # Check minimum samples
    if n_valid < min_samples:
        issues.append(f"Insufficient valid samples: {n_valid} < {min_samples}")
    
    # Check not all NaN
    if n_valid == 0:
        return ValidationResult.failure(["All target values are NaN"])
    
    valid_values = values[valid_mask]
    
    # Check range
    if expected_range is not None:
        min_val, max_val = expected_range
        out_of_range = np.sum((valid_values < min_val) | (valid_values > max_val))
        if out_of_range > 0:
            warnings.append(f"{out_of_range} values outside expected range [{min_val}, {max_val}]")
    
    # Check variance
    variance = np.var(valid_values)
    if variance < 1e-10:
        issues.append("Target has zero variance (all identical values)")
    
    # Check for suspicious patterns
    unique_values = np.unique(valid_values)
    if len(unique_values) <= 3:
        warnings.append(f"Target has only {len(unique_values)} unique values")
    
    valid = len(issues) == 0
    metadata = {
        "n_total": len(values),
        "n_valid": int(n_valid),
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
    if logger is None:
        logger = logging.getLogger(__name__)
    
    issues = []
    warnings = []
    
    if data is None:
        return ValidationResult.failure(["Data is None"])
    
    if data.ndim != 3:
        issues.append(f"Expected 3D array (epochs, channels, times), got {data.ndim}D")
    else:
        if data.shape[0] != n_epochs:
            issues.append(f"Epoch count mismatch: {data.shape[0]} != {n_epochs}")
        if data.shape[1] != n_channels:
            issues.append(f"Channel count mismatch: {data.shape[1]} != {n_channels}")
    
    # Check for NaN
    nan_fraction = np.mean(~np.isfinite(data))
    if nan_fraction > 0:
        if nan_fraction > 0.1:
            issues.append(f"Data contains {nan_fraction:.1%} NaN/Inf values")
        else:
            warnings.append(f"Data contains {nan_fraction:.2%} NaN/Inf values")
    
    valid = len(issues) == 0
    metadata = {
        "shape": data.shape if data is not None else None,
        "nan_fraction": float(nan_fraction) if data is not None else None,
    }
    
    result = ValidationResult(
        valid=valid,
        issues=issues,
        warnings=warnings,
        metadata=metadata,
    )
    
    return result


###################################################################
# Data Format Helpers
###################################################################


def detect_data_format(
    data: np.ndarray, 
    data_format: Optional[str] = None, 
    percent_threshold: Optional[float] = None,
    config: Optional[Any] = None
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
    
    if percent_threshold is None:
        # Try to get from config object if available
        if config and hasattr(config, "get"):
            percent_threshold = float(config.get("io.constants.percent_threshold", 5.0))
        # Fallback to looking up by attribute if not a dict-like
        elif config and hasattr(config, "io") and hasattr(config.io, "constants"):
             percent_threshold = config.io.constants.percent_threshold
        else:
             percent_threshold = 5.0
    
    data_finite = data[np.isfinite(data)]
    if data_finite.size == 0:
        return False
    data_abs_max = float(np.nanmax(np.abs(data_finite)))
    return data_abs_max > percent_threshold


###################################################################
# Alignment Validation
###################################################################


def _handle_alignment_error(msg: str, strict: bool, logger: Optional[logging.Logger]) -> None:
    if strict:
        raise ValueError(msg)
    if logger:
        logger.warning(msg)


def ensure_aligned_lengths(
    *arrays_or_series,
    context: str = "",
    strict: Optional[bool] = None,
    logger: Optional[logging.Logger] = None
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
    strict = strict if strict is not None else True
    
    non_null = [obj for obj in arrays_or_series if obj is not None]
    if len(non_null) < 2:
        return
    
    lengths = {len(obj) for obj in non_null}
    if len(lengths) > 1:
        msg = f"{context}: Length mismatch detected: {[len(obj) for obj in non_null]}"
        _handle_alignment_error(msg, strict, logger)
        return
    
    ref_index = None
    for obj in non_null:
        idx = getattr(obj, "index", None)
        if idx is None:
            continue
        if ref_index is None:
            ref_index = idx
        elif len(idx) != len(ref_index) or not idx.equals(ref_index):
            msg = f"{context}: Index misalignment detected; align before analysis"
            _handle_alignment_error(msg, strict, logger)
            return


###################################################################
# Pipeline Specific Validation
###################################################################


def validate_epochs_for_plotting(epochs: mne.Epochs, logger: Optional[logging.Logger] = None) -> bool:
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


def require_epochs_tfr(tfr, context_msg: str, logger: Optional[logging.Logger] = None) -> bool:
    """Check if object is EpochsTFR."""
    if not isinstance(tfr, mne.time_frequency.EpochsTFR):
        if logger:
            logger.warning(f"{context_msg} requires EpochsTFR; skipping.")
        return False
    return True


def validate_predictor_file(df: pd.DataFrame, predictor_type: str, target: str, logger: logging.Logger) -> bool:
    """Validate predictor file columns."""
    if predictor_type == "Channel":
        if "channel" not in df.columns or "band" not in df.columns:
            logger.debug(
                f"Skipping combined file for target '{target}' - missing required columns "
                f"(expected 'channel' and 'band')"
            )
            return False
    return True
