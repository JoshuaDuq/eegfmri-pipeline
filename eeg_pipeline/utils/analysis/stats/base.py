"""
Base Statistics Utilities
=========================

Constants, config helpers, and core data structures for statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from scipy import stats

from eeg_pipeline.utils.config.loader import (
    ensure_config,
    get_config_value,
    get_constants,
    load_config,
)


@dataclass
class CorrelationStats:
    """Container for correlation statistics."""

    correlation: float = np.nan
    p_value: float = np.nan
    ci_low: float = np.nan
    ci_high: float = np.nan
    r_partial: float = np.nan
    p_partial: float = np.nan
    n_partial: int = 0
    r_partial_temp: float = np.nan
    p_partial_temp: float = np.nan
    n_partial_temp: int = 0
    p_perm: float = np.nan
    p_partial_perm: float = np.nan
    p_partial_temp_perm: float = np.nan


def get_statistics_constants(config=None):
    """Load statistics constants from config."""
    try:
        constants = get_constants("statistics", config)
        _normalize_cluster_structure(constants)
        _ensure_min_samples_default(constants)
        return constants
    except ValueError:
        pass

    config = ensure_config(config)
    if config is None:
        raise ValueError("Config required for statistics constants.")

    constants = config.get("statistics.constants")
    if constants is None:
        return {
            "min_samples_for_correlation": 5,
            "fisher_z_clip_min": -0.999999,
            "fisher_z_clip_max": 0.999999,
        }

    result = dict(constants)
    _normalize_cluster_structure(result)
    _ensure_min_samples_default(result)
    return result


def _normalize_cluster_structure(constants: dict) -> None:
    """Convert cluster_structure_2d to numpy array if present."""
    if "cluster_structure_2d" in constants:
        constants["cluster_structure_2d"] = np.array(
            constants["cluster_structure_2d"], dtype=int
        )


def _ensure_min_samples_default(constants: dict) -> None:
    """Ensure min_samples_for_correlation has a default value."""
    if "min_samples_for_correlation" not in constants:
        constants["min_samples_for_correlation"] = 5


def get_fdr_alpha(config: Optional[Any] = None) -> float:
    """Get FDR alpha from config."""
    default_alpha = get_config_value(config, "statistics.sig_alpha", 0.05)
    fdr_alpha = get_config_value(config, "statistics.fdr_alpha", default_alpha)
    return float(fdr_alpha)


def get_ci_level(config: Optional[Any] = None) -> float:
    """Get confidence interval level from config."""
    return float(get_config_value(config, "statistics.ci_level", 0.95))


def get_z_critical_value(ci_level: float) -> float:
    """Compute z-critical value for given confidence interval level.
    
    Consolidated utility function for computing z-critical values used
    throughout the statistics modules.
    
    Parameters
    ----------
    ci_level : float
        Confidence level (e.g., 0.95 for 95% CI)
        
    Returns
    -------
    float
        Z-critical value for two-tailed test
    """
    return float(stats.norm.ppf((1 + ci_level) / 2))


def filter_finite_values(values: np.ndarray, flatten: bool = False) -> np.ndarray:
    """Extract finite values from array.
    
    Consolidated utility for filtering finite values used throughout
    the statistics modules.
    
    Parameters
    ----------
    values : np.ndarray
        Input array
    flatten : bool
        If True, flatten array before filtering (default: False)
        
    Returns
    -------
    np.ndarray
        Array containing only finite values
    """
    if flatten:
        values = np.asarray(values).ravel()
    else:
        values = np.asarray(values)
    return values[np.isfinite(values)]


def get_n_permutations(config: Optional[Any] = None) -> int:
    """Get number of permutations from config."""
    return int(get_config_value(config, "statistics.n_permutations", 1000))


def get_n_bootstrap(config: Optional[Any] = None) -> int:
    """Get number of bootstrap samples from config."""
    return int(get_config_value(config, "statistics.n_bootstrap", 1000))


def get_epsilon_std(config: Optional[Any] = None) -> float:
    """Get epsilon for standard deviation/division operations."""
    return float(get_config_value(config, "epsilon_std", 1e-12))


def get_epsilon_psd(config: Optional[Any] = None) -> float:
    """Get epsilon for power spectral density/log operations."""
    return float(get_config_value(config, "epsilon_psd", 1e-20))


def get_epsilon_amp(config: Optional[Any] = None) -> float:
    """Get epsilon for amplitude calculations."""
    return float(get_config_value(config, "epsilon_amp", 1e-10))


def get_min_samples_for_correlation(config: Optional[Any] = None) -> int:
    """Get minimum samples required for correlation calculations."""
    return int(get_config_value(config, "statistics.constants.min_samples_for_correlation", 5))


def get_subject_seed(base_seed: int, subject: str) -> int:
    """Generate reproducible seed from subject ID.
    
    Useful for ensuring bootstrap/permutation results are reproducible
    per subject while varying across subjects.
    
    Parameters
    ----------
    base_seed : int
        Base random seed.
    subject : str
        Subject identifier
        
    Returns
    -------
    int
        Subject-specific seed
    """
    import hashlib
    digest = hashlib.sha256(f"{base_seed}:{subject}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2**31)


def safe_get_config_value(config: Any, key: str, default: Any) -> Any:
    """Safely get a value from config object, returning default if unavailable.
    
    This is a convenience wrapper around get_config_value that handles
    cases where config might be None or not have the expected structure.
    
    Parameters
    ----------
    config : Any
        Configuration object (dict-like or object with get method)
    key : str
        Configuration key to retrieve
    default : Any
        Default value if key is not found
        
    Returns
    -------
    Any
        Configuration value or default
    """
    if config is None:
        return default
    try:
        if hasattr(config, "get"):
            return config.get(key, default)
    except (AttributeError, KeyError, TypeError):
        pass
    # Fallback to get_config_value from loader if available
    try:
        return get_config_value(config, key, default)
    except (AttributeError, KeyError, TypeError, ValueError):
        return default


def _safe_float(value: Any) -> float:
    """Safely convert value to float, returning NaN on failure.
    
    Consolidated utility for safely converting values to float used throughout
    the statistics modules. Handles None, non-numeric types, and infinite values.
    
    Parameters
    ----------
    value : Any
        Value to convert to float
        
    Returns
    -------
    float
        Converted float value, or NaN if conversion fails or value is infinite
    """
    try:
        f = float(value)
        return f if np.isfinite(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")

