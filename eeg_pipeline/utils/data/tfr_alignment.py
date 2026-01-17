from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def compute_aligned_data_length(tfr, events_df: Optional[pd.DataFrame]) -> int:
    """Compute the minimum length between TFR epochs and events DataFrame."""
    n_epochs = tfr.data.shape[0]
    n_meta = len(events_df) if events_df is not None else n_epochs
    return min(n_epochs, n_meta)


def _has_metadata(tfr) -> bool:
    """Check if TFR object has metadata attribute with DataFrame."""
    return hasattr(tfr, "metadata") and tfr.metadata is not None


def extract_aligned_column_vector(
    tfr,
    events_df: Optional[pd.DataFrame],
    column_name: str,
    n_samples: int,
) -> Optional[pd.Series]:
    """Extract a column vector from TFR metadata or events DataFrame.
    
    Args:
        tfr: TFR object that may have metadata attribute
        events_df: Optional events DataFrame
        column_name: Name of column to extract
        n_samples: Number of samples to extract
        
    Returns:
        Series with numeric values, or None if column not found
    """
    if _has_metadata(tfr) and column_name in tfr.metadata.columns:
        return pd.to_numeric(tfr.metadata.iloc[:n_samples][column_name], errors="coerce")
    if events_df is not None and column_name in events_df.columns:
        return pd.to_numeric(events_df.iloc[:n_samples][column_name], errors="coerce")
    return None


def extract_pain_vector(
    tfr,
    events_df: Optional[pd.DataFrame],
    pain_col: Optional[str],
    n_samples: int,
) -> Optional[pd.Series]:
    """Extract pain vector as Series from TFR or events DataFrame.
    
    Args:
        tfr: TFR object that may have metadata attribute
        events_df: Optional events DataFrame
        pain_col: Name of pain column, or None
        n_samples: Number of samples to extract
        
    Returns:
        Series with integer pain values, or None if column not found
    """
    if pain_col is None:
        return None
    pain_vec = extract_aligned_column_vector(tfr, events_df, pain_col, n_samples)
    if pain_vec is None:
        return None
    return pd.to_numeric(pain_vec, errors="coerce").fillna(0).astype(int)


def extract_pain_vector_array(
    tfr,
    events_df: Optional[pd.DataFrame],
    pain_col: Optional[str],
    n_samples: int,
) -> Optional[np.ndarray]:
    """Extract pain vector as numpy array from TFR or events DataFrame.
    
    Args:
        tfr: TFR object that may have metadata attribute
        events_df: Optional events DataFrame
        pain_col: Name of pain column, or None
        n_samples: Number of samples to extract
        
    Returns:
        Array with integer pain values, or None if column not found
    """
    pain_series = extract_pain_vector(tfr, events_df, pain_col, n_samples)
    if pain_series is None:
        return None
    return pain_series.values


def extract_temperature_series(
    tfr,
    events_df: Optional[pd.DataFrame],
    temp_col: Optional[str],
    n_samples: int,
) -> Optional[pd.Series]:
    """Extract temperature series from TFR or events DataFrame.
    
    Args:
        tfr: TFR object that may have metadata attribute
        events_df: Optional events DataFrame
        temp_col: Name of temperature column, or None
        n_samples: Number of samples to extract
        
    Returns:
        Series with numeric temperature values, or None if column not found
    """
    if temp_col is None:
        return None
    return extract_aligned_column_vector(tfr, events_df, temp_col, n_samples)


def extract_time_frequency_grid(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract unique frequency and time values from DataFrame.
    
    Args:
        df: DataFrame with 'freq' and 'time_start' columns
        
    Returns:
        Tuple of (frequencies, times) arrays, or empty arrays if columns missing
    """
    if "freq" not in df.columns or "time_start" not in df.columns:
        return np.array([]), np.array([])
    freqs = np.unique(np.round(df["freq"].to_numpy(dtype=float), 6))
    times = np.unique(np.round(df["time_start"].to_numpy(dtype=float), 6))
    return freqs, times


def _extract_unique_temperatures(
    temp_series: pd.Series,
    temperature_rounding_decimals: int,
) -> list[float]:
    """Extract sorted unique temperature values from series.
    
    Args:
        temp_series: Series with temperature values
        temperature_rounding_decimals: Decimal places for rounding
        
    Returns:
        Sorted list of unique temperature values
    """
    rounded_series = pd.to_numeric(temp_series, errors="coerce").round(
        temperature_rounding_decimals
    )
    unique_temps = rounded_series.dropna().unique()
    return sorted(map(float, unique_temps))


def _create_temperature_mask(
    temp_series: pd.Series,
    target_temp: float,
    temperature_rounding_decimals: int,
) -> np.ndarray:
    """Create boolean mask for samples matching target temperature.
    
    Args:
        temp_series: Series with temperature values
        target_temp: Target temperature to match
        temperature_rounding_decimals: Decimal places for rounding
        
    Returns:
        Boolean array indicating matching samples
    """
    rounded_series = pd.to_numeric(temp_series, errors="coerce").round(
        temperature_rounding_decimals
    )
    target_rounded = round(target_temp, temperature_rounding_decimals)
    return np.asarray(rounded_series == target_rounded, dtype=bool)


def get_temperature_range(
    temp_series: pd.Series,
    temperature_rounding_decimals: int = 1,
    min_temperatures_required: int = 2,
) -> tuple[Optional[float], Optional[float]]:
    """Extract minimum and maximum temperature values from series.
    
    Args:
        temp_series: Series with temperature values, or None
        temperature_rounding_decimals: Decimal places for rounding
        min_temperatures_required: Minimum unique temperatures needed
        
    Returns:
        Tuple of (min_temp, max_temp), or (None, None) if insufficient data
    """
    if temp_series is None:
        return None, None

    unique_temps = _extract_unique_temperatures(temp_series, temperature_rounding_decimals)
    if len(unique_temps) < min_temperatures_required:
        return None, None

    return float(min(unique_temps)), float(max(unique_temps))


def create_temperature_masks(
    temp_series: pd.Series,
    temperature_rounding_decimals: int = 1,
    min_temperatures_required: int = 2,
) -> tuple[Optional[float], Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
    """Create masks for minimum and maximum temperature conditions.
    
    Args:
        temp_series: Series with temperature values, or None
        temperature_rounding_decimals: Decimal places for rounding
        min_temperatures_required: Minimum unique temperatures needed
        
    Returns:
        Tuple of (t_min, t_max, mask_min, mask_max), with None values if insufficient data
    """
    if temp_series is None:
        return None, None, None, None

    unique_temps = _extract_unique_temperatures(temp_series, temperature_rounding_decimals)
    if len(unique_temps) < min_temperatures_required:
        return None, None, None, None

    t_min = float(min(unique_temps))
    t_max = float(max(unique_temps))
    mask_min = _create_temperature_mask(temp_series, t_min, temperature_rounding_decimals)
    mask_max = _create_temperature_mask(temp_series, t_max, temperature_rounding_decimals)
    return t_min, t_max, mask_min, mask_max


def create_temperature_masks_from_range(
    temp_series: pd.Series,
    t_min: float,
    t_max: float,
    temperature_rounding_decimals: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create masks for specified minimum and maximum temperatures.
    
    Args:
        temp_series: Series with temperature values, or None
        t_min: Minimum temperature value
        t_max: Maximum temperature value
        temperature_rounding_decimals: Decimal places for rounding
        
    Returns:
        Tuple of (mask_min, mask_max) boolean arrays, or empty arrays if invalid input
    """
    if temp_series is None or t_min is None or t_max is None:
        return np.array([], dtype=bool), np.array([], dtype=bool)

    mask_min = _create_temperature_mask(temp_series, t_min, temperature_rounding_decimals)
    mask_max = _create_temperature_mask(temp_series, t_max, temperature_rounding_decimals)
    return mask_min, mask_max


__all__ = [
    "compute_aligned_data_length",
    "extract_aligned_column_vector",
    "extract_pain_vector",
    "extract_pain_vector_array",
    "extract_temperature_series",
    "extract_time_frequency_grid",
    "create_temperature_masks",
    "get_temperature_range",
    "create_temperature_masks_from_range",
]
