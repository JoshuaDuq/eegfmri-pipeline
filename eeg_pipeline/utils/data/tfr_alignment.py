from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def compute_aligned_data_length(tfr, events_df: Optional[pd.DataFrame]) -> int:
    n_epochs = tfr.data.shape[0]
    n_meta = len(events_df) if events_df is not None else n_epochs
    return min(n_epochs, n_meta)


def extract_aligned_column_vector(
    tfr,
    events_df: Optional[pd.DataFrame],
    column_name: str,
    n: int,
) -> Optional[pd.Series]:
    if getattr(tfr, "metadata", None) is not None and column_name in tfr.metadata.columns:
        return pd.to_numeric(tfr.metadata.iloc[:n][column_name], errors="coerce")
    if events_df is not None and column_name in events_df.columns:
        return pd.to_numeric(events_df.iloc[:n][column_name], errors="coerce")
    return None


def extract_pain_vector_array(
    tfr,
    events_df: Optional[pd.DataFrame],
    pain_col: Optional[str],
    n: int,
) -> Optional[np.ndarray]:
    if pain_col is None:
        return None

    pain_vec = extract_aligned_column_vector(tfr, events_df, pain_col, n)
    if pain_vec is None:
        return None

    pain_vec = pd.to_numeric(pain_vec, errors="coerce").fillna(0).astype(int)
    return pain_vec.values


def extract_pain_vector(
    tfr,
    events_df: Optional[pd.DataFrame],
    pain_col: Optional[str],
    n: int,
) -> Optional[pd.Series]:
    if pain_col is None:
        return None
    pain_vec = extract_aligned_column_vector(tfr, events_df, pain_col, n)
    if pain_vec is None:
        return None
    return pd.to_numeric(pain_vec, errors="coerce").fillna(0).astype(int)


def extract_temperature_series(
    tfr,
    events_df: Optional[pd.DataFrame],
    temp_col: Optional[str],
    n: int,
) -> Optional[pd.Series]:
    if temp_col is None:
        return None
    return extract_aligned_column_vector(tfr, events_df, temp_col, n)


def extract_time_frequency_grid(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if "frequency" not in df.columns or "time" not in df.columns:
        return np.array([]), np.array([])
    freqs = np.unique(np.round(df["frequency"].to_numpy(dtype=float), 6))
    times = np.unique(np.round(df["time"].to_numpy(dtype=float), 6))
    return freqs, times


def create_temperature_masks(
    temp_series: pd.Series,
    temperature_rounding_decimals: int = 1,
    min_temperatures_required: int = 2,
) -> tuple[Optional[float], Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
    if temp_series is None:
        return None, None, None, None

    s_round = pd.to_numeric(temp_series, errors="coerce").round(temperature_rounding_decimals)
    temps = sorted(map(float, s_round.dropna().unique()))
    if len(temps) < min_temperatures_required:
        return None, None, None, None

    t_min = float(min(temps))
    t_max = float(max(temps))
    mask_min = np.asarray(s_round == round(t_min, temperature_rounding_decimals), dtype=bool)
    mask_max = np.asarray(s_round == round(t_max, temperature_rounding_decimals), dtype=bool)
    return t_min, t_max, mask_min, mask_max


def get_temperature_range(
    temp_series: pd.Series,
    temperature_rounding_decimals: int = 1,
    min_temperatures_required: int = 2,
) -> tuple[Optional[float], Optional[float]]:
    if temp_series is None:
        return None, None

    s_round = pd.to_numeric(temp_series, errors="coerce").round(temperature_rounding_decimals)
    temps = sorted(map(float, s_round.dropna().unique()))
    if len(temps) < min_temperatures_required:
        return None, None

    return float(min(temps)), float(max(temps))


def create_temperature_masks_from_range(
    temp_series: pd.Series,
    t_min: float,
    t_max: float,
    temperature_rounding_decimals: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    if temp_series is None or t_min is None or t_max is None:
        return np.array([], dtype=bool), np.array([], dtype=bool)

    s_round = pd.to_numeric(temp_series, errors="coerce").round(temperature_rounding_decimals)
    mask_min = np.asarray(s_round == round(t_min, temperature_rounding_decimals), dtype=bool)
    mask_max = np.asarray(s_round == round(t_max, temperature_rounding_decimals), dtype=bool)
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
