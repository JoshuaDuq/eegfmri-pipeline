"""Column finding utilities for events and metadata."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import mne
import pandas as pd

# Constants for target column matching
VAS_KEYWORD = "vas"
RATING_KEYWORD = "rating"


def _extract_column_names_from_config(config: Any) -> Dict[str, List[str]]:
    """Extract column name lists from config for pain and temperature."""
    if config is None:
        raise ValueError("config is required")

    event_columns = config.get("event_columns", {})
    return {
        "temperature_column_names": event_columns.get("temperature", []),
        "pain_column_names": event_columns.get("pain_binary", []),
    }


def _find_column_in_dataframe(df: pd.DataFrame, column_names: List[str]) -> Optional[str]:
    """Find first matching column name from list in dataframe.
    
    This function uses find_column from data.manipulation as the canonical implementation.
    """
    from eeg_pipeline.utils.data.manipulation import find_column
    return find_column(df, column_names)


def find_column_in_events(events_df: pd.DataFrame, column_names: List[str]) -> Optional[str]:
    """Find first matching column in events dataframe."""
    return _find_column_in_dataframe(events_df, column_names)


def find_pain_column_in_events(events_df: pd.DataFrame, config: Any) -> Optional[str]:
    """Find pain column in events dataframe using config."""
    column_names = _extract_column_names_from_config(config)["pain_column_names"]
    return _find_column_in_dataframe(events_df, column_names)


def find_temperature_column_in_events(events_df: pd.DataFrame, config: Any) -> Optional[str]:
    """Find temperature column in events dataframe using config."""
    column_names = _extract_column_names_from_config(config)["temperature_column_names"]
    return _find_column_in_dataframe(events_df, column_names)


def find_column_in_metadata(epochs: mne.Epochs, config_key: str, config: Any) -> Optional[str]:
    """Find column in epochs metadata using config key."""
    if not hasattr(epochs, "metadata") or epochs.metadata is None:
        return None

    column_names = config.get(config_key)
    if not column_names:
        return None

    return _find_column_in_dataframe(epochs.metadata, column_names)


def find_pain_column_in_metadata(epochs: mne.Epochs, config: Any) -> Optional[str]:
    """Find pain column in epochs metadata using config."""
    return find_column_in_metadata(epochs, "event_columns.pain_binary", config)


def find_temperature_column_in_metadata(epochs: mne.Epochs, config: Any) -> Optional[str]:
    """Find temperature column in epochs metadata using config."""
    return find_column_in_metadata(epochs, "event_columns.temperature", config)


def get_column_from_config(
    config: Any,
    column_key: str,
    events_df: Optional[pd.DataFrame] = None,
) -> Optional[str]:
    """Get column name from config, optionally matching against events dataframe."""
    if config is None:
        raise ValueError("config is required")

    columns = config.get(column_key)
    if columns is None:
        raise ValueError(f"{column_key} not found in config")

    if not columns:
        return None

    if events_df is not None:
        return _find_column_in_dataframe(events_df, columns)

    return columns[0]


def get_pain_column_from_config(config: Any, events_df: Optional[pd.DataFrame] = None) -> Optional[str]:
    """Get pain column name from config, optionally matching against events dataframe."""
    return get_column_from_config(config, "event_columns.pain_binary", events_df)


def get_temperature_column_from_config(config: Any, events_df: Optional[pd.DataFrame] = None) -> Optional[str]:
    """Get temperature column name from config, optionally matching against events dataframe."""
    return get_column_from_config(config, "event_columns.temperature", events_df)


def pick_target_column(df: pd.DataFrame, *, target_columns: List[str]) -> Optional[str]:
    """Pick target column from dataframe, matching target list or VAS/rating patterns."""
    for col in target_columns:
        if col in df.columns:
            return col

    for col in df.columns:
        col_lower = str(col).lower()
        has_vas_or_rating = VAS_KEYWORD in col_lower or RATING_KEYWORD in col_lower
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        if has_vas_or_rating and is_numeric:
            return col

    return None


__all__ = [
    "_extract_column_names_from_config",
    "_find_column_in_dataframe",
    "find_column_in_events",
    "find_pain_column_in_events",
    "find_temperature_column_in_events",
    "find_column_in_metadata",
    "find_pain_column_in_metadata",
    "find_temperature_column_in_metadata",
    "get_column_from_config",
    "get_pain_column_from_config",
    "get_temperature_column_from_config",
    "pick_target_column",
]
