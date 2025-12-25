"""Column finding utilities for events and metadata."""

from __future__ import annotations

import functools
from typing import Any, Dict, List, Optional

import mne
import pandas as pd


def _get_io_constants(config: Any) -> Dict[str, Any]:
    if config is None:
        raise ValueError("config is required")

    event_columns = config.get("event_columns", {})
    return {
        "temperature_column_names": event_columns.get("temperature", []),
        "pain_column_names": event_columns.get("pain_binary", []),
    }


def _get_config_key_for_column_type(column_type: Optional[str]) -> Optional[str]:
    config_key_map = {
        "pain": "event_columns.pain_binary",
        "temperature": "event_columns.temperature",
    }
    return config_key_map.get(column_type) if column_type else None


def _find_column_in_dataframe(df: pd.DataFrame, column_names: List[str]) -> Optional[str]:
    return next((col for col in column_names if col in df.columns), None)


def find_column_in_events(events_df: pd.DataFrame, column_names: List[str]) -> Optional[str]:
    return _find_column_in_dataframe(events_df, column_names)


def find_pain_column_in_events(events_df: pd.DataFrame, config: Any) -> Optional[str]:
    constants = _get_io_constants(config)
    return _find_column_in_dataframe(events_df, constants["pain_column_names"])


def find_temperature_column_in_events(events_df: pd.DataFrame, config: Any) -> Optional[str]:
    constants = _get_io_constants(config)
    return _find_column_in_dataframe(events_df, constants["temperature_column_names"])


def find_column_in_metadata(epochs: mne.Epochs, config_key: str, config: Any) -> Optional[str]:
    if not hasattr(epochs, "metadata") or epochs.metadata is None:
        return None
    column_names = config.get(config_key)
    if not column_names:
        return None
    return _find_column_in_dataframe(epochs.metadata, column_names)


def find_pain_column_in_metadata(epochs: mne.Epochs, config: Any) -> Optional[str]:
    return find_column_in_metadata(epochs, "event_columns.pain_binary", config)


def find_temperature_column_in_metadata(epochs: mne.Epochs, config: Any) -> Optional[str]:
    return find_column_in_metadata(epochs, "event_columns.temperature", config)


def get_column_from_config(
    config: Any,
    column_key: str,
    events_df: Optional[pd.DataFrame] = None,
) -> Optional[str]:
    if config is None:
        raise ValueError("config is required")

    columns = config.get(column_key)
    if columns is None:
        raise ValueError(f"{column_key} not found in config")
    if not columns:
        return None

    if events_df is not None:
        return _find_column_in_dataframe(events_df, columns)

    return columns[0] if columns else None


def get_pain_column_from_config(config: Any, events_df: Optional[pd.DataFrame] = None) -> Optional[str]:
    return get_column_from_config(config, "event_columns.pain_binary", events_df)


def get_temperature_column_from_config(config: Any, events_df: Optional[pd.DataFrame] = None) -> Optional[str]:
    return get_column_from_config(config, "event_columns.temperature", events_df)


def pick_target_column(df: pd.DataFrame, *, target_columns: List[str]) -> Optional[str]:
    for col in target_columns:
        if col in df.columns:
            return col

    for col in df.columns:
        col_lower = str(col).lower()
        if ("vas" in col_lower or "rating" in col_lower) and pd.api.types.is_numeric_dtype(df[col]):
            return col

    return None


__all__ = [
    "_get_io_constants",
    "_get_config_key_for_column_type",
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
