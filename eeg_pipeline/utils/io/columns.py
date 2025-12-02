"""
Column finding utilities for events and metadata.

This module provides functions for finding and extracting columns
from events DataFrames and epochs metadata.
"""

from __future__ import annotations

from typing import Optional, List, Any, Dict
import functools
import pandas as pd
import mne

try:
    from ..config.loader import load_settings, get_config_value
except ImportError:
    load_settings = None
    def get_config_value(config, key, default):
        if config is None:
            return default
        if hasattr(config, "get"):
            return config.get(key, default)
        if isinstance(config, dict):
            keys = key.split('.')
            value = config
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
        return default


@functools.lru_cache(maxsize=None)
def _get_io_constants(config=None):
    if config is None:
        if load_settings is not None:
            config = load_settings()
    
    if config is None:
        raise ValueError("Config is required. Cannot load IO constants without config.")
    
    event_columns = config.get("event_columns", {})
    return {
        "temperature_column_names": event_columns.get("temperature", []),
        "pain_column_names": event_columns.get("pain_binary", []),
    }


def _get_config_key_for_column_type(column_type: Optional[str]) -> Optional[str]:
    config_key_map = {
        "pain": "event_columns.pain_binary",
        "temperature": "event_columns.temperature"
    }
    return config_key_map.get(column_type) if column_type else None


def _find_column_in_dataframe(df: pd.DataFrame, column_names: List[str]) -> Optional[str]:
    return next((col for col in column_names if col in df.columns), None)


def _get_column_names_for_type(column_type: str, config: Optional[Any] = None) -> Optional[List[str]]:
    constants = _get_io_constants(config)
    column_map = {
        "pain": constants["pain_column_names"],
        "temperature": constants["temperature_column_names"]
    }
    return column_map.get(column_type)


def find_column_in_events(events_df: pd.DataFrame, column_names: List[str]) -> Optional[str]:
    return _find_column_in_dataframe(events_df, column_names)


def find_pain_column_in_events(events_df: pd.DataFrame, config: Optional[Any] = None) -> Optional[str]:
    column_names = _get_column_names_for_type("pain", config)
    if not column_names:
        return None
    return _find_column_in_dataframe(events_df, column_names)


def find_temperature_column_in_events(events_df: pd.DataFrame, config: Optional[Any] = None) -> Optional[str]:
    column_names = _get_column_names_for_type("temperature", config)
    if not column_names:
        return None
    return _find_column_in_dataframe(events_df, column_names)


def find_column_in_metadata(epochs: mne.Epochs, config_key: str, config) -> Optional[str]:
    if not hasattr(epochs, "metadata") or epochs.metadata is None:
        return None
    column_names = config.get(config_key)
    if not column_names:
        return None
    return _find_column_in_dataframe(epochs.metadata, column_names)


def find_pain_column_in_metadata(epochs: mne.Epochs, config) -> Optional[str]:
    config_key = _get_config_key_for_column_type("pain")
    if not config_key:
        return None
    return find_column_in_metadata(epochs, config_key, config)


def find_temperature_column_in_metadata(epochs: mne.Epochs, config) -> Optional[str]:
    config_key = _get_config_key_for_column_type("temperature")
    if not config_key:
        return None
    return find_column_in_metadata(epochs, config_key, config)


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


def get_pain_column_from_config(config, events_df: Optional[pd.DataFrame] = None) -> Optional[str]:
    return get_column_from_config(config, "event_columns.pain_binary", events_df)


def get_temperature_column_from_config(config, events_df: Optional[pd.DataFrame] = None) -> Optional[str]:
    return get_column_from_config(config, "event_columns.temperature", events_df)


def _pick_target_column(df: pd.DataFrame, constants: Dict[str, Any]) -> Optional[str]:
    if constants is None:
        raise ValueError("constants is required for _pick_target_column")
    
    target_columns = tuple(constants.get("TARGET_COLUMNS", ()))
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
    "_get_column_names_for_type",
    "find_column_in_events",
    "find_pain_column_in_events",
    "find_temperature_column_in_events",
    "find_column_in_metadata",
    "find_pain_column_in_metadata",
    "find_temperature_column_in_metadata",
    "get_column_from_config",
    "get_pain_column_from_config",
    "get_temperature_column_from_config",
    "_pick_target_column",
]

