"""Column finding utilities for events and metadata."""

from __future__ import annotations

from typing import Any, List, Optional

import pandas as pd

VAS_KEYWORD = "vas"
RATING_KEYWORD = "rating"


def find_column_in_events(events_df: pd.DataFrame, column_names: List[str]) -> Optional[str]:
    """Find first matching column in events dataframe."""
    from eeg_pipeline.utils.data.manipulation import find_column
    return find_column(events_df, column_names)


def find_binary_outcome_column_in_events(events_df: pd.DataFrame, config: Any) -> Optional[str]:
    """Find pain column in events dataframe using config."""
    if config is None:
        raise ValueError("config is required")
    
    event_columns = config.get("event_columns", {})
    column_names = event_columns.get("binary_outcome", [])
    return find_column_in_events(events_df, column_names)


def find_temperature_column_in_events(events_df: pd.DataFrame, config: Any) -> Optional[str]:
    """Find temperature column in events dataframe using config."""
    if config is None:
        raise ValueError("config is required")
    
    event_columns = config.get("event_columns", {})
    column_names = event_columns.get("temperature", [])
    return find_column_in_events(events_df, column_names)


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
        return find_column_in_events(events_df, columns)

    return columns[0]


def get_binary_outcome_column_from_config(
    config: Any,
    events_df: Optional[pd.DataFrame] = None,
) -> Optional[str]:
    """Get pain column name from config, optionally matching against events dataframe."""
    return get_column_from_config(config, "event_columns.binary_outcome", events_df)


def get_temperature_column_from_config(
    config: Any,
    events_df: Optional[pd.DataFrame] = None,
) -> Optional[str]:
    """Get temperature column name from config, optionally matching against events dataframe."""
    return get_column_from_config(config, "event_columns.temperature", events_df)


def get_rating_column_from_config(
    config: Any,
    events_df: Optional[pd.DataFrame] = None,
) -> Optional[str]:
    """Get rating column name from config, optionally matching against events dataframe."""
    return get_column_from_config(config, "event_columns.rating", events_df)


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


def _get_explicit_behavior_column(
    config: Any,
    *,
    key: str,
) -> Optional[str]:
    """Resolve an explicit behavior-analysis column key from config."""
    if config is None or not hasattr(config, "get"):
        return None
    raw = config.get(key)
    if raw is None:
        return None
    value = str(raw).strip()
    return value if value else None


def resolve_outcome_column(
    events_df: pd.DataFrame,
    config: Any,
) -> Optional[str]:
    """Resolve behavior outcome column from explicit config, then rating aliases."""
    explicit = _get_explicit_behavior_column(
        config,
        key="behavior_analysis.outcome_column",
    )
    if explicit and explicit in events_df.columns:
        return explicit

    if "rating" in events_df.columns:
        return "rating"

    rating_candidates = []
    if config is not None and hasattr(config, "get"):
        rating_candidates = list(config.get("event_columns.rating", []) or [])
    return pick_target_column(events_df, target_columns=rating_candidates)


def resolve_predictor_column(
    events_df: pd.DataFrame,
    config: Any,
) -> Optional[str]:
    """Resolve behavior predictor column from explicit config, then temperature aliases."""
    explicit = _get_explicit_behavior_column(
        config,
        key="behavior_analysis.predictor_column",
    )
    if explicit and explicit in events_df.columns:
        return explicit

    if "temperature" in events_df.columns:
        return "temperature"

    temperature_candidates: List[str] = []
    if config is not None and hasattr(config, "get"):
        temperature_candidates = list(config.get("event_columns.temperature", []) or [])

    for candidate in temperature_candidates:
        if candidate in events_df.columns:
            return candidate
    return None


__all__ = [
    "find_column_in_events",
    "find_binary_outcome_column_in_events",
    "find_temperature_column_in_events",
    "get_column_from_config",
    "get_binary_outcome_column_from_config",
    "get_temperature_column_from_config",
    "get_rating_column_from_config",
    "pick_target_column",
    "resolve_outcome_column",
    "resolve_predictor_column",
]
