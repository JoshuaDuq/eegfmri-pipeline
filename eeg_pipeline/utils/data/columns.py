"""Column finding utilities for events and metadata."""

from __future__ import annotations

from typing import Any, List, Optional

import pandas as pd


def _is_numeric_series(df: pd.DataFrame, column: str) -> bool:
    return bool(column in df.columns and pd.api.types.is_numeric_dtype(df[column]))


def find_column_in_events(events_df: pd.DataFrame, column_names: List[str]) -> Optional[str]:
    """Find first matching column in events dataframe."""
    from eeg_pipeline.utils.data.manipulation import find_column
    return find_column(events_df, column_names)


def find_binary_outcome_column_in_events(events_df: pd.DataFrame, config: Any) -> Optional[str]:
    """Find binary outcome column in events dataframe using config."""
    if config is None:
        raise ValueError("config is required")
    
    event_columns = config.get("event_columns", {})
    column_names = event_columns.get("binary_outcome", [])
    return find_column_in_events(events_df, column_names)


def find_condition_column_in_events(events_df: pd.DataFrame, config: Any) -> Optional[str]:
    """Find generic condition column in events dataframe using config."""
    if config is None:
        raise ValueError("config is required")

    event_columns = config.get("event_columns", {})
    column_names = event_columns.get("condition", [])
    return find_column_in_events(events_df, column_names)


def find_predictor_column_in_events(events_df: pd.DataFrame, config: Any) -> Optional[str]:
    """Find predictor column in events dataframe using config."""
    if config is None:
        raise ValueError("config is required")

    event_columns = config.get("event_columns", {})
    column_names = event_columns.get("predictor", [])
    return find_column_in_events(events_df, column_names)


def get_column_from_config(
    config: Any,
    column_key: str,
    events_df: Optional[pd.DataFrame] = None,
) -> Optional[str]:
    """Get an optional configured column alias, optionally matched against a dataframe."""
    if config is None:
        raise ValueError("config is required")

    columns = config.get(column_key)
    if columns is None:
        return None

    if not columns:
        return None

    if events_df is not None:
        return find_column_in_events(events_df, columns)

    return columns[0]


def get_binary_outcome_column_from_config(
    config: Any,
    events_df: Optional[pd.DataFrame] = None,
) -> Optional[str]:
    """Get binary outcome column from config, optionally matching against events dataframe."""
    return get_column_from_config(config, "event_columns.binary_outcome", events_df)


def get_condition_column_from_config(
    config: Any,
    events_df: Optional[pd.DataFrame] = None,
) -> Optional[str]:
    """Get condition column from config, optionally matching against events dataframe."""
    return get_column_from_config(config, "event_columns.condition", events_df)


def get_predictor_column_from_config(
    config: Any,
    events_df: Optional[pd.DataFrame] = None,
) -> Optional[str]:
    """Get predictor column name from config, optionally matching against events dataframe."""
    return get_column_from_config(config, "event_columns.predictor", events_df)


def get_outcome_column_from_config(
    config: Any,
    events_df: Optional[pd.DataFrame] = None,
) -> Optional[str]:
    """Get outcome column from config, optionally matching against events dataframe."""
    return get_column_from_config(config, "event_columns.outcome", events_df)


def pick_target_column(df: pd.DataFrame, *, target_columns: List[str]) -> Optional[str]:
    """Pick numeric target column from dataframe using explicit candidate names only."""
    for column in target_columns:
        if _is_numeric_series(df, column):
            return column
    return None


def _available_numeric_columns(events_df: pd.DataFrame) -> List[str]:
    """Return numeric columns available for behavior column resolution."""
    return [
        str(column)
        for column in events_df.columns
        if _is_numeric_series(events_df, str(column))
    ]


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
    """Resolve behavior outcome column from explicit config, then outcome aliases."""
    explicit = _get_explicit_behavior_column(
        config,
        key="behavior_analysis.outcome_column",
    )
    if explicit and _is_numeric_series(events_df, explicit):
        return explicit

    outcome_candidates: List[str] = []
    if config is not None and hasattr(config, "get"):
        outcome_candidates = list(config.get("event_columns.outcome", []) or [])
    return pick_target_column(events_df, target_columns=outcome_candidates)


def resolve_predictor_column(
    events_df: pd.DataFrame,
    config: Any,
) -> Optional[str]:
    """Resolve behavior predictor column from explicit config, then predictor aliases."""
    explicit = _get_explicit_behavior_column(
        config,
        key="behavior_analysis.predictor_column",
    )
    if explicit and _is_numeric_series(events_df, explicit):
        return explicit

    predictor_candidates: List[str] = []
    if config is not None and hasattr(config, "get"):
        predictor_candidates = list(config.get("event_columns.predictor", []) or [])
    return pick_target_column(events_df, target_columns=predictor_candidates)


def require_outcome_column(
    events_df: pd.DataFrame,
    config: Any,
) -> str:
    """Resolve the behavior outcome column or raise a configuration error."""
    resolved = resolve_outcome_column(events_df, config)
    if resolved:
        return resolved
    raise ValueError(
        "Could not resolve a numeric behavior outcome column. "
        "Set 'behavior_analysis.outcome_column' or configure "
        f"'event_columns.outcome'. Available numeric columns: {_available_numeric_columns(events_df)}"
    )


def require_predictor_column(
    events_df: pd.DataFrame,
    config: Any,
) -> str:
    """Resolve the behavior predictor column or raise a configuration error."""
    resolved = resolve_predictor_column(events_df, config)
    if resolved:
        return resolved
    raise ValueError(
        "Could not resolve a numeric behavior predictor column. "
        "Set 'behavior_analysis.predictor_column' or configure "
        f"'event_columns.predictor'. Available numeric columns: {_available_numeric_columns(events_df)}"
    )


__all__ = [
    "find_column_in_events",
    "find_binary_outcome_column_in_events",
    "find_condition_column_in_events",
    "find_predictor_column_in_events",
    "get_column_from_config",
    "get_binary_outcome_column_from_config",
    "get_condition_column_from_config",
    "get_predictor_column_from_config",
    "get_outcome_column_from_config",
    "pick_target_column",
    "require_outcome_column",
    "require_predictor_column",
    "resolve_outcome_column",
    "resolve_predictor_column",
]
