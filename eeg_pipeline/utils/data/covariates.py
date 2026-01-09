"""
Covariate Extraction Utilities
==============================

Functions for extracting and managing alignment covariates (e.g., temperature, trials).
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any, Tuple

import pandas as pd

from ..config.loader import load_config


###################################################################
# Constants
###################################################################


TRIAL_COLUMN_CANDIDATES = ["trial", "trial_number", "trial_index", "run", "block"]

TEMPERATURE_ALIASES = {"stimulus_temp", "stimulus_temperature", "temp", "temperature"}

TRIAL_ALIASES = {"trial", "trial_number", "trial_index", "run", "block"}


###################################################################
# Config Loading
###################################################################


def _load_config_safely() -> Optional[Any]:
    """Load config, returning None if unavailable."""
    try:
        return load_config()
    except (OSError, ValueError):
        return None


###################################################################
# Column Selection
###################################################################


def _pick_first_column(df: Optional[pd.DataFrame], candidates: List[str]) -> Optional[str]:
    """Return first matching column from candidates, or None."""
    if df is None or not candidates:
        return None
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


###################################################################
# Name Resolution
###################################################################


def _canonical_covariate_name(name: Optional[str], config: Optional[Any] = None) -> Optional[str]:
    """Resolve covariate name to canonical form (temperature, trial, etc.)."""
    if name is None:
        return None
    
    normalized_name = str(name).lower()
    
    if config is None:
        config = _load_config_safely()
    
    temperature_aliases = TEMPERATURE_ALIASES.copy()
    trial_aliases = TRIAL_ALIASES.copy()
    
    if config is not None and hasattr(config, "get"):
        config_temp_cols = config.get("event_columns.temperature", [])
        temperature_aliases.update(str(col).lower() for col in config_temp_cols)
    
    if normalized_name in temperature_aliases:
        return "temperature"
    if normalized_name in trial_aliases:
        return "trial"
    
    return normalized_name


###################################################################
# Extraction Functions
###################################################################


def extract_default_covariates(events_df: pd.DataFrame, config: Any) -> List[str]:
    """Extract default covariates (temperature, trial) from events."""
    covariates = []
    
    temperature_columns = config.get("event_columns.temperature")
    temperature_column = _pick_first_column(events_df, temperature_columns)
    if temperature_column:
        covariates.append(temperature_column)
    
    trial_column = _pick_first_column(events_df, TRIAL_COLUMN_CANDIDATES)
    if trial_column:
        covariates.append(trial_column)
    
    return covariates


def extract_temperature_data(
    aligned_events: Optional[pd.DataFrame],
    config: Any,
) -> Tuple[Optional[pd.Series], Optional[str]]:
    """Extract temperature series and column name from aligned events."""
    if aligned_events is None:
        return None, None
    
    temperature_columns = config.get("event_columns.temperature")
    temperature_column = _pick_first_column(aligned_events, temperature_columns)
    if not temperature_column:
        return None, None
    
    temperature_series = pd.to_numeric(
        aligned_events[temperature_column], errors="coerce"
    )
    return temperature_series, temperature_column


###################################################################
# Matrix Building
###################################################################


def _resolve_requested_covariate(
    covariate_name: str,
    events_df: pd.DataFrame,
    config: Any,
    temperature_candidates: List[str],
    covariate_columns: List[str],
    column_name_map: Dict[str, str],
) -> None:
    """Resolve a single requested covariate name to a column."""
    if covariate_name in events_df.columns:
        canonical_name = (
            _canonical_covariate_name(covariate_name, config=config) or covariate_name
        )
        covariate_columns.append(covariate_name)
        column_name_map[covariate_name] = canonical_name
        return
    
    canonical_name = _canonical_covariate_name(covariate_name, config=config)
    if canonical_name == "temperature":
        temperature_column = _pick_first_column(events_df, temperature_candidates)
        if temperature_column:
            covariate_columns.append(temperature_column)
            column_name_map[temperature_column] = canonical_name


def _resolve_default_covariates(
    events_df: pd.DataFrame,
    config: Any,
    temperature_candidates: List[str],
    covariate_columns: List[str],
    column_name_map: Dict[str, str],
) -> None:
    """Resolve default covariates (temperature and trial)."""
    temperature_column = _pick_first_column(events_df, temperature_candidates)
    if temperature_column:
        covariate_columns.append(temperature_column)
        column_name_map[temperature_column] = "temperature"
    
    trial_column = _pick_first_column(events_df, TRIAL_COLUMN_CANDIDATES)
    if trial_column:
        canonical_name = (
            _canonical_covariate_name(trial_column, config=config) or trial_column
        )
        covariate_columns.append(trial_column)
        column_name_map[trial_column] = canonical_name


def _resolve_covariate_columns(
    events_df: pd.DataFrame,
    requested_covariates: Optional[List[str]],
    config: Any,
) -> Tuple[List[str], Dict[str, str]]:
    """Resolve which columns to use as covariates."""
    if config is None:
        raise ValueError("config is required")
    
    covariate_columns: List[str] = []
    column_name_map: Dict[str, str] = {}
    temperature_candidates = config.get("event_columns.temperature")

    if requested_covariates:
        for covariate_name in requested_covariates:
            _resolve_requested_covariate(
                covariate_name,
                events_df,
                config,
                temperature_candidates,
                covariate_columns,
                column_name_map,
            )
    else:
        _resolve_default_covariates(
            events_df,
            config,
            temperature_candidates,
            covariate_columns,
            column_name_map,
        )

    return covariate_columns, column_name_map


def _build_covariate_dataframe(
    events_df: pd.DataFrame,
    covariate_columns: List[str],
    column_name_map: Dict[str, str],
) -> Optional[pd.DataFrame]:
    """Build DataFrame from resolved covariate columns."""
    covariates_df = pd.DataFrame()
    for column_name in covariate_columns:
        if column_name in events_df.columns:
            canonical_name = column_name_map.get(column_name, column_name)
            covariates_df[canonical_name] = pd.to_numeric(
                events_df[column_name], errors="coerce"
            )
    
    return None if covariates_df.empty else covariates_df


def _remove_temperature_column(
    covariates_df: pd.DataFrame,
    temperature_column: Optional[str],
    config: Optional[Any],
) -> Optional[pd.DataFrame]:
    """Remove temperature column from covariates DataFrame."""
    if temperature_column is None:
        return covariates_df.copy()
    
    columns_to_drop = []
    if temperature_column in covariates_df.columns:
        columns_to_drop.append(temperature_column)
    
    temperature_canonical = _canonical_covariate_name(temperature_column, config=config)
    if (
        temperature_canonical
        and temperature_canonical in covariates_df.columns
        and temperature_canonical != temperature_column
    ):
        columns_to_drop.append(temperature_canonical)
    
    if not columns_to_drop:
        return covariates_df.copy()
    
    result = covariates_df.drop(columns=columns_to_drop, errors="ignore")
    return None if result.empty else result


def _build_covariate_matrices(
    events_df: Optional[pd.DataFrame],
    requested_covariates: Optional[List[str]],
    temperature_column: Optional[str],
    config: Optional[Any] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Build covariate matrices for analysis."""
    if events_df is None:
        return None, None

    if config is None:
        config = _load_config_safely()
        if config is None:
            return None, None

    covariate_columns, column_name_map = _resolve_covariate_columns(
        events_df, requested_covariates, config
    )
    
    if not covariate_columns:
        return None, None

    covariates_df = _build_covariate_dataframe(
        events_df, covariate_columns, column_name_map
    )
    
    if covariates_df is None:
        return None, None

    covariates_without_temp = _remove_temperature_column(
        covariates_df, temperature_column, config
    )

    return covariates_df, covariates_without_temp


def build_covariate_matrix(
    aligned_events: Optional[pd.DataFrame],
    requested_covariates: Optional[List[str]],
    config: Optional[Any] = None,
) -> Optional[pd.DataFrame]:
    """Build covariate matrix from aligned events.
    
    Parameters
    ----------
    aligned_events : DataFrame or None
        Events DataFrame with aligned data
    requested_covariates : list of str or None
        Covariate names to include. If None, uses defaults (temperature, trial).
    config : Any, optional
        Configuration object. If None, attempts to load from file.
        
    Returns
    -------
    DataFrame or None
        Covariate matrix with canonical column names, or None if unavailable.
    """
    if aligned_events is None:
        return None
    
    _, temperature_column = extract_temperature_data(aligned_events, config)
    covariates_df, _ = _build_covariate_matrices(
        aligned_events, requested_covariates, temperature_column, config
    )
    return covariates_df


def build_covariates_without_temp(
    covariates_df: Optional[pd.DataFrame],
    temperature_column: Optional[str],
    config: Optional[Any] = None,
) -> Optional[pd.DataFrame]:
    """Build covariate matrix excluding temperature column.
    
    Parameters
    ----------
    covariates_df : DataFrame or None
        Full covariate matrix
    temperature_column : str or None
        Name of temperature column to exclude
    config : Any, optional
        Configuration object for canonical name resolution
        
    Returns
    -------
    DataFrame or None
        Covariates without temperature, or None if empty
    """
    if covariates_df is None or covariates_df.empty:
        return None
    
    return _remove_temperature_column(covariates_df, temperature_column, config)


__all__ = [
    "extract_temperature_data",
    "extract_default_covariates",
    "build_covariate_matrix",
    "build_covariates_without_temp",
]
