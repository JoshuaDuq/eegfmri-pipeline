"""
Covariate Extraction Utilities
==============================

Functions for extracting and managing alignment covariates (e.g., temperature, trials).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Dict, Any, Tuple

import pandas as pd

from ..config.loader import load_settings


###################################################################
# Covariate Helpers
###################################################################


def _pick_first_column(df: Optional[pd.DataFrame], candidates: List[str]) -> Optional[str]:
    """Helper to pick first matching column from specific candidates."""
    if df is None or not candidates:
        return None
    for cand in candidates:
        if cand in df.columns:
            return cand
    return None


def _canonical_covariate_name(name: Optional[str], config=None) -> Optional[str]:
    """Resolve covariate name to canonical form (temperature, trial, etc.)."""
    if name is None:
        return None
    
    n = str(name).lower()
    
    if config is None:
        try:
            config = load_settings()
        except (OSError, ValueError):
            # Config file not found or invalid - use defaults
            config = None
    
    temp_aliases = {"stimulus_temp", "stimulus_temperature", "temp", "temperature"}
    trial_aliases = {"trial", "trial_number", "trial_index", "run", "block"}
    
    if config is not None:
        # Resolve config aliases if available
        if hasattr(config, "get"):
            temp_cols = config.get("event_columns.temperature", [])
            temp_aliases.update(str(c).lower() for c in temp_cols)
    
    if n in temp_aliases:
        return "temperature"
    if n in trial_aliases:
        return "trial"
    
    return n


###################################################################
# Extraction Functions
###################################################################


def extract_default_covariates(events_df: pd.DataFrame, config) -> List[str]:
    """Extract default covariates (temperature, trial) from events."""
    covariates = []
    
    temperature_columns = config.get("event_columns.temperature")
    temperature_column = _pick_first_column(events_df, temperature_columns)
    if temperature_column:
        covariates.append(temperature_column)
    
    trial_column_candidates = ["trial", "trial_number", "trial_index", "run", "block"]
    trial_col = _pick_first_column(events_df, trial_column_candidates)
    if trial_col:
        covariates.append(trial_col)
    
    return covariates


def extract_temperature_data(
    aligned_events: Optional[pd.DataFrame],
    config,
) -> Tuple[Optional[pd.Series], Optional[str]]:
    """Extract temperature series and column name from aligned events."""
    if aligned_events is None:
        return None, None
    
    psych_temp_columns = config.get("event_columns.temperature")
    temp_col = _pick_first_column(aligned_events, psych_temp_columns)
    if not temp_col:
        return None, None
    
    temp_series = pd.to_numeric(aligned_events[temp_col], errors="coerce")
    return temp_series, temp_col


###################################################################
# Matrix Building
###################################################################


def _add_covariate_column(
    covariate_columns: List[str],
    column_name_map: Dict[str, str],
    col_name: str,
    canonical_name: str
) -> None:
    covariate_columns.append(col_name)
    column_name_map[col_name] = canonical_name


def _resolve_covariate_columns(
    df_events: pd.DataFrame,
    partial_covars: Optional[List[str]],
    config: Optional[Any],
) -> Tuple[List[str], Dict[str, str]]:
    """Resolve which columns to use as covariates."""
    covariate_columns: List[str] = []
    column_name_map: Dict[str, str] = {}
    
    if config is None:
        raise ValueError("config is required")
    
    temperature_candidates = config.get("event_columns.temperature")

    if partial_covars:
        for covariate in partial_covars:
            if covariate in df_events.columns:
                canonical_name = _canonical_covariate_name(covariate, config=config) or covariate
                _add_covariate_column(covariate_columns, column_name_map, covariate, canonical_name)
                continue
            
            # Try to resolve semantic name
            canonical_name = _canonical_covariate_name(covariate, config=config)
            if canonical_name == "temperature":
                temp_column = _pick_first_column(df_events, temperature_candidates)
                if temp_column:
                    _add_covariate_column(covariate_columns, column_name_map, temp_column, canonical_name)
    else:
        # Default behavior: grab temperature and trial index
        temp_column = _pick_first_column(df_events, temperature_candidates)
        if temp_column:
            _add_covariate_column(covariate_columns, column_name_map, temp_column, "temperature")
        
        trial_candidates = ["trial", "trial_number", "trial_index", "run", "block"]
        for candidate in trial_candidates:
            if candidate in df_events.columns:
                canonical_name = _canonical_covariate_name(candidate, config=config) or candidate
                _add_covariate_column(covariate_columns, column_name_map, candidate, canonical_name)
                break

    return covariate_columns, column_name_map


def _build_covariate_matrices(
    df_events: Optional[pd.DataFrame],
    partial_covars: Optional[List[str]],
    temp_col: Optional[str],
    config: Optional[Any] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Build covariate matrices for analysis."""
    if df_events is None:
        return None, None

    if config is None:
        try:
            config = load_settings()
        except (OSError, ValueError):
            # Config file not found or invalid - proceed without config-based covariate resolution
            config = None

    covariate_columns, column_name_map = _resolve_covariate_columns(df_events, partial_covars, config)
    
    if not covariate_columns:
        return None, None

    covariates_df = pd.DataFrame()
    for covariate in covariate_columns:
        if covariate in df_events.columns:
            canonical_name = column_name_map.get(covariate, covariate)
            covariates_df[canonical_name] = pd.to_numeric(df_events[covariate], errors="coerce")

    if covariates_df.empty:
        return None, None

    temp_canonical = _canonical_covariate_name(temp_col, config=config) if temp_col else None
    
    # Create version without temperature (for partial correlation control)
    if temp_canonical and temp_canonical in covariates_df.columns:
        covariates_without_temp = covariates_df.drop(columns=[temp_canonical], errors="ignore")
    else:
        covariates_without_temp = covariates_df.copy()
    
    if covariates_without_temp.shape[1] == 0:
        covariates_without_temp = None

    return covariates_df, covariates_without_temp
