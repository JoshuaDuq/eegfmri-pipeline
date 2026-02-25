"""
Covariate Extraction Utilities
==============================

Functions for extracting and managing alignment covariates (e.g., predictor, trials).
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Dict, Any, Tuple

import pandas as pd

from ..config.loader import load_config


###################################################################
# Constants
###################################################################


# WARNING: Order matters! Prefer true within-run trial indices over run/block IDs.
# Using run/block as "trial order" is statistically meaningless (categorical coerced to numeric).
TRIAL_COLUMN_CANDIDATES = ["trial_index", "trial_number", "trial"]

# These are NOT valid trial order covariates - they are categorical grouping variables
RUN_BLOCK_COLUMNS = ["run", "block", "run_number", "block_number"]

# Canonical predictor alias plus configurable event column aliases.
PREDICTOR_ALIASES = {"predictor"}

# Only true within-run trial indices should be used as order covariates
TRIAL_ALIASES = {"trial", "trial_number", "trial_index"}


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
    """Resolve covariate name to canonical form (predictor, trial, etc.)."""
    if name is None:
        return None
    
    normalized_name = str(name).lower()
    
    if config is None:
        config = _load_config_safely()
    
    predictor_aliases = PREDICTOR_ALIASES.copy()
    trial_aliases = TRIAL_ALIASES.copy()
    
    if config is not None and hasattr(config, "get"):
        config_pred_cols = config.get("event_columns.predictor", [])
        predictor_aliases.update(str(col).lower() for col in config_pred_cols)
        explicit_predictor = config.get("behavior_analysis.predictor_column", None)
        if explicit_predictor is not None:
            explicit_predictor_norm = str(explicit_predictor).strip().lower()
            if explicit_predictor_norm:
                predictor_aliases.add(explicit_predictor_norm)
    
    if normalized_name in predictor_aliases:
        return "predictor"
    if normalized_name in trial_aliases:
        return "trial"
    
    return normalized_name


###################################################################
# Extraction Functions
###################################################################


def extract_predictor_data(
    aligned_events: Optional[pd.DataFrame],
    config: Any,
) -> Tuple[Optional[pd.Series], Optional[str]]:
    """Extract predictor series and column name from aligned events."""
    if aligned_events is None:
        return None, None

    from eeg_pipeline.utils.data.columns import resolve_predictor_column

    predictor_column = resolve_predictor_column(aligned_events, config)
    if predictor_column is None or predictor_column not in aligned_events.columns:
        return None, None

    predictor_series = pd.to_numeric(
        aligned_events[predictor_column], errors="coerce"
    )
    return predictor_series, predictor_column


###################################################################
# Matrix Building
###################################################################


def _resolve_requested_covariate(
    covariate_name: str,
    events_df: pd.DataFrame,
    config: Any,
    predictor_candidates: List[str],
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
    if canonical_name == "predictor":
        predictor_column = _pick_first_column(events_df, predictor_candidates)
        if predictor_column:
            covariate_columns.append(predictor_column)
            column_name_map[predictor_column] = canonical_name


def _resolve_default_covariates(
    events_df: pd.DataFrame,
    config: Any,
    predictor_candidates: List[str],
    covariate_columns: List[str],
    column_name_map: Dict[str, str],
) -> None:
    """Resolve default covariates (predictor and trial)."""
    predictor_column = _pick_first_column(events_df, predictor_candidates)
    if predictor_column:
        covariate_columns.append(predictor_column)
        column_name_map[predictor_column] = "predictor"
    
    trial_column = _pick_first_column(events_df, TRIAL_COLUMN_CANDIDATES)
    if trial_column:
        canonical_name = (
            _canonical_covariate_name(trial_column, config=config) or trial_column
        )
        covariate_columns.append(trial_column)
        column_name_map[trial_column] = canonical_name
    else:
        run_block_col = _pick_first_column(events_df, RUN_BLOCK_COLUMNS)
        if run_block_col:
            warnings.warn(
                f"No true trial index column found (tried: {TRIAL_COLUMN_CANDIDATES}). "
                f"Found '{run_block_col}' but run/block IDs are categorical grouping variables, "
                f"not valid trial order covariates. Using them as numeric covariates is "
                f"statistically meaningless. Add a true within-run trial index column.",
                UserWarning,
                stacklevel=3,
            )


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
    predictor_candidates = config.get("event_columns.predictor")

    if requested_covariates:
        for covariate_name in requested_covariates:
            _resolve_requested_covariate(
                covariate_name,
                events_df,
                config,
                predictor_candidates,
                covariate_columns,
                column_name_map,
            )
    else:
        _resolve_default_covariates(
            events_df,
            config,
            predictor_candidates,
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


def _remove_predictor_column(
    covariates_df: pd.DataFrame,
    predictor_column: Optional[str],
    config: Optional[Any],
) -> Optional[pd.DataFrame]:
    """Remove predictor column from covariates DataFrame."""
    if predictor_column is None:
        return covariates_df.copy()
    
    columns_to_drop = {predictor_column}
    predictor_canonical = _canonical_covariate_name(predictor_column, config=config)
    if predictor_canonical and predictor_canonical != predictor_column:
        columns_to_drop.add(predictor_canonical)
    
    columns_to_drop = [col for col in columns_to_drop if col in covariates_df.columns]
    if not columns_to_drop:
        return covariates_df.copy()
    
    result = covariates_df.drop(columns=columns_to_drop, errors="ignore")
    return None if result.empty else result


def _build_covariate_matrices(
    events_df: Optional[pd.DataFrame],
    requested_covariates: Optional[List[str]],
    predictor_column: Optional[str],
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

    covariates_without_predictor = _remove_predictor_column(
        covariates_df, predictor_column, config
    )

    return covariates_df, covariates_without_predictor


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
        Covariate names to include. If None, uses defaults (predictor, trial).
    config : Any, optional
        Configuration object. If None, attempts to load from file.
        
    Returns
    -------
    DataFrame or None
        Covariate matrix with canonical column names, or None if unavailable.
    """
    if aligned_events is None:
        return None
    
    _, predictor_column = extract_predictor_data(aligned_events, config)
    covariates_df, _ = _build_covariate_matrices(
        aligned_events, requested_covariates, predictor_column, config
    )
    return covariates_df


def build_covariates_without_predictor(
    covariates_df: Optional[pd.DataFrame],
    predictor_column: Optional[str],
    config: Optional[Any] = None,
) -> Optional[pd.DataFrame]:
    """Build covariate matrix excluding predictor column.
    
    Parameters
    ----------
    covariates_df : DataFrame or None
        Full covariate matrix
    predictor_column : str or None
        Name of predictor column to exclude
    config : Any, optional
        Configuration object for canonical name resolution
        
    Returns
    -------
    DataFrame or None
        Covariates without predictor, or None if empty
    """
    if covariates_df is None or covariates_df.empty:
        return None
    
    return _remove_predictor_column(covariates_df, predictor_column, config)


__all__ = [
    "extract_predictor_data",
    "build_covariate_matrix",
    "build_covariates_without_predictor",
]
