"""
Event analysis utilities.

Functions for extracting specific event types and masks from event DataFrames.
"""

from typing import Any, Optional
import numpy as np
import pandas as pd

from eeg_pipeline.utils.data.columns import get_pain_column_from_config


def extract_pain_mask(events_df: pd.DataFrame, config: Any = None) -> Optional[np.ndarray]:
    """Extract boolean mask for pain condition (1/True/High = Pain).
    
    Args:
        events_df: Events DataFrame.
        config: Pipeline configuration object (optional).
        
    Returns:
        Boolean numpy array mask where True indicates Pain condition, 
        or None if pain column cannot be identified.
    """
    if config is None:
        return None

    try:
        col = get_pain_column_from_config(config, events_df)
    except Exception:
        col = None

    if not col or col not in events_df.columns:
        return None

    # Prefer the plotting comparison values (value2 is treated as "pain/high") when available.
    from eeg_pipeline.utils.config.loader import get_config_value

    vals_spec = get_config_value(config, "plotting.comparisons.comparison_values", None)
    pain_value = None
    if isinstance(vals_spec, (list, tuple)) and len(vals_spec) >= 2:
        pain_value = vals_spec[1]
    if pain_value is None:
        pain_value = 1

    series = events_df[col]

    # Numeric match first
    try:
        series_num = pd.to_numeric(series, errors="coerce")
        pain_value_num = pd.to_numeric(str(pain_value), errors="coerce")
        if not np.isnan(pain_value_num) and not pd.isna(series_num).all():
            return (series_num == pain_value_num).to_numpy()
    except Exception:
        pass

    return (series.astype(str) == str(pain_value)).to_numpy()


def _resolve_comparison_column(events_df: pd.DataFrame, config: Any) -> str:
    from eeg_pipeline.utils.config.loader import get_config_value

    col = get_config_value(config, "plotting.comparisons.comparison_column", None)
    col = str(col).strip() if col is not None else ""
    if col:
        return col

    try:
        return str(get_pain_column_from_config(config, events_df) or "").strip()
    except Exception:
        return ""


def _resolve_comparison_values(config: Any) -> tuple[Any, Any]:
    from eeg_pipeline.utils.config.loader import get_config_value

    vals_spec = get_config_value(config, "plotting.comparisons.comparison_values", []) or []
    if not isinstance(vals_spec, (list, tuple)) or len(vals_spec) < 2:
        return 0, 1
    return vals_spec[0], vals_spec[1]


def _resolve_default_labels(config: Any, *, col: str, v1: Any, v2: Any) -> tuple[str, str]:
    """Return display labels for a 2-level comparison."""
    from eeg_pipeline.utils.config.loader import get_config_value

    # If this is the configured pain-binary column and values look like 0/1, keep the legacy labels.
    pain_candidates = get_config_value(config, "event_columns.pain_binary", []) or []
    is_pain_binary = bool(col) and (col in pain_candidates)

    v1s, v2s = str(v1), str(v2)
    if is_pain_binary:
        if (v1s, v2s) == ("0", "1"):
            return "Non-pain", "Pain"
        if (v1s, v2s) == ("1", "0"):
            return "Pain", "Non-pain"

    return v1s, v2s


def extract_comparison_mask(
    events_df: pd.DataFrame,
    config: Any,
    *,
    require_enabled: bool = True,
) -> Optional[tuple[np.ndarray, np.ndarray, str, str]]:
    """Extract dual masks for flexible comparison based on config.
    
    Returns:
        tuple of (mask1, mask2, label1, label2) or None.
    """
    from eeg_pipeline.utils.config.loader import get_config_value
    
    compare_cols = bool(get_config_value(config, "plotting.comparisons.compare_columns", False))
    if require_enabled and not compare_cols:
        return None

    col = _resolve_comparison_column(events_df, config)

    if not col or col not in events_df.columns:
        return None

    v1, v2 = _resolve_comparison_values(config)
    v1_str, v2_str = str(v1), str(v2)
    label1, label2 = _resolve_default_labels(config, col=col, v1=v1, v2=v2)
    
    # Try numeric match first
    try:
        col_vals = pd.to_numeric(events_df[col], errors='coerce')
        v1_num = pd.to_numeric(v1_str, errors='coerce')
        v2_num = pd.to_numeric(v2_str, errors='coerce')
        
        if not np.isnan(v1_num) and not np.isnan(v2_num):
            m1 = (col_vals == v1_num).values
            m2 = (col_vals == v2_num).values
            if np.any(m1) or np.any(m2):
                return m1, m2, label1, label2
    except Exception:
        pass
        
    # String match fallback
    col_vals_str = events_df[col].astype(str)
    m1 = (col_vals_str == v1_str).values
    m2 = (col_vals_str == v2_str).values
    
    if not np.any(m1) and not np.any(m2):
        return None
        
    return m1, m2, label1, label2


def resolve_comparison_spec(
    events_df: pd.DataFrame,
    config: Any,
    *,
    require_enabled: bool = True,
) -> Optional[tuple[str, Any, Any, str, str]]:
    """Resolve (column, value1, value2, label1, label2) for a configured comparison."""
    from eeg_pipeline.utils.config.loader import get_config_value

    compare_cols = bool(get_config_value(config, "plotting.comparisons.compare_columns", False))
    if require_enabled and not compare_cols:
        return None

    col = _resolve_comparison_column(events_df, config)
    if not col or col not in events_df.columns:
        return None

    v1, v2 = _resolve_comparison_values(config)
    label1, label2 = _resolve_default_labels(config, col=col, v1=v1, v2=v2)
    return col, v1, v2, label1, label2
