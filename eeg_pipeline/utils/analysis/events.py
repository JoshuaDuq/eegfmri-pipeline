"""
Event analysis utilities.

Functions for extracting specific event types and masks from event DataFrames.
"""

from typing import Any, Optional
import numpy as np
import pandas as pd

def _resolve_comparison_column(events_df: pd.DataFrame, config: Any) -> str:
    from eeg_pipeline.utils.config.loader import get_config_value

    col = get_config_value(config, "plotting.comparisons.comparison_column", None)
    col = str(col).strip() if col is not None else ""
    return col


def _resolve_comparison_values(config: Any) -> tuple[Any, Any]:
    from eeg_pipeline.utils.config.loader import get_config_value

    vals_spec = get_config_value(config, "plotting.comparisons.comparison_values", []) or []
    if not isinstance(vals_spec, (list, tuple)) or len(vals_spec) < 2:
        return 0, 1
    return vals_spec[0], vals_spec[1]


def _resolve_comparison_labels(config: Any, *, v1: Any, v2: Any) -> tuple[str, str]:
    """Return display labels for a 2-level comparison (value1/value2)."""
    from eeg_pipeline.utils.config.loader import get_config_value

    labels_spec = get_config_value(config, "plotting.comparisons.comparison_labels", None)
    if isinstance(labels_spec, (list, tuple)) and len(labels_spec) >= 2:
        l1 = str(labels_spec[0]).strip()
        l2 = str(labels_spec[1]).strip()
        if l1 and l2:
            return l1, l2

    return str(v1), str(v2)


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
    label1, label2 = _resolve_comparison_labels(config, v1=v1, v2=v2)
    
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
    label1, label2 = _resolve_comparison_labels(config, v1=v1, v2=v2)
    return col, v1, v2, label1, label2
