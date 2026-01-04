"""
Event analysis utilities.

Functions for extracting specific event types and masks from event DataFrames.
"""

from typing import Any, Optional
import numpy as np
import pandas as pd

from eeg_pipeline.utils.data.columns import find_column_in_events, get_pain_column_from_config


def extract_pain_mask(events_df: pd.DataFrame, config: Any = None) -> Optional[np.ndarray]:
    """Extract boolean mask for pain condition (1/True/High = Pain).
    
    Args:
        events_df: Events DataFrame.
        config: Pipeline configuration object (optional).
        
    Returns:
        Boolean numpy array mask where True indicates Pain condition, 
        or None if pain column cannot be identified.
    """
    col = None
    if config:
        try:
            col = get_pain_column_from_config(config, events_df)
        except (ValueError, KeyError):
            pass
    
    # Fallback to heuristics if not found via config or config not provided
    if not col:
         # Simplified heuristic logic from previous implementations
        candidates = ['pain_stimulus', 'pain', 'is_pain', 'condition', 'stimulus_type']
        for c in candidates:
            if c in events_df.columns:
                col = c
                break
    
    if not col:
        return None
        
    # Check values
    try:
        vals = pd.to_numeric(events_df[col], errors='coerce')
        if not pd.isna(vals).all():
            # Numeric: 1 vs 0 (assuming 1 is pain/high)
            # Or high vs low values? Let's assume binary 0/1 for now as per previous logic
            return (vals == 1).values
    except Exception:
        pass
        
    # String matching
    vals_str = events_df[col].astype(str).str.lower()
    return vals_str.isin(['pain', 'high', 'hot', 'stimulus']).values


def extract_comparison_mask(events_df: pd.DataFrame, config: Any) -> Optional[tuple[np.ndarray, np.ndarray, str, str]]:
    """Extract dual masks for flexible comparison based on config.
    
    Returns:
        tuple of (mask1, mask2, label1, label2) or None.
    """
    from eeg_pipeline.utils.config.loader import get_config_value
    
    compare_cols = get_config_value(config, "plotting.comparisons.compare_columns", False)
    if not compare_cols:
        return None
        
    col = get_config_value(config, "plotting.comparisons.comparison_column", None)
    vals_spec = get_config_value(config, "plotting.comparisons.comparison_values", [])
    
    if not col or not vals_spec or len(vals_spec) < 2:
        return None
        
    if col not in events_df.columns:
        return None
        
    v1_str, v2_str = str(vals_spec[0]), str(vals_spec[1])
    
    # Try numeric match first
    try:
        col_vals = pd.to_numeric(events_df[col], errors='coerce')
        v1_num = pd.to_numeric(v1_str, errors='coerce')
        v2_num = pd.to_numeric(v2_str, errors='coerce')
        
        if not np.isnan(v1_num) and not np.isnan(v2_num):
            m1 = (col_vals == v1_num).values
            m2 = (col_vals == v2_num).values
            if np.any(m1) or np.any(m2):
                return m1, m2, v1_str, v2_str
    except Exception:
        pass
        
    # String match fallback
    col_vals_str = events_df[col].astype(str)
    m1 = (col_vals_str == v1_str).values
    m2 = (col_vals_str == v2_str).values
    
    if not np.any(m1) and not np.any(m2):
        return None
        
    return m1, m2, v1_str, v2_str
