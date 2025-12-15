"""
Event analysis utilities.

Functions for extracting specific event types and masks from event DataFrames.
"""

from typing import Any, Optional
import numpy as np
import pandas as pd

from eeg_pipeline.io.columns import find_column_in_events, get_pain_column_from_config


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
