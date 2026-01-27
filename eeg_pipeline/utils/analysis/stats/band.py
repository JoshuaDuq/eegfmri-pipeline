"""
Band Statistics
===============

Inter-band correlations and power statistics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from eeg_pipeline.utils.config.loader import get_fisher_z_clip_values
from eeg_pipeline.domain.features.naming import NamingSchema

from .correlation import compute_correlation, fisher_z_transform_mean


def compute_band_correlations(
    pow_df: pd.DataFrame,
    y: pd.Series,
    band: str,
    min_samples: int = 3,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Compute channel-wise correlations for a band."""
    band_lower = str(band).lower()

    band_columns: List[str] = []
    channel_names: List[str] = []
    for col in pow_df.columns:
        parsed = NamingSchema.parse(str(col))
        if not parsed.get("valid"):
            continue
        if str(parsed.get("group", "")).lower() != "power":
            continue
        if str(parsed.get("scope", "")).lower() != "ch":
            continue
        if str(parsed.get("band", "")).lower() != band_lower:
            continue
        channel = parsed.get("identifier")
        if not channel:
            continue
        band_columns.append(str(col))
        channel_names.append(str(channel))

    if not band_columns:
        return [], np.array([]), np.array([])
    
    correlations = []
    p_values = []
    
    for col in band_columns:
        x_values = pow_df[col].to_numpy()
        y_values = y.to_numpy()
        
        valid_mask = np.isfinite(x_values) & np.isfinite(y_values)
        x_valid = x_values[valid_mask]
        y_valid = y_values[valid_mask]
        
        if len(x_valid) < min_samples:
            correlation = np.nan
            p_value = 1.0
        else:
            correlation, _ = compute_correlation(x_valid, y_valid, method="spearman")
            if np.isfinite(correlation):
                _, p_value = stats.spearmanr(x_valid, y_valid)
            else:
                p_value = 1.0
        
        correlations.append(correlation)
        p_values.append(p_value)
    
    return channel_names, np.array(correlations), np.array(p_values)
