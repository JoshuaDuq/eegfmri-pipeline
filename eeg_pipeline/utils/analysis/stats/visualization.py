"""
Visualization Statistics
========================

Statistics for visualization and plotting.
"""

from __future__ import annotations

from typing import Dict, List, Union

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


def compute_kde_scale(
    data: pd.Series,
    hist_bins: int = 15,
    kde_points: int = 100,
) -> float:
    """Compute KDE scaling factor for overlay on histogram."""
    hist_counts, _ = np.histogram(data, bins=hist_bins)
    kde = gaussian_kde(data)
    data_range = np.linspace(data.min(), data.max(), kde_points)
    kde_vals = kde(data_range)
    if kde_vals.max() > 0:
        return hist_counts.max() / kde_vals.max()
    return 1.0


def compute_correlation_vmax(data: Union[np.ndarray, List[Dict]]) -> float:
    """Compute symmetric vmax for correlation heatmaps."""
    if isinstance(data, np.ndarray):
        finite_vals = data[np.isfinite(data)]
        if len(finite_vals) == 0:
            return 0.5
        return max(abs(np.min(finite_vals)), abs(np.max(finite_vals)))
    
    all_corrs = []
    for bd in data:
        sig_mask = bd.get('significant_mask', np.ones(len(bd['correlations']), dtype=bool))
        sig_corrs = bd['correlations'][sig_mask]
        all_corrs.extend(sig_corrs[np.isfinite(sig_corrs)])
    
    if all_corrs:
        return max(abs(np.min(all_corrs)), abs(np.max(all_corrs)))
    
    all_corrs = []
    for bd in data:
        all_corrs.extend(bd['correlations'][np.isfinite(bd['correlations'])])
    return max(abs(np.min(all_corrs)), abs(np.max(all_corrs))) if all_corrs else 0.5





