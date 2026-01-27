"""
Visualization Statistics
========================

Statistics for visualization and plotting, including diagnostic plots
for permutation distributions, cluster-mass histograms, and p-p plots.
"""

from __future__ import annotations

from typing import Dict, List, Union

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde



# Constants
DEFAULT_CORRELATION_VMAX = 0.5


###################################################################
# Core Visualization Statistics
###################################################################


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
            return DEFAULT_CORRELATION_VMAX
        return max(abs(np.min(finite_vals)), abs(np.max(finite_vals)))
    
    all_corrs = []
    for bd in data:
        correlations = bd['correlations']
        sig_mask = bd.get('significant_mask', np.ones(len(correlations), dtype=bool))
        sig_corrs = correlations[sig_mask]
        finite_sig = sig_corrs[np.isfinite(sig_corrs)]
        if len(finite_sig) > 0:
            all_corrs.extend(finite_sig)
    
    if not all_corrs:
        for bd in data:
            correlations = bd['correlations']
            finite_corrs = correlations[np.isfinite(correlations)]
            all_corrs.extend(finite_corrs)
    
    if not all_corrs:
        return DEFAULT_CORRELATION_VMAX
    
    return max(abs(np.min(all_corrs)), abs(np.max(all_corrs)))






