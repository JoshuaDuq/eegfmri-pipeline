"""
Visualization Statistics
========================

Statistics for visualization and plotting, including diagnostic plots
for permutation distributions, cluster-mass histograms, and p-p plots.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde



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
