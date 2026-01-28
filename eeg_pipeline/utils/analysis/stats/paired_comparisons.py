"""
Paired comparison: Cohen's d for paired samples.
"""

from __future__ import annotations

import numpy as np


MIN_STD_FOR_COHENS_D = 1e-10


def compute_paired_cohens_d(before: np.ndarray, after: np.ndarray) -> float:
    """Compute Cohen's d for paired samples using difference scores."""
    before = np.asarray(before).ravel()
    after = np.asarray(after).ravel()
    
    valid = np.isfinite(before) & np.isfinite(after)
    before = before[valid]
    after = after[valid]
    
    if len(before) < 2:
        return np.nan
    
    diff = after - before
    std_diff = np.std(diff, ddof=1)
    
    if std_diff < MIN_STD_FOR_COHENS_D:
        return 0.0
    
    return float(np.mean(diff) / std_diff)


