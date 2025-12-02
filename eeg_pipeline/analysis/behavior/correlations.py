"""
Analysis configuration and correlation statistics helpers.

This module provides:
- AnalysisConfig: Lightweight config dataclass for analysis functions
- Helper functions for building temperature records and ROI statistics

All core statistical functions are in eeg_pipeline.utils.analysis.stats
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

from eeg_pipeline.utils.analysis.stats import (
    prepare_aligned_data,
    compute_correlation,
    compute_bootstrap_ci,
    compute_partial_correlations,
    compute_permutation_pvalues,
    compute_temp_permutation_pvalues,
    CorrelationStats,
)

if TYPE_CHECKING:
    from eeg_pipeline.analysis.behavior.core import BehaviorContext


@dataclass
class AnalysisConfig:
    """Lightweight configuration extracted from BehaviorContext."""
    subject: str
    config: Any
    logger: Any
    rng: np.random.Generator
    stats_dir: Optional[Path] = None
    bootstrap: int = 0
    n_perm: int = 0
    use_spearman: bool = True
    method: str = "spearman"
    min_samples_channel: int = 10
    min_samples_roi: int = 20
    groups: Optional[np.ndarray] = None

    @classmethod
    def from_context(cls, ctx: "BehaviorContext") -> "AnalysisConfig":
        return cls(
            subject=ctx.subject,
            config=ctx.config,
            logger=ctx.logger,
            rng=ctx.rng or np.random.default_rng(42),
            stats_dir=ctx.stats_dir,
            bootstrap=ctx.bootstrap,
            n_perm=ctx.n_perm,
            use_spearman=ctx.use_spearman,
            method=ctx.method,
            min_samples_channel=ctx.min_samples_channel,
            min_samples_roi=ctx.min_samples_roi,
        )


def _align_groups_to_series(
    series: pd.Series,
    groups: Optional[Union[pd.Series, np.ndarray]]
) -> Optional[np.ndarray]:
    """Align group labels to a pandas Series index."""
    if groups is None:
        return None
    try:
        if isinstance(groups, pd.Series):
            return groups.loc[series.index].to_numpy()
        arr = np.asarray(groups)
        return arr if arr.size == len(series) else None
    except Exception:
        return None


def _build_temp_record_unified(
    x: pd.Series,
    temp: Optional[pd.Series],
    cov_no_temp: Optional[pd.DataFrame],
    identifier: str,
    id_key: str,
    band: str,
    cfg: AnalysisConfig,
    groups: Optional[np.ndarray] = None,
    **extra
) -> Optional[Dict[str, Any]]:
    """Build a temperature correlation record with optional bootstrap/permutation."""
    if temp is None or (hasattr(temp, 'empty') and temp.empty):
        return None
    
    from eeg_pipeline.analysis.behavior.core import safe_correlation, build_correlation_record
    
    min_s = cfg.min_samples_channel if id_key == "channel" else cfg.min_samples_roi
    x_a, temp_a, cov_a, _, _ = prepare_aligned_data(x, temp, cov_no_temp)
    
    if len(x_a) == 0 or len(temp_a) == 0:
        return None
    
    grp = _align_groups_to_series(x_a, groups if groups is not None else cfg.groups)
    r, p, _ = safe_correlation(x_a, temp_a, cfg.method, min_s)
    
    ci_lo, ci_hi = np.nan, np.nan
    p_perm = np.nan
    
    if cfg.bootstrap > 0:
        ci_lo, ci_hi = compute_bootstrap_ci(
            x_a, temp_a, cfg.bootstrap, 0.95,
            "spearman" if cfg.use_spearman else "pearson", cfg.rng
        )
    
    if cfg.n_perm > 0:
        p_perm, _ = compute_temp_permutation_pvalues(
            x_a, temp_a, cov_a, cfg.method, cfg.n_perm, cfg.rng,
            band, identifier, cfg.logger, groups=grp
        )
    
    return build_correlation_record(
        identifier, band, r, p, len(x_a), cfg.method,
                                    ci_low=ci_lo, ci_high=ci_hi, p_perm=p_perm,
        identifier_type=id_key, **extra
    ).to_dict()


def _compute_roi_correlation_stats(
    x: pd.Series,
    y: pd.Series,
    x_a: np.ndarray,
    y_a: np.ndarray,
    cov: Optional[pd.DataFrame],
    temp: Optional[pd.Series],
    n_eff: int,
    band: str,
    roi: str,
    context: str,
    cfg: AnalysisConfig,
    groups: Optional[np.ndarray] = None,
    me_records: Optional[List[Dict]] = None,
) -> CorrelationStats:
    """Compute comprehensive correlation statistics for an ROI."""
    r, p = compute_correlation(x_a, y_a, cfg.use_spearman)
    
    r_part, p_part, n_part, r_part_temp, p_part_temp, n_part_temp = compute_partial_correlations(
        x, y, cov, temp, cfg.method, context, cfg.logger, cfg.min_samples_roi
    )
    
    ci_lo, ci_hi = compute_bootstrap_ci(
        x_a, y_a, cfg.bootstrap, 0.95,
        "spearman" if cfg.use_spearman else "pearson", cfg.rng
    )
    
    x_series = pd.Series(x_a) if not isinstance(x_a, pd.Series) else x_a
    y_series = pd.Series(y_a) if not isinstance(y_a, pd.Series) else y_a
    
    p_perm, p_part_perm, p_part_temp_perm = compute_permutation_pvalues(
        x_series, y_series, cov, temp, cfg.method, cfg.n_perm, n_eff, cfg.rng,
        band, roi, groups=groups
    )
    
    return CorrelationStats(
        correlation=r,
        p_value=p,
        ci_low=ci_lo,
        ci_high=ci_hi,
        r_partial=r_part,
        p_partial=p_part,
        n_partial=n_part,
        r_partial_temp=r_part_temp,
        p_partial_temp=p_part_temp,
        n_partial_temp=n_part_temp,
        p_perm=p_perm,
        p_partial_perm=p_part_perm,
        p_partial_temp_perm=p_part_temp_perm,
    )
