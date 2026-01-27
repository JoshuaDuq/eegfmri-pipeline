"""
Paired Comparison Statistics
============================

Comprehensive paired comparison tests for time window and condition-based comparisons.
Computes Wilcoxon signed-rank tests (paired) and Mann-Whitney U tests (unpaired)
with FDR correction, effect sizes, and bootstrap confidence intervals.

This module pre-computes all paired comparisons that the plotting pipeline
previously computed on-the-fly, enabling faster plotting and consistent statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd
from scipy import stats

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.utils.analysis.stats.fdr import fdr_bh
from eeg_pipeline.utils.analysis.stats.effect_size import cohens_d, hedges_g
from eeg_pipeline.utils.analysis.stats.bootstrap import (
    bootstrap_mean_diff_ci,
)
from eeg_pipeline.utils.analysis.spatial import get_roi_definitions


###################################################################
# Constants
###################################################################


MIN_STD_FOR_COHENS_D = 1e-10
DEFAULT_FDR_ALPHA = 0.05
DEFAULT_N_BOOT = 1000
DEFAULT_N_PERM = 0
DEFAULT_MIN_SAMPLES_PAIRED = 5
DEFAULT_MIN_SAMPLES_UNPAIRED = 3

FEATURE_TYPE_GROUPS = {
    "connectivity": "conn",
    "ratios": "ratio",
    "asymmetry": "asym",
    "complexity": "comp",
}


###################################################################
# Data Classes
###################################################################


@dataclass
class PairedComparisonResult:
    """Result of a single paired comparison test."""
    feature_type: str
    comparison_type: str
    condition1: str
    condition2: str
    identifier: str
    n_pairs: int
    mean1: float
    mean2: float
    std1: float
    std2: float
    mean_diff: float
    statistic: float
    p_value: float
    effect_size_d: float
    effect_size_g: float
    ci_low: float
    ci_high: float
    test_method: str
    p_perm: float = np.nan  # Permutation p-value
    n_permutations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_type": self.feature_type,
            "comparison_type": self.comparison_type,
            "condition1": self.condition1,
            "condition2": self.condition2,
            "identifier": self.identifier,
            "n_pairs": self.n_pairs,
            "mean1": self.mean1,
            "mean2": self.mean2,
            "std1": self.std1,
            "std2": self.std2,
            "mean_diff": self.mean_diff,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "effect_size_d": self.effect_size_d,
            "effect_size_g": self.effect_size_g,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "test_method": self.test_method,
            "p_perm": self.p_perm,
            "n_permutations": self.n_permutations,
        }


@dataclass
class PairedComparisonSummary:
    """Summary of all paired comparison results."""
    results: List[PairedComparisonResult] = field(default_factory=list)
    n_tests: int = 0
    n_significant_raw: int = 0
    n_significant_fdr: int = 0
    fdr_alpha: float = 0.05
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()
        df = pd.DataFrame([r.to_dict() for r in self.results])
        return df


###################################################################
# Core Statistical Functions
###################################################################


def safe_wilcoxon(
    x: np.ndarray,
    y: np.ndarray,
    min_n: int = DEFAULT_MIN_SAMPLES_PAIRED,
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """Safely compute Wilcoxon signed-rank test for paired samples.
    
    Args:
        x: First paired sample
        y: Second paired sample
        min_n: Minimum samples required
        alternative: Alternative hypothesis
    
    Returns:
        Tuple of (statistic, p_value), or (np.nan, 1.0) if test fails
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    
    if len(x) != len(y) or len(x) < min_n:
        return np.nan, 1.0
    
    valid = np.isfinite(x) & np.isfinite(y)
    x_valid = x[valid]
    y_valid = y[valid]
    
    if len(x_valid) < min_n:
        return np.nan, 1.0
    
    diff = y_valid - x_valid
    if np.all(diff == 0):
        return np.nan, 1.0
    
    try:
        stat, p = stats.wilcoxon(x_valid, y_valid, alternative=alternative)
        return float(stat), float(p)
    except ValueError:
        return np.nan, 1.0


def safe_mannwhitneyu(
    group1: np.ndarray,
    group2: np.ndarray,
    min_n: int = DEFAULT_MIN_SAMPLES_UNPAIRED,
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """Safely compute Mann-Whitney U test for independent samples.
    
    Args:
        group1: First group values
        group2: Second group values
        min_n: Minimum samples required per group
        alternative: Alternative hypothesis
    
    Returns:
        Tuple of (statistic, p_value), or (np.nan, 1.0) if test fails
    """
    g1 = np.asarray(group1).ravel()
    g2 = np.asarray(group2).ravel()
    
    g1 = g1[np.isfinite(g1)]
    g2 = g2[np.isfinite(g2)]
    
    if len(g1) < min_n or len(g2) < min_n:
        return np.nan, 1.0
    
    try:
        stat, p = stats.mannwhitneyu(g1, g2, alternative=alternative)
        return float(stat), float(p)
    except ValueError:
        return np.nan, 1.0


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


def permutation_paired_pvalue(
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Compute permutation p-value for paired samples (sign-flip test).
    
    For paired samples, we randomly flip the signs of difference scores
    to create a null distribution under the hypothesis of no difference.
    
    Args:
        x: First paired sample
        y: Second paired sample
        n_perm: Number of permutations
        rng: Random number generator
    
    Returns:
        Two-tailed permutation p-value
    """
    if rng is None:
        rng = np.random.default_rng()
    
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    
    if len(x) != len(y):
        return np.nan
    
    valid = np.isfinite(x) & np.isfinite(y)
    diff = y[valid] - x[valid]
    n = len(diff)
    
    if n < DEFAULT_MIN_SAMPLES_PAIRED:
        return np.nan
    
    observed_stat = np.abs(np.mean(diff))
    
    null_stats = np.zeros(n_perm)
    for i in range(n_perm):
        signs = rng.choice([-1, 1], size=n)
        null_stats[i] = np.abs(np.mean(diff * signs))
    
    n_extreme = np.sum(null_stats >= observed_stat)
    p_perm = (n_extreme + 1) / (n_perm + 1)
    
    return float(p_perm)


def permutation_unpaired_pvalue(
    group1: np.ndarray,
    group2: np.ndarray,
    n_perm: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Compute permutation p-value for unpaired samples.
    
    For unpaired samples, we randomly shuffle group labels to create
    a null distribution under the hypothesis of no group difference.
    
    Args:
        group1: First group values
        group2: Second group values
        n_perm: Number of permutations
        rng: Random number generator
    
    Returns:
        Two-tailed permutation p-value
    """
    if rng is None:
        rng = np.random.default_rng()
    
    g1 = np.asarray(group1).ravel()
    g2 = np.asarray(group2).ravel()
    
    g1 = g1[np.isfinite(g1)]
    g2 = g2[np.isfinite(g2)]
    
    if len(g1) < DEFAULT_MIN_SAMPLES_UNPAIRED or len(g2) < DEFAULT_MIN_SAMPLES_UNPAIRED:
        return np.nan
    
    combined = np.concatenate([g1, g2])
    n1 = len(g1)
    n_total = len(combined)
    
    observed_stat = np.abs(np.mean(g2) - np.mean(g1))
    
    null_stats = np.zeros(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(combined)
        perm_g1 = shuffled[:n1]
        perm_g2 = shuffled[n1:]
        null_stats[i] = np.abs(np.mean(perm_g2) - np.mean(perm_g1))
    
    n_extreme = np.sum(null_stats >= observed_stat)
    p_perm = (n_extreme + 1) / (n_perm + 1)
    
    return float(p_perm)


###################################################################
# Feature Extraction Helpers
###################################################################


def get_channels_from_columns(columns: List[str]) -> List[str]:
    """Extract unique channel names from feature column names."""
    channels = set()
    for col in columns:
        parsed = NamingSchema.parse(str(col))
        if parsed.get("valid") and parsed.get("scope") == "ch":
            ch = parsed.get("identifier")
            if ch:
                channels.add(str(ch))
    return sorted(channels)


def get_roi_channels(
    roi_spec: Union[List[str], Dict[str, Any]],
    available_channels: List[str],
) -> List[str]:
    """Get channels belonging to an ROI."""
    if isinstance(roi_spec, list):
        return [ch for ch in roi_spec if ch in available_channels]
    if isinstance(roi_spec, dict):
        ch_list = roi_spec.get("channels", [])
        return [ch for ch in ch_list if ch in available_channels]
    return []


def collect_segment_data(
    df: pd.DataFrame,
    group: str,
    segment: str,
    band: str,
    roi_channels: Optional[List[str]] = None,
) -> pd.Series:
    """Collect and aggregate data for a specific segment/band/ROI combination."""
    cols = []
    roi_set = set(roi_channels) if roi_channels else None
    
    for c in df.columns:
        parsed = NamingSchema.parse(str(c))
        if not parsed.get("valid"):
            continue
        if parsed.get("group") != group:
            continue
        if parsed.get("segment") != segment:
            continue
        if parsed.get("band") != band:
            continue
        
        if roi_set:
            ch = parsed.get("identifier")
            if ch and ch not in roi_set:
                continue
        
        cols.append(c)
    
    if not cols:
        return pd.Series(dtype=float)
    
    return df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)


###################################################################
# Paired Comparison Computation
###################################################################


def compute_window_comparison(
    df: pd.DataFrame,
    group: str,
    segment1: str,
    segment2: str,
    band: str,
    roi_name: str,
    roi_channels: Optional[List[str]],
    min_samples: int = DEFAULT_MIN_SAMPLES_PAIRED,
    n_boot: int = DEFAULT_N_BOOT,
    n_perm: int = DEFAULT_N_PERM,
    rng: Optional[np.random.Generator] = None,
) -> Optional[PairedComparisonResult]:
    """Compute paired comparison between two time windows."""
    s1 = collect_segment_data(df, group, segment1, band, roi_channels)
    s2 = collect_segment_data(df, group, segment2, band, roi_channels)
    
    if s1.empty or s2.empty:
        return None
    
    valid_mask = s1.notna() & s2.notna()
    v1 = s1[valid_mask].values
    v2 = s2[valid_mask].values
    
    if len(v1) < min_samples:
        return None
    
    stat, p = safe_wilcoxon(v1, v2, min_n=min_samples)
    d = compute_paired_cohens_d(v1, v2)
    g = hedges_g(v1, v2)
    _, ci_low, ci_high = bootstrap_mean_diff_ci(v1, v2, n_boot=n_boot, rng=rng, config=None)
    
    hedges_g_finite = g if np.isfinite(g) else d
    
    # Compute permutation p-value if requested
    p_perm = np.nan
    n_permutations = 0
    if n_perm > 0:
        p_perm = permutation_paired_pvalue(v1, v2, n_perm=n_perm, rng=rng)
        n_permutations = n_perm
    
    return PairedComparisonResult(
        feature_type=group,
        comparison_type="window",
        condition1=segment1,
        condition2=segment2,
        identifier=f"{band}_{roi_name}",
        n_pairs=len(v1),
        mean1=float(np.nanmean(v1)),
        mean2=float(np.nanmean(v2)),
        std1=float(np.nanstd(v1, ddof=1)),
        std2=float(np.nanstd(v2, ddof=1)),
        mean_diff=float(np.nanmean(v2 - v1)),
        statistic=stat,
        p_value=p,
        effect_size_d=d,
        effect_size_g=hedges_g_finite,
        ci_low=ci_low,
        ci_high=ci_high,
        test_method="wilcoxon",
        p_perm=p_perm,
        n_permutations=n_permutations,
    )


def compute_condition_comparison(
    df: pd.DataFrame,
    mask1: np.ndarray,
    mask2: np.ndarray,
    group: str,
    segment: str,
    band: str,
    roi_name: str,
    roi_channels: Optional[List[str]],
    label1: str,
    label2: str,
    min_samples: int = DEFAULT_MIN_SAMPLES_UNPAIRED,
    n_boot: int = DEFAULT_N_BOOT,
    n_perm: int = DEFAULT_N_PERM,
    rng: Optional[np.random.Generator] = None,
) -> Optional[PairedComparisonResult]:
    """Compute unpaired comparison between two conditions."""
    series = collect_segment_data(df, group, segment, band, roi_channels)
    
    if series.empty:
        return None
    
    v1 = series[mask1].dropna().values
    v2 = series[mask2].dropna().values
    
    if len(v1) < min_samples or len(v2) < min_samples:
        return None
    
    stat, p = safe_mannwhitneyu(v1, v2, min_n=min_samples)
    d = cohens_d(v2, v1, pooled=True)
    g = hedges_g(v2, v1)
    
    _, ci_low, ci_high = bootstrap_mean_diff_ci(v1, v2, n_boot=n_boot, rng=rng, config=None)
    
    cohens_d_finite = d if np.isfinite(d) else np.nan
    hedges_g_finite = g if np.isfinite(g) else np.nan
    
    # Compute permutation p-value if requested
    p_perm = np.nan
    n_permutations = 0
    if n_perm > 0:
        p_perm = permutation_unpaired_pvalue(v1, v2, n_perm=n_perm, rng=rng)
        n_permutations = n_perm
    
    return PairedComparisonResult(
        feature_type=group,
        comparison_type="condition",
        condition1=label1,
        condition2=label2,
        identifier=f"{band}_{roi_name}",
        n_pairs=len(v1) + len(v2),
        mean1=float(np.nanmean(v1)),
        mean2=float(np.nanmean(v2)),
        std1=float(np.nanstd(v1, ddof=1)),
        std2=float(np.nanstd(v2, ddof=1)),
        mean_diff=float(np.nanmean(v2) - np.nanmean(v1)),
        statistic=stat,
        p_value=p,
        effect_size_d=cohens_d_finite,
        effect_size_g=hedges_g_finite,
        ci_low=ci_low,
        ci_high=ci_high,
        test_method="mannwhitneyu",
        p_perm=p_perm,
        n_permutations=n_permutations,
    )


###################################################################
# Main Computation Functions
###################################################################


def _extract_segments_from_df(df: pd.DataFrame, group: str) -> List[str]:
    """Extract all unique segment names from a feature DataFrame."""
    segments = set()
    for col in df.columns:
        parsed = NamingSchema.parse(str(col))
        if parsed.get("valid") and parsed.get("group") == group:
            seg = parsed.get("segment")
            if seg:
                segments.add(str(seg))
    return sorted(segments)


def _extract_bands_from_df(df: pd.DataFrame, group: str) -> List[str]:
    """Extract all unique band/metric names from a feature DataFrame."""
    bands = set()
    for col in df.columns:
        parsed = NamingSchema.parse(str(col))
        if parsed.get("valid") and parsed.get("group") == group:
            band = parsed.get("band")
            if band:
                bands.add(str(band))
    return sorted(bands)


def _extract_stats_from_df(df: pd.DataFrame, group: str) -> List[str]:
    """Extract all unique statistic names from a feature DataFrame (for aperiodic, etc.)."""
    stats = set()
    for col in df.columns:
        parsed = NamingSchema.parse(str(col))
        if parsed.get("valid") and parsed.get("group") == group:
            stat = parsed.get("stat")
            if stat:
                stats.add(str(stat))
    return sorted(stats)


def compute_all_paired_comparisons(
    feature_dfs: Dict[str, pd.DataFrame],
    events_df: Optional[pd.DataFrame],
    config: Any,
    logger: Optional[logging.Logger] = None,
    min_samples: int = DEFAULT_MIN_SAMPLES_PAIRED,
    n_boot: int = DEFAULT_N_BOOT,
    n_perm: int = DEFAULT_N_PERM,
    fdr_alpha: float = DEFAULT_FDR_ALPHA,
    rng: Optional[np.random.Generator] = None,
) -> PairedComparisonSummary:
    """Compute ALL possible paired comparisons for all feature types.
    
    This function exhaustively computes all paired comparisons:
    - All pairwise segment comparisons (window comparisons)
    - All ROIs (including "all" for global)
    - All frequency bands / metrics
    - Condition comparisons (pain vs non-pain) for each segment
    
    Args:
        feature_dfs: Dict mapping feature type to DataFrame (e.g., {"power": power_df})
        events_df: Events DataFrame for condition comparisons
        config: Pipeline configuration
        logger: Logger instance
        min_samples: Minimum samples for statistical tests
        n_boot: Number of bootstrap iterations
        n_perm: Number of permutations for p-values (0 to skip)
        fdr_alpha: FDR significance threshold
        rng: Random number generator
    
    Returns:
        PairedComparisonSummary with all results
    """
    if rng is None:
        rng = np.random.default_rng()
    
    results: List[PairedComparisonResult] = []
    
    rois = get_roi_definitions(config) or {}
    
    condition_masks = None
    condition_labels = None
    if events_df is not None:
        mask_result = _extract_condition_masks(events_df, config)
        if mask_result is not None:
            condition_masks, condition_labels = mask_result
    
    all_segments_compared = set()
    
    for feat_name, df in feature_dfs.items():
        if df is None or df.empty:
            continue
        
        group = FEATURE_TYPE_GROUPS.get(feat_name, feat_name)
        all_channels = get_channels_from_columns(list(df.columns))
        
        segments = _extract_segments_from_df(df, group)
        if not segments:
            if logger:
                logger.warning(f"No segments detected for feature group {group}; skipping.")
            continue
        
        if feat_name == "aperiodic":
            identifiers = _extract_stats_from_df(df, group)
        else:
            identifiers = _extract_bands_from_df(df, group)
            
        if not identifiers:
            if logger:
                logger.warning(f"No frequency bands or metrics detected for feature group {group}; skipping.")
            continue
        
        roi_names = ["all"]
        if rois:
            roi_names.extend(list(rois.keys()))
        
        for roi_name in roi_names:
            if roi_name == "all":
                roi_channels = all_channels
            else:
                roi_channels = get_roi_channels(rois.get(roi_name, []), all_channels)
            
            if not roi_channels and roi_name != "all":
                continue
            
            for identifier in identifiers:
                for i, seg1 in enumerate(segments):
                    for seg2 in segments[i+1:]:
                        all_segments_compared.add((seg1, seg2))
                        
                        result = compute_window_comparison(
                            df=df,
                            group=group,
                            segment1=seg1,
                            segment2=seg2,
                            band=identifier,
                            roi_name=roi_name,
                            roi_channels=roi_channels if roi_name != "all" else None,
                            min_samples=min_samples,
                            n_boot=n_boot,
                            n_perm=n_perm,
                            rng=rng,
                        )
                        if result is not None:
                            results.append(result)
                
                if condition_masks is not None:
                    m1, m2 = condition_masks
                    l1, l2 = condition_labels
                    
                    for segment in segments:
                        result = compute_condition_comparison(
                            df=df,
                            mask1=m1,
                            mask2=m2,
                            group=group,
                            segment=segment,
                            band=identifier,
                            roi_name=roi_name,
                            roi_channels=roi_channels if roi_name != "all" else None,
                            label1=l1,
                            label2=l2,
                            min_samples=min_samples,
                            n_boot=n_boot,
                            n_perm=n_perm,
                            rng=rng,
                        )
                        if result is not None:
                            results.append(result)
    
    if logger:
        logger.info(f"Computed {len(results)} paired comparisons (exhaustive)")
    
    summary = PairedComparisonSummary(
        results=results,
        n_tests=len(results),
        fdr_alpha=fdr_alpha,
        metadata={
            "segments_compared": sorted([f"{s1}_vs_{s2}" for s1, s2 in all_segments_compared]),
            "n_boot": n_boot,
            "min_samples": min_samples,
            "exhaustive": True,
        },
    )
    
    if results:
        p_values = np.array([r.p_value for r in results])
        q_values = fdr_bh(p_values, alpha=fdr_alpha, config=config)
        
        summary.n_significant_raw = int(np.sum(p_values < fdr_alpha))
        summary.n_significant_fdr = int(np.sum(q_values < fdr_alpha))
        
        summary.metadata["q_values"] = q_values.tolist()
    
    return summary


def _extract_condition_masks(
    events_df: pd.DataFrame,
    config: Any,
) -> Optional[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[str, str]]]:
    """Extract condition masks from events DataFrame."""
    from eeg_pipeline.utils.analysis.events import extract_comparison_mask
    
    result = extract_comparison_mask(events_df, config)
    if result is None:
        return None
    
    m1, m2, l1, l2 = result
    return (m1, m2), (l1, l2)


