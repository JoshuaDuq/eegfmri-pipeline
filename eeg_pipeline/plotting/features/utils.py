"""Shared utilities for feature visualization plotting.

This module consolidates common functionality used across multiple plotting modules:
- FDR correction for multiple comparisons
- Effect size calculations (Cohen's d)
- Bootstrap confidence intervals for means
- Normality testing (Shapiro-Wilk)
- Significance annotation formatting with standardized p/q values
- Common statistical helpers
- Config-driven accessors for bands and colors
"""

from __future__ import annotations

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats

from eeg_pipeline.utils.config.loader import get_frequency_band_names, get_config_value
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.io.figures import get_band_color
from eeg_pipeline.utils.analysis.stats.fdr import fdr_bh
from eeg_pipeline.utils.analysis.stats.effect_size import cohens_d as _cohens_d
from eeg_pipeline.utils.analysis.stats.bootstrap import (
    bootstrap_mean_ci as _bootstrap_mean_ci,
    bootstrap_mean_diff_ci as _bootstrap_mean_diff_ci,
)
from eeg_pipeline.utils.analysis.stats.validation import check_normality_shapiro
from eeg_pipeline.domain.features.naming import NamingSchema


###################################################################
# CONFIG-DRIVEN ACCESSORS
###################################################################

_DEFAULT_BANDS = ["delta", "theta", "alpha", "beta", "gamma"]

_DEFAULT_BAND_RANGES = {
    "delta": "1-4 Hz",
    "theta": "4-8 Hz",
    "alpha": "8-13 Hz",
    "beta": "13-30 Hz",
    "gamma": "30-100 Hz"
}


def get_band_names(config: Any = None) -> List[str]:
    """Return frequency band names from config (falls back to defaults)."""
    bands = get_frequency_band_names(config)
    if bands:
        return list(bands)
    return _DEFAULT_BANDS


def get_band_colors(config: Any = None) -> Dict[str, str]:
    """Return band color mapping from config (falls back to defaults)."""
    colors = {band: get_band_color(band, config) for band in get_band_names(config)}
    if colors:
        return colors
    return {
        "delta": "#440154",
        "theta": "#3b528b",
        "alpha": "#21918c",
        "beta": "#5ec962",
        "gamma": "#fde725",
    }


def get_band_ranges(config: Any = None) -> Dict[str, str]:
    """Return band range labels from config (falls back to defaults)."""
    freq_bands = get_config_value(config, "time_frequency_analysis.bands", {}) or getattr(config, "frequency_bands", {})
    if isinstance(freq_bands, dict) and freq_bands:
        return {name: f"{vals[0]}-{vals[1]} Hz" for name, vals in freq_bands.items()}
    return _DEFAULT_BAND_RANGES


def get_condition_colors(config: Any = None) -> Dict[str, str]:
    """Return condition color mapping from config (pain/nonpain)."""
    plot_cfg = get_plot_config(config)
    return {
        "pain": plot_cfg.get_color("pain"),
        "nonpain": plot_cfg.get_color("nonpain"),
    }


def get_fdr_alpha(config: Any = None, default: float = 0.05) -> float:
    """Return FDR alpha threshold from config."""
    return float(
        get_config_value(
            config,
            "behavior_analysis.statistics.fdr_alpha",
            get_config_value(config, "statistics.fdr_alpha", default),
        )
    )


def get_numeric_feature_columns(
    df: pd.DataFrame,
    *,
    exclude: Optional[List[str]] = None,
) -> List[str]:
    """Return numeric feature columns excluding common metadata columns."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []

    exclude_set = set(exclude or [])
    exclude_set |= {"epoch", "trial", "subject", "index", "condition"}

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in exclude_set]


def get_named_segments(
    df: pd.DataFrame,
    *,
    group: Optional[str] = None,
) -> List[str]:
    """Return available NamingSchema segments for a feature group."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    segments = set()
    for col in df.columns:
        parsed = NamingSchema.parse(str(col))
        if not parsed.get("valid"):
            continue
        if group and parsed.get("group") != group:
            continue
        segment = parsed.get("segment")
        if segment:
            segments.add(str(segment))
    return sorted(segments)


def get_named_bands(
    df: pd.DataFrame,
    *,
    group: Optional[str] = None,
    segment: Optional[str] = None,
) -> List[str]:
    """Return available NamingSchema bands for a feature group/segment."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    bands = set()
    for col in df.columns:
        parsed = NamingSchema.parse(str(col))
        if not parsed.get("valid"):
            continue
        if group and parsed.get("group") != group:
            continue
        if segment and str(parsed.get("segment") or "") != str(segment):
            continue
        band = parsed.get("band")
        if band:
            bands.add(str(band))
    return sorted(bands)


def select_named_columns(
    df: pd.DataFrame,
    *,
    group: str,
    segment: str,
    band: str,
    identifier: Optional[str] = None,
    stat_preference: Optional[List[str]] = None,
    scope_preference: Optional[List[str]] = None,
) -> Tuple[List[str], Optional[str], Optional[str]]:
    """Return columns and matched scope/stat for NamingSchema features."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return [], None, None

    stat_preference = list(stat_preference or [])
    scope_preference = list(scope_preference or [])
    if not stat_preference:
        stat_preference = [None]
    if not scope_preference:
        scope_preference = [None]

    for scope in scope_preference:
        for stat in stat_preference:
            cols = []
            for col in df.columns:
                parsed = NamingSchema.parse(str(col))
                if not parsed.get("valid"):
                    continue
                if parsed.get("group") != group:
                    continue
                if str(parsed.get("segment") or "") != str(segment):
                    continue
                if str(parsed.get("band") or "") != str(band):
                    continue
                if scope and str(parsed.get("scope") or "") != str(scope):
                    continue
                if identifier is not None and str(parsed.get("identifier") or "") != str(identifier):
                    continue
                if stat and str(parsed.get("stat") or "") != str(stat):
                    continue
                cols.append(str(col))
            if cols:
                return cols, scope, stat
    return [], None, None


def collect_named_series(
    df: pd.DataFrame,
    *,
    group: str,
    segment: str,
    band: str,
    identifier: Optional[str] = None,
    stat_preference: Optional[List[str]] = None,
    scope_preference: Optional[List[str]] = None,
) -> Tuple[pd.Series, Optional[str], Optional[str]]:
    """Return per-trial series aggregated across matching NamingSchema columns."""
    cols, scope, stat = select_named_columns(
        df,
        group=group,
        segment=segment,
        band=band,
        identifier=identifier,
        stat_preference=stat_preference,
        scope_preference=scope_preference,
    )
    if not cols:
        return pd.Series(dtype=float), None, None

    if len(cols) == 1:
        series = pd.to_numeric(df[cols[0]], errors="coerce")
    else:
        series = df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    return series, scope, stat


###################################################################
# FDR CORRECTION
###################################################################

def apply_fdr_correction(
    pvalues: List[float],
    alpha: Optional[float] = None,
    method: str = "fdr_bh",
    config: Any = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply FDR correction to multiple p-values.
    
    Args:
        pvalues: List of p-values to correct
        alpha: Significance threshold (default from config.statistics.fdr_alpha)
        method: Correction method (default 'fdr_bh' = Benjamini-Hochberg)
        config: Config object for pulling defaults
    
    Returns:
        Tuple of (rejected, qvalues, corrected_alpha)
        - rejected: Boolean array of which tests are significant
        - qvalues: Corrected p-values (q-values)
        - corrected_alpha: The corrected significance threshold
    """
    if alpha is None:
        alpha = float(get_config_value(config, "statistics.fdr_alpha", 0.05))

    if not pvalues:
        return np.array([]), np.array([]), alpha
    
    pvalues_arr = np.asarray(pvalues, dtype=float)
    qvals = fdr_bh(pvalues_arr, alpha=alpha, config=config)
    rejected = np.isfinite(qvals) & (qvals < float(alpha))
    return rejected, qvals, np.asarray(alpha, dtype=float)


###################################################################
# EFFECT SIZE
###################################################################

def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups.
    
    Uses pooled standard deviation for unequal group sizes.
    
    Args:
        group1: First group values
        group2: Second group values
    
    Returns:
        Cohen's d effect size (positive = group2 > group1)
    """
    g1 = np.asarray(group1).ravel()
    g2 = np.asarray(group2).ravel()
    g1 = g1[np.isfinite(g1)]
    g2 = g2[np.isfinite(g2)]
    if g1.size < 2 or g2.size < 2:
        return 0.0

    d = _cohens_d(g2, g1, pooled=True)
    return float(d) if np.isfinite(d) else 0.0


def compute_paired_cohens_d(before: np.ndarray, after: np.ndarray) -> float:
    """Compute Cohen's d for paired samples.
    
    Args:
        before: Values before intervention
        after: Values after intervention
    
    Returns:
        Cohen's d effect size (positive = increase after)
    """
    if len(before) != len(after) or len(before) < 2:
        return 0.0
    
    diff = after - before
    std_diff = np.std(diff, ddof=1)
    
    if std_diff < 1e-10:
        return 0.0
    return np.mean(diff) / std_diff


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d magnitude.
    
    Args:
        d: Cohen's d value
    
    Returns:
        Interpretation string
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


###################################################################
# SIGNIFICANCE FORMATTING
###################################################################

def format_significance_annotation(
    q: float,
    d: float,
    significant: bool,
    include_d: bool = True,
    newline: bool = True
) -> str:
    """Format significance annotation for plots.
    
    Args:
        q: FDR-corrected q-value
        d: Cohen's d effect size
        significant: Whether test is significant after FDR
        include_d: Include Cohen's d in annotation
        newline: Use newline between q and d
    
    Returns:
        Formatted annotation string
    """
    sig_marker = "†" if significant else ""
    sep = "\n" if newline else " "
    
    if include_d:
        return f"q={q:.3f}{sig_marker}{sep}d={d:.2f}"
    else:
        return f"q={q:.3f}{sig_marker}"


def get_significance_color(significant: bool, config: Any = None) -> str:
    """Get color for significance annotation.
    
    Args:
        significant: Whether test is significant
        config: Config object for pulling palette
    
    Returns:
        Color hex code
    """
    plot_cfg = get_plot_config(config)
    style_colors = getattr(plot_cfg, "style", None)
    if style_colors and hasattr(style_colors, "colors"):
        sig_color = getattr(style_colors.colors, "significant", "#d62728")
        nonsig_color = getattr(style_colors.colors, "nonsignificant", "#333333")
        return sig_color if significant else nonsig_color
    return "#d62728" if significant else "#333333"


###################################################################
# STATISTICAL HELPERS
###################################################################

def safe_mannwhitneyu(
    group1: np.ndarray,
    group2: np.ndarray,
    min_n: int = 3
) -> Tuple[float, float]:
    """Safely compute Mann-Whitney U test.
    
    Args:
        group1: First group values
        group2: Second group values
        min_n: Minimum samples required
    
    Returns:
        Tuple of (statistic, p_value), or (np.nan, 1.0) if test fails
    """
    if len(group1) < min_n or len(group2) < min_n:
        return np.nan, 1.0
    
    try:
        stat, p = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        return stat, p
    except ValueError:
        return np.nan, 1.0


def safe_wilcoxon(
    x: np.ndarray,
    y: np.ndarray,
    min_n: int = 5
) -> Tuple[float, float]:
    """Safely compute Wilcoxon signed-rank test.
    
    Args:
        x: First paired sample
        y: Second paired sample
        min_n: Minimum samples required
    
    Returns:
        Tuple of (statistic, p_value), or (np.nan, 1.0) if test fails
    """
    if len(x) != len(y) or len(x) < min_n:
        return np.nan, 1.0
    
    try:
        stat, p = stats.wilcoxon(x, y)
        return stat, p
    except ValueError:
        return np.nan, 1.0


###################################################################
# BOOTSTRAP CONFIDENCE INTERVALS FOR MEANS
###################################################################

def bootstrap_mean_ci(
    data: np.ndarray,
    n_boot: int = 1000,
    ci_level: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for the mean.
    
    Args:
        data: 1D array of values
        n_boot: Number of bootstrap iterations
        ci_level: Confidence level (default 0.95 for 95% CI)
        rng: Random number generator
    
    Returns:
        Tuple of (mean, ci_low, ci_high)
    """
    if rng is None:
        rng = np.random.default_rng()
    return _bootstrap_mean_ci(data, n_boot=n_boot, ci_level=ci_level, rng=rng)


def bootstrap_mean_diff_ci(
    group1: np.ndarray,
    group2: np.ndarray,
    n_boot: int = 1000,
    ci_level: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """Compute bootstrap CI for difference in means (group2 - group1).
    
    Args:
        group1: First group values
        group2: Second group values
        n_boot: Number of bootstrap iterations
        ci_level: Confidence level
        rng: Random number generator
    
    Returns:
        Tuple of (mean_diff, ci_low, ci_high)
    """
    if rng is None:
        rng = np.random.default_rng()
    return _bootstrap_mean_diff_ci(group1, group2, n_boot=n_boot, ci_level=ci_level, rng=rng)


###################################################################
# NORMALITY TESTING
###################################################################

def test_normality(
    data: np.ndarray,
    alpha: float = 0.05,
    max_n: int = 5000,
) -> Tuple[bool, float, str]:
    """Test normality using Shapiro-Wilk test.
    
    Args:
        data: 1D array of values
        alpha: Significance threshold
        max_n: Maximum sample size for Shapiro-Wilk (uses random subset if exceeded)
    
    Returns:
        Tuple of (is_normal, p_value, interpretation)
        - is_normal: True if p > alpha (fail to reject normality)
        - p_value: Shapiro-Wilk p-value
        - interpretation: Human-readable string
    """
    data = np.asarray(data).ravel()
    data = data[np.isfinite(data)]
    
    if len(data) < 3:
        return True, np.nan, "Insufficient data (n<3)"
    
    if len(data) > max_n:
        rng = np.random.default_rng(42)
        data = rng.choice(data, size=max_n, replace=False)
    
    try:
        res = check_normality_shapiro(data, alpha=alpha)
    except Exception:
        return True, np.nan, "Test failed"

    if not np.isfinite(res.p_value):
        return True, np.nan, "Test failed"

    p = float(res.p_value)
    is_normal = p > alpha

    if p < 0.001:
        interpretation = "Strongly non-normal (p<.001)"
    elif p < 0.01:
        interpretation = "Non-normal (p<.01)"
    elif p < 0.05:
        interpretation = "Marginally non-normal (p<.05)"
    elif p < 0.10:
        interpretation = "Approximately normal (p>.05)"
    else:
        interpretation = "Normal (p>.10)"

    return is_normal, p, interpretation


###################################################################
# STANDARDIZED STATISTICAL ANNOTATION
###################################################################

def compute_condition_stats(
    group1: np.ndarray,
    group2: np.ndarray,
    test: str = "mannwhitneyu",
    n_boot: int = 1000,
    config: Any = None,
) -> Dict[str, Any]:
    """Compute comprehensive statistics for condition comparison.
    
    Returns raw p-value, effect size, and bootstrap CI for difference.
    FDR correction should be applied separately across all tests.
    
    Args:
        group1: First group values (e.g., non-pain)
        group2: Second group values (e.g., pain)
        test: Statistical test ("mannwhitneyu" or "ttest")
        n_boot: Bootstrap iterations for CI
        config: Config object
    
    Returns:
        Dict with keys:
        - p_raw: Raw p-value from test
        - statistic: Test statistic
        - cohens_d: Effect size
        - d_interpretation: Effect size interpretation
        - mean_diff: Mean difference (group2 - group1)
        - ci_low: Bootstrap CI lower bound
        - ci_high: Bootstrap CI upper bound
        - n1, n2: Sample sizes
    """
    g1 = np.asarray(group1).ravel()
    g2 = np.asarray(group2).ravel()
    g1 = g1[np.isfinite(g1)]
    g2 = g2[np.isfinite(g2)]
    
    result = {
        "p_raw": np.nan,
        "statistic": np.nan,
        "cohens_d": np.nan,
        "d_interpretation": "N/A",
        "mean_diff": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "n1": len(g1),
        "n2": len(g2),
    }
    
    if len(g1) < 3 or len(g2) < 3:
        return result
    
    if test == "mannwhitneyu":
        try:
            stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            result["statistic"] = float(stat)
            result["p_raw"] = float(p)
        except ValueError:
            pass
    elif test == "ttest":
        try:
            stat, p = stats.ttest_ind(g1, g2)
            result["statistic"] = float(stat)
            result["p_raw"] = float(p)
        except ValueError:
            pass
    
    result["cohens_d"] = compute_cohens_d(g1, g2)
    result["d_interpretation"] = interpret_cohens_d(result["cohens_d"])
    
    diff, ci_lo, ci_hi = _bootstrap_mean_diff_ci(g1, g2, n_boot=n_boot, config=config)
    result["mean_diff"] = diff
    result["ci_low"] = ci_lo
    result["ci_high"] = ci_hi
    
    return result


def format_stats_annotation(
    p_raw: float,
    q_fdr: Optional[float] = None,
    cohens_d: Optional[float] = None,
    ci_low: Optional[float] = None,
    ci_high: Optional[float] = None,
    compact: bool = True,
) -> str:
    """Format standardized statistical annotation for plots.
    
    Always shows both raw p and FDR-corrected q when available.
    
    Args:
        p_raw: Raw p-value
        q_fdr: FDR-corrected q-value (optional)
        cohens_d: Cohen's d effect size (optional)
        ci_low: Bootstrap CI lower bound (optional)
        ci_high: Bootstrap CI upper bound (optional)
        compact: Use compact format (default True)
    
    Returns:
        Formatted annotation string
    """
    lines = []
    
    if np.isfinite(p_raw):
        p_str = f"p={p_raw:.3f}" if p_raw >= 0.001 else "p<.001"
        if q_fdr is not None and np.isfinite(q_fdr):
            q_str = f"q={q_fdr:.3f}" if q_fdr >= 0.001 else "q<.001"
            sig_marker = "†" if q_fdr < 0.05 else ""
            if compact:
                lines.append(f"{p_str}, {q_str}{sig_marker}")
            else:
                lines.append(f"{p_str}")
                lines.append(f"{q_str}{sig_marker}")
        else:
            sig_marker = "*" if p_raw < 0.05 else ""
            lines.append(f"{p_str}{sig_marker}")
    
    if cohens_d is not None and np.isfinite(cohens_d):
        lines.append(f"d={cohens_d:.2f}")
    
    if ci_low is not None and ci_high is not None:
        if np.isfinite(ci_low) and np.isfinite(ci_high):
            lines.append(f"95%CI [{ci_low:.2f}, {ci_high:.2f}]")
    
    return "\n".join(lines) if lines else ""


def format_footer_annotation(
    n_tests: int,
    correction_method: str = "FDR-BH",
    alpha: float = 0.05,
    n_significant: Optional[int] = None,
    additional_info: Optional[str] = None,
) -> str:
    """Format footer annotation showing multiple comparison info.
    
    Args:
        n_tests: Total number of statistical tests
        correction_method: Correction method used
        alpha: Significance threshold
        n_significant: Number of significant tests after correction
        additional_info: Additional text to append
    
    Returns:
        Footer annotation string
    """
    parts = [f"{n_tests} tests"]
    
    if correction_method:
        parts.append(f"{correction_method} α={alpha}")
    
    if n_significant is not None:
        parts.append(f"{n_significant} significant")
    
    footer = " | ".join(parts)
    
    if additional_info:
        footer = f"{footer} | {additional_info}"
    
    return footer


###################################################################
# VARIABILITY METRICS
###################################################################

def compute_variability_metrics(
    data: np.ndarray,
) -> Dict[str, float]:
    """Compute variability metrics for trial-to-trial analysis.
    
    Args:
        data: 1D array of values (e.g., power per trial)
    
    Returns:
        Dict with:
        - cv: Coefficient of variation (std/|mean|)
        - fano: Fano factor (var/mean) - meaningful for positive data
        - std: Standard deviation
        - iqr: Interquartile range
        - mad: Median absolute deviation
    """
    data = np.asarray(data).ravel()
    data = data[np.isfinite(data)]
    
    if len(data) < 2:
        return {
            "cv": np.nan,
            "fano": np.nan,
            "std": np.nan,
            "iqr": np.nan,
            "mad": np.nan,
        }
    
    mean_val = np.mean(data)
    std_val = np.std(data, ddof=1)
    var_val = np.var(data, ddof=1)
    
    cv = std_val / np.abs(mean_val) if np.abs(mean_val) > 1e-10 else np.nan
    
    fano = var_val / mean_val if mean_val > 1e-10 else np.nan
    
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    
    mad = np.median(np.abs(data - np.median(data)))
    
    return {
        "cv": float(cv),
        "fano": float(fano),
        "std": float(std_val),
        "iqr": float(iqr),
        "mad": float(mad),
    }


###################################################################
# UNIFIED PAIRED COMPARISON PLOTTING
###################################################################

def plot_paired_comparison(
    data_by_band: Dict[str, Tuple[np.ndarray, np.ndarray]],
    subject: str,
    save_path: "Path",
    feature_label: str,
    config: Any = None,
    logger: Any = None,
    *,
    label1: str = "Condition 1",
    label2: str = "Condition 2",
    roi_name: Optional[str] = None,
) -> None:
    """Unified paired comparison plot.
    
    Creates a single-row figure with one subplot per frequency band, showing
    paired comparisons with box plots, scatter points, and connecting lines.
    Uses Wilcoxon signed-rank test with FDR correction.
    
    Args:
        data_by_band: Dict mapping band names to (values1, values2) tuples.
                      Both arrays must have the same length for paired comparison.
        subject: Subject identifier for title.
        save_path: Path to save the figure (without extension).
        feature_label: Human-readable feature name (e.g., "Band Power").
        config: Configuration object.
        logger: Logger instance.
        label1: Label for first condition (from user config).
        label2: Label for second condition (from user config).
        roi_name: Optional ROI name for title.
    """
    from pathlib import Path
    from scipy.stats import wilcoxon
    import matplotlib.pyplot as plt
    from eeg_pipeline.plotting.io.figures import save_fig
    
    if not data_by_band:
        if logger:
            logger.warning(f"No data provided for {feature_label} paired comparison")
        return
    
    # Get bands in order
    band_order = get_band_names(config)
    bands = [b for b in band_order if b in data_by_band]
    bands += [b for b in data_by_band if b not in bands]
    
    if not bands:
        return
    
    n_bands = len(bands)
    plot_cfg = get_plot_config(config)
    band_colors = get_band_colors(config)
    
    # Consistent colors
    segment_colors = {
        "v1": "#5a7d9a",  # first condition
        "v2": "#c44e52",  # second condition
    }
    
    # Compute statistics for each band
    all_pvalues = []
    pvalue_keys = []
    
    for band in bands:
        v1, v2 = data_by_band[band]
        
        if len(v1) > 5 and len(v2) > 5 and len(v1) == len(v2):
            try:
                _, p = wilcoxon(v2, v1)
                diff = v2 - v1
                pooled_std = np.std(diff, ddof=1)
                d = np.mean(diff) / pooled_std if pooled_std > 0 else 0
                all_pvalues.append(p)
                pvalue_keys.append((band, p, d))
            except Exception:
                pass
    
    # FDR correction
    qvalues = {}
    n_significant = 0
    if all_pvalues:
        rejected, qvals, _ = apply_fdr_correction(all_pvalues, config=config)
        for i, (band, p, d) in enumerate(pvalue_keys):
            qvalues[band] = (p, qvals[i], d, rejected[i])
        n_significant = int(np.sum(rejected))
    
    # Create figure
    fig, axes = plt.subplots(1, n_bands, figsize=(3 * n_bands, 5), squeeze=False)
    
    for band_idx, band in enumerate(bands):
        ax = axes.flatten()[band_idx]
        v1, v2 = data_by_band[band]
        
        if len(v1) == 0 or len(v2) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                   transform=ax.transAxes, fontsize=plot_cfg.font.title, color="gray")
            ax.set_xticks([])
            continue
        
        # Box plots
        bp = ax.boxplot([v1, v2], positions=[0, 1], widths=0.4, patch_artist=True)
        bp["boxes"][0].set_facecolor(segment_colors["v1"])
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor(segment_colors["v2"])
        bp["boxes"][1].set_alpha(0.6)
        
        # Scatter points
        ax.scatter(np.random.uniform(-0.08, 0.08, len(v1)),
                  v1, c=segment_colors["v1"], alpha=0.3, s=6)
        ax.scatter(1 + np.random.uniform(-0.08, 0.08, len(v2)),
                  v2, c=segment_colors["v2"], alpha=0.3, s=6)
        
        # Paired connecting lines
        if len(v1) == len(v2) and len(v1) <= 100:
            for i in range(len(v1)):
                ax.plot([0, 1], [v1[i], v2[i]], c="gray", alpha=0.15, lw=0.5)
        
        # Y-axis limits with room for annotation
        all_vals = np.concatenate([v1, v2])
        ymin, ymax = np.nanmin(all_vals), np.nanmax(all_vals)
        yrange = ymax - ymin if ymax > ymin else 0.1
        ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.3 * yrange)
        
        # Statistical annotation
        if band in qvalues:
            _, q, d, sig = qvalues[band]
            sig_marker = "†" if sig else ""
            sig_color = get_significance_color(sig, config)
            ax.annotate(
                f"q={q:.3f}{sig_marker}\nd={d:.2f}",
                xy=(0.5, ymax + 0.05 * yrange),
                ha="center",
                fontsize=plot_cfg.font.medium,
                color=sig_color,
                fontweight="bold" if sig else "normal",
            )
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels([label1, label2], fontsize=9)
        ax.set_title(band.capitalize(), fontweight="bold", 
                    color=band_colors.get(band, "gray"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    # Build title - uses user-provided labels only
    n_trials = len(data_by_band[bands[0]][0]) if bands else 0
    n_tests = len(all_pvalues)
    
    title_parts = [f"{feature_label}: {label1} vs {label2} (Paired Comparison)"]
    
    info_parts = [f"Subject: {subject}"]
    if roi_name:
        roi_display = roi_name.replace("_", " ").title() if roi_name != "all" else "All Channels"
        info_parts.append(f"ROI: {roi_display}")
    info_parts.extend([
        f"N: {n_trials} trials",
        "Wilcoxon signed-rank",
        f"FDR: {n_significant}/{n_tests} significant (†=q<0.05)"
    ])
    title_parts.append(" | ".join(info_parts))
    
    fig.suptitle("\n".join(title_parts), fontsize=plot_cfg.font.suptitle, 
                fontweight="bold", y=1.02)
    
    plt.tight_layout()
    save_fig(fig, save_path, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    
    if logger:
        logger.info(f"Saved {feature_label} paired comparison ({n_significant}/{n_tests} FDR significant)")

