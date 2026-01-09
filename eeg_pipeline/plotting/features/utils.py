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

from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Union
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
    """Return condition color mapping from config (condition_1/condition_2)."""
    plot_cfg = get_plot_config(config)
    return {
        "condition_1": plot_cfg.get_color("condition_1"),
        "condition_2": plot_cfg.get_color("condition_2"),
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

    default_exclude = {"epoch", "trial", "subject", "index", "condition"}
    exclude_set = set(exclude or []) | default_exclude

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if col not in exclude_set]


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
        parsed_segment = str(parsed.get("segment") or "")
        if segment and parsed_segment != str(segment):
            continue
        band = parsed.get("band")
        if band:
            bands.add(str(band))
    
    return sorted(bands)


def _matches_naming_schema_criteria(
    parsed: Dict[str, Any],
    group: str,
    segment: str,
    band: str,
    identifier: Optional[str] = None,
    scope: Optional[str] = None,
    stat: Optional[str] = None,
) -> bool:
    """Check if parsed NamingSchema matches all specified criteria.
    
    Args:
        parsed: Parsed NamingSchema dictionary
        group: Required group name
        segment: Required segment name
        band: Required band name
        identifier: Optional identifier filter
        scope: Optional scope filter
        stat: Optional stat filter
    
    Returns:
        True if all criteria match
    """
    if not parsed.get("valid"):
        return False
    if parsed.get("group") != group:
        return False
    if str(parsed.get("segment") or "") != str(segment):
        return False
    if str(parsed.get("band") or "") != str(band):
        return False
    if scope and str(parsed.get("scope") or "") != str(scope):
        return False
    if identifier is not None and str(parsed.get("identifier") or "") != str(identifier):
        return False
    if stat and str(parsed.get("stat") or "") != str(stat):
        return False
    return True


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

    stat_prefs = list(stat_preference or [None])
    scope_prefs = list(scope_preference or [None])

    for scope in scope_prefs:
        for stat in stat_prefs:
            matching_columns = []
            for col in df.columns:
                parsed = NamingSchema.parse(str(col))
                if _matches_naming_schema_criteria(
                    parsed, group, segment, band, identifier, scope, stat
                ):
                    matching_columns.append(str(col))
            
            if matching_columns:
                return matching_columns, scope, stat
    
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
    matching_columns, matched_scope, matched_stat = select_named_columns(
        df,
        group=group,
        segment=segment,
        band=band,
        identifier=identifier,
        stat_preference=stat_preference,
        scope_preference=scope_preference,
    )
    if not matching_columns:
        return pd.Series(dtype=float), None, None

    if len(matching_columns) == 1:
        series = pd.to_numeric(df[matching_columns[0]], errors="coerce")
    else:
        numeric_data = df[matching_columns].apply(pd.to_numeric, errors="coerce")
        series = numeric_data.mean(axis=1)
    return series, matched_scope, matched_stat


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
    group1_clean = np.asarray(group1).ravel()
    group2_clean = np.asarray(group2).ravel()
    group1_clean = group1_clean[np.isfinite(group1_clean)]
    group2_clean = group2_clean[np.isfinite(group2_clean)]
    
    if group1_clean.size < 2 or group2_clean.size < 2:
        return 0.0

    effect_size = _cohens_d(group2_clean, group1_clean, pooled=True)
    return float(effect_size) if np.isfinite(effect_size) else 0.0


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
) -> str:
    """Format significance annotation for plots with q-value and effect size.
    
    Args:
        q: FDR-corrected q-value
        d: Cohen's d effect size
        significant: Whether test is significant after FDR
    
    Returns:
        Formatted annotation string with q-value and effect size on separate lines
    """
    sig_marker = "†" if significant else ""
    return f"q={q:.3f}{sig_marker}\nd={d:.2f}"


def format_significance_annotation_compact(
    q: float,
    d: float,
    significant: bool,
) -> str:
    """Format significance annotation for plots in compact single-line format.
    
    Args:
        q: FDR-corrected q-value
        d: Cohen's d effect size
        significant: Whether test is significant after FDR
    
    Returns:
        Formatted annotation string with q-value and effect size on one line
    """
    sig_marker = "†" if significant else ""
    return f"q={q:.3f}{sig_marker} d={d:.2f}"


def format_significance_annotation_q_only(
    q: float,
    significant: bool,
) -> str:
    """Format significance annotation for plots with q-value only.
    
    Args:
        q: FDR-corrected q-value
        significant: Whether test is significant after FDR
    
    Returns:
        Formatted annotation string with q-value only
    """
    sig_marker = "†" if significant else ""
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
        result = check_normality_shapiro(data, alpha=alpha)
    except (ValueError, RuntimeError) as e:
        return True, np.nan, f"Test failed: {type(e).__name__}"

    if not np.isfinite(result.p_value):
        return True, np.nan, "Test failed: invalid p-value"

    p_value = float(result.p_value)
    is_normal = p_value > alpha

    if p_value < 0.001:
        interpretation = "Strongly non-normal (p<.001)"
    elif p_value < 0.01:
        interpretation = "Non-normal (p<.01)"
    elif p_value < 0.05:
        interpretation = "Marginally non-normal (p<.05)"
    elif p_value < 0.10:
        interpretation = "Approximately normal (p>.05)"
    else:
        interpretation = "Normal (p>.10)"

    return is_normal, p_value, interpretation


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
    group1_clean = np.asarray(group1).ravel()
    group2_clean = np.asarray(group2).ravel()
    group1_clean = group1_clean[np.isfinite(group1_clean)]
    group2_clean = group2_clean[np.isfinite(group2_clean)]
    
    result = {
        "p_raw": np.nan,
        "statistic": np.nan,
        "cohens_d": np.nan,
        "d_interpretation": "N/A",
        "mean_diff": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "n1": len(group1_clean),
        "n2": len(group2_clean),
    }
    
    if len(group1_clean) < 3 or len(group2_clean) < 3:
        return result
    
    if test == "mannwhitneyu":
        try:
            statistic, p_value = stats.mannwhitneyu(
                group1_clean, group2_clean, alternative="two-sided"
            )
            result["statistic"] = float(statistic)
            result["p_raw"] = float(p_value)
        except ValueError:
            pass
    elif test == "ttest":
        try:
            statistic, p_value = stats.ttest_ind(group1_clean, group2_clean)
            result["statistic"] = float(statistic)
            result["p_raw"] = float(p_value)
        except ValueError:
            pass
    
    result["cohens_d"] = compute_cohens_d(group1_clean, group2_clean)
    result["d_interpretation"] = interpret_cohens_d(result["cohens_d"])
    
    mean_diff, ci_low, ci_high = _bootstrap_mean_diff_ci(
        group1_clean, group2_clean, n_boot=n_boot, config=config
    )
    result["mean_diff"] = mean_diff
    result["ci_low"] = ci_low
    result["ci_high"] = ci_high
    
    return result


def format_stats_annotation(
    p_raw: float,
    q_fdr: Optional[float] = None,
    cohens_d: Optional[float] = None,
    ci_low: Optional[float] = None,
    ci_high: Optional[float] = None,
) -> str:
    """Format standardized statistical annotation for plots.
    
    Always shows both raw p and FDR-corrected q when available.
    
    Args:
        p_raw: Raw p-value
        q_fdr: FDR-corrected q-value (optional)
        cohens_d: Cohen's d effect size (optional)
        ci_low: Bootstrap CI lower bound (optional)
        ci_high: Bootstrap CI upper bound (optional)
    
    Returns:
        Formatted annotation string with each metric on separate lines
    """
    lines = []
    
    if np.isfinite(p_raw):
        p_str = f"p={p_raw:.3f}" if p_raw >= 0.001 else "p<.001"
        if q_fdr is not None and np.isfinite(q_fdr):
            q_str = f"q={q_fdr:.3f}" if q_fdr >= 0.001 else "q<.001"
            sig_marker = "†" if q_fdr < 0.05 else ""
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


def format_stats_annotation_compact(
    p_raw: float,
    q_fdr: Optional[float] = None,
    cohens_d: Optional[float] = None,
    ci_low: Optional[float] = None,
    ci_high: Optional[float] = None,
) -> str:
    """Format standardized statistical annotation for plots in compact format.
    
    Args:
        p_raw: Raw p-value
        q_fdr: FDR-corrected q-value (optional)
        cohens_d: Cohen's d effect size (optional)
        ci_low: Bootstrap CI lower bound (optional)
        ci_high: Bootstrap CI upper bound (optional)
    
    Returns:
        Formatted annotation string with p and q on one line
    """
    lines = []
    
    if np.isfinite(p_raw):
        p_str = f"p={p_raw:.3f}" if p_raw >= 0.001 else "p<.001"
        if q_fdr is not None and np.isfinite(q_fdr):
            q_str = f"q={q_fdr:.3f}" if q_fdr >= 0.001 else "q<.001"
            sig_marker = "†" if q_fdr < 0.05 else ""
            lines.append(f"{p_str}, {q_str}{sig_marker}")
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
    data_clean = np.asarray(data).ravel()
    data_clean = data_clean[np.isfinite(data_clean)]
    
    if len(data_clean) < 2:
        return {
            "cv": np.nan,
            "fano": np.nan,
            "std": np.nan,
            "iqr": np.nan,
            "mad": np.nan,
        }
    
    mean_value = np.mean(data_clean)
    std_value = np.std(data_clean, ddof=1)
    variance_value = np.var(data_clean, ddof=1)
    
    min_denominator = 1e-10
    coefficient_of_variation = (
        std_value / np.abs(mean_value)
        if np.abs(mean_value) > min_denominator
        else np.nan
    )
    
    fano_factor = (
        variance_value / mean_value
        if mean_value > min_denominator
        else np.nan
    )
    
    q75 = np.percentile(data_clean, 75)
    q25 = np.percentile(data_clean, 25)
    interquartile_range = q75 - q25
    
    median_value = np.median(data_clean)
    median_absolute_deviation = np.median(np.abs(data_clean - median_value))
    
    return {
        "cv": float(coefficient_of_variation),
        "fano": float(fano_factor),
        "std": float(std_value),
        "iqr": float(interquartile_range),
        "mad": float(median_absolute_deviation),
    }


###################################################################
# UNIFIED PAIRED COMPARISON PLOTTING
###################################################################


def _compute_paired_wilcoxon_stats(
    condition1_values: np.ndarray,
    condition2_values: np.ndarray,
) -> Tuple[float, float]:
    """Compute Wilcoxon signed-rank test and effect size for paired data.
    
    Args:
        condition1_values: First condition values
        condition2_values: Second condition values
    
    Returns:
        Tuple of (p_value, effect_size_d)
    """
    from scipy.stats import wilcoxon
    
    differences = condition2_values - condition1_values
    std_diff = np.std(differences, ddof=1)
    effect_size = np.mean(differences) / std_diff if std_diff > 0 else 0.0
    _, p_value = wilcoxon(condition2_values, condition1_values)
    return float(p_value), float(effect_size)


def _plot_single_band_comparison(
    ax: Any,
    condition1_values: np.ndarray,
    condition2_values: np.ndarray,
    band: str,
    label1: str,
    label2: str,
    band_color: str,
    condition1_color: str,
    condition2_color: str,
    q_value: Optional[float],
    effect_size: Optional[float],
    is_significant: bool,
    plot_cfg: Any,
    config: Any,
) -> None:
    """Plot single band comparison with box plots, scatter, and connecting lines.
    
    Args:
        ax: Matplotlib axes
        condition1_values: First condition values
        condition2_values: Second condition values
        band: Band name for title
        label1: Label for first condition
        label2: Label for second condition
        band_color: Color for band title
        condition1_color: Color for condition 1
        condition2_color: Color for condition 2
        q_value: FDR-corrected q-value (optional)
        effect_size: Cohen's d effect size (optional)
        is_significant: Whether test is significant
        plot_cfg: Plot configuration object
        config: Config object
    """
    if len(condition1_values) == 0 or len(condition2_values) == 0:
        ax.text(
            0.5, 0.5, "No data", ha="center", va="center",
            transform=ax.transAxes, fontsize=plot_cfg.font.title, color="gray"
        )
        ax.set_xticks([])
        return
    
    box_positions = [0, 1]
    box_width = 0.4
    
    boxplot = ax.boxplot(
        [condition1_values, condition2_values],
        positions=box_positions,
        widths=box_width,
        patch_artist=True
    )
    boxplot["boxes"][0].set_facecolor(condition1_color)
    boxplot["boxes"][0].set_alpha(0.6)
    boxplot["boxes"][1].set_facecolor(condition2_color)
    boxplot["boxes"][1].set_alpha(0.6)
    
    jitter_range = 0.08
    condition1_jitter = np.random.uniform(-jitter_range, jitter_range, len(condition1_values))
    condition2_jitter = np.random.uniform(-jitter_range, jitter_range, len(condition2_values))
    
    ax.scatter(
        condition1_jitter, condition1_values,
        c=condition1_color, alpha=0.3, s=6
    )
    ax.scatter(
        1 + condition2_jitter, condition2_values,
        c=condition2_color, alpha=0.3, s=6
    )
    
    max_paired_lines = 100
    if len(condition1_values) == len(condition2_values) and len(condition1_values) <= max_paired_lines:
        for i in range(len(condition1_values)):
            ax.plot(
                [0, 1], [condition1_values[i], condition2_values[i]],
                c="gray", alpha=0.15, lw=0.5
            )
    
    all_values = np.concatenate([condition1_values, condition2_values])
    y_min = np.nanmin(all_values)
    y_max = np.nanmax(all_values)
    y_range = y_max - y_min if y_max > y_min else 0.1
    padding_bottom = 0.1 * y_range
    padding_top = 0.3 * y_range
    ax.set_ylim(y_min - padding_bottom, y_max + padding_top)
    
    if q_value is not None and effect_size is not None:
        significance_marker = "†" if is_significant else ""
        significance_color = get_significance_color(is_significant, config)
        annotation_text = f"q={q_value:.3f}{significance_marker}\nd={effect_size:.2f}"
        annotation_y = y_max + 0.05 * y_range
        
        ax.annotate(
            annotation_text,
            xy=(0.5, annotation_y),
            ha="center",
            fontsize=plot_cfg.font.medium,
            color=significance_color,
            fontweight="bold" if is_significant else "normal",
        )
    
    ax.set_xticks(box_positions)
    ax.set_xticklabels([label1, label2], fontsize=9)
    ax.set_title(band.capitalize(), fontweight="bold", color=band_color)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _load_condition_effects_files(
    stats_dir: Path,
    comparison_type: str,
    suffix: str,
) -> List[pd.DataFrame]:
    """Load condition effects files for a specific comparison type.
    
    Args:
        stats_dir: Path to stats directory
        comparison_type: "window" or "column"
        suffix: Optional file suffix
    
    Returns:
        List of loaded DataFrames
    """
    from eeg_pipeline.infra.tsv import read_tsv
    
    result_dfs = []
    condition_subdir = stats_dir / "condition"
    search_dirs = [condition_subdir, stats_dir]
    
    base_filename = f"condition_effects_{comparison_type}"
    patterns = [
        f"{base_filename}{suffix}.tsv",
        f"{base_filename}*.tsv"
    ]
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for pattern in patterns:
            glob_pattern = pattern.replace("*", "**/*") if "**" not in pattern else pattern
            for path in search_dir.glob(glob_pattern):
                if path.is_file():
                    df = read_tsv(path)
                    if df is not None and not df.empty:
                        normalized_df = _normalize_condition_effects_df(df, comparison_type)
                        if normalized_df is not None and not normalized_df.empty:
                            result_dfs.append(normalized_df)
    
    return result_dfs


def load_precomputed_paired_stats(
    stats_dir: Union[Path, str],
    feature_type: Optional[str] = None,
    comparison_type: Optional[str] = None,
    condition1: Optional[str] = None,
    condition2: Optional[str] = None,
    roi_name: Optional[str] = None,
    suffix: str = "",
) -> Optional[pd.DataFrame]:
    """Load pre-computed paired comparison statistics from behavior pipeline.
    
    Checks for statistics files in the following order:
    1. condition_effects_window*.tsv or condition_effects_column*.tsv (new format from behavior pipeline)
    2. paired_comparisons*.tsv (legacy format)
    
    Args:
        stats_dir: Path to stats directory
        feature_type: Optional feature type filter (e.g., "power", "aperiodic")
        comparison_type: Optional "window" or "column" filter
        condition1: Optional first condition label filter
        condition2: Optional second condition label filter
        roi_name: Optional ROI name filter
        suffix: Optional file suffix
    
    Returns:
        DataFrame with pre-computed statistics or None if not found
    """
    from pathlib import Path
    from eeg_pipeline.utils.analysis.stats.paired_comparisons import load_paired_comparisons
    
    stats_dir_path = Path(stats_dir)
    result_dfs = []
    
    comparison_types_to_try = []
    if comparison_type is None:
        comparison_types_to_try = ["window", "column"]
    elif comparison_type in ("window", "column"):
        comparison_types_to_try = [comparison_type]
    
    for comp_type in comparison_types_to_try:
        loaded_dfs = _load_condition_effects_files(stats_dir_path, comp_type, suffix)
        result_dfs.extend(loaded_dfs)
    
    if result_dfs:
        combined_df = pd.concat(result_dfs, ignore_index=True)
        return _apply_stats_filters(
            combined_df, feature_type, comparison_type, condition1, condition2, roi_name
        )
    
    legacy_df = load_paired_comparisons(stats_dir_path, suffix=suffix)
    if legacy_df is None or legacy_df.empty:
        return None
    
    return _apply_stats_filters(
        legacy_df, feature_type, comparison_type, condition1, condition2, roi_name
    )


def _normalize_condition_effects_df(
    df: pd.DataFrame,
    comparison_type: str,
) -> Optional[pd.DataFrame]:
    """Normalize condition effects DataFrame to expected schema.
    
    Converts from behavior pipeline output format to plotting expected format.
    Works with any user-defined columns and windows.
    """
    if df is None or df.empty:
        return None
    
    # Check for required columns
    if "feature" not in df.columns:
        return None
    
    result = df.copy()
    
    # Map feature to identifier (for lookup)
    if "identifier" not in result.columns:
        result["identifier"] = result["feature"].astype(str)
    
    # Handle window comparison - map window1/window2 to condition1/condition2
    if comparison_type == "window":
        if "window1" in result.columns and "condition1" not in result.columns:
            result["condition1"] = result["window1"]
        if "window2" in result.columns and "condition2" not in result.columns:
            result["condition2"] = result["window2"]
    
    # Ensure comparison_type is set
    if "comparison_type" not in result.columns:
        result["comparison_type"] = comparison_type
    
    # Map effect size columns
    if "effect_size_d" not in result.columns:
        if "hedges_g" in result.columns:
            result["effect_size_d"] = result["hedges_g"]
        elif "cohens_d" in result.columns:
            result["effect_size_d"] = result["cohens_d"]
        else:
            result["effect_size_d"] = 0.0
    
    # Map q_value if not present
    if "q_value" not in result.columns:
        if "p_fdr" in result.columns:
            result["q_value"] = result["p_fdr"]
        elif "p_value" in result.columns:
            result["q_value"] = result["p_value"]
    
    # Mark significant results
    if "significant_fdr" not in result.columns:
        q_vals = pd.to_numeric(result.get("q_value", pd.Series(dtype=float)), errors="coerce")
        result["significant_fdr"] = q_vals < 0.05
    
    return result


def _apply_stats_filters(
    df: pd.DataFrame,
    feature_type: Optional[str] = None,
    comparison_type: Optional[str] = None,
    condition1: Optional[str] = None,
    condition2: Optional[str] = None,
    roi_name: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Apply optional filters to pre-computed stats DataFrame."""
    if df is None or df.empty:
        return None
    
    mask = pd.Series(True, index=df.index)
    
    # Filter by feature_type
    if feature_type and "feature_type" in df.columns:
        ft_lower = feature_type.lower()
        mask &= df["feature_type"].str.lower().str.contains(ft_lower, na=False)
    
    # Also check if feature name contains the feature type
    if feature_type and "identifier" in df.columns:
        ft_lower = feature_type.lower()
        mask |= df["identifier"].str.lower().str.contains(ft_lower, na=False)
    
    # Filter by comparison_type
    if comparison_type and "comparison_type" in df.columns:
        mask &= df["comparison_type"].str.lower() == comparison_type.lower()
    
    # Filter by conditions (flexible matching)
    if condition1 and "condition1" in df.columns:
        c1_lower = condition1.lower()
        mask &= df["condition1"].str.lower() == c1_lower
    if condition2 and "condition2" in df.columns:
        c2_lower = condition2.lower()
        mask &= df["condition2"].str.lower() == c2_lower
    
    # Filter by ROI
    if roi_name and roi_name.lower() != "all" and "identifier" in df.columns:
        mask &= df["identifier"].str.lower().str.contains(roi_name.lower(), na=False)
    
    filtered = df[mask]
    return filtered if not filtered.empty else None


def get_precomputed_qvalues(
    precomputed_df: Optional[pd.DataFrame],
    feature_keys: List[str],
    roi_name: str = "all",
) -> Dict[str, Tuple[float, float, float, bool]]:
    """Extract q-values from pre-computed statistics DataFrame.
    
    Flexible matching: looks for feature_keys in the identifier column using
    substring matching, which works with any user-defined feature names.
    
    Args:
        precomputed_df: Pre-computed statistics DataFrame
        feature_keys: List of feature keys to look for (e.g., band names, metric names)
        roi_name: ROI name for context
    
    Returns:
        Dict mapping feature_key to (p_value, q_value, effect_size_d, significant)
    """
    qvalues = {}
    
    if precomputed_df is None or precomputed_df.empty:
        return qvalues
    
    if "identifier" not in precomputed_df.columns:
        return qvalues
    
    for key in feature_keys:
        key_lower = key.lower()
        
        # Try exact match with roi first
        if roi_name and roi_name.lower() != "all":
            pattern = f"{key_lower}_{roi_name.lower()}"
            match = precomputed_df[
                precomputed_df["identifier"].str.lower() == pattern
            ]
        else:
            match = pd.DataFrame()
        
        # Try partial match on key
        if match.empty:
            match = precomputed_df[
                precomputed_df["identifier"].str.lower().str.contains(key_lower, na=False)
            ]
        
        # If still no match, try matching just the key anywhere
        if match.empty:
            match = precomputed_df[
                precomputed_df["identifier"].str.lower().str.contains(f"_{key_lower}_", na=False) |
                precomputed_df["identifier"].str.lower().str.endswith(f"_{key_lower}", na=False) |
                precomputed_df["identifier"].str.lower().str.startswith(f"{key_lower}_", na=False)
            ]
        
        if not match.empty:
            row = match.iloc[0]
            p = float(row.get("p_value", row.get("p_raw", 1.0)))
            q = float(row.get("q_value", row.get("p_fdr", p)))
            d = float(row.get("effect_size_d", row.get("hedges_g", row.get("cohens_d", 0.0))))
            sig = bool(row.get("significant_fdr", q < 0.05)) if "significant_fdr" in row else (q < 0.05)
            qvalues[key] = (p, q, d, sig)
    
    return qvalues


def compute_or_load_column_stats(
    stats_dir: Optional[Union[Path, str]],
    feature_type: str,
    feature_keys: List[str],
    cell_data: Dict[int, Optional[Dict[str, np.ndarray]]],
    config: Any = None,
    logger: Any = None,
) -> Tuple[Dict[int, Tuple[float, float, float, bool]], int, bool]:
    """Compute or load column comparison statistics.
    
    This is a central helper for all plotting functions that need column
    comparison statistics. It first tries to load pre-computed stats from
    the behavior pipeline. If not available, computes on-the-fly.
    
    Args:
        stats_dir: Optional path to stats directory for pre-computed stats
        feature_type: Feature type (e.g., "power", "spectral", "aperiodic")
        feature_keys: List of feature keys (e.g., band names) matching cell_data keys by index
        cell_data: Dict mapping column index to {"v1": array, "v2": array} or None
        config: Configuration object
        logger: Logger instance
    
    Returns:
        Tuple of:
        - qvalues: Dict mapping column_idx to (p_value, q_value, effect_size_d, significant)
        - n_significant: Number of significant tests
        - use_precomputed: Whether pre-computed stats were used
    """
    qvalues: Dict[int, Tuple[float, float, float, bool]] = {}
    n_significant = 0
    use_precomputed = False
    
    # Try to load pre-computed stats
    if stats_dir is not None:
        precomputed = load_precomputed_paired_stats(
            stats_dir=stats_dir,
            feature_type=feature_type,
            comparison_type="column",
        )
        
        if precomputed is not None and not precomputed.empty:
            use_precomputed = True
            if logger:
                import logging
                if hasattr(logger, "info"):
                    logger.info(f"Using pre-computed column stats for {feature_type} ({len(precomputed)} entries)")
            
            # Map pre-computed stats to feature_keys
            precomputed_qvals = get_precomputed_qvalues(precomputed, feature_keys, roi_name="all")
            
            for col_idx, key in enumerate(feature_keys):
                if key in precomputed_qvals:
                    qvalues[col_idx] = precomputed_qvals[key]
            
            n_significant = sum(1 for v in qvalues.values() if v[3])
            return qvalues, n_significant, use_precomputed
    
    # Fall back to computing on-the-fly
    all_pvals = []
    pvalue_keys = []
    
    for col_idx, key in enumerate(feature_keys):
        data = cell_data.get(col_idx)
        if data is None:
            continue
        
        condition1_values = data.get("v1", np.array([]))
        condition2_values = data.get("v2", np.array([]))
        
        if len(condition1_values) >= 3 and len(condition2_values) >= 3:
            try:
                _, p_value = stats.mannwhitneyu(
                    condition1_values, condition2_values, alternative="two-sided"
                )
                mean_diff = np.mean(condition2_values) - np.mean(condition1_values)
                n1, n2 = len(condition1_values), len(condition2_values)
                variance1 = np.var(condition1_values, ddof=1)
                variance2 = np.var(condition2_values, ddof=1)
                pooled_variance = ((n1 - 1) * variance1 + (n2 - 1) * variance2) / (n1 + n2 - 2)
                pooled_std = np.sqrt(pooled_variance)
                effect_size = mean_diff / pooled_std if pooled_std > 0 else 0
                all_pvals.append(p_value)
                pvalue_keys.append((col_idx, p_value, effect_size))
            except (ValueError, RuntimeError) as e:
                if logger:
                    logger.debug(f"Failed to compute stats for column {col_idx}: {e}")
                pass
    
    if all_pvals:
        rejected, qvals, _ = apply_fdr_correction(all_pvals, config=config)
        for i, (col_idx, p, d) in enumerate(pvalue_keys):
            qvalues[col_idx] = (p, qvals[i], d, rejected[i])
        n_significant = int(np.sum(rejected))
    
    return qvalues, n_significant, use_precomputed


def plot_paired_comparison(
    data_by_band: Dict[str, Tuple[np.ndarray, np.ndarray]],
    subject: str,
    save_path: Union[Path, str],
    feature_label: str,
    config: Any = None,
    logger: Any = None,
    *,
    label1: str = "Condition 1",
    label2: str = "Condition 2",
    roi_name: Optional[str] = None,
    precomputed_stats: Optional[pd.DataFrame] = None,
    stats_dir: Optional[Union[Path, str]] = None,
) -> None:
    """Unified paired comparison plot.
    
    Creates a single-row figure with one subplot per frequency band, showing
    paired comparisons with box plots, scatter points, and connecting lines.
    
    If precomputed_stats or stats_dir is provided, uses pre-computed statistics
    from the behavior pipeline instead of computing on-the-fly.
    
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
        precomputed_stats: Optional pre-computed statistics DataFrame.
        stats_dir: Optional path to stats directory to load pre-computed stats.
    """
    from pathlib import Path
    from scipy.stats import wilcoxon
    import matplotlib.pyplot as plt
    from eeg_pipeline.plotting.io.figures import save_fig
    
    if not data_by_band:
        if logger:
            logger.warning(f"No data provided for {feature_label} paired comparison")
        return
    
    band_order = get_band_names(config)
    bands_in_order = [b for b in band_order if b in data_by_band]
    bands_in_order += [b for b in data_by_band if b not in bands_in_order]
    
    if not bands_in_order:
        return
    
    n_bands = len(bands_in_order)
    plot_cfg = get_plot_config(config)
    band_colors = get_band_colors(config)
    
    condition1_color = "#5a7d9a"
    condition2_color = "#c44e52"
    
    feature_type_map = {
        "Band Power": "power",
        "Aperiodic": "aperiodic",
        "Connectivity": "connectivity",
        "Spectral": "spectral",
        "ERDS": "erds",
        "Band Ratios": "ratios",
        "Asymmetry": "asymmetry",
        "ITPC": "itpc",
        "PAC": "pac",
        "Complexity": "complexity",
    }
    feature_type = feature_type_map.get(feature_label, feature_label.lower())
    
    if precomputed_stats is None and stats_dir is not None:
        precomputed_stats = load_precomputed_paired_stats(
            stats_dir=stats_dir,
            feature_type=feature_type,
            comparison_type="window",
            condition1=label1.lower(),
            condition2=label2.lower(),
            roi_name=roi_name,
        )
    
    qvalues = {}
    n_significant = 0
    use_precomputed = precomputed_stats is not None and not precomputed_stats.empty
    
    if use_precomputed:
        qvalues = get_precomputed_qvalues(precomputed_stats, bands_in_order, roi_name or "all")
        n_significant = sum(1 for stats_tuple in qvalues.values() if stats_tuple[3])
        if logger:
            logger.debug(f"Using pre-computed statistics for {feature_label} ({len(qvalues)} bands)")
    else:
        all_pvalues = []
        pvalue_keys = []
        min_samples = int(get_config_value(config, "behavior_analysis.min_samples.default", 5))
        
        for band in bands_in_order:
            condition1_values, condition2_values = data_by_band[band]
            
            has_sufficient_samples = (
                len(condition1_values) >= min_samples and
                len(condition2_values) >= min_samples and
                len(condition1_values) == len(condition2_values)
            )
            
            if has_sufficient_samples:
                try:
                    p_value, effect_size = _compute_paired_wilcoxon_stats(
                        condition1_values, condition2_values
                    )
                    all_pvalues.append(p_value)
                    pvalue_keys.append((band, p_value, effect_size))
                except (ValueError, RuntimeError) as e:
                    if logger:
                        logger.debug(f"Failed to compute stats for band {band}: {e}")
                    pass
        
        if all_pvalues:
            rejected, qvals, _ = apply_fdr_correction(all_pvalues, config=config)
            for i, (band, p_value, effect_size) in enumerate(pvalue_keys):
                qvalues[band] = (p_value, qvals[i], effect_size, rejected[i])
            n_significant = int(np.sum(rejected))
    
    fig_width_per_band = 3
    fig_height = 5
    fig, axes = plt.subplots(
        1, n_bands, figsize=(fig_width_per_band * n_bands, fig_height), squeeze=False
    )
    
    for band_idx, band in enumerate(bands_in_order):
        ax = axes.flatten()[band_idx]
        condition1_values, condition2_values = data_by_band[band]
        
        q_value = None
        effect_size = None
        is_significant = False
        if band in qvalues:
            _, q_value, effect_size, is_significant = qvalues[band]
        
        _plot_single_band_comparison(
            ax=ax,
            condition1_values=condition1_values,
            condition2_values=condition2_values,
            band=band,
            label1=label1,
            label2=label2,
            band_color=band_colors.get(band, "gray"),
            condition1_color=condition1_color,
            condition2_color=condition2_color,
            q_value=q_value,
            effect_size=effect_size,
            is_significant=is_significant,
            plot_cfg=plot_cfg,
            config=config,
        )
    
    n_trials = len(data_by_band[bands_in_order[0]][0]) if bands_in_order else 0
    n_tests = len(all_pvalues) if not use_precomputed else len(qvalues)
    
    title_parts = [f"{feature_label}: {label1} vs {label2} (Paired Comparison)"]
    
    info_parts = [f"Subject: {subject}"]
    if roi_name:
        roi_display = (
            roi_name.replace("_", " ").title() if roi_name != "all" else "All Channels"
        )
        info_parts.append(f"ROI: {roi_display}")
    info_parts.extend([
        f"N: {n_trials} trials",
        "Wilcoxon signed-rank",
        f"FDR: {n_significant}/{n_tests} significant (†=q<0.05)"
    ])
    title_parts.append(" | ".join(info_parts))
    
    fig.suptitle(
        "\n".join(title_parts),
        fontsize=plot_cfg.font.suptitle,
        fontweight="bold",
        y=1.02
    )
    
    plt.tight_layout()
    save_fig(
        fig, save_path,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches
    )
    plt.close(fig)
    
    if logger:
        logger.info(
            f"Saved {feature_label} paired comparison "
            f"({n_significant}/{n_tests} FDR significant)"
        )
