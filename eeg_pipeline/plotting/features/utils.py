"""Shared utilities for feature visualization plotting.

This module consolidates common functionality used across multiple plotting modules:
- FDR correction for multiple comparisons
- Effect size calculations (Cohen's d)
- Normality testing (Shapiro-Wilk)
- Config-driven accessors for bands and colors
- NamingSchema-based feature column selection
- Pre-computed statistics loading and matching
- Paired comparison plotting
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Union
from itertools import combinations
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy import stats

from eeg_pipeline.utils.config.loader import get_frequency_band_names, get_config_value
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.io.figures import get_band_color
from eeg_pipeline.utils.analysis.stats.fdr import fdr_bh
from eeg_pipeline.utils.analysis.stats.effect_size import cohens_d as _cohens_d
from eeg_pipeline.utils.analysis.stats.paired_comparisons import compute_paired_cohens_d
from eeg_pipeline.domain.features.naming import NamingSchema


###################################################################
# CONFIG-DRIVEN ACCESSORS
###################################################################

_DEFAULT_BANDS = ["delta", "theta", "alpha", "beta", "gamma"]


def _is_valid_dataframe(df: Any) -> bool:
    """Check if input is a non-empty DataFrame."""
    return df is not None and isinstance(df, pd.DataFrame) and not df.empty


def get_band_names(config: Any = None) -> List[str]:
    """Return frequency band names from config (falls back to defaults)."""
    bands = get_frequency_band_names(config)
    if bands:
        return list(bands)
    return _DEFAULT_BANDS


def get_band_colors(config: Any = None) -> Dict[str, str]:
    """Return band color mapping from config (falls back to defaults)."""
    return {band: get_band_color(band, config) for band in get_band_names(config)}


def get_condition_colors(config: Any = None) -> Dict[str, str]:
    """Return condition color mapping from config (condition_1/condition_2)."""
    plot_cfg = get_plot_config(config)
    return {
        "condition_1": plot_cfg.get_color("condition_1"),
        "condition_2": plot_cfg.get_color("condition_2"),
    }


def get_fdr_alpha(config: Any = None, default: float = 0.05) -> float:
    """Return FDR alpha threshold from config."""
    return float(get_config_value(config, "statistics.fdr_alpha", default))


def _format_count_range(counts: List[int]) -> str:
    """Format a list of sample counts as a compact exact value or range."""
    valid_counts = [int(count) for count in counts if int(count) > 0]
    if not valid_counts:
        return "0"
    count_min = min(valid_counts)
    count_max = max(valid_counts)
    if count_min == count_max:
        return str(count_min)
    return f"{count_min}-{count_max}"


def _t_critical_95(sample_size: int) -> float:
    """Return the two-sided 95% t critical value for a sample size."""
    n_samples = int(sample_size)
    if n_samples < 2:
        return 0.0
    return float(stats.t.ppf(0.975, df=n_samples - 1))


def _format_fdr_stars_legend() -> str:
    """Return the q-value legend used for FDR-significance star annotations."""
    return "(*q<.05, **q<.01, ***q<.001)"


def get_named_segments(
    df: pd.DataFrame,
    *,
    group: Optional[str] = None,
) -> List[str]:
    """Return available NamingSchema segments for a feature group."""
    if not _is_valid_dataframe(df):
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


def extract_multi_segment_data(
    df: pd.DataFrame,
    group: str,
    bands: List[str],
    segments: List[str],
    identifiers: Optional[List[str]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Extract feature data by band for multiple segments.
    
    Generic function that works for any feature type (power, aperiodic, connectivity, etc.).
    
    Args:
        df: Feature DataFrame with NamingSchema columns
        group: Feature group (e.g., 'power', 'aperiodic', 'connectivity')
        bands: List of frequency bands to extract
        segments: List of segment names (e.g., ['baseline', 'plateau', 'rampdown', 'rampup'])
        identifiers: Optional list of channel/ROI identifiers to filter by
    
    Returns:
        Dict mapping band -> {segment_name -> values array}
    """
    data_by_band: Dict[str, Dict[str, np.ndarray]] = {}
    identifier_set = set(identifiers) if identifiers else None
    
    for band in bands:
        segment_cols: Dict[str, List[str]] = {seg: [] for seg in segments}
        
        for col in df.columns:
            parsed = NamingSchema.parse(str(col))
            if not parsed.get("valid"):
                continue
            if parsed.get("group") != group:
                continue
            col_band = str(parsed.get("band") or "")
            if col_band != band:
                continue
            col_segment = str(parsed.get("segment") or "")
            if col_segment not in segment_cols:
                continue
            if identifier_set:
                col_id = str(parsed.get("identifier") or "")
                if col_id and col_id not in identifier_set:
                    continue
            segment_cols[col_segment].append(col)
        
        segment_series = {}
        for seg, cols in segment_cols.items():
            if cols:
                segment_series[seg] = df[cols].mean(axis=1)
        
        if len(segment_series) < 2:
            continue
        
        valid_mask = pd.Series(True, index=df.index)
        for series in segment_series.values():
            valid_mask &= series.notna()
        
        segment_values = {}
        for seg, series in segment_series.items():
            vals = series[valid_mask].values
            if len(vals) > 0:
                segment_values[seg] = vals
        
        if len(segment_values) >= 2:
            data_by_band[band] = segment_values
    
    return data_by_band


###################################################################
# FDR CORRECTION
###################################################################

def apply_fdr_correction(
    pvalues: List[float],
    alpha: Optional[float] = None,
    config: Any = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply FDR correction to multiple p-values using Benjamini-Hochberg method.
    
    Returns:
        Tuple of (rejected, qvalues, corrected_alpha)
    """
    if alpha is None:
        alpha = float(get_config_value(config, "statistics.fdr_alpha", 0.05))

    if not pvalues:
        return np.array([]), np.array([]), alpha
    
    pvalues_arr = np.asarray(pvalues, dtype=float)
    qvals = fdr_bh(pvalues_arr, alpha=alpha, config=config)
    rejected = np.isfinite(qvals) & (qvals < alpha)
    return rejected, qvals, np.asarray(alpha, dtype=float)


###################################################################
# EFFECT SIZE
###################################################################

def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups using pooled standard deviation.
    
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


###################################################################
# SIGNIFICANCE FORMATTING
###################################################################

def get_significance_color(significant: bool, config: Any = None) -> str:
    """Get color for significance annotation."""
    default_sig_color = "#d62728"
    default_nonsig_color = "#333333"
    
    plot_cfg = get_plot_config(config)
    style_colors = getattr(plot_cfg, "style", None)
    if style_colors and hasattr(style_colors, "colors"):
        sig_color = getattr(style_colors.colors, "significant", default_sig_color)
        nonsig_color = getattr(style_colors.colors, "nonsignificant", default_nonsig_color)
        return sig_color if significant else nonsig_color
    
    return default_sig_color if significant else default_nonsig_color


def _resolve_feature_type_name(feature_label: str) -> str:
    """Map plot labels to canonical feature family names."""
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
    return feature_type_map.get(feature_label, feature_label.lower())


###################################################################
# UNIFIED PAIRED COMPARISON PLOTTING
###################################################################


def _compute_paired_wilcoxon_stats(
    condition1_values: np.ndarray,
    condition2_values: np.ndarray,
) -> Tuple[float, float]:
    """Compute Wilcoxon signed-rank test and rank-biserial effect size for paired data.
    
    Returns:
        Tuple of (p_value, rank_biserial_r)
    """
    from scipy.stats import wilcoxon

    paired_condition1 = np.asarray(condition1_values, dtype=float)
    paired_condition2 = np.asarray(condition2_values, dtype=float)
    finite_mask = np.isfinite(paired_condition1) & np.isfinite(paired_condition2)
    differences = paired_condition2[finite_mask] - paired_condition1[finite_mask]

    if differences.size < 2 or np.allclose(differences, 0.0):
        return 1.0, 0.0

    _, p_value = wilcoxon(differences, zero_method="wilcox", alternative="two-sided")
    return float(p_value), _compute_paired_rank_biserial(differences)


def _compute_paired_rank_biserial(differences: np.ndarray) -> float:
    """Compute the matched-pairs rank-biserial correlation for paired differences."""
    from scipy.stats import rankdata

    finite_differences = np.asarray(differences, dtype=float)
    finite_differences = finite_differences[np.isfinite(finite_differences)]
    nonzero_mask = ~np.isclose(finite_differences, 0.0)
    nonzero_differences = finite_differences[nonzero_mask]

    if nonzero_differences.size == 0:
        return 0.0

    ranks = rankdata(np.abs(nonzero_differences), method="average")
    positive_rank_sum = float(np.sum(ranks[nonzero_differences > 0]))
    negative_rank_sum = float(np.sum(ranks[nonzero_differences < 0]))
    total_rank_sum = positive_rank_sum + negative_rank_sum
    if np.isclose(total_rank_sum, 0.0):
        return 0.0
    return float((positive_rank_sum - negative_rank_sum) / total_rank_sum)


def _format_qvalue_label(q_value: float) -> str:
    """Format q-values consistently for compact panel annotations."""
    return "q<.001" if q_value < 0.001 else f"q={q_value:.3f}"


def _compute_mean_ci(values: np.ndarray) -> Tuple[float, float]:
    """Return mean and 95% t-interval half-width for finite values."""
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return np.nan, 0.0
    mean_value = float(np.nanmean(finite_values))
    if finite_values.size < 2:
        return mean_value, 0.0
    sem = float(np.nanstd(finite_values, ddof=1) / np.sqrt(finite_values.size))
    return mean_value, _t_critical_95(finite_values.size) * sem


def _compute_paired_differences(
    condition1_values: np.ndarray,
    condition2_values: np.ndarray,
) -> np.ndarray:
    """Return finite paired differences using the plotting convention condition2 - condition1."""
    paired_condition1 = np.asarray(condition1_values, dtype=float)
    paired_condition2 = np.asarray(condition2_values, dtype=float)
    if paired_condition1.shape != paired_condition2.shape:
        return np.array([], dtype=float)

    differences = paired_condition2 - paired_condition1
    return differences[np.isfinite(differences)]


def _draw_paired_difference_summary(
    ax: Any,
    differences: np.ndarray,
    *,
    x_position: float,
    color: str,
) -> None:
    """Draw a compact paired-difference summary column on an existing comparison axis."""
    if differences.size == 0:
        return

    strip_left = x_position - 0.22
    strip_right = x_position + 0.22
    ax.axvspan(strip_left, strip_right, color=color, alpha=0.08, zorder=0)
    ax.hlines(
        0.0,
        strip_left,
        strip_right,
        color="0.55",
        linewidth=0.8,
        linestyle="--",
        alpha=0.8,
        zorder=1,
    )

    rng = np.random.default_rng(1234)
    jitter = rng.uniform(x_position - 0.08, x_position + 0.08, size=differences.size)
    ax.scatter(
        jitter,
        differences,
        s=14,
        color=color,
        alpha=0.7,
        zorder=12,
        linewidths=0,
    )

    mean_difference, ci_half_width = _compute_mean_ci(differences)
    ax.errorbar(
        [x_position],
        [mean_difference],
        yerr=[[ci_half_width], [ci_half_width]],
        fmt="o",
        color="black",
        markerfacecolor="white",
        markersize=5.0,
        capsize=2.5,
        linewidth=1.0,
        zorder=20,
    )


def _compute_axis_limits(values: np.ndarray) -> Tuple[float, float]:
    """Compute padded y-axis limits for a one-dimensional numeric series."""
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return -0.05, 0.05

    value_min = float(np.nanmin(finite_values))
    value_max = float(np.nanmax(finite_values))
    value_range = value_max - value_min
    if value_range <= 0.0:
        scale = max(abs(value_min), abs(value_max), 1.0)
        padding = 0.1 * scale
    else:
        padding = 0.1 * value_range
    return value_min - padding, value_max + padding


def _compute_difference_axis_limits(differences: np.ndarray) -> Tuple[float, float]:
    """Compute padded y-axis limits for paired differences, always including zero."""
    finite_differences = np.asarray(differences, dtype=float)
    finite_differences = finite_differences[np.isfinite(finite_differences)]
    if finite_differences.size == 0:
        return -0.05, 0.05

    value_min = min(float(np.nanmin(finite_differences)), 0.0)
    value_max = max(float(np.nanmax(finite_differences)), 0.0)
    value_range = value_max - value_min
    if value_range <= 0.0:
        scale = max(abs(value_min), abs(value_max), 1.0)
        padding = 0.1 * scale
    else:
        padding = 0.12 * value_range
    return value_min - padding, value_max + padding


def _style_difference_axis(
    delta_ax: Any,
    *,
    color: str,
) -> None:
    """Apply the shared style for the dedicated paired-difference axis."""
    delta_ax.set_facecolor((1.0, 1.0, 1.0, 0.0))
    delta_ax.patch.set_facecolor(color)
    delta_ax.patch.set_alpha(0.08)
    delta_ax.yaxis.tick_right()
    delta_ax.yaxis.set_label_position("right")
    delta_ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    delta_ax.tick_params(axis="y", labelsize=7, colors="0.35", pad=2)
    delta_ax.tick_params(axis="x", labelsize=9, colors="0.2")
    delta_ax.grid(axis="y", alpha=0.12, linewidth=0.5)
    delta_ax.grid(axis="x", visible=False)
    delta_ax.spines["top"].set_visible(False)
    delta_ax.spines["left"].set_visible(False)
    delta_ax.spines["right"].set_linewidth(0.8)
    delta_ax.spines["bottom"].set_linewidth(0.8)


def _plot_single_band_comparison(
    ax: Any,
    delta_ax: Any,
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
    """Plot a paired band comparison with raw values and an explicit delta summary.
    
    Args:
        ax: Matplotlib axes
        delta_ax: Dedicated axis for the paired-difference summary
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
        delta_ax.set_visible(False)
        return
    
    box_positions = [0.0, 1.0]
    condition1_values = np.asarray(condition1_values, dtype=float)
    condition2_values = np.asarray(condition2_values, dtype=float)
    differences = _compute_paired_differences(condition1_values, condition2_values)

    if differences.size > 0:
        pair_order = np.argsort(0.5 * (condition1_values + condition2_values))
        paired_condition1 = condition1_values[pair_order]
        paired_condition2 = condition2_values[pair_order]
        ax.plot(
            [box_positions[0], box_positions[1]],
            np.vstack([paired_condition1, paired_condition2]),
            color="0.75",
            linewidth=0.6,
            alpha=0.6,
            zorder=1,
        )

    colors = [condition1_color, condition2_color]
    data_list = [condition1_values, condition2_values]
    clip_bounds = [
        (-np.inf, box_positions[0]),
        (box_positions[1], np.inf),
    ]

    for j, (data, color) in enumerate(zip(data_list, colors)):
        pos = box_positions[j]
        clip_min, clip_max = clip_bounds[j]

        violin = ax.violinplot(data, positions=[pos], showextrema=False, widths=0.72)
        for body in violin["bodies"]:
            vertices = body.get_paths()[0].vertices
            vertices[:, 0] = np.clip(vertices[:, 0], clip_min, clip_max)
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.22)
            body.set_linewidth(0.8)

        box_center = pos - 0.11 if j == 0 else pos + 0.11
        ax.boxplot(
            data,
            positions=[box_center],
            widths=0.16,
            showfliers=False,
            patch_artist=True,
            boxprops=dict(facecolor="white", color=color, linewidth=1.0),
            medianprops=dict(color="black", linewidth=1.2),
            whiskerprops=dict(color=color, linewidth=1.0),
            capprops=dict(color=color, linewidth=1.0),
        )

        rng = np.random.default_rng(42 + j)
        jitter_low = pos - 0.22 if j == 0 else pos + 0.02
        jitter_high = pos - 0.03 if j == 0 else pos + 0.22
        jitter = rng.uniform(jitter_low, jitter_high, size=len(data))
        ax.scatter(
            jitter,
            data,
            s=12,
            color=color,
            alpha=0.65,
            zorder=10,
            linewidths=0,
        )

        mean_value, ci_half_width = _compute_mean_ci(data)
        ax.errorbar(
            [box_center],
            [mean_value],
            yerr=[[ci_half_width], [ci_half_width]],
            fmt="o",
            color="black",
            markersize=4.5,
            capsize=2.5,
            linewidth=1.0,
            zorder=20,
        )

    condition_axis_values = np.concatenate([condition1_values, condition2_values])
    main_y_min, main_y_max = _compute_axis_limits(condition_axis_values)
    ax.set_ylim(main_y_min, main_y_max)

    if differences.size > 0:
        _style_difference_axis(delta_ax, color=band_color)
        delta_ax.set_xlim(-0.45, 0.45)
        delta_y_min, delta_y_max = _compute_difference_axis_limits(differences)
        delta_ax.set_ylim(delta_y_min, delta_y_max)
        _draw_paired_difference_summary(
            delta_ax,
            differences,
            x_position=0.0,
            color=band_color,
        )
        delta_ax.set_xticks([0.0])
        delta_ax.set_xticklabels(["Δ"])
    else:
        delta_ax.set_visible(False)

    if q_value is not None and effect_size is not None:
        sig_marker = "  *" if is_significant else ""
        annotation_text = f"{_format_qvalue_label(q_value)} | r={effect_size:.2f}{sig_marker}"
        significance_color = get_significance_color(is_significant, config)
        ax.text(
            0.5,
            1.01,
            annotation_text,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=plot_cfg.font.small,
            color=significance_color,
            fontweight="bold" if is_significant else "normal",
        )

    ax.set_xlim(-0.4, 1.4)
    ax.set_xticks(box_positions)
    ax.set_xticklabels([label1, label2], fontsize=9)
    ax.set_title(band.capitalize(), fontweight="bold", color=band_color, pad=12)
    ax.yaxis.grid(True, alpha=0.18, linewidth=0.6)
    ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


def _load_condition_effects_files(
    stats_dir: Path,
    comparison_type: str,
    suffix: str,
) -> List[pd.DataFrame]:
    """Load condition effects files for a specific comparison type."""
    from eeg_pipeline.infra.tsv import read_tsv, read_table
    
    result_dfs = []
    condition_subdir = stats_dir / "condition_effects"
    search_dirs = [condition_subdir, stats_dir]
    
    base_filename = f"condition_effects_{comparison_type}"
    patterns = [
        f"{base_filename}{suffix}.parquet",
        f"{base_filename}*.parquet",
        f"{base_filename}{suffix}.tsv",
        f"{base_filename}*.tsv"
    ]
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for pattern in patterns:
            for path in search_dir.glob(pattern):
                if path.is_file():
                    if path.suffix.lower() == ".parquet":
                        df = read_table(path)
                    else:
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
    
    Loads condition_effects_window*.tsv, condition_effects_column*.tsv,
    or condition_effects_multigroup*.tsv files.
    """
    stats_dir_path = Path(stats_dir)
    result_dfs = []
    
    if comparison_type is None:
        comparison_types_to_try = ["window", "column", "multigroup"]
    elif comparison_type in ("window", "column", "multigroup"):
        comparison_types_to_try = [comparison_type]
    else:
        comparison_types_to_try = []
    
    for comp_type in comparison_types_to_try:
        loaded_dfs = _load_condition_effects_files(stats_dir_path, comp_type, suffix)
        result_dfs.extend(loaded_dfs)
    
    if result_dfs:
        combined_df = pd.concat(result_dfs, ignore_index=True)
        return _apply_stats_filters(
            combined_df, feature_type, comparison_type, condition1, condition2, roi_name
        )
    
    return None


def _normalize_condition_effects_df(
    df: pd.DataFrame,
    comparison_type: str,
) -> Optional[pd.DataFrame]:
    """Normalize condition effects DataFrame to expected schema.
    
    Converts from behavior pipeline output format to plotting expected format.
    """
    if not _is_valid_dataframe(df) or "feature" not in df.columns:
        return None
    
    result = df.copy()
    
    if "identifier" not in result.columns:
        result["identifier"] = result["feature"].astype(str)
    
    if "comparison_type" not in result.columns:
        result["comparison_type"] = comparison_type
    
    if "effect_size_d" not in result.columns:
        if "cohens_d" in result.columns:
            result["effect_size_d"] = pd.to_numeric(result["cohens_d"], errors="coerce")
        elif "hedges_g" in result.columns:
            result["effect_size_d"] = pd.to_numeric(result["hedges_g"], errors="coerce")
        else:
            result["effect_size_d"] = 0.0
    
    if "q_value" not in result.columns:
        p_series = result.get("p_value", pd.Series(dtype=float, index=result.index))
        result["q_value"] = pd.to_numeric(p_series, errors="coerce")
    
    if "significant_fdr" not in result.columns:
        q_vals = pd.to_numeric(result["q_value"], errors="coerce")
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
    if not _is_valid_dataframe(df):
        return None
    
    mask = pd.Series(True, index=df.index)
    
    # Filter by feature_type
    if feature_type:
        ft_lower = feature_type.lower()
        feature_mask: Optional[pd.Series] = None
        if "feature_type" in df.columns:
            feature_mask = df["feature_type"].astype(str).str.lower().str.contains(ft_lower, na=False)
        if "identifier" in df.columns:
            identifier_lower = df["identifier"].astype(str).str.lower()
            if _identifier_has_feature_family(identifier_lower, ft_lower):
                identifier_mask = identifier_lower.str.contains(ft_lower, na=False)
                feature_mask = identifier_mask if feature_mask is None else (feature_mask | identifier_mask)
        if feature_mask is not None:
            mask &= feature_mask
    
    # Filter by comparison_type
    if comparison_type and "comparison_type" in df.columns:
        mask &= df["comparison_type"].str.lower() == comparison_type.lower()
    
    # Filter by conditions (flexible matching)
    if condition1:
        c1_lower = condition1.lower()
        condition1_columns = [
            col for col in ("condition1", "window1", "condition_value1") if col in df.columns
        ]
        if condition1_columns:
            condition1_mask = pd.Series(False, index=df.index)
            for col in condition1_columns:
                condition1_mask |= df[col].astype(str).str.lower() == c1_lower
            mask &= condition1_mask
        else:
            mask &= False

    if condition2:
        c2_lower = condition2.lower()
        condition2_columns = [
            col for col in ("condition2", "window2", "condition_value2") if col in df.columns
        ]
        if condition2_columns:
            condition2_mask = pd.Series(False, index=df.index)
            for col in condition2_columns:
                condition2_mask |= df[col].astype(str).str.lower() == c2_lower
            mask &= condition2_mask
        else:
            mask &= False

    # Filter by ROI
    if roi_name and roi_name.lower() != "all" and "identifier" in df.columns:
        mask &= df["identifier"].str.lower().str.contains(roi_name.lower(), na=False)
    
    filtered = df[mask]
    return filtered if not filtered.empty else None


def _identifier_has_feature_family(
    identifier_lower: pd.Series,
    requested_feature_type: str,
) -> bool:
    """Return whether identifiers encode feature-family names for strict filtering."""
    known_prefixes = (
        "power_",
        "aperiodic_",
        "complexity_",
        "bursts_",
        "erds_",
        "spectral_",
        "ratios_",
        "asymmetry_",
        "itpc_",
        "pac_",
        "connectivity_",
    )
    if identifier_lower.str.startswith(known_prefixes).any():
        return True
    return identifier_lower.str.contains(requested_feature_type, na=False).any()


def get_precomputed_qvalues(
    precomputed_df: Optional[pd.DataFrame],
    feature_keys: List[str],
    roi_name: str = "all",
) -> Dict[str, Tuple[float, float, float, bool]]:
    """Extract q-values from pre-computed statistics DataFrame.
    
    Matches feature_keys in identifier column using flexible substring matching.
    Returns dict mapping feature_key to (p_value, q_value, effect_size_d, significant).
    """
    qvalues = {}
    
    if not _is_valid_dataframe(precomputed_df) or "identifier" not in precomputed_df.columns:
        return qvalues
    
    identifier_lower = precomputed_df["identifier"].str.lower()
    
    for key in feature_keys:
        key_lower = key.lower()
        match = pd.DataFrame()
        roi_lower = roi_name.lower() if roi_name else "all"

        if roi_lower != "all":
            pattern = f"{key_lower}_{roi_name.lower()}"
            match = precomputed_df[identifier_lower == pattern]
        if match.empty:
            exact_match = precomputed_df[identifier_lower == key_lower]
            if not exact_match.empty:
                match = exact_match

        if match.empty:
            key_mask = identifier_lower.str.contains(key_lower, na=False)
            if roi_lower != "all":
                key_mask &= identifier_lower.str.contains(roi_lower, na=False)
            match = precomputed_df[key_mask]
        
        if not match.empty:
            row = match.iloc[0]
            p = float(row.get("p_value", 1.0))
            q = float(row.get("q_value", p))
            d = float(row.get("effect_size_d", 0.0))
            sig = bool(row.get("significant_fdr", q < 0.05)) if "significant_fdr" in row else (q < 0.05)
            qvalues[key] = (p, q, d, sig)
    
    return qvalues


def get_precomputed_window_qvalues(
    precomputed_df: Optional[pd.DataFrame],
    bands: List[str],
    segments: List[str],
    *,
    roi_name: str = "all",
) -> Dict[Tuple[str, str, str], Tuple[float, float, float, bool]]:
    """Extract pairwise window q-values keyed by `(band, segment1, segment2)`."""
    qvalues: Dict[Tuple[str, str, str], Tuple[float, float, float, bool]] = {}
    if not _is_valid_dataframe(precomputed_df) or "identifier" not in precomputed_df.columns:
        return qvalues

    identifier_lower = precomputed_df["identifier"].astype(str).str.lower()
    window1_cols = [col for col in ("window1", "condition1", "condition_value1") if col in precomputed_df.columns]
    window2_cols = [col for col in ("window2", "condition2", "condition_value2") if col in precomputed_df.columns]
    if not window1_cols or not window2_cols:
        return qvalues

    roi_lower = str(roi_name or "all").lower()
    for band in bands:
        band_lower = str(band).lower()
        band_mask = identifier_lower.str.contains(band_lower, na=False)
        if roi_lower != "all":
            band_mask &= identifier_lower.str.contains(roi_lower, na=False)
        band_rows = precomputed_df[band_mask]
        if band_rows.empty:
            continue

        for seg1, seg2 in combinations(segments, 2):
            seg1_lower = str(seg1).lower()
            seg2_lower = str(seg2).lower()
            pair_mask = pd.Series(False, index=band_rows.index)
            for left_col in window1_cols:
                left_values = band_rows[left_col].astype(str).str.lower()
                for right_col in window2_cols:
                    right_values = band_rows[right_col].astype(str).str.lower()
                    pair_mask |= (left_values == seg1_lower) & (right_values == seg2_lower)
                    pair_mask |= (left_values == seg2_lower) & (right_values == seg1_lower)

            match = band_rows[pair_mask]
            if match.empty:
                continue

            row = match.iloc[0]
            p = float(row.get("p_value", 1.0))
            q = float(row.get("q_value", p))
            d = float(row.get("effect_size_d", 0.0))
            sig = bool(row.get("significant_fdr", q < 0.05)) if "significant_fdr" in row else (q < 0.05)
            qvalues[(band, seg1, seg2)] = (p, q, d, sig)

    return qvalues


def compute_or_load_column_stats(
    stats_dir: Optional[Union[Path, str]],
    feature_type: str,
    feature_keys: List[str],
    cell_data: Dict[int, Optional[Dict[str, np.ndarray]]],
    config: Any = None,
    logger: Any = None,
    *,
    roi_name: str = "all",
    require_precomputed_stats: bool = False,
    precomputed_df: Optional[pd.DataFrame] = None,
) -> Tuple[Dict[int, Tuple[float, float, float, bool]], int, bool]:
    """Compute or load column comparison statistics.
    
    Tries to load pre-computed stats first, falls back to on-the-fly computation.
    
    Returns:
        Tuple of (qvalues dict, n_significant, use_precomputed)
    """
    qvalues: Dict[int, Tuple[float, float, float, bool]] = {}
    n_significant = 0
    use_precomputed = False
    
    # Try to load pre-computed stats
    precomputed = precomputed_df
    if precomputed is None and stats_dir is not None:
        precomputed = load_precomputed_paired_stats(
            stats_dir=stats_dir,
            feature_type=feature_type,
            comparison_type="column",
            roi_name=roi_name,
        )

    if precomputed is not None and not precomputed.empty:
            use_precomputed = True
            if logger and hasattr(logger, "info"):
                logger.info(f"Using pre-computed column stats for {feature_type} ({len(precomputed)} entries)")
            
            # Map pre-computed stats to feature_keys
            precomputed_qvals = get_precomputed_qvalues(precomputed, feature_keys, roi_name=roi_name)
            
            for col_idx, key in enumerate(feature_keys):
                if key in precomputed_qvals:
                    qvalues[col_idx] = precomputed_qvals[key]
            
            n_significant = sum(1 for v in qvalues.values() if v[3])
            return qvalues, n_significant, use_precomputed

    if require_precomputed_stats:
        raise ValueError(
            f"{feature_type} column comparison requires pre-computed statistics; "
            "on-the-fly row-wise tests are disabled for this plot."
        )
    
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
                from scipy.stats import mannwhitneyu
                _, p_value = mannwhitneyu(
                    condition1_values, condition2_values, alternative="two-sided"
                )
                effect_size = compute_cohens_d(condition1_values, condition2_values)
                all_pvals.append(p_value)
                pvalue_keys.append((col_idx, p_value, effect_size))
            except (ValueError, RuntimeError) as e:
                if logger:
                    logger.debug(f"Failed to compute stats for column {col_idx}: {e}")
    
    if all_pvals:
        rejected, qvals, _ = apply_fdr_correction(all_pvals, config=config)
        for i, (col_idx, p, d) in enumerate(pvalue_keys):
            qvalues[col_idx] = (p, qvals[i], d, rejected[i])
        n_significant = int(np.sum(rejected))
    
    return qvalues, n_significant, use_precomputed


def _get_significance_stars(q_value: float) -> str:
    """Return significance stars based on q-value thresholds."""
    if q_value < 0.001:
        return "***"
    elif q_value < 0.01:
        return "**"
    elif q_value < 0.05:
        return "*"
    return "ns"


def _draw_significance_bracket(
    ax: Any,
    x1: float,
    x2: float,
    y: float,
    text: str,
    is_significant: bool,
    bracket_height: float = 0.02,
    text_offset: float = 0.01,
) -> float:
    """Draw a significance bracket with text annotation.
    
    Returns the y position of the top of the bracket for stacking.
    """
    color = "#d62728" if is_significant else "#666666"
    fontweight = "bold" if is_significant else "normal"
    
    ax.plot([x1, x1, x2, x2], [y, y + bracket_height, y + bracket_height, y],
            color=color, linewidth=1.2)
    ax.text((x1 + x2) / 2, y + bracket_height + text_offset, text,
            ha="center", va="bottom", fontsize=8, color=color, fontweight=fontweight)
    
    return y + bracket_height + text_offset + 0.03


def _summarize_paired_sample_counts(
    data_by_band: Dict[str, Tuple[np.ndarray, np.ndarray]],
    sample_unit: str,
) -> str:
    """Summarize complete paired sample counts across bands."""
    paired_counts = [len(values1) for values1, values2 in data_by_band.values() if len(values1) == len(values2)]
    return f"N: {_format_count_range(paired_counts)} {sample_unit}"


def _summarize_multi_window_sample_counts(
    data_by_band: Dict[str, Dict[str, np.ndarray]],
    sample_unit: str,
) -> str:
    """Summarize per-window sample counts across bands for multi-window plots."""
    counts = [
        len(values)
        for segment_data in data_by_band.values()
        for values in segment_data.values()
    ]
    return f"N per window: {_format_count_range(counts)} {sample_unit}"


def _get_displayed_multi_window_pairs(
    available_segments: List[str],
) -> List[Tuple[str, str]]:
    """Return the window pairs actually drawn on the figure."""
    if len(available_segments) < 2:
        return []
    return list(combinations(available_segments, 2))


def _filter_displayed_multi_window_qvalues(
    qvalues: Dict[Tuple[str, str, str], Tuple[float, float, float, bool]],
    bands_in_order: List[str],
    segments: List[str],
) -> Dict[Tuple[str, str, str], Tuple[float, float, float, bool]]:
    """Keep only the comparisons that are rendered on the figure."""
    displayed_pairs = set(_get_displayed_multi_window_pairs(segments))
    filtered: Dict[Tuple[str, str, str], Tuple[float, float, float, bool]] = {}
    for band in bands_in_order:
        for seg1, seg2 in displayed_pairs:
            key = (band, seg1, seg2)
            reverse_key = (band, seg2, seg1)
            if key in qvalues:
                filtered[key] = qvalues[key]
            elif reverse_key in qvalues:
                filtered[key] = qvalues[reverse_key]
    return filtered


def _format_displayed_multi_window_pairs(segments: List[str]) -> str:
    """Format the displayed comparison pairs for figure metadata."""
    displayed_pairs = _get_displayed_multi_window_pairs(segments)
    return ", ".join(f"{seg1} vs {seg2}" for seg1, seg2 in displayed_pairs)


def plot_multi_window_comparison(
    data_by_band: Dict[str, Dict[str, np.ndarray]],
    subject: str,
    save_path: Union[Path, str],
    feature_label: str,
    segments: List[str],
    config: Any = None,
    logger: Any = None,
    *,
    roi_name: Optional[str] = None,
    stats_dir: Optional[Union[Path, str]] = None,
    precomputed_stats: Optional[pd.DataFrame] = None,
    sample_unit: str = "trials",
    comparison_dimension_name: str = "windows",
    require_precomputed_stats: bool = False,
) -> None:
    """Multi-window paired comparison plot with significance brackets.
    
    Creates a figure with one subplot per frequency band, showing all windows
    as grouped boxplots with pairwise comparison brackets and significance asterisks.
    
    Args:
        data_by_band: Dict mapping band -> {segment_name -> values array}
        subject: Subject identifier
        save_path: Path to save figure
        feature_label: Label for the feature type (e.g., "Band Power")
        segments: List of segment names in display order
        config: Configuration object
        logger: Logger instance
        roi_name: ROI name for title
        stats_dir: Directory containing pre-computed statistics
    """
    import matplotlib.pyplot as plt
    from scipy.stats import wilcoxon
    from eeg_pipeline.plotting.io.figures import save_fig
    
    if not data_by_band:
        if logger:
            logger.warning(f"No data provided for {feature_label} multi-window comparison")
        return
    
    band_order = get_band_names(config)
    bands_in_order = [b for b in band_order if b in data_by_band]
    bands_in_order += [b for b in data_by_band if b not in bands_in_order]
    
    if not bands_in_order:
        return
    
    n_bands = len(bands_in_order)
    n_segments = len(segments)
    n_pairs = n_segments * (n_segments - 1) // 2
    
    plot_cfg = get_plot_config(config)
    band_colors = get_band_colors(config)
    
    segment_colors = plt.cm.Set2(np.linspace(0, 1, max(n_segments, 3)))
    segment_color_map = {seg: segment_colors[i] for i, seg in enumerate(segments)}
    
    feature_type = _resolve_feature_type_name(feature_label)
    if precomputed_stats is None and stats_dir is not None:
        precomputed_stats = load_precomputed_paired_stats(
            stats_dir=stats_dir,
            feature_type=feature_type,
            comparison_type="window",
            roi_name=roi_name,
        )

    use_precomputed = precomputed_stats is not None and not precomputed_stats.empty
    if require_precomputed_stats and not use_precomputed:
        raise ValueError(
            f"{feature_label} multi-window comparison requires pre-computed statistics; "
            "on-the-fly row-wise tests are disabled for this plot."
        )

    qvalues: Dict[Tuple[str, str, str], Tuple[float, float, float, bool]] = {}
    n_significant = 0
    if use_precomputed:
        qvalues = get_precomputed_window_qvalues(
            precomputed_stats,
            bands_in_order,
            segments,
            roi_name=roi_name or "all",
        )
        if require_precomputed_stats and not qvalues:
            raise ValueError(
                f"{feature_label} multi-window comparison could not match the requested "
                "bands/windows in the pre-computed statistics table."
            )
        qvalues = _filter_displayed_multi_window_qvalues(qvalues, bands_in_order, segments)
        n_significant = sum(1 for stats_tuple in qvalues.values() if stats_tuple[3])
    else:
        all_pvalues = []
        pvalue_keys = []
        min_samples = int(get_config_value(config, "behavior_analysis.min_samples.default", 5))
        displayed_pairs = set(_get_displayed_multi_window_pairs(segments))

        for band in bands_in_order:
            segment_data = data_by_band[band]
            for seg1, seg2 in combinations(segments, 2):
                if (seg1, seg2) not in displayed_pairs:
                    continue
                if seg1 not in segment_data or seg2 not in segment_data:
                    continue
                v1, v2 = segment_data[seg1], segment_data[seg2]
                if len(v1) >= min_samples and len(v2) >= min_samples and len(v1) == len(v2):
                    try:
                        _, p_value = wilcoxon(v2, v1)
                        effect_size = compute_paired_cohens_d(v1, v2)
                        all_pvalues.append(p_value)
                        pvalue_keys.append((band, seg1, seg2, p_value, effect_size))
                    except (ValueError, RuntimeError):
                        pass

        if all_pvalues:
            rejected, qvals, _ = apply_fdr_correction(all_pvalues, config=config)
            for i, (band, seg1, seg2, p_value, effect_size) in enumerate(pvalue_keys):
                qvalues[(band, seg1, seg2)] = (p_value, qvals[i], effect_size, rejected[i])
            n_significant = int(np.sum(rejected))
    
    fig_width_per_band = 2.5 + 0.5 * n_segments
    fig_height = 5 + 0.4 * n_pairs
    fig, axes = plt.subplots(
        1, n_bands, figsize=(fig_width_per_band * n_bands, fig_height), squeeze=False
    )
    
    for band_idx, band in enumerate(bands_in_order):
        ax = axes.flatten()[band_idx]
        segment_data = data_by_band[band]
        
        available_segments = [s for s in segments if s in segment_data]
        if not available_segments:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=plot_cfg.font.title, color="gray")
            ax.set_xticks([])
            continue
        
        positions = list(range(len(available_segments)))
        
        box_data = [segment_data[seg] for seg in available_segments]
        for i, (seg, data) in enumerate(zip(available_segments, box_data)):
            color = segment_color_map[seg]
            pos = positions[i]
            
            violin = ax.violinplot(data, positions=[pos], showextrema=False, widths=0.6)
            for body in violin["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.22)
                body.set_linewidth(0.8)
            
            # Boxplot
            ax.boxplot(data, positions=[pos - 0.1], widths=0.1, showfliers=False,
                       patch_artist=True, boxprops=dict(facecolor="white", color=color),
                       medianprops=dict(color="black", linewidth=1.5),
                       whiskerprops=dict(color=color), capprops=dict(color=color))
            
            # Jitter scatter
            rng = np.random.default_rng(42 + i)
            jitter = rng.uniform(pos - 0.12, pos + 0.12, size=len(data))
            ax.scatter(jitter, data, s=10, color=color, alpha=0.55, zorder=10, linewidths=0)
        
        all_values = np.concatenate([segment_data[seg] for seg in available_segments])
        y_min = np.nanmin(all_values)
        y_max = np.nanmax(all_values)
        y_range = y_max - y_min if y_max > y_min else 0.1
        
        bracket_y = y_max + 0.08 * y_range
        bracket_spacing = 0.12 * y_range

        pair_list = list(combinations(range(len(available_segments)), 2))
        drawn_brackets = 0
        for pair_idx, (i, j) in enumerate(pair_list):
            seg1, seg2 = available_segments[i], available_segments[j]
            key = (band, seg1, seg2)
            if key not in qvalues:
                key = (band, seg2, seg1)
            
            if key in qvalues:
                _, q_val, d, is_sig = qvalues[key]
                stars = _get_significance_stars(q_val)
                text = f"{stars}" if is_sig else "ns"
                
                current_y = bracket_y + drawn_brackets * bracket_spacing
                _draw_significance_bracket(
                    ax, positions[i], positions[j], current_y, text, is_sig,
                    bracket_height=0.02 * y_range, text_offset=0.01 * y_range
                )
                drawn_brackets += 1
        
        top_bracket_y = bracket_y + drawn_brackets * bracket_spacing
        ax.set_ylim(y_min - 0.1 * y_range, top_bracket_y + 0.1 * y_range)
        
        ax.set_xticks(positions)
        ax.set_xticklabels([s.capitalize() for s in available_segments], fontsize=9, rotation=30, ha="right")
        ax.set_title(band.capitalize(), fontweight="bold", color=band_colors.get(band, "gray"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    n_tests = len(qvalues)
    
    roi_display = roi_name.replace("_", " ").title() if roi_name and roi_name != "all" else "All Channels"
    
    dimension_name = str(comparison_dimension_name).strip() or "windows"
    displayed_pair_text = _format_displayed_multi_window_pairs(segments)
    title_parts = [
        f"{feature_label}: Multi-{dimension_name.capitalize()} Comparison "
        f"({n_segments} {dimension_name}, {len(_get_displayed_multi_window_pairs(segments))} displayed pairs)"
    ]
    info_parts = [
        f"Subject: {subject}",
        f"ROI: {roi_display}",
        _summarize_multi_window_sample_counts(data_by_band, sample_unit),
        "Wilcoxon signed-rank",
        f"Displayed comparisons: {displayed_pair_text}",
        f"FDR: {n_significant}/{n_tests} significant {_format_fdr_stars_legend()}"
    ]
    title_parts.append(" | ".join(info_parts))
    
    fig.suptitle(
        title_parts[0],
        fontsize=plot_cfg.font.suptitle,
        fontweight="bold",
        y=0.99,
    )
    footer = " | ".join(info_parts)
    save_fig(
        fig, save_path,
        footer=footer,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
        tight_layout_rect=(0, 0.04, 1, 0.97),
        config=config
    )
    
    if logger:
        logger.info(
            f"Saved {feature_label} multi-window comparison "
            f"({n_significant}/{n_tests} FDR significant)"
        )


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
    sample_unit: str = "trials",
    require_precomputed_stats: bool = False,
) -> None:
    """Unified paired comparison plot.
    
    Creates single-row figure with one subplot per frequency band.
    Uses pre-computed statistics if provided, otherwise computes on-the-fly.
    """
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
    condition_colors = get_condition_colors(config)
    condition1_color = condition_colors.get("condition_1", "#5a7d9a")
    condition2_color = condition_colors.get("condition_2", "#c44e52")
    
    feature_type = _resolve_feature_type_name(feature_label)
    
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
    if require_precomputed_stats and not use_precomputed:
        raise ValueError(
            f"{feature_label} paired comparison requires pre-computed statistics; "
            "on-the-fly row-wise tests are disabled for this plot."
        )
    
    if use_precomputed:
        qvalues = get_precomputed_qvalues(precomputed_stats, bands_in_order, roi_name or "all")
        if require_precomputed_stats and not qvalues:
            raise ValueError(
                f"{feature_label} paired comparison could not match the requested bands "
                "in the pre-computed statistics table."
            )
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
        
        if all_pvalues:
            rejected, qvals, _ = apply_fdr_correction(all_pvalues, config=config)
            for i, (band, p_value, effect_size) in enumerate(pvalue_keys):
                qvalues[band] = (p_value, qvals[i], effect_size, rejected[i])
            n_significant = int(np.sum(rejected))
    
    fig_width_per_band = 4.2
    fig_height = 5
    fig = plt.figure(figsize=(fig_width_per_band * n_bands, fig_height))
    outer_grid = fig.add_gridspec(1, n_bands, wspace=0.34)
    
    for band_idx, band in enumerate(bands_in_order):
        band_grid = outer_grid[0, band_idx].subgridspec(1, 2, width_ratios=[4.8, 1.0], wspace=0.08)
        ax = fig.add_subplot(band_grid[0, 0])
        delta_ax = fig.add_subplot(band_grid[0, 1])
        condition1_values, condition2_values = data_by_band[band]
        
        q_value = None
        effect_size = None
        is_significant = False
        if band in qvalues:
            _, q_value, effect_size, is_significant = qvalues[band]
        
        _plot_single_band_comparison(
            ax=ax,
            delta_ax=delta_ax,
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
    
    n_tests = len(qvalues)
    
    title_parts = [f"{feature_label}: {label1} vs {label2}"]

    info_parts = [f"Subject: {subject}"]
    if roi_name:
        roi_display = (
            roi_name.replace("_", " ").title() if roi_name != "all" else "All Channels"
        )
        info_parts.append(f"ROI: {roi_display}")
    info_parts.extend([
        _summarize_paired_sample_counts(data_by_band, sample_unit),
        "Wilcoxon signed-rank",
        f"Δ: {label2} - {label1}",
        f"FDR: {n_significant}/{n_tests} significant"
    ])
    fig.suptitle(
        title_parts[0],
        fontsize=plot_cfg.font.suptitle,
        fontweight="bold",
        y=0.99,
    )
    footer = " | ".join(info_parts)
    save_fig(
        fig, save_path,
        footer=footer,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
        tight_layout_rect=(0, 0.04, 1, 0.97),
        config=config
    )
    
    if logger:
        logger.info(
            f"Saved {feature_label} paired comparison "
            f"({n_significant}/{n_tests} FDR significant)"
        )


def load_multigroup_stats(
    stats_dir: Union[Path, str],
    feature_type: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Load pre-computed multi-group comparison statistics.
    
    Args:
        stats_dir: Directory containing pre-computed statistics
        feature_type: Optional filter by feature type
        
    Returns:
        DataFrame with multi-group stats or None if not found
    """
    stats = load_precomputed_paired_stats(
        stats_dir=stats_dir,
        feature_type=feature_type,
        comparison_type="multigroup",
    )
    
    if stats is None or stats.empty:
        return None
    
    required_cols = {"feature", "group1", "group2", "q_value", "significant_fdr"}
    if not required_cols.issubset(set(stats.columns)):
        return None
    
    return stats


def resolve_complete_multigroup_plot_groups(
    events_df: pd.DataFrame,
    config: Any,
    *,
    context: str,
    minimum_groups: int = 2,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Resolve configured multi-group labels and require every group to have trials."""
    from eeg_pipeline.utils.analysis.events import extract_multi_group_masks

    multi_group_info = extract_multi_group_masks(events_df, config, require_enabled=True)
    if not multi_group_info:
        raise ValueError(f"{context} could not resolve configured multi-group masks.")

    masks_dict, group_labels = multi_group_info
    resolved_masks: Dict[str, np.ndarray] = {}
    missing_labels: List[str] = []

    for label in group_labels:
        mask = masks_dict.get(label)
        if mask is None:
            missing_labels.append(str(label))
            continue

        mask_bool = np.asarray(mask, dtype=bool)
        if int(mask_bool.sum()) == 0:
            missing_labels.append(str(label))
            continue

        resolved_masks[str(label)] = mask_bool

    if missing_labels:
        raise ValueError(
            f"{context}: missing configured group(s) with matching trials: {missing_labels}"
        )

    ordered_labels = [str(label) for label in group_labels]
    if len(ordered_labels) < minimum_groups:
        raise ValueError(
            f"{context} requires at least {minimum_groups} configured groups with matching trials."
        )

    return resolved_masks, ordered_labels


def plot_multi_group_column_comparison(
    data_by_band: Dict[str, Dict[str, np.ndarray]],
    subject: str,
    save_path: Union[Path, str],
    feature_label: str,
    groups: List[str],
    config: Any = None,
    logger: Any = None,
    *,
    roi_name: Optional[str] = None,
    stats_dir: Optional[Union[Path, str]] = None,
    multigroup_stats: Optional[pd.DataFrame] = None,
    stats_match_terms: Optional[Dict[str, Tuple[str, ...]]] = None,
) -> None:
    """Multi-group unpaired comparison plot with significance brackets.
    
    Requires pre-computed statistics from the behavior pipeline. Will not plot
    if stats_dir is not provided or stats are not found. Run the behavior
    pipeline with 3+ comparison values to generate multi-group stats.
    
    When more than 3 groups, creates one plot per band to avoid overcrowding.
    
    Args:
        data_by_band: Dict mapping band -> {group_name -> values array}
        subject: Subject identifier
        save_path: Path to save figure
        feature_label: Label for the feature type (e.g., "Band Power")
        groups: List of group names in display order
        config: Configuration object
        logger: Logger instance
        roi_name: ROI name for title
        stats_dir: Directory containing pre-computed statistics (required if multigroup_stats not provided)
        multigroup_stats: Pre-loaded multigroup stats DataFrame (optional, avoids reloading)
        stats_match_terms: Optional mapping from plotted feature key to identifier match terms
    """
    import matplotlib.pyplot as plt
    
    if not data_by_band:
        return
    
    required_stat_columns = {"feature", "group1", "group2", "q_value", "significant_fdr"}

    if multigroup_stats is None:
        if stats_dir is None:
            raise ValueError(
                f"Multi-group comparison for {feature_label} requires pre-computed stats. "
                "Run behavior pipeline with 3+ comparison values first."
            )
        
        multigroup_stats = load_multigroup_stats(stats_dir)

        if multigroup_stats is None or multigroup_stats.empty:
            raise ValueError(
                f"No pre-computed multi-group stats found for {feature_label}. "
                "Run behavior pipeline with 3+ comparison values first."
            )
    else:
        multigroup_stats = multigroup_stats.copy()

    missing_columns = required_stat_columns.difference(set(multigroup_stats.columns))
    if missing_columns:
        raise ValueError(
            "Invalid multigroup stats table: missing required columns "
            f"{sorted(missing_columns)}."
        )

    if multigroup_stats.empty and stats_dir is not None:
        raise ValueError(
            f"No pre-computed multi-group stats found for {feature_label}. "
            "Run behavior pipeline with 3+ comparison values first."
        )
    
    plot_cfg = get_plot_config(config)
    bands_in_order = list(data_by_band.keys())
    n_bands = len(bands_in_order)
    
    if n_bands == 0:
        return
    
    n_groups = len(groups)
    group_colors = plt.cm.Set2(np.linspace(0, 1, max(n_groups, 3)))
    
    qvalues_map, total_significant, total_tests = _resolve_multigroup_qvalues_map(
        data_by_band=data_by_band,
        groups=groups,
        multigroup_stats=multigroup_stats,
        feature_keys=bands_in_order,
        roi_name=roi_name,
        stats_match_terms=stats_match_terms,
    )
    
    # Decide layout: separate plots per band if more than 3 groups
    separate_plots_per_band = n_groups > 3
    
    if separate_plots_per_band:
        _plot_multi_group_separate_bands(
            data_by_band=data_by_band,
            bands_in_order=bands_in_order,
            groups=groups,
            group_colors=group_colors,
            qvalues_map=qvalues_map,
            subject=subject,
            save_path=save_path,
            feature_label=feature_label,
            plot_cfg=plot_cfg,
            config=config,
            logger=logger,
            roi_name=roi_name,
            total_significant=total_significant,
            total_tests=total_tests,
        )
    else:
        _plot_multi_group_combined(
            data_by_band=data_by_band,
            bands_in_order=bands_in_order,
            groups=groups,
            group_colors=group_colors,
            qvalues_map=qvalues_map,
            subject=subject,
            save_path=save_path,
            feature_label=feature_label,
            plot_cfg=plot_cfg,
            config=config,
            logger=logger,
            roi_name=roi_name,
            total_significant=total_significant,
            total_tests=total_tests,
        )


def _plot_multi_group_separate_bands(
    data_by_band: Dict[str, Dict[str, np.ndarray]],
    bands_in_order: List[str],
    groups: List[str],
    group_colors: np.ndarray,
    qvalues_map: Dict[Tuple[int, str, str], Tuple[float, bool]],
    subject: str,
    save_path: Union[Path, str],
    feature_label: str,
    plot_cfg: Any,
    config: Any,
    logger: Any,
    roi_name: Optional[str],
    total_significant: int,
    total_tests: int,
) -> None:
    """Plot each band as a separate figure (for >3 groups).
    
    Only draws brackets for significant comparisons to avoid clutter.
    """
    import matplotlib.pyplot as plt
    from itertools import combinations
    from eeg_pipeline.plotting.io.figures import save_fig
    
    save_path = Path(save_path)
    base_stem = save_path.stem
    save_dir = save_path.parent
    
    for band_idx, band in enumerate(bands_in_order):
        band_data = data_by_band[band]
        band_color = get_band_color(band, config)
        
        available_groups = [g for g in groups if g in band_data and len(band_data[g]) > 0]
        
        if not available_groups:
            continue
        
        # Collect significant pairs for this band
        sig_pairs = []
        band_sig = 0
        band_tests = 0
        for g1, g2 in combinations(available_groups, 2):
            key = (band_idx, g1, g2)
            if key not in qvalues_map:
                key = (band_idx, g2, g1)
            
            if key in qvalues_map:
                qval, is_sig = qvalues_map[key]
                band_tests += 1
                if is_sig:
                    band_sig += 1
                    sig_pairs.append((g1, g2, qval))
        
        # Figure size based on number of significant brackets (not all pairs)
        n_sig_brackets = len(sig_pairs)
        fig_width = max(1.2 * len(available_groups), 6)
        fig_height = 5 + 0.25 * n_sig_brackets
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        positions = list(range(len(available_groups)))
        box_data = [band_data[g] for g in available_groups]
        
        for i, (g, data) in enumerate(zip(available_groups, box_data)):
            color = group_colors[groups.index(g) % len(group_colors)]
            pos = positions[i]
            
            violin = ax.violinplot(data, positions=[pos], showextrema=False, widths=0.6)
            for body in violin["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.22)
                body.set_linewidth(0.8)
            
            # Boxplot
            ax.boxplot(data, positions=[pos - 0.1], widths=0.1, showfliers=False,
                       patch_artist=True, boxprops=dict(facecolor="white", color=color),
                       medianprops=dict(color="black", linewidth=1.5),
                       whiskerprops=dict(color=color), capprops=dict(color=color))
            
            # Scatter jitter
            rng = np.random.default_rng(42 + i)
            jitter = rng.uniform(pos - 0.12, pos + 0.12, size=len(data))
            ax.scatter(jitter, data, s=15, color=color, alpha=0.5, zorder=3, linewidths=0)
        
        all_vals = np.concatenate([band_data[g] for g in available_groups])
        y_min, y_max = np.nanmin(all_vals), np.nanmax(all_vals)
        y_range = y_max - y_min if y_max > y_min else 0.1
        
        # Only draw significant brackets
        bracket_y = y_max + 0.05 * y_range
        bracket_step = 0.08 * y_range
        
        for g1, g2, qval in sig_pairs:
            text = _get_significance_stars(qval)
            x1 = available_groups.index(g1)
            x2 = available_groups.index(g2)
            
            bracket_y = _draw_significance_bracket(
                ax, x1, x2, bracket_y, text, is_significant=True,
                bracket_height=0.02 * y_range
            )
            bracket_y += bracket_step * 0.3
        
        # Set y limits based on actual bracket extent
        top_margin = 0.15 * y_range if not sig_pairs else 0.05 * y_range
        ax.set_ylim(y_min - 0.1 * y_range, bracket_y + top_margin)
        
        ax.set_title(f"{band.capitalize()} Band", fontsize=plot_cfg.font.title, 
                    fontweight="bold", color=band_color)
        ax.set_xticks(positions)
        ax.set_xticklabels([str(g) for g in available_groups], 
                          fontsize=plot_cfg.font.small, rotation=45, ha="right")
        ax.set_ylabel(feature_label, fontsize=plot_cfg.font.label)
        ax.tick_params(axis="y", labelsize=plot_cfg.font.small)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        title_parts = [f"{feature_label}: Multi-Group Comparison"]
        info_parts = [f"Subject: {subject}"]
        if roi_name:
            roi_display = roi_name.replace("_", " ").title() if roi_name != "all" else "All Channels"
            info_parts.append(f"ROI: {roi_display}")
        info_parts.extend([
            f"{len(groups)} groups",
            f"FDR: {band_sig}/{band_tests} sig {_format_fdr_stars_legend()}"
        ])
        title_parts.append(" | ".join(info_parts))
        
        fig.suptitle("\n".join(title_parts), fontsize=plot_cfg.font.suptitle,
                    fontweight="bold", y=1.02)
        
        plt.tight_layout()
        band_save_path = save_dir / f"{base_stem}_band-{band}"
        save_fig(fig, band_save_path, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
                bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches, config=config)
        plt.close(fig)
    
    if logger:
        logger.info(f"Saved {feature_label} multi-group comparison per band "
                   f"({total_significant}/{total_tests} total FDR significant)")


def _plot_multi_group_combined(
    data_by_band: Dict[str, Dict[str, np.ndarray]],
    bands_in_order: List[str],
    groups: List[str],
    group_colors: np.ndarray,
    qvalues_map: Dict[Tuple[int, str, str], Tuple[float, bool]],
    subject: str,
    save_path: Union[Path, str],
    feature_label: str,
    plot_cfg: Any,
    config: Any,
    logger: Any,
    roi_name: Optional[str],
    total_significant: int,
    total_tests: int,
) -> None:
    """Plot all bands in a single row (for <=3 groups)."""
    import matplotlib.pyplot as plt
    from itertools import combinations
    from eeg_pipeline.plotting.io.figures import save_fig
    
    n_bands = len(bands_in_order)
    
    fig_width = max(3 * n_bands, 8)
    fig, axes = plt.subplots(1, n_bands, figsize=(fig_width, 5), squeeze=False)
    axes = axes.flatten()
    
    for band_idx, band in enumerate(bands_in_order):
        ax = axes[band_idx]
        band_data = data_by_band[band]
        band_color = get_band_color(band, config)
        
        available_groups = [g for g in groups if g in band_data and len(band_data[g]) > 0]
        
        if not available_groups:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                   transform=ax.transAxes, fontsize=plot_cfg.font.title, color="gray")
            ax.set_title(band.capitalize(), fontsize=plot_cfg.font.title, fontweight="bold")
            ax.set_xticks([])
            continue
        
        positions = list(range(len(available_groups)))
        box_data = [band_data[g] for g in available_groups]
        
        bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True)
        
        for i, (box, g) in enumerate(zip(bp["boxes"], available_groups)):
            box.set_facecolor(group_colors[groups.index(g) % len(group_colors)])
            box.set_alpha(0.7)
        
        rng = np.random.default_rng(42)
        for i, g in enumerate(available_groups):
            vals = band_data[g]
            jitter = rng.uniform(-0.15, 0.15, len(vals))
            ax.scatter(i + jitter, vals, 
                      c=[group_colors[groups.index(g) % len(group_colors)]], 
                      alpha=0.5, s=15, zorder=3)
        
        all_vals = np.concatenate([band_data[g] for g in available_groups])
        y_min, y_max = np.nanmin(all_vals), np.nanmax(all_vals)
        y_range = y_max - y_min if y_max > y_min else 0.1
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.5 * y_range)
        
        bracket_y = y_max + 0.05 * y_range
        bracket_step = 0.12 * y_range
        
        for g1, g2 in combinations(available_groups, 2):
            key = (band_idx, g1, g2)
            if key not in qvalues_map:
                key = (band_idx, g2, g1)
            
            if key in qvalues_map:
                qval, is_sig = qvalues_map[key]
                text = _get_significance_stars(qval)
                
                x1 = available_groups.index(g1)
                x2 = available_groups.index(g2)
                
                bracket_y = _draw_significance_bracket(
                    ax, x1, x2, bracket_y, text, is_sig,
                    bracket_height=0.02 * y_range
                )
                bracket_y += bracket_step * 0.3
        
        ax.set_title(band.capitalize(), fontsize=plot_cfg.font.title, fontweight="bold",
                    color=band_color)
        ax.set_xticks(positions)
        ax.set_xticklabels([str(g) for g in available_groups], 
                          fontsize=plot_cfg.font.small, rotation=45, ha="right")
        ax.set_ylabel(feature_label if band_idx == 0 else "", fontsize=plot_cfg.font.label)
        ax.tick_params(axis="y", labelsize=plot_cfg.font.small)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    title_parts = [f"{feature_label}: Multi-Group Comparison (Unpaired)"]
    
    info_parts = [f"Subject: {subject}"]
    if roi_name:
        roi_display = roi_name.replace("_", " ").title() if roi_name != "all" else "All Channels"
        info_parts.append(f"ROI: {roi_display}")
    info_parts.extend([
        f"Groups: {', '.join(str(g) for g in groups)}",
        "Pre-computed stats",
        f"FDR: {total_significant}/{total_tests} significant (*=q<0.05)"
    ])
    title_parts.append(" | ".join(info_parts))
    
    fig.suptitle("\n".join(title_parts), fontsize=plot_cfg.font.suptitle,
                fontweight="bold", y=1.02)
    
    plt.tight_layout()
    save_fig(fig, save_path, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
            bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches, config=config)
    plt.close(fig)
    
    if logger:
        logger.info(f"Saved {feature_label} multi-group column comparison "
                   f"({total_significant}/{total_tests} FDR significant, pre-computed)")


def _build_multigroup_qvalues_map(
    multigroup_stats: pd.DataFrame,
    feature_keys: List[str],
    groups: List[str],
    roi_name: Optional[str],
    stats_match_terms: Optional[Dict[str, Tuple[str, ...]]] = None,
) -> Tuple[Dict[Tuple[int, str, str], Tuple[float, bool]], int, int]:
    """Map precomputed multi-group stats to the requested feature keys."""
    stats = multigroup_stats.copy()
    if "identifier" not in stats.columns:
        stats["identifier"] = stats["feature"].astype(str)

    identifier_normalized = stats["identifier"].astype(str).map(_normalize_stats_match_text)
    if roi_name and roi_name.lower() != "all":
        roi_term = _normalize_stats_match_text(roi_name)
        roi_mask = identifier_normalized.str.contains(roi_term, na=False)
        stats = stats[roi_mask].copy()
        identifier_normalized = stats["identifier"].astype(str).map(_normalize_stats_match_text)

    qvalues_map: Dict[Tuple[int, str, str], Tuple[float, bool]] = {}
    total_significant = 0
    total_tests = 0
    expected_pairs = {tuple(sorted((g1, g2))) for g1, g2 in combinations(groups, 2)}

    for feature_idx, feature_key in enumerate(feature_keys):
        match_terms = _resolve_stats_match_terms(feature_key, stats_match_terms)
        feature_mask = _build_stats_match_mask(identifier_normalized, match_terms)
        feature_rows = stats[feature_mask]
        if feature_rows.empty:
            raise ValueError(
                "No multigroup stats matched plotted feature "
                f"{feature_key!r} for roi {roi_name!r} using match terms {match_terms!r}."
            )

        seen_pairs = set()
        for _, row in feature_rows.iterrows():
            g1 = str(row.get("group1", ""))
            g2 = str(row.get("group2", ""))
            if g1 not in groups or g2 not in groups:
                continue

            pair_key = tuple(sorted((g1, g2)))
            if pair_key not in expected_pairs:
                continue
            if pair_key in seen_pairs:
                raise ValueError(
                    "Ambiguous multigroup stats for "
                    f"feature {feature_key!r}, groups {g1!r}/{g2!r}, roi {roi_name!r}."
                )
            seen_pairs.add(pair_key)

            row_key = (feature_idx, g1, g2)
            if row_key in qvalues_map:
                raise ValueError(
                    "Ambiguous multigroup stats for "
                    f"feature {feature_key!r}, groups {g1!r}/{g2!r}, roi {roi_name!r}."
                )

            q_value = float(row.get("q_value", 1.0))
            is_sig = bool(row.get("significant_fdr", False))
            qvalues_map[row_key] = (q_value, is_sig)
            total_tests += 1
            if is_sig:
                total_significant += 1

        missing_pairs = sorted(expected_pairs.difference(seen_pairs))
        if missing_pairs:
            raise ValueError(
                "Missing multigroup stats for "
                f"feature {feature_key!r}, roi {roi_name!r}: {missing_pairs!r}."
            )

    return qvalues_map, total_significant, total_tests


def _resolve_multigroup_qvalues_map(
    *,
    data_by_band: Dict[str, Dict[str, np.ndarray]],
    groups: List[str],
    multigroup_stats: pd.DataFrame,
    feature_keys: List[str],
    roi_name: Optional[str],
    stats_match_terms: Optional[Dict[str, Tuple[str, ...]]] = None,
) -> Tuple[Dict[Tuple[int, str, str], Tuple[float, bool]], int, int]:
    """Resolve multigroup q-values from precomputed stats."""
    return _build_multigroup_qvalues_map(
        multigroup_stats=multigroup_stats,
        feature_keys=feature_keys,
        groups=groups,
        roi_name=roi_name,
        stats_match_terms=stats_match_terms,
    )


def _resolve_stats_match_terms(
    feature_key: str,
    stats_match_terms: Optional[Dict[str, Tuple[str, ...]]],
) -> Tuple[str, ...]:
    """Return the identifier terms that must match for one plotted feature key."""
    if stats_match_terms and feature_key in stats_match_terms:
        terms = tuple(str(term) for term in stats_match_terms[feature_key] if str(term).strip())
    else:
        terms = (str(feature_key),)

    if not terms:
        raise ValueError(f"No stats match terms were provided for feature {feature_key!r}.")
    return terms


def _normalize_stats_match_text(value: str) -> str:
    """Normalize stats identifiers and match terms to comparable lowercase tokens."""
    normalized = re.sub(r"[^0-9a-z]+", " ", str(value).lower())
    return " ".join(normalized.split())


def _build_stats_match_mask(
    identifier_normalized: pd.Series,
    match_terms: Tuple[str, ...],
) -> pd.Series:
    """Return rows whose normalized identifier contains every requested term."""
    mask = pd.Series(True, index=identifier_normalized.index)
    for term in match_terms:
        normalized_term = _normalize_stats_match_text(term)
        if normalized_term == "":
            raise ValueError(f"Invalid empty stats match term in {match_terms!r}.")
        mask &= identifier_normalized.str.contains(normalized_term, na=False)
    return mask
