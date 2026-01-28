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

from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd

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


def get_numeric_feature_columns(
    df: pd.DataFrame,
    *,
    exclude: Optional[List[str]] = None,
) -> List[str]:
    """Return numeric feature columns excluding common metadata columns."""
    if not _is_valid_dataframe(df):
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


def get_named_bands(
    df: pd.DataFrame,
    *,
    group: Optional[str] = None,
    segment: Optional[str] = None,
) -> List[str]:
    """Return available NamingSchema bands for a feature group/segment."""
    if not _is_valid_dataframe(df):
        return []
    
    bands = set()
    for col in df.columns:
        parsed = NamingSchema.parse(str(col))
        if not parsed.get("valid"):
            continue
        if group and parsed.get("group") != group:
            continue
        parsed_segment = parsed.get("segment") or ""
        if segment and str(parsed_segment) != str(segment):
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
    if not _is_valid_dataframe(df):
        return [], None, None

    stat_prefs = list(stat_preference or [None])
    scope_prefs = list(scope_preference or [None])

    for scope in scope_prefs:
        for stat in stat_prefs:
            matching_columns = []
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


def extract_multi_segment_data(
    df: pd.DataFrame,
    group: str,
    bands: List[str],
    segments: List[str],
    identifiers: Optional[List[str]] = None,
    stat_preference: Optional[List[str]] = None,
    scope_preference: Optional[List[str]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Extract feature data by band for multiple segments.
    
    Generic function that works for any feature type (power, aperiodic, connectivity, etc.).
    
    Args:
        df: Feature DataFrame with NamingSchema columns
        group: Feature group (e.g., 'power', 'aperiodic', 'connectivity')
        bands: List of frequency bands to extract
        segments: List of segment names (e.g., ['baseline', 'plateau', 'rampdown', 'rampup'])
        identifiers: Optional list of channel/ROI identifiers to filter by
        stat_preference: Preferred stat types (e.g., ['mean', 'median'])
        scope_preference: Preferred scope types (e.g., ['ch', 'roi', 'global'])
    
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


###################################################################
# UNIFIED PAIRED COMPARISON PLOTTING
###################################################################


def _compute_paired_wilcoxon_stats(
    condition1_values: np.ndarray,
    condition2_values: np.ndarray,
) -> Tuple[float, float]:
    """Compute Wilcoxon signed-rank test and effect size for paired data.
    
    Returns:
        Tuple of (p_value, effect_size_d)
    """
    from scipy.stats import wilcoxon
    
    _, p_value = wilcoxon(condition2_values, condition1_values)
    effect_size = compute_paired_cohens_d(condition1_values, condition2_values)
    return float(p_value), float(effect_size) if np.isfinite(effect_size) else 0.0


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
    rng = np.random.default_rng(42)
    condition1_jitter = rng.uniform(-jitter_range, jitter_range, len(condition1_values))
    condition2_jitter = rng.uniform(-jitter_range, jitter_range, len(condition2_values))
    
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
        sig_marker = "†" if is_significant else ""
        q_str = f"q={q_value:.3f}" if q_value >= 0.001 else "q<.001"
        annotation_text = f"{q_str}{sig_marker}\nd={effect_size:.2f}"
        significance_color = get_significance_color(is_significant, config)
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
        if "feature_type" in df.columns:
            mask &= df["feature_type"].str.lower().str.contains(ft_lower, na=False)
        if "identifier" in df.columns:
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
        
        if roi_name and roi_name.lower() != "all":
            pattern = f"{key_lower}_{roi_name.lower()}"
            match = precomputed_df[identifier_lower == pattern]
        
        if match.empty:
            match = precomputed_df[identifier_lower.str.contains(key_lower, na=False)]
        
        if not match.empty:
            row = match.iloc[0]
            p = float(row.get("p_value", 1.0))
            q = float(row.get("q_value", p))
            d = float(row.get("effect_size_d", 0.0))
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
    
    Tries to load pre-computed stats first, falls back to on-the-fly computation.
    
    Returns:
        Tuple of (qvalues dict, n_significant, use_precomputed)
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
            if logger and hasattr(logger, "info"):
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
    from itertools import combinations
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
    
    all_pvalues = []
    pvalue_keys = []
    min_samples = int(get_config_value(config, "behavior_analysis.min_samples.default", 5))
    
    for band in bands_in_order:
        segment_data = data_by_band[band]
        for seg1, seg2 in combinations(segments, 2):
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
    
    qvalues: Dict[Tuple[str, str, str], Tuple[float, float, float, bool]] = {}
    n_significant = 0
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
        box_width = 0.6
        
        box_data = [segment_data[seg] for seg in available_segments]
        boxplot = ax.boxplot(box_data, positions=positions, widths=box_width, patch_artist=True)
        
        for i, seg in enumerate(available_segments):
            boxplot["boxes"][i].set_facecolor(segment_color_map[seg])
            boxplot["boxes"][i].set_alpha(0.6)
            
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(segment_data[seg]))
            ax.scatter(positions[i] + jitter, segment_data[seg],
                      c=[segment_color_map[seg]], alpha=0.3, s=8)
        
        all_values = np.concatenate([segment_data[seg] for seg in available_segments])
        y_min = np.nanmin(all_values)
        y_max = np.nanmax(all_values)
        y_range = y_max - y_min if y_max > y_min else 0.1
        
        bracket_y = y_max + 0.08 * y_range
        bracket_spacing = 0.12 * y_range
        
        pair_list = list(combinations(range(len(available_segments)), 2))
        for pair_idx, (i, j) in enumerate(pair_list):
            seg1, seg2 = available_segments[i], available_segments[j]
            key = (band, seg1, seg2)
            if key not in qvalues:
                key = (band, seg2, seg1)
            
            if key in qvalues:
                _, q_val, d, is_sig = qvalues[key]
                stars = _get_significance_stars(q_val)
                text = f"{stars}" if is_sig else "ns"
                
                current_y = bracket_y + pair_idx * bracket_spacing
                _draw_significance_bracket(
                    ax, positions[i], positions[j], current_y, text, is_sig,
                    bracket_height=0.02 * y_range, text_offset=0.01 * y_range
                )
        
        top_bracket_y = bracket_y + len(pair_list) * bracket_spacing
        ax.set_ylim(y_min - 0.1 * y_range, top_bracket_y + 0.1 * y_range)
        
        ax.set_xticks(positions)
        ax.set_xticklabels([s.capitalize() for s in available_segments], fontsize=9, rotation=30, ha="right")
        ax.set_title(band.capitalize(), fontweight="bold", color=band_colors.get(band, "gray"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    first_band = bands_in_order[0]
    n_trials = len(data_by_band[first_band][segments[0]]) if segments[0] in data_by_band[first_band] else 0
    n_tests = len(qvalues)
    
    roi_display = roi_name.replace("_", " ").title() if roi_name and roi_name != "all" else "All Channels"
    
    title_parts = [f"{feature_label}: Multi-Window Comparison ({n_segments} windows, {n_pairs} pairs)"]
    info_parts = [
        f"Subject: {subject}",
        f"ROI: {roi_display}",
        f"N: {n_trials} trials",
        "Wilcoxon signed-rank",
        f"FDR: {n_significant}/{n_tests} significant (*p<.05, **p<.01, ***p<.001)"
    ]
    title_parts.append(" | ".join(info_parts))
    
    fig.suptitle("\n".join(title_parts), fontsize=plot_cfg.font.suptitle, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    save_fig(
        fig, save_path,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
        config=config
    )
    plt.close(fig)
    
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
    n_tests = len(qvalues)
    
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
        pad_inches=plot_cfg.pad_inches,
        config=config
    )
    plt.close(fig)
    
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
    """
    import matplotlib.pyplot as plt
    from itertools import combinations
    from eeg_pipeline.plotting.io.figures import save_fig
    
    if not data_by_band:
        return
    
    if multigroup_stats is None:
        if stats_dir is None:
            if logger:
                logger.warning(
                    f"Multi-group comparison for {feature_label} requires pre-computed stats. "
                    "Run behavior pipeline with 3+ comparison values first."
                )
            return
        
        multigroup_stats = load_multigroup_stats(stats_dir)
    
    if multigroup_stats is None or multigroup_stats.empty:
        if logger:
            logger.warning(
                f"No pre-computed multi-group stats found for {feature_label}. "
                "Run behavior pipeline with 3+ comparison values first."
            )
        return
    
    plot_cfg = get_plot_config(config)
    bands_in_order = list(data_by_band.keys())
    n_bands = len(bands_in_order)
    
    if n_bands == 0:
        return
    
    n_groups = len(groups)
    group_colors = plt.cm.Set2(np.linspace(0, 1, max(n_groups, 3)))
    
    # Build qvalues map for all bands
    qvalues_map: Dict[Tuple[int, str, str], Tuple[float, bool]] = {}
    total_significant = 0
    total_tests = 0
    
    for _, row in multigroup_stats.iterrows():
        feature = str(row.get("feature", ""))
        g1 = str(row.get("group1", ""))
        g2 = str(row.get("group2", ""))
        q_value = float(row.get("q_value", 1.0))
        is_sig = bool(row.get("significant_fdr", False))
        
        for band_idx, band in enumerate(bands_in_order):
            if band.lower() in feature.lower():
                qvalues_map[(band_idx, g1, g2)] = (q_value, is_sig)
                total_tests += 1
                if is_sig:
                    total_significant += 1
                break
    
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
            f"FDR: {band_sig}/{band_tests} sig (*p<.05)"
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
