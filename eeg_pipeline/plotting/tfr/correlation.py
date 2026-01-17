"""
Time-frequency correlation plotting functions.

Functions for creating group-level time-frequency correlation visualizations,
including correlation heatmaps and statistical significance testing.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist

from eeg_pipeline.infra.tsv import read_tsv
from eeg_pipeline.infra.paths import deriv_group_stats_path, deriv_group_plots_path
from ...utils.data.tfr_alignment import extract_time_frequency_grid
from ...utils.analysis.stats import (
    fdr_bh_values as _fdr_bh_values,
)
from ...utils.config.loader import get_fisher_z_clip_values, get_config_value
from ..config import get_plot_config
from ..core.utils import log
from .channels import _save_fig


def _discover_subjects_with_data(
    roi_suffix: str,
    method_suffix: str,
    config,
    allowed_subjects: Optional[List[str]] = None,
) -> List[str]:
    """Discover subjects that have time-frequency correlation statistics files.
    
    Args:
        roi_suffix: ROI suffix for the file name
        method_suffix: Method suffix (e.g., '_spearman', '_pearson')
        config: Configuration object with deriv_root
        allowed_subjects: Optional list of allowed subject IDs
        
    Returns:
        List of subject IDs that have the required files
    """
    subject_ids = []
    for subject_dir in sorted(config.deriv_root.glob("sub-*")):
        if not subject_dir.is_dir():
            continue
        subject_id = subject_dir.name[4:]
        if allowed_subjects is not None and subject_id not in allowed_subjects:
            continue
        unified_path = (
            subject_dir
            / "eeg"
            / "stats"
            / "temporal_correlations"
            / f"tf_grid{roi_suffix}{method_suffix}.tsv"
        )
        if unified_path.exists():
            subject_ids.append(subject_id)
    return subject_ids


def _load_subject_correlation_data(
    subject_id: str,
    roi_suffix: str,
    method_suffix: str,
    config,
) -> Optional[pd.DataFrame]:
    """Load time-frequency correlation statistics for a single subject.
    
    Args:
        subject_id: Subject ID
        roi_suffix: ROI suffix for the file name
        method_suffix: Method suffix (e.g., '_spearman', '_pearson')
        config: Configuration object with deriv_root
        
    Returns:
        DataFrame with correlation statistics, or None if file doesn't exist
    """
    unified_path = (
        config.deriv_root
        / f"sub-{subject_id}"
        / "eeg"
        / "stats"
        / "temporal_correlations"
        / f"tf_grid{roi_suffix}{method_suffix}.tsv"
    )
    return read_tsv(unified_path) if unified_path.exists() else None


def _get_baseline_window(config) -> List[float]:
    """Extract baseline window from configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        Baseline window as [start, end] in seconds
    """
    if not config:
        return [-0.5, -0.01]
    return config.get(
        "plotting.tfr.default_baseline_window",
        config.get("time_frequency_analysis.baseline_window", [-0.5, -0.01]),
    )


def _annotate_correlation_figure(
    fig: plt.Figure,
    config,
    alpha: float,
) -> None:
    """Add annotation text to time-frequency correlation figure.
    
    Args:
        fig: Matplotlib figure to annotate
        config: Configuration object
        alpha: Significance threshold (FDR alpha)
    """
    baseline_window = _get_baseline_window(config)
    baseline_start = float(baseline_window[0])
    baseline_end = float(baseline_window[1])
    fdr_text = f"FDR BH α={alpha}"
    annotation_text = (
        f"Group TF correlation | Baseline: [{baseline_start:.2f}, {baseline_end:.2f}] s | "
        f"{fdr_text}"
    )
    plot_config = get_plot_config(config)
    font_size = plot_config.font.label
    fig.text(0.01, 0.01, annotation_text, fontsize=font_size, alpha=0.8)


def _find_available_correlation_method(
    roi_suffix: str,
    min_subjects: int,
    config,
    allowed_subjects: Optional[List[str]],
) -> Optional[Tuple[str, List[str]]]:
    """Find first available correlation method with sufficient subjects.
    
    Args:
        roi_suffix: ROI suffix for the file name
        min_subjects: Minimum number of subjects required
        config: Configuration object
        allowed_subjects: Optional list of allowed subject IDs
        
    Returns:
        Tuple of (method_suffix, subjects_found) or None if none found
    """
    for method_suffix in ("_spearman", "_pearson"):
        subjects_found = _discover_subjects_with_data(
            roi_suffix, method_suffix, config, allowed_subjects
        )
        if len(subjects_found) >= min_subjects:
            return method_suffix, subjects_found
    return None


def _select_correlation_method(
    method: str,
    roi_suffix: str,
    min_subjects: int,
    config,
    allowed_subjects: Optional[List[str]],
    subjects_param: Optional[List[str]],
    logger: Optional[logging.Logger],
) -> Tuple[Optional[str], Optional[List[str]]]:
    """Select correlation method and discover subjects with available data.
    
    Args:
        method: Correlation method ('auto', 'spearman', or 'pearson')
        roi_suffix: ROI suffix for the file name
        min_subjects: Minimum number of subjects required
        config: Configuration object
        allowed_subjects: Optional list of allowed subject IDs
        subjects_param: Optional list of subjects to use
        logger: Optional logger
        
    Returns:
        Tuple of (method_suffix, subjects_found) or (None, None) if insufficient data
    """
    if method == "auto":
        result = _find_available_correlation_method(
            roi_suffix, min_subjects, config, allowed_subjects
        )
        if result is None:
            roi_display = roi_suffix or "all"
            log(
                f"Group TF correlation skipped for ROI '{roi_display}' — insufficient subject heatmaps.",
                logger,
                "warning",
            )
            return None, None
        return result
    
    method_suffix = f"_{method.lower()}"
    subjects_found = (
        subjects_param
        or _discover_subjects_with_data(
            roi_suffix, method_suffix, config, allowed_subjects
        )
    )
    return method_suffix, subjects_found


def _get_default_alpha(config) -> float:
    """Get default significance alpha from configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        Significance alpha value
    """
    from ...utils.config.loader import ensure_config
    
    config = ensure_config(config)
    plot_config = get_plot_config(config)
    if plot_config:
        tfr_config = plot_config.plot_type_configs.get("tfr", {})
        return tfr_config.get(
            "default_significance_alpha",
            get_config_value(config, "statistics.sig_alpha", 0.05),
        )
    return get_config_value(config, "statistics.sig_alpha", 0.05)


def _get_default_min_subjects(config) -> int:
    """Get default minimum subjects from configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        Minimum number of subjects required
    """
    if not config:
        return 3
    return int(config.get("analysis.min_subjects_for_topomaps", 3))


def _normalize_roi_name(roi: Optional[str]) -> str:
    """Normalize ROI name to file-safe suffix.
    
    Args:
        roi: Optional ROI name (e.g., 'frontal', 'parietal')
        
    Returns:
        ROI suffix string (empty if roi is None)
    """
    if roi is None:
        return ""
    roi_sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", roi.lower())
    return f"_{roi_sanitized}"


def _load_valid_subject_data(
    subject_ids: List[str],
    roi_suffix: str,
    method_suffix: str,
    config,
) -> Tuple[List[pd.DataFrame], List[str]]:
    """Load and validate correlation data for subjects.
    
    Args:
        subject_ids: List of subject IDs to load
        roi_suffix: ROI suffix for file names
        method_suffix: Method suffix for file names
        config: Configuration object
        
    Returns:
        Tuple of (valid_dataframes, valid_subject_ids)
    """
    valid_dataframes = []
    valid_subject_ids = []
    
    for subject_id in subject_ids:
        dataframe = _load_subject_correlation_data(
            subject_id, roi_suffix, method_suffix, config
        )
        if dataframe is None or dataframe.empty:
            continue
        
        required_columns = ["r", "freq", "time_start"]
        missing = [c for c in required_columns if c not in dataframe.columns]
        if missing:
            continue
        dataframe_clean = dataframe.dropna(subset=required_columns)
        if dataframe_clean.empty:
            continue
        valid_dataframes.append(dataframe_clean)
        valid_subject_ids.append(subject_id)
    
    return valid_dataframes, valid_subject_ids


def _find_common_time_frequency_grid(
    dataframes: List[pd.DataFrame],
) -> Tuple[np.ndarray, np.ndarray]:
    """Find common time-frequency grid across all subject dataframes.
    
    Args:
        dataframes: List of subject dataframes with frequency and time columns
        
    Returns:
        Tuple of (common_frequencies, common_times)
    """
    if not dataframes:
        return np.array([]), np.array([])
    
    frequencies_common, times_common = extract_time_frequency_grid(dataframes[0])
    for dataframe in dataframes[1:]:
        frequencies, times = extract_time_frequency_grid(dataframe)
        frequencies_common = np.intersect1d(frequencies_common, frequencies)
        times_common = np.intersect1d(times_common, times)
    
    return frequencies_common, times_common


def _build_correlation_matrices(
    dataframes: List[pd.DataFrame],
    frequencies_common: np.ndarray,
    times_common: np.ndarray,
) -> List[np.ndarray]:
    """Build correlation matrices from subject dataframes.
    
    Args:
        dataframes: List of subject dataframes
        frequencies_common: Common frequency values
        times_common: Common time values
        
    Returns:
        List of correlation matrices (one per subject)
    """
    correlation_matrices = []
    rounding_precision = 6
    
    for dataframe in dataframes:
        dataframe_copy = dataframe.copy()
        dataframe_copy["freq"] = np.round(
            dataframe_copy["freq"].astype(float), rounding_precision
        )
        dataframe_copy["time_start"] = np.round(
            dataframe_copy["time_start"].astype(float), rounding_precision
        )
        pivot_table = dataframe_copy.pivot_table(
            index="freq",
            columns="time_start",
            values="r",
            aggfunc="mean",
        )
        pivot_table_aligned = pivot_table.reindex(
            index=frequencies_common, columns=times_common
        )
        correlation_matrices.append(pivot_table_aligned.to_numpy())
    
    return correlation_matrices


def _compute_group_statistics(
    correlation_matrices: List[np.ndarray],
    alpha: float,
    min_subjects: int,
    config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute group-level statistics on correlation matrices.
    
    Performs Fisher z-transform, t-test, and FDR correction.
    
    Args:
        correlation_matrices: List of subject correlation matrices
        alpha: Significance threshold for FDR correction
        min_subjects: Minimum number of subjects required
        config: Configuration object
        
    Returns:
        Tuple of (r_mean, z_mean, n, p_values, q_values, significant_mask)
    """
    clip_min, clip_max = get_fisher_z_clip_values(config)
    
    fisher_z_matrices = np.stack([
        np.arctanh(np.clip(matrix, clip_min, clip_max))
        for matrix in correlation_matrices
    ], axis=0)
    
    z_mean = np.nanmean(fisher_z_matrices, axis=0)
    z_std = np.nanstd(fisher_z_matrices, axis=0, ddof=1)
    n_valid = np.sum(np.isfinite(fisher_z_matrices), axis=0)
    
    with np.errstate(divide="ignore", invalid="ignore"):
        standard_error = z_std / np.sqrt(np.maximum(n_valid, 1))
        standard_error[standard_error == 0] = np.nan
        t_statistic = z_mean / standard_error
    
    p_values = np.full_like(t_statistic, np.nan, dtype=float)
    valid_mask = np.isfinite(t_statistic) & (n_valid > 1)
    if np.any(valid_mask):
        degrees_of_freedom = np.maximum(n_valid[valid_mask] - 1, 1)
        t_absolute = np.abs(t_statistic[valid_mask])
        p_values[valid_mask] = 2.0 * t_dist.sf(t_absolute, df=degrees_of_freedom)
    
    p_values_flat = p_values[np.isfinite(p_values)]
    rejected, q_values_flat = _fdr_bh_values(p_values_flat, alpha=alpha)
    
    q_values = np.full_like(p_values, np.nan)
    if q_values_flat is not None:
        q_values[np.isfinite(p_values)] = q_values_flat
    
    significant_mask = np.zeros_like(p_values, dtype=bool)
    if rejected is not None:
        significant_mask[np.isfinite(p_values)] = rejected.astype(bool)
    significant_mask &= (n_valid >= min_subjects)
    
    r_mean = np.tanh(z_mean)
    
    return r_mean, z_mean, n_valid, p_values, q_values, significant_mask


def _save_group_statistics(
    frequencies: np.ndarray,
    times: np.ndarray,
    r_mean: np.ndarray,
    z_mean: np.ndarray,
    n_valid: np.ndarray,
    p_values: np.ndarray,
    q_values: np.ndarray,
    significant_mask: np.ndarray,
    roi_suffix: str,
    method_suffix: str,
    config,
) -> Path:
    """Save group-level statistics to TSV file.
    
    Args:
        frequencies: Frequency values
        times: Time values
        r_mean: Mean correlation values
        z_mean: Mean Fisher z values
        n_valid: Number of valid observations per time-frequency point
        p_values: P-values
        q_values: FDR-corrected q-values
        significant_mask: Boolean mask of significant points
        roi_suffix: ROI suffix for file name
        method_suffix: Method suffix for file name
        config: Configuration object
        
    Returns:
        Path to saved TSV file
    """
    stats_dir = deriv_group_stats_path(config.deriv_root)
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = stats_dir / f"tf_corr_group{roi_suffix}{method_suffix}.tsv"
    output_dataframe = pd.DataFrame({
        "frequency": np.repeat(frequencies, len(times)),
        "time": np.tile(times, len(frequencies)),
        "r_mean": r_mean.flatten(),
        "z_mean": z_mean.flatten(),
        "n": n_valid.flatten(),
        "p": p_values.flatten(),
        "q": q_values.flatten(),
        "significant": significant_mask.flatten(),
    })
    output_dataframe.to_csv(output_path, sep="\t", index=False)
    
    return output_path


def _create_correlation_heatmap(
    data: np.ndarray,
    frequencies: np.ndarray,
    times: np.ndarray,
    title: str,
    colorbar_label: str,
    roi_suffix: str,
    method_suffix: str,
    alpha: float,
    config,
    logger: Optional[logging.Logger],
    filename_suffix: str,
) -> Tuple[plt.Figure, List[Path]]:
    """Create correlation heatmap plot.
    
    Args:
        data: Correlation matrix to plot
        frequencies: Frequency values
        times: Time values
        title: Plot title
        colorbar_label: Colorbar label
        roi_suffix: ROI suffix for file name
        method_suffix: Method suffix
        alpha: Significance threshold
        config: Configuration object
        logger: Optional logger
        filename_suffix: Suffix for output filename
        
    Returns:
        Tuple of (figure, figure_paths)
    """
    plot_config = get_plot_config(config)
    figure_size = plot_config.get_figure_size("small", plot_type="tfr")
    
    extent = [times[0], times[-1], frequencies[0], frequencies[-1]]
    colormap = "RdBu_r"
    vmin, vmax = -0.6, 0.6
    
    figure, axis = plt.subplots(figsize=figure_size)
    image = axis.imshow(
        data,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
    )
    axis.axvline(0.0, color="k", linestyle="--", alpha=0.6)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Frequency (Hz)")
    axis.set_title(title)
    
    colorbar = plt.colorbar(image, ax=axis)
    colorbar.set_label(colorbar_label)
    plt.tight_layout()
    
    _annotate_correlation_figure(figure, config, alpha)
    
    plots_dir = deriv_group_plots_path(config.deriv_root, "tf_corr")
    plots_dir.mkdir(parents=True, exist_ok=True)
    filename_base = f"tf_corr_group_{filename_suffix}{roi_suffix}{method_suffix}"
    
    _save_fig(figure, plots_dir, filename_base, config, logger=logger)
    
    figure_paths = [
        plots_dir / f"{filename_base}.{ext}" for ext in plot_config.formats
    ]
    
    return figure, figure_paths


def _create_mean_correlation_plot(
    r_mean: np.ndarray,
    frequencies: np.ndarray,
    times: np.ndarray,
    roi: Optional[str],
    roi_suffix: str,
    method_suffix: str,
    alpha: float,
    config,
    logger: Optional[logging.Logger],
) -> Tuple[plt.Figure, List[Path]]:
    """Create plot showing mean correlation values.
    
    Args:
        r_mean: Mean correlation matrix
        frequencies: Frequency values
        times: Time values
        roi: Optional ROI name for display
        roi_suffix: ROI suffix for file name
        method_suffix: Method suffix
        alpha: Significance threshold
        config: Configuration object
        logger: Optional logger
        
    Returns:
        Tuple of (figure, figure_paths)
    """
    roi_display = roi or "All channels"
    method_display = method_suffix.strip("_").title()
    title = f"Group TF correlation — mean r ({method_display}, {roi_display})"
    
    return _create_correlation_heatmap(
        r_mean,
        frequencies,
        times,
        title,
        "r",
        roi_suffix,
        method_suffix,
        alpha,
        config,
        logger,
        "rmean",
    )


def _create_significant_correlation_plot(
    r_mean: np.ndarray,
    significant_mask: np.ndarray,
    frequencies: np.ndarray,
    times: np.ndarray,
    roi: Optional[str],
    roi_suffix: str,
    method_suffix: str,
    alpha: float,
    config,
    logger: Optional[logging.Logger],
) -> Tuple[plt.Figure, List[Path]]:
    """Create plot showing only significant correlation values.
    
    Args:
        r_mean: Mean correlation matrix
        significant_mask: Boolean mask of significant points
        frequencies: Frequency values
        times: Time values
        roi: Optional ROI name for display
        roi_suffix: ROI suffix for file name
        method_suffix: Method suffix
        alpha: Significance threshold
        config: Configuration object
        logger: Optional logger
        
    Returns:
        Tuple of (figure, figure_paths)
    """
    significant_correlations = np.where(significant_mask, r_mean, np.nan)
    roi_display = roi or "All channels"
    method_display = method_suffix.strip("_").title()
    title = f"Group TF correlation — FDR<{alpha:g} ({method_display}, {roi_display})"
    
    return _create_correlation_heatmap(
        significant_correlations,
        frequencies,
        times,
        title,
        "r (significant)",
        roi_suffix,
        method_suffix,
        alpha,
        config,
        logger,
        "sig",
    )


def group_tf_correlation(
    subjects: Optional[List[str]] = None,
    roi: Optional[str] = None,
    method: str = "auto",
    alpha: Optional[float] = None,
    min_subjects: Optional[int] = None,
    config=None,
    logger: Optional[logging.Logger] = None,
) -> Optional[Tuple[Path, List[Path]]]:
    """Create group-level time-frequency correlation plots.
    
    Loads subject-level correlation statistics, performs group-level statistical
    testing (Fisher z-transform, t-test, FDR correction), and creates visualization
    heatmaps showing mean correlation and significant regions.
    
    Args:
        subjects: Optional list of subject IDs to include
        roi: Optional ROI name (e.g., 'frontal', 'parietal')
        method: Correlation method ('auto', 'spearman', or 'pearson')
        alpha: Significance threshold for FDR correction (defaults to config)
        min_subjects: Minimum number of subjects required (defaults to config)
        config: Configuration object
        logger: Optional logger
        
    Returns:
        Tuple of (output_tsv_path, figure_paths) or None if skipped
    """
    from ...utils.config.loader import ensure_config
    
    config = ensure_config(config)
    
    if alpha is None:
        alpha = _get_default_alpha(config)
    if min_subjects is None:
        min_subjects = _get_default_min_subjects(config)
    
    roi_suffix = _normalize_roi_name(roi)
    allowed_subjects = set(subjects) if subjects else None
    
    method_suffix, subjects_to_use = _select_correlation_method(
        method, roi_suffix, min_subjects, config, allowed_subjects, subjects, logger
    )
    if method_suffix is None or not subjects_to_use:
        if not subjects_to_use:
            roi_display = roi or "all"
            log(
                f"Group TF correlation skipped for ROI '{roi_display}' — no subject files for method '{method}'.",
                logger,
                "warning",
            )
        return None
    
    subject_dataframes, valid_subject_ids = _load_valid_subject_data(
        subjects_to_use, roi_suffix, method_suffix, config
    )
    
    if len(subject_dataframes) < min_subjects:
        roi_display = roi or "all"
        log(
            f"Group TF correlation skipped for ROI '{roi_display}' — fewer than {min_subjects} subjects with valid data.",
            logger,
            "warning",
        )
        return None
    
    frequencies_common, times_common = _find_common_time_frequency_grid(
        subject_dataframes
    )
    
    if frequencies_common.size == 0 or times_common.size == 0:
        roi_display = roi or "all"
        log(
            f"Group TF correlation skipped for ROI '{roi_display}' — unable to find common TF grid.",
            logger,
            "warning",
        )
        return None
    
    correlation_matrices = _build_correlation_matrices(
        subject_dataframes, frequencies_common, times_common
    )
    
    r_mean, z_mean, n_valid, p_values, q_values, significant_mask = (
        _compute_group_statistics(
            correlation_matrices, alpha, min_subjects, config
        )
    )
    
    output_tsv_path = _save_group_statistics(
        frequencies_common,
        times_common,
        r_mean,
        z_mean,
        n_valid,
        p_values,
        q_values,
        significant_mask,
        roi_suffix,
        method_suffix,
        config,
    )
    
    _, mean_plot_paths = _create_mean_correlation_plot(
        r_mean,
        frequencies_common,
        times_common,
        roi,
        roi_suffix,
        method_suffix,
        alpha,
        config,
        logger,
    )
    
    _, sig_plot_paths = _create_significant_correlation_plot(
        r_mean,
        significant_mask,
        frequencies_common,
        times_common,
        roi,
        roi_suffix,
        method_suffix,
        alpha,
        config,
        logger,
    )
    
    figure_paths = mean_plot_paths + sig_plot_paths
    
    roi_display = roi or "all"
    method_display = method_suffix.strip("_")
    log(
        f"Group TF correlation saved (ROI={roi_display}, method={method_display}): {output_tsv_path}",
        logger,
        "info",
    )
    
    return output_tsv_path, figure_paths


__all__ = [
    "group_tf_correlation",
]
