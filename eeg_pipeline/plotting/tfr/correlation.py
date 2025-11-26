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

from ...utils.io.general import (
    deriv_group_stats_path,
    deriv_group_plots_path,
)
from ...utils.data.loading import (
    extract_time_frequency_grid,
)
from ...utils.analysis.stats import (
    fdr_bh_values as _fdr_bh_values,
)
from ..config import get_plot_config
from ..core.utils import log
from .channels import _save_fig


###################################################################
# Helper Functions
###################################################################


def _discover_subjects_with_tf(roi_suffix: str, method_suffix: str, config, allowed_subjects: Optional[List[str]] = None) -> List[str]:
    """Discover subjects that have time-frequency correlation statistics files.
    
    Args:
        roi_suffix: ROI suffix for the file name
        method_suffix: Method suffix (e.g., '_spearman', '_pearson')
        config: Configuration object with deriv_root
        allowed_subjects: Optional list of allowed subject IDs
        
    Returns:
        List of subject IDs that have the required files
    """
    subs = []
    for sd in sorted(config.deriv_root.glob("sub-*")):
        if not sd.is_dir():
            continue
        sub = sd.name[4:]
        if allowed_subjects is not None and sub not in allowed_subjects:
            continue
        cand = sd / "eeg" / "stats" / f"tf_corr_stats{roi_suffix}{method_suffix}.tsv"
        if cand.exists():
            subs.append(sub)
    return subs


def _load_subject_tf(sub: str, roi_suffix: str, method_suffix: str, config) -> Optional[pd.DataFrame]:
    """Load time-frequency correlation statistics for a single subject.
    
    Args:
        sub: Subject ID
        roi_suffix: ROI suffix for the file name
        method_suffix: Method suffix (e.g., '_spearman', '_pearson')
        config: Configuration object with deriv_root
        
    Returns:
        DataFrame with correlation statistics, or None if file doesn't exist
    """
    p = config.deriv_root / f"sub-{sub}" / "eeg" / "stats" / f"tf_corr_stats{roi_suffix}{method_suffix}.tsv"
    return pd.read_csv(p, sep="\t") if p.exists() else None


def _annotate_tf_correlation_figure(fig: plt.Figure, config, alpha: float) -> None:
    """Add annotation text to time-frequency correlation figure.
    
    Args:
        fig: Matplotlib figure to annotate
        config: Configuration object
        alpha: Significance threshold (FDR alpha)
    """
    try:
        default_baseline = [-0.5, -0.01]
        bwin = config.get("time_frequency_analysis.baseline_window", default_baseline) if config else default_baseline
        corr_txt = f"FDR BH α={alpha}"
        text = (
            f"Group TF correlation | Baseline: [{float(bwin[0]):.2f}, {float(bwin[1]):.2f}] s | "
            f"{corr_txt}"
        )
        plot_cfg_annotate = get_plot_config(config)
        font_size_label = plot_cfg_annotate.font.label
        fig.text(0.01, 0.01, text, fontsize=font_size_label, alpha=0.8)
    except Exception:
        pass


def _select_tf_correlation_method(
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
        for method_name in ("_spearman", "_pearson"):
            subjects_found = _discover_subjects_with_tf(roi_suffix, method_name, config, allowed_subjects)
            if len(subjects_found) >= min_subjects:
                return method_name, subjects_found
        log(f"Group TF correlation skipped for ROI '{roi_suffix or 'all'}' — insufficient subject heatmaps.", logger, "warning")
        return None, None
    else:
        method_suffix = f"_{method.lower()}"
        subjects_found = subjects_param or _discover_subjects_with_tf(roi_suffix, method_suffix, config, allowed_subjects)
        return method_suffix, subjects_found


###################################################################
# Main Function
###################################################################


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
    if alpha is None:
        plot_cfg_alpha = get_plot_config(config) if config else None
        if plot_cfg_alpha:
            tfr_config_alpha = plot_cfg_alpha.plot_type_configs.get("tfr", {})
            default_significance_alpha = tfr_config_alpha.get("default_significance_alpha", 0.05)
        else:
            default_significance_alpha = 0.05
        alpha = config.get("statistics.sig_alpha", default_significance_alpha) if config else default_significance_alpha
    if min_subjects is None:
        min_subjects = int(config.get("analysis.min_subjects_for_topomaps", 3)) if config else 3
    
    roi_raw = roi.lower() if isinstance(roi, str) else None
    roi_suffix = f"_{re.sub(r'[^A-Za-z0-9._-]+', '_', roi_raw)}" if roi_raw else ""
    allowed_subjects = set(subjects) if subjects else None

    method_suffix, subjects_to_use = _select_tf_correlation_method(method, roi_suffix, min_subjects, config, allowed_subjects, subjects, logger)
    if method_suffix is None or not subjects_to_use:
        if not subjects_to_use:
            log(f"Group TF correlation skipped for ROI '{roi or 'all'}' — no subject files for method '{method}'.", logger, "warning")
        return None

    dfs = []
    used_subjects = []
    for sub in subjects_to_use:
        df = _load_subject_tf(sub, roi_suffix, method_suffix, config)
        if df is None or df.empty or df.dropna(subset=["correlation", "frequency", "time"]).empty:
            continue
        dfs.append(df.dropna(subset=["correlation", "frequency", "time"]))
        used_subjects.append(sub)

    if len(dfs) < min_subjects:
        log(f"Group TF correlation skipped for ROI '{roi or 'all'}' — fewer than {min_subjects} subjects with valid data.", logger, "warning")
        return None

    f_common, t_common = extract_time_frequency_grid(dfs[0])
    for df in dfs[1:]:
        f, t = extract_time_frequency_grid(df)
        f_common = np.intersect1d(f_common, f)
        t_common = np.intersect1d(t_common, t)

    if f_common.size == 0 or t_common.size == 0:
        log(f"Group TF correlation skipped for ROI '{roi or 'all'}' — unable to find common TF grid.", logger, "warning")
        return None

    mats: list[np.ndarray] = []
    for df in dfs:
        df_use = df.copy()
        df_use["frequency"] = np.round(df_use["frequency"].astype(float), 6)
        df_use["time"] = np.round(df_use["time"].astype(float), 6)
        pivot = df_use.pivot_table(index="frequency", columns="time", values="correlation", aggfunc="mean")
        pivot = pivot.reindex(index=f_common, columns=t_common)
        mats.append(pivot.to_numpy())

    Z = np.stack([np.arctanh(np.clip(m, -0.999999, 0.999999)) for m in mats], axis=0)
    z_mean = np.nanmean(Z, axis=0)
    z_sd = np.nanstd(Z, axis=0, ddof=1)
    n = np.sum(np.isfinite(Z), axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        denom = z_sd / np.sqrt(np.maximum(n, 1))
        denom[denom == 0] = np.nan
        t_stat = z_mean / denom

    p_vals = np.full_like(t_stat, np.nan, dtype=float)
    finite = np.isfinite(t_stat) & (n > 1)
    if np.any(finite):
        df = np.maximum(n[finite] - 1, 1)
        t_abs = np.abs(t_stat[finite])
        p_vals[finite] = 2.0 * t_dist.sf(t_abs, df=df)

    rej, q_flat = _fdr_bh_values(p_vals[np.isfinite(p_vals)], alpha=alpha)
    q_vals = np.full_like(p_vals, np.nan)
    if q_flat is not None:
        q_vals[np.isfinite(p_vals)] = q_flat

    sig_mask = np.zeros_like(p_vals, dtype=bool)
    if rej is not None:
        sig_mask[np.isfinite(p_vals)] = rej.astype(bool)
    sig_mask &= (n >= min_subjects)
    r_mean = np.tanh(z_mean)

    stats_dir = deriv_group_stats_path(config.deriv_root)
    plots_dir = deriv_group_plots_path(config.deriv_root, "tf_corr")
    stats_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    out_tsv = stats_dir / f"tf_corr_group{roi_suffix}{method_suffix}.tsv"
    df_out = pd.DataFrame(
        {
            "frequency": np.repeat(f_common, len(t_common)),
            "time": np.tile(t_common, len(f_common)),
            "r_mean": r_mean.flatten(),
            "z_mean": z_mean.flatten(),
            "n": n.flatten(),
            "p": p_vals.flatten(),
            "q": q_vals.flatten(),
            "significant": sig_mask.flatten(),
        }
    )
    df_out.to_csv(out_tsv, sep="\t", index=False)

    extent = [t_common[0], t_common[-1], f_common[0], f_common[-1]]
    cmap = "RdBu_r"
    vmin = -0.6
    vmax = 0.6
    figure_paths = []

    plot_cfg_small = get_plot_config(config)
    fig_size_small = plot_cfg_small.get_figure_size("small", plot_type="tfr")
    fig1, ax1 = plt.subplots(figsize=fig_size_small)
    im1 = ax1.imshow(
        r_mean,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax1.axvline(0.0, color="k", linestyle="--", alpha=0.6)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Frequency (Hz)")
    title_roi = roi or "All channels"
    title_method = method_suffix.strip("_").title()
    ax1.set_title(f"Group TF correlation — mean r ({title_method}, {title_roi})")
    cb1 = plt.colorbar(im1, ax=ax1)
    cb1.set_label("r")
    plt.tight_layout()
    _annotate_tf_correlation_figure(fig1, config, alpha)
    _save_fig(
        fig1,
        plots_dir,
        f"tf_corr_group_rmean{roi_suffix}{method_suffix}",
        config,
        logger=logger,
    )
    for ext in plot_cfg_small.formats:
        figure_paths.append(
            plots_dir / f"tf_corr_group_rmean{roi_suffix}{method_suffix}.{ext}"
        )

    fig2, ax2 = plt.subplots(figsize=fig_size_small)
    im2 = ax2.imshow(
        np.where(sig_mask, r_mean, np.nan),
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax2.axvline(0.0, color="k", linestyle="--", alpha=0.6)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (Hz)")
    sig_title = f"Group TF correlation — FDR<{alpha:g} ({title_method}, {title_roi})"
    ax2.set_title(sig_title)
    cb2 = plt.colorbar(im2, ax=ax2)
    cb2.set_label("r (significant)")
    plt.tight_layout()
    _annotate_tf_correlation_figure(fig2, config, alpha)
    _save_fig(
        fig2,
        plots_dir,
        f"tf_corr_group_sig{roi_suffix}{method_suffix}",
        config,
        logger=logger,
    )
    for ext in plot_cfg_small.formats:
        figure_paths.append(
            plots_dir / f"tf_corr_group_sig{roi_suffix}{method_suffix}.{ext}"
        )

    log(
        f"Group TF correlation saved (ROI={roi or 'all'}, method={method_suffix.strip('_')}): {out_tsv}",
        logger,
        "info"
    )
    return out_tsv, figure_paths


__all__ = [
    "group_tf_correlation",
]

