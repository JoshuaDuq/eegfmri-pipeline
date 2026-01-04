"""
Aperiodic component visualization plotting functions.

Functions for creating aperiodic component plots including topomaps, QC histograms,
residual spectra, run trajectories, and behavior scatter plots.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.infra.paths import ensure_dir, deriv_stats_path
from eeg_pipeline.plotting.io.figures import save_fig, log_if_present
from eeg_pipeline.infra.logging import get_logger
from eeg_pipeline.utils.data.columns import get_pain_column_from_config
from ..config import get_plot_config
from ...utils.analysis.stats import fdr_bh
from .utils import get_condition_colors, get_fdr_alpha


def _extract_aperiodic_data(
    features_df: pd.DataFrame,
    metric: str,
    info: mne.Info,
    mask: Optional[pd.Series] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Extract aperiodic metric values for channels in info.
    
    Args:
        features_df: Features DataFrame
        metric: Metric name ("slope" or "offset")
        info: MNE Info object
        mask: Optional boolean mask to subset trials
    
    Returns:
        Tuple of (array of values, list of channel names found)
    """
    data = []
    found_chs = []
    df_masked = features_df if mask is None else features_df[mask]
    
    for ch_name in info.ch_names:
        col = NamingSchema.build(
            "aperiodic",
            "active",
            "broadband",
            "ch",
            metric,
            channel=ch_name,
        )
        if col not in df_masked.columns:
            continue
        val = pd.to_numeric(df_masked[col], errors="coerce").mean()
        if not np.isnan(val):
            data.append(val)
            found_chs.append(ch_name)

    return np.array(data), found_chs


def _load_aperiodic_qc(subject: str, config: Any, logger: logging.Logger):
    """Load aperiodic QC data from npz file.
    
    Args:
        subject: Subject identifier
        config: Configuration object
        logger: Logger instance
    
    Returns:
        Loaded npz data or None if not found
    """
    try:
        stats_dir = deriv_stats_path(config.deriv_root, subject)
        qc_path = stats_dir / "aperiodic_qc.npz"
    except (AttributeError, TypeError):
        qc_path = None
    if qc_path is None or not qc_path.exists():
        log_if_present(logger, "warning", "Aperiodic QC sidecar not found; skipping QC plots")
        return None
    try:
        return np.load(qc_path, allow_pickle=True)
    except Exception as exc:
        log_if_present(logger, "warning", f"Failed to load aperiodic QC npz: {exc}")
        return None


def plot_aperiodic_residual_spectra(
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    """Plot aperiodic residual spectra showing fit quality across frequencies.
    
    Args:
        subject: Subject identifier
        save_dir: Directory to save plot
        logger: Logger instance
        config: Configuration object
    """
    qc = _load_aperiodic_qc(subject, config, logger)
    if qc is None or "residual_mean" not in qc or "freqs" not in qc:
        return

    residual_mean = qc["residual_mean"]
    freqs = qc["freqs"]
    if residual_mean.size == 0:
        return

    plot_cfg = get_plot_config(config)
    mean_resid = np.nanmean(residual_mean, axis=0)
    p5 = np.nanpercentile(residual_mean, 5, axis=0)
    p95 = np.nanpercentile(residual_mean, 95, axis=0)

    fig_size = plot_cfg.get_figure_size("standard", plot_type="features")
    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(freqs, mean_resid, label="Mean residual", color="k")
    ax.fill_between(freqs, p5, p95, color="gray", alpha=0.3, label="5–95% channels")
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Residual (log PSD – fit)")
    ax.set_title(f"Aperiodic residual spectra (sub-{subject})")
    ax.legend()

    output = save_dir / f"sub-{subject}_aperiodic_residual_spectra"
    save_fig(fig, output, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    log_if_present(logger, "info", "Saved aperiodic residual spectra")


def plot_aperiodic_run_trajectories(
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    """Plot aperiodic slope and offset trajectories across runs.
    
    Args:
        subject: Subject identifier
        save_dir: Directory to save plot
        logger: Logger instance
        config: Configuration object
    """
    qc = _load_aperiodic_qc(subject, config, logger)
    if qc is None or "slopes" not in qc or "offsets" not in qc:
        return
    run_labels = qc.get("run_labels")
    if run_labels is None:
        log_if_present(logger, "warning", "No run labels in QC; skipping run-wise trajectories")
        return

    slopes = qc["slopes"]
    offsets = qc["offsets"]
    runs_unique = pd.unique(run_labels)
    if runs_unique.size == 0:
        return

    slope_means = []
    slope_sems = []
    offset_means = []
    offset_sems = []
    run_order = []
    for run in runs_unique:
        mask = run_labels == run
        if not np.any(mask):
            continue
        slope_vals = slopes[mask, :].ravel()
        offset_vals = offsets[mask, :].ravel()
        slope_means.append(np.nanmean(slope_vals))
        slope_sems.append(np.nanstd(slope_vals) / np.sqrt(np.sum(np.isfinite(slope_vals))))
        offset_means.append(np.nanmean(offset_vals))
        offset_sems.append(np.nanstd(offset_vals) / np.sqrt(np.sum(np.isfinite(offset_vals))))
        run_order.append(run)

    if not run_order:
        return

    plot_cfg = get_plot_config(config)
    fig_size = plot_cfg.get_figure_size("wide", plot_type="features")
    fig, axes = plt.subplots(1, 2, figsize=fig_size)
    axes[0].errorbar(run_order, slope_means, yerr=slope_sems, fmt="o-", color="purple", capsize=3)
    axes[0].set_title("Slope by run")
    axes[0].set_ylabel("Aperiodic slope")
    axes[0].set_xlabel("Run")
    axes[0].grid(alpha=0.3)

    axes[1].errorbar(run_order, offset_means, yerr=offset_sems, fmt="o-", color="teal", capsize=3)
    axes[1].set_title("Offset by run")
    axes[1].set_ylabel("Aperiodic offset")
    axes[1].set_xlabel("Run")
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"Aperiodic trajectories by run (sub-{subject})")
    plt.tight_layout()
    output = save_dir / f"sub-{subject}_aperiodic_run_trajectories"
    save_fig(fig, output, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    log_if_present(logger, "info", "Saved aperiodic run-wise trajectories")


def plot_aperiodic_topomaps(
    features_df: pd.DataFrame,
    events_df: Optional[pd.DataFrame],
    info: mne.Info,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    """Plot topomaps for aperiodic slope and offset, split by pain condition.
    
    Plots topomaps for aperiodic slope and offset, split by pain condition
    when a pain column is available. Also plots a pain-minus-nonpain contrast.
    
    Args:
        features_df: Features DataFrame
        events_df: Optional events DataFrame
        info: MNE Info object
        subject: Subject identifier
        save_dir: Directory to save plot
        logger: Logger instance
        config: Configuration object
    """
    if features_df is None or features_df.empty:
        log_if_present(logger, "warning", "No feature data for aperiodic topomaps")
        return

    plot_cfg = get_plot_config(config)

    pain_col = get_pain_column_from_config(config, events_df) if events_df is not None else None
    pain_mask = None
    nonpain_mask = None
    if pain_col and len(features_df) == len(events_df):
        pain_series = pd.to_numeric(events_df[pain_col], errors="coerce")
        nonpain_mask = pain_series == 0
        pain_mask = pain_series == 1
        if nonpain_mask.sum() == 0 or pain_mask.sum() == 0:
            log_if_present(logger, "warning", "Pain/non-pain topomaps skipped (missing one condition)")
            pain_mask = nonpain_mask = None
    elif pain_col:
        log_if_present(logger, "warning", "Pain column found but events/features length mismatch; using overall mean.")

    all_pvals: List[float] = []
    per_metric_pvals: Dict[str, List[float]] = {}
    per_metric_common: Dict[str, Dict[str, Any]] = {}

    for metric in ["slope", "offset"]:
        data_overall, found_chs_overall = _extract_aperiodic_data(features_df, metric, info)
        if len(data_overall) == 0:
            log_if_present(logger, "warning", f"No {metric} data found for topomap")
            continue
        per_metric_pvals[metric] = []

        try:
            picks = mne.pick_channels(info.ch_names, found_chs_overall)
            info_subset = mne.pick_info(info, picks)
        except Exception as e:
            log_if_present(logger, "warning", f"Failed to pick channels for {metric} topomap: {e}")
            continue

        per_metric_common[metric] = {
            "info_subset": info_subset,
            "data_overall": data_overall,
            "found_chs_overall": found_chs_overall,
            "pair_data": None,
        }

        if pain_mask is not None and nonpain_mask is not None:
            data_nonpain, ch_nonpain = _extract_aperiodic_data(features_df, metric, info, mask=nonpain_mask)
            data_pain, ch_pain = _extract_aperiodic_data(features_df, metric, info, mask=pain_mask)
            common_chs = [ch for ch in found_chs_overall if ch in ch_nonpain and ch in ch_pain]
            if common_chs:
                run_col = None
                for cand in ["run_id", "run", "block"]:
                    if cand in events_df.columns:
                        run_col = cand
                        break
                if run_col is None:
                    log_if_present(logger, "warning", "No run/block column found; skipping pain vs non-pain channel tests to avoid non-independent trials.")
                    continue

                p_vals = []
                min_samples = int(config.get("statistics.min_samples_per_channel", 5))
                for ch in common_chs:
                    # Try new naming first
                    col_new = f"aperiodic_active_broadband_ch_{ch}_{metric}"
                    col_old = f"aper_{metric}_{ch}"
                    col = col_new if col_new in features_df.columns else col_old
                    if col not in features_df.columns:
                        p_vals.append(np.nan)
                        continue
                    series = pd.to_numeric(features_df[col], errors="coerce")
                    if run_col:
                        runs = events_df[run_col]
                        vals_pain = series[pain_mask].groupby(runs[pain_mask]).mean().dropna()
                        vals_nonpain = series[nonpain_mask].groupby(runs[nonpain_mask]).mean().dropna()
                    else:
                        vals_pain = series[pain_mask].dropna()
                        vals_nonpain = series[nonpain_mask].dropna()

                    if len(vals_pain) < min_samples or len(vals_nonpain) < min_samples:
                        p_vals.append(np.nan)
                        continue
                    try:
                        _, pval = stats.mannwhitneyu(vals_pain, vals_nonpain, alternative="two-sided")
                    except ValueError:
                        pval = np.nan
                    p_vals.append(pval)

                per_metric_pvals[metric] = p_vals
                all_pvals.extend([p for p in p_vals if np.isfinite(p)])
                
                def get_col(ch, metric):
                    c_new = f"aperiodic_active_broadband_ch_{ch}_{metric}"
                    c_old = f"aper_{metric}_{ch}"
                    return c_new if c_new in features_df.columns else c_old
                
                per_metric_common[metric]["pair_data"] = {
                    "common_chs": common_chs,
                    "data_nonpain": np.array([features_df.loc[nonpain_mask, get_col(ch, metric)].mean() for ch in common_chs]),
                    "data_pain": np.array([features_df.loc[pain_mask, get_col(ch, metric)].mean() for ch in common_chs]),
                }

    per_metric_qvals: Dict[str, np.ndarray] = {}
    if all_pvals:
        q_all = fdr_bh(all_pvals, config=config)
        idx = 0
        for metric, p_vals in per_metric_pvals.items():
            q_vals = []
            for p in p_vals:
                if np.isfinite(p):
                    q_vals.append(float(q_all[idx]))
                    idx += 1
                else:
                    q_vals.append(np.nan)
            per_metric_qvals[metric] = np.array(q_vals, dtype=float)

    alpha = get_fdr_alpha(config)
    
    metrics = list(per_metric_common.keys())
    if not metrics:
        log_if_present(logger, "warning", "No aperiodic metrics available for topomap grid")
        return
    
    has_pain_data = any(meta.get("pair_data") is not None for meta in per_metric_common.values())
    n_cols = 4 if has_pain_data else 1
    n_rows = len(metrics)
    
    fig_width = float(plot_cfg.plot_type_configs.get("aperiodic", {}).get("width_per_column", 5.0)) * n_cols
    fig_height = float(plot_cfg.plot_type_configs.get("aperiodic", {}).get("height_per_row", 4.5)) * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    
    for row_idx, metric in enumerate(metrics):
        meta = per_metric_common[metric]
        info_subset = meta["info_subset"]
        data_overall = meta["data_overall"]
        cmap = "RdBu_r" if metric == "slope" else "viridis"
        
        ax = axes[row_idx, 0]
        im, _ = mne.viz.plot_topomap(
            data_overall, info_subset, axes=ax, show=False, cmap=cmap, contours=6
        )
        ax.set_title(f"{metric.capitalize()} - Overall")
        
        if not has_pain_data or n_cols == 1:
            for col_idx in range(1, n_cols):
                axes[row_idx, col_idx].axis("off")
            continue
        
        pair_data = meta.get("pair_data")
        if pair_data is not None:
            common_chs = pair_data["common_chs"]
            data_nonpain = pair_data["data_nonpain"]
            data_pain = pair_data["data_pain"]
            picks_common = mne.pick_channels(info.ch_names, common_chs)
            info_common = mne.pick_info(info, picks_common)
            diff = data_pain - data_nonpain
            q_vals = per_metric_qvals.get(metric, np.full(len(common_chs), np.nan))
            sig_mask = np.isfinite(q_vals) & (q_vals < alpha)
            
            ax = axes[row_idx, 1]
            mne.viz.plot_topomap(
                data_nonpain, info_common, axes=ax, show=False, cmap=cmap, contours=6
            )
            ax.set_title(f"{metric.capitalize()} - Non-pain")
            
            ax = axes[row_idx, 2]
            mne.viz.plot_topomap(
                data_pain, info_common, axes=ax, show=False, cmap=cmap, contours=6
            )
            ax.set_title(f"{metric.capitalize()} - Pain")
            
            ax = axes[row_idx, 3]
            mne.viz.plot_topomap(
                diff,
                info_common,
                axes=ax,
                show=False,
                cmap="RdBu_r",
                contours=6,
                mask=sig_mask,
                mask_params=dict(
                    marker="o",
                    markerfacecolor="none",
                    markeredgecolor="black",
                    linewidth=1.0,
                    markersize=8,
                ) if np.any(sig_mask) else None,
            )
            title = f"{metric.capitalize()} - Contrast"
            if np.any(sig_mask):
                q_min = np.nanmin(q_vals[sig_mask])
                title += f"\\n(FDR<{alpha:.2f}, min q={q_min:.3f})"
            ax.set_title(title)
        else:
            for col_idx in range(1, n_cols):
                axes[row_idx, col_idx].axis("off")
    
    fig.suptitle(f"Aperiodic Component Topomaps (sub-{subject})", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_name = f"sub-{subject}_aperiodic_topomaps_grid"
    save_fig(
        fig,
        save_dir / output_name,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches
    )
    plt.close(fig)
    log_if_present(logger, "info", f"Saved aperiodic topomap grid ({n_rows} metrics × {n_cols} conditions)")


def plot_aperiodic_vs_pain(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    """Scatter plot of mean aperiodic slope (across channels) vs pain ratings.
    
    Args:
        features_df: Features DataFrame
        events_df: Events DataFrame
        subject: Subject identifier
        save_dir: Directory to save plot
        logger: Logger instance
        config: Configuration object
    """
    if features_df is None or features_df.empty or events_df is None or events_df.empty:
        return

    rating_col = None
    potential_cols = ["rating", "vas", "pain_rating", "response"]
    config_rating = config.get("event_columns.rating")
    
    if config_rating:
        if isinstance(config_rating, list):
             for c in reversed(config_rating):
                 potential_cols.insert(0, c)
        else:
             potential_cols.insert(0, config_rating)
        
    for col in potential_cols:
        if col in events_df.columns:
            rating_col = col
            break
            
    if rating_col is None:
        log_if_present(logger, "warning", "No rating column found for aperiodic scatter plot")
        return

    slope_cols = [c for c in features_df.columns if c.startswith("aper_slope_")]
    if not slope_cols:
        return
    
    if "aper_slope_" in slope_cols[0]:
        if events_df is not None and rating_col in events_df.columns:
            pain_col = get_pain_column_from_config(config, events_df)
        else:
            pain_col = None
        if pain_col and len(features_df) == len(events_df):
            pain_series = pd.to_numeric(events_df[pain_col], errors="coerce")
            nonpain_mask = pain_series == 0
            pain_mask = pain_series == 1
            common_chs = []
            for col in slope_cols:
                vals = pd.to_numeric(features_df[col], errors="coerce")
                if vals[nonpain_mask].notna().all() and vals[pain_mask].notna().all():
                    common_chs.append(col)
        else:
            common_chs = [
                col for col in slope_cols
                if pd.to_numeric(features_df[col], errors="coerce").notna().all()
            ]
        min_ch = int(config.get("statistics.min_channels_for_aperiodic_corr", 10))
        if len(common_chs) < min_ch:
            log_if_present(logger, "warning", f"Insufficient common aperiodic channels across conditions ({len(common_chs)}<{min_ch}); skipping slope vs pain plot")
            return
        mean_slopes = features_df[common_chs].mean(axis=1)
    else:
        mean_slopes = features_df[slope_cols].mean(axis=1)
    ratings = pd.to_numeric(events_df[rating_col], errors="coerce")
    
    if len(mean_slopes) != len(ratings):
        log_if_present(logger, "warning", "Mismatch in features and events length for scatter plot")
        return

    valid_mask = mean_slopes.notna() & ratings.notna()
    mean_slopes = mean_slopes[valid_mask]
    ratings = ratings[valid_mask]
    events_aligned = events_df.loc[valid_mask]

    good_chs: List[str] = []
    for col in slope_cols:
        vals = pd.to_numeric(features_df[col], errors="coerce").loc[valid_mask]
        if vals.notna().all():
            good_chs.append(col)
    min_ch = int(config.get("statistics.min_channels_for_aperiodic_corr", 10))
    if len(good_chs) < min_ch:
        log_if_present(logger, "warning", f"Insufficient common aperiodic channels ({len(good_chs)}<{min_ch}); skipping slope vs pain plot")
        return

    run_col = None
    for cand in ["run_id", "run", "block"]:
        if cand in events_aligned.columns:
            run_col = cand
            break

    if run_col:
        grouped = events_aligned.assign(mean_slope=mean_slopes).groupby(run_col)
        slopes_agg = grouped["mean_slope"].mean()
        ratings_agg = grouped[rating_col].mean()
        agg_label = f"run-mean ({run_col})"
    else:
        log_if_present(logger, "warning", "No run/block column found; skipping aperiodic slope vs pain (cannot assume independent trials).")
        return

    if len(slopes_agg) < 5:
        return

    r_val, _ = stats.spearmanr(ratings_agg, slopes_agg)

    n_perm = int(get_config_value(config, "plotting.plots.aperiodic.n_perm", get_config_value(config, "behavior_analysis.statistics.n_permutations", 1000)))
    rng = np.random.default_rng(int(config.get("project.random_state", 42)))
    observed = abs(r_val)
    perm_ge = 0
    ratings_arr = ratings_agg.to_numpy()
    slopes_arr = slopes_agg.to_numpy()
    n_iter = max(10, n_perm)
    for _ in range(n_iter):
        shuffled = rng.permutation(ratings_arr)
        r_perm, _ = stats.spearmanr(shuffled, slopes_arr)
        if abs(r_perm) >= observed:
            perm_ge += 1
    p_perm = (perm_ge + 1) / (n_iter + 1)

    p_perm_for_bh = [p_perm]
    if any(c.startswith("aper_offset_") for c in features_df.columns):
        offset_cols = [c for c in features_df.columns if c.startswith("aper_offset_")]
        mean_offset = features_df[offset_cols].mean(axis=1)[valid_mask]
        if len(mean_offset) == len(ratings):
            if run_col:
                grouped_off = events_aligned.assign(mean_offset=mean_offset).groupby(run_col)
                off_agg = grouped_off["mean_offset"].mean()
                ratings_off = grouped_off[rating_col].mean()
            else:
                off_agg = mean_offset
                ratings_off = ratings
            if len(off_agg) >= 5:
                r_off, _ = stats.spearmanr(ratings_off, off_agg)
                observed_off = abs(r_off)
                perm_ge_off = 0
                off_arr = off_agg.to_numpy()
                ratings_off_arr = ratings_off.to_numpy()
                for _ in range(n_iter):
                    shuffled_off = rng.permutation(ratings_off_arr)
                    r_perm_off, _ = stats.spearmanr(shuffled_off, off_arr)
                    if abs(r_perm_off) >= observed_off:
                        perm_ge_off += 1
                p_perm_off = (perm_ge_off + 1) / (n_iter + 1)
                p_perm_for_bh.append(p_perm_off)
    q_vals_perm = fdr_bh(p_perm_for_bh, config=config)
    q_perm = q_vals_perm[0] if q_vals_perm.size else p_perm

    plot_cfg = get_plot_config(config)
    fig_size = plot_cfg.get_figure_size("standard", plot_type="features")
    fig, ax = plt.subplots(figsize=fig_size)
    
    sns.scatterplot(x=ratings_agg, y=slopes_agg, ax=ax, alpha=0.6)
    sns.regplot(x=ratings_agg, y=slopes_agg, ax=ax, scatter=False, lowess=True, line_kws={'color': 'red'})
    
    ax.set_xlabel(f"Pain Rating ({rating_col}, {agg_label})")
    ax.set_ylabel("Mean Aperiodic Slope")
    ax.set_title(
        f"Aperiodic Slope vs Pain (sub-{subject})\n"
        f"Spearman r={r_val:.2f}, perm q={q_perm:.3f}"
    )
    
    output_name = f"sub-{subject}_aperiodic_slope_vs_pain"
    save_fig(
        fig,
        save_dir / output_name,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches
    )
    plt.close(fig)
    log_if_present(logger, "info", "Saved aperiodic slope vs pain scatter")


def plot_aperiodic_by_condition(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    stats_dir: Optional[Path] = None,
) -> None:
    """Compare aperiodic features between conditions per metric (slope/offset).
    
    For window comparisons (paired): Uses the unified plot_paired_comparison helper.
    For column comparisons (unpaired): Uses Mann-Whitney U test with consistent styling.
    Creates one figure per ROI.
    
    If stats_dir is provided, uses pre-computed statistics from the behavior pipeline.
    """
    if features_df is None or features_df.empty or events_df is None:
        return

    from eeg_pipeline.utils.config.loader import get_config_value
    from eeg_pipeline.utils.analysis.events import extract_comparison_mask
    from eeg_pipeline.plotting.features.utils import (
        plot_paired_comparison,
        apply_fdr_correction,
        get_named_segments,
    )
    from eeg_pipeline.plotting.features.roi import get_roi_definitions, get_roi_channels

    compare_wins = get_config_value(config, "plotting.comparisons.compare_windows", True)
    compare_cols = get_config_value(config, "plotting.comparisons.compare_columns", False)
    
    # Get segments from config or auto-detect from data
    segments = get_config_value(config, "plotting.comparisons.comparison_windows", [])
    if not segments or len(segments) < 2:
        detected = get_named_segments(features_df, group="aperiodic")
        if len(detected) >= 2:
            segments = detected[:2]
            if logger:
                logger.info(f"Auto-detected segments for aperiodic comparison: {segments}")
    
    # Aperiodic uses metrics (slope, offset) instead of bands
    metrics = ["slope", "offset"]
    metric_colors = {"slope": "#8E44AD", "offset": "#16A085"}  # Purple for slope, teal for offset
    
    # Get ROI definitions
    rois = get_roi_definitions(config)
    all_channels = set()
    for col in features_df.columns:
        parsed = NamingSchema.parse(str(col))
        if parsed.get("valid") and parsed.get("group") == "aperiodic" and parsed.get("scope") == "ch":
            ch = parsed.get("identifier")
            if ch:
                all_channels.add(str(ch))
    all_channels = list(all_channels)
    
    comp_rois = get_config_value(config, "plotting.comparisons.comparison_rois", [])
    if comp_rois:
        roi_names = []
        for r in comp_rois:
            if r.lower() == "all":
                if "all" not in roi_names:
                    roi_names.append("all")
            elif r in rois:
                roi_names.append(r)
    else:
        roi_names = ["all"]
        if rois:
            roi_names.extend(list(rois.keys()))
    
    if logger:
        logger.info(f"Aperiodic comparison: segments={segments}, ROIs={roi_names}, metrics={metrics}, compare_windows={compare_wins}, compare_columns={compare_cols}")
    
    plot_cfg = get_plot_config(config)
    ensure_dir(save_dir)
    
    # Helper to get aperiodic columns for a segment/metric/ROI
    def get_aperiodic_columns(segment, metric, roi_name):
        """Get aperiodic columns filtered by segment, metric, and ROI."""
        cols = []
        roi_channels = all_channels if roi_name == "all" else get_roi_channels(rois.get(roi_name, []), all_channels)
        roi_set = set(roi_channels) if roi_channels else set(all_channels)
        
        for col in features_df.columns:
            parsed = NamingSchema.parse(str(col))
            if not parsed.get("valid"):
                continue
            if parsed.get("group") != "aperiodic":
                continue
            if str(parsed.get("segment") or "") != segment:
                continue
            if str(parsed.get("stat") or "") != metric:
                continue
            # Accept global scope or ch scope in ROI
            scope = parsed.get("scope") or ""
            if scope in ("global", "roi"):
                cols.append(col)
            elif scope == "ch":
                ch_id = str(parsed.get("identifier") or "")
                if ch_id in roi_set:
                    cols.append(col)
        return cols
    
    # Window comparison (paired) - use unified helper
    if compare_wins and len(segments) >= 2:
        seg1, seg2 = segments[0], segments[1]
        
        for roi_name in roi_names:
            # For aperiodic, we use metrics instead of bands
            data_by_band = {}  # Reusing the dict name for compatibility with helper
            for metric in metrics:
                cols1 = get_aperiodic_columns(seg1, metric, roi_name)
                cols2 = get_aperiodic_columns(seg2, metric, roi_name)
                
                if not cols1 or not cols2:
                    continue
                
                s1 = features_df[cols1].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                s2 = features_df[cols2].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                
                valid_mask = s1.notna() & s2.notna()
                v1, v2 = s1[valid_mask].values, s2[valid_mask].values
                
                if len(v1) > 0:
                    data_by_band[metric] = (v1, v2)
            
            if data_by_band:
                roi_safe = roi_name.replace(" ", "_").lower() if roi_name != "all" else ""
                suffix = f"_roi-{roi_safe}" if roi_safe else ""
                save_path = save_dir / f"sub-{subject}_aperiodic_by_condition{suffix}_window"
                
                plot_paired_comparison(
                    data_by_band=data_by_band,
                    subject=subject,
                    save_path=save_path,
                    feature_label="Aperiodic",
                    config=config,
                    logger=logger,
                    label1=seg1.capitalize(),
                    label2=seg2.capitalize(),
                    roi_name=roi_name,
                    stats_dir=stats_dir,
                )
        
        log_if_present(logger, "info", f"Saved aperiodic paired comparison plots for {len(roi_names)} ROIs")

    # Column comparison (unpaired)
    if compare_cols:
        comp_mask_info = extract_comparison_mask(events_df, config)
        if not comp_mask_info:
            if logger:
                logger.debug("Column comparison requested but config incomplete")
        else:
            m1, m2, label1, label2 = comp_mask_info
            seg_name = get_config_value(config, "plotting.comparisons.comparison_segment", "active")
            
            segment_colors = {"v1": "#5a7d9a", "v2": "#c44e52"}
            n_metrics = len(metrics)
            n_trials = len(features_df)
            
            for roi_name in roi_names:
                all_pvals, pvalue_keys, cell_data = [], [], {}
                
                for col_idx, metric in enumerate(metrics):
                    cols = get_aperiodic_columns(seg_name, metric, roi_name)
                    
                    if not cols:
                        cell_data[col_idx] = None
                        continue
                    
                    val_series = features_df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                    v1 = val_series[m1].dropna().values
                    v2 = val_series[m2].dropna().values
                    
                    cell_data[col_idx] = {"v1": v1, "v2": v2}
                    
                    if len(v1) >= 3 and len(v2) >= 3:
                        try:
                            _, p = stats.mannwhitneyu(v1, v2, alternative="two-sided")
                            diff = np.mean(v2) - np.mean(v1)
                            pooled_std = np.sqrt(((len(v1)-1)*np.var(v1, ddof=1) + (len(v2)-1)*np.var(v2, ddof=1)) / (len(v1)+len(v2)-2))
                            d = diff / pooled_std if pooled_std > 0 else 0
                            all_pvals.append(p)
                            pvalue_keys.append((col_idx, p, d))
                        except Exception:
                            pass
                
                qvalues = {}
                n_significant = 0
                if all_pvals:
                    rejected, qvals, _ = apply_fdr_correction(all_pvals, config=config)
                    for i, (key, p, d) in enumerate(pvalue_keys):
                        qvalues[key] = (p, qvals[i], d, rejected[i])
                    n_significant = int(np.sum(rejected))
                
                fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5), squeeze=False)
                
                for col_idx, metric in enumerate(metrics):
                    ax = axes.flatten()[col_idx]
                    data = cell_data.get(col_idx)
                    
                    if data is None or len(data.get("v1", [])) == 0 or len(data.get("v2", [])) == 0:
                        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                               transform=ax.transAxes, fontsize=plot_cfg.font.title, color="gray")
                        ax.set_xticks([])
                        continue
                    
                    v1, v2 = data["v1"], data["v2"]
                    
                    bp = ax.boxplot([v1, v2], positions=[0, 1], widths=0.4, patch_artist=True)
                    bp["boxes"][0].set_facecolor(segment_colors["v1"])
                    bp["boxes"][0].set_alpha(0.6)
                    bp["boxes"][1].set_facecolor(segment_colors["v2"])
                    bp["boxes"][1].set_alpha(0.6)
                    
                    ax.scatter(np.random.uniform(-0.08, 0.08, len(v1)), v1, c=segment_colors["v1"], alpha=0.3, s=6)
                    ax.scatter(1 + np.random.uniform(-0.08, 0.08, len(v2)), v2, c=segment_colors["v2"], alpha=0.3, s=6)
                    
                    all_vals = np.concatenate([v1, v2])
                    ymin, ymax = np.nanmin(all_vals), np.nanmax(all_vals)
                    yrange = ymax - ymin if ymax > ymin else 0.1
                    ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.3 * yrange)
                    
                    if col_idx in qvalues:
                        _, q, d, sig = qvalues[col_idx]
                        sig_marker = "†" if sig else ""
                        sig_color = "#d62728" if sig else "#333333"
                        ax.annotate(f"q={q:.3f}{sig_marker}\nd={d:.2f}", xy=(0.5, ymax + 0.05 * yrange),
                                   ha="center", fontsize=plot_cfg.font.medium, color=sig_color,
                                   fontweight="bold" if sig else "normal")
                    
                    ax.set_xticks([0, 1])
                    ax.set_xticklabels([label1, label2], fontsize=9)
                    ax.set_title(metric.capitalize(), fontweight="bold", color=metric_colors.get(metric, "gray"))
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                
                n_tests = len(all_pvals)
                roi_display = roi_name.replace("_", " ").title() if roi_name != "all" else "All Channels"
                
                title = (f"Aperiodic: {label1} vs {label2} (Column Comparison)\n"
                         f"Subject: {subject} | ROI: {roi_display} | N: {n_trials} trials | Mann-Whitney U | "
                         f"FDR: {n_significant}/{n_tests} significant (†=q<0.05)")
                fig.suptitle(title, fontsize=plot_cfg.font.suptitle, fontweight="bold", y=1.02)
                
                plt.tight_layout()
                
                roi_safe = roi_name.replace(" ", "_").lower() if roi_name != "all" else ""
                suffix = f"_roi-{roi_safe}" if roi_safe else ""
                filename = f"sub-{subject}_aperiodic_by_condition{suffix}_column"
                
                save_fig(fig, save_dir / filename, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
                         bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
                plt.close(fig)
            
            log_if_present(logger, "info", f"Saved aperiodic column comparison plots for {len(roi_names)} ROIs")


