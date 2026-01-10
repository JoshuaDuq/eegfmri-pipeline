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
from eeg_pipeline.utils.config.loader import get_config_value
from eeg_pipeline.utils.data.columns import get_pain_column_from_config
from ..config import get_plot_config
from ...utils.analysis.stats import fdr_bh
from .utils import get_condition_colors, get_fdr_alpha


# Constants
MIN_SAMPLES_PER_CHANNEL_DEFAULT = 5
MIN_CHANNELS_FOR_APERIODIC_CORR_DEFAULT = 10
MIN_RUNS_FOR_CORRELATION = 5
MIN_PERMUTATIONS = 10
DEFAULT_N_PERMUTATIONS = 1000
DEFAULT_RANDOM_STATE = 42
PERCENTILE_LOW = 5
PERCENTILE_HIGH = 95
TOPO_CONTOURS = 6
SCATTER_ALPHA = 0.6
BOX_ALPHA = 0.6
SCATTER_SIZE = 6
RUN_COLUMN_CANDIDATES = ["run_id", "run", "block"]
RATING_COLUMN_CANDIDATES = ["rating", "vas", "pain_rating", "response"]
APERIODIC_METRICS = ["slope", "offset"]
SLOPE_COLOR = "#8E44AD"
OFFSET_COLOR = "#16A085"
CONDITION_COLOR_V1 = "#5a7d9a"
CONDITION_COLOR_V2 = "#c44e52"
SIGNIFICANT_COLOR = "#d62728"
NON_SIGNIFICANT_COLOR = "#333333"


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


def _load_aperiodic_qc(
    subject: str, config: Any, logger: logging.Logger
) -> Optional[Dict[str, Any]]:
    """Load aperiodic QC data from TSV file.

    Args:
        subject: Subject identifier
        config: Configuration object
        logger: Logger instance

    Returns:
        Loaded QC data as dict or None if not found
    """
    try:
        stats_dir = deriv_stats_path(config.deriv_root, subject)
        qc_path = stats_dir / "aperiodic_qc.tsv"
    except (AttributeError, TypeError):
        qc_path = None

    if qc_path is None or not qc_path.exists():
        log_if_present(
            logger, "warning", "Aperiodic QC sidecar not found; skipping QC plots"
        )
        return None

    try:
        df = read_table(qc_path)
        if df.empty:
            return None

        qc_data: Dict[str, Any] = {}

        residual_rows = df[df["residual_mean"].notna() & df["frequency"].notna()]
        if not residual_rows.empty:
            channels = residual_rows["channel"].unique()
            freqs = residual_rows["frequency"].unique()
            residual_mean = np.full((len(channels), len(freqs)), np.nan)

            for ch_idx, ch in enumerate(channels):
                ch_data = residual_rows[residual_rows["channel"] == ch]
                for freq_idx, freq in enumerate(freqs):
                    row = ch_data[ch_data["frequency"] == freq]
                    if not row.empty:
                        residual_mean[ch_idx, freq_idx] = row["residual_mean"].values[0]

            qc_data["residual_mean"] = residual_mean
            qc_data["freqs"] = freqs

        r2_rows = df[df["r2"].notna()]
        if not r2_rows.empty:
            channels = r2_rows["channel"].unique()
            r2 = np.full(len(channels), np.nan)
            for ch_idx, ch in enumerate(channels):
                ch_data = r2_rows[r2_rows["channel"] == ch]
                if not ch_data.empty:
                    r2[ch_idx] = ch_data["r2"].values[0]
            qc_data["r2"] = r2

        slope_rows = df[df["slope"].notna()]
        if not slope_rows.empty:
            trials = slope_rows["trial"].unique()
            channels = slope_rows["channel"].unique()
            slopes = np.full((len(trials), len(channels)), np.nan)
            offsets = np.full((len(trials), len(channels)), np.nan)

            for trial_idx, trial in enumerate(trials):
                trial_data = slope_rows[slope_rows["trial"] == trial]
                for ch_idx, ch in enumerate(channels):
                    ch_data = trial_data[trial_data["channel"] == ch]
                    if not ch_data.empty:
                        slopes[trial_idx, ch_idx] = ch_data["slope"].values[0]
                        offsets[trial_idx, ch_idx] = ch_data["offset"].values[0]

            qc_data["slopes"] = slopes
            qc_data["offsets"] = offsets
            qc_data["run_labels"] = trials

        if "channel" in df.columns:
            qc_data["channel_names"] = df["channel"].unique().tolist()

        return qc_data

    except (OSError, IOError, ValueError, KeyError) as exc:
        log_if_present(
            logger, "warning", f"Failed to load aperiodic QC TSV: {exc}"
        )
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
    p5 = np.nanpercentile(residual_mean, PERCENTILE_LOW, axis=0)
    p95 = np.nanpercentile(residual_mean, PERCENTILE_HIGH, axis=0)

    fig_size = plot_cfg.get_figure_size("standard", plot_type="features")
    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(freqs, mean_resid, label="Mean residual", color="k")
    percentile_label = f"{PERCENTILE_LOW}–{PERCENTILE_HIGH}% channels"
    ax.fill_between(freqs, p5, p95, color="gray", alpha=0.3, label=percentile_label)
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
    axes[0].errorbar(
        run_order, slope_means, yerr=slope_sems, fmt="o-", color="purple", capsize=3
    )
    axes[0].set_title("Slope by run")
    axes[0].set_ylabel("Aperiodic slope")
    axes[0].set_xlabel("Run")
    axes[0].grid(alpha=0.3)

    axes[1].errorbar(
        run_order, offset_means, yerr=offset_sems, fmt="o-", color="teal", capsize=3
    )
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


def _extract_condition_masks(
    features_df: pd.DataFrame,
    events_df: Optional[pd.DataFrame],
    config: Any,
    logger: logging.Logger,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str], Optional[str]]:
    """Extract condition masks from events DataFrame.
    
    Returns:
        Tuple of (nonpain_mask, pain_mask, label1, label2) or (None, None, None, None)
    """
    if events_df is None or len(features_df) != len(events_df):
        if events_df is not None:
            log_if_present(
                logger,
                "warning",
                "Events/features length mismatch; using overall mean."
            )
        return None, None, None, None
    
    from eeg_pipeline.utils.analysis.events import extract_comparison_mask
    
    comp = extract_comparison_mask(events_df, config, require_enabled=False)
    if comp is None:
        return None, None, None, None
    
    nonpain_mask, pain_mask, label1, label2 = comp
    nonpain_mask = np.asarray(nonpain_mask, dtype=bool)
    pain_mask = np.asarray(pain_mask, dtype=bool)
    
    if nonpain_mask.sum() == 0 or pain_mask.sum() == 0:
        log_if_present(
            logger,
            "warning",
            "Condition topomaps skipped (missing one condition)"
        )
        return None, None, None, None
    
    return nonpain_mask, pain_mask, label1, label2


def _find_run_column(events_df: pd.DataFrame, config: Any) -> Optional[str]:
    """Find run/block column name from config or common candidates.
    
    Returns:
        Column name or None if not found
    """
    run_col = None
    if config is not None:
        run_col = config.get("behavior_analysis.run_adjustment.column", None)
        if run_col is not None:
            run_col = str(run_col).strip()
            if run_col and run_col in events_df.columns:
                return run_col
    
    for candidate in RUN_COLUMN_CANDIDATES:
        if candidate in events_df.columns:
            return candidate
    
    return None


def _get_aperiodic_column_name(channel: str, metric: str, features_df: pd.DataFrame) -> Optional[str]:
    """Get aperiodic column name using new or old naming scheme.
    
    Returns:
        Column name or None if not found
    """
    col_new = f"aperiodic_active_broadband_ch_{channel}_{metric}"
    col_old = f"aper_{metric}_{channel}"
    
    if col_new in features_df.columns:
        return col_new
    if col_old in features_df.columns:
        return col_old
    return None


def _compute_channel_pvalues(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    common_channels: List[str],
    metric: str,
    pain_mask: np.ndarray,
    nonpain_mask: np.ndarray,
    run_col: Optional[str],
    config: Any,
) -> List[float]:
    """Compute p-values for each channel comparing pain vs nonpain.
    
    Returns:
        List of p-values (may contain NaN)
    """
    min_samples = int(
        get_config_value(
            config, "statistics.min_samples_per_channel", MIN_SAMPLES_PER_CHANNEL_DEFAULT
        )
    )
    p_values = []
    
    for channel in common_channels:
        col = _get_aperiodic_column_name(channel, metric, features_df)
        if col is None:
            p_values.append(np.nan)
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
            p_values.append(np.nan)
            continue
        
        try:
            _, pval = stats.mannwhitneyu(
                vals_pain, vals_nonpain, alternative="two-sided"
            )
        except ValueError:
            pval = np.nan
        p_values.append(pval)
    
    return p_values


def _extract_pair_data(
    features_df: pd.DataFrame,
    common_channels: List[str],
    metric: str,
    pain_mask: np.ndarray,
    nonpain_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract mean values for pain and nonpain conditions.
    
    Returns:
        Tuple of (data_nonpain, data_pain) arrays
    """
    data_nonpain = []
    data_pain = []
    
    for channel in common_channels:
        col = _get_aperiodic_column_name(channel, metric, features_df)
        if col is None:
            continue
        data_nonpain.append(features_df.loc[nonpain_mask, col].mean())
        data_pain.append(features_df.loc[pain_mask, col].mean())
    
    return np.array(data_nonpain), np.array(data_pain)


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
    
    nonpain_mask, pain_mask, label1, label2 = _extract_condition_masks(
        features_df, events_df, config, logger
    )

    all_pvals: List[float] = []
    per_metric_pvals: Dict[str, List[float]] = {}
    per_metric_common: Dict[str, Dict[str, Any]] = {}

    for metric in APERIODIC_METRICS:
        data_overall, found_chs_overall = _extract_aperiodic_data(features_df, metric, info)
        if len(data_overall) == 0:
            log_if_present(logger, "warning", f"No {metric} data found for topomap")
            continue
        per_metric_pvals[metric] = []

        try:
            picks = mne.pick_channels(info.ch_names, found_chs_overall)
            info_subset = mne.pick_info(info, picks)
        except (ValueError, RuntimeError) as e:
            log_if_present(
                logger,
                "warning",
                f"Failed to pick channels for {metric} topomap: {e}"
            )
            continue

        per_metric_common[metric] = {
            "info_subset": info_subset,
            "data_overall": data_overall,
            "found_chs_overall": found_chs_overall,
            "pair_data": None,
        }

        if pain_mask is not None and nonpain_mask is not None:
            data_nonpain, ch_nonpain = _extract_aperiodic_data(
                features_df, metric, info, mask=nonpain_mask
            )
            data_pain, ch_pain = _extract_aperiodic_data(
                features_df, metric, info, mask=pain_mask
            )
            common_chs = [
                ch for ch in found_chs_overall
                if ch in ch_nonpain and ch in ch_pain
            ]
            
            if common_chs and events_df is not None:
                run_col = _find_run_column(events_df, config)
                if run_col is None:
                    log_if_present(
                        logger,
                        "warning",
                        "No run/block column found; skipping pain vs non-pain "
                        "channel tests to avoid non-independent trials."
                    )
                else:
                    p_vals = _compute_channel_pvalues(
                        features_df,
                        events_df,
                        common_chs,
                        metric,
                        pain_mask,
                        nonpain_mask,
                        run_col,
                        config,
                    )
                    
                    per_metric_pvals[metric] = p_vals
                    all_pvals.extend([p for p in p_vals if np.isfinite(p)])
                    
                    data_nonpain_arr, data_pain_arr = _extract_pair_data(
                        features_df, common_chs, metric, pain_mask, nonpain_mask
                    )
                    
                    per_metric_common[metric]["pair_data"] = {
                        "common_chs": common_chs,
                        "data_nonpain": data_nonpain_arr,
                        "data_pain": data_pain_arr,
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
    
    has_pain_data = any(
        meta.get("pair_data") is not None for meta in per_metric_common.values()
    )
    n_cols = 4 if has_pain_data else 1
    n_rows = len(metrics)
    
    aperiodic_config = plot_cfg.plot_type_configs.get("aperiodic", {})
    width_per_column = float(aperiodic_config.get("width_per_column", 5.0))
    height_per_row = float(aperiodic_config.get("height_per_row", 4.5))
    fig_width = width_per_column * n_cols
    fig_height = height_per_row * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    
    for row_idx, metric in enumerate(metrics):
        meta = per_metric_common[metric]
        info_subset = meta["info_subset"]
        data_overall = meta["data_overall"]
        cmap = "RdBu_r" if metric == "slope" else "viridis"
        
        ax = axes[row_idx, 0]
        mne.viz.plot_topomap(
            data_overall, info_subset, axes=ax, show=False, cmap=cmap,
            contours=TOPO_CONTOURS
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
            
            cond1_label = str(label1) if label1 is not None else "Condition 1"
            cond2_label = str(label2) if label2 is not None else "Condition 2"
            
            ax = axes[row_idx, 1]
            mne.viz.plot_topomap(
                data_nonpain, info_common, axes=ax, show=False, cmap=cmap,
                contours=TOPO_CONTOURS
            )
            ax.set_title(f"{metric.capitalize()} - {cond1_label}")
            
            ax = axes[row_idx, 2]
            mne.viz.plot_topomap(
                data_pain, info_common, axes=ax, show=False, cmap=cmap,
                contours=TOPO_CONTOURS
            )
            ax.set_title(f"{metric.capitalize()} - {cond2_label}")
            
            ax = axes[row_idx, 3]
            mask_params = None
            if np.any(sig_mask):
                mask_params = dict(
                    marker="o",
                    markerfacecolor="none",
                    markeredgecolor="black",
                    linewidth=1.0,
                    markersize=8,
                )
            mne.viz.plot_topomap(
                diff,
                info_common,
                axes=ax,
                show=False,
                cmap="RdBu_r",
                contours=TOPO_CONTOURS,
                mask=sig_mask,
                mask_params=mask_params,
            )
            title = f"{metric.capitalize()} - {cond2_label} minus {cond1_label}"
            if np.any(sig_mask):
                q_min = np.nanmin(q_vals[sig_mask])
                title += f"\\n(FDR<{alpha:.2f}, min q={q_min:.3f})"
            ax.set_title(title)
        else:
            for col_idx in range(1, n_cols):
                axes[row_idx, col_idx].axis("off")
    
    fig.suptitle(
        f"Aperiodic Component Topomaps (sub-{subject})",
        fontsize=14,
        fontweight="bold"
    )
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
    log_if_present(
        logger,
        "info",
        f"Saved aperiodic topomap grid ({n_rows} metrics × {n_cols} conditions)"
    )


def _find_rating_column(events_df: pd.DataFrame, config: Any) -> Optional[str]:
    """Find rating column from config or common candidates.
    
    Returns:
        Column name or None if not found
    """
    potential_cols = list(RATING_COLUMN_CANDIDATES)
    config_rating = config.get("event_columns.rating") if config is not None else None
    
    if config_rating:
        if isinstance(config_rating, list):
            for col in reversed(config_rating):
                potential_cols.insert(0, col)
        else:
            potential_cols.insert(0, config_rating)
    
    for col in potential_cols:
        if col in events_df.columns:
            return col
    
    return None


def _find_common_slope_columns(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    config: Any,
) -> List[str]:
    """Find slope columns common across conditions.
    
    Returns:
        List of column names
    """
    slope_cols = [c for c in features_df.columns if c.startswith("aper_slope_")]
    if not slope_cols:
        return []
    
    from eeg_pipeline.utils.analysis.events import extract_comparison_mask
    
    comp = extract_comparison_mask(events_df, config, require_enabled=False)
    if comp is None:
        return [
            col for col in slope_cols
            if pd.to_numeric(features_df[col], errors="coerce").notna().all()
        ]
    
    mask1, mask2, _, _ = comp
    mask1 = np.asarray(mask1, dtype=bool)
    mask2 = np.asarray(mask2, dtype=bool)
    
    common_cols = []
    for col in slope_cols:
        vals = pd.to_numeric(features_df[col], errors="coerce")
        if vals[mask1].notna().all() and vals[mask2].notna().all():
            common_cols.append(col)
    
    return common_cols


def _compute_permutation_pvalue(
    ratings: np.ndarray,
    slopes: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> float:
    """Compute permutation p-value for correlation.
    
    Returns:
        Permutation p-value
    """
    observed_correlation, _ = stats.spearmanr(ratings, slopes)
    observed_abs = abs(observed_correlation)
    
    n_iter = max(MIN_PERMUTATIONS, n_permutations)
    perm_ge_count = 0
    
    for _ in range(n_iter):
        shuffled_ratings = rng.permutation(ratings)
        perm_correlation, _ = stats.spearmanr(shuffled_ratings, slopes)
        if abs(perm_correlation) >= observed_abs:
            perm_ge_count += 1
    
    return (perm_ge_count + 1) / (n_iter + 1)


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

    rating_col = _find_rating_column(events_df, config)
    if rating_col is None:
        log_if_present(
            logger,
            "warning",
            "No rating column found for aperiodic scatter plot"
        )
        return

    if len(features_df) != len(events_df):
        log_if_present(
            logger,
            "warning",
            "Mismatch in features and events length for scatter plot"
        )
        return

    common_slope_cols = _find_common_slope_columns(features_df, events_df, config)
    min_channels = int(
        get_config_value(
            config,
            "statistics.min_channels_for_aperiodic_corr",
            MIN_CHANNELS_FOR_APERIODIC_CORR_DEFAULT
        )
    )
    
    if len(common_slope_cols) < min_channels:
        log_if_present(
            logger,
            "warning",
            f"Insufficient common aperiodic channels across conditions "
            f"({len(common_slope_cols)}<{min_channels}); skipping slope vs pain plot"
        )
        return
    
    mean_slopes = features_df[common_slope_cols].mean(axis=1)
    ratings = pd.to_numeric(events_df[rating_col], errors="coerce")
    valid_mask = mean_slopes.notna() & ratings.notna()
    mean_slopes_valid = mean_slopes[valid_mask]
    ratings_valid = ratings[valid_mask]
    events_aligned = events_df.loc[valid_mask]

    run_col = _find_run_column(events_aligned, config)
    if run_col is None:
        log_if_present(
            logger,
            "warning",
            "No run/block column found; skipping aperiodic slope vs pain "
            "(cannot assume independent trials)."
        )
        return

    grouped = events_aligned.assign(mean_slope=mean_slopes_valid).groupby(run_col)
    slopes_agg = grouped["mean_slope"].mean()
    ratings_agg = grouped[rating_col].mean()
    agg_label = f"run-mean ({run_col})"

    if len(slopes_agg) < MIN_RUNS_FOR_CORRELATION:
        return

    correlation, _ = stats.spearmanr(ratings_agg, slopes_agg)

    n_permutations = int(
        get_config_value(
            config,
            "plotting.plots.aperiodic.n_perm",
            get_config_value(
                config,
                "behavior_analysis.statistics.n_permutations",
                DEFAULT_N_PERMUTATIONS
            )
        )
    )
    random_state = int(
        get_config_value(config, "project.random_state", DEFAULT_RANDOM_STATE)
    )
    rng = np.random.default_rng(random_state)
    
    p_perm_slope = _compute_permutation_pvalue(
        ratings_agg.to_numpy(),
        slopes_agg.to_numpy(),
        n_permutations,
        rng,
    )

    p_perm_for_bh = [p_perm_slope]
    offset_cols = [c for c in features_df.columns if c.startswith("aper_offset_")]
    if offset_cols:
        mean_offset = features_df[offset_cols].mean(axis=1)[valid_mask]
        if len(mean_offset) == len(ratings_valid):
            grouped_off = events_aligned.assign(mean_offset=mean_offset).groupby(run_col)
            off_agg = grouped_off["mean_offset"].mean()
            ratings_off = grouped_off[rating_col].mean()
            
            if len(off_agg) >= MIN_RUNS_FOR_CORRELATION:
                p_perm_offset = _compute_permutation_pvalue(
                    ratings_off.to_numpy(),
                    off_agg.to_numpy(),
                    n_permutations,
                    rng,
                )
                p_perm_for_bh.append(p_perm_offset)
    
    q_vals_perm = fdr_bh(p_perm_for_bh, config=config)
    q_perm = q_vals_perm[0] if q_vals_perm.size > 0 else p_perm_slope

    plot_cfg = get_plot_config(config)
    fig_size = plot_cfg.get_figure_size("standard", plot_type="features")
    fig, ax = plt.subplots(figsize=fig_size)
    
    sns.scatterplot(x=ratings_agg, y=slopes_agg, ax=ax, alpha=SCATTER_ALPHA)
    sns.regplot(
        x=ratings_agg,
        y=slopes_agg,
        ax=ax,
        scatter=False,
        lowess=True,
        line_kws={"color": "red"}
    )
    
    ax.set_xlabel(f"Rating ({rating_col}, {agg_label})")
    ax.set_ylabel("Mean Aperiodic Slope")
    ax.set_title(
        f"Aperiodic Slope vs Rating (sub-{subject})\n"
        f"Spearman r={correlation:.2f}, perm q={q_perm:.3f}"
    )
    
    output_name = f"sub-{subject}_aperiodic_slope_vs_rating"
    save_fig(
        fig,
        save_dir / output_name,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches
    )
    plt.close(fig)
    log_if_present(logger, "info", "Saved aperiodic slope vs rating scatter")


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
    
    metrics = APERIODIC_METRICS
    metric_colors = {"slope": SLOPE_COLOR, "offset": OFFSET_COLOR}
    
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
            
            from eeg_pipeline.plotting.features.utils import compute_or_load_column_stats
            
            segment_colors = {"v1": CONDITION_COLOR_V1, "v2": CONDITION_COLOR_V2}
            n_metrics = len(metrics)
            n_trials = len(features_df)
            
            for roi_name in roi_names:
                # Collect cell data first
                cell_data = {}
                for col_idx, metric in enumerate(metrics):
                    cols = get_aperiodic_columns(seg_name, metric, roi_name)
                    
                    if not cols:
                        cell_data[col_idx] = None
                        continue
                    
                    val_series = features_df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                    v1 = val_series[m1].dropna().values
                    v2 = val_series[m2].dropna().values
                    
                    cell_data[col_idx] = {"v1": v1, "v2": v2}
                
                # Compute or load column comparison stats
                qvalues, n_significant, use_precomputed = compute_or_load_column_stats(
                    stats_dir=stats_dir,
                    feature_type="aperiodic",
                    feature_keys=metrics,
                    cell_data=cell_data,
                    config=config,
                    logger=logger,
                )
                
                fig_width_per_metric = 4
                fig_height = 5
                fig, axes = plt.subplots(
                    1, n_metrics,
                    figsize=(fig_width_per_metric * n_metrics, fig_height),
                    squeeze=False
                )
                
                for col_idx, metric in enumerate(metrics):
                    ax = axes.flatten()[col_idx]
                    data = cell_data.get(col_idx)
                    
                    if data is None or len(data.get("v1", [])) == 0 or len(data.get("v2", [])) == 0:
                        ax.text(
                            0.5, 0.5, "No data", ha="center", va="center",
                            transform=ax.transAxes, fontsize=plot_cfg.font.title,
                            color="gray"
                        )
                        ax.set_xticks([])
                        continue
                    
                    v1, v2 = data["v1"], data["v2"]
                    
                    bp = ax.boxplot([v1, v2], positions=[0, 1], widths=0.4, patch_artist=True)
                    bp["boxes"][0].set_facecolor(segment_colors["v1"])
                    bp["boxes"][0].set_alpha(BOX_ALPHA)
                    bp["boxes"][1].set_facecolor(segment_colors["v2"])
                    bp["boxes"][1].set_alpha(BOX_ALPHA)
                    
                    scatter_jitter = 0.08
                    ax.scatter(
                        np.random.uniform(-scatter_jitter, scatter_jitter, len(v1)),
                        v1,
                        c=segment_colors["v1"],
                        alpha=0.3,
                        s=SCATTER_SIZE
                    )
                    ax.scatter(
                        1 + np.random.uniform(-scatter_jitter, scatter_jitter, len(v2)),
                        v2,
                        c=segment_colors["v2"],
                        alpha=0.3,
                        s=SCATTER_SIZE
                    )
                    
                    all_vals = np.concatenate([v1, v2])
                    ymin, ymax = np.nanmin(all_vals), np.nanmax(all_vals)
                    yrange = ymax - ymin if ymax > ymin else 0.1
                    ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.3 * yrange)
                    
                    if col_idx in qvalues:
                        _, q, d, sig = qvalues[col_idx]
                        sig_marker = "†" if sig else ""
                        sig_color = SIGNIFICANT_COLOR if sig else NON_SIGNIFICANT_COLOR
                        annotation_y = ymax + 0.05 * yrange
                        ax.annotate(
                            f"q={q:.3f}{sig_marker}\nd={d:.2f}",
                            xy=(0.5, annotation_y),
                            ha="center",
                            fontsize=plot_cfg.font.medium,
                            color=sig_color,
                            fontweight="bold" if sig else "normal"
                        )
                    
                    ax.set_xticks([0, 1])
                    ax.set_xticklabels([label1, label2], fontsize=9)
                    metric_color = metric_colors.get(metric, "gray")
                    ax.set_title(
                        metric.capitalize(), fontweight="bold", color=metric_color
                    )
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                
                n_tests = len(qvalues)
                roi_display = (
                    roi_name.replace("_", " ").title()
                    if roi_name != "all"
                    else "All Channels"
                )
                
                stats_source = (
                    "pre-computed" if use_precomputed else "Mann-Whitney U"
                )
                title = (
                    f"Aperiodic: {label1} vs {label2} (Column Comparison)\n"
                    f"Subject: {subject} | ROI: {roi_display} | "
                    f"N: {n_trials} trials | {stats_source} | "
                    f"FDR: {n_significant}/{n_tests} significant (†=q<0.05)"
                )
                fig.suptitle(
                    title,
                    fontsize=plot_cfg.font.suptitle,
                    fontweight="bold",
                    y=1.02
                )
                
                plt.tight_layout()
                
                roi_safe = (
                    roi_name.replace(" ", "_").lower()
                    if roi_name != "all"
                    else ""
                )
                suffix = f"_roi-{roi_safe}" if roi_safe else ""
                filename = f"sub-{subject}_aperiodic_by_condition{suffix}_column"
                
                save_fig(
                    fig,
                    save_dir / filename,
                    formats=plot_cfg.formats,
                    dpi=plot_cfg.dpi,
                    bbox_inches=plot_cfg.bbox_inches,
                    pad_inches=plot_cfg.pad_inches
                )
                plt.close(fig)
            
            log_if_present(
                logger,
                "info",
                f"Saved aperiodic column comparison plots for {len(roi_names)} ROIs"
            )
