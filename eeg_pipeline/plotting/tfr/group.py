"""
Group-level TFR plotting functions.

Functions for creating group-level time-frequency representations (TFR) plots,
including group contrasts, aggregations, and multi-subject visualizations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

from ...utils.io.general import (
    robust_sym_vlim,
    extract_eeg_picks,
    logratio_to_pct,
    get_pain_column_from_config,
    get_temperature_column_from_config,
    plot_topomap_on_ax,
    unwrap_figure,
    sanitize_label,
    get_viz_params,
    ensure_aligned_lengths,
)
from ...utils.analysis.tfr import (
    apply_baseline_and_crop,
    create_tfr_subset,
    get_bands_for_tfr,
    average_tfr_band,
    extract_trial_band_power,
    clip_time_range,
    create_time_windows_fixed_size,
    create_time_windows_fixed_count,
    avg_by_mask_to_avg_tfr,
    align_avg_tfrs,
    align_paired_avg_tfrs,
    get_rois,
    extract_roi_from_tfr,
    extract_roi_contrast_data,
)
from ...utils.analysis.stats import (
    cluster_test_two_sample_arrays as _cluster_test_two_sample_arrays,
    fdr_bh_mask as _fdr_bh_mask,
    fdr_bh_values as _fdr_bh_values,
    format_cluster_ann,
)
from ...utils.data.loading import (
    compute_aligned_data_length,
    extract_pain_vector_array,
    extract_temperature_series,
    create_temperature_masks,
)
from ..config import get_plot_config
from ..core.utils import get_font_sizes, log
from ..core.statistics import get_strict_mode, compute_cluster_significance, build_statistical_title
from ..core.colorbars import add_normalized_colorbar, add_diff_colorbar, create_difference_colorbar
from ..core.topomaps import build_topomap_percentage_label, build_topomap_diff_label
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors
from ..core.annotations import (
    apply_fdr_correction_to_roi_pvalues,
    build_significance_info,
    render_roi_annotations,
)
from ..core.annotations import get_sig_marker_text
from .contrasts import (
    _get_baseline_window,
    _create_pain_masks_from_vector,
    _create_pain_masks_from_events,
    _align_and_trim_masks,
    _compute_band_diff_data,
    _compute_cluster_significance_from_combined,
    _plot_diff_topomap_with_label,
    _plot_topomap_with_percentage_label,
)
from .channels import _save_fig, _plot_single_tfr_figure
from .scalpmean import _plot_scalpmean_tfr
from .topomaps import _add_roi_annotations, _plot_temporal_topomaps_for_bands


###################################################################
# Helper Functions
###################################################################


def _collect_pain_nonpain_avg_tfrs(
    powers: List["mne.time_frequency.EpochsTFR"],
    events_by_subj: List[Optional[pd.DataFrame]],
    config,
    baseline: Tuple[Optional[float], Optional[float]],
    logger: Optional[logging.Logger] = None,
) -> Tuple[List["mne.time_frequency.AverageTFR"], List["mne.time_frequency.AverageTFR"]]:
    """Collect pain and non-pain average TFRs from multiple subjects.
    
    Args:
        powers: List of EpochsTFR objects (one per subject)
        events_by_subj: List of events DataFrames (one per subject)
        config: Configuration object
        baseline: Baseline window tuple
        logger: Optional logger instance
        
    Returns:
        Tuple of (avg_pain_list, avg_non_list)
    """
    avg_pain = []
    avg_non = []
    
    for power, ev in zip(powers, events_by_subj):
        if ev is None:
            continue
        pain_col = get_pain_column_from_config(config, ev)
        if pain_col is None:
            continue
        pain_mask, non_mask = _create_pain_masks_from_events(ev, pain_col)
        if pain_mask is None or pain_mask.sum() == 0 or non_mask.sum() == 0:
            continue
        a_p = avg_by_mask_to_avg_tfr(power, pain_mask, baseline=baseline, logger=logger)
        a_n = avg_by_mask_to_avg_tfr(power, non_mask, baseline=baseline, logger=logger)
        if a_p is not None and a_n is not None:
            avg_pain.append(a_p)
            avg_non.append(a_n)
    
    return avg_pain, avg_non


def _create_group_roi_tfr(
    template_tfr: "mne.time_frequency.AverageTFR",
    data: np.ndarray,
    info: mne.Info,
    nave: int,
    comment: str,
) -> "mne.time_frequency.AverageTFR":
    """Create a group ROI TFR object from template and data.
    
    Args:
        template_tfr: Template AverageTFR object
        data: Data array to use
        info: MNE Info object
        nave: Number of averages
        comment: Comment string
        
    Returns:
        New AverageTFR object
    """
    grp = template_tfr.copy()
    grp.data = data
    grp.info = info
    grp.nave = nave
    grp.comment = comment
    return grp


def _create_group_scalpmean_tfr(
    template_tfr: "mne.time_frequency.AverageTFR",
    data: np.ndarray,
    sfreq: float,
    nave: int,
    comment: str,
) -> "mne.time_frequency.AverageTFR":
    """Create a group scalp-mean TFR object.
    
    Args:
        template_tfr: Template AverageTFR object
        data: Data array to use
        sfreq: Sampling frequency
        nave: Number of averages
        comment: Comment string
        
    Returns:
        New AverageTFR object with scalp-mean channel
    """
    tfr = template_tfr.copy()
    tfr.data = data
    tfr.info = mne.create_info(["AllEEG"], sfreq=sfreq, ch_types='eeg')
    tfr.nave = nave
    tfr.comment = comment
    return tfr


def _collect_roi_contrast_data(
    powers: List["mne.time_frequency.EpochsTFR"],
    events_by_subj: List[Optional[pd.DataFrame]],
    roi: str,
    roi_map: Optional[Dict[str, List[str]]],
    config,
    baseline: Tuple[Optional[float], Optional[float]],
    logger: Optional[logging.Logger] = None,
) -> Tuple[List["mne.time_frequency.AverageTFR"], List["mne.time_frequency.AverageTFR"]]:
    """Collect ROI contrast data for pain/non-pain conditions.
    
    Args:
        powers: List of EpochsTFR objects (one per subject)
        events_by_subj: List of events DataFrames (one per subject)
        roi: ROI name
        roi_map: Optional ROI map dictionary
        config: Configuration object
        baseline: Baseline window tuple
        logger: Optional logger instance
        
    Returns:
        Tuple of (roi_pain_list, roi_non_list)
    """
    roi_p_list = []
    roi_n_list = []
    
    for power, ev in zip(powers, events_by_subj):
        contrast_result = extract_roi_contrast_data(power, ev, roi, roi_map, config, baseline=baseline, logger=logger)
        r_p, r_n = (None, None) if contrast_result is None else contrast_result
        if r_p is not None and r_n is not None:
            roi_p_list.append(r_p)
            roi_n_list.append(r_n)
    
    if len(roi_p_list) < 1 or len(roi_n_list) < 1:
        if roi_map is not None:
            roi_p_list = []
            roi_n_list = []
            for power, ev in zip(powers, events_by_subj):
                contrast_result = extract_roi_contrast_data(power, ev, roi, None, config, baseline=baseline, logger=logger)
                r_p, r_n = (None, None) if contrast_result is None else contrast_result
                if r_p is not None and r_n is not None:
                    roi_p_list.append(r_p)
                    roi_n_list.append(r_n)
    
    return roi_p_list, roi_n_list


def _combine_multiple_tfr_groups(
    tfr_lists_dict: Dict[str, List["mne.time_frequency.AverageTFR"]],
    min_count: int,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Optional["mne.time_frequency.AverageTFR"]]:
    """Combine multiple TFR groups into combined averages.
    
    Args:
        tfr_lists_dict: Dictionary mapping condition names to lists of AverageTFR objects
        min_count: Minimum number of TFRs required for combination
        logger: Optional logger instance
        
    Returns:
        Dictionary mapping condition names to combined AverageTFR objects (or None)
    """
    combined = {}
    for key, tfr_list in tfr_lists_dict.items():
        if len(tfr_list) >= min_count:
            combined[key] = _combine_avg_tfrs_group(tfr_list, logger)
        else:
            combined[key] = None
    return combined


def _combine_avg_tfrs_group(
    avg_tfr_list: List["mne.time_frequency.AverageTFR"],
    logger: Optional[logging.Logger] = None,
) -> Optional["mne.time_frequency.AverageTFR"]:
    """Combine multiple AverageTFR objects into a grand average.
    
    Args:
        avg_tfr_list: List of AverageTFR objects to combine
        logger: Optional logger instance
        
    Returns:
        Combined AverageTFR object, or None if combination fails
    """
    if not avg_tfr_list:
        return None
    avg_tfr_list = [t for t in avg_tfr_list if t is not None]
    if not avg_tfr_list:
        return None
    
    base = avg_tfr_list[0]
    base_times = np.asarray(base.times)
    base_freqs = np.asarray(base.freqs)
    base_chs = list(base.info["ch_names"])
    
    valid_tfrs = [base]
    for tfr in avg_tfr_list[1:]:
        if np.allclose(tfr.times, base_times) and np.allclose(tfr.freqs, base_freqs):
            valid_tfrs.append(tfr)
        elif logger:
            log("Skipping TFR: times/freqs mismatch for group alignment", logger, "warning")
    
    if len(valid_tfrs) == 0:
        return None
    
    ch_sets = [set(t.info["ch_names"]) for t in valid_tfrs]
    common_chs = list(sorted(set.intersection(*ch_sets))) if ch_sets else []
    if len(common_chs) == 0:
        if logger:
            log("No common channels across subjects; cannot combine TFRs", logger, "warning")
        return None
    
    data_list = []
    for tfr in valid_tfrs:
        ch_indices = [tfr.info["ch_names"].index(ch) for ch in common_chs]
        data_list.append(np.asarray(tfr.data)[ch_indices, :, :])
    
    combined_data = np.mean(np.stack(data_list, axis=0), axis=0)
    
    pick_inds = [base_chs.index(ch) for ch in common_chs]
    info_common = mne.pick_info(base.info, pick_inds)
    
    combined_tfr = mne.time_frequency.AverageTFR(
        info=info_common,
        data=combined_data,
        times=base_times,
        freqs=base_freqs,
        nave=len(valid_tfrs),
        method=None,
        comment=base.comment if hasattr(base, 'comment') else 'grand_average'
    )
    return combined_tfr


def _combine_multiple_epochs_tfr_groups(
    tfr_lists_dict: Dict[str, List["mne.time_frequency.EpochsTFR"]],
    min_count: int,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Optional["mne.time_frequency.EpochsTFR"]]:
    """Combine multiple EpochsTFR groups into concatenated groups.
    
    Args:
        tfr_lists_dict: Dictionary mapping condition names to lists of EpochsTFR objects
        min_count: Minimum number of TFRs required for combination
        logger: Optional logger instance
        
    Returns:
        Dictionary mapping condition names to combined EpochsTFR objects (or None)
    """
    combined = {}
    for key, tfr_list in tfr_lists_dict.items():
        if len(tfr_list) >= min_count:
            combined[key] = _concatenate_epochs_tfr_group(tfr_list, logger)
        else:
            combined[key] = None
    return combined


def _concatenate_epochs_tfr_group(
    tfr_list: List["mne.time_frequency.EpochsTFR"],
    logger: Optional[logging.Logger] = None,
) -> Optional["mne.time_frequency.EpochsTFR"]:
    """Concatenate multiple EpochsTFR objects into a single group-level object.
    
    Args:
        tfr_list: List of EpochsTFR objects to concatenate
        logger: Optional logger instance
        
    Returns:
        Concatenated EpochsTFR object, or None if concatenation fails
    """
    if not tfr_list:
        return None
    tfr_list = [t for t in tfr_list if t is not None]
    if not tfr_list:
        return None
    
    base = tfr_list[0]
    base_times = np.asarray(base.times)
    base_freqs = np.asarray(base.freqs)
    base_chs = list(base.info["ch_names"])
    
    valid_tfrs = [base]
    for tfr in tfr_list[1:]:
        if np.allclose(tfr.times, base_times) and np.allclose(tfr.freqs, base_freqs):
            valid_tfrs.append(tfr)
        elif logger:
            log("Skipping TFR: times/freqs mismatch for group concatenation", logger, "warning")
    
    if len(valid_tfrs) == 0:
        return None
    
    ch_sets = [set(t.info["ch_names"]) for t in valid_tfrs]
    common_chs = list(sorted(set.intersection(*ch_sets))) if ch_sets else []
    if len(common_chs) == 0:
        if logger:
            log("No common channels across subjects; cannot concatenate TFRs", logger, "warning")
        return None
    
    data_list = []
    events_list = []
    for tfr in valid_tfrs:
        ch_indices = [tfr.info["ch_names"].index(ch) for ch in common_chs]
        data_subj = np.asarray(tfr.data)[:, ch_indices, :, :]
        data_list.append(data_subj)
        if hasattr(tfr, 'events') and tfr.events is not None:
            events_list.append(tfr.events)
    
    combined_data = np.concatenate(data_list, axis=0)
    combined_events = np.concatenate(events_list, axis=0) if events_list and all(e is not None for e in events_list) else base.events
    
    pick_inds = [base_chs.index(ch) for ch in common_chs]
    info_common = mne.pick_info(base.info, pick_inds)
    
    combined_tfr = mne.time_frequency.EpochsTFR(
        info_common,
        combined_data,
        base_times,
        base_freqs,
        events=combined_events,
        event_id=base.event_id if hasattr(base, 'event_id') else None,
        metadata=base.metadata if hasattr(base, 'metadata') and base.metadata is not None else None,
        method=None,
        verbose=False
    )
    return combined_tfr


###################################################################
# Group Plotting Functions
###################################################################


def group_contrast_maxmin_temperature(
    powers: List["mne.time_frequency.EpochsTFR"],
    events_by_subj: List[Optional[pd.DataFrame]],
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    plateau_window: Tuple[float, float] = (3.0, 10.5),
    logger: Optional[logging.Logger] = None,
) -> None:
    """Plot group-level topomaps comparing max vs min temperature conditions.
    
    Creates topomap grids showing group-level differences between maximum and
    minimum temperature conditions across frequency bands.
    
    Args:
        powers: List of EpochsTFR objects (one per subject)
        events_by_subj: List of events DataFrames (one per subject)
        out_dir: Output directory path
        config: Configuration object
        baseline: Optional baseline window tuple (defaults to config)
        plateau_window: Plateau window tuple for statistics
        logger: Optional logger instance
    """
    baseline = _get_baseline_window(config, baseline)
    if not powers:
        return
    temps = []
    for ev in events_by_subj:
        if ev is None:
            continue
        tcol = get_temperature_column_from_config(config, ev)
        if tcol is None:
            continue
        vals = pd.to_numeric(ev[tcol], errors="coerce").round(1)
        temps.extend(list(vals.dropna().unique()))
    temps = sorted(set(map(float, temps)))
    if len(temps) < 2:
        log("Group max/min: fewer than 2 temperature levels; skipping", logger)
        return
    t_min, t_max = float(min(temps)), float(max(temps))

    avg_min: List["mne.time_frequency.AverageTFR"] = []
    avg_max: List["mne.time_frequency.AverageTFR"] = []
    for power, ev in zip(powers, events_by_subj):
        if ev is None:
            continue
        tcol = get_temperature_column_from_config(config, ev)
        if tcol is None:
            continue
        vals = pd.to_numeric(ev[tcol], errors="coerce").round(1)
        mask_min = np.asarray(vals == round(t_min, 1), dtype=bool)
        mask_max = np.asarray(vals == round(t_max, 1), dtype=bool)
        if mask_min.sum() == 0 or mask_max.sum() == 0:
            continue
        a_min = avg_by_mask_to_avg_tfr(power, mask_min, baseline=baseline, logger=logger)
        a_max = avg_by_mask_to_avg_tfr(power, mask_max, baseline=baseline, logger=logger)
        if a_min is not None and a_max is not None:
            avg_min.append(a_min)
            avg_max.append(a_max)

    info_common, data_min, data_max = align_paired_avg_tfrs(avg_min, avg_max, logger=logger)
    if info_common is None or data_min is None or data_max is None:
        log("Group max/min: could not align paired min/max TFRs; skipping", logger)
        return

    mean_min = data_min.mean(axis=0)
    mean_max = data_max.mean(axis=0)
    freqs = np.asarray(avg_min[0].freqs if avg_min else avg_max[0].freqs)
    times = np.asarray(avg_min[0].times if avg_min else avg_max[0].times)
    fmax_available = float(freqs.max())
    bands = get_bands_for_tfr(max_freq_available=fmax_available, config=config)
    tmin_req, tmax_req = plateau_window
    tmin = float(max(times.min(), tmin_req))
    tmax = float(min(times.max(), tmax_req))

    n_rows, n_cols = 3, len(bands)
    row_labels = [f"Max {t_max:.1f}°C (n={data_max.shape[0]})", f"Min {t_min:.1f}°C (n={data_min.shape[0]})", "Max - Min"]
    plot_cfg_large2 = get_plot_config(config)
    fig_size_per_col_large2 = plot_cfg_large2.get_figure_size("tfr_per_col_large", plot_type="tfr")[0]
    fig_size_per_row_large = plot_cfg_large2.get_figure_size("tfr_per_row_large", plot_type="tfr")[1]
    tfr_config_large2 = plot_cfg_large2.plot_type_configs.get("tfr", {})
    topomap_config_large2 = tfr_config_large2.get("topomap", {})
    topo_wspace_large2 = topomap_config_large2.get("wspace", 1.2)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_size_per_col_large2 * n_cols, fig_size_per_row_large * n_rows),
        squeeze=False,
        gridspec_kw={"wspace": topo_wspace_large2, "hspace": 0.35}
    )

    for c, (band, (fmin, fmax)) in enumerate(bands.items()):
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            for r in range(n_rows):
                axes[r, c].axis('off')
            continue
        fmask = (freqs >= fmin) & (freqs <= fmax_eff)
        tmask = (times >= tmin) & (times < tmax)
        if fmask.sum() == 0 or tmask.sum() == 0:
            for r in range(n_rows):
                axes[r, c].axis('off')
            continue
        v_max = mean_max[:, fmask, :][:, :, tmask].mean(axis=(1, 2))
        v_min = mean_min[:, fmask, :][:, :, tmask].mean(axis=(1, 2))
        v_diff = v_max - v_min
        vabs_pn = robust_sym_vlim([v_max, v_min])
        vabs_diff = robust_sym_vlim(v_diff)

        _plot_topomap_with_percentage_label(axes[0, c], v_max, info_common, -vabs_pn, +vabs_pn, config)
        _plot_topomap_with_percentage_label(axes[1, c], v_min, info_common, -vabs_pn, +vabs_pn, config)
        sig_mask = cluster_p_min = cluster_k = cluster_mass = None
        fdr_txt = ""
        p_ch_used = None
        is_cluster_used = False
        viz_params = get_viz_params(config)
        if viz_params["diff_annotation_enabled"]:
            subj_max = data_max[:, :, fmask, :][:, :, :, tmask].mean(axis=(2, 3))
            subj_min = data_min[:, :, fmask, :][:, :, :, tmask].mean(axis=(2, 3))
            sig_mask, cluster_p_min, cluster_k, cluster_mass = _cluster_test_two_sample_arrays(
                subj_max, subj_min, info_common, alpha=config.get("statistics.sig_alpha", 0.05), paired=True, n_permutations=config.get("statistics.cluster_n_perm", 100), config=config
            )
            if sig_mask is not None and cluster_p_min is not None:
                is_cluster_used = True
            elif sig_mask is None:
                res = ttest_rel(subj_max, subj_min, axis=0, nan_policy="omit")
                p_ch_used = np.asarray(res.pvalue)
                sig_mask = _fdr_bh_mask(p_ch_used, alpha=config.get("statistics.sig_alpha", 0.05))
                rej, q = _fdr_bh_values(p_ch_used, alpha=config.get("statistics.sig_alpha", 0.05))
                k_rej = int(np.nansum(rej)) if rej is not None else 0
                q_min = float(np.nanmin(q)) if q is not None and np.isfinite(q).any() else None
                fdr_txt = format_cluster_ann(q_min, k_rej if k_rej > 0 else None, config=config)

        ax = axes[2, c]
        plot_topomap_on_ax(
            ax,
            v_diff,
            info_common,
            mask=(sig_mask if viz_params["diff_annotation_enabled"] else None),
            mask_params=viz_params["sig_mask_params"],
            vmin=-vabs_diff,
            vmax=+vabs_diff,
        )
        _add_roi_annotations(
            ax, v_diff, info_common, config=config,
            sig_mask=(sig_mask if viz_params["diff_annotation_enabled"] else None),
            p_ch=p_ch_used,
            cluster_p_min=cluster_p_min, cluster_k=cluster_k, cluster_mass=cluster_mass,
            is_cluster=is_cluster_used
        )
        mu_d = float(np.nanmean(v_diff))
        pct_d = logratio_to_pct(mu_d)
        cl_txt = (format_cluster_ann(cluster_p_min, cluster_k, cluster_mass, config=config) or fdr_txt) if viz_params["diff_annotation_enabled"] else ""
        label = f"Δ%={pct_d:+.1f}%" + (f" | {cl_txt}" if cl_txt else "")
        ax.text(0.5, 1.02, label, transform=ax.transAxes, ha="center", va="top", fontsize=9)
        axes[0, c].set_title(f"{band} ({fmin:.0f}-{fmax_eff:.0f} Hz)", fontsize=9, pad=4, y=1.04)

    add_normalized_colorbar(fig, [axes[0, :].ravel().tolist(), axes[1, :].ravel().tolist()], -vabs_pn, +vabs_pn, viz_params["topo_cmap"], config)
    add_diff_colorbar(fig, axes[2, :].ravel().tolist(), vabs_diff, viz_params["topo_cmap"], config)

    axes[0, 0].set_ylabel(row_labels[0], fontsize=10)
    axes[1, 0].set_ylabel(row_labels[1], fontsize=10)
    axes[2, 0].set_ylabel(row_labels[2], fontsize=10)
    fig.suptitle(
        f"Group Topomaps: Max vs Min temperature (baseline logratio; t=[{tmin:.1f}, {tmax:.1f}] s)",
        fontsize=12,
    )
    _save_fig(fig, out_dir, "group_topomap_grid_bands_maxmin_temp_diff_baseline_logratio.png", config=config, logger=logger)


def group_rois_all_trials(
    powers: List["mne.time_frequency.EpochsTFR"],
    out_dir: Path,
    config,
    baseline: Tuple[Optional[float], Optional[float]],
    logger: Optional[logging.Logger] = None,
    roi_map: Optional[Dict[str, list[str]]] = None,
) -> None:
    """Plot group-level ROI TFRs for all trials.
    
    Creates group-averaged TFR plots for each ROI across all subjects.
    
    Args:
        powers: List of EpochsTFR objects (one per subject)
        out_dir: Output directory path
        config: Configuration object
        baseline: Baseline window tuple
        logger: Optional logger instance
        roi_map: Optional ROI map dictionary
    """
    if not powers:
        return

    avg_list = []
    for p in powers:
        t = p.copy()
        t_avg = t.average()
        baseline_used = apply_baseline_and_crop(t_avg, baseline=baseline, mode="logratio", logger=logger)
        avg_list.append(t_avg)

    if not avg_list:
        return

    if roi_map is not None:
        rois = list(roi_map.keys())
    else:
        if config is None:
            raise ValueError("Either roi_map or config is required for group_rois_all_trials")
        roi_defs = get_rois(config)
        rois = list(roi_defs.keys())

    for roi in rois:
        per_subj: List["mne.time_frequency.AverageTFR"] = []
        for a in avg_list:
            ra = extract_roi_from_tfr(a, roi, roi_map, config)
            if ra is not None:
                per_subj.append(ra)

        if len(per_subj) < 1:
            log(f"Group ROI all-trials: no subjects contributed to ROI '{roi}'", logger)
            continue

        info_c, data_c = align_avg_tfrs(per_subj, logger=logger)
        if info_c is None or data_c is None:
            continue

        mean_roi = data_c.mean(axis=0)
        grp = per_subj[0].copy()
        grp.data = mean_roi
        grp.info = info_c
        grp.nave = int(data_c.shape[0])
        grp.comment = f"Group ROI:{roi}"
        ch = grp.info['ch_names'][0]
        title_fontsize = 12
        fig = unwrap_figure(grp.plot(picks=ch, show=False))
        fig.suptitle(f"Group ROI: {roi} — all trials (baseline logratio, n={data_c.shape[0]})", fontsize=title_fontsize)
        _save_fig(fig, out_dir, f"group_tfr_ROI-{sanitize_label(roi)}_all_trials_baseline_logratio.png", config=config, logger=logger, baseline_used=baseline)


def group_contrast_pain_nonpain_rois(
    powers: List["mne.time_frequency.EpochsTFR"],
    events_by_subj: List[Optional[pd.DataFrame]],
    out_dir: Path,
    config,
    baseline: Tuple[Optional[float], Optional[float]],
    logger: Optional[logging.Logger] = None,
    roi_map: Optional[Dict[str, list[str]]] = None,
) -> None:
    """Plot group-level ROI contrasts for pain vs non-pain conditions.
    
    Creates group-averaged TFR plots showing pain, non-pain, and difference
    for each ROI across all subjects.
    
    Args:
        powers: List of EpochsTFR objects (one per subject)
        events_by_subj: List of events DataFrames (one per subject)
        out_dir: Output directory path
        config: Configuration object
        baseline: Baseline window tuple
        logger: Optional logger instance
        roi_map: Optional ROI map dictionary
    """
    if not powers:
        return

    if roi_map is not None:
        rois = list(roi_map.keys())
    else:
        if config is None:
            raise ValueError("Either roi_map or config is required for group_contrast_pain_nonpain_rois")
        roi_defs = get_rois(config)
        rois = list(roi_defs.keys())
    for roi in rois:
        roi_p_list, roi_n_list = _collect_roi_contrast_data(powers, events_by_subj, roi, roi_map, config, baseline=baseline, logger=logger)
        
        if len(roi_p_list) < 1 or len(roi_n_list) < 1:
            log(f"Group ROI pain/non: no subjects contributed to ROI '{roi}'", logger)
            continue

        info_p, data_p = align_avg_tfrs(roi_p_list, logger=logger)
        info_n, data_n = align_avg_tfrs(roi_n_list, logger=logger)
        if info_p is None or info_n is None or data_p is None or data_n is None:
            continue

        mean_p = data_p.mean(axis=0)
        mean_n = data_n.mean(axis=0)

        grp_p = _create_group_roi_tfr(roi_p_list[0], mean_p, info_p, int(data_p.shape[0]), f"Group ROI:{roi} Pain")
        grp_n = _create_group_roi_tfr(roi_n_list[0], mean_n, info_n, int(data_n.shape[0]), f"Group ROI:{roi} Non")
        diff = mean_p - mean_n
        grp_d = _create_group_roi_tfr(roi_p_list[0], diff, info_p, int(min(data_p.shape[0], data_n.shape[0])), f"Group ROI:{roi} Diff")
        
        ch = grp_p.info['ch_names'][0]

        _plot_single_tfr_figure(
            grp_p, ch, None, f"Group ROI: {roi} — Pain (baseline logratio, n={data_p.shape[0]})",
            f"group_tfr_ROI-{sanitize_label(roi)}_pain_baseline_logratio.png", out_dir, config, logger, baseline
        )
        _plot_single_tfr_figure(
            grp_n, ch, None, f"Group ROI: {roi} — Non-pain (baseline logratio, n={data_n.shape[0]})",
            f"group_tfr_ROI-{sanitize_label(roi)}_nonpain_baseline_logratio.png", out_dir, config, logger, baseline
        )
        n_diff = min(data_p.shape[0], data_n.shape[0])
        _plot_single_tfr_figure(
            grp_d, ch, None, f"Group ROI: {roi} — Pain minus Non (baseline logratio, n={n_diff})",
            f"group_tfr_ROI-{sanitize_label(roi)}_pain_minus_non_baseline_logratio.png", out_dir, config, logger, baseline
        )


def group_contrast_pain_nonpain_scalpmean(
    powers: List["mne.time_frequency.EpochsTFR"],
    events_by_subj: List[Optional[pd.DataFrame]],
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Plot group-level scalp-mean TFR contrasts for pain vs non-pain conditions.
    
    Creates group-averaged scalp-mean TFR plots showing pain, non-pain, and
    difference conditions across all subjects.
    
    Args:
        powers: List of EpochsTFR objects (one per subject)
        events_by_subj: List of events DataFrames (one per subject)
        out_dir: Output directory path
        config: Configuration object
        baseline: Optional baseline window tuple (defaults to config)
        logger: Optional logger instance
    """
    baseline = _get_baseline_window(config, baseline)
    if not powers:
        return

    avg_pain, avg_non = _collect_pain_nonpain_avg_tfrs(powers, events_by_subj, config, baseline, logger)
    if len(avg_pain) < 1 or len(avg_non) < 1:
        return

    info_p, data_p, data_n = align_paired_avg_tfrs(avg_pain, avg_non, logger=logger)
    if info_p is None or data_p is None or data_n is None:
        return

    n_subj = int(min(data_p.shape[0], data_n.shape[0]))
    mean_p = data_p.mean(axis=0)
    mean_n = data_n.mean(axis=0)

    data_p_sm = np.asarray(mean_p).mean(axis=0, keepdims=True)
    data_n_sm = np.asarray(mean_n).mean(axis=0, keepdims=True)
    diff_sm = data_p_sm - data_n_sm

    tmpl = avg_pain[0].copy()
    sfreq = tmpl.info['sfreq']

    grp_p = _create_group_scalpmean_tfr(tmpl, data_p_sm, sfreq, n_subj, "Group AllEEG Pain")
    grp_n = _create_group_scalpmean_tfr(tmpl, data_n_sm, sfreq, n_subj, "Group AllEEG Non")
    grp_d = _create_group_scalpmean_tfr(tmpl, diff_sm, sfreq, n_subj, "Group AllEEG Diff")

    _plot_scalpmean_tfr(grp_p, f"Group TFR: All EEG — Pain (baseline logratio, n={n_subj})", "group_tfr_AllEEG_pain_baseline_logratio.png", None, out_dir, config, logger, baseline, None, None)
    _plot_scalpmean_tfr(grp_n, f"Group TFR: All EEG — Non-pain (baseline logratio, n={n_subj})", "group_tfr_AllEEG_nonpain_baseline_logratio.png", None, out_dir, config, logger, baseline, None, None)


###################################################################
# Group Plotting Functions - Complex Multi-Band and Temporal Topomaps
###################################################################


def group_plot_bands_pain_temp_contrasts(
    powers: List["mne.time_frequency.EpochsTFR"],
    events_list: List[Optional[pd.DataFrame]],
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    plateau_window: Tuple[float, float] = (3.0, 10.5),
    logger: Optional[logging.Logger] = None,
) -> None:
    baseline = _get_baseline_window(config, baseline)
    
    if len(powers) < 2:
        log("Group contrast requires at least 2 subjects; skipping.", logger, "warning")
        return
    
    pain_col = get_pain_column_from_config(config)
    temp_col = get_temperature_column_from_config(config)
    
    if pain_col is None or temp_col is None:
        log("Group contrast: missing pain or temperature column; skipping.", logger, "warning")
        return
    
    tfr_pain_avg_list = []
    tfr_non_avg_list = []
    tfr_max_avg_list = []
    tfr_min_avg_list = []
    tfr_pain_epochs_list = []
    tfr_non_epochs_list = []
    tfr_max_epochs_list = []
    tfr_min_epochs_list = []
    
    for tfr, events_df in zip(powers, events_list):
        if events_df is None:
            continue
        
        n = compute_aligned_data_length(tfr, events_df)
        
        pain_vec = extract_pain_vector_array(tfr, events_df, pain_col, n)
        if pain_vec is None:
            continue
        
        pain_mask, non_mask = _create_pain_masks_from_vector(pain_vec)
        if pain_mask is None:
            continue
        
        temp_series = extract_temperature_series(tfr, events_df, temp_col, n)
        if temp_series is None:
            continue
        
        temp_result = create_temperature_masks(temp_series)
        if temp_result[0] is None:
            continue
        
        t_min, t_max, mask_min, mask_max = temp_result
        
        if pain_mask.sum() == 0 or non_mask.sum() == 0 or mask_min.sum() == 0 or mask_max.sum() == 0:
            continue
        
        tfr_sub = create_tfr_subset(tfr, n)
        aligned = _align_and_trim_masks(
            tfr_sub,
            {
                "Pain contrast": (pain_mask, non_mask),
                "Temperature contrast": (mask_min, mask_max)
            },
            config, logger
        )
        if aligned is None:
            continue
        
        pain_mask, non_mask = aligned["Pain contrast"]
        mask_min, mask_max = aligned["Temperature contrast"]
        
        tfr_sub_stats = tfr_sub.copy()
        baseline_used = apply_baseline_and_crop(tfr_sub_stats, baseline=baseline, mode="logratio", logger=logger)
        
        tfr_pain_raw = tfr_sub[pain_mask].average()
        tfr_non_raw = tfr_sub[non_mask].average()
        tfr_max_raw = tfr_sub[mask_max].average()
        tfr_min_raw = tfr_sub[mask_min].average()
        
        apply_baseline_and_crop(tfr_pain_raw, baseline=baseline_used, mode="logratio", logger=logger)
        apply_baseline_and_crop(tfr_non_raw, baseline=baseline_used, mode="logratio", logger=logger)
        apply_baseline_and_crop(tfr_max_raw, baseline=baseline_used, mode="logratio", logger=logger)
        apply_baseline_and_crop(tfr_min_raw, baseline=baseline_used, mode="logratio", logger=logger)
        
        tfr_pain_avg_list.append(tfr_pain_raw)
        tfr_non_avg_list.append(tfr_non_raw)
        tfr_max_avg_list.append(tfr_max_raw)
        tfr_min_avg_list.append(tfr_min_raw)
        
        tfr_pain_epochs_list.append(tfr_sub_stats[pain_mask])
        tfr_non_epochs_list.append(tfr_sub_stats[non_mask])
        tfr_max_epochs_list.append(tfr_sub_stats[mask_max])
        tfr_min_epochs_list.append(tfr_sub_stats[mask_min])
    
    if len(tfr_pain_avg_list) < 2:
        log("Group contrast: insufficient subjects with valid data; skipping.", logger, "warning")
        return
    
    combined_avg = _combine_multiple_tfr_groups({
        "pain": tfr_pain_avg_list,
        "non": tfr_non_avg_list,
        "max": tfr_max_avg_list,
        "min": tfr_min_avg_list
    }, 2, logger)
    
    tfr_pain = combined_avg["pain"]
    tfr_non = combined_avg["non"]
    tfr_max = combined_avg["max"]
    tfr_min = combined_avg["min"]
    
    if tfr_pain is None or tfr_non is None or tfr_max is None or tfr_min is None:
        log("Group contrast: failed to combine TFRs; skipping.", logger, "warning")
        return
    
    combined_epochs = _combine_multiple_epochs_tfr_groups({
        "pain": tfr_pain_epochs_list,
        "non": tfr_non_epochs_list,
        "max": tfr_max_epochs_list,
        "min": tfr_min_epochs_list
    }, 2, logger)
    
    tfr_pain_combined = combined_epochs["pain"]
    tfr_non_combined = combined_epochs["non"]
    tfr_max_combined = combined_epochs["max"]
    tfr_min_combined = combined_epochs["min"]
    
    times = np.asarray(tfr_pain.times)
    tmin_req, tmax_req = plateau_window
    tmin_eff = float(max(times.min(), tmin_req))
    tmax_eff = float(min(times.max(), tmax_req))
    tmin, tmax = tmin_eff, tmax_eff
    
    fmax_available = float(np.max(tfr_pain.freqs))
    bands = get_bands_for_tfr(max_freq_available=fmax_available, config=config)
    
    n_rows = 2
    n_cols = len(bands)
    plot_cfg_large2_pain = get_plot_config(config)
    fig_size_per_col_large2_pain = plot_cfg_large2_pain.get_figure_size("tfr_per_col_large", plot_type="tfr")[0]
    fig_size_per_row_large_pain = plot_cfg_large2_pain.get_figure_size("tfr_per_row_large", plot_type="tfr")[1]
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_size_per_col_large2_pain * n_cols, fig_size_per_row_large_pain * n_rows),
        squeeze=False,
        gridspec_kw={"wspace": 1.2, "hspace": 0.25},
    )
    
    viz_params = get_viz_params(config)
    n_pain = len(tfr_pain_avg_list)
    n_non = len(tfr_non_avg_list)
    n_max = len(tfr_max_avg_list)
    n_min = len(tfr_min_avg_list)
    
    all_pain_diff = []
    all_temp_diff = []
    
    for c, (band, (fmin, fmax)) in enumerate(bands.items()):
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            for r in range(n_rows):
                axes[r, c].axis('off')
            continue
        
        pain_data = average_tfr_band(tfr_pain, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
        non_data = average_tfr_band(tfr_non, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
        max_data = average_tfr_band(tfr_max, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
        min_data = average_tfr_band(tfr_min, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
        
        if pain_data is None or non_data is None or max_data is None or min_data is None:
            for r in range(n_rows):
                axes[r, c].axis('off')
            continue
        
        pain_diff_data = pain_data - non_data
        temp_diff_data = max_data - min_data
        
        all_pain_diff.append(pain_diff_data)
        all_temp_diff.append(temp_diff_data)
    
    pain_diff_abs = robust_sym_vlim(all_pain_diff) if len(all_pain_diff) > 0 else 0.0
    temp_diff_abs = robust_sym_vlim(all_temp_diff) if len(all_temp_diff) > 0 else 0.0
    
    for c, (band, (fmin, fmax)) in enumerate(bands.items()):
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            continue
        
        pain_diff_data, temp_diff_data = _compute_band_diff_data(tfr_pain, tfr_non, tfr_max, tfr_min, fmin, fmax_eff, tmin, tmax)
        
        if pain_diff_data is None or temp_diff_data is None:
            continue
        
        pain_sig_mask = None
        pain_cluster_p_min = pain_cluster_k = pain_cluster_mass = None
        if viz_params["diff_annotation_enabled"]:
            pain_sig_mask, pain_cluster_p_min, pain_cluster_k, pain_cluster_mass = _compute_cluster_significance_from_combined(
                tfr_pain_epochs_list, tfr_non_epochs_list, fmin, fmax_eff, tmin, tmax, config, len(pain_diff_data), logger
            )
        
        temp_sig_mask = None
        temp_cluster_p_min = temp_cluster_k = temp_cluster_mass = None
        if viz_params["diff_annotation_enabled"]:
            temp_sig_mask, temp_cluster_p_min, temp_cluster_k, temp_cluster_mass = _compute_cluster_significance_from_combined(
                tfr_max_epochs_list, tfr_min_epochs_list, fmin, fmax_eff, tmin, tmax, config, len(temp_diff_data), logger
            )
        
        pain_data_group_a = extract_trial_band_power(tfr_pain_combined, fmin, fmax_eff, tmin, tmax)
        pain_data_group_b = extract_trial_band_power(tfr_non_combined, fmin, fmax_eff, tmin, tmax)
        temp_data_group_a = extract_trial_band_power(tfr_max_combined, fmin, fmax_eff, tmin, tmax)
        temp_data_group_b = extract_trial_band_power(tfr_min_combined, fmin, fmax_eff, tmin, tmax)
        
        _plot_diff_topomap_with_label(
            axes[0, c], pain_diff_data, tfr_pain.info, pain_diff_abs,
            pain_sig_mask, pain_cluster_p_min, pain_cluster_k, pain_cluster_mass, config, viz_params,
            data_group_a=pain_data_group_a,
            data_group_b=pain_data_group_b,
            paired=False
        )
        
        _plot_diff_topomap_with_label(
            axes[1, c], temp_diff_data, tfr_max.info, temp_diff_abs,
            temp_sig_mask, temp_cluster_p_min, temp_cluster_k, temp_cluster_mass, config, viz_params,
            data_group_a=temp_data_group_a,
            data_group_b=temp_data_group_b,
            paired=False
        )
        
        axes[0, c].set_title(f"{band} ({fmin:.0f}-{fmax_eff:.0f} Hz)", fontsize=9, pad=4, y=1.04)
    
    create_difference_colorbar(fig, axes[0, :].ravel().tolist(), pain_diff_abs, viz_params["topo_cmap"], config=config)
    create_difference_colorbar(fig, axes[1, :].ravel().tolist(), temp_diff_abs, viz_params["topo_cmap"], config=config)
    
    font_sizes = get_font_sizes()
    axes[0, 0].set_ylabel(f"Pain - Non (N={n_pain} subjects)", fontsize=font_sizes["ylabel"])
    axes[1, 0].set_ylabel(f"Max - Min temp (N={n_max} subjects)", fontsize=font_sizes["ylabel"])
    sig_text = get_sig_marker_text(config)
    fig.suptitle(f"Group Topomaps (baseline: logratio; t=[{tmin:.1f}, {tmax:.1f}] s, N={n_pain} subjects){sig_text}", fontsize=font_sizes["figure_title"])
    fig.supxlabel("Frequency bands", fontsize=font_sizes["ylabel"])
    _save_fig(fig, out_dir, "group_topomap_grid_bands_pain_temp_contrasts_bl.png", config=config, logger=logger, baseline_used=baseline)


def group_plot_topomap_grid_baseline_temps(
    powers: List["mne.time_frequency.EpochsTFR"],
    events_list: List[Optional[pd.DataFrame]],
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    plateau_window: Tuple[float, float] = (3.0, 10.5),
    logger: Optional[logging.Logger] = None,
) -> None:
    baseline = _get_baseline_window(config, baseline)
    
    if len(powers) < 2:
        log("Group temperature grid requires at least 2 subjects; skipping.", logger, "warning")
        return
    
    temp_col = next((c for c in config.get("event_columns.temperature", []) if c), None)
    if temp_col is None:
        log("Group temperature grid: no temperature column found; skipping.", logger, "warning")
        return
    
    all_temps = set()
    tfr_all_list = []
    tfr_by_temp: Dict[float, List["mne.time_frequency.AverageTFR"]] = {}
    
    for tfr, events_df in zip(powers, events_list):
        if events_df is None:
            continue
        
        tfr_corr = tfr.copy()
        tfr_avg_all = tfr_corr.average()
        baseline_used = apply_baseline_and_crop(tfr_avg_all, baseline=baseline, mode="percent", logger=logger)
        tfr_all_list.append(tfr_avg_all)
        
        temps = (
            pd.to_numeric(events_df[temp_col], errors="coerce")
            .round(1)
            .dropna()
            .unique()
        )
        temps = sorted(map(float, temps))
        all_temps.update(temps)
        
        for tval in temps:
            temp_values = pd.to_numeric(events_df[temp_col], errors="coerce")
            mask = np.abs(temp_values - float(tval)) < 0.05
            mask = np.asarray(mask, dtype=bool)
            if mask.sum() == 0:
                continue
            tfr_temp_raw = tfr_corr.copy()[mask].average()
            apply_baseline_and_crop(tfr_temp_raw, baseline=baseline_used, mode="percent", logger=logger)
            tfr_by_temp.setdefault(float(tval), []).append(tfr_temp_raw)
    
    if len(tfr_all_list) < 2:
        log("Group temperature grid: insufficient subjects; skipping.", logger, "warning")
        return
    
    tfr_avg_all_combined = _combine_avg_tfrs_group(tfr_all_list, logger)
    if tfr_avg_all_combined is None:
        log("Group temperature grid: failed to combine TFRs; skipping.", logger, "warning")
        return
    
    times_corr = np.asarray(tfr_avg_all_combined.times)
    tmin_req, tmax_req = plateau_window
    tmin = float(max(times_corr.min(), tmin_req))
    tmax = float(min(times_corr.max(), tmax_req))
    
    fmax_available = float(np.max(tfr_avg_all_combined.freqs))
    bands = get_bands_for_tfr(max_freq_available=fmax_available, config=config)
    
    cond_tfrs: list[tuple[str, "mne.time_frequency.AverageTFR", int, float]] = []
    cond_tfrs.append(("All trials", tfr_avg_all_combined, len(tfr_all_list), np.nan))
    
    sorted_temps = sorted(all_temps)
    for tval in sorted_temps:
        if tval in tfr_by_temp and len(tfr_by_temp[tval]) >= 2:
            tfr_combined = _combine_avg_tfrs_group(tfr_by_temp[tval], logger)
            if tfr_combined is not None:
                cond_tfrs.append((f"{tval:.1f}°C", tfr_combined, len(tfr_by_temp[tval]), float(tval)))
    
    plot_cfg = get_plot_config(config)
    tfr_config = plot_cfg.plot_type_configs.get("tfr", {})
    topomap_config = tfr_config.get("topomap", {})
    fig_size_per_col_large = plot_cfg.get_figure_size("tfr_per_col_large", plot_type="tfr")[0]
    fig_size_per_row_large = plot_cfg.get_figure_size("tfr_per_row_large", plot_type="tfr")[1]
    
    n_cols, n_rows = len(cond_tfrs), len(bands)
    fig, axes = plt.subplots(
        n_rows, n_cols, 
        figsize=(fig_size_per_col_large * n_cols, fig_size_per_row_large * n_rows), 
        squeeze=False,
        gridspec_kw={"wspace": 1.2, "hspace": 0.25},
    )
    
    for r, (band, (fmin, fmax)) in enumerate(bands.items()):
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            for c in range(n_cols):
                axes[r, c].axis("off")
            continue
        
        diff_datas: list[Optional[np.ndarray]] = []
        for _, tfr_cond, _, _ in cond_tfrs:
            d = average_tfr_band(tfr_cond, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
            diff_datas.append(d)
        
        vals = [v for v in diff_datas if v is not None and np.isfinite(v).any()]
        if len(vals) == 0:
            for c in range(n_cols):
                axes[r, c].axis("off")
            continue
        
        diff_abs = robust_sym_vlim(vals, cap=100.0)
        if not np.isfinite(diff_abs) or diff_abs == 0:
            diff_abs = 1e-6
        
        for idx, (label, tfr_cond, n_cond, _tval) in enumerate(cond_tfrs, start=0):
            ax = axes[r, idx]
            data = diff_datas[idx]
            if data is None:
                ax.axis("off")
                continue
            
            plot_topomap_on_ax(ax, data, tfr_cond.info, vmin=-diff_abs, vmax=+diff_abs)
            _add_roi_annotations(ax, data, tfr_cond.info, config=config, data_format="percent")
            eeg_picks = extract_eeg_picks(tfr_cond, exclude_bads=False)
            mu = float(np.nanmean(data[eeg_picks])) if len(eeg_picks) > 0 else float(np.nanmean(data))
            title_y = topomap_config.get("title_y", 1.04)
            title_pad = topomap_config.get("title_pad", 4)
            build_topomap_percentage_label(data, ax, mu, plot_cfg.font.title)
            if r == 0:
                ax.set_title(f"{label} (N={n_cond} subjects)", fontsize=plot_cfg.font.title, pad=title_pad, y=title_y)
        
        axes[r, 0].set_ylabel(f"{band} ({fmin:.0f}-{fmax_eff:.0f} Hz)", fontsize=plot_cfg.font.ylabel)
        
        create_difference_colorbar(
            fig, axes[r, :].ravel().tolist(), diff_abs, get_viz_params(config)["topo_cmap"],
            label="Percent change from baseline (%)", config=config, fontsize=plot_cfg.font.title
        )
    
    fig.suptitle(
        f"Group Topomaps by temperature: % change from baseline over plateau t=[{tmin:.1f}, {tmax:.1f}] s (N={len(tfr_all_list)} subjects)",
        fontsize=plot_cfg.font.figure_title,
    )
    _save_fig(fig, out_dir, "group_topomap_grid_bands_alltrials_plus_temperatures_baseline_percent.png", config=config, logger=logger, baseline_used=baseline)


def group_plot_pain_nonpain_temporal_topomaps_diff_allbands(
    powers: List["mne.time_frequency.EpochsTFR"],
    events_list: List[Optional[pd.DataFrame]],
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    plateau_window: Tuple[float, float] = (3.0, 10.5),
    window_size_ms: float = 100.0,
    logger: Optional[logging.Logger] = None,
) -> None:
    baseline = _get_baseline_window(config, baseline)
    
    if len(powers) < 2:
        log("Group temporal topomaps require at least 2 subjects; skipping.", logger, "warning")
        return
    
    pain_col = next((c for c in config.get("event_columns.pain_binary", []) if c), None)
    temp_col = next((c for c in config.get("event_columns.temperature", []) if c), None)
    
    if pain_col is None:
        log("Group temporal topomaps: pain column required; skipping.", logger, "warning")
        return
    
    tfr_pain_avg_list = []
    tfr_non_avg_list = []
    tfr_max_avg_list = []
    tfr_min_avg_list = []
    tfr_pain_epochs_list = []
    tfr_non_epochs_list = []
    tfr_max_epochs_list = []
    tfr_min_epochs_list = []
    has_temp = False
    baseline_used = None
    
    for tfr, events_df in zip(powers, events_list):
        if events_df is None:
            continue
        
        n = compute_aligned_data_length(tfr, events_df)
        
        pain_vec = extract_pain_vector_array(tfr, events_df, pain_col, n)
        if pain_vec is None:
            continue
        
        pain_mask, non_mask = _create_pain_masks_from_vector(pain_vec)
        if pain_mask is None:
            continue
        
        if pain_mask.sum() == 0 or non_mask.sum() == 0:
            continue
        
        tfr_sub = create_tfr_subset(tfr, n)
        aligned = _align_and_trim_masks(
            tfr_sub,
            {"Pain contrast": (pain_mask, non_mask)},
            config, logger
        )
        if aligned is None:
            continue
        
        pain_mask, non_mask = aligned["Pain contrast"]
        
        baseline_used = apply_baseline_and_crop(tfr_sub, baseline=baseline, mode="logratio", logger=logger)
        tfr_pain_avg_list.append(tfr_sub[pain_mask].average())
        tfr_non_avg_list.append(tfr_sub[non_mask].average())
        tfr_pain_epochs_list.append(tfr_sub[pain_mask])
        tfr_non_epochs_list.append(tfr_sub[non_mask])
        
        temp_series = extract_temperature_series(tfr, events_df, temp_col, n)
        if temp_series is not None:
            temp_result = create_temperature_masks(temp_series)
            if temp_result[0] is not None:
                t_min, t_max, mask_min, mask_max = temp_result
                
                if mask_min.sum() > 0 and mask_max.sum() > 0:
                    try:
                        ensure_aligned_lengths(
                            tfr_sub, mask_min, mask_max,
                            context=f"Temperature contrast",
                            strict=get_strict_mode(config),
                            logger=logger
                        )
                        if len(mask_min) != len(tfr_sub) or len(mask_max) != len(tfr_sub):
                            mask_min = mask_min[:len(tfr_sub)]
                            mask_max = mask_max[:len(tfr_sub)]
                        tfr_min_avg_list.append(tfr_sub[mask_min].average())
                        tfr_max_avg_list.append(tfr_sub[mask_max].average())
                        tfr_min_epochs_list.append(tfr_sub[mask_min])
                        tfr_max_epochs_list.append(tfr_sub[mask_max])
                        has_temp = True
                    except ValueError:
                        pass
    
    if len(tfr_pain_avg_list) < 2:
        log("Group temporal topomaps: insufficient subjects; skipping.", logger, "warning")
        return
    
    combined_avg = _combine_multiple_tfr_groups({
        "pain": tfr_pain_avg_list,
        "non": tfr_non_avg_list,
        "max": tfr_max_avg_list if has_temp else [],
        "min": tfr_min_avg_list if has_temp else []
    }, 2, logger)
    
    tfr_pain = combined_avg["pain"]
    tfr_non = combined_avg["non"]
    tfr_max = combined_avg["max"] if has_temp else None
    tfr_min = combined_avg["min"] if has_temp else None
    
    if tfr_pain is None or tfr_non is None:
        log("Group temporal topomaps: failed to combine TFRs; skipping.", logger, "warning")
        return
    
    combined_epochs = _combine_multiple_epochs_tfr_groups({
        "pain": tfr_pain_epochs_list,
        "non": tfr_non_epochs_list,
        "max": tfr_max_epochs_list if has_temp else [],
        "min": tfr_min_epochs_list if has_temp else []
    }, 2, logger)
    
    tfr_pain_combined = combined_epochs["pain"]
    tfr_non_combined = combined_epochs["non"]
    tfr_max_combined = combined_epochs["max"] if has_temp else None
    tfr_min_combined = combined_epochs["min"] if has_temp else None
    
    times = np.asarray(tfr_pain.times)
    tmin_req, tmax_req = plateau_window
    tmin_start = float(times.min())
    tmax_clip = float(min(times.max(), tmax_req))
    
    if not np.isfinite(tmin_start) or not np.isfinite(tmax_clip) or (tmax_clip <= tmin_start):
        log("Group temporal topomaps: no valid time interval; skipping.", logger, "warning")
        return
    
    window_starts, window_ends = create_time_windows_fixed_size(tmin_start, tmax_clip, window_size_ms)
    n_windows = len(window_starts)
    if n_windows == 0:
        log("Group temporal topomaps: no valid windows; skipping.", logger, "warning")
        return
    
    fmax_available = float(np.max(tfr_pain.freqs))
    bands = get_bands_for_tfr(max_freq_available=fmax_available, config=config)
    
    valid_bands = {}
    all_band_pain_diff_data = {}
    all_band_temp_diff_data = {}
    
    for band_name, (fmin, fmax) in bands.items():
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            continue
        
        pain_diff_data_windows = []
        temp_diff_data_windows = []
        
        for tmin_win, tmax_win in zip(window_starts, window_ends):
            pain_data = average_tfr_band(tfr_pain, fmin=fmin, fmax=fmax_eff, tmin=tmin_win, tmax=tmax_win)
            non_data = average_tfr_band(tfr_non, fmin=fmin, fmax=fmax_eff, tmin=tmin_win, tmax=tmax_win)
            
            if pain_data is not None and non_data is not None:
                diff_data = pain_data - non_data
                pain_diff_data_windows.append(diff_data)
            else:
                pain_diff_data_windows.append(None)
            
            if has_temp and tfr_max is not None and tfr_min is not None:
                max_data = average_tfr_band(tfr_max, fmin=fmin, fmax=fmax_eff, tmin=tmin_win, tmax=tmax_win)
                min_data = average_tfr_band(tfr_min, fmin=fmin, fmax=fmax_eff, tmin=tmin_win, tmax=tmax_win)
                
                if max_data is not None and min_data is not None:
                    temp_diff_data = max_data - min_data
                    temp_diff_data_windows.append(temp_diff_data)
                else:
                    temp_diff_data_windows.append(None)
            else:
                temp_diff_data_windows.append(None)
        
        pain_diff_data_valid = [d for d in pain_diff_data_windows if d is not None]
        if len(pain_diff_data_valid) == 0:
            continue
        
        valid_bands[band_name] = (fmin, fmax_eff, pain_diff_data_windows, temp_diff_data_windows if has_temp else None)
        all_band_pain_diff_data[band_name] = pain_diff_data_valid
        if has_temp:
            all_band_temp_diff_data[band_name] = [d for d in temp_diff_data_windows if d is not None]
    
    if len(valid_bands) == 0:
        log("Group temporal topomaps: no valid bands; skipping.", logger, "warning")
        return
    
    all_pain_diff_data = [d for data_list in all_band_pain_diff_data.values() for d in data_list]
    all_temp_diff_data = [d for data_list in all_band_temp_diff_data.values() for d in data_list] if has_temp else []
    all_diff_data = all_pain_diff_data + all_temp_diff_data
    vabs_diff = robust_sym_vlim(all_diff_data) if len(all_diff_data) > 0 else 1e-6
    
    n_bands = len(valid_bands)
    n_rows = n_bands * 2 if has_temp else n_bands
    plot_cfg = get_plot_config(config)
    fig_size_per_col_large = plot_cfg.get_figure_size("tfr_per_col_large", plot_type="tfr")[0]
    fig_size_per_row_large = plot_cfg.get_figure_size("tfr_per_row_large", plot_type="tfr")[1]
    fig, axes = plt.subplots(
        n_rows, n_windows, 
        figsize=(fig_size_per_col_large * n_windows, fig_size_per_row_large * n_rows), 
        squeeze=False,
        gridspec_kw={"hspace": 0.15, "wspace": 0.8}
    )
    
    for band_idx, (band_name, (fmin, fmax_eff, pain_diff_data_windows, temp_diff_data_windows)) in enumerate(valid_bands.items()):
        freq_label = f"{band_name} ({fmin:.0f}-{fmax_eff:.0f}Hz)"
        row_pain = band_idx * 2 if has_temp else band_idx
        row_temp = band_idx * 2 + 1 if has_temp else None
        
        font_sizes = get_font_sizes()
        axes[row_pain, 0].set_ylabel(f"Pain - Non\n{freq_label}", fontsize=font_sizes["ylabel"], labelpad=10)
        
        for col, (tmin_win, tmax_win) in enumerate(zip(window_starts, window_ends)):
            if row_pain == 0:
                time_label = f"{tmin_win:.2f}s"
                font_sizes = get_font_sizes()
                axes[row_pain, col].set_title(time_label, fontsize=font_sizes["title"], pad=12, y=1.07)
            
            pain_diff_data = pain_diff_data_windows[col]
            if pain_diff_data is not None:
                pain_sig_mask = pain_cluster_p_min = pain_cluster_k = pain_cluster_mass = None
                viz_params = get_viz_params(config)
                if viz_params["diff_annotation_enabled"]:
                    pain_sig_mask, pain_cluster_p_min, pain_cluster_k, pain_cluster_mass = _compute_cluster_significance_from_combined(
                        tfr_pain_epochs_list, tfr_non_epochs_list, fmin, fmax_eff, tmin_win, tmax_win, config, len(pain_diff_data), logger
                    )
                
                plot_topomap_on_ax(
                    axes[row_pain, col], pain_diff_data, tfr_pain.info,
                    vmin=-vabs_diff, vmax=+vabs_diff,
                    mask=(pain_sig_mask if viz_params["diff_annotation_enabled"] else None), 
                    mask_params=viz_params["sig_mask_params"],
                    config=config
                )
                pain_data_group_a = extract_trial_band_power(tfr_pain_combined, fmin, fmax_eff, tmin_win, tmax_win)
                pain_data_group_b = extract_trial_band_power(tfr_non_combined, fmin, fmax_eff, tmin_win, tmax_win)
                _add_roi_annotations(
                    axes[row_pain, col], pain_diff_data, tfr_pain.info, config=config,
                    sig_mask=(pain_sig_mask if viz_params["diff_annotation_enabled"] else None),
                    cluster_p_min=pain_cluster_p_min, cluster_k=pain_cluster_k, cluster_mass=pain_cluster_mass,
                    is_cluster=(pain_sig_mask is not None and pain_cluster_p_min is not None),
                    data_group_a=pain_data_group_a,
                    data_group_b=pain_data_group_b,
                    paired=False
                )
                label = build_topomap_diff_label(pain_diff_data, pain_cluster_p_min, pain_cluster_k, pain_cluster_mass, config, viz_params, paired=False)
                axes[row_pain, col].text(0.5, 1.08, label, transform=axes[row_pain, col].transAxes, ha="center", va="bottom", fontsize=font_sizes["label"])
            else:
                axes[row_pain, col].axis('off')
            
            if has_temp and row_temp is not None and temp_diff_data_windows is not None:
                temp_diff_data = temp_diff_data_windows[col]
                if temp_diff_data is not None:
                    temp_sig_mask = temp_cluster_p_min = temp_cluster_k = temp_cluster_mass = None
                    if viz_params["diff_annotation_enabled"]:
                        temp_sig_mask, temp_cluster_p_min, temp_cluster_k, temp_cluster_mass = _compute_cluster_significance_from_combined(
                            tfr_max_epochs_list, tfr_min_epochs_list, fmin, fmax_eff, tmin_win, tmax_win, config, len(temp_diff_data), logger
                        )
                    
                    plot_topomap_on_ax(
                        axes[row_temp, col], temp_diff_data, tfr_max.info,
                        vmin=-vabs_diff, vmax=+vabs_diff,
                        mask=(temp_sig_mask if viz_params["diff_annotation_enabled"] else None), 
                        mask_params=viz_params["sig_mask_params"],
                        config=config
                    )
                    _add_roi_annotations(
                        axes[row_temp, col], temp_diff_data, tfr_max.info, config=config,
                        sig_mask=(temp_sig_mask if viz_params["diff_annotation_enabled"] else None),
                        cluster_p_min=temp_cluster_p_min, cluster_k=temp_cluster_k, cluster_mass=temp_cluster_mass,
                        is_cluster=(temp_sig_mask is not None and temp_cluster_p_min is not None)
                    )
                    mu = float(np.nanmean(temp_diff_data))
                    pct = logratio_to_pct(mu)
                    cl_txt = format_cluster_ann(temp_cluster_p_min, temp_cluster_k, temp_cluster_mass, config=config) if viz_params["diff_annotation_enabled"] else ""
                    label = f"Δ%={pct:+.1f}%" + (f" | {cl_txt}" if cl_txt else "")
                    axes[row_temp, col].text(0.5, 1.08, label, transform=axes[row_temp, col].transAxes, ha="center", va="bottom", fontsize=font_sizes["label"])
                else:
                    axes[row_temp, col].axis('off')
                
                if col == 0:
                    axes[row_temp, 0].set_ylabel(f"Max - Min temp\n{freq_label}", fontsize=font_sizes["ylabel"], labelpad=10)
    
    viz_params = get_viz_params(config)
    create_difference_colorbar(
        fig, axes, vabs_diff, viz_params["topo_cmap"],
        label="log10(power/baseline) difference",
        config=config
    )
    
    baseline_str = f"bl{abs(baseline_used[0]):.1f}to{abs(baseline_used[1]):.2f}" if baseline_used else "bl"
    sig_text = get_sig_marker_text(config)
    title_parts = [f"Group Temporal topomaps: Pain - Non-pain difference (all bands, {tmin_start:.1f}–{tmax_clip:.1f}s; {n_windows} windows @ {window_size_ms}ms, N={len(tfr_pain_avg_list)} subjects)"]
    if has_temp:
        title_parts.append(f"Max - Min temp")
    title_parts.append(f"log10(power/baseline) difference, vlim ±{vabs_diff:.2f}")
    
    stat_title = build_statistical_title(
        config, baseline_used, paired=False,
        n_subjects=len(tfr_pain_avg_list), is_group=True
    )
    
    full_title = f"{' | '.join(title_parts)}"
    if stat_title:
        full_title += f"\n{stat_title}"
    if sig_text:
        full_title += sig_text
    
    font_sizes = get_font_sizes()
    fig.suptitle(full_title, fontsize=font_sizes["figure_title"], y=0.995)
    
    filename = f"group_temporal_topomaps_pain_minus_nonpain_diff_allbands_{tmin_start:.0f}-{tmax_clip:.0f}s_{n_windows}windows_{baseline_str}.png"
    _save_fig(fig, out_dir, filename, config=config, logger=logger, baseline_used=baseline_used)


def group_plot_temporal_topomaps_allbands_plateau(
    powers: List["mne.time_frequency.EpochsTFR"],
    events_list: List[Optional[pd.DataFrame]],
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    plateau_window: Tuple[float, float] = (3.0, 10.5),
    window_count: int = 5,
    logger: Optional[logging.Logger] = None,
) -> None:
    baseline = _get_baseline_window(config, baseline)
    
    if len(powers) < 2:
        log("Group temporal topomaps require at least 2 subjects; skipping.", logger, "warning")
        return
    
    pain_col = next((c for c in config.get("event_columns.pain_binary", []) if c), None)
    temp_col = next((c for c in config.get("event_columns.temperature", []) if c), None)
    
    if pain_col is None:
        log("Group temporal topomaps: pain column required; skipping.", logger, "warning")
        return
    
    tfr_pain_avg_list = []
    tfr_non_avg_list = []
    tfr_max_avg_list = []
    tfr_min_avg_list = []
    tfr_pain_epochs_list = []
    tfr_non_epochs_list = []
    tfr_max_epochs_list = []
    tfr_min_epochs_list = []
    has_temp = False
    baseline_used = None
    
    for tfr, events_df in zip(powers, events_list):
        if events_df is None:
            continue
        
        n = compute_aligned_data_length(tfr, events_df)
        
        pain_vec = extract_pain_vector_array(tfr, events_df, pain_col, n)
        if pain_vec is None:
            continue
        
        pain_mask, non_mask = _create_pain_masks_from_vector(pain_vec)
        if pain_mask is None:
            continue
        
        if pain_mask.sum() == 0 or non_mask.sum() == 0:
            continue
        
        tfr_sub = create_tfr_subset(tfr, n)
        aligned = _align_and_trim_masks(
            tfr_sub,
            {"Pain contrast": (pain_mask, non_mask)},
            config, logger
        )
        if aligned is None:
            continue
        
        pain_mask, non_mask = aligned["Pain contrast"]
        
        tfr_sub_stats = tfr_sub.copy()
        baseline_used = apply_baseline_and_crop(tfr_sub_stats, baseline=baseline, mode="logratio", logger=logger)
        
        tfr_pain_raw = tfr_sub[pain_mask].average()
        tfr_non_raw = tfr_sub[non_mask].average()
        
        apply_baseline_and_crop(tfr_pain_raw, baseline=baseline_used, mode="logratio", logger=logger)
        apply_baseline_and_crop(tfr_non_raw, baseline=baseline_used, mode="logratio", logger=logger)
        
        tfr_pain_avg_list.append(tfr_pain_raw)
        tfr_non_avg_list.append(tfr_non_raw)
        tfr_pain_epochs_list.append(tfr_sub_stats[pain_mask])
        tfr_non_epochs_list.append(tfr_sub_stats[non_mask])
        
        temp_series = extract_temperature_series(tfr, events_df, temp_col, n)
        if temp_series is not None:
            temp_result = create_temperature_masks(temp_series)
            if temp_result[0] is not None:
                t_min, t_max, mask_min, mask_max = temp_result
                
                if mask_min.sum() > 0 and mask_max.sum() > 0:
                    try:
                        ensure_aligned_lengths(
                            tfr_sub, mask_min, mask_max,
                            context=f"Temperature contrast",
                            strict=get_strict_mode(config),
                            logger=logger
                        )
                        if len(mask_min) != len(tfr_sub) or len(mask_max) != len(tfr_sub):
                            mask_min = mask_min[:len(tfr_sub)]
                            mask_max = mask_max[:len(tfr_sub)]
                        
                        tfr_min_raw = tfr_sub[mask_min].average()
                        tfr_max_raw = tfr_sub[mask_max].average()
                        apply_baseline_and_crop(tfr_min_raw, baseline=baseline_used, mode="logratio", logger=logger)
                        apply_baseline_and_crop(tfr_max_raw, baseline=baseline_used, mode="logratio", logger=logger)
                        
                        tfr_min_avg_list.append(tfr_min_raw)
                        tfr_max_avg_list.append(tfr_max_raw)
                        tfr_min_epochs_list.append(tfr_sub_stats[mask_min])
                        tfr_max_epochs_list.append(tfr_sub_stats[mask_max])
                        has_temp = True
                    except ValueError:
                        pass
    
    if len(tfr_pain_avg_list) < 2:
        log("Group temporal topomaps: insufficient subjects; skipping.", logger, "warning")
        return
    
    combined_avg = _combine_multiple_tfr_groups({
        "pain": tfr_pain_avg_list,
        "non": tfr_non_avg_list,
        "max": tfr_max_avg_list if has_temp else [],
        "min": tfr_min_avg_list if has_temp else []
    }, 2, logger)
    
    tfr_pain = combined_avg["pain"]
    tfr_non = combined_avg["non"]
    tfr_max = combined_avg["max"] if has_temp else None
    tfr_min = combined_avg["min"] if has_temp else None
    
    if tfr_pain is None or tfr_non is None:
        log("Group temporal topomaps: failed to combine TFRs; skipping.", logger, "warning")
        return
    
    combined_epochs = _combine_multiple_epochs_tfr_groups({
        "pain": tfr_pain_epochs_list,
        "non": tfr_non_epochs_list,
        "max": tfr_max_epochs_list if has_temp else [],
        "min": tfr_min_epochs_list if has_temp else []
    }, 2, logger)
    
    tfr_pain_combined = combined_epochs["pain"]
    tfr_non_combined = combined_epochs["non"]
    tfr_max_combined = combined_epochs["max"] if has_temp else None
    tfr_min_combined = combined_epochs["min"] if has_temp else None
    
    times = np.asarray(tfr_pain.times)
    tmin_req, tmax_req = plateau_window
    clipped = clip_time_range(times, tmin_req, tmax_req)
    if clipped is None:
        return None
    tmin_clip, tmax_clip = clipped
    
    if tmin_clip is None or tmax_clip is None:
        log("Group temporal topomaps: no valid time interval; skipping.", logger, "warning")
        return
    
    window_starts, window_ends = create_time_windows_fixed_count(tmin_clip, tmax_clip, window_count)
    n_windows = len(window_starts)
    
    if n_windows == 0:
        log("Group temporal topomaps: no valid windows; skipping.", logger, "warning")
        return
    
    fmax_available = float(np.max(tfr_pain.freqs))
    bands = get_bands_for_tfr(max_freq_available=fmax_available, config=config)
    
    valid_bands = {}
    all_band_pain_diff_data = {}
    all_band_temp_diff_data = {}
    
    for band_name, (fmin, fmax) in bands.items():
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            continue
        
        pain_diff_data_windows = []
        temp_diff_data_windows = []
        
        for tmin_win, tmax_win in zip(window_starts, window_ends):
            pain_data = average_tfr_band(tfr_pain, fmin=fmin, fmax=fmax_eff, tmin=tmin_win, tmax=tmax_win)
            non_data = average_tfr_band(tfr_non, fmin=fmin, fmax=fmax_eff, tmin=tmin_win, tmax=tmax_win)
            
            if pain_data is not None and non_data is not None:
                diff_data = pain_data - non_data
                pain_diff_data_windows.append(diff_data)
            else:
                pain_diff_data_windows.append(None)
            
            if has_temp and tfr_max is not None and tfr_min is not None:
                max_data = average_tfr_band(tfr_max, fmin=fmin, fmax=fmax_eff, tmin=tmin_win, tmax=tmax_win)
                min_data = average_tfr_band(tfr_min, fmin=fmin, fmax=fmax_eff, tmin=tmin_win, tmax=tmax_win)
                
                if max_data is not None and min_data is not None:
                    temp_diff_data = max_data - min_data
                    temp_diff_data_windows.append(temp_diff_data)
                else:
                    temp_diff_data_windows.append(None)
            else:
                temp_diff_data_windows.append(None)
        
        pain_diff_data_valid = [d for d in pain_diff_data_windows if d is not None]
        if len(pain_diff_data_valid) == 0:
            continue
        
        valid_bands[band_name] = (fmin, fmax_eff, pain_diff_data_windows, temp_diff_data_windows if has_temp else None)
        all_band_pain_diff_data[band_name] = pain_diff_data_valid
        if has_temp:
            all_band_temp_diff_data[band_name] = [d for d in temp_diff_data_windows if d is not None]
    
    if len(valid_bands) == 0:
        log("Group temporal topomaps: no valid bands; skipping.", logger, "warning")
        return
    
    all_pain_diff_data = [d for data_list in all_band_pain_diff_data.values() for d in data_list]
    all_temp_diff_data = [d for data_list in all_band_temp_diff_data.values() for d in data_list] if has_temp else []
    all_diff_data = all_pain_diff_data + all_temp_diff_data
    vabs_diff = robust_sym_vlim(all_diff_data) if len(all_diff_data) > 0 else 1e-6
    
    n_bands = len(valid_bands)
    n_rows = n_bands * 2 if has_temp else n_bands
    plot_cfg = get_plot_config(config)
    fig_size_per_col_large = plot_cfg.get_figure_size("tfr_per_col_large", plot_type="tfr")[0]
    fig_size_per_row_large = plot_cfg.get_figure_size("tfr_per_row_large", plot_type="tfr")[1]
    fig, axes = plt.subplots(
        n_rows, n_windows, 
        figsize=(fig_size_per_col_large * n_windows, fig_size_per_row_large * n_rows), 
        squeeze=False,
        gridspec_kw={"hspace": 0.15, "wspace": 0.8}
    )
    
    for band_idx, (band_name, (fmin, fmax_eff, pain_diff_data_windows, temp_diff_data_windows)) in enumerate(valid_bands.items()):
        freq_label = f"{band_name} ({fmin:.0f}-{fmax_eff:.0f}Hz)"
        row_pain = band_idx * 2 if has_temp else band_idx
        row_temp = band_idx * 2 + 1 if has_temp else None
        
        font_sizes = get_font_sizes()
        axes[row_pain, 0].set_ylabel(f"Pain - Non\n{freq_label}", fontsize=font_sizes["ylabel"], labelpad=10)
        
        for col, (tmin_win, tmax_win) in enumerate(zip(window_starts, window_ends)):
            if row_pain == 0:
                time_label = f"{tmin_win:.2f}s"
                font_sizes = get_font_sizes()
                axes[row_pain, col].set_title(time_label, fontsize=font_sizes["title"], pad=12, y=1.07)
            
            pain_diff_data = pain_diff_data_windows[col]
            if pain_diff_data is not None:
                pain_sig_mask = pain_cluster_p_min = pain_cluster_k = pain_cluster_mass = None
                viz_params = get_viz_params(config)
                if viz_params["diff_annotation_enabled"]:
                    pain_sig_mask, pain_cluster_p_min, pain_cluster_k, pain_cluster_mass = _compute_cluster_significance_from_combined(
                        tfr_pain_epochs_list, tfr_non_epochs_list, fmin, fmax_eff, tmin_win, tmax_win, config, len(pain_diff_data), logger
                    )
                
                plot_topomap_on_ax(
                    axes[row_pain, col], pain_diff_data, tfr_pain.info,
                    vmin=-vabs_diff, vmax=+vabs_diff,
                    mask=(pain_sig_mask if viz_params["diff_annotation_enabled"] else None), 
                    mask_params=viz_params["sig_mask_params"],
                    config=config
                )
                pain_data_group_a = extract_trial_band_power(tfr_pain_combined, fmin, fmax_eff, tmin_win, tmax_win)
                pain_data_group_b = extract_trial_band_power(tfr_non_combined, fmin, fmax_eff, tmin_win, tmax_win)
                _add_roi_annotations(
                    axes[row_pain, col], pain_diff_data, tfr_pain.info, config=config,
                    sig_mask=(pain_sig_mask if viz_params["diff_annotation_enabled"] else None),
                    cluster_p_min=pain_cluster_p_min, cluster_k=pain_cluster_k, cluster_mass=pain_cluster_mass,
                    is_cluster=(pain_sig_mask is not None and pain_cluster_p_min is not None),
                    data_group_a=pain_data_group_a,
                    data_group_b=pain_data_group_b,
                    paired=False
                )
                label = build_topomap_diff_label(pain_diff_data, pain_cluster_p_min, pain_cluster_k, pain_cluster_mass, config, viz_params, paired=False)
                axes[row_pain, col].text(0.5, 1.08, label, transform=axes[row_pain, col].transAxes, ha="center", va="bottom", fontsize=font_sizes["label"])
            else:
                axes[row_pain, col].axis('off')
            
            if has_temp and row_temp is not None and temp_diff_data_windows is not None:
                temp_diff_data = temp_diff_data_windows[col]
                if temp_diff_data is not None:
                    temp_sig_mask = temp_cluster_p_min = temp_cluster_k = temp_cluster_mass = None
                    if viz_params["diff_annotation_enabled"]:
                        temp_sig_mask, temp_cluster_p_min, temp_cluster_k, temp_cluster_mass = _compute_cluster_significance_from_combined(
                            tfr_max_epochs_list, tfr_min_epochs_list, fmin, fmax_eff, tmin_win, tmax_win, config, len(temp_diff_data), logger
                        )
                    
                    plot_topomap_on_ax(
                        axes[row_temp, col], temp_diff_data, tfr_max.info,
                        vmin=-vabs_diff, vmax=+vabs_diff,
                        mask=(temp_sig_mask if viz_params["diff_annotation_enabled"] else None), 
                        mask_params=viz_params["sig_mask_params"],
                        config=config
                    )
                    _add_roi_annotations(
                        axes[row_temp, col], temp_diff_data, tfr_max.info, config=config,
                        sig_mask=(temp_sig_mask if viz_params["diff_annotation_enabled"] else None),
                        cluster_p_min=temp_cluster_p_min, cluster_k=temp_cluster_k, cluster_mass=temp_cluster_mass,
                        is_cluster=(temp_sig_mask is not None and temp_cluster_p_min is not None)
                    )
                    mu = float(np.nanmean(temp_diff_data))
                    pct = logratio_to_pct(mu)
                    cl_txt = format_cluster_ann(temp_cluster_p_min, temp_cluster_k, temp_cluster_mass, config=config) if viz_params["diff_annotation_enabled"] else ""
                    label = f"Δ%={pct:+.1f}%" + (f" | {cl_txt}" if cl_txt else "")
                    axes[row_temp, col].text(0.5, 1.08, label, transform=axes[row_temp, col].transAxes, ha="center", va="bottom", fontsize=font_sizes["label"])
                else:
                    axes[row_temp, col].axis('off')
                
                if col == 0:
                    axes[row_temp, 0].set_ylabel(f"Max - Min temp\n{freq_label}", fontsize=font_sizes["ylabel"], labelpad=10)
    
    viz_params = get_viz_params(config)
    create_difference_colorbar(
        fig, axes, vabs_diff, viz_params["topo_cmap"],
        label="log10(power/baseline) difference",
        config=config
    )
    
    baseline_str = f"bl{abs(baseline_used[0]):.1f}to{abs(baseline_used[1]):.2f}" if baseline_used else "bl"
    sig_text = get_sig_marker_text(config)
    title_parts = [f"Group Temporal topomaps: Pain - Non-pain difference (all bands, plateau {tmin_clip:.0f}–{tmax_clip:.0f}s; {n_windows} windows, N={len(tfr_pain_avg_list)} subjects)"]
    if has_temp:
        title_parts.append(f"Max - Min temp")
    title_parts.append(f"log10(power/baseline) difference, vlim ±{vabs_diff:.2f}")
    
    stat_title = build_statistical_title(
        config, baseline_used, paired=False,
        n_subjects=len(tfr_pain_avg_list), is_group=True
    )
    
    full_title = f"{' | '.join(title_parts)}"
    if stat_title:
        full_title += f"\n{stat_title}"
    if sig_text:
        full_title += sig_text
    
    font_sizes = get_font_sizes()
    fig.suptitle(full_title, fontsize=font_sizes["figure_title"], y=0.995)
    
    filename = f"group_temporal_topomaps_allbands_plateau_{tmin_clip:.0f}-{tmax_clip:.0f}s_{n_windows}windows_{baseline_str}.png"
    _save_fig(fig, out_dir, filename, config=config, logger=logger, baseline_used=baseline_used)


__all__ = [
    "group_contrast_maxmin_temperature",
    "group_rois_all_trials",
    "group_contrast_pain_nonpain_rois",
    "group_contrast_pain_nonpain_scalpmean",
    "group_plot_bands_pain_temp_contrasts",
    "group_plot_topomap_grid_baseline_temps",
    "group_plot_pain_nonpain_temporal_topomaps_diff_allbands",
    "group_plot_temporal_topomaps_allbands_plateau",
]
