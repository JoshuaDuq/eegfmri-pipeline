from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.core.annotations import find_annotation_x_position, get_sig_marker_text
from eeg_pipeline.plotting.core.colorbars import create_difference_colorbar
from eeg_pipeline.plotting.core.utils import get_font_sizes
from eeg_pipeline.utils.analysis.stats import compute_band_correlations, compute_correlation_vmax
from eeg_pipeline.utils.analysis.tfr import build_roi_channel_mask, build_rois_from_info
from eeg_pipeline.utils.data.loading import prepare_topomap_correlation_data
from eeg_pipeline.io.logging import get_default_logger as _get_default_logger
from eeg_pipeline.io.paths import ensure_dir
from eeg_pipeline.plotting.io.figures import (
    get_behavior_footer as _get_behavior_footer,
    get_default_config as _get_default_config,
    get_viz_params,
    plot_topomap_on_ax,
    robust_sym_vlim,
    save_fig,
)


def _add_correlation_roi_annotations(
    ax,
    corr_data,
    p_uncorr,
    p_fdr,
    info,
    config=None,
    roi_map=None,
    fdr_alpha=0.05,
):
    if config is None and roi_map is None:
        return

    if roi_map is None and config is not None:
        roi_map = build_rois_from_info(info, config=config)
    if not roi_map:
        return

    ch_names = info["ch_names"]
    if len(corr_data) != len(ch_names):
        return

    plot_cfg = get_plot_config(config) if config else None
    tfr_config = plot_cfg.plot_type_configs.get("tfr", {}) if plot_cfg else {}
    font_sizes = get_font_sizes(plot_cfg)
    annotation_fontsize = font_sizes["annotation"]

    x_pos_ax = find_annotation_x_position(ax, plot_cfg)
    annotation_y_start = tfr_config.get("annotation_y_start", 0.98)
    y_pos_ax = annotation_y_start
    annotation_line_height = tfr_config.get("annotation_line_height", 0.045)
    annotation_min_spacing = tfr_config.get("annotation_min_spacing", 0.03)
    annotation_spacing_multiplier = tfr_config.get("annotation_spacing_multiplier", 0.3)

    annotations = []
    for roi, roi_chs in roi_map.items():
        mask_vec = build_roi_channel_mask(ch_names, roi_chs)
        if not mask_vec.any():
            continue

        roi_corrs = corr_data[mask_vec]
        roi_corrs_finite = roi_corrs[np.isfinite(roi_corrs)]
        if len(roi_corrs_finite) == 0:
            continue

        mean_corr = np.nanmean(roi_corrs_finite)

        roi_p_uncorr = None
        roi_p_fdr = None
        if p_uncorr is not None:
            roi_p_uncorr_vals = p_uncorr[mask_vec]
            roi_p_uncorr_finite = roi_p_uncorr_vals[np.isfinite(roi_p_uncorr_vals)]
            if len(roi_p_uncorr_finite) > 0:
                roi_p_uncorr = np.nanmin(roi_p_uncorr_finite)

        if p_fdr is not None:
            roi_p_fdr_vals = p_fdr[mask_vec]
            roi_p_fdr_finite = roi_p_fdr_vals[np.isfinite(roi_p_fdr_vals)]
            if len(roi_p_fdr_finite) > 0:
                roi_p_fdr = np.nanmin(roi_p_fdr_finite)

        annotations.append((roi, mean_corr, roi_p_uncorr, roi_p_fdr))

    for i, (roi, mean_corr, roi_p_uncorr, roi_p_fdr) in enumerate(annotations):
        if not np.isfinite(mean_corr):
            continue

        label = f"{roi}: r={mean_corr:+.2f}"

        if roi_p_fdr is not None and np.isfinite(roi_p_fdr) and roi_p_fdr < fdr_alpha:
            label += f" (q={roi_p_fdr:.3f})"
        elif roi_p_uncorr is not None and np.isfinite(roi_p_uncorr) and roi_p_uncorr < 0.05:
            label += f" (p={roi_p_uncorr:.3f})"

        ax.text(
            x_pos_ax,
            y_pos_ax,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=annotation_fontsize,
        )

        if i < len(annotations) - 1:
            spacing_ax = annotation_min_spacing + (
                annotation_line_height * annotation_spacing_multiplier
            )
            y_pos_ax -= (annotation_line_height + spacing_ax)


def _load_global_fdr_for_temporal_correlations(
    stats_dir: Path,
    use_spearman: bool,
    logger: logging.Logger,
) -> Optional[Dict[Tuple[str, str, float, float, str], bool]]:
    use_spearman_suffix = "_spearman" if use_spearman else "_pearson"
    tsv_path = stats_dir / f"corr_stats_temporal_all{use_spearman_suffix}.tsv"

    if not tsv_path.exists():
        logger.debug(f"TSV file not found for global FDR: {tsv_path}")
        return None

    try:
        df = pd.read_csv(tsv_path, sep="\t")
    except Exception as e:
        logger.warning(f"Failed to load TSV for global FDR: {e}")
        return None

    if df.empty:
        logger.warning(f"TSV file is empty: {tsv_path}")
        return None

    required_cols = ["condition", "band", "time_start", "time_end", "channel"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"TSV file missing required columns: {missing_cols}")
        return None

    fdr_col = "fdr_reject_global" if "fdr_reject_global" in df.columns else "fdr_reject"
    if fdr_col not in df.columns:
        logger.warning("TSV file missing FDR column (fdr_reject_global or fdr_reject)")
        return None

    global_fdr_map: Dict[Tuple[str, str, float, float, str], bool] = {}

    for _, row in df.iterrows():
        try:
            condition = str(row["condition"]).strip()
            band = str(row["band"]).strip()
            channel = str(row["channel"]).strip()

            if pd.isna(row["time_start"]) or pd.isna(row["time_end"]):
                continue

            time_start = float(row["time_start"])
            time_end = float(row["time_end"])

            if pd.isna(row.get(fdr_col)):
                fdr_reject = False
            else:
                fdr_reject = bool(row[fdr_col])

            if not condition or not band or not channel:
                continue

            key = (condition, band, time_start, time_end, channel)
            global_fdr_map[key] = fdr_reject
        except (ValueError, TypeError):
            continue

    if not global_fdr_map:
        logger.warning(f"No valid global FDR entries found in {tsv_path.name}")
        return None

    logger.debug(f"Loaded global FDR for {len(global_fdr_map)} entries from {tsv_path.name}")
    return global_fdr_map


def plot_temporal_correlation_topomaps_by_pain(
    subject: str,
    task: str,
    plots_dir: Path,
    stats_dir: Path,
    config,
    logger: logging.Logger,
    use_spearman: bool = True,
) -> None:
    use_spearman_suffix = "_spearman" if use_spearman else "_pearson"
    data_path = stats_dir / f"temporal_correlations_by_pain{use_spearman_suffix}.npz"

    if not data_path.exists():
        logger.warning(f"Temporal correlation data not found: {data_path}")
        return

    logger.info("Plotting temporal correlation topomaps by pain condition...")

    global_fdr_map = _load_global_fdr_for_temporal_correlations(stats_dir, use_spearman, logger)
    use_global_fdr = global_fdr_map is not None

    data = np.load(data_path, allow_pickle=True)
    info = data.get("info", None)
    if info is None:
        logger.warning("Info not found in data file")
        return

    if isinstance(info, np.ndarray) and info.dtype == object:
        info = info.item()

    ch_names = data.get("ch_names", None)
    if ch_names is None:
        logger.warning("Channel names not found in data file")
        return

    if isinstance(ch_names, np.ndarray):
        ch_names = ch_names.tolist()

    if "pain" not in data or "non_pain" not in data:
        logger.warning("Pain/non-pain data not found in file")
        return

    info_ch_names = info["ch_names"]
    if len(ch_names) != len(info_ch_names) or set(ch_names) != set(info_ch_names):
        picks = mne.pick_channels(info_ch_names, include=ch_names, exclude=[])
        info = mne.pick_info(info, picks)

    viz_params = get_viz_params(config)
    font_sizes = get_font_sizes()
    sig_text = get_sig_marker_text(config)

    result_pain = data["pain"].item()
    result_non = data["non_pain"].item()

    band_names = result_pain["band_names"]
    band_ranges = result_pain["band_ranges"]
    window_starts = result_pain["window_starts"]
    window_ends = result_pain["window_ends"]
    n_windows = len(window_starts)

    n_rows = 2

    all_corr_data = []
    for result in [result_pain, result_non]:
        for band_idx in range(len(band_names)):
            corr_data = result["correlations"][band_idx]
            all_corr_data.extend([c for c in corr_data.flatten() if np.isfinite(c)])

    vabs_corr = robust_sym_vlim(all_corr_data) if all_corr_data else 0.6

    plot_cfg = get_plot_config(config) if config else None
    tfr_config = plot_cfg.plot_type_configs.get("tfr", {}) if plot_cfg else {}
    topomap_config = tfr_config.get("topomap", {})
    tfr_specific = topomap_config.get("tfr_specific", {})
    hspace = tfr_specific.get("hspace", 0.25)
    wspace = tfr_specific.get("wspace", 1.2)

    fig_size_per_col_large = (
        plot_cfg.get_figure_size("tfr_per_col_large", plot_type="tfr")[0]
        if plot_cfg
        else 10.0
    )
    fig_size_per_row_large = (
        plot_cfg.get_figure_size("tfr_per_row_large", plot_type="tfr")[1]
        if plot_cfg
        else 10.0
    )

    for band_idx, band_name in enumerate(band_names):
        fmin, fmax = band_ranges[band_idx]
        freq_label = f"{band_name} ({fmin:.0f}-{fmax:.0f}Hz)"

        fig, axes = plt.subplots(
            n_rows,
            n_windows,
            figsize=(fig_size_per_col_large * n_windows, fig_size_per_row_large * n_rows),
            squeeze=False,
            gridspec_kw={"hspace": hspace, "wspace": wspace},
        )

        row_labels = ["Non-Pain", "Pain"]
        results = [result_non, result_pain]

        for row_idx, (row_label, result) in enumerate(zip(row_labels, results)):
            correlations = result["correlations"][band_idx]
            p_values = result["p_values"][band_idx]
            p_corrected = result["p_corrected"][band_idx]

            condition_name = "non_pain" if row_idx == 0 else "pain"

            axes[row_idx, 0].set_ylabel(
                f"{row_label}\n{freq_label}", fontsize=font_sizes["ylabel"], labelpad=10
            )

            for col, (tmin_win, tmax_win) in enumerate(zip(window_starts, window_ends)):
                if row_idx == 0:
                    time_label = f"{tmin_win:.2f}s"
                    axes[row_idx, col].set_title(
                        time_label, fontsize=font_sizes["title"], pad=12, y=1.07
                    )

                corr_data = correlations[col, :]
                p_uncorr = p_values[col, :]
                p_fdr = p_corrected[col, :]

                sig_mask_uncorr = (p_uncorr < 0.05) & np.isfinite(p_uncorr)

                if use_global_fdr and global_fdr_map is not None:
                    sig_mask_fdr = np.zeros(len(ch_names), dtype=bool)
                    for ch_idx, ch_name in enumerate(ch_names):
                        key = (condition_name, band_name, tmin_win, tmax_win, ch_name)
                        if key in global_fdr_map:
                            sig_mask_fdr[ch_idx] = global_fdr_map[key]
                else:
                    sig_mask_fdr = (p_fdr < 0.05) & np.isfinite(p_fdr)

                plot_topomap_on_ax(
                    axes[row_idx, col],
                    corr_data,
                    info,
                    vmin=-vabs_corr,
                    vmax=+vabs_corr,
                    mask=sig_mask_fdr,
                    mask_params=dict(
                        marker="o",
                        markerfacecolor="green",
                        markeredgecolor="green",
                        markersize=4,
                    ),
                    config=config,
                )

                if sig_mask_uncorr.sum() > 0:
                    uncorr_chs = np.where(sig_mask_uncorr & ~sig_mask_fdr)[0]
                    if len(uncorr_chs) > 0:
                        try:
                            from mne.channels.layout import _find_topomap_coords

                            pos = _find_topomap_coords(info, picks=None)
                            axes[row_idx, col].plot(
                                pos[uncorr_chs, 0],
                                pos[uncorr_chs, 1],
                                "o",
                                markerfacecolor="white",
                                markeredgecolor="black",
                                markersize=4,
                                markeredgewidth=1,
                                zorder=10,
                            )
                        except Exception:
                            pass

                _add_correlation_roi_annotations(
                    axes[row_idx, col],
                    corr_data,
                    p_uncorr,
                    p_fdr,
                    info,
                    config=config,
                    fdr_alpha=0.05,
                )

        create_difference_colorbar(
            fig, axes, vabs_corr, viz_params["topo_cmap"], label="Correlation coefficient"
        )

        window_label = f"{window_starts[0]:.1f}–{window_ends[-1]:.1f}s; {n_windows} windows"
        method_name = "Spearman" if use_spearman else "Pearson"
        fig.suptitle(
            (
                f"Temporal correlation topomaps by pain condition ({band_name}, {window_label})\n"
                f"{method_name} correlation, vlim ±{vabs_corr:.2f}{sig_text}\n"
            ),
            fontsize=font_sizes["suptitle"],
            y=0.995,
        )

        topomap_dir = plots_dir / "topomaps"
        ensure_dir(topomap_dir)
        filename = f"sub-{subject}_temporal_correlations_by_pain_{band_name}.png"
        save_fig(
            fig,
            topomap_dir / filename,
            formats=plot_cfg.formats if plot_cfg else ["png", "svg"],
            dpi=plot_cfg.dpi if plot_cfg else None,
            bbox_inches=plot_cfg.bbox_inches if plot_cfg else "tight",
            pad_inches=plot_cfg.pad_inches if plot_cfg else None,
            footer=_get_behavior_footer(config),
        )
        plt.close(fig)


def _get_behavioral_config(plot_cfg):
    return plot_cfg.plot_type_configs.get("behavioral", {})


def _add_colorbar(fig, axes, successful_plots, config=None):
    if not successful_plots:
        return

    config = config or _get_default_config()
    plot_cfg = get_plot_config(config)
    behavioral_config = _get_behavioral_config(plot_cfg)
    colorbar_config = behavioral_config.get("colorbar", {})

    width_fraction = colorbar_config.get("width_fraction", 0.55)
    left_offset_fraction = colorbar_config.get("left_offset_fraction", 0.225)
    bottom_offset = colorbar_config.get("bottom_offset", 0.12)
    min_bottom = colorbar_config.get("min_bottom", 0.04)
    height = colorbar_config.get("height", 0.028)
    label_fontsize = colorbar_config.get("label_fontsize", 11)
    tick_fontsize = colorbar_config.get("tick_fontsize", 9)
    tick_pad = colorbar_config.get("tick_pad", 2)

    left = min(ax.get_position().x0 for ax in axes)
    right = max(ax.get_position().x1 for ax in axes)
    bottom = min(ax.get_position().y0 for ax in axes)
    span = right - left
    cb_width = width_fraction * span
    cb_left = left + left_offset_fraction * span
    cb_bottom = max(min_bottom, bottom - bottom_offset)
    cax = fig.add_axes([cb_left, cb_bottom, cb_width, height])
    cbar = fig.colorbar(successful_plots[-1], cax=cax, orientation="horizontal")
    cbar.set_label("Spearman correlation (ρ)", fontweight="bold", fontsize=label_fontsize)
    cbar.ax.tick_params(pad=tick_pad, labelsize=tick_fontsize)


def plot_significant_correlations_topomap(
    pow_df: pd.DataFrame,
    y: pd.Series,
    bands: List[str],
    info: mne.Info,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config=None,
    alpha: float = 0.05,
) -> None:
    config = config or _get_default_config()
    plot_cfg = get_plot_config(config)
    behavioral_config = _get_behavioral_config(plot_cfg)
    power_prefix = behavioral_config.get("power_prefix", "pow_")
    min_samples_for_plot = plot_cfg.validation.get("min_samples_for_plot", 5)

    bands_with_data = []
    for band in bands:
        ch_names, correlations, p_values = compute_band_correlations(
            pow_df,
            y,
            band,
            power_prefix=power_prefix,
            min_samples=min_samples_for_plot,
        )
        if len(ch_names) == 0:
            continue

        sig_mask = p_values < alpha
        bands_with_data.append(
            {
                "band": band,
                "channels": ch_names,
                "correlations": correlations,
                "p_values": p_values,
                "significant_mask": sig_mask,
            }
        )

    if not bands_with_data:
        logger.warning("No significant correlations found across any frequency band")
        return

    topomap_config = behavioral_config.get("topomap", {})
    tfr_config = plot_cfg.plot_type_configs.get("tfr", {}) if plot_cfg else {}
    topomap_config_tfr = tfr_config.get("topomap", {})
    tfr_specific = topomap_config_tfr.get("tfr_specific", {})
    wspace = tfr_specific.get("wspace", 1.2)

    fig_size_per_col_large = (
        plot_cfg.get_figure_size("tfr_per_col_large", plot_type="tfr")[0]
        if plot_cfg
        else 10.0
    )
    fig_size_per_row_large = (
        plot_cfg.get_figure_size("tfr_per_row_large", plot_type="tfr")[1]
        if plot_cfg
        else 10.0
    )

    n_bands = len(bands_with_data)
    fig, axes = plt.subplots(
        1,
        n_bands,
        figsize=(fig_size_per_col_large * n_bands, fig_size_per_row_large),
        squeeze=False,
        gridspec_kw={"wspace": wspace},
    )
    axes = axes[0]

    vmax = compute_correlation_vmax(bands_with_data)
    successful_plots = []

    for i, band_data in enumerate(bands_with_data):
        ax = axes[i]
        topo_data, topo_mask = prepare_topomap_correlation_data(band_data, info)

        picks = mne.pick_types(info, meg=False, eeg=True, exclude="bads")
        if len(picks) == 0:
            continue

        plot_cfg = get_plot_config(config)
        topomap_plot_config = plot_cfg.plot_type_configs.get("topomap", {})
        colormap = topomap_plot_config.get("colormap", "RdBu_r")
        contours = topomap_plot_config.get("contours", 6)

        im, _ = mne.viz.plot_topomap(
            topo_data[picks],
            mne.pick_info(info, picks),
            axes=ax,
            show=False,
            cmap=colormap,
            vlim=(-vmax, vmax),
            contours=contours,
            mask=topo_mask[picks],
            mask_params=dict(
                marker=topomap_config.get("mask_marker", "o"),
                markerfacecolor=topomap_config.get("mask_markerfacecolor", "white"),
                markeredgecolor=topomap_config.get("mask_markeredgecolor", "black"),
                linewidth=topomap_config.get("mask_linewidth", 1),
                markersize=topomap_config.get("mask_markersize", 6),
            ),
        )

        successful_plots.append(im)

        n_sig = topo_mask[picks].sum()
        n_total = len([ch for ch in band_data["channels"] if ch in info["ch_names"]])
        title_fontsize = topomap_config.get("title_fontsize", 12)
        title_pad = topomap_config.get("title_pad", 10)
        ax.set_title(
            f"{band_data['band'].upper()}\n{n_sig}/{n_total} significant",
            fontweight="bold",
            fontsize=title_fontsize,
            pad=title_pad,
        )

    suptitle_fontsize = topomap_config.get("suptitle_fontsize", 14)
    suptitle_y = topomap_config.get("suptitle_y", 1.02)
    plt.suptitle(
        f"Significant EEG-Pain Correlations (p < {alpha})\nSubject {subject}",
        fontweight="bold",
        fontsize=suptitle_fontsize,
        y=suptitle_y,
    )

    _add_colorbar(fig, axes, successful_plots, config)

    tight_layout_rect = topomap_config.get("tight_layout_rect", [0, 0.15, 1, 1])
    topomap_dir = save_dir / "topomaps"
    ensure_dir(topomap_dir)
    save_fig(
        fig,
        topomap_dir / f"sub-{subject}_significant_correlations_topomap",
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
        footer=_get_behavior_footer(config),
        tight_layout_rect=tight_layout_rect,
    )
    plt.close(fig)

    logger.info(
        "Created topomaps for %d frequency bands: %s",
        len(bands_with_data),
        [bd["band"] for bd in bands_with_data],
    )
