from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from eeg_pipeline.infra.paths import ensure_dir
from eeg_pipeline.infra.tsv import read_tsv
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.core.annotations import find_annotation_x_position, get_sig_marker_text
from eeg_pipeline.plotting.core.colorbars import create_difference_colorbar
from eeg_pipeline.plotting.core.utils import get_font_sizes
from eeg_pipeline.plotting.features.utils import get_fdr_alpha
from eeg_pipeline.plotting.io.figures import (
    get_behavior_footer as _get_behavior_footer,
    get_default_config as _get_default_config,
    get_viz_params,
    plot_topomap_on_ax,
    robust_sym_vlim,
    save_fig,
)
from eeg_pipeline.utils.analysis.stats import compute_band_correlations, compute_correlation_vmax
from eeg_pipeline.utils.analysis.tfr import build_roi_channel_mask, build_rois_from_info
from eeg_pipeline.utils.config.loader import get_config_value
from eeg_pipeline.utils.data.manipulation import prepare_topomap_correlation_data

DEFAULT_ALPHA = 0.05
DEFAULT_UNCORRECTED_ALPHA = 0.05
DEFAULT_ANNOTATION_Y_START = 0.98
DEFAULT_ANNOTATION_LINE_HEIGHT = 0.045
DEFAULT_ANNOTATION_MIN_SPACING = 0.03
DEFAULT_ANNOTATION_SPACING_MULTIPLIER = 0.3
DEFAULT_FIGURE_SIZE = 10.0


def _compute_roi_statistics(
    corr_data: np.ndarray,
    p_uncorr: Optional[np.ndarray],
    p_fdr: Optional[np.ndarray],
    ch_names: List[str],
    roi_map: Dict[str, List[str]],
) -> List[Tuple[str, float, Optional[float], Optional[float]]]:
    """Compute mean correlation and p-values for each ROI."""
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
        if p_uncorr is not None:
            roi_p_uncorr_vals = p_uncorr[mask_vec]
            roi_p_uncorr_finite = roi_p_uncorr_vals[np.isfinite(roi_p_uncorr_vals)]
            if len(roi_p_uncorr_finite) > 0:
                roi_p_uncorr = np.nanmin(roi_p_uncorr_finite)

        roi_p_fdr = None
        if p_fdr is not None:
            roi_p_fdr_vals = p_fdr[mask_vec]
            roi_p_fdr_finite = roi_p_fdr_vals[np.isfinite(roi_p_fdr_vals)]
            if len(roi_p_fdr_finite) > 0:
                roi_p_fdr = np.nanmin(roi_p_fdr_finite)

        annotations.append((roi, mean_corr, roi_p_uncorr, roi_p_fdr))
    return annotations


def _format_roi_label(
    roi: str,
    mean_corr: float,
    roi_p_uncorr: Optional[float],
    roi_p_fdr: Optional[float],
    fdr_alpha: float,
) -> str:
    """Format ROI annotation label with correlation and significance."""
    label = f"{roi}: r={mean_corr:+.2f}"
    if roi_p_fdr is not None and np.isfinite(roi_p_fdr) and roi_p_fdr < fdr_alpha:
        label += f" (q={roi_p_fdr:.3f})"
    elif roi_p_uncorr is not None and np.isfinite(roi_p_uncorr) and roi_p_uncorr < DEFAULT_UNCORRECTED_ALPHA:
        label += f" (p={roi_p_uncorr:.3f})"
    return label


def _get_annotation_config(plot_cfg: Optional[Any]) -> Dict[str, float]:
    """Extract annotation positioning configuration."""
    if plot_cfg is None:
        return {
            "y_start": DEFAULT_ANNOTATION_Y_START,
            "line_height": DEFAULT_ANNOTATION_LINE_HEIGHT,
            "min_spacing": DEFAULT_ANNOTATION_MIN_SPACING,
            "spacing_multiplier": DEFAULT_ANNOTATION_SPACING_MULTIPLIER,
        }

    tfr_config = plot_cfg.plot_type_configs.get("tfr", {})
    return {
        "y_start": tfr_config.get("annotation_y_start", DEFAULT_ANNOTATION_Y_START),
        "line_height": tfr_config.get("annotation_line_height", DEFAULT_ANNOTATION_LINE_HEIGHT),
        "min_spacing": tfr_config.get("annotation_min_spacing", DEFAULT_ANNOTATION_MIN_SPACING),
        "spacing_multiplier": tfr_config.get("annotation_spacing_multiplier", DEFAULT_ANNOTATION_SPACING_MULTIPLIER),
    }


def _add_correlation_roi_annotations(
    ax,
    corr_data: np.ndarray,
    p_uncorr: Optional[np.ndarray],
    p_fdr: Optional[np.ndarray],
    info: mne.Info,
    config: Optional[Any] = None,
    roi_map: Optional[Dict[str, List[str]]] = None,
    fdr_alpha: float = DEFAULT_ALPHA,
) -> None:
    """Add ROI correlation annotations to a topomap axis."""
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
    font_sizes = get_font_sizes(plot_cfg)
    annotation_fontsize = font_sizes["annotation"]
    annotation_config = _get_annotation_config(plot_cfg)

    x_pos_ax = find_annotation_x_position(ax, plot_cfg)
    y_pos_ax = annotation_config["y_start"]
    line_height = annotation_config["line_height"]
    min_spacing = annotation_config["min_spacing"]
    spacing_multiplier = annotation_config["spacing_multiplier"]

    annotations = _compute_roi_statistics(corr_data, p_uncorr, p_fdr, ch_names, roi_map)

    for i, (roi, mean_corr, roi_p_uncorr, roi_p_fdr) in enumerate(annotations):
        if not np.isfinite(mean_corr):
            continue

        label = _format_roi_label(roi, mean_corr, roi_p_uncorr, roi_p_fdr, fdr_alpha)

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
            spacing_ax = min_spacing + (line_height * spacing_multiplier)
            y_pos_ax -= (line_height + spacing_ax)


def _get_correlation_suffix(use_spearman: bool) -> str:
    """Get file suffix based on correlation method."""
    return "_spearman" if use_spearman else "_pearson"


def _validate_fdr_dataframe(df: pd.DataFrame, logger: logging.Logger) -> Optional[str]:
    """Validate FDR dataframe structure and return FDR column name if valid."""
    required_cols = ["condition", "band", "time_start", "time_end", "channel"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"TSV file missing required columns: {missing_cols}")
        return None

    fdr_col = "fdr_reject_global" if "fdr_reject_global" in df.columns else "fdr_reject"
    if fdr_col not in df.columns:
        logger.warning("TSV file missing FDR column (fdr_reject_global or fdr_reject)")
        return None

    return fdr_col


def _parse_fdr_row(
    row: pd.Series,
    fdr_col: str,
) -> Optional[Tuple[Tuple[str, str, float, float, str], bool]]:
    """Parse a single row from FDR dataframe into key-value pair."""
    try:
        condition = str(row["condition"]).strip()
        band = str(row["band"]).strip()
        channel = str(row["channel"]).strip()

        if not condition or not band or not channel:
            return None

        if pd.isna(row["time_start"]) or pd.isna(row["time_end"]):
            return None

        time_start = float(row["time_start"])
        time_end = float(row["time_end"])

        fdr_reject = False if pd.isna(row.get(fdr_col)) else bool(row[fdr_col])

        key = (condition, band, time_start, time_end, channel)
        return (key, fdr_reject)
    except (ValueError, TypeError):
        return None


def _load_global_fdr_for_temporal_correlations(
    stats_dir: Path,
    use_spearman: bool,
    logger: logging.Logger,
) -> Optional[Dict[Tuple[str, str, float, float, str], bool]]:
    """Load global FDR correction map from temporal correlations TSV file."""
    suffix = _get_correlation_suffix(use_spearman)
    tsv_path = stats_dir / "temporal_correlations" / f"temporal_correlations{suffix}.tsv"
    
    if not tsv_path.exists():
        logger.debug(f"TSV file not found for global FDR: {tsv_path.name}")
        return None

    try:
        df = read_tsv(tsv_path)
    except Exception as e:
        logger.warning(f"Failed to load TSV for global FDR: {e}")
        return None

    if df.empty:
        logger.warning(f"TSV file is empty: {tsv_path}")
        return None

    fdr_col = _validate_fdr_dataframe(df, logger)
    if fdr_col is None:
        return None

    global_fdr_map: Dict[Tuple[str, str, float, float, str], bool] = {}
    for _, row in df.iterrows():
        parsed = _parse_fdr_row(row, fdr_col)
        if parsed is not None:
            key, fdr_reject = parsed
            global_fdr_map[key] = fdr_reject

    if not global_fdr_map:
        logger.warning(f"No valid global FDR entries found in {tsv_path.name}")
        return None

    logger.debug(f"Loaded global FDR for {len(global_fdr_map)} entries from {tsv_path.name}")
    return global_fdr_map


def _load_temporal_correlation_data(
    stats_dir: Path,
    use_spearman: bool,
    logger: logging.Logger,
) -> Optional[Dict[str, Any]]:
    """Load temporal correlation data from NPZ file."""
    suffix = _get_correlation_suffix(use_spearman)
    
    path = stats_dir / "temporal_correlations" / f"temporal_correlations_by_condition{suffix}.npz"
    if path.exists():
        data = np.load(path, allow_pickle=True)
        return dict(data)

    logger.warning(f"Temporal correlation data not found: {path}")
    return None


def _extract_info_from_data(data: Dict[str, Any], ch_names: List[str], logger: logging.Logger) -> Optional[mne.Info]:
    """Extract and validate MNE Info object from loaded data."""
    info = data.get("info", None)
    if info is None:
        logger.warning("Info not found in data file")
        return None

    if isinstance(info, np.ndarray) and info.dtype == object:
        info = info.item()

    info_ch_names = info["ch_names"]
    if len(ch_names) != len(info_ch_names) or set(ch_names) != set(info_ch_names):
        picks = mne.pick_channels(info_ch_names, include=ch_names, exclude=[])
        info = mne.pick_info(info, picks)

    return info


def _validate_temporal_results(
    condition_results: Dict[str, Dict[str, Any]],
    data_path: Path,
    subject: str,
    logger: logging.Logger,
) -> bool:
    """Validate that all condition results contain required keys."""
    required_result_keys = [
        "correlations",
        "p_values",
        "p_corrected",
        "band_names",
        "band_ranges",
        "window_starts",
        "window_ends",
    ]
    for condition_name, result in condition_results.items():
        if not isinstance(result, dict):
            continue
        missing = [k for k in required_result_keys if k not in result]
        if missing:
            logger.error(
                "Temporal correlation stats file is missing required fields (%s) for condition '%s': %s. "
                "Regenerate temporal stats (this overwrites stale outputs) via: "
                "python -m eeg_pipeline.cli.main behavior compute --subject %s --computations temporal",
                data_path.name,
                condition_name,
                ", ".join(missing),
                subject,
            )
            return False
    return True


def _compute_correlation_vlim(
    condition_results: Dict[str, Dict[str, Any]],
    band_names: List[str],
) -> float:
    """Compute symmetric vlim for correlation topomaps across all conditions."""
    all_corr_data = []
    for result in condition_results.values():
        if not isinstance(result, dict) or "correlations" not in result:
            continue
        for band_idx in range(len(band_names)):
            corr_data = result["correlations"][band_idx]
            all_corr_data.extend([c for c in corr_data.flatten() if np.isfinite(c)])

    default_vlim = 0.6
    return robust_sym_vlim(all_corr_data) if all_corr_data else default_vlim


def _get_figure_layout_config(plot_cfg: Optional[Any]) -> Dict[str, Any]:
    """Extract figure layout configuration."""
    if plot_cfg is None:
        return {
            "hspace": 0.25,
            "wspace": 1.2,
            "fig_size_per_col": DEFAULT_FIGURE_SIZE,
            "fig_size_per_row": DEFAULT_FIGURE_SIZE,
        }

    tfr_config = plot_cfg.plot_type_configs.get("tfr", {})
    topomap_config = tfr_config.get("topomap", {})
    tfr_specific = topomap_config.get("tfr_specific", {})

    fig_size_per_col = (
        plot_cfg.get_figure_size("tfr_per_col_large", plot_type="tfr")[0]
        if plot_cfg
        else DEFAULT_FIGURE_SIZE
    )
    fig_size_per_row = (
        plot_cfg.get_figure_size("tfr_per_row_large", plot_type="tfr")[1]
        if plot_cfg
        else DEFAULT_FIGURE_SIZE
    )

    return {
        "hspace": tfr_specific.get("hspace", 0.25),
        "wspace": tfr_specific.get("wspace", 1.2),
        "fig_size_per_col": fig_size_per_col,
        "fig_size_per_row": fig_size_per_row,
    }


def _get_condition_labels(config: Optional[Any]) -> List[str]:
    """Extract condition labels from config."""
    labels_spec = get_config_value(config, "plotting.comparisons.comparison_labels", None)
    if isinstance(labels_spec, (list, tuple)) and len(labels_spec) >= 2:
        return [str(labels_spec[0]), str(labels_spec[1])]
    return ["Condition 1", "Condition 2"]


def _build_global_fdr_mask(
    ch_names: List[str],
    condition_name: str,
    band_name: str,
    tmin_win: float,
    tmax_win: float,
    global_fdr_map: Dict[Tuple[str, str, float, float, str], bool],
) -> np.ndarray:
    """Build significance mask from global FDR map."""
    sig_mask = np.zeros(len(ch_names), dtype=bool)
    for ch_idx, ch_name in enumerate(ch_names):
        key = (condition_name, band_name, tmin_win, tmax_win, ch_name)
        if key in global_fdr_map:
            sig_mask[ch_idx] = global_fdr_map[key]
    return sig_mask


def _plot_uncorrected_markers(
    ax: plt.Axes,
    info: mne.Info,
    uncorr_chs: np.ndarray,
) -> None:
    """Plot markers for uncorrected significant channels."""
    if len(uncorr_chs) == 0:
        return

    try:
        from mne.channels.layout import _find_topomap_coords

        pos = _find_topomap_coords(info, picks=None)
        ax.plot(
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


def _plot_single_band_topomaps(
    band_idx: int,
    band_name: str,
    band_ranges: List[Tuple[float, float]],
    window_starts: np.ndarray,
    window_ends: np.ndarray,
    result_row1: Dict[str, Any],
    result_row2: Dict[str, Any],
    condition_names: List[str],
    info: mne.Info,
    ch_names: List[str],
    row_labels: List[str],
    vabs_corr: float,
    global_fdr_map: Optional[Dict[Tuple[str, str, float, float, str], bool]],
    use_global_fdr: bool,
    config: Optional[Any],
    font_sizes: Dict[str, int],
    layout_config: Dict[str, Any],
    n_windows: int,
) -> Tuple[plt.Figure, np.ndarray]:
    """Create topomap figure for a single frequency band."""
    fmin, fmax = band_ranges[band_idx]
    freq_label = f"{band_name} ({fmin:.0f}-{fmax:.0f}Hz)"

    n_rows = 2
    fig, axes = plt.subplots(
        n_rows,
        n_windows,
        figsize=(
            layout_config["fig_size_per_col"] * n_windows,
            layout_config["fig_size_per_row"] * n_rows,
        ),
        squeeze=False,
        gridspec_kw={"hspace": layout_config["hspace"], "wspace": layout_config["wspace"]},
    )

    results = [result_row1, result_row2]
    fdr_alpha = get_fdr_alpha(config, DEFAULT_ALPHA)

    for row_idx, (row_label, result) in enumerate(zip(row_labels, results)):
        correlations = result["correlations"][band_idx]
        p_values = result["p_values"][band_idx]
        p_corrected = result["p_corrected"][band_idx]
        condition_name = condition_names[row_idx] if row_idx < len(condition_names) else condition_names[0]

        axes[row_idx, 0].set_ylabel(
            f"{row_label}\n{freq_label}",
            fontsize=font_sizes["ylabel"],
            labelpad=10,
        )

        for col, (tmin_win, tmax_win) in enumerate(zip(window_starts, window_ends)):
            if row_idx == 0:
                time_label = f"{tmin_win:.2f}s"
                axes[row_idx, col].set_title(
                    time_label,
                    fontsize=font_sizes["title"],
                    pad=12,
                    y=1.07,
                )

            corr_data = correlations[col, :]
            p_uncorr = p_values[col, :]
            p_fdr = p_corrected[col, :]

            sig_mask_uncorr = (p_uncorr < DEFAULT_UNCORRECTED_ALPHA) & np.isfinite(p_uncorr)

            if use_global_fdr and global_fdr_map is not None:
                sig_mask_fdr = _build_global_fdr_mask(
                    ch_names, condition_name, band_name, tmin_win, tmax_win, global_fdr_map
                )
            else:
                sig_mask_fdr = (p_fdr < fdr_alpha) & np.isfinite(p_fdr)

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
                _plot_uncorrected_markers(axes[row_idx, col], info, uncorr_chs)

            _add_correlation_roi_annotations(
                axes[row_idx, col],
                corr_data,
                p_uncorr,
                p_fdr,
                info,
                config=config,
                fdr_alpha=fdr_alpha,
            )

    return fig, axes


def _save_band_topomap(
    fig: plt.Figure,
    subject: str,
    band_name: str,
    window_starts: np.ndarray,
    window_ends: np.ndarray,
    n_windows: int,
    vabs_corr: float,
    use_spearman: bool,
    sig_text: str,
    plots_dir: Path,
    config: Optional[Any],
    font_sizes: Dict[str, int],
) -> None:
    """Save topomap figure for a single band."""
    window_label = f"{window_starts[0]:.1f}–{window_ends[-1]:.1f}s; {n_windows} windows"
    method_name = "Spearman" if use_spearman else "Pearson"
    fig.suptitle(
        (
            f"Temporal correlation topomaps by condition ({band_name}, {window_label})\n"
            f"{method_name} correlation, vlim ±{vabs_corr:.2f}{sig_text}\n"
        ),
        fontsize=font_sizes["suptitle"],
        y=0.995,
    )

    topomap_dir = plots_dir / "topomaps"
    ensure_dir(topomap_dir)
    filename = f"sub-{subject}_temporal_correlations_by_condition_{band_name}.png"
    plot_cfg = get_plot_config(config) if config else None

    save_fig(
        fig,
        topomap_dir / filename,
        formats=plot_cfg.formats if plot_cfg else ["png", "svg"],
        dpi=plot_cfg.dpi if plot_cfg else None,
        bbox_inches=plot_cfg.bbox_inches if plot_cfg else "tight",
        pad_inches=plot_cfg.pad_inches if plot_cfg else None,
        footer=_get_behavior_footer(config),
        config=config,
    )
    plt.close(fig)


def _extract_condition_results(
    data: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Dict[str, Any]]:
    """Extract condition results from NPZ data."""
    condition_results = {}
    
    if "condition_names" not in data:
        return condition_results
    
    condition_names = data["condition_names"]
    if isinstance(condition_names, np.ndarray):
        condition_names = condition_names.tolist()
    
    for cond_name in condition_names:
        if cond_name in data:
            result = data[cond_name]
            if isinstance(result, np.ndarray) and result.dtype == object:
                result = result.item()
            condition_results[cond_name] = result
    
    return condition_results


def plot_temporal_correlation_topomaps_by_pain(
    subject: str,
    task: str,
    plots_dir: Path,
    stats_dir: Path,
    config: Optional[Any],
    logger: logging.Logger,
    use_spearman: bool = True,
) -> None:
    """Plot temporal correlation topomaps by condition.
    
    Supports user-configurable conditions (not just pain/non-pain).
    Conditions are determined by the temporal.condition_column and 
    temporal.condition_values settings in the config.
    """
    data = _load_temporal_correlation_data(stats_dir, use_spearman, logger)
    if data is None:
        return

    logger.info("Plotting temporal correlation topomaps by condition...")

    global_fdr_map = _load_global_fdr_for_temporal_correlations(stats_dir, use_spearman, logger)
    use_global_fdr = global_fdr_map is not None

    ch_names = data.get("ch_names", None)
    if ch_names is None:
        logger.warning("Channel names not found in data file")
        return

    if isinstance(ch_names, np.ndarray):
        ch_names = ch_names.tolist()

    condition_results = _extract_condition_results(data, logger)
    if not condition_results:
        logger.warning("No condition results found in data file")
        return
    
    condition_names = list(condition_results.keys())
    logger.info(f"Found {len(condition_names)} conditions: {condition_names}")

    info = _extract_info_from_data(data, ch_names, logger)
    if info is None:
        return

    viz_params = get_viz_params(config)
    font_sizes = get_font_sizes()
    sig_text = get_sig_marker_text(config)

    suffix = _get_correlation_suffix(use_spearman)
    data_path = stats_dir / "temporal_correlations" / f"temporal_correlations_by_condition{suffix}.npz"
    
    if not _validate_temporal_results(condition_results, data_path, subject, logger):
        return

    # Use first condition to get band/window metadata
    first_result = list(condition_results.values())[0]
    band_names = first_result["band_names"]
    band_ranges = first_result["band_ranges"]
    window_starts = first_result["window_starts"]
    window_ends = first_result["window_ends"]
    n_windows = len(window_starts)

    vabs_corr = _compute_correlation_vlim(condition_results, band_names)

    plot_cfg = get_plot_config(config) if config else None
    layout_config = _get_figure_layout_config(plot_cfg)
    
    # Get condition labels (use actual condition names if not configured)
    row_labels = _get_condition_labels(config)
    if row_labels is None or len(row_labels) != len(condition_names):
        row_labels = condition_names

    if len(condition_results) >= 2:
        result_row1 = condition_results[condition_names[0]]
        result_row2 = condition_results[condition_names[1]]
    else:
        # Single condition: duplicate for both rows
        result_row1 = result_row2 = list(condition_results.values())[0]
        row_labels = [condition_names[0], condition_names[0]]

    for band_idx, band_name in enumerate(band_names):
        fig, axes = _plot_single_band_topomaps(
            band_idx,
            band_name,
            band_ranges,
            window_starts,
            window_ends,
            result_row1,
            result_row2,
            condition_names,
            info,
            ch_names,
            row_labels,
            vabs_corr,
            global_fdr_map,
            use_global_fdr,
            config,
            font_sizes,
            layout_config,
            n_windows,
        )

        create_difference_colorbar(
            fig, axes, vabs_corr, viz_params["topo_cmap"], label="Correlation coefficient"
        )

        _save_band_topomap(
            fig,
            subject,
            band_name,
            window_starts,
            window_ends,
            n_windows,
            vabs_corr,
            use_spearman,
            sig_text,
            plots_dir,
            config,
            font_sizes,
        )


def _get_behavioral_config(plot_cfg: Optional[Any]) -> Dict[str, Any]:
    """Extract behavioral plotting configuration."""
    if plot_cfg is None:
        return {}
    return plot_cfg.plot_type_configs.get("behavioral", {})


def _get_colorbar_config(plot_cfg: Optional[Any]) -> Dict[str, Any]:
    """Extract colorbar configuration."""
    behavioral_config = _get_behavioral_config(plot_cfg)
    colorbar_config = behavioral_config.get("colorbar", {})
    return {
        "width_fraction": colorbar_config.get("width_fraction", 0.55),
        "left_offset_fraction": colorbar_config.get("left_offset_fraction", 0.225),
        "bottom_offset": colorbar_config.get("bottom_offset", 0.12),
        "min_bottom": colorbar_config.get("min_bottom", 0.04),
        "height": colorbar_config.get("height", 0.028),
        "label_fontsize": colorbar_config.get("label_fontsize", 11),
        "tick_fontsize": colorbar_config.get("tick_fontsize", 9),
        "tick_pad": colorbar_config.get("tick_pad", 2),
    }


def _add_colorbar(
    fig: plt.Figure,
    axes: np.ndarray,
    successful_plots: List[Any],
    config: Optional[Any] = None,
) -> None:
    """Add colorbar to figure with successful plots."""
    if not successful_plots:
        return

    config = config or _get_default_config()
    plot_cfg = get_plot_config(config)
    cb_config = _get_colorbar_config(plot_cfg)

    left = min(ax.get_position().x0 for ax in axes)
    right = max(ax.get_position().x1 for ax in axes)
    bottom = min(ax.get_position().y0 for ax in axes)
    span = right - left

    cb_width = cb_config["width_fraction"] * span
    cb_left = left + cb_config["left_offset_fraction"] * span
    cb_bottom = max(cb_config["min_bottom"], bottom - cb_config["bottom_offset"])

    cax = fig.add_axes([cb_left, cb_bottom, cb_width, cb_config["height"]])
    cbar = fig.colorbar(successful_plots[-1], cax=cax, orientation="horizontal")
    cbar.set_label("Spearman correlation (ρ)", fontweight="bold", fontsize=cb_config["label_fontsize"])
    cbar.ax.tick_params(pad=cb_config["tick_pad"], labelsize=cb_config["tick_fontsize"])


def _compute_bands_with_significant_correlations(
    pow_df: pd.DataFrame,
    y: pd.Series,
    bands: List[str],
    power_prefix: str,
    min_samples: int,
    alpha: float,
) -> List[Dict[str, Any]]:
    """Compute correlations and identify significant bands."""
    bands_with_data = []
    for band in bands:
        ch_names, correlations, p_values = compute_band_correlations(
            pow_df,
            y,
            band,
            min_samples=min_samples,
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
    return bands_with_data


def _get_topomap_plot_config(plot_cfg: Optional[Any]) -> Dict[str, Any]:
    """Extract topomap plotting configuration."""
    if plot_cfg is None:
        return {
            "colormap": "RdBu_r",
            "contours": 6,
            "wspace": 1.2,
            "fig_size_per_col": DEFAULT_FIGURE_SIZE,
            "fig_size_per_row": DEFAULT_FIGURE_SIZE,
        }

    topomap_plot_config = plot_cfg.plot_type_configs.get("topomap", {})
    tfr_config = plot_cfg.plot_type_configs.get("tfr", {})
    topomap_config_tfr = tfr_config.get("topomap", {})
    tfr_specific = topomap_config_tfr.get("tfr_specific", {})

    fig_size_per_col = (
        plot_cfg.get_figure_size("tfr_per_col_large", plot_type="tfr")[0]
        if plot_cfg
        else DEFAULT_FIGURE_SIZE
    )
    fig_size_per_row = (
        plot_cfg.get_figure_size("tfr_per_row_large", plot_type="tfr")[1]
        if plot_cfg
        else DEFAULT_FIGURE_SIZE
    )

    return {
        "colormap": topomap_plot_config.get("colormap", "RdBu_r"),
        "contours": topomap_plot_config.get("contours", 6),
        "wspace": tfr_specific.get("wspace", 1.2),
        "fig_size_per_col": fig_size_per_col,
        "fig_size_per_row": fig_size_per_row,
    }


def _plot_band_topomap(
    ax: plt.Axes,
    band_data: Dict[str, Any],
    info: mne.Info,
    vmax: float,
    topomap_config: Dict[str, Any],
    plot_config: Dict[str, Any],
) -> Optional[Any]:
    """Plot a single band topomap and return image handle."""
    topo_data, topo_mask = prepare_topomap_correlation_data(band_data, info)

    picks = mne.pick_types(info, meg=False, eeg=True, exclude="bads")
    if len(picks) == 0:
        return None

    im, _ = mne.viz.plot_topomap(
        topo_data[picks],
        mne.pick_info(info, picks),
        axes=ax,
        show=False,
        cmap=plot_config["colormap"],
        vlim=(-vmax, vmax),
        contours=plot_config["contours"],
        mask=topo_mask[picks],
        mask_params=dict(
            marker=topomap_config.get("mask_marker", "o"),
            markerfacecolor=topomap_config.get("mask_markerfacecolor", "white"),
            markeredgecolor=topomap_config.get("mask_markeredgecolor", "black"),
            linewidth=topomap_config.get("mask_linewidth", 1),
            markersize=topomap_config.get("mask_markersize", 6),
        ),
    )

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

    return im


def plot_significant_correlations_topomap(
    pow_df: pd.DataFrame,
    y: pd.Series,
    bands: List[str],
    info: mne.Info,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Optional[Any] = None,
    alpha: Optional[float] = None,
) -> None:
    """Plot topomaps showing significant correlations between power and outcome."""
    config = config or _get_default_config()
    plot_cfg = get_plot_config(config)
    behavioral_config = _get_behavioral_config(plot_cfg)

    if alpha is None:
        alpha = get_fdr_alpha(config, DEFAULT_ALPHA)

    power_prefix = behavioral_config.get("power_prefix", "pow_")
    min_samples_for_plot = plot_cfg.validation.get("min_samples_for_plot", 5)

    bands_with_data = _compute_bands_with_significant_correlations(
        pow_df, y, bands, power_prefix, min_samples_for_plot, alpha
    )

    if not bands_with_data:
        logger.warning("No significant correlations found across any frequency band")
        return

    topomap_config = behavioral_config.get("topomap", {})
    plot_config = _get_topomap_plot_config(plot_cfg)

    n_bands = len(bands_with_data)
    fig, axes = plt.subplots(
        1,
        n_bands,
        figsize=(
            plot_config["fig_size_per_col"] * n_bands,
            plot_config["fig_size_per_row"],
        ),
        squeeze=False,
        gridspec_kw={"wspace": plot_config["wspace"]},
    )
    axes = axes[0]

    vmax = compute_correlation_vmax(bands_with_data)
    successful_plots = []

    for i, band_data in enumerate(bands_with_data):
        im = _plot_band_topomap(
            axes[i], band_data, info, vmax, topomap_config, plot_config
        )
        if im is not None:
            successful_plots.append(im)

    suptitle_fontsize = topomap_config.get("suptitle_fontsize", 14)
    suptitle_y = topomap_config.get("suptitle_y", 1.02)
    plt.suptitle(
        f"Significant EEG-Outcome Correlations (p < {alpha})\nSubject {subject}",
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
        config=config,
    )
    plt.close(fig)

    logger.info(
        "Created topomaps for %d frequency bands: %s",
        len(bands_with_data),
        [bd["band"] for bd in bands_with_data],
    )
