"""
Power distribution and PSD plotting functions.

Extracted from plot_features.py for modular organization.
"""

from __future__ import annotations

from pathlib import Path
import textwrap
from typing import Any, Dict, List, Optional, Tuple
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import mne
from mne.viz import plot_topomap

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.plotting.io.figures import get_band_color, save_fig
from eeg_pipeline.plotting.io.figures import get_viz_params, robust_sym_vlim
from eeg_pipeline.utils.data.columns import (
    find_predictor_column_in_events,
)
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.features.roi import (
    get_roi_definitions,
    get_roi_channels,
    extract_channels_from_columns,
)
from eeg_pipeline.utils.analysis.tfr import (
    apply_baseline_and_crop,
)
from eeg_pipeline.utils.config.loader import get_config_value, get_frequency_bands, require_config_value
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon

logger = logging.getLogger(__name__)


###################################################################
# Constants
###################################################################

FDR_ALPHA_DEFAULT = 0.05
MIN_TRIALS_FOR_STATISTICS = 3
MIN_CHANNELS_FOR_TOPO = 3
HEATMAP_TEXT_THRESHOLD = 200
MIN_TRIALS_FOR_VARIABILITY = 5
MIN_EPOCHS_FOR_SEM = 2
BAR_LABEL_OFFSET = 0.02
HISTOGRAM_BINS = 15
MIN_FONT_SIZE = 6
MAX_FONT_SIZE = 10
HEATMAP_EFFECT_ANNOTATION_THRESHOLD = 40
TOPO_MASK_MARKER_SIZE = 4.0
TIMECOURSE_BASELINE_MODE = "ratio"
TOPO_COLORBAR_PAD = 0.02
TOPO_COLORBAR_WIDTH = 0.012
FOREST_BOOTSTRAP_SAMPLES = 2000


###################################################################
# Helper Functions
###################################################################


def _get_comparison_segments(
    power_df: pd.DataFrame,
    config: Any,
    logger: Optional[logging.Logger],
) -> List[str]:
    """Extract comparison segments from config (no auto-detection)."""
    segments = require_config_value(config, "plotting.comparisons.comparison_windows")
    if not isinstance(segments, (list, tuple)) or len(segments) < 2:
        raise ValueError(
            "plotting.comparisons.comparison_windows must be a list/tuple with at least 2 window names "
            f"(got {segments!r})"
        )
    return [str(s) for s in segments]


def _get_condition_color_map(labels: List[str], config: Any) -> Dict[str, Any]:
    """Return a stable condition palette aligned with plot configuration."""
    plot_cfg = get_plot_config(config)
    normalized_labels = [str(label) for label in labels]
    if len(normalized_labels) == 1:
        return {normalized_labels[0]: plot_cfg.get_color("condition_1")}
    if len(normalized_labels) == 2:
        return {
            normalized_labels[0]: plot_cfg.get_color("condition_1"),
            normalized_labels[1]: plot_cfg.get_color("condition_2"),
        }

    palette = sns.color_palette("colorblind", n_colors=len(normalized_labels))
    return {label: palette[index] for index, label in enumerate(normalized_labels)}


def _build_channel_data_array(
    channel_values: Dict[str, float],
    epochs_info: mne.Info,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project channel-level values into epochs_info order."""
    data_array = np.full(len(epochs_info.ch_names), np.nan, dtype=float)
    valid_mask = np.zeros(len(epochs_info.ch_names), dtype=bool)

    for channel_index, channel_name in enumerate(epochs_info.ch_names):
        if channel_name not in channel_values:
            continue
        value = channel_values[channel_name]
        if np.isfinite(value):
            data_array[channel_index] = float(value)
            valid_mask[channel_index] = True

    return data_array, valid_mask


def _compute_shared_topomap_vlim(
    topomap_arrays: List[np.ndarray],
    config: Any,
    *,
    symmetric: bool = True,
) -> Tuple[float, float]:
    """Compute a robust value range for a set of topomap panels."""
    if symmetric:
        vmax = float(robust_sym_vlim(topomap_arrays, config=config))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        return -vmax, vmax

    flat_arrays = [
        np.asarray(array, dtype=float).ravel()
        for array in topomap_arrays
        if array is not None
    ]
    if not flat_arrays:
        return 0.0, 1.0

    finite_data = np.concatenate(flat_arrays)
    finite_data = finite_data[np.isfinite(finite_data)]
    if finite_data.size == 0:
        return 0.0, 1.0

    q_low = float(get_config_value(config, "visualization.robust_vlim.q_low", 0.01))
    q_high = float(get_config_value(config, "visualization.robust_vlim.q_high", 0.99))
    min_value = float(get_config_value(config, "visualization.robust_vlim.min_v", 1e-6))

    vmin = float(np.nanquantile(finite_data, q_low))
    vmax = float(np.nanquantile(finite_data, q_high))
    if vmin >= 0:
        vmin = 0.0
    if vmax <= 0:
        vmax = 0.0

    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax):
        vmax = 1.0
    if vmax < vmin:
        vmax = vmin + max(abs(vmin) * 0.05, min_value)
    if np.isclose(vmin, vmax):
        padding = max(abs(vmax) * 0.05, min_value)
        vmin -= padding
        vmax += padding
    return vmin, vmax


def _resolve_topomap_colormap(
    vmin: float,
    vmax: float,
    *,
    symmetric: bool,
) -> str:
    """Choose a colormap that matches the value range semantics."""
    if symmetric or (vmin < 0 < vmax):
        return "RdBu_r"
    if vmax <= 0:
        return "Blues_r"
    return "Reds"


def _format_condition_display_label(
    label: Optional[str],
    config: Any,
) -> str:
    """Format raw comparison values into readable condition labels."""
    label_text = str(label).strip() if label is not None else ""
    if not label_text:
        return ""

    comparison_column = str(get_config_value(config, "plotting.comparisons.comparison_column", "") or "").strip()
    comparison_values = get_config_value(config, "plotting.comparisons.comparison_values", [])
    comparison_labels = get_config_value(config, "plotting.comparisons.comparison_labels", [])

    configured_label_set = {
        str(item).strip()
        for item in comparison_labels
        if str(item).strip()
    }
    configured_value_set = {
        str(item).strip()
        for item in comparison_values
        if str(item).strip()
    }

    if label_text in configured_label_set or comparison_column == "":
        return label_text
    if label_text in configured_value_set:
        return f"{comparison_column}={label_text}"
    return label_text


def _format_topomap_condition_title_label(
    label: Optional[str],
    config: Any,
) -> str:
    """Format raw comparison values into readable topomap condition titles."""
    return _format_condition_display_label(label, config)


def _add_topomap_colorbar(
    fig: plt.Figure,
    axes: Any,
    image: Any,
    *,
    label: str,
) -> None:
    """Place the topomap colorbar in a dedicated right-side axis."""
    axes_list = list(np.atleast_1d(axes))
    visible_axes = [ax for ax in axes_list if ax.get_visible()]
    if not visible_axes:
        return

    max_right = max(ax.get_position().x1 for ax in visible_axes)
    bottom = min(ax.get_position().y0 for ax in visible_axes)
    top = max(ax.get_position().y1 for ax in visible_axes)

    colorbar_left = min(max_right + TOPO_COLORBAR_PAD, 0.98 - TOPO_COLORBAR_WIDTH)
    colorbar_height = max(top - bottom, 0.1)
    cax = fig.add_axes([colorbar_left, bottom, TOPO_COLORBAR_WIDTH, colorbar_height])
    fig.colorbar(image, cax=cax, label=label)


def _add_topomap_colorbar_in_rect(
    fig: plt.Figure,
    image: Any,
    *,
    rect: Tuple[float, float, float, float],
    label: str,
) -> None:
    """Place a topomap colorbar into an explicit figure rectangle."""
    cax = fig.add_axes(list(rect))
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label(label, fontsize=10)
    colorbar.ax.tick_params(labelsize=8)


def _format_triptych_condition_label(
    label: str,
    config: Any,
) -> str:
    """Return a concise panel title label for a comparison condition."""
    comparison_values = get_config_value(config, "plotting.comparisons.comparison_values", [])
    comparison_labels = get_config_value(config, "plotting.comparisons.comparison_labels", [])

    normalized_label = str(label).strip()
    normalized_value_map = {
        str(value).strip(): str(value).strip()
        for value in comparison_values
        if str(value).strip()
    }
    normalized_label_set = {
        str(value).strip()
        for value in comparison_labels
        if str(value).strip()
    }

    if normalized_label in normalized_label_set:
        return normalized_label
    if normalized_label in normalized_value_map:
        return normalized_value_map[normalized_label]
    return _format_condition_display_label(normalized_label, config)


def _wrap_topomap_panel_title(
    title: str,
    *,
    width: int = 18,
) -> str:
    """Wrap a topomap panel title onto compact lines."""
    return textwrap.fill(str(title), width=width, break_long_words=False)


def _build_topomap_panel(
    channel_values: Dict[str, float],
    epochs_info: mne.Info,
) -> Optional[Tuple[np.ndarray, mne.Info]]:
    """Build one topomap panel from channel values."""
    data_array, present_mask = _build_channel_data_array(channel_values, epochs_info)
    if int(present_mask.sum()) <= MIN_CHANNELS_FOR_TOPO:
        return None
    return data_array[present_mask], mne.pick_info(epochs_info, np.where(present_mask)[0])


def _save_band_topomap_triptych(
    *,
    descriptive_panel_1: Tuple[np.ndarray, mne.Info],
    descriptive_panel_2: Tuple[np.ndarray, mne.Info],
    contrast_panel: Tuple[np.ndarray, np.ndarray, mne.Info],
    band: str,
    subject: str,
    save_path: Path,
    logger: logging.Logger,
    config: Any,
    segment_label: str,
    label1: str,
    label2: str,
    descriptive_value_label: str,
    contrast_value_label: str,
    footer: str,
) -> None:
    """Render a single-band triptych: condition/window 1, 2, and contrast."""
    plot_cfg = get_plot_config(config)
    viz_params = get_viz_params(config)

    desc_data_1, desc_info_1 = descriptive_panel_1
    desc_data_2, desc_info_2 = descriptive_panel_2
    contrast_data, contrast_sig, contrast_info = contrast_panel

    desc_vmin, desc_vmax = _compute_shared_topomap_vlim(
        [desc_data_1, desc_data_2],
        config,
        symmetric=False,
    )
    desc_cmap = _resolve_topomap_colormap(desc_vmin, desc_vmax, symmetric=False)
    contrast_vmin, contrast_vmax = _compute_shared_topomap_vlim(
        [contrast_data],
        config,
        symmetric=True,
    )

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.9))
    fig.patch.set_facecolor("white")

    display_label1 = _format_condition_display_label(label1, config)
    display_label2 = _format_condition_display_label(label2, config)
    panel_label1 = _format_triptych_condition_label(label1, config)
    panel_label2 = _format_triptych_condition_label(label2, config)
    band_title = str(band).upper()
    band_color = get_band_color(band, config)

    for ax in axes:
        ax.set_facecolor("white")

    desc_image = None
    desc_image, _ = plot_topomap(
        desc_data_1,
        desc_info_1,
        axes=axes[0],
        show=False,
        cmap=desc_cmap,
        contours=viz_params.get("topo_contours"),
        vlim=(desc_vmin, desc_vmax),
    )
    axes[0].set_title(
        _wrap_topomap_panel_title(panel_label1),
        fontsize=10,
        fontweight="bold",
        pad=12,
    )

    plot_topomap(
        desc_data_2,
        desc_info_2,
        axes=axes[1],
        show=False,
        cmap=desc_cmap,
        contours=viz_params.get("topo_contours"),
        vlim=(desc_vmin, desc_vmax),
    )
    axes[1].set_title(
        _wrap_topomap_panel_title(panel_label2),
        fontsize=10,
        fontweight="bold",
        pad=12,
    )

    contrast_image, _ = plot_topomap(
        contrast_data,
        contrast_info,
        axes=axes[2],
        show=False,
        cmap="RdBu_r",
        contours=viz_params.get("topo_contours"),
        vlim=(contrast_vmin, contrast_vmax),
        mask=contrast_sig,
        mask_params=_build_topomap_mask_params(config),
    )
    axes[2].set_title(
        _wrap_topomap_panel_title(f"{panel_label2} - {panel_label1}"),
        fontsize=10,
        fontweight="bold",
        pad=12,
    )

    if desc_image is not None:
        _add_topomap_colorbar_in_rect(
            fig,
            desc_image,
            rect=(0.89, 0.56, TOPO_COLORBAR_WIDTH, 0.28),
            label=descriptive_value_label,
        )
    _add_topomap_colorbar_in_rect(
        fig,
        contrast_image,
        rect=(0.89, 0.18, TOPO_COLORBAR_WIDTH, 0.28),
        label=contrast_value_label,
    )

    fig.suptitle(
        f"{band_title} topomap comparison | {segment_label}",
        fontsize=plot_cfg.font.figure_title,
        fontweight="bold",
        color=band_color,
        y=0.97,
    )
    save_fig(
        fig,
        save_path,
        footer=(
            f"{footer} | Comparison: {display_label2} - {display_label1}"
        ),
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
        tight_layout_rect=(0, 0.04, 0.87, 0.94),
        config=config,
    )
    logger.debug("Saved band topomap triptych: %s", save_path.name)


def _build_topomap_mask_params(config: Any) -> Dict[str, Any]:
    """Resolve readable significance mask styling for publication plots."""
    viz_params = get_viz_params(config)
    mask_params = dict(viz_params.get("sig_mask_params", {}))
    if "markersize" not in mask_params:
        mask_params["markersize"] = TOPO_MASK_MARKER_SIZE
    if "linewidth" in mask_params and "markeredgewidth" not in mask_params:
        mask_params["markeredgewidth"] = mask_params.pop("linewidth")
    else:
        mask_params.pop("linewidth", None)
    mask_params["linestyle"] = "none"
    return mask_params


def _style_publication_axis(ax: Any) -> None:
    """Apply consistent manuscript-style axis styling."""
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


def _annotate_frequency_bands(
    ax: Any,
    freq_bands: Dict[str, Tuple[float, float]],
    max_frequency: float,
    config: Any,
) -> None:
    """Add restrained frequency-band shading and labels to a PSD axis."""
    for band_name, (fmin, fmax) in freq_bands.items():
        if fmin >= max_frequency:
            continue
        fmax_clipped = min(fmax, max_frequency)
        ax.axvspan(
            fmin,
            fmax_clipped,
            alpha=0.035,
            color=get_band_color(str(band_name), config),
            linewidth=0,
            zorder=0,
        )
        band_midpoint = 0.5 * (fmin + fmax_clipped)
        if band_midpoint >= max_frequency:
            continue
        ax.text(
            band_midpoint,
            0.985,
            str(band_name).capitalize(),
            transform=ax.get_xaxis_transform(),
            fontsize=7,
            ha="center",
            va="top",
            color="0.45",
        )


def _get_comparison_rois(
    config: Any,
    rois: Dict[str, Any],
) -> List[str]:
    """Determine which ROIs to plot for comparisons."""
    from eeg_pipeline.utils.config.loader import get_config_value
    
    comp_rois = get_config_value(config, "plotting.comparisons.comparison_rois", [])
    if comp_rois:
        roi_names = []
        for r in comp_rois:
            if r.lower() == "all":
                if "all" not in roi_names:
                    roi_names.append("all")
            elif r in rois:
                roi_names.append(r)
        return roi_names
    
    roi_names = ["all"]
    if rois:
        roi_names.extend(list(rois.keys()))
    return roi_names


def _get_power_columns_for_roi(
    power_df: pd.DataFrame,
    segment: str,
    band: str,
    roi_channels: List[str],
) -> List[str]:
    """Get power columns for a specific segment, band, and ROI channels."""
    roi_set = set(roi_channels)
    cols = []
    for c in power_df.columns:
        parsed = NamingSchema.parse(str(c))
        if not (parsed.get("valid") and parsed.get("group") == "power"):
            continue
        if str(parsed.get("segment") or "") != segment:
            continue
        if str(parsed.get("band") or "") != band:
            continue
        channel_id = str(parsed.get("identifier") or "")
        if channel_id and channel_id not in roi_set:
            continue
        cols.append(c)
    return cols


def _extract_band_data_for_roi(
    power_df: pd.DataFrame,
    bands: List[str],
    segments: List[str],
    roi_channels: List[str],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Extract power data by band for two segments within an ROI."""
    roi_set = set(roi_channels)
    data_by_band = {}
    
    for band in bands:
        cols1, cols2 = [], []
        for c in power_df.columns:
            parsed = NamingSchema.parse(str(c))
            if not (parsed.get("valid") and parsed.get("group") == "power"):
                continue
            channel_id = str(parsed.get("identifier") or "")
            if channel_id and channel_id not in roi_set:
                continue
            if str(parsed.get("segment") or "") == segments[0] and str(parsed.get("band") or "") == band:
                cols1.append(c)
            if str(parsed.get("segment") or "") == segments[1] and str(parsed.get("band") or "") == band:
                cols2.append(c)
        
        if cols1 and cols2:
            s1 = power_df[cols1].mean(axis=1)
            s2 = power_df[cols2].mean(axis=1)
            valid_mask = s1.notna() & s2.notna()
            v1, v2 = s1[valid_mask].values, s2[valid_mask].values
            if len(v1) > 0:
                data_by_band[band] = (v1, v2)
    
    return data_by_band


def _extract_multi_segment_data_for_roi(
    power_df: pd.DataFrame,
    bands: List[str],
    segments: List[str],
    roi_channels: List[str],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Extract power data by band for multiple segments within an ROI.
    
    Returns:
        Dict mapping band -> {segment_name -> values array}
    """
    from .utils import extract_multi_segment_data
    return extract_multi_segment_data(
        df=power_df,
        group="power",
        bands=bands,
        segments=segments,
        identifiers=roi_channels,
    )


def _extract_column_comparison_data(
    power_df: pd.DataFrame,
    bands: List[str],
    seg_name: str,
    roi_channels: List[str],
) -> Dict[int, pd.Series]:
    """Extract power data by band for column comparison within an ROI."""
    roi_set = set(roi_channels)
    cell_data = {}
    
    for col_idx, band in enumerate(bands):
        cols = []
        for c in power_df.columns:
            parsed = NamingSchema.parse(str(c))
            if not (parsed.get("valid") and parsed.get("group") == "power"):
                continue
            channel_id = str(parsed.get("identifier") or "")
            if channel_id and channel_id not in roi_set:
                continue
            if str(parsed.get("segment") or "") == seg_name and str(parsed.get("band") or "") == band:
                cols.append(c)
        
        if not cols:
            cell_data[col_idx] = None
        else:
            val_series = power_df[cols].mean(axis=1)
            cell_data[col_idx] = val_series
    
    return cell_data


def _compute_column_comparison_statistics(
    cell_data: Dict[int, Optional[pd.Series]],
    m1: np.ndarray,
    m2: np.ndarray,
    bands: List[str],
    config: Any,
) -> Tuple[Dict[int, Tuple[float, float, float, bool]], int]:
    """Compute Mann-Whitney U statistics for column comparison."""
    from .utils import apply_fdr_correction
    
    all_pvals = []
    pvalue_keys = []
    
    for col_idx, band in enumerate(bands):
        val_series = cell_data.get(col_idx)
        if val_series is None:
            continue
        
        v1 = val_series[m1].dropna().values
        v2 = val_series[m2].dropna().values
        
        if len(v1) < MIN_TRIALS_FOR_STATISTICS or len(v2) < MIN_TRIALS_FOR_STATISTICS:
            continue
        
        try:
            _, p = mannwhitneyu(v1, v2, alternative="two-sided")
            mean_diff = np.mean(v2) - np.mean(v1)
            pooled_std = np.sqrt(
                ((len(v1) - 1) * np.var(v1, ddof=1) + (len(v2) - 1) * np.var(v2, ddof=1))
                / (len(v1) + len(v2) - 2)
            )
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0
            all_pvals.append(p)
            pvalue_keys.append((col_idx, p, cohens_d))
        except Exception as exc:
            logger.warning(
                "Failed Mann-Whitney computation for band=%s (column index %s): %s",
                band,
                col_idx,
                exc,
            )
    
    qvalues = {}
    n_significant = 0
    if all_pvals:
        rejected, qvals, _ = apply_fdr_correction(all_pvals, config=config)
        for i, (key, p, d) in enumerate(pvalue_keys):
            qvalues[key] = (p, qvals[i], d, rejected[i])
        n_significant = int(np.sum(rejected))
    
    return qvalues, n_significant


def _compute_window_effect_summary(
    power_df: pd.DataFrame,
    bands: List[str],
    segments: List[str],
    roi_names: List[str],
    rois: Dict[str, Any],
    all_channels: List[str],
    config: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize paired window effects across ROI and band."""
    from eeg_pipeline.plotting.features.utils import apply_fdr_correction
    from eeg_pipeline.utils.analysis.stats.paired_comparisons import compute_paired_cohens_d

    if len(segments) != 2:
        raise ValueError("_compute_window_effect_summary requires exactly 2 segments.")

    effect_df = pd.DataFrame(np.nan, index=roi_names, columns=bands, dtype=float)
    pvalue_df = pd.DataFrame(np.nan, index=roi_names, columns=bands, dtype=float)
    segment1, segment2 = segments

    for roi_name in roi_names:
        roi_channels = all_channels if roi_name == "all" else get_roi_channels(rois[roi_name], all_channels)
        if not roi_channels:
            continue

        band_data = _extract_band_data_for_roi(
            power_df=power_df,
            bands=bands,
            segments=[segment1, segment2],
            roi_channels=roi_channels,
        )
        for band, (values1, values2) in band_data.items():
            if len(values1) < MIN_TRIALS_FOR_STATISTICS or len(values2) < MIN_TRIALS_FOR_STATISTICS:
                continue
            if len(values1) != len(values2):
                continue

            effect_df.loc[roi_name, band] = compute_paired_cohens_d(values1, values2)
            diffs = values2 - values1
            if diffs.size == 0 or not np.isfinite(diffs).any() or np.allclose(diffs, 0):
                pvalue_df.loc[roi_name, band] = 1.0
            else:
                pvalue_df.loc[roi_name, band] = float(
                    wilcoxon(diffs, zero_method="wilcox", alternative="two-sided").pvalue
                )

    finite_pvalues = pvalue_df.to_numpy(dtype=float)
    finite_mask = np.isfinite(finite_pvalues)
    qvalue_df = pd.DataFrame(np.nan, index=roi_names, columns=bands, dtype=float)
    if finite_mask.any():
        rejected, qvals, _ = apply_fdr_correction(finite_pvalues[finite_mask].tolist(), config=config)
        qvalue_values = qvalue_df.to_numpy(dtype=float, copy=True)
        qvalue_values[finite_mask] = qvals
        qvalue_df.iloc[:, :] = qvalue_values

    return effect_df, qvalue_df


def _compute_column_effect_summary(
    power_df: pd.DataFrame,
    events_df: pd.DataFrame,
    bands: List[str],
    seg_name: str,
    roi_names: List[str],
    rois: Dict[str, Any],
    all_channels: List[str],
    config: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """Summarize condition effects across ROI and band."""
    from eeg_pipeline.plotting.features.utils import apply_fdr_correction, compute_cohens_d
    from eeg_pipeline.utils.analysis.events import extract_comparison_mask

    comp_mask_info = extract_comparison_mask(events_df, config, require_enabled=True)
    if not comp_mask_info:
        raise ValueError("Configured condition comparison could not resolve comparison masks.")

    mask1, mask2, label1, label2 = comp_mask_info
    effect_df = pd.DataFrame(np.nan, index=roi_names, columns=bands, dtype=float)
    pvalue_df = pd.DataFrame(np.nan, index=roi_names, columns=bands, dtype=float)

    for roi_name in roi_names:
        roi_channels = all_channels if roi_name == "all" else get_roi_channels(rois[roi_name], all_channels)
        if not roi_channels:
            continue

        for band in bands:
            columns = _get_power_columns_for_roi(power_df, seg_name, band, roi_channels)
            if not columns:
                continue

            value_series = power_df[columns].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            values1 = value_series[mask1].dropna().values
            values2 = value_series[mask2].dropna().values
            if len(values1) < MIN_TRIALS_FOR_STATISTICS or len(values2) < MIN_TRIALS_FOR_STATISTICS:
                continue

            effect_df.loc[roi_name, band] = compute_cohens_d(values1, values2)
            pvalue_df.loc[roi_name, band] = float(
                mannwhitneyu(values1, values2, alternative="two-sided").pvalue
            )

    finite_pvalues = pvalue_df.to_numpy(dtype=float)
    finite_mask = np.isfinite(finite_pvalues)
    qvalue_df = pd.DataFrame(np.nan, index=roi_names, columns=bands, dtype=float)
    if finite_mask.any():
        rejected, qvals, _ = apply_fdr_correction(finite_pvalues[finite_mask].tolist(), config=config)
        qvalue_values = qvalue_df.to_numpy(dtype=float, copy=True)
        qvalue_values[finite_mask] = qvals
        qvalue_df.iloc[:, :] = qvalue_values

    return effect_df, qvalue_df, str(label1), str(label2)


def _compute_group_paired_effect_summary(
    subject_values: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    bands: List[str],
    roi_names: List[str],
    labels: Tuple[str, str],
    config: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize paired group effects across ROI and band."""
    from eeg_pipeline.plotting.features.utils import apply_fdr_correction
    from eeg_pipeline.utils.analysis.stats.paired_comparisons import compute_paired_cohens_d

    label1, label2 = labels
    effect_df = pd.DataFrame(np.nan, index=roi_names, columns=bands, dtype=float)
    pvalue_df = pd.DataFrame(np.nan, index=roi_names, columns=bands, dtype=float)

    for roi_name in roi_names:
        roi_band_values = subject_values.get(roi_name, {})
        if not roi_band_values:
            continue

        for band in bands:
            subject_band_values = roi_band_values.get(band, {})
            values1: List[float] = []
            values2: List[float] = []

            for condition_values in subject_band_values.values():
                value1 = condition_values.get(label1)
                value2 = condition_values.get(label2)
                try:
                    value1_float = float(value1)
                    value2_float = float(value2)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(value1_float) or not np.isfinite(value2_float):
                    continue
                values1.append(value1_float)
                values2.append(value2_float)

            if len(values1) < 2:
                continue

            values1_array = np.asarray(values1, dtype=float)
            values2_array = np.asarray(values2, dtype=float)
            effect_df.loc[roi_name, band] = compute_paired_cohens_d(values1_array, values2_array)

            diffs = values2_array - values1_array
            if np.allclose(diffs, 0.0):
                pvalue_df.loc[roi_name, band] = 1.0
                continue

            pvalue_df.loc[roi_name, band] = float(
                wilcoxon(diffs, zero_method="wilcox", alternative="two-sided").pvalue
            )

    finite_pvalues = pvalue_df.to_numpy(dtype=float)
    finite_mask = np.isfinite(finite_pvalues)
    qvalue_df = pd.DataFrame(np.nan, index=roi_names, columns=bands, dtype=float)
    if finite_mask.any():
        _, qvalues, _ = apply_fdr_correction(finite_pvalues[finite_mask].tolist(), config=config)
        qvalue_values = qvalue_df.to_numpy(dtype=float, copy=True)
        qvalue_values[finite_mask] = qvalues
        qvalue_df.iloc[:, :] = qvalue_values

    return effect_df, qvalue_df


def _extract_group_paired_cell_values(
    subject_values: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    *,
    roi_name: str,
    band: str,
    labels: Tuple[str, str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract finite paired values for one ROI x band cell."""
    label1, label2 = labels
    roi_band_values = subject_values.get(roi_name, {})
    subject_band_values = roi_band_values.get(band, {})
    values1: List[float] = []
    values2: List[float] = []

    for condition_values in subject_band_values.values():
        value1 = condition_values.get(label1)
        value2 = condition_values.get(label2)
        try:
            value1_float = float(value1)
            value2_float = float(value2)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value1_float) or not np.isfinite(value2_float):
            continue
        values1.append(value1_float)
        values2.append(value2_float)

    return np.asarray(values1, dtype=float), np.asarray(values2, dtype=float)


def _compute_effect_size_ci(
    values1: np.ndarray,
    values2: np.ndarray,
) -> Tuple[float, float]:
    """Estimate a deterministic bootstrap CI for paired repeated-measures effect size."""
    sample_size = len(values1)
    if sample_size < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(42)
    bootstrap_effects = np.empty(FOREST_BOOTSTRAP_SAMPLES, dtype=float)
    for index in range(FOREST_BOOTSTRAP_SAMPLES):
        sample_indices = rng.integers(0, sample_size, size=sample_size)
        bootstrap_effects[index] = _compute_plot_paired_effect_size(
            values1[sample_indices],
            values2[sample_indices],
        )

    finite_effects = bootstrap_effects[np.isfinite(bootstrap_effects)]
    if finite_effects.size == 0:
        return np.nan, np.nan
    return (
        float(np.nanquantile(finite_effects, 0.025)),
        float(np.nanquantile(finite_effects, 0.975)),
    )


def _compute_plot_paired_effect_size(
    values1: np.ndarray,
    values2: np.ndarray,
) -> float:
    """Return a finite paired repeated-measures effect size for plotting.

    Uses Cohen's d_av so constant paired shifts remain plottable.
    """
    before = np.asarray(values1, dtype=float).ravel()
    after = np.asarray(values2, dtype=float).ravel()
    finite_mask = np.isfinite(before) & np.isfinite(after)
    before = before[finite_mask]
    after = after[finite_mask]
    if before.size < 2:
        return np.nan

    pooled_sd = np.sqrt((np.var(before, ddof=1) + np.var(after, ddof=1)) / 2.0)
    mean_diff = float(np.mean(after - before))
    if not np.isfinite(pooled_sd) or pooled_sd <= 0:
        if np.isclose(mean_diff, 0.0):
            return 0.0
        pooled_sd = np.finfo(float).eps

    return float(mean_diff / pooled_sd)


def _compute_group_paired_effect_forest_data(
    subject_values: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    bands: List[str],
    roi_names: List[str],
    labels: Tuple[str, str],
    config: Any,
) -> pd.DataFrame:
    """Return paired group effect statistics for forest plotting."""
    from eeg_pipeline.plotting.features.utils import apply_fdr_correction

    rows: List[Dict[str, Any]] = []
    pvalues: List[float] = []

    for band in bands:
        for roi_name in roi_names:
            values1, values2 = _extract_group_paired_cell_values(
                subject_values,
                roi_name=roi_name,
                band=band,
                labels=labels,
            )
            if len(values1) < 2:
                continue

            effect_size = float(_compute_plot_paired_effect_size(values1, values2))
            differences = values2 - values1
            if np.allclose(differences, 0.0):
                pvalue = 1.0
            else:
                pvalue = float(
                    wilcoxon(differences, zero_method="wilcox", alternative="two-sided").pvalue
                )

            ci_low, ci_high = _compute_effect_size_ci(values1, values2)
            rows.append(
                {
                    "band": band,
                    "roi_name": roi_name,
                    "effect_size": effect_size,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": pvalue,
                    "n": int(len(values1)),
                }
            )
            pvalues.append(pvalue)

    forest_df = pd.DataFrame(rows)
    if forest_df.empty:
        return forest_df

    rejected, qvalues, _ = apply_fdr_correction(pvalues, config=config)
    forest_df["q_value"] = np.asarray(qvalues, dtype=float)
    forest_df["significant_fdr"] = np.asarray(rejected, dtype=bool)
    return forest_df


def _compute_group_paired_sample_count_summary(
    subject_values: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    bands: List[str],
    roi_names: List[str],
    labels: Tuple[str, str],
) -> pd.DataFrame:
    """Count paired subjects contributing to each ROI x band cell."""
    label1, label2 = labels
    count_df = pd.DataFrame(0, index=roi_names, columns=bands, dtype=int)

    for roi_name in roi_names:
        roi_band_values = subject_values.get(roi_name, {})
        if not roi_band_values:
            continue

        for band in bands:
            subject_band_values = roi_band_values.get(band, {})
            paired_count = 0
            for condition_values in subject_band_values.values():
                value1 = condition_values.get(label1)
                value2 = condition_values.get(label2)
                try:
                    value1_float = float(value1)
                    value2_float = float(value2)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(value1_float) or not np.isfinite(value2_float):
                    continue
                paired_count += 1

            count_df.loc[roi_name, band] = paired_count

    return count_df


def _plot_power_effect_summary_heatmap(
    effect_df: pd.DataFrame,
    qvalue_df: pd.DataFrame,
    subject: str,
    save_path: Path,
    logger: logging.Logger,
    config: Any,
    *,
    title: str,
    footer: str,
) -> None:
    """Render an ROI x band effect-size heatmap with FDR markers."""
    finite_effects = effect_df.to_numpy(dtype=float)
    if not np.isfinite(finite_effects).any():
        if logger:
            logger.warning("No finite effect sizes available for power summary heatmap.")
        return

    plot_cfg = get_plot_config(config)
    n_rows, n_cols = effect_df.shape
    fig_width = max(4.8, 1.15 * n_cols + 2.4)
    fig_height = max(3.4, 0.48 * n_rows + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    vmax = float(robust_sym_vlim([finite_effects], config=config))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    annot = n_rows * n_cols <= HEATMAP_EFFECT_ANNOTATION_THRESHOLD
    roi_labels = ["All Channels" if roi_name == "all" else roi_name.replace("_", " ") for roi_name in effect_df.index]
    band_labels = [str(column).upper() for column in effect_df.columns]

    sns.heatmap(
        effect_df,
        ax=ax,
        cmap="RdBu_r",
        center=0.0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Effect size (d)"},
        annot=annot,
        fmt=".2f",
        annot_kws={"fontsize": plot_cfg.font.small},
    )

    ax.set_yticklabels(roi_labels, rotation=0, fontsize=plot_cfg.font.small)
    ax.set_xticklabels(band_labels, rotation=0, fontsize=plot_cfg.font.medium, fontweight="bold")
    ax.set_xlabel("Frequency band", fontsize=plot_cfg.font.label)
    ax.set_ylabel("ROI", fontsize=plot_cfg.font.label)
    ax.set_title(title, fontsize=plot_cfg.font.figure_title, fontweight="bold", pad=12)

    significant_cells = np.argwhere((qvalue_df.to_numpy(dtype=float) < FDR_ALPHA_DEFAULT) & np.isfinite(qvalue_df.to_numpy(dtype=float)))
    for row_index, col_index in significant_cells:
        ax.scatter(
            col_index + 0.5,
            row_index + 0.5,
            s=36,
            facecolors="none",
            edgecolors="black",
            linewidths=1.0,
            zorder=5,
        )

    save_fig(
        fig,
        save_path,
        footer=footer,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
        tight_layout_rect=(0, 0.04, 1, 0.97),
        config=config,
    )
    if logger:
        logger.debug("Saved power ROI x band effect summary heatmap: %s", save_path.name)


def _plot_power_effect_forest(
    forest_df: pd.DataFrame,
    *,
    bands: List[str],
    roi_names: List[str],
    subject: str,
    save_path: Path,
    logger: logging.Logger,
    config: Any,
    title: str,
    footer: str,
) -> None:
    """Render a band-by-ROI forest plot using paired effect-size estimates and CIs."""
    if forest_df.empty:
        if logger:
            logger.warning("No paired forest statistics available for power forest plot.")
        return

    finite_effects = forest_df["effect_size"].to_numpy(dtype=float)
    finite_effects = finite_effects[np.isfinite(finite_effects)]
    if finite_effects.size == 0:
        if logger:
            logger.warning("No finite effect sizes available for power forest plot.")
        return

    plot_cfg = get_plot_config(config)
    available_bands = [band for band in bands if band in set(forest_df["band"])]
    if not available_bands:
        return

    range_values = forest_df[["effect_size", "ci_low", "ci_high"]].to_numpy(dtype=float)
    max_abs = float(np.nanmax(np.abs(range_values)))
    if not np.isfinite(max_abs) or max_abs <= 0:
        max_abs = 1.0
    x_limit = max_abs * 1.1

    fig_width = max(8.0, 2.15 * len(available_bands) + 1.6)
    fig_height = max(4.6, 0.45 * len(roi_names) + 1.8)
    fig, axes = plt.subplots(
        1,
        len(available_bands),
        figsize=(fig_width, fig_height),
        sharey=True,
        squeeze=False,
    )
    fig.patch.set_facecolor("white")

    y_positions = np.arange(len(roi_names))[::-1]
    roi_display_labels = [
        "All Channels" if roi_name == "all" else roi_name.replace("_", " ")
        for roi_name in roi_names
    ]

    for axis_index, band in enumerate(available_bands):
        ax = axes[0, axis_index]
        _style_publication_axis(ax)
        band_color = get_band_color(band, config)
        band_df = forest_df[forest_df["band"] == band].copy().set_index("roi_name")

        ax.axvline(0.0, color="0.45", linestyle="--", linewidth=0.9, alpha=0.7, zorder=0)

        for roi_index, roi_name in enumerate(roi_names):
            if roi_name not in band_df.index:
                continue

            row = band_df.loc[roi_name]
            y_position = y_positions[roi_index]
            effect_size = float(row["effect_size"])
            ci_low = float(row["ci_low"])
            ci_high = float(row["ci_high"])
            significant = bool(row["significant_fdr"])

            if np.isfinite(ci_low) and np.isfinite(ci_high):
                ax.hlines(
                    y_position,
                    ci_low,
                    ci_high,
                    color=band_color,
                    linewidth=1.6,
                    alpha=0.75,
                    zorder=1,
                )

            marker_edge = "black" if significant else band_color
            marker_face = "white" if significant else band_color
            marker_alpha = 1.0 if significant else 0.75
            ax.scatter(
                effect_size,
                y_position,
                s=34,
                facecolor=marker_face,
                edgecolor=marker_edge,
                linewidth=1.1,
                alpha=marker_alpha,
                zorder=3,
            )

        ax.set_xlim(-x_limit, x_limit)
        ax.set_title(
            band.upper(),
            fontweight="bold",
            color=band_color,
            fontsize=plot_cfg.font.title,
        )
        ax.set_xlabel("Effect size (d)", fontsize=plot_cfg.font.medium)
        ax.tick_params(axis="x", labelsize=plot_cfg.font.small)
        ax.tick_params(axis="y", labelsize=plot_cfg.font.small)
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, alpha=0.16, linewidth=0.6)
        ax.set_yticks(y_positions)
        if axis_index == 0:
            ax.set_yticklabels(roi_display_labels, fontsize=plot_cfg.font.small)
        else:
            ax.set_yticklabels([])

    fig.suptitle(
        title,
        fontsize=plot_cfg.font.figure_title,
        fontweight="bold",
        y=0.99,
    )
    save_fig(
        fig,
        save_path,
        footer=footer,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
        tight_layout_rect=(0, 0.04, 1, 0.98),
        config=config,
    )
    plt.close(fig)

    if logger:
        logger.debug("Saved power ROI x band forest plot: %s", save_path.name)


def _plot_power_sample_count_heatmap(
    count_df: pd.DataFrame,
    save_path: Path,
    logger: logging.Logger,
    config: Any,
    *,
    title: str,
    footer: str,
) -> None:
    """Render an ROI x band sample-count heatmap."""
    count_values = count_df.to_numpy(dtype=float)
    if not np.isfinite(count_values).any() or np.nanmax(count_values) <= 0:
        logger.warning("No positive sample counts available for power count heatmap.")
        return

    plot_cfg = get_plot_config(config)
    n_rows, n_cols = count_df.shape
    fig_width = max(4.8, 1.15 * n_cols + 2.4)
    fig_height = max(3.4, 0.48 * n_rows + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    roi_labels = ["All Channels" if roi_name == "all" else roi_name.replace("_", " ") for roi_name in count_df.index]
    band_labels = [str(column).upper() for column in count_df.columns]

    sns.heatmap(
        count_df,
        ax=ax,
        cmap="Greys",
        vmin=0,
        vmax=float(np.nanmax(count_values)),
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Paired subjects"},
        annot=True,
        fmt=".0f",
        annot_kws={"fontsize": plot_cfg.font.small},
    )

    ax.set_yticklabels(roi_labels, rotation=0, fontsize=plot_cfg.font.small)
    ax.set_xticklabels(band_labels, rotation=0, fontsize=plot_cfg.font.medium, fontweight="bold")
    ax.set_xlabel("Frequency band", fontsize=plot_cfg.font.label)
    ax.set_ylabel("ROI", fontsize=plot_cfg.font.label)
    ax.set_title(title, fontsize=plot_cfg.font.figure_title, fontweight="bold", pad=12)

    save_fig(
        fig,
        save_path,
        footer=footer,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
        tight_layout_rect=(0, 0.04, 1, 0.97),
        config=config,
    )
    logger.debug("Saved power ROI x band sample count heatmap: %s", save_path.name)


def _plot_window_comparison(
    power_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    segments: List[str],
    bands: List[str],
    roi_names: List[str],
    rois: Dict[str, Any],
    all_channels: List[str],
    stats_dir: Optional[Path],
) -> None:
    """Plot window comparison (paired) for power by condition.
    
    Supports both 2-window comparison (simple paired) and multi-window comparison
    (3+ windows with all pairwise brackets and significance asterisks).
    """
    from .utils import plot_paired_comparison, plot_multi_window_comparison
    from eeg_pipeline.utils.formatting import sanitize_label
    
    use_multi_window = len(segments) > 2
    
    for roi_name in roi_names:
        if roi_name == "all":
            roi_channels = all_channels
        else:
            roi_channels = get_roi_channels(rois[roi_name], all_channels)
        
        if not roi_channels:
            continue
        
        roi_safe = sanitize_label(roi_name).lower() if roi_name != "all" else ""
        suffix = f"_roi-{roi_safe}" if roi_safe else ""
        
        if use_multi_window:
            data_by_band = _extract_multi_segment_data_for_roi(
                power_df, bands, segments, roi_channels
            )
            
            if not data_by_band:
                continue
            
            save_path = save_dir / f"sub-{subject}_power_by_condition{suffix}_multiwindow"
            
            plot_multi_window_comparison(
                data_by_band=data_by_band,
                subject=subject,
                save_path=save_path,
                feature_label="Band Power",
                segments=segments,
                config=config,
                logger=logger,
                roi_name=roi_name,
                stats_dir=stats_dir,
            )
        else:
            seg1, seg2 = segments[0], segments[1]
            data_by_band = _extract_band_data_for_roi(
                power_df, bands, [seg1, seg2], roi_channels
            )
            
            if not data_by_band:
                continue
            
            save_path = save_dir / f"sub-{subject}_power_by_condition{suffix}_window"
            
            plot_paired_comparison(
                data_by_band=data_by_band,
                subject=subject,
                save_path=save_path,
                feature_label="Band Power",
                config=config,
                logger=logger,
                label1=seg1.capitalize(),
                label2=seg2.capitalize(),
                roi_name=roi_name,
                stats_dir=stats_dir,
            )


def _plot_column_comparison(
    power_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    bands: List[str],
    roi_names: List[str],
    rois: Dict[str, Any],
    all_channels: List[str],
    stats_dir: Optional[Path],
) -> None:
    """Plot column comparison (unpaired) for power by condition.
    
    Supports both 2-group comparison (simple unpaired) and multi-group comparison
    (3+ groups with all pairwise brackets and significance asterisks).
    """
    from eeg_pipeline.utils.analysis.events import extract_comparison_mask, extract_multi_group_masks
    from eeg_pipeline.utils.config.loader import get_config_value
    from .utils import load_precomputed_paired_stats, get_precomputed_qvalues, plot_multi_group_column_comparison, get_named_segments
    
    values_spec = get_config_value(config, "plotting.comparisons.comparison_values", [])
    use_multi_group = isinstance(values_spec, (list, tuple)) and len(values_spec) > 2
    
    if use_multi_group:
        multi_group_info = extract_multi_group_masks(events_df, config, require_enabled=True)
        if not multi_group_info:
            raise ValueError("Multi-group column comparison requested but could not resolve group masks.")
        
        masks_dict, group_labels = multi_group_info
        seg_name = str(require_config_value(config, "plotting.comparisons.comparison_segment")).strip()
        if seg_name == "":
            raise ValueError("plotting.comparisons.comparison_segment must be a non-empty string")
        
        from .utils import load_multigroup_stats
        multigroup_stats = load_multigroup_stats(stats_dir) if stats_dir else None
        
        for roi_name in roi_names:
            if roi_name == "all":
                roi_channels = all_channels
            else:
                roi_channels = get_roi_channels(rois[roi_name], all_channels)
            
            if not roi_channels:
                continue
            
            data_by_band: Dict[str, Dict[str, np.ndarray]] = {}
            for band in bands:
                cols = _get_power_columns_for_roi(power_df, seg_name, band, roi_channels)
                if not cols:
                    continue
                
                val_series = power_df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                
                group_values = {}
                for label, mask in masks_dict.items():
                    vals = val_series[mask].dropna().values
                    if len(vals) > 0:
                        group_values[label] = vals
                
                if len(group_values) >= 2:
                    data_by_band[band] = group_values
            
            if data_by_band:
                from eeg_pipeline.utils.formatting import sanitize_label
                roi_safe = sanitize_label(roi_name).lower() if roi_name != "all" else ""
                suffix = f"_roi-{roi_safe}" if roi_safe else ""
                save_path = save_dir / f"sub-{subject}_power_by_condition{suffix}_multigroup"
                
                plot_multi_group_column_comparison(
                    data_by_band=data_by_band,
                    subject=subject,
                    save_path=save_path,
                    feature_label="Band Power",
                    groups=group_labels,
                    config=config,
                    logger=logger,
                    roi_name=roi_name,
                    stats_dir=stats_dir,
                    multigroup_stats=multigroup_stats,
                )
        
        if logger:
            logger.debug(
                "Saved power multi-group column comparison for %s ROIs",
                len(roi_names),
            )
        return
    
    comp_mask_info = extract_comparison_mask(events_df, config, require_enabled=True)
    if not comp_mask_info:
        raise ValueError("Column comparison requested but could not resolve comparison masks.")
    
    m1, m2, label1, label2 = comp_mask_info
    
    seg_name = str(require_config_value(config, "plotting.comparisons.comparison_segment")).strip()
    if seg_name == "":
        raise ValueError("plotting.comparisons.comparison_segment must be a non-empty string")
    
    available_segments = get_named_segments(power_df, group="power")
    if seg_name not in available_segments:
        raise ValueError(
            f"Configured segment '{seg_name}' not found in data. Available segments: {available_segments}"
        )
    
    plot_cfg = get_plot_config(config)
    band_colors = {band: get_band_color(band, config) for band in bands}
    label_colors = _get_condition_color_map([label1, label2], config)
    
    precomputed_column_stats = None
    if stats_dir is not None:
        precomputed_column_stats = load_precomputed_paired_stats(
            stats_dir=stats_dir,
            feature_type="power",
            comparison_type="column",
            condition1=label1.lower(),
            condition2=label2.lower(),
            roi_name=None,
        )
        if precomputed_column_stats is not None and not precomputed_column_stats.empty:
            if logger:
                logger.debug(
                    "Using pre-computed column comparison stats (%s entries)",
                    len(precomputed_column_stats),
                )
    
    use_precomputed = precomputed_column_stats is not None and not precomputed_column_stats.empty
    
    for roi_name in roi_names:
        if roi_name == "all":
            roi_channels = all_channels
        else:
            roi_channels = get_roi_channels(rois[roi_name], all_channels)
        
        if not roi_channels:
            continue
        
        cell_data = _extract_column_comparison_data(
            power_df, bands, seg_name, roi_channels
        )
        
        bands_with_data = [band for col_idx, band in enumerate(bands) if cell_data.get(col_idx) is not None]
        if not bands_with_data:
            if logger:
                logger.error(
                    f"No power data found for segment '{seg_name}' in ROI {roi_name}. "
                    f"Skipping column comparison plot for this ROI."
                )
            continue
        
        plot_data = {}
        for col_idx, band in enumerate(bands):
            val_series = cell_data.get(col_idx)
            if val_series is None:
                plot_data[col_idx] = None
                continue
            v1 = val_series[m1].dropna().values
            v2 = val_series[m2].dropna().values
            if len(v1) == 0 or len(v2) == 0:
                if logger:
                    logger.warning(
                        f"No data for band {band} after filtering by conditions. "
                        f"v1: {len(v1)} trials, v2: {len(v2)} trials"
                    )
                plot_data[col_idx] = None
                continue
            plot_data[col_idx] = {"v1": v1, "v2": v2}
        
        if use_precomputed:
            qvalues = get_precomputed_qvalues(precomputed_column_stats, bands, roi_name or "all")
            n_significant = sum(1 for v in qvalues.values() if v[3])
            
            for col_idx, band in enumerate(bands):
                if band in qvalues:
                    p, q, d, sig = qvalues[band]
                    qvalues[col_idx] = (p, q, d, sig)
        else:
            qvalues, n_significant = _compute_column_comparison_statistics(
                cell_data, m1, m2, bands, config
            )
        
        fig, axes = plt.subplots(1, len(bands), figsize=(3 * len(bands), 5), squeeze=False)
        
        for col_idx, band in enumerate(bands):
            ax = axes.flatten()[col_idx]
            data = plot_data.get(col_idx)
            
            if data is None or len(data.get("v1", [])) == 0 or len(data.get("v2", [])) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                       transform=ax.transAxes, fontsize=plot_cfg.font.title, color="gray")
                ax.set_xticks([])
                continue
            
            v1, v2 = data["v1"], data["v2"]
            
            for position, values, label in ((0.0, v1, label1), (1.0, v2, label2)):
                color = label_colors[label]
                violin = ax.violinplot(values, positions=[position], showextrema=False, widths=0.68)
                for body in violin["bodies"]:
                    body.set_facecolor(color)
                    body.set_edgecolor(color)
                    body.set_alpha(0.22)
                    body.set_linewidth(0.8)

                box_center = position - 0.11 if position == 0.0 else position + 0.11
                ax.boxplot(
                    values,
                    positions=[box_center],
                    widths=0.16,
                    showfliers=False,
                    patch_artist=True,
                    boxprops=dict(facecolor="white", color=color, linewidth=1.0),
                    medianprops=dict(color="black", linewidth=1.2),
                    whiskerprops=dict(color=color, linewidth=1.0),
                    capprops=dict(color=color, linewidth=1.0),
                )
                jitter_low = position - 0.18 if position == 0.0 else position + 0.02
                jitter_high = position - 0.02 if position == 0.0 else position + 0.18
                jitter = np.random.default_rng(42 + int(position * 10)).uniform(jitter_low, jitter_high, len(values))
                ax.scatter(jitter, values, c=[color], alpha=0.65, s=12, linewidths=0, zorder=4)

                mean_value = float(np.nanmean(values))
                ci_half_width = 0.0
                if len(values) > 1:
                    ci_half_width = 1.96 * float(np.nanstd(values, ddof=1) / np.sqrt(len(values)))
                ax.errorbar(
                    [box_center],
                    [mean_value],
                    yerr=[[ci_half_width], [ci_half_width]],
                    fmt="o",
                    color="black",
                    markersize=4.5,
                    linewidth=1.0,
                    capsize=2.5,
                    zorder=6,
                )
            
            all_vals = np.concatenate([v1, v2])
            ymin = np.nanmin(all_vals)
            ymax = np.nanmax(all_vals)
            yrange = ymax - ymin if ymax > ymin else 0.1
            y_padding_bottom = 0.1 * yrange
            y_padding_top = 0.3 * yrange
            ax.set_ylim(ymin - y_padding_bottom, ymax + y_padding_top)
            
            if col_idx in qvalues:
                _, q, d, sig = qvalues[col_idx]
                sig_color = "#d62728" if sig else "#333333"
                q_label = "q<.001" if q < 0.001 else f"q={q:.3f}"
                ax.text(
                    0.5,
                    1.01,
                    f"{q_label} | d={d:.2f}" + ("  *" if sig else ""),
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=plot_cfg.font.small,
                    color=sig_color,
                    fontweight="bold" if sig else "normal",
                )
            
            ax.set_xticks([0, 1])
            ax.set_xticklabels([label1, label2], fontsize=9)
            ax.set_xlim(-0.35, 1.35)
            ax.set_title(band.capitalize(), fontweight="bold", color=band_colors[band], pad=12)
            ax.yaxis.grid(True, alpha=0.18, linewidth=0.6)
            ax.xaxis.grid(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(0.8)
            ax.spines["bottom"].set_linewidth(0.8)
        
        n_trials = len(power_df)
        roi_display = roi_name.replace("_", " ").title() if roi_name != "all" else "All Channels"
        n_tests = len(qvalues)
        
        stats_source = "pre-computed" if use_precomputed else "Mann-Whitney U"
        title = f"Band Power: {label1} vs {label2}"
        footer = (
            f"Subject: {subject} | ROI: {roi_display} | N: {n_trials} trials | "
            f"{stats_source} | FDR: {n_significant}/{n_tests} significant"
        )
        fig.suptitle(title, fontsize=plot_cfg.font.suptitle, fontweight="bold", y=0.99)
        
        from eeg_pipeline.utils.formatting import sanitize_label
        roi_safe = sanitize_label(roi_name).lower() if roi_name != "all" else ""
        suffix = f"_roi-{roi_safe}" if roi_safe else ""
        filename = f"sub-{subject}_power_by_condition{suffix}_column"
        
        save_fig(
            fig,
            save_dir / filename,
            footer=footer,
            formats=plot_cfg.formats,
            dpi=plot_cfg.dpi,
            bbox_inches=plot_cfg.bbox_inches,
            pad_inches=plot_cfg.pad_inches,
            tight_layout_rect=(0, 0.04, 1, 0.97),
            config=config,
        )
        
        if logger:
            logger.debug(
                "Saved power column comparison for ROI %s (%s/%s FDR significant, %s)",
                roi_display,
                n_significant,
                n_tests,
                stats_source,
            )


def plot_power_by_condition(
    power_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    stats_dir: Optional[Path] = None,
) -> None:
    """Compare power between conditions per band.
    
    For window comparisons (paired): Uses the unified plot_paired_comparison helper.
    For column comparisons (unpaired): Uses Mann-Whitney U test.
    
    If stats_dir is provided, uses pre-computed statistics from the behavior pipeline.
    """
    if power_df is None or power_df.empty or events_df is None:
        return

    from eeg_pipeline.utils.config.loader import get_config_value, get_frequency_band_names

    compare_wins = get_config_value(config, "plotting.comparisons.compare_windows", True)
    compare_cols = get_config_value(config, "plotting.comparisons.compare_columns", False)
    
    segments = _get_comparison_segments(power_df, config, logger)
    bands = list(get_frequency_band_names(config) or ["delta", "theta", "alpha", "beta", "gamma"])
    
    rois = get_roi_definitions(config)
    all_channels = extract_channels_from_columns(list(power_df.columns))
    roi_names = _get_comparison_rois(config, rois)
    
    if logger:
        logger.debug(
            "Power comparison: segments=%s, ROIs=%s, compare_windows=%s, compare_columns=%s",
            segments,
            roi_names,
            compare_wins,
            compare_cols,
        )
    
    if compare_wins and len(segments) >= 2:
        _plot_window_comparison(
            power_df, events_df, subject, save_dir, logger, config,
            segments, bands, roi_names, rois, all_channels, stats_dir
        )
        if len(segments) == 2:
            effect_df, qvalue_df = _compute_window_effect_summary(
                power_df=power_df,
                bands=bands,
                segments=segments,
                roi_names=roi_names,
                rois=rois,
                all_channels=all_channels,
                config=config,
            )
            _plot_power_effect_summary_heatmap(
                effect_df=effect_df,
                qvalue_df=qvalue_df,
                subject=subject,
                save_path=save_dir / f"sub-{subject}_power_roi_band_summary_window",
                logger=logger,
                config=config,
                title=f"Power window effects: {segments[1].capitalize()} - {segments[0].capitalize()}",
                footer=(
                    f"Subject: {subject} | Paired effect size (d) | "
                    f"FDR across ROI x band cells | open circles: q<0.05"
                ),
            )

    if compare_cols:
        _plot_column_comparison(
            power_df, events_df, subject, save_dir, logger, config,
            bands, roi_names, rois, all_channels, stats_dir
        )
        comparison_segment = str(require_config_value(config, "plotting.comparisons.comparison_segment")).strip()
        effect_df, qvalue_df, label1, label2 = _compute_column_effect_summary(
            power_df=power_df,
            events_df=events_df,
            bands=bands,
            seg_name=comparison_segment,
            roi_names=roi_names,
            rois=rois,
            all_channels=all_channels,
            config=config,
        )
        _plot_power_effect_summary_heatmap(
            effect_df=effect_df,
            qvalue_df=qvalue_df,
            subject=subject,
            save_path=save_dir / f"sub-{subject}_power_roi_band_summary_column",
            logger=logger,
            config=config,
            title=f"Power condition effects: {label2} - {label1}",
            footer=(
                f"Subject: {subject} | Segment: {comparison_segment} | "
                f"Unpaired effect size (d) | FDR across ROI x band cells | open circles: q<0.05"
            ),
        )





def _setup_subplot_grid(n_items: int, n_cols: int = 2, config: Any = None) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Create a subplot grid for multiple plots.
    
    Args:
        n_items: Number of subplots needed
        n_cols: Number of columns (default: 2)
        config: Configuration object
    
    Returns:
        Tuple of (figure, list of axes)
    """
    plot_cfg = get_plot_config(config)
    n_rows = (n_items + n_cols - 1) // n_cols
    width_per_col = float(plot_cfg.plot_type_configs.get("power", {}).get("width_per_col", 6.0))
    height_per_row = float(plot_cfg.plot_type_configs.get("power", {}).get("height_per_row", 4.0))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width_per_col * n_cols, height_per_row * n_rows))
    
    if n_items == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = axes.flatten()
    
    return fig, axes


def _validate_epochs_tfr(tfr: Any, function_name: str, logger: logging.Logger) -> bool:
    """Validate that TFR is EpochsTFR (4D) and raise if AverageTFR (3D).
    
    Args:
        tfr: TFR object to validate
        function_name: Name of calling function for error messages
        logger: Logger instance
    
    Returns:
        True if valid
    
    Raises:
        TypeError: If TFR is not EpochsTFR
        ValueError: If TFR data shape is incorrect
    """
    if not isinstance(tfr, mne.time_frequency.EpochsTFR):
        if isinstance(tfr, mne.time_frequency.AverageTFR):
            error_msg = (
                f"{function_name} requires EpochsTFR (4D: n_epochs, n_channels, n_freqs, n_times), "
                f"but received AverageTFR (3D: n_channels, n_freqs, n_times). "
                f"Cannot split by epochs/conditions with averaged data."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)
        else:
            error_msg = (
                f"{function_name} requires EpochsTFR, but received {type(tfr).__name__}"
            )
            logger.error(error_msg)
            raise TypeError(error_msg)
    
    if len(tfr.data.shape) != 4:
        error_msg = (
            f"{function_name} requires 4D TFR data (n_epochs, n_channels, n_freqs, n_times), "
            f"but received shape {tfr.data.shape}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    return True


def _get_active_window(config: Any) -> List[float]:
    """Get active window from config."""
    active_window = require_config_value(config, "time_frequency_analysis.active_window")
    if not isinstance(active_window, (list, tuple)) or len(active_window) < 2:
        raise ValueError(
            "time_frequency_analysis.active_window must be a list/tuple of length 2 "
            f"(got {active_window!r})"
        )
    return [float(active_window[0]), float(active_window[1])]


def _get_plotting_tfr_baseline_window(config: Any) -> tuple[float, float]:
    """Resolve baseline window for plotting."""
    baseline = require_config_value(config, "time_frequency_analysis.baseline_window")
    if not isinstance(baseline, (list, tuple)) or len(baseline) < 2:
        raise ValueError(
            "time_frequency_analysis.baseline_window must be a list/tuple of length 2 "
            f"(got {baseline!r})"
        )
    return float(baseline[0]), float(baseline[1])


def _crop_tfr_to_active(tfr: Any, active_window: List[float], logger: logging.Logger) -> Optional[Any]:
    """Crop TFR to active window.
    
    Args:
        tfr: TFR object to crop
        active_window: List of [start, end] times
        logger: Logger instance
    
    Returns:
        Cropped TFR or None if window is invalid
    """
    times = np.asarray(tfr.times)
    active_start = float(active_window[0])
    active_end = float(active_window[1])
    tmin = max(times.min(), active_start)
    tmax = min(times.max(), active_end)
    
    if tmax <= tmin:
        logger.warning("Invalid active window; skipping PSD")
        return None
    
    return tfr.copy().crop(tmin, tmax)


def _compute_mean_ci(values: np.ndarray) -> Tuple[float, float]:
    """Return the mean and 95% CI half-width for finite values."""
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return np.nan, 0.0
    mean_value = float(np.nanmean(finite_values))
    if finite_values.size < 2:
        return mean_value, 0.0
    sem = float(np.nanstd(finite_values, ddof=1) / np.sqrt(finite_values.size))
    return mean_value, 1.96 * sem


def _compute_window_mean_series(
    sample_matrix: np.ndarray,
    times: np.ndarray,
    window: Tuple[float, float],
    *,
    context: str,
) -> np.ndarray:
    """Return one mean value per sample across the requested time window."""
    values = np.asarray(sample_matrix, dtype=float)
    time_axis = np.asarray(times, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"{context} requires a 2D sample x time matrix.")
    if time_axis.ndim != 1 or values.shape[1] != time_axis.size:
        raise ValueError(
            f"{context} requires a 1D time axis matching matrix width; got {values.shape} and {time_axis.shape}."
        )

    window_mask = (time_axis >= float(window[0])) & (time_axis <= float(window[1]))
    if not np.any(window_mask):
        raise ValueError(f"{context} window {window!r} does not overlap the plotted time axis.")

    return np.nanmean(values[:, window_mask], axis=1)


def _draw_active_window_summary_inset(
    ax: Any,
    *,
    summary_by_label: Dict[str, np.ndarray],
    condition_labels: List[str],
    condition_color_map: Dict[str, Any],
    config: Any,
) -> None:
    """Draw a compact active-window summary inset with raw values and mean ± CI."""
    plotted_labels = [label for label in condition_labels if label in summary_by_label]
    if not plotted_labels:
        return

    inset_ax = ax.inset_axes([0.68, 0.56, 0.28, 0.32])
    inset_ax.set_facecolor("white")
    inset_ax.axhline(1.0, color="0.55", linestyle="--", linewidth=0.8, alpha=0.7, zorder=0)

    if len(plotted_labels) == 2:
        values_1 = np.asarray(summary_by_label[plotted_labels[0]], dtype=float)
        values_2 = np.asarray(summary_by_label[plotted_labels[1]], dtype=float)
        if values_1.shape == values_2.shape:
            finite_mask = np.isfinite(values_1) & np.isfinite(values_2)
            if np.any(finite_mask):
                inset_ax.plot(
                    [0.0, 1.0],
                    np.vstack([values_1[finite_mask], values_2[finite_mask]]),
                    color="0.78",
                    linewidth=0.55,
                    alpha=0.55,
                    zorder=1,
                )

    for index, label in enumerate(plotted_labels):
        values = np.asarray(summary_by_label[label], dtype=float)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            continue

        color = condition_color_map[label]
        rng = np.random.default_rng(900 + index)
        jitter = rng.uniform(index - 0.08, index + 0.08, size=finite_values.size)
        inset_ax.scatter(
            jitter,
            finite_values,
            s=10,
            color=color,
            alpha=0.65,
            linewidths=0,
            zorder=3,
        )

        mean_value, ci_half_width = _compute_mean_ci(finite_values)
        inset_ax.errorbar(
            [index],
            [mean_value],
            yerr=[[ci_half_width], [ci_half_width]],
            fmt="o",
            color="black",
            markerfacecolor="white",
            markersize=4.2,
            capsize=2.0,
            linewidth=0.9,
            zorder=5,
        )

    plot_cfg = get_plot_config(config)
    inset_labels = [
        textwrap.fill(_format_condition_display_label(label, config), width=10)
        for label in plotted_labels
    ]
    inset_ax.set_xlim(-0.35, max(len(plotted_labels) - 0.35, 0.35))
    inset_ax.set_xticks(list(range(len(plotted_labels))))
    inset_ax.set_xticklabels(inset_labels, fontsize=max(plot_cfg.font.small - 1, 6))
    inset_ax.tick_params(axis="y", labelsize=max(plot_cfg.font.small - 1, 6))
    inset_ax.set_title("Active window", fontsize=plot_cfg.font.small, pad=3)
    inset_ax.yaxis.grid(True, alpha=0.15, linewidth=0.5)
    inset_ax.xaxis.grid(False)
    sns.despine(ax=inset_ax, trim=True)


def _compute_paired_effect_matrix(
    condition1_values: np.ndarray,
    condition2_values: np.ndarray,
    *,
    context: str,
) -> np.ndarray:
    """Return the paired effect matrix using the plotting convention condition2 - condition1."""
    first = np.asarray(condition1_values, dtype=float)
    second = np.asarray(condition2_values, dtype=float)
    if first.ndim != 2 or second.ndim != 2:
        raise ValueError(f"{context} requires 2D subject x time arrays.")
    if first.shape != second.shape:
        raise ValueError(f"{context} requires arrays with matching shapes, got {first.shape} and {second.shape}.")
    return second - first


def _draw_effect_window_summary_inset(
    ax: Any,
    *,
    effect_values: np.ndarray,
    color: Any,
    config: Any,
) -> None:
    """Draw a compact active-window effect inset for paired subject differences."""
    finite_values = np.asarray(effect_values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return

    inset_ax = ax.inset_axes([0.68, 0.56, 0.28, 0.32])
    inset_ax.set_facecolor("white")
    inset_ax.axhline(0.0, color="0.55", linestyle="--", linewidth=0.8, alpha=0.7, zorder=0)

    rng = np.random.default_rng(1400)
    jitter = rng.uniform(-0.08, 0.08, size=finite_values.size)
    inset_ax.scatter(
        jitter,
        finite_values,
        s=10,
        color=color,
        alpha=0.68,
        linewidths=0,
        zorder=3,
    )

    mean_value, ci_half_width = _compute_mean_ci(finite_values)
    inset_ax.errorbar(
        [0.0],
        [mean_value],
        yerr=[[ci_half_width], [ci_half_width]],
        fmt="o",
        color="black",
        markerfacecolor="white",
        markersize=4.2,
        capsize=2.0,
        linewidth=0.9,
        zorder=5,
    )

    plot_cfg = get_plot_config(config)
    inset_ax.set_xlim(-0.22, 0.22)
    inset_ax.set_xticks([0.0])
    inset_ax.set_xticklabels(["Δ"], fontsize=max(plot_cfg.font.small - 1, 6))
    inset_ax.tick_params(axis="y", labelsize=max(plot_cfg.font.small - 1, 6))
    inset_ax.set_title("Active-window effect", fontsize=plot_cfg.font.small, pad=3)
    inset_ax.yaxis.grid(True, alpha=0.15, linewidth=0.5)
    inset_ax.xaxis.grid(False)
    sns.despine(ax=inset_ax, trim=True)


def _validate_predictor_data(
    tfr: Any,
    events_df: Optional[pd.DataFrame],
    *,
    config: Any,
    subject: str,
    logger: logging.Logger,
) -> Optional[pd.Series]:
    """Validate and extract predictor data from events DataFrame.
    
    Args:
        tfr: TFR object
        events_df: Events DataFrame
        subject: Subject identifier
        logger: Logger instance
    
    Returns:
        Series of predictor values or None if validation fails
    """
    if config is None:
        logger.warning("Config is required for predictor plotting; skipping.")
        return None

    if events_df is None or events_df.empty:
        logger.warning("No events DataFrame provided for predictor analysis")
        return None
        
    temp_col = find_predictor_column_in_events(events_df, config)
    if temp_col is None:
        logger.warning("No predictor column found in events")
        return None
        
    temps = pd.to_numeric(events_df[temp_col], errors="coerce")
    if len(tfr) != len(temps):
        logger.warning(
            f"TFR window ({len(tfr)} epochs) and events "
            f"({len(temps)} rows) length mismatch for subject {subject}"
        )
        return None
        
    return temps


def _get_band_frequency_mask(tfr: Any, band: str, config: Any, logger: logging.Logger) -> Optional[np.ndarray]:
    """Get frequency mask for a given band.
    
    Args:
        tfr: TFR object
        band: Band name (e.g., 'alpha')
        config: Configuration object
        logger: Logger instance
    
    Returns:
        Boolean mask array or None if band not found
    """
    if config is None:
        logger.warning("Config is required to get band frequency mask")
        return None
        
    freq_bands = get_frequency_bands(config)
    if not freq_bands or band not in freq_bands:
        logger.warning(f"Band '{band}' not found in configuration")
        return None
        
    fmin, fmax = freq_bands[band]
    mask = (tfr.freqs >= fmin) & (tfr.freqs <= fmax)
    
    if not mask.any():
        logger.warning(f"No frequencies found for band '{band}' ({fmin}-{fmax} Hz)")
        return None
        
    return mask


###################################################################
# Power Spectral Density Plotting
###################################################################


def _plot_psd_by_conditions(
    tfr_epochs: Any,
    conditions: List[Tuple[str, np.ndarray]],
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    roi_suffix: str = "",
    roi_name: Optional[str] = None,
) -> bool:
    """Plot PSD by condition with uncertainty visualization and frequency band annotations.
    
    Computes PSD per trial, then averages across channels and time windows.
    Shows mean ± SEM with shaded confidence intervals for scientific rigor.
    Includes frequency band annotations and optional statistical comparison.
    
    Args:
        tfr_epochs: EpochsTFR object
        conditions: List of (label, mask) tuples
        subject: Subject identifier
        save_dir: Directory to save plots
        logger: Logger instance
        config: Configuration object
    
    Returns:
        True if plot was created
    
    Raises:
        ValueError: If insufficient conditions or no valid data
    """
    if len(conditions) < 1:
        raise ValueError(
            f"power_spectral_density requires at least 1 condition, got {len(conditions)}"
        )
    
    plot_cfg = get_plot_config(config)
    fig_size = plot_cfg.get_figure_size("medium", plot_type="features")
    fig, ax = plt.subplots(figsize=fig_size)
    fig.patch.set_facecolor("white")
    _style_publication_axis(ax)
    
    active_window = _get_active_window(config)
    tfr_baseline = _get_plotting_tfr_baseline_window(config)
    
    condition_color_map = _get_condition_color_map([label for label, _ in conditions], config)
    
    freq_bands = get_frequency_bands(config)
    features_freq_bands = {name: tuple(freqs) for name, freqs in freq_bands.items()}
    
    psd_data_by_condition = []
    
    for idx, (label, mask) in enumerate(conditions):
        n_trials_cond = int(mask.sum())
        if n_trials_cond < 1:
            continue
        
        tfr_cond = tfr_epochs[mask]
        if len(tfr_cond) == 0:
            continue
        
        tfr_cond_avg = tfr_cond.average()
        apply_baseline_and_crop(tfr_cond_avg, baseline=tfr_baseline, mode="logratio", logger=logger)
        tfr_cond_win = _crop_tfr_to_active(tfr_cond_avg, active_window, logger)
        
        if tfr_cond_win is None:
            continue
        
        psd_mean = tfr_cond_win.data.mean(axis=(0, 2))
        
        if len(psd_mean) != len(tfr_cond_win.freqs):
            logger.warning(f"Frequency dimension mismatch: {len(psd_mean)} vs {len(tfr_cond_win.freqs)}")
            continue
        
        freqs = tfr_cond_win.freqs
        
        if len(tfr_cond) >= MIN_EPOCHS_FOR_SEM:
            psd_per_trial = []
            for trial_idx in range(len(tfr_cond)):
                tfr_trial = tfr_cond[[trial_idx]]
                tfr_trial_avg = tfr_trial.average()
                apply_baseline_and_crop(tfr_trial_avg, baseline=tfr_baseline, mode="logratio", logger=logger)
                tfr_trial_win = _crop_tfr_to_active(tfr_trial_avg, active_window, logger)
                if tfr_trial_win is not None:
                    psd_trial = tfr_trial_win.data.mean(axis=(0, 2))
                    if len(psd_trial) == len(freqs):
                        psd_per_trial.append(psd_trial)
            
            if len(psd_per_trial) >= MIN_EPOCHS_FOR_SEM:
                psd_per_trial = np.array(psd_per_trial)
                psd_sem = psd_per_trial.std(axis=0, ddof=1) / np.sqrt(len(psd_per_trial))
                ci_multiplier = 1.96
                psd_ci_lower = psd_mean - ci_multiplier * psd_sem
                psd_ci_upper = psd_mean + ci_multiplier * psd_sem
            else:
                psd_sem = np.zeros_like(psd_mean)
                psd_ci_lower = psd_mean
                psd_ci_upper = psd_mean
        else:
            psd_sem = np.zeros_like(psd_mean)
            psd_ci_lower = psd_mean
            psd_ci_upper = psd_mean
        
        psd_data_by_condition.append({
            'label': label,
            'freqs': freqs,
            'mean': psd_mean,
            'sem': psd_sem,
            'ci_lower': psd_ci_lower,
            'ci_upper': psd_ci_upper,
            'n_trials': n_trials_cond,
            'color': condition_color_map[label],
            'stacked': psd_per_trial if len(tfr_cond) >= MIN_EPOCHS_FOR_SEM and len(psd_per_trial) >= MIN_EPOCHS_FOR_SEM else None,
        })
    
    if not psd_data_by_condition:
        plt.close(fig)
        raise ValueError(
            "power_spectral_density plot failed: no conditions had valid trials. "
            "Check that conditions have sufficient data."
        )
    
    for psd_data in psd_data_by_condition:
        has_uncertainty = np.any(psd_data['sem'] > 0)
        
        if has_uncertainty:
            ax.fill_between(
                psd_data['freqs'],
                psd_data['ci_lower'],
                psd_data['ci_upper'],
                color=psd_data['color'],
                alpha=0.15,
                linewidth=0,
                zorder=1,
            )
        
        ax.plot(
            psd_data['freqs'],
            psd_data['mean'],
            color=psd_data['color'],
            linewidth=2.0,
            label=(
                f"{_format_condition_display_label(psd_data['label'], config)} "
                f"(n={psd_data['n_trials']})"
            ),
            zorder=3,
        )

    _annotate_frequency_bands(
        ax,
        features_freq_bands,
        float(psd_data_by_condition[0]['freqs'].max()),
        config,
    )

    n_significant_bands = 0
    if len(psd_data_by_condition) == 2:
        first_stacked = psd_data_by_condition[0]["stacked"]
        second_stacked = psd_data_by_condition[1]["stacked"]
        if first_stacked is not None and second_stacked is not None:
            first_stacked = np.asarray(first_stacked, dtype=float)
            second_stacked = np.asarray(second_stacked, dtype=float)
            if first_stacked.shape == second_stacked.shape and first_stacked.ndim == 2:
                significant_mask = _compute_group_curve_significance_mask(first_stacked, second_stacked, config)
                _draw_curve_significance_strip(ax, psd_data_by_condition[0]["freqs"], significant_mask, label="PSD FDR q<0.05")
                band_stats = _compute_group_band_summary_stats(
                    first_stacked,
                    second_stacked,
                    psd_data_by_condition[0]["freqs"],
                    features_freq_bands,
                    config,
                )
                _draw_psd_band_summary_strip(ax, band_stats)
                n_significant_bands = sum(
                    1 for stats in band_stats.values() if bool(stats.get("significant", False))
                )
    
    ax.axhline(0, color="0.4", linewidth=1.0, alpha=0.5, linestyle='--', zorder=2)
    ax.set_xscale('log')
    import matplotlib.ticker as ticker
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.set_xticks([2, 4, 8, 16, 32, 64])
    ax.set_xlabel("Frequency (Hz)", fontsize=plot_cfg.font.ylabel, fontweight='medium')
    ax.set_ylabel(r"$\log_{10}$(power / baseline)", fontsize=plot_cfg.font.ylabel, fontweight='medium')
    ax.legend(loc='upper left', fontsize=plot_cfg.font.medium, frameon=False, handlelength=1.5)
    
    if roi_name:
        roi_display = roi_name.replace("_", " ").title() if roi_name != "all" else "All Channels"
        title = f"Descriptive power spectral density | {roi_display}"
        fig.suptitle(title, fontsize=plot_cfg.font.figure_title, fontweight="bold", y=0.99)

    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, zorder=0)
    ax.tick_params(labelsize=plot_cfg.font.small)
    
    footer_text = (
        f"Subject: {subject} | Baseline: [{tfr_baseline[0]:.2f}, {tfr_baseline[1]:.2f}] s | "
        f"Window: [{active_window[0]:.1f}, {active_window[1]:.1f}] s | "
        f"Band sig={n_significant_bands}/{len(features_freq_bands)} | "
        "Within-subject mean ± 95% CI"
    )
    output_path = save_dir / f'sub-{subject}_power_spectral_density_by_condition{roi_suffix}'
    save_fig(
        fig,
        output_path,
        footer=footer_text,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
        tight_layout_rect=(0, 0.04, 1, 0.98),
        config=config,
    )
    logger.debug(
        "Saved PSD by condition (Induced) with uncertainty visualization%s",
        roi_suffix,
    )
    return True


def _plot_psd_by_predictor(
    tfr_epochs: Any,
    temps: pd.Series,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any
) -> bool:
    """Plot PSD by predictor condition with uncertainty visualization and frequency band annotations.
    
    Computes PSD per trial, then averages across channels and time windows.
    Shows mean ± SEM with shaded confidence intervals for scientific rigor.
    Includes frequency band annotations.
    
    Args:
        tfr_epochs: EpochsTFR object
        preds: Series of predictor values
        subject: Subject identifier
        save_dir: Directory to save plots
        logger: Logger instance
        config: Configuration object
    
    Returns:
        True if plot was created, False otherwise
    """
    MIN_TEMPERATURES_FOR_COMPARISON = 2
    unique_temps = sorted(temps.dropna().unique())
    if len(unique_temps) < MIN_TEMPERATURES_FOR_COMPARISON:
        return False
    
    plot_cfg = get_plot_config(config)
    fig_size = plot_cfg.get_figure_size("medium", plot_type="features")
    fig, ax = plt.subplots(figsize=fig_size)
    fig.patch.set_facecolor("white")
    _style_publication_axis(ax)
    temp_palette = sns.color_palette("coolwarm", n_colors=len(unique_temps))
    
    active_window = _get_active_window(config)
    tfr_baseline = _get_plotting_tfr_baseline_window(config)
    
    freq_bands = get_frequency_bands(config)
    features_freq_bands = {name: tuple(freqs) for name, freqs in freq_bands.items()}
    
    psd_data_by_temp = []
    
    for idx, temp in enumerate(unique_temps):
        temp_mask = (temps == temp).to_numpy()
        n_trials_temp = int(temp_mask.sum())
        if n_trials_temp < 1:
            continue
        
        tfr_temp = tfr_epochs[temp_mask]
        if len(tfr_temp) == 0:
            continue
        
        tfr_temp_avg = tfr_temp.average()
        apply_baseline_and_crop(tfr_temp_avg, baseline=tfr_baseline, mode="logratio", logger=logger)
        tfr_temp_win = _crop_tfr_to_active(tfr_temp_avg, active_window, logger)
        
        if tfr_temp_win is None:
            continue
        
        psd_mean = tfr_temp_win.data.mean(axis=(0, 2))
        
        if len(psd_mean) != len(tfr_temp_win.freqs):
            logger.warning(f"Frequency dimension mismatch: {len(psd_mean)} vs {len(tfr_temp_win.freqs)}")
            continue
        
        freqs = tfr_temp_win.freqs
        
        if len(tfr_temp) >= MIN_EPOCHS_FOR_SEM:
            psd_per_trial = []
            for trial_idx in range(len(tfr_temp)):
                tfr_trial = tfr_temp[[trial_idx]]
                tfr_trial_avg = tfr_trial.average()
                apply_baseline_and_crop(tfr_trial_avg, baseline=tfr_baseline, mode="logratio", logger=logger)
                tfr_trial_win = _crop_tfr_to_active(tfr_trial_avg, active_window, logger)
                if tfr_trial_win is not None:
                    psd_trial = tfr_trial_win.data.mean(axis=(0, 2))
                    if len(psd_trial) == len(freqs):
                        psd_per_trial.append(psd_trial)
            
            if len(psd_per_trial) >= MIN_EPOCHS_FOR_SEM:
                psd_per_trial = np.array(psd_per_trial)
                psd_sem = psd_per_trial.std(axis=0, ddof=1) / np.sqrt(len(psd_per_trial))
                ci_multiplier = 1.96
                psd_ci_lower = psd_mean - ci_multiplier * psd_sem
                psd_ci_upper = psd_mean + ci_multiplier * psd_sem
            else:
                psd_sem = np.zeros_like(psd_mean)
                psd_ci_lower = psd_mean
                psd_ci_upper = psd_mean
        else:
            psd_sem = np.zeros_like(psd_mean)
            psd_ci_lower = psd_mean
            psd_ci_upper = psd_mean
        
        psd_data_by_temp.append({
            'label': f'{temp:.0f}°C',
            'freqs': freqs,
            'mean': psd_mean,
            'sem': psd_sem,
            'ci_lower': psd_ci_lower,
            'ci_upper': psd_ci_upper,
            'n_trials': n_trials_temp,
            'color': temp_palette[idx],
        })
    
    if not psd_data_by_temp:
        plt.close(fig)
        return False
    
    for psd_data in psd_data_by_temp:
        has_uncertainty = np.any(psd_data['sem'] > 0)
        
        if has_uncertainty:
            ax.fill_between(
                psd_data['freqs'],
                psd_data['ci_lower'],
                psd_data['ci_upper'],
                color=psd_data['color'],
                alpha=0.15,
                linewidth=0,
                zorder=1,
            )
        
        ax.plot(
            psd_data['freqs'],
            psd_data['mean'],
            color=psd_data['color'],
            linewidth=2.0,
            label=f"{psd_data['label']} (n={psd_data['n_trials']})",
            zorder=3,
        )

    _annotate_frequency_bands(
        ax,
        features_freq_bands,
        float(psd_data_by_temp[0]['freqs'].max()),
        config,
    )
    
    ax.axhline(0, color="0.4", linewidth=1.0, alpha=0.5, linestyle='--', zorder=2)
    ax.set_xscale('log')
    import matplotlib.ticker as ticker
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.set_xticks([2, 4, 8, 16, 32, 64])
    ax.set_xlabel("Frequency (Hz)", fontsize=plot_cfg.font.ylabel, fontweight='medium')
    ax.set_ylabel(r"$\log_{10}$(power / baseline)", fontsize=plot_cfg.font.ylabel, fontweight='medium')
    ax.legend(loc='best', fontsize=plot_cfg.font.medium, frameon=False, handlelength=1.5)
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, zorder=0)
    ax.tick_params(labelsize=plot_cfg.font.small)
    
    footer_text = (
        f"Subject: {subject} | Baseline: [{tfr_baseline[0]:.2f}, {tfr_baseline[1]:.2f}] s | "
        f"Window: [{active_window[0]:.1f}, {active_window[1]:.1f}] s | "
        "Within-subject mean ± 95% CI"
    )
    output_path = save_dir / f'sub-{subject}_power_spectral_density_by_predictor'
    save_fig(
        fig,
        output_path,
        footer=footer_text,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
        tight_layout_rect=(0, 0.04, 1, 0.98),
        config=config,
    )
    logger.debug("Saved PSD by predictor (Induced) with uncertainty visualization")
    return True


def _plot_psd_overall(
    tfr_avg_win: Any,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any
) -> None:
    """Plot overall PSD (internal helper).
    
    Args:
        tfr_avg_win: AverageTFR object (already averaged, baselined, and cropped)
        subject: Subject identifier
        save_dir: Directory to save plots
        logger: Logger instance
        config: Configuration object
    """
    psd_avg = tfr_avg_win.data.mean(axis=(0, 2))
    
    fig, ax = plt.subplots(figsize=(4.0, 2.5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    _style_publication_axis(ax)
    ax.plot(tfr_avg_win.freqs, psd_avg, color="0.2", linewidth=1.0)

    ax.axhline(0, color="0.7", linewidth=0.5, alpha=0.6)
    
    freq_bands = get_frequency_bands(config)
    features_freq_bands = {name: tuple(freqs) for name, freqs in freq_bands.items()}

    _annotate_frequency_bands(
        ax,
        features_freq_bands,
        float(tfr_avg_win.freqs.max()),
        config,
    )
    
    plot_cfg = get_plot_config(config)
    ax.set_xscale('log')
    import matplotlib.ticker as ticker
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.set_xticks([2, 4, 8, 16, 32, 64])
    ax.set_xlabel("Frequency (Hz)", fontsize=plot_cfg.font.medium)
    ax.set_ylabel(r"$\log_{10}$(power/baseline)", fontsize=plot_cfg.font.medium)
    ax.tick_params(labelsize=plot_cfg.font.small)
    sns.despine(ax=ax, trim=True)
    
    output_path = save_dir / f'sub-{subject}_power_spectral_density'
    save_fig(fig, output_path, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches, config=config)
    plt.close(fig)
    logger.debug("Saved PSD (Induced)")


def plot_power_spectral_density(
    tfr: Any,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    events_df: Optional[pd.DataFrame] = None,
    config: Optional[Any] = None
) -> None:
    """Plot power spectral density by condition, one plot per ROI.
    
    Requires condition selection to be configured via plotting.comparisons.
    Creates one plot per ROI specified in the TUI.
    No fallbacks - errors will surface if conditions cannot be extracted.
    
    Args:
        tfr: EpochsTFR object
        subject: Subject identifier
        save_dir: Directory to save plots
        logger: Logger instance
        events_df: Events DataFrame (required)
        config: Configuration object (required)
    
    Raises:
        ValueError: If events_df or config is missing, or if conditions cannot be extracted
    """
    _validate_epochs_tfr(tfr, "plot_power_spectral_density", logger)
    
    if events_df is None or events_df.empty:
        raise ValueError(
            "plot_power_spectral_density requires events_df. "
            "No fallback will be used."
        )
    
    if config is None:
        raise ValueError(
            "plot_power_spectral_density requires config. "
            "No fallback will be used."
        )
    
    if len(tfr) != len(events_df):
        raise ValueError(
            f"TFR window ({len(tfr)} epochs) and events "
            f"({len(events_df)} rows) length mismatch for subject {subject}"
        )
    
    from eeg_pipeline.utils.analysis.events import extract_comparison_mask, extract_multi_group_masks
    from eeg_pipeline.utils.config.loader import get_config_value, require_config_value
    from eeg_pipeline.utils.formatting import sanitize_label
    
    rois = get_roi_definitions(config)
    all_channels = tfr.ch_names
    roi_names = _get_comparison_rois(config, rois)
    
    if logger:
        logger.debug("PSD plotting: ROIs=%s", roi_names)
    
    column = require_config_value(config, "plotting.comparisons.comparison_column")
    values_spec = get_config_value(config, "plotting.comparisons.comparison_values", [])
    labels_spec = get_config_value(config, "plotting.comparisons.comparison_labels", None)
    
    if not isinstance(values_spec, (list, tuple)) or len(values_spec) < 1:
        raise ValueError(
            "power_spectral_density requires plotting.comparisons.comparison_values with at least 1 value. "
            "Configure via TUI plot-specific settings or CLI."
        )
    
    if len(values_spec) == 1:
        val = values_spec[0]
        if isinstance(labels_spec, (list, tuple)) and len(labels_spec) >= 1:
            label = str(labels_spec[0]).strip()
        else:
            label = str(val)
        
        column_values = events_df[column]
        try:
            numeric_val = float(val)
            mask = (pd.to_numeric(column_values, errors="coerce") == numeric_val).values
        except (ValueError, TypeError):
            val_str = str(val).strip().lower()
            mask = (column_values.astype(str).str.strip().str.lower() == val_str).values
        
        if int(mask.sum()) == 0:
            raise ValueError(
                f"power_spectral_density: no trials found for value {val!r} in column {column!r}"
            )
        
        conditions = [(label, mask)]
    elif len(values_spec) == 2:
        comp_mask_info = extract_comparison_mask(events_df, config, require_enabled=True)
        if not comp_mask_info:
            raise ValueError(
                "power_spectral_density plot requested but could not resolve comparison masks. "
                "Configure plotting.comparisons.comparison_column and comparison_values."
            )
        mask1, mask2, label1, label2 = comp_mask_info
        conditions = [(label1, mask1), (label2, mask2)]
    else:
        multi_group_info = extract_multi_group_masks(events_df, config, require_enabled=True)
        if not multi_group_info:
            raise ValueError(
                "power_spectral_density plot requested but could not resolve multi-group masks. "
                "Configure plotting.comparisons.comparison_column and comparison_values."
            )
        masks_dict, group_labels = multi_group_info
        conditions = [(label, masks_dict[label]) for label in group_labels]
    
    for roi_name in roi_names:
        if roi_name == "all":
            roi_channels = all_channels
        else:
            roi_channels = get_roi_channels(rois[roi_name], all_channels)
        
        if not roi_channels:
            if logger:
                logger.warning(f"No channels found for ROI {roi_name}, skipping PSD plot")
            continue
        
        tfr_roi = tfr.copy().pick(roi_channels)
        if len(tfr_roi.ch_names) == 0:
            if logger:
                logger.warning(f"No valid channels after filtering for ROI {roi_name}, skipping PSD plot")
            continue
        
        roi_safe = sanitize_label(roi_name).lower() if roi_name != "all" else ""
        roi_suffix = f"_roi-{roi_safe}" if roi_safe else ""
        
        _plot_psd_by_conditions(tfr_roi, conditions, subject, save_dir, logger, config, roi_suffix=roi_suffix, roi_name=roi_name)


def plot_power_timecourse_by_condition(
    tfr: Any,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    events_df: Optional[pd.DataFrame] = None,
    config: Optional[Any] = None,
) -> None:
    """Plot time-resolved band power trajectories for configured conditions."""
    _validate_epochs_tfr(tfr, "plot_power_timecourse_by_condition", logger)

    if events_df is None or events_df.empty:
        raise ValueError(
            "plot_power_timecourse_by_condition requires events_df. "
            "No fallback will be used."
        )
    if config is None:
        raise ValueError(
            "plot_power_timecourse_by_condition requires config. "
            "No fallback will be used."
        )
    if len(tfr) != len(events_df):
        raise ValueError(
            f"TFR window ({len(tfr)} epochs) and events "
            f"({len(events_df)} rows) length mismatch for subject {subject}"
        )

    from eeg_pipeline.utils.analysis.events import extract_comparison_mask, extract_multi_group_masks
    from eeg_pipeline.utils.formatting import sanitize_label

    rois = get_roi_definitions(config)
    all_channels = tfr.ch_names
    roi_names = _get_comparison_rois(config, rois)

    column = require_config_value(config, "plotting.comparisons.comparison_column")
    values_spec = get_config_value(config, "plotting.comparisons.comparison_values", [])
    labels_spec = get_config_value(config, "plotting.comparisons.comparison_labels", None)

    if not isinstance(values_spec, (list, tuple)) or len(values_spec) < 1:
        raise ValueError(
            "plot_power_timecourse_by_condition requires plotting.comparisons.comparison_values "
            "with at least 1 value."
        )

    if len(values_spec) == 1:
        value = values_spec[0]
        label = str(labels_spec[0]).strip() if isinstance(labels_spec, (list, tuple)) and len(labels_spec) >= 1 else str(value)
        column_values = events_df[column]
        try:
            numeric_value = float(value)
            mask = (pd.to_numeric(column_values, errors="coerce") == numeric_value).values
        except (TypeError, ValueError):
            value_string = str(value).strip().lower()
            mask = (column_values.astype(str).str.strip().str.lower() == value_string).values
        if int(mask.sum()) == 0:
            raise ValueError(
                f"plot_power_timecourse_by_condition: no trials found for value {value!r} in column {column!r}"
            )
        conditions = [(label, mask)]
    elif len(values_spec) == 2:
        comp_mask_info = extract_comparison_mask(events_df, config, require_enabled=True)
        if not comp_mask_info:
            raise ValueError(
                "plot_power_timecourse_by_condition could not resolve configured comparison masks."
            )
        mask1, mask2, label1, label2 = comp_mask_info
        conditions = [(label1, mask1), (label2, mask2)]
    else:
        multi_group_info = extract_multi_group_masks(events_df, config, require_enabled=True)
        if not multi_group_info:
            raise ValueError(
                "plot_power_timecourse_by_condition could not resolve configured multi-group masks."
            )
        masks_dict, group_labels = multi_group_info
        conditions = [(label, masks_dict[label]) for label in group_labels]

    for roi_name in roi_names:
        if roi_name == "all":
            roi_channels = all_channels
        else:
            roi_channels = get_roi_channels(rois[roi_name], all_channels)

        if not roi_channels:
            if logger:
                logger.warning("No channels found for ROI %s, skipping timecourse plot", roi_name)
            continue

        tfr_roi = tfr.copy().pick(roi_channels)
        if len(tfr_roi.ch_names) == 0:
            if logger:
                logger.warning("No valid channels after filtering for ROI %s, skipping timecourse plot", roi_name)
            continue

        roi_safe = sanitize_label(roi_name).lower() if roi_name != "all" else ""
        roi_suffix = f"_roi-{roi_safe}" if roi_safe else ""
        plot_band_power_evolution(
            tfr_epochs=tfr_roi,
            conditions=conditions,
            subject=subject,
            save_dir=save_dir,
            logger=logger,
            config=config,
            roi_suffix=roi_suffix,
            roi_name=roi_name,
        )

def plot_band_power_topomaps(
    pow_df: pd.DataFrame,
    epochs_info: mne.Info,
    bands: List[str],
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    segment: str,
    events_df: Optional[pd.DataFrame] = None,
    *,
    sample_unit: str = "trials",
    label_suffix: Optional[str] = None,
) -> None:
    """Band power topomaps showing spatial distribution per frequency band.
    
    Creates MNE topomaps for each frequency band for a single segment.
    Supports condition-based filtering and optional column-based contrasts.
    
    Args:
        segment: Time window segment name (required, no fallback).
        events_df: Optional events DataFrame for condition-based filtering.
        label_suffix: Optional label to append to the title/filename when not using
            condition-based filtering (e.g., group mean for a specific condition).
    """
    if pow_df is None or epochs_info is None:
        return
    
    if not segment or segment.strip() == "":
        if logger:
            logger.error("plot_band_power_topomaps requires segment parameter. No fallback will be used.")
        return
    
    from eeg_pipeline.utils.analysis.events import extract_comparison_mask
    from eeg_pipeline.utils.config.loader import get_config_value


    compare_columns = bool(get_config_value(config, "plotting.comparisons.compare_columns", True))
    comparison_column = str(get_config_value(config, "plotting.comparisons.comparison_column", "") or "").strip()
    comparison_values = get_config_value(config, "plotting.comparisons.comparison_values", [])
    has_column_spec = (
        comparison_column != ""
        and isinstance(comparison_values, (list, tuple))
        and len(comparison_values) >= 2
    )

    if has_column_spec and len(comparison_values) != 2:
        raise ValueError(
            f"band_power_topomaps compare_columns requires exactly 2 comparison_values, "
            f"got {len(comparison_values)}: {comparison_values}"
        )

    conditions: Optional[List[Tuple[str, np.ndarray]]] = None
    if events_df is not None and has_column_spec:
        comp_mask_info = extract_comparison_mask(events_df, config, require_enabled=False)
        if not comp_mask_info:
            raise ValueError(
                "band_power_topomaps column comparison requested but could not resolve "
                f"comparison masks for column={comparison_column!r}, values={comparison_values!r}"
            )
        mask1, mask2, label1, label2 = comp_mask_info
        conditions = [(label1, mask1), (label2, mask2)]

    _plot_band_power_topomaps_single_segment(
        pow_df,
        epochs_info,
        bands,
        subject,
        save_dir,
        logger,
        config,
        segment,
        conditions,
        compare_columns,
        events_df,
        sample_unit,
        label_suffix,
    )
    
    
def _plot_band_power_topomaps_single_segment(
    pow_df: pd.DataFrame,
    epochs_info: mne.Info,
    bands: List[str],
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    segment: str,
    conditions: Optional[List[Tuple[str, np.ndarray]]],
    compare_columns: bool,
    events_df: Optional[pd.DataFrame],
    sample_unit: str,
    label_suffix: Optional[str] = None,
) -> None:
    """Internal helper to plot topomaps."""
    plot_cfg = get_plot_config(config)
    
    def extract_band_data(power_subset: pd.DataFrame) -> Tuple[List[str], Dict[str, Dict[str, float]], set]:
        """Extract band power data from a power DataFrame subset."""
        valid_bands = []
        band_data = {}
        detected_stats = set()
        
        for band in bands:
            cols = []
            for c in power_subset.columns:
                parsed = NamingSchema.parse(str(c))
                if not parsed.get("valid"):
                    continue
                if parsed.get("group") != "power":
                    continue
                if parsed.get("segment") != segment:
                    continue
                if parsed.get("band") != band:
                    continue
                if parsed.get("scope") != "ch":
                    continue
                cols.append(c)
                stat = parsed.get("stat", "")
                if stat:
                    detected_stats.add(stat)
            
            if not cols:
                continue
            
            ch_power = {}
            for col in cols:
                parsed = NamingSchema.parse(str(col))
                if parsed.get("valid") and parsed.get("identifier"):
                    ch_name = parsed["identifier"]
                    ch_power[ch_name] = power_subset[col].mean()
            
            if ch_power:
                valid_bands.append(band)
                band_data[band] = ch_power
        
        return valid_bands, band_data, detected_stats
    
    if conditions:
        valid_bands, band_data, detected_stats = extract_band_data(pow_df)
        if not valid_bands:
            if logger:
                logger.warning("No valid band power columns found for condition-based topomaps")
            return
    else:
        valid_bands, band_data, detected_stats = extract_band_data(pow_df)
    
    if not valid_bands:
        if logger:
            power_cols = [c for c in pow_df.columns if c.startswith("power_")]
            sample_cols = power_cols[:5] if len(power_cols) > 0 else []
            available_segments = set()
            for c in power_cols:
                parsed = NamingSchema.parse(str(c))
                if parsed.get("valid") and parsed.get("group") == "power":
                    seg = parsed.get("segment")
                    if seg:
                        available_segments.add(seg)
            logger.warning(
                f"No valid band power columns found for segment '{segment}'. "
                f"Available segments: {sorted(available_segments)}. "
                f"Available power columns: {len(power_cols)}. "
                f"Sample columns: {sample_cols}"
            )
        return
    
    # Determine unit label based on detected stat types
    # Use LaTeX math mode for proper rendering of subscripts
    STAT_TO_LABEL = {
        "logratio": r"$\log_{10}$(ratio)",
        "mean": r"power ($\mu$V²)",
        "baselined": "power (baseline-corrected)",
        "log10raw": r"$\log_{10}$(power)",
    }
    
    # Use the most common stat, or default to logratio if multiple stats found
    primary_stat = None
    if detected_stats:
        # Prefer logratio > mean > baselined > log10raw
        priority_order = ["logratio", "mean", "baselined", "log10raw"]
        for stat in priority_order:
            if stat in detected_stats:
                primary_stat = stat
                break
        if primary_stat is None:
            primary_stat = sorted(detected_stats)[0]
    
    unit_label = STAT_TO_LABEL.get(primary_stat, "power")
    value_label = f"mean {unit_label}" if primary_stat else "mean power"
    
    width_per_band = float(plot_cfg.plot_type_configs.get("power", {}).get("width_per_band", 3.5))
    
    from eeg_pipeline.utils.formatting import sanitize_label
    from eeg_pipeline.plotting.features.utils import apply_fdr_correction, get_fdr_alpha
    from scipy.stats import mannwhitneyu
    
    def save_topomap_plot(
        condition_pow_df: pd.DataFrame,
        condition_label: Optional[str],
        n_samples: int,
        value: str,
    ) -> None:
        """Create and save a topomap plot for a condition."""
        cond_valid_bands, cond_band_data, _ = extract_band_data(condition_pow_df)
        if not cond_valid_bands:
            return

        panel_arrays: Dict[str, np.ndarray] = {}
        panel_infos: Dict[str, mne.Info] = {}
        for band in cond_valid_bands:
            data_array, present_mask = _build_channel_data_array(cond_band_data[band], epochs_info)
            if present_mask.sum() <= MIN_CHANNELS_FOR_TOPO:
                continue
            panel_arrays[band] = data_array[present_mask]
            panel_infos[band] = mne.pick_info(epochs_info, np.where(present_mask)[0])

        if not panel_arrays:
            return

        vmin, vmax = _compute_shared_topomap_vlim(
            list(panel_arrays.values()),
            config,
            symmetric=False,
        )
        cmap = _resolve_topomap_colormap(vmin, vmax, symmetric=False)
        fig, axes = plt.subplots(1, len(cond_valid_bands), figsize=(width_per_band * len(cond_valid_bands), 4))
        fig.patch.set_facecolor("white")
        if len(cond_valid_bands) == 1:
            axes = [axes]

        shared_image = None
        for i, band in enumerate(cond_valid_bands):
            ax = axes[i]
            ax.set_facecolor("white")
            if band not in panel_arrays:
                ax.set_axis_off()
                continue

            shared_image, _ = plot_topomap(
                panel_arrays[band],
                panel_infos[band],
                axes=ax,
                show=False,
                cmap=cmap,
                contours=get_viz_params(config).get("topo_contours"),
                vlim=(vmin, vmax),
            )
            band_color = get_band_color(band, config)
            ax.set_title(f"{band.upper()}", fontweight="bold", color=band_color, fontsize=12)

        if shared_image is not None:
            _add_topomap_colorbar(fig, axes, shared_image, label=value)

        title_text = f"Descriptive band power topomaps | {segment.capitalize()}"
        effective_label = condition_label or (str(label_suffix).strip() if label_suffix is not None else "")
        effective_label = _format_topomap_condition_title_label(effective_label, config)
        if effective_label:
            title_text += f" | {effective_label}"
        fig.suptitle(title_text, fontsize=plot_cfg.font.figure_title, fontweight="bold", y=0.99)

        unit_label = str(sample_unit).strip() or "trials"
        footer = (
            f"Subject: {subject} | Descriptive mean across {n_samples} {unit_label} | "
            f"Values: {value}"
        )
        
        if effective_label:
            condition_safe = sanitize_label(effective_label).lower().replace(" ", "_")
            filename = f"sub-{subject}_band_power_topomaps_{segment}_{condition_safe}"
        else:
            filename = f"sub-{subject}_band_power_topomaps_{segment}"

        save_fig(
            fig,
            save_dir / filename,
            footer=footer,
            formats=plot_cfg.formats,
            dpi=plot_cfg.dpi,
            bbox_inches=plot_cfg.bbox_inches,
            pad_inches=plot_cfg.pad_inches,
            tight_layout_rect=(0, 0.04, 0.9, 0.98),
            config=config,
        )

    def _band_channel_columns(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
        """Map band -> channel -> column for this segment."""
        mapping: Dict[str, Dict[str, str]] = {str(b): {} for b in bands}
        for c in df.columns:
            parsed = NamingSchema.parse(str(c))
            if not parsed.get("valid"):
                continue
            if parsed.get("group") != "power":
                continue
            if parsed.get("segment") != segment:
                continue
            if parsed.get("scope") != "ch":
                continue
            band = str(parsed.get("band") or "")
            ch = str(parsed.get("identifier") or "")
            if band in mapping and ch:
                mapping[band][ch] = str(c)
        return mapping

    def save_column_comparison_topomaps() -> None:
        """If compare_columns, save statistical contrast topomaps with significance masks."""
        if not conditions:
            return
        if events_df is None:
            raise ValueError("band_power_topomaps compare_columns requires events_df.")
        if len(conditions) != 2:
            raise ValueError("band_power_topomaps compare_columns requires exactly 2 conditions.")

        band_to_ch_col = _band_channel_columns(pow_df)
        alpha = float(get_fdr_alpha(config))
        label1, mask1 = conditions[0]
        label2, mask2 = conditions[1]

        n_samples_1 = min(len(pow_df), len(mask1))
        n_samples_2 = min(len(pow_df), len(mask2))
        condition_df_1 = pow_df.iloc[:n_samples_1].loc[np.asarray(mask1[:n_samples_1], dtype=bool)]
        condition_df_2 = pow_df.iloc[:n_samples_2].loc[np.asarray(mask2[:n_samples_2], dtype=bool)]

        triptych_band_data_1 = extract_band_data(condition_df_1)[1]
        triptych_band_data_2 = extract_band_data(condition_df_2)[1]

        # Collect p-values across all band×channel tests for global FDR within this plot.
        tests: List[Tuple[str, str, float]] = []  # (band, ch, p)
        effect_map: Dict[Tuple[str, str], float] = {}

        for band in [str(b) for b in bands]:
            ch_to_col = band_to_ch_col.get(band, {})
            for ch_name in epochs_info.ch_names:
                col = ch_to_col.get(ch_name)
                if not col:
                    continue

                group_arrays: List[np.ndarray] = []
                group_means: List[float] = []
                for label, mask in conditions:
                    n_samples = min(len(pow_df), len(mask))
                    mask_array = np.asarray(mask[:n_samples], dtype=bool)
                    series = pd.to_numeric(pow_df.iloc[:n_samples][col], errors="coerce")
                    vals = series[mask_array].dropna().values
                    group_arrays.append(vals)
                    group_means.append(float(np.nanmean(vals)) if len(vals) else np.nan)

                if len(group_arrays) != 2:
                    continue

                v1, v2 = group_arrays[0], group_arrays[1]
                if len(v1) < MIN_TRIALS_FOR_STATISTICS or len(v2) < MIN_TRIALS_FOR_STATISTICS:
                    p = 1.0
                else:
                    p = float(mannwhitneyu(v1, v2, alternative="two-sided").pvalue)
                tests.append((band, ch_name, p))
                effect_map[(band, ch_name)] = (group_means[1] - group_means[0])

        if not tests:
            return

        rejected, qvals, _ = apply_fdr_correction([p for _, _, p in tests], config=config)
        q_map: Dict[Tuple[str, str], float] = {}
        sig_map: Dict[Tuple[str, str], bool] = {}
        for (band, ch, _), q, rej in zip(tests, qvals, rejected):
            q_map[(band, ch)] = float(q)
            sig_map[(band, ch)] = bool(rej) and float(q) < alpha

        viz_params = get_viz_params(config)
        n_bands = len([str(b) for b in bands])
        fig, axes = plt.subplots(1, n_bands, figsize=(width_per_band * n_bands, 4))
        fig.patch.set_facecolor("white")
        if n_bands == 1:
            axes = [axes]

        total_sig = 0
        total_tests = 0
        panel_arrays: List[np.ndarray] = []
        band_panel_data: Dict[str, Tuple[np.ndarray, np.ndarray, mne.Info]] = {}

        for band in [str(b) for b in bands]:
            data_array = np.full(len(epochs_info.ch_names), np.nan, dtype=float)
            sig_mask_full = np.zeros(len(epochs_info.ch_names), dtype=bool)

            for ch_idx, ch_name in enumerate(epochs_info.ch_names):
                key = (band, ch_name)
                if key not in effect_map:
                    continue
                data_array[ch_idx] = float(effect_map[key])
                if sig_map.get(key, False):
                    sig_mask_full[ch_idx] = True

            present = np.isfinite(data_array)
            if present.sum() <= MIN_CHANNELS_FOR_TOPO:
                continue

            valid_data = data_array[present]
            valid_info = mne.pick_info(epochs_info, np.where(present)[0])
            valid_sig = sig_mask_full[present]
            panel_arrays.append(valid_data)
            band_panel_data[band] = (valid_data, valid_sig, valid_info)

        if not band_panel_data:
            plt.close(fig)
            return

        vmin, vmax = _compute_shared_topomap_vlim(panel_arrays, config)
        shared_image = None

        for i, band in enumerate([str(b) for b in bands]):
            ax = axes[i]
            ax.set_facecolor("white")
            if band not in band_panel_data:
                ax.set_axis_off()
                continue

            valid_data, valid_sig, valid_info = band_panel_data[band]
            shared_image, _ = plot_topomap(
                valid_data,
                valid_info,
                axes=ax,
                show=False,
                cmap="RdBu_r",
                contours=viz_params.get("topo_contours"),
                vlim=(vmin, vmax),
                mask=valid_sig,
                mask_params=_build_topomap_mask_params(config),
            )

            band_color = get_band_color(band, config)
            ax.set_title(f"{band.upper()}", fontweight="bold", color=band_color, fontsize=12)

            # Count stats for footer
            for ch_name in epochs_info.ch_names:
                key = (band, ch_name)
                if key in q_map:
                    total_tests += 1
                    if sig_map.get(key, False):
                        total_sig += 1

        if shared_image is not None:
            _add_topomap_colorbar(fig, axes, shared_image, label=f"Δ {unit_label}")

        display_label1 = _format_condition_display_label(label1, config)
        display_label2 = _format_condition_display_label(label2, config)
        title = (
            f"Band power contrast | {segment.capitalize()} | "
            f"{display_label2} - {display_label1}"
        )
        footer = (
            f"Subject: {subject} | n={len(pow_df)} trials | "
            f"FDR α={alpha:.3f} | sig={total_sig}/{max(total_tests, 1)}"
        )
        fig.suptitle(title, fontsize=plot_cfg.font.figure_title, fontweight="bold", y=0.99)

        label1_safe = sanitize_label(str(conditions[0][0])).lower().replace(" ", "_")
        label2_safe = sanitize_label(str(conditions[1][0])).lower().replace(" ", "_")
        out = save_dir / f"sub-{subject}_band_power_topomaps_{segment}_contrast_{label2_safe}_minus_{label1_safe}"
        save_fig(
            fig,
            out,
            footer=footer,
            formats=plot_cfg.formats,
            dpi=plot_cfg.dpi,
            bbox_inches=plot_cfg.bbox_inches,
            pad_inches=plot_cfg.pad_inches,
            tight_layout_rect=(0, 0.04, 0.9, 0.98),
            config=config,
        )

        for band in [str(b) for b in bands]:
            condition_band_1 = triptych_band_data_1.get(band)
            condition_band_2 = triptych_band_data_2.get(band)
            contrast_panel_data = band_panel_data.get(band)
            if condition_band_1 is None or condition_band_2 is None or contrast_panel_data is None:
                continue

            descriptive_panel_1 = _build_topomap_panel(condition_band_1, epochs_info)
            descriptive_panel_2 = _build_topomap_panel(condition_band_2, epochs_info)
            if descriptive_panel_1 is None or descriptive_panel_2 is None:
                continue

            contrast_data, contrast_sig, contrast_info = contrast_panel_data
            band_total_tests = sum(1 for ch_name in epochs_info.ch_names if (band, ch_name) in q_map)
            band_total_sig = sum(
                1 for ch_name in epochs_info.ch_names if sig_map.get((band, ch_name), False)
            )
            _save_band_topomap_triptych(
                descriptive_panel_1=descriptive_panel_1,
                descriptive_panel_2=descriptive_panel_2,
                contrast_panel=(contrast_data, contrast_sig, contrast_info),
                band=band,
                subject=subject,
                save_path=save_dir / (
                    f"sub-{subject}_band_power_topomap_triptych_{segment}_band-{band}_"
                    f"{label2_safe}_minus_{label1_safe}"
                ),
                logger=logger,
                config=config,
                segment_label=segment.capitalize(),
                label1=label1,
                label2=label2,
                descriptive_value_label=value_label,
                contrast_value_label=f"Δ {unit_label}",
                footer=(
                    f"Subject: {subject} | n={len(condition_df_1)} vs {len(condition_df_2)} {sample_unit} | "
                    f"Band: {band.upper()} | FDR α={alpha:.3f} | sig={band_total_sig}/{max(band_total_tests, 1)}"
                ),
            )
    
    if conditions:
        for condition_label, condition_mask in conditions:
            n_samples = min(len(pow_df), len(condition_mask))
            if n_samples <= 0:
                continue
            
            condition_mask_array = np.asarray(condition_mask[:n_samples], dtype=bool)
            if int(condition_mask_array.sum()) == 0:
                continue
            
            condition_pow_df = pow_df.loc[condition_mask_array]
            n_trials = int(condition_mask_array.sum())
            save_topomap_plot(condition_pow_df, condition_label, n_trials, value_label)
        
        if logger:
            logger.debug(
                "Saved band power topomaps (%s) for %s conditions",
                segment,
                len(conditions),
            )
        if compare_columns:
            save_column_comparison_topomaps()
    else:
        save_topomap_plot(pow_df, None, len(pow_df), value_label)
        if logger:
            label_for_log = str(label_suffix).strip() if label_suffix is not None else ""
            if label_for_log:
                logger.debug(
                    "Saved band power topomaps (%s) | %s",
                    segment,
                    label_for_log,
                )
            else:
                logger.debug("Saved band power topomaps (%s)", segment)


def plot_band_power_topomaps_window_contrast(
    pow_df: pd.DataFrame,
    epochs_info: mne.Info,
    bands: List[str],
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    *,
    window1: str,
    window2: str,
) -> None:
    """Paired window contrast topomaps (window2 − window1) with FDR significance marks."""
    if pow_df is None or pow_df.empty or epochs_info is None:
        return

    if not window1 or not window2:
        raise ValueError("plot_band_power_topomaps_window_contrast requires window1 and window2.")

    from eeg_pipeline.plotting.features.utils import apply_fdr_correction, get_fdr_alpha
    from scipy.stats import wilcoxon

    plot_cfg = get_plot_config(config)
    viz_params = get_viz_params(config)
    alpha = float(get_fdr_alpha(config))
    stat_label = r"$\log_{10}$(ratio)"

    def _band_channel_columns(df: pd.DataFrame, segment_name: str) -> Dict[str, Dict[str, str]]:
        mapping: Dict[str, Dict[str, str]] = {str(b): {} for b in bands}
        for c in df.columns:
            parsed = NamingSchema.parse(str(c))
            if not parsed.get("valid"):
                continue
            if parsed.get("group") != "power":
                continue
            if parsed.get("segment") != segment_name:
                continue
            if parsed.get("scope") != "ch":
                continue
            band = str(parsed.get("band") or "")
            ch = str(parsed.get("identifier") or "")
            if band in mapping and ch:
                mapping[band][ch] = str(c)
        return mapping

    cols_w1 = _band_channel_columns(pow_df, window1)
    cols_w2 = _band_channel_columns(pow_df, window2)

    tests: List[Tuple[str, str, float]] = []  # (band, ch, p)
    diff_map: Dict[Tuple[str, str], float] = {}

    for band in [str(b) for b in bands]:
        ch_to_col1 = cols_w1.get(band, {})
        ch_to_col2 = cols_w2.get(band, {})
        for ch_name in epochs_info.ch_names:
            col1 = ch_to_col1.get(ch_name)
            col2 = ch_to_col2.get(ch_name)
            if not col1 or not col2:
                continue
            s1 = pd.to_numeric(pow_df[col1], errors="coerce")
            s2 = pd.to_numeric(pow_df[col2], errors="coerce")
            valid = s1.notna() & s2.notna()
            if int(valid.sum()) < MIN_TRIALS_FOR_STATISTICS:
                p = 1.0
                mean_diff = float(np.nanmean((s2 - s1).values))
            else:
                diffs = (s2[valid] - s1[valid]).values
                mean_diff = float(np.nanmean(diffs)) if diffs.size else np.nan
                if diffs.size == 0 or not np.isfinite(diffs).any() or np.allclose(diffs, 0):
                    p = 1.0
                else:
                    p = float(wilcoxon(diffs, zero_method="wilcox", alternative="two-sided").pvalue)
            tests.append((band, ch_name, p))
            diff_map[(band, ch_name)] = mean_diff

    if not tests:
        return

    rejected, qvals, _ = apply_fdr_correction([p for _, _, p in tests], config=config)
    q_map: Dict[Tuple[str, str], float] = {}
    sig_map: Dict[Tuple[str, str], bool] = {}
    for (band, ch, _), q, rej in zip(tests, qvals, rejected):
        q_map[(band, ch)] = float(q)
        sig_map[(band, ch)] = bool(rej) and float(q) < alpha

    width_per_band = float(plot_cfg.plot_type_configs.get("power", {}).get("width_per_band", 3.5))
    fig, axes = plt.subplots(1, len(bands), figsize=(width_per_band * len(bands), 4))
    fig.patch.set_facecolor("white")
    if len(bands) == 1:
        axes = [axes]

    total_sig = 0
    total_tests = 0
    panel_arrays: List[np.ndarray] = []
    band_panel_data: Dict[str, Tuple[np.ndarray, np.ndarray, mne.Info]] = {}

    for band in [str(b) for b in bands]:
        data_array = np.full(len(epochs_info.ch_names), np.nan, dtype=float)
        sig_mask_full = np.zeros(len(epochs_info.ch_names), dtype=bool)

        for ch_idx, ch_name in enumerate(epochs_info.ch_names):
            key = (band, ch_name)
            if key not in diff_map:
                continue
            data_array[ch_idx] = float(diff_map[key])
            if sig_map.get(key, False):
                sig_mask_full[ch_idx] = True
            if key in q_map:
                total_tests += 1
                if sig_map.get(key, False):
                    total_sig += 1

        present = np.isfinite(data_array)
        if present.sum() <= MIN_CHANNELS_FOR_TOPO:
            continue

        valid_data = data_array[present]
        valid_info = mne.pick_info(epochs_info, np.where(present)[0])
        valid_sig = sig_mask_full[present]
        band_panel_data[band] = (valid_data, valid_sig, valid_info)
        panel_arrays.append(valid_data)

    if not band_panel_data:
        plt.close(fig)
        return

    vmin, vmax = _compute_shared_topomap_vlim(panel_arrays, config)
    shared_image = None

    for i, band in enumerate([str(b) for b in bands]):
        ax = axes[i]
        ax.set_facecolor("white")
        if band not in band_panel_data:
            ax.set_axis_off()
            continue

        valid_data, valid_sig, valid_info = band_panel_data[band]
        shared_image, _ = plot_topomap(
            valid_data,
            valid_info,
            axes=ax,
            show=False,
            cmap="RdBu_r",
            contours=viz_params.get("topo_contours"),
            vlim=(vmin, vmax),
            mask=valid_sig,
            mask_params=_build_topomap_mask_params(config),
        )

        band_color = get_band_color(band, config)
        ax.set_title(f"{band.upper()}", fontweight="bold", color=band_color, fontsize=12)

    if shared_image is not None:
        _add_topomap_colorbar(fig, axes, shared_image, label="Δ power")

    fig.suptitle(
        f"Band power window contrast | {window2} - {window1}",
        fontsize=plot_cfg.font.figure_title,
        fontweight="bold",
        y=0.99,
    )
    footer = f"Subject: {subject} | n={len(pow_df)} trials | FDR α={alpha:.3f} | sig={total_sig}/{max(total_tests,1)}"
    save_fig(
        fig,
        save_dir / f"sub-{subject}_band_power_topomaps_contrast_{window2}_minus_{window1}",
        footer=footer,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
        tight_layout_rect=(0, 0.04, 0.9, 0.98),
        config=config,
    )

    for band in [str(b) for b in bands]:
        window1_columns = cols_w1.get(band, {})
        window2_columns = cols_w2.get(band, {})
        contrast_panel = band_panel_data.get(band)
        if not window1_columns or not window2_columns or contrast_panel is None:
            continue

        window1_values = {
            ch_name: float(pd.to_numeric(pow_df[col], errors="coerce").mean())
            for ch_name, col in window1_columns.items()
        }
        window2_values = {
            ch_name: float(pd.to_numeric(pow_df[col], errors="coerce").mean())
            for ch_name, col in window2_columns.items()
        }
        descriptive_panel_1 = _build_topomap_panel(window1_values, epochs_info)
        descriptive_panel_2 = _build_topomap_panel(window2_values, epochs_info)
        if descriptive_panel_1 is None or descriptive_panel_2 is None:
            continue

        contrast_data, contrast_sig, contrast_info = contrast_panel
        band_total_tests = sum(1 for ch_name in epochs_info.ch_names if (band, ch_name) in q_map)
        band_total_sig = sum(1 for ch_name in epochs_info.ch_names if sig_map.get((band, ch_name), False))
        _save_band_topomap_triptych(
            descriptive_panel_1=descriptive_panel_1,
            descriptive_panel_2=descriptive_panel_2,
            contrast_panel=(contrast_data, contrast_sig, contrast_info),
            band=band,
            subject=subject,
            save_path=save_dir / (
                f"sub-{subject}_band_power_topomap_triptych_{window2}_minus_{window1}_band-{band}"
            ),
            logger=logger,
            config=config,
            segment_label=f"{window1.capitalize()} vs {window2.capitalize()}",
            label1=window1.capitalize(),
            label2=window2.capitalize(),
            descriptive_value_label=f"mean {stat_label}",
            contrast_value_label="Δ power",
            footer=(
                f"Subject: {subject} | n={len(pow_df)} trials | Band: {band.upper()} | "
                f"FDR α={alpha:.3f} | sig={band_total_sig}/{max(band_total_tests, 1)}"
            ),
        )


def plot_band_power_topomaps_group_condition_contrast(
    *,
    pow_df_condition1: pd.DataFrame,
    pow_df_condition2: pd.DataFrame,
    epochs_info: mne.Info,
    bands: List[str],
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    segment: str,
    label1: str,
    label2: str,
    sample_unit: str = "subjects",
) -> None:
    """Paired condition contrast topomaps (label2 − label1) across samples (typically subjects).

    This is intended for group-level plotting, where each row in pow_df_condition*
    represents a subject-level summary (e.g., mean across trials within a condition).

    Statistical testing:
    - Paired Wilcoxon signed-rank test per band×channel.
    - Global FDR correction across all band×channel tests within this (segment) contrast.
    - If fewer than MIN_TRIALS_FOR_STATISTICS paired samples exist for a given test,
      p-values default to 1.0 and no significance markers will be shown.
    """
    if pow_df_condition1 is None or pow_df_condition2 is None:
        return
    if pow_df_condition1.empty or pow_df_condition2.empty or epochs_info is None:
        return

    if not segment or segment.strip() == "":
        if logger:
            logger.error(
                "plot_band_power_topomaps_group_condition_contrast requires segment parameter. No fallback will be used."
            )
        return

    from eeg_pipeline.plotting.features.utils import apply_fdr_correction, get_fdr_alpha
    from eeg_pipeline.utils.formatting import sanitize_label

    plot_cfg = get_plot_config(config)
    viz_params = get_viz_params(config)
    alpha = float(get_fdr_alpha(config))

    def _band_channel_columns(df: pd.DataFrame, segment_name: str) -> Dict[str, Dict[str, str]]:
        mapping: Dict[str, Dict[str, str]] = {str(b): {} for b in bands}
        for c in df.columns:
            parsed = NamingSchema.parse(str(c))
            if not parsed.get("valid"):
                continue
            if parsed.get("group") != "power":
                continue
            if parsed.get("segment") != segment_name:
                continue
            if parsed.get("scope") != "ch":
                continue
            band = str(parsed.get("band") or "")
            ch = str(parsed.get("identifier") or "")
            if band in mapping and ch:
                mapping[band][ch] = str(c)
        return mapping

    cols1 = _band_channel_columns(pow_df_condition1, segment)
    cols2 = _band_channel_columns(pow_df_condition2, segment)

    tests: List[Tuple[str, str, float]] = []  # (band, ch, p)
    diff_map: Dict[Tuple[str, str], float] = {}

    for band in [str(b) for b in bands]:
        ch_to_col1 = cols1.get(band, {})
        ch_to_col2 = cols2.get(band, {})
        for ch_name in epochs_info.ch_names:
            col1 = ch_to_col1.get(ch_name)
            col2 = ch_to_col2.get(ch_name)
            if not col1 or not col2:
                continue

            s1 = pd.to_numeric(pow_df_condition1[col1], errors="coerce")
            s2 = pd.to_numeric(pow_df_condition2[col2], errors="coerce")
            valid = s1.notna() & s2.notna()
            diffs = (s2[valid] - s1[valid]).values

            mean_diff = float(np.nanmean(diffs)) if diffs.size else np.nan
            diff_map[(band, ch_name)] = mean_diff

            if int(valid.sum()) < MIN_TRIALS_FOR_STATISTICS:
                p = 1.0
            else:
                # Wilcoxon cannot run on all-zero diffs.
                if diffs.size == 0 or not np.isfinite(diffs).any() or np.allclose(diffs, 0):
                    p = 1.0
                else:
                    try:
                        p = float(wilcoxon(diffs, zero_method="wilcox", alternative="two-sided").pvalue)
                    except ValueError:
                        p = 1.0
            tests.append((band, ch_name, p))

    if not tests:
        return

    rejected, qvals, _ = apply_fdr_correction([p for _, _, p in tests], config=config)
    q_map: Dict[Tuple[str, str], float] = {}
    sig_map: Dict[Tuple[str, str], bool] = {}
    for (band, ch, _), q, rej in zip(tests, qvals, rejected):
        q_map[(band, ch)] = float(q)
        sig_map[(band, ch)] = bool(rej) and float(q) < alpha

    width_per_band = float(plot_cfg.plot_type_configs.get("power", {}).get("width_per_band", 3.5))
    fig, axes = plt.subplots(1, len(bands), figsize=(width_per_band * len(bands), 4))
    fig.patch.set_facecolor("white")
    if len(bands) == 1:
        axes = [axes]

    total_sig = 0
    total_tests = 0
    panel_arrays: List[np.ndarray] = []
    band_panel_data: Dict[str, Tuple[np.ndarray, np.ndarray, mne.Info]] = {}

    for band in [str(b) for b in bands]:
        data_array = np.full(len(epochs_info.ch_names), np.nan, dtype=float)
        sig_mask_full = np.zeros(len(epochs_info.ch_names), dtype=bool)

        for ch_idx, ch_name in enumerate(epochs_info.ch_names):
            key = (band, ch_name)
            if key not in diff_map:
                continue
            data_array[ch_idx] = float(diff_map[key])
            if sig_map.get(key, False):
                sig_mask_full[ch_idx] = True

        present = np.isfinite(data_array)
        if present.sum() <= MIN_CHANNELS_FOR_TOPO:
            continue

        valid_data = data_array[present]
        valid_info = mne.pick_info(epochs_info, np.where(present)[0])
        valid_sig = sig_mask_full[present]
        band_panel_data[band] = (valid_data, valid_sig, valid_info)
        panel_arrays.append(valid_data)

        for ch_name in epochs_info.ch_names:
            key = (band, ch_name)
            if key in q_map:
                total_tests += 1
                if sig_map.get(key, False):
                    total_sig += 1

    if not band_panel_data:
        plt.close(fig)
        return

    vmin, vmax = _compute_shared_topomap_vlim(panel_arrays, config)
    shared_image = None

    for i, band in enumerate([str(b) for b in bands]):
        ax = axes[i]
        ax.set_facecolor("white")
        if band not in band_panel_data:
            ax.set_axis_off()
            continue

        valid_data, valid_sig, valid_info = band_panel_data[band]
        shared_image, _ = plot_topomap(
            valid_data,
            valid_info,
            axes=ax,
            show=False,
            cmap="RdBu_r",
            contours=viz_params.get("topo_contours"),
            vlim=(vmin, vmax),
            mask=valid_sig,
            mask_params=_build_topomap_mask_params(config),
        )

        band_color = get_band_color(band, config)
        ax.set_title(f"{band.upper()}", fontweight="bold", color=band_color, fontsize=12)

    if shared_image is not None:
        _add_topomap_colorbar(fig, axes, shared_image, label="Δ power")

    display_label1 = _format_condition_display_label(label1, config)
    display_label2 = _format_condition_display_label(label2, config)
    descriptive_value_label = "mean power"
    fig.suptitle(
        f"Band power contrast | {segment.capitalize()} | {display_label2} - {display_label1}",
        fontsize=plot_cfg.font.figure_title,
        fontweight="bold",
        y=0.99,
    )
    footer = (
        f"Subject: {subject} | n={len(pow_df_condition1)} {sample_unit} | "
        f"FDR α={alpha:.3f} | sig={total_sig}/{max(total_tests,1)}"
    )

    label1_safe = sanitize_label(str(label1)).lower().replace(" ", "_")
    label2_safe = sanitize_label(str(label2)).lower().replace(" ", "_")
    out = save_dir / f"sub-{subject}_band_power_topomaps_{segment}_contrast_{label2_safe}_minus_{label1_safe}"
    save_fig(
        fig,
        out,
        footer=footer,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
        tight_layout_rect=(0, 0.04, 0.9, 0.98),
        config=config,
    )

    for band in [str(b) for b in bands]:
        window_condition_1 = cols1.get(band, {})
        window_condition_2 = cols2.get(band, {})
        contrast_panel = band_panel_data.get(band)
        if not window_condition_1 or not window_condition_2 or contrast_panel is None:
            continue

        condition_values_1 = {
            ch_name: float(pd.to_numeric(pow_df_condition1[col], errors="coerce").mean())
            for ch_name, col in window_condition_1.items()
        }
        condition_values_2 = {
            ch_name: float(pd.to_numeric(pow_df_condition2[col], errors="coerce").mean())
            for ch_name, col in window_condition_2.items()
        }
        descriptive_panel_1 = _build_topomap_panel(condition_values_1, epochs_info)
        descriptive_panel_2 = _build_topomap_panel(condition_values_2, epochs_info)
        if descriptive_panel_1 is None or descriptive_panel_2 is None:
            continue

        contrast_data, contrast_sig, contrast_info = contrast_panel
        band_total_tests = sum(1 for ch_name in epochs_info.ch_names if (band, ch_name) in q_map)
        band_total_sig = sum(1 for ch_name in epochs_info.ch_names if sig_map.get((band, ch_name), False))
        _save_band_topomap_triptych(
            descriptive_panel_1=descriptive_panel_1,
            descriptive_panel_2=descriptive_panel_2,
            contrast_panel=(contrast_data, contrast_sig, contrast_info),
            band=band,
            subject=subject,
            save_path=save_dir / (
                f"sub-{subject}_band_power_topomap_triptych_{segment}_band-{band}_"
                f"{label2_safe}_minus_{label1_safe}"
            ),
            logger=logger,
            config=config,
            segment_label=segment.capitalize(),
            label1=label1,
            label2=label2,
            descriptive_value_label=descriptive_value_label,
            contrast_value_label="Δ power",
            footer=(
                f"Group: {subject} | n={len(pow_df_condition1)} {sample_unit} | "
                f"Band: {band.upper()} | FDR α={alpha:.3f} | sig={band_total_sig}/{max(band_total_tests, 1)}"
            ),
        )


def _compute_group_curve_significance_mask(
    condition1_values: np.ndarray,
    condition2_values: np.ndarray,
    config: Any,
) -> np.ndarray:
    """Return an FDR-corrected significance mask across curve samples for paired group data."""
    from eeg_pipeline.plotting.features.utils import apply_fdr_correction, get_fdr_alpha

    if condition1_values.ndim != 2 or condition2_values.ndim != 2:
        raise ValueError("Group curve significance expects 2D subject x sample arrays.")
    if condition1_values.shape != condition2_values.shape:
        raise ValueError("Group curve significance arrays must share the same shape.")

    n_subjects, n_samples = condition1_values.shape
    significant_mask = np.zeros(n_samples, dtype=bool)
    if n_subjects < MIN_TRIALS_FOR_STATISTICS:
        return significant_mask

    pvalues: List[float] = []
    valid_sample_indices: List[int] = []

    for sample_index in range(n_samples):
        values1 = np.asarray(condition1_values[:, sample_index], dtype=float)
        values2 = np.asarray(condition2_values[:, sample_index], dtype=float)
        finite_mask = np.isfinite(values1) & np.isfinite(values2)
        if int(finite_mask.sum()) < MIN_TRIALS_FOR_STATISTICS:
            continue

        diffs = values2[finite_mask] - values1[finite_mask]
        if diffs.size == 0 or np.allclose(diffs, 0):
            p_value = 1.0
        else:
            p_value = float(wilcoxon(diffs, zero_method="wilcox", alternative="two-sided").pvalue)

        pvalues.append(p_value)
        valid_sample_indices.append(sample_index)

    if not pvalues:
        return significant_mask

    rejected, qvalues, _ = apply_fdr_correction(pvalues, config=config)
    alpha = float(get_fdr_alpha(config))
    for sample_index, rejected_flag, q_value in zip(valid_sample_indices, rejected, qvalues):
        significant_mask[sample_index] = bool(rejected_flag) and float(q_value) < alpha

    return significant_mask


def _compute_group_timecourse_significance_mask(
    condition1_values: np.ndarray,
    condition2_values: np.ndarray,
    config: Any,
) -> np.ndarray:
    """Return an FDR-corrected significance mask across time for paired group traces."""
    return _compute_group_curve_significance_mask(condition1_values, condition2_values, config)


def _compute_group_band_summary_stats(
    condition1_values: np.ndarray,
    condition2_values: np.ndarray,
    freqs: np.ndarray,
    frequency_bands: Dict[str, Tuple[float, float]],
    config: Any,
) -> Dict[str, Dict[str, float]]:
    """Return paired band-summary statistics for group PSD curves."""
    from eeg_pipeline.plotting.features.utils import apply_fdr_correction, get_fdr_alpha

    if freqs.ndim != 1:
        raise ValueError("Group band summary requires a 1D frequency axis.")
    if condition1_values.ndim != 2 or condition2_values.ndim != 2:
        raise ValueError("Group band summary requires 2D subject x frequency arrays.")
    if condition1_values.shape != condition2_values.shape:
        raise ValueError("Group band summary arrays must share the same shape.")
    if condition1_values.shape[1] != len(freqs):
        raise ValueError("Group band summary arrays must match the frequency axis length.")

    band_stats: Dict[str, Dict[str, float]] = {}
    pvalues: List[float] = []
    tested_bands: List[str] = []

    for band_name, bounds in frequency_bands.items():
        if not isinstance(bounds, (list, tuple)) or len(bounds) < 2:
            continue

        band_min = float(bounds[0])
        band_max = float(bounds[1])
        band_mask = (freqs >= band_min) & (freqs <= band_max)
        if not np.any(band_mask):
            continue

        values1 = np.nanmean(condition1_values[:, band_mask], axis=1)
        values2 = np.nanmean(condition2_values[:, band_mask], axis=1)
        finite_mask = np.isfinite(values1) & np.isfinite(values2)
        if int(finite_mask.sum()) < MIN_TRIALS_FOR_STATISTICS:
            continue

        finite_values1 = values1[finite_mask]
        finite_values2 = values2[finite_mask]
        effect_size = _compute_plot_paired_effect_size(finite_values1, finite_values2)
        diffs = finite_values2 - finite_values1
        if diffs.size == 0 or np.allclose(diffs, 0):
            p_value = 1.0
            include_in_fdr = False
        else:
            p_value = float(wilcoxon(diffs, zero_method="wilcox", alternative="two-sided").pvalue)
            include_in_fdr = True

        band_stats[str(band_name)] = {
            "fmin": band_min,
            "fmax": band_max,
            "effect_size": float(effect_size),
            "p_value": p_value,
            "q_value": 1.0 if not include_in_fdr else np.nan,
            "significant": False,
        }
        if include_in_fdr:
            pvalues.append(p_value)
            tested_bands.append(str(band_name))

    if not pvalues:
        return band_stats

    rejected, qvalues, _ = apply_fdr_correction(pvalues, config=config)
    alpha = float(get_fdr_alpha(config))
    for band_name, rejected_flag, q_value in zip(tested_bands, rejected, qvalues):
        band_stats[band_name]["q_value"] = float(q_value)
        band_stats[band_name]["significant"] = bool(rejected_flag) and float(q_value) < alpha

    return band_stats


def _iter_true_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return inclusive index runs where the boolean mask is true."""
    runs: List[Tuple[int, int]] = []
    run_start: Optional[int] = None

    for index, flag in enumerate(np.asarray(mask, dtype=bool)):
        if flag and run_start is None:
            run_start = index
        if not flag and run_start is not None:
            runs.append((run_start, index - 1))
            run_start = None

    if run_start is not None:
        runs.append((run_start, len(mask) - 1))
    return runs


def _draw_curve_significance_strip(
    ax: Any,
    x_values: np.ndarray,
    significant_mask: np.ndarray,
    *,
    label: str = "FDR q<0.05",
) -> None:
    """Draw a compact significance strip along the bottom of a 1D curve axis."""
    if x_values.ndim != 1:
        raise ValueError("Curve significance strip expects a 1D axis.")
    if significant_mask.shape != x_values.shape:
        raise ValueError("Curve significance mask must match the axis shape.")
    if not significant_mask.any():
        return

    if len(x_values) > 1:
        sample_step = float(np.nanmedian(np.diff(x_values)))
    else:
        sample_step = 0.0

    for start_index, end_index in _iter_true_runs(significant_mask):
        start_value = float(x_values[start_index] - 0.5 * sample_step)
        end_value = float(x_values[end_index] + 0.5 * sample_step)
        ax.axvspan(
            start_value,
            end_value,
            ymin=0.02,
            ymax=0.055,
            color="0.15",
            alpha=0.16,
            linewidth=0,
            zorder=0,
        )

    ax.text(
        0.98,
        0.06,
        label,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.35",
    )


def _draw_timecourse_significance_strip(
    ax: Any,
    times: np.ndarray,
    significant_mask: np.ndarray,
) -> None:
    """Draw a compact significance strip along the bottom of the timecourse axis."""
    _draw_curve_significance_strip(ax, times, significant_mask, label="FDR q<0.05")


def _draw_psd_band_summary_strip(
    ax: Any,
    band_stats: Dict[str, Dict[str, float]],
) -> None:
    """Draw a compact canonical-band significance strip above the curve strip."""
    significant_bands = [
        stats
        for stats in band_stats.values()
        if bool(stats.get("significant", False))
    ]
    if not significant_bands:
        return

    for stats in significant_bands:
        ax.axvspan(
            float(stats["fmin"]),
            float(stats["fmax"]),
            ymin=0.07,
            ymax=0.11,
            color="0.1",
            alpha=0.18,
            linewidth=0,
            zorder=0,
        )

    ax.text(
        0.98,
        0.115,
        "Band q<0.05",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.35",
    )


def _draw_group_subject_traces(
    ax: Any,
    times: np.ndarray,
    subject_matrix: np.ndarray,
    *,
    color: Any,
) -> None:
    """Draw restrained subject-level traces behind the group mean."""
    if times.ndim != 1:
        raise ValueError("Group subject traces require a 1D time axis.")
    if subject_matrix.ndim != 2:
        raise ValueError("Group subject traces require a 2D subject x time matrix.")
    if subject_matrix.shape[1] != len(times):
        raise ValueError("Group subject trace matrix must match the time axis length.")

    n_subjects = subject_matrix.shape[0]
    linewidth = 0.7 if n_subjects <= 12 else 0.55
    alpha = 0.14 if n_subjects <= 12 else 0.08

    for subject_trace in subject_matrix:
        trace = np.asarray(subject_trace, dtype=float)
        finite_mask = np.isfinite(trace)
        if int(finite_mask.sum()) < 2:
            continue
        ax.plot(
            times[finite_mask],
            trace[finite_mask],
            color=color,
            lw=linewidth,
            alpha=alpha,
            zorder=1,
        )


def plot_group_band_power_evolution(
    *,
    times: np.ndarray,
    group_series_by_band: Dict[str, Dict[str, np.ndarray]],
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    baseline_window: Tuple[float, float],
    active_window: Tuple[float, float],
    roi_suffix: str = "",
    roi_name: Optional[str] = None,
) -> bool:
    """Plot one group-level timecourse figure per band using subject-level summaries."""
    if times.ndim != 1 or len(times) == 0:
        raise ValueError("plot_group_band_power_evolution requires a non-empty 1D time axis.")
    if not group_series_by_band:
        raise ValueError("plot_group_band_power_evolution requires grouped subject timecourses.")

    plot_cfg = get_plot_config(config)
    condition_labels = list(
        dict.fromkeys(
            label
            for band_series in group_series_by_band.values()
            for label in band_series.keys()
        )
    )
    if not condition_labels:
        raise ValueError("plot_group_band_power_evolution requires at least one condition label.")

    condition_color_map = _get_condition_color_map(condition_labels, config)
    band_colors = {band: get_band_color(band, config) for band in group_series_by_band}
    roi_display = roi_name.replace("_", " ").title() if roi_name and roi_name != "all" else "All Channels"
    rendered_bands = 0

    for band_name, label_series in group_series_by_band.items():
        if not label_series:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(5.4, 4.1))
        fig.patch.set_facecolor("white")
        _style_publication_axis(ax)

        subject_count: Optional[int] = None
        plotted_labels = 0

        for label in condition_labels:
            subject_matrix = label_series.get(label)
            if subject_matrix is None:
                continue
            subject_matrix = np.asarray(subject_matrix, dtype=float)
            if subject_matrix.ndim != 2 or subject_matrix.shape[1] != len(times):
                raise ValueError(
                    f"Group timecourse for band {band_name!r}, condition {label!r} has invalid shape "
                    f"{subject_matrix.shape}; expected (n_subjects, {len(times)})."
                )
            if subject_matrix.shape[0] < 2:
                continue

            color = condition_color_map[label]
            _draw_group_subject_traces(
                ax,
                times,
                subject_matrix,
                color=color,
            )

            mean_trace = np.nanmean(subject_matrix, axis=0)
            sem_trace = np.nanstd(subject_matrix, axis=0, ddof=1) / np.sqrt(subject_matrix.shape[0])
            ci_lower = mean_trace - 1.96 * sem_trace
            ci_upper = mean_trace + 1.96 * sem_trace

            ax.plot(
                times,
                mean_trace,
                color=color,
                lw=2,
                label=(
                    f"{_format_condition_display_label(label, config)} "
                    f"(n={subject_matrix.shape[0]})"
                ),
            )
            ax.fill_between(times, ci_lower, ci_upper, color=color, alpha=0.2, lw=0)
            subject_count = subject_matrix.shape[0]
            plotted_labels += 1

        if plotted_labels == 0 or subject_count is None:
            plt.close(fig)
            continue

        if len(condition_labels) == 2:
            first = label_series.get(condition_labels[0])
            second = label_series.get(condition_labels[1])
            if first is not None and second is not None:
                first = np.asarray(first, dtype=float)
                second = np.asarray(second, dtype=float)
                if first.shape == second.shape and first.ndim == 2 and first.shape[0] >= MIN_TRIALS_FOR_STATISTICS:
                    significant_mask = _compute_group_timecourse_significance_mask(first, second, config)
                    _draw_timecourse_significance_strip(ax, times, significant_mask)

        ax.axvspan(baseline_window[0], baseline_window[1], color="0.88", alpha=0.7, zorder=0)
        ax.axvspan(active_window[0], active_window[1], color=band_colors.get(band_name, "0.7"), alpha=0.08, zorder=0)
        ax.axhline(1.0, color="0.4", linestyle="--", alpha=0.5, lw=1)
        ax.axvline(0.0, color="0.4", linestyle="--", alpha=0.5, lw=1)
        ax.axvline(active_window[0], color="0.6", linestyle=":", alpha=0.8, lw=0.9)
        ax.set_xlim(float(times[0]), float(times[-1]))
        ax.set_title(
            band_name.capitalize(),
            color=band_colors.get(band_name, "0.2"),
            fontweight="bold",
        )
        ax.set_xlabel("Time (s)", fontsize=plot_cfg.font.medium)
        ax.set_ylabel("Power / baseline", fontsize=plot_cfg.font.medium)
        ax.legend(loc="upper left", frameon=False, fontsize=plot_cfg.font.small)
        ax.yaxis.grid(True, alpha=0.18, linewidth=0.6)
        ax.xaxis.grid(False)
        ax.text(
            0.02,
            0.02,
            "Thin lines: subjects | thick line: group mean",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=plot_cfg.font.small,
            color="0.4",
        )
        sns.despine(ax=ax, trim=True)

        fig.suptitle(
            f"Group time-resolved band power | {roi_display} | {band_name.capitalize()}",
            fontsize=plot_cfg.font.figure_title,
            fontweight="bold",
            y=0.99,
        )
        footer = (
            f"Group: {subject} | n={subject_count} subjects | "
            f"Baseline: [{baseline_window[0]:.2f}, {baseline_window[1]:.2f}] s | "
            f"Active window: [{active_window[0]:.2f}, {active_window[1]:.2f}] s | "
            "Thin lines: subject means | thick line: between-subject mean ± 95% CI"
        )
        save_fig(
            fig,
            save_dir / f"sub-{subject}_power_timecourse_band-{band_name}{roi_suffix}",
            footer=footer,
            formats=plot_cfg.formats,
            dpi=plot_cfg.dpi,
            bbox_inches=plot_cfg.bbox_inches,
            pad_inches=plot_cfg.pad_inches,
            tight_layout_rect=(0, 0.04, 1, 0.98),
            config=config,
        )
        rendered_bands += 1

    if rendered_bands == 0:
        logger.warning("No group band power timecourses were rendered.")
        return False
    return True


def plot_group_band_power_effect_evolution(
    *,
    times: np.ndarray,
    group_series_by_band: Dict[str, Dict[str, np.ndarray]],
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    baseline_window: Tuple[float, float],
    active_window: Tuple[float, float],
    roi_suffix: str = "",
    roi_name: Optional[str] = None,
) -> bool:
    """Plot one paired group-level effect figure per band using condition2 - condition1."""
    if times.ndim != 1 or len(times) == 0:
        raise ValueError("plot_group_band_power_effect_evolution requires a non-empty 1D time axis.")
    if not group_series_by_band:
        raise ValueError("plot_group_band_power_effect_evolution requires grouped subject timecourses.")

    plot_cfg = get_plot_config(config)
    condition_labels = list(
        dict.fromkeys(
            label
            for band_series in group_series_by_band.values()
            for label in band_series.keys()
        )
    )
    if len(condition_labels) != 2:
        raise ValueError(
            "plot_group_band_power_effect_evolution requires exactly two condition labels."
        )

    label1, label2 = condition_labels
    band_colors = {band: get_band_color(band, config) for band in group_series_by_band}
    roi_display = roi_name.replace("_", " ").title() if roi_name and roi_name != "all" else "All Channels"
    rendered_bands = 0

    for band_name, label_series in group_series_by_band.items():
        first = label_series.get(label1)
        second = label_series.get(label2)
        if first is None or second is None:
            continue

        effect_matrix = _compute_paired_effect_matrix(
            first,
            second,
            context=f"group timecourse effect for {band_name}",
        )
        if effect_matrix.shape[0] < 2 or effect_matrix.shape[1] != len(times):
            continue

        fig, ax = plt.subplots(1, 1, figsize=(5.4, 4.1))
        fig.patch.set_facecolor("white")
        _style_publication_axis(ax)

        effect_color = band_colors.get(band_name, "0.2")
        _draw_group_subject_traces(
            ax,
            times,
            effect_matrix,
            color=effect_color,
        )

        mean_trace = np.nanmean(effect_matrix, axis=0)
        sem_trace = np.nanstd(effect_matrix, axis=0, ddof=1) / np.sqrt(effect_matrix.shape[0])
        ci_lower = mean_trace - 1.96 * sem_trace
        ci_upper = mean_trace + 1.96 * sem_trace

        display_label_1 = _format_condition_display_label(label1, config)
        display_label_2 = _format_condition_display_label(label2, config)
        ax.plot(
            times,
            mean_trace,
            color=effect_color,
            lw=2.2,
            label=f"{display_label_2} - {display_label_1} (n={effect_matrix.shape[0]})",
        )
        ax.fill_between(times, ci_lower, ci_upper, color=effect_color, alpha=0.2, lw=0)

        significant_mask = _compute_group_curve_significance_mask(
            np.asarray(first, dtype=float),
            np.asarray(second, dtype=float),
            config,
        )
        _draw_timecourse_significance_strip(ax, times, significant_mask)

        ax.axvspan(baseline_window[0], baseline_window[1], color="0.88", alpha=0.7, zorder=0)
        ax.axvspan(active_window[0], active_window[1], color=effect_color, alpha=0.08, zorder=0)
        ax.axhline(0.0, color="0.4", linestyle="--", alpha=0.6, lw=1)
        ax.axvline(0.0, color="0.4", linestyle="--", alpha=0.5, lw=1)
        ax.axvline(active_window[0], color="0.6", linestyle=":", alpha=0.8, lw=0.9)
        ax.set_xlim(float(times[0]), float(times[-1]))
        ax.set_title(
            band_name.capitalize(),
            color=effect_color,
            fontweight="bold",
        )
        ax.set_xlabel("Time (s)", fontsize=plot_cfg.font.medium)
        ax.set_ylabel("Δ power / baseline", fontsize=plot_cfg.font.medium)
        ax.legend(loc="upper left", frameon=False, fontsize=plot_cfg.font.small)
        _draw_effect_window_summary_inset(
            ax,
            effect_values=_compute_window_mean_series(
                effect_matrix,
                times,
                active_window,
                context=f"group timecourse active-window effect for {band_name}",
            ),
            color=effect_color,
            config=config,
        )
        ax.yaxis.grid(True, alpha=0.18, linewidth=0.6)
        ax.xaxis.grid(False)
        ax.text(
            0.02,
            0.02,
            "Thin lines: paired subject effects | thick line: group mean effect",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=plot_cfg.font.small,
            color="0.4",
        )
        sns.despine(ax=ax, trim=True)

        fig.suptitle(
            f"Group time-resolved power effect | {roi_display} | {band_name.capitalize()}",
            fontsize=plot_cfg.font.figure_title,
            fontweight="bold",
            y=0.99,
        )
        footer = (
            f"Group: {subject} | n={effect_matrix.shape[0]} subjects | "
            f"Effect: {display_label_2} - {display_label_1} | "
            f"Baseline: [{baseline_window[0]:.2f}, {baseline_window[1]:.2f}] s | "
            f"Active window: [{active_window[0]:.2f}, {active_window[1]:.2f}] s | "
            "Thin lines: paired subject effects | thick line: mean paired effect ± 95% CI | "
            "Inset: active-window paired effects"
        )
        save_fig(
            fig,
            save_dir / f"sub-{subject}_power_timecourse_effect_band-{band_name}{roi_suffix}",
            footer=footer,
            formats=plot_cfg.formats,
            dpi=plot_cfg.dpi,
            bbox_inches=plot_cfg.bbox_inches,
            pad_inches=plot_cfg.pad_inches,
            tight_layout_rect=(0, 0.04, 1, 0.98),
            config=config,
        )
        rendered_bands += 1

    if rendered_bands == 0:
        logger.warning("No group band power effect timecourses were rendered.")
        return False
    return True


def plot_band_power_evolution(
    tfr_epochs: Any,
    conditions: List[Tuple[str, np.ndarray]],
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    roi_suffix: str = "",
    roi_name: Optional[str] = None,
) -> bool:
    """Plot continuous time-resolved band power trace with 95% CI.
    
    Shows the Time vs Power trajectory per frequency band to better 
    illustrate the onset/offset of ERS/ERD dynamics relative to events.
    """
    if len(conditions) < 1:
        raise ValueError(
            f"plot_band_power_evolution requires at least 1 condition, got {len(conditions)}"
        )

    from eeg_pipeline.utils.config.loader import get_frequency_bands
    from eeg_pipeline.plotting.config import get_plot_config
    
    plot_cfg = get_plot_config(config)
    freq_bands = get_frequency_bands(config)
    
    active_window = _get_active_window(config)
    tfr_baseline = _get_plotting_tfr_baseline_window(config)
    display_start = float(min(tfr_baseline[0], 0.0))
    display_end = float(active_window[1])
    
    # Check if there are valid trials to avoid empty plots
    has_valid_data = False
    for label, mask in conditions:
        if int(mask.sum()) > 0:
            has_valid_data = True
            break
            
    if not has_valid_data:
        logger.warning("No valid trials found across conditions for plot_band_power_evolution.")
        return False

    condition_color_map = _get_condition_color_map([label for label, _ in conditions], config)
    band_colors = {band: get_band_color(band, config) for band in freq_bands.keys()}
    band_series: Dict[str, Dict[str, Any]] = {band_name: {} for band_name in freq_bands.keys()}

    for label, mask in conditions:
        n_trials_cond = int(mask.sum())
        if n_trials_cond < 1:
            continue
            
        tfr_cond = tfr_epochs[mask]
        
        psd_per_trial = []
        for trial_idx in range(len(tfr_cond)):
            tfr_trial = tfr_cond[[trial_idx]].average()
            apply_baseline_and_crop(
                tfr_trial,
                baseline=tfr_baseline,
                mode=TIMECOURSE_BASELINE_MODE,
                logger=logger,
            )
            times = np.asarray(tfr_trial.times)
            tmin = max(float(times.min()), display_start)
            tmax = min(float(times.max()), display_end)
            if tmax <= tmin:
                continue
            tfr_trial_display = tfr_trial.copy().crop(tmin, tmax)
            psd_per_trial.append(tfr_trial_display.data)
                
        if not psd_per_trial:
            continue
            
        psd_per_trial = np.array(psd_per_trial)
        psd_per_trial_roi = psd_per_trial.mean(axis=1)
        freqs = tfr_trial_display.freqs
        times = tfr_trial_display.times
        
        for band_name, (fmin, fmax) in freq_bands.items():
            fmask = (freqs >= fmin) & (freqs <= fmax)
            if not np.any(fmask):
                continue
                
            band_power_trials = psd_per_trial_roi[:, fmask, :].mean(axis=1)
            band_mean = band_power_trials.mean(axis=0)
            band_sem = band_power_trials.std(axis=0, ddof=1) / np.sqrt(len(band_power_trials))
            band_series[band_name][label] = {
                "times": times,
                "mean": band_mean,
                "ci_lower": band_mean - 1.96 * band_sem,
                "ci_upper": band_mean + 1.96 * band_sem,
                "n_trials": n_trials_cond,
            }

    roi_display = roi_name.replace("_", " ").title() if roi_name and roi_name != "all" else "All Channels"
    footer = (
        f"Subject: {subject} | Baseline: [{tfr_baseline[0]:.2f}, {tfr_baseline[1]:.2f}] s | "
        f"Active window: [{active_window[0]:.2f}, {active_window[1]:.2f}] s | "
        "Within-subject mean ± 95% CI"
    )

    for band_name in freq_bands.keys():
        if not band_series[band_name]:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.0))
        fig.patch.set_facecolor("white")
        _style_publication_axis(ax)

        for label, series in band_series[band_name].items():
            color = condition_color_map[label]
            ax.plot(
                series["times"],
                series["mean"],
                label=(
                    f"{_format_condition_display_label(label, config)} "
                    f"(n={series['n_trials']})"
                ),
                color=color,
                lw=2,
            )
            ax.fill_between(
                series["times"],
                series["ci_lower"],
                series["ci_upper"],
                color=color,
                alpha=0.2,
                lw=0,
            )

        ax.set_title(
            band_name.capitalize(),
            color=band_colors.get(band_name, "0.2"),
            fontweight="bold",
        )
        ax.axvspan(tfr_baseline[0], tfr_baseline[1], color="0.88", alpha=0.7, zorder=0)
        ax.axvspan(active_window[0], active_window[1], color=band_colors.get(band_name, "0.7"), alpha=0.08, zorder=0)
        ax.axhline(1.0, color="0.4", linestyle="--", alpha=0.5, lw=1)
        ax.axvline(0, color="0.4", linestyle="--", alpha=0.5, lw=1)
        ax.axvline(active_window[0], color="0.6", linestyle=":", alpha=0.8, lw=0.9)
        ax.set_xlim(display_start, display_end)
        ax.set_xlabel("Time (s)", fontsize=plot_cfg.font.medium)
        ax.set_ylabel("Power / baseline", fontsize=plot_cfg.font.medium)
        ax.legend(loc="upper left", frameon=False, fontsize=plot_cfg.font.small)
        ax.yaxis.grid(True, alpha=0.18, linewidth=0.6)
        ax.xaxis.grid(False)
        ax.text(
            0.02,
            0.02,
            "Descriptive within-subject trace",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=plot_cfg.font.small,
            color="0.4",
        )
        sns.despine(ax=ax, trim=True)

        fig.suptitle(
            f"Descriptive time-resolved band power | {roi_display} | {band_name.capitalize()}",
            fontsize=plot_cfg.font.figure_title,
            fontweight="bold",
            y=0.99,
        )
        output_path = save_dir / f"sub-{subject}_power_continuous_trace_by_condition_band-{band_name}{roi_suffix}"
        save_fig(
            fig,
            output_path,
            footer=footer,
            formats=plot_cfg.formats,
            dpi=plot_cfg.dpi,
            bbox_inches=plot_cfg.bbox_inches,
            pad_inches=plot_cfg.pad_inches,
            tight_layout_rect=(0, 0.04, 1, 0.98),
            config=config,
        )
    return True
