from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from eeg_pipeline.infra.paths import ensure_dir
from eeg_pipeline.infra.tsv import read_table, read_tsv
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.core.annotations import find_annotation_x_position, get_sig_marker_text
from eeg_pipeline.plotting.core.colorbars import create_difference_colorbar
from eeg_pipeline.plotting.core.utils import get_font_sizes
from eeg_pipeline.plotting.io.figures import (
    get_behavior_footer as _get_behavior_footer,
    get_default_config as _get_default_config,
    get_viz_params,
    plot_topomap_on_ax,
    robust_sym_vlim,
    save_fig,
)
from eeg_pipeline.utils.analysis.tfr import build_roi_channel_mask, build_rois_from_info
from eeg_pipeline.utils.config.loader import get_config_value

DEFAULT_ALPHA = 0.05
DEFAULT_UNCORRECTED_ALPHA = 0.05
DEFAULT_ANNOTATION_Y_START = 0.98
DEFAULT_ANNOTATION_LINE_HEIGHT = 0.045
DEFAULT_ANNOTATION_MIN_SPACING = 0.03
DEFAULT_ANNOTATION_SPACING_MULTIPLIER = 0.3
DEFAULT_FIGURE_SIZE = 10.0


def _get_behavior_fdr_alpha(config: Optional[Any], default: float = DEFAULT_ALPHA) -> float:
    """Use the behavioral analysis alpha (falls back to global statistics alpha)."""
    return float(
        get_config_value(
            config,
            "behavior_analysis.statistics.fdr_alpha",
            get_config_value(config, "statistics.fdr_alpha", default),
        )
    )


def _round_time_value(value: float, *, decimals: int = 6) -> float:
    """Canonicalize time keys across TSV/NPZ float formatting."""
    return float(np.round(float(value), decimals=decimals))


def _temporal_key(
    condition: str,
    band: str,
    time_start: float,
    time_end: float,
    channel: str,
) -> Tuple[str, str, float, float, str]:
    return (
        str(condition).strip(),
        str(band).strip(),
        _round_time_value(time_start),
        _round_time_value(time_end),
        str(channel).strip(),
    )


@dataclass(frozen=True)
class TemporalPrimaryLookup:
    """Lookup for the *primary* multiple-comparison decision used by behavior pipeline."""

    correction_method: str
    alpha: float
    primary_value_label: str  # "q" for FDR, otherwise "p"
    primary_p_map: Dict[Tuple[str, str, float, float, str], float]
    primary_sig_map: Dict[Tuple[str, str, float, float, str], bool]


def _compute_roi_statistics(
    corr_data: np.ndarray,
    p_uncorr: Optional[np.ndarray],
    p_primary: Optional[np.ndarray],
    sig_primary: Optional[np.ndarray],
    ch_names: List[str],
    roi_map: Dict[str, List[str]],
    alpha: float,
) -> List[Tuple[str, float, Optional[float], Optional[float], int, Optional[int]]]:
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

        n_roi = int(mask_vec.sum())

        roi_p_uncorr = None
        if p_uncorr is not None:
            roi_p_uncorr_vals = p_uncorr[mask_vec]
            roi_p_uncorr_finite = roi_p_uncorr_vals[np.isfinite(roi_p_uncorr_vals)]
            if len(roi_p_uncorr_finite) > 0:
                roi_p_uncorr = np.nanmin(roi_p_uncorr_finite)

        roi_primary_val = None
        if p_primary is not None:
            roi_primary_vals = p_primary[mask_vec]
            roi_primary_finite = roi_primary_vals[np.isfinite(roi_primary_vals)]
            if len(roi_primary_finite) > 0:
                roi_primary_val = np.nanmin(roi_primary_finite)

        n_sig_primary = None
        if sig_primary is not None and n_roi > 0:
            n_sig_primary = int(np.sum(sig_primary[mask_vec]))

        annotations.append(
            (
                roi,
                mean_corr,
                roi_p_uncorr,
                roi_primary_val,
                n_roi,
                n_sig_primary,
            )
        )
    return annotations



def _format_roi_label(
    roi: str,
    mean_corr: float,
    roi_p_uncorr: Optional[float],
    roi_primary: Optional[float],
    n_roi: int,
    n_sig_primary: Optional[int],
    correction_method: str,
    alpha: float,
    primary_value_label: str,
) -> str:
    """Format ROI annotation label with correlation and significance."""
    label = f"{roi}: r={mean_corr:+.2f}"

    if n_sig_primary is not None and n_roi > 0 and n_sig_primary > 0:
        label += f" ({n_sig_primary}/{n_roi} {correction_method})"
        if roi_primary is not None and np.isfinite(roi_primary) and roi_primary < alpha:
            label += f", min {primary_value_label}={roi_primary:.3f}"
        return label

    if roi_p_uncorr is not None and np.isfinite(roi_p_uncorr) and roi_p_uncorr < DEFAULT_UNCORRECTED_ALPHA:
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
    p_primary: Optional[np.ndarray],
    sig_primary: Optional[np.ndarray],
    info: mne.Info,
    config: Optional[Any] = None,
    roi_map: Optional[Dict[str, List[str]]] = None,
    correction_method: str = "primary",
    alpha: float = DEFAULT_ALPHA,
    primary_value_label: str = "p",
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

    annotations = _compute_roi_statistics(
        corr_data,
        p_uncorr,
        p_primary,
        sig_primary,
        ch_names,
        roi_map,
        alpha=alpha,
    )

    for i, (roi, mean_corr, roi_p_uncorr, roi_primary, n_roi, n_sig_primary) in enumerate(annotations):
        if not np.isfinite(mean_corr):
            continue

        label = _format_roi_label(
            roi,
            mean_corr,
            roi_p_uncorr,
            roi_primary,
            n_roi,
            n_sig_primary,
            correction_method,
            alpha,
            primary_value_label,
        )

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


def _resolve_single_candidate(
    candidates: List[Path],
    *,
    logger: logging.Logger,
    label: str,
) -> Optional[Path]:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    # Prefer the most recently modified file (common when overwrite=false).
    try:
        chosen = max(candidates, key=lambda p: p.stat().st_mtime)
    except OSError:
        chosen = sorted(candidates)[-1]
    logger.warning("Multiple %s candidates found; using most recent: %s", label, chosen)
    return chosen


def _load_temporal_primary_lookup(
    stats_dir: Path,
    use_spearman: bool,
    logger: logging.Logger,
    *,
    feature_folder: Optional[str] = None,
    config: Optional[Any] = None,
) -> Optional[TemporalPrimaryLookup]:
    """Load the behavior pipeline's *primary* correction decision for temporal stats.

    This is scientifically important: the NPZ `p_corrected` field is cluster-corrected
    (time-only clustering per channel) when cluster permutations are enabled, which
    is not the same as the behavior pipeline's configured family-wise correction
    (`behavior_analysis.temporal.correction_method`, default: FDR).
    """

    def _candidate_paths(filename: str) -> List[Path]:
        if feature_folder:
            return sorted(stats_dir.glob(f"temporal_correlations*/{feature_folder}/{filename}"))
        return sorted(stats_dir.glob(f"temporal_correlations*/*/{filename}"))

    suffix = _get_correlation_suffix(use_spearman)
    base = f"temporal_correlations{suffix}"
    path = _resolve_single_candidate(
        [*_candidate_paths(f"{base}.parquet"), *_candidate_paths(f"{base}.tsv")],
        logger=logger,
        label=f"{base} table",
    )
    if path is None or not path.exists():
        return None

    try:
        df = read_table(path)
    except Exception as exc:
        logger.warning("Failed to load temporal correlations table %s: %s", path.name, exc)
        return None
    if df is None or df.empty:
        return None

    required = {"condition", "band", "time_start", "time_end", "channel"}
    missing = sorted(required - set(df.columns))
    if missing:
        logger.warning("Temporal correlations table missing required columns %s: %s", missing, path.name)
        return None

    # Determine correction method and alpha (prefer table metadata if present).
    correction_method = str(
        df.get("correction_method", pd.Series([None])).dropna().astype(str).iloc[0]
        if "correction_method" in df.columns and not df["correction_method"].dropna().empty
        else get_config_value(config, "behavior_analysis.temporal.correction_method", "fdr")
    ).strip().lower()
    alpha = _get_behavior_fdr_alpha(config, DEFAULT_ALPHA)

    # Determine primary p-like value and significance column, matching behavior stage.
    primary_value_label = "q" if correction_method == "fdr" else "p"

    if "p_primary" in df.columns:
        p_primary = pd.to_numeric(df["p_primary"], errors="coerce")
    elif correction_method == "fdr" and "p_fdr" in df.columns:
        p_primary = pd.to_numeric(df["p_fdr"], errors="coerce")
    elif correction_method == "cluster" and "p_cluster" in df.columns:
        p_primary = pd.to_numeric(df["p_cluster"], errors="coerce")
    elif correction_method == "bonferroni" and "p_bonferroni" in df.columns:
        p_primary = pd.to_numeric(df["p_bonferroni"], errors="coerce")
    else:
        fallback_col = "p_raw" if "p_raw" in df.columns else ("p" if "p" in df.columns else None)
        if fallback_col is None:
            logger.warning("Temporal correlations table missing p-values: %s", path.name)
            return None
        p_primary = pd.to_numeric(df[fallback_col], errors="coerce")

    sig_primary: Optional[pd.Series]
    if correction_method == "fdr" and "sig_fdr" in df.columns:
        sig_primary = df["sig_fdr"].fillna(False).astype(bool)
    elif correction_method == "cluster" and "sig_cluster" in df.columns:
        sig_primary = df["sig_cluster"].fillna(False).astype(bool)
    elif correction_method == "bonferroni" and "sig_bonferroni" in df.columns:
        sig_primary = df["sig_bonferroni"].fillna(False).astype(bool)
    elif correction_method == "none" and "sig_raw" in df.columns:
        sig_primary = df["sig_raw"].fillna(False).astype(bool)
    else:
        sig_primary = p_primary < alpha

    primary_p_map: Dict[Tuple[str, str, float, float, str], float] = {}
    primary_sig_map: Dict[Tuple[str, str, float, float, str], bool] = {}

    for (cond, band, t0, t1, ch), p_val, sig_val in zip(
        df["condition"],
        df["band"],
        df["time_start"],
        df["time_end"],
        df["channel"],
        p_primary,
        sig_primary,
    ):
        if pd.isna(cond) or pd.isna(band) or pd.isna(t0) or pd.isna(t1) or pd.isna(ch):
            continue
        key = _temporal_key(str(cond), str(band), float(t0), float(t1), str(ch))
        primary_p_map[key] = float(p_val) if np.isfinite(p_val) else np.nan
        primary_sig_map[key] = bool(sig_val) if pd.notna(sig_val) else False

    if not primary_p_map:
        return None

    return TemporalPrimaryLookup(
        correction_method=correction_method,
        alpha=alpha,
        primary_value_label=primary_value_label,
        primary_p_map=primary_p_map,
        primary_sig_map=primary_sig_map,
    )


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
    feature_folder: Optional[str] = None,
) -> Optional[Dict[Tuple[str, str, float, float, str], bool]]:
    """Load global FDR correction map from temporal correlations TSV file."""
    def _resolve_temporal_file(filename: str) -> Optional[Path]:
        if feature_folder:
            candidates = sorted(stats_dir.glob(f"temporal_correlations*/{feature_folder}/{filename}"))
        else:
            candidates = sorted(stats_dir.glob(f"temporal_correlations*/*/{filename}"))
        if not candidates:
            return None
        return _resolve_single_candidate(candidates, logger=logger, label=filename)

    suffix = _get_correlation_suffix(use_spearman)
    tsv_path = _resolve_temporal_file(f"temporal_correlations{suffix}.tsv")
    
    if tsv_path is None or not tsv_path.exists():
        logger.debug("TSV file not found for global FDR.")
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
            # Canonicalize time keys to avoid float formatting mismatches.
            cond, band, t0, t1, ch = key
            key_norm = _temporal_key(cond, band, t0, t1, ch)
            global_fdr_map[key_norm] = fdr_reject

    if not global_fdr_map:
        logger.warning(f"No valid global FDR entries found in {tsv_path.name}")
        return None

    logger.debug(f"Loaded global FDR for {len(global_fdr_map)} entries from {tsv_path.name}")
    return global_fdr_map


def _load_temporal_correlation_data(
    stats_dir: Path,
    use_spearman: bool,
    logger: logging.Logger,
    feature_folder: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Load temporal correlation data from NPZ file."""
    def _resolve_temporal_file(filename: str) -> Optional[Path]:
        if feature_folder:
            candidates = sorted(stats_dir.glob(f"temporal_correlations*/{feature_folder}/{filename}"))
        else:
            candidates = sorted(stats_dir.glob(f"temporal_correlations*/*/{filename}"))
        if not candidates:
            return None
        return _resolve_single_candidate(candidates, logger=logger, label=filename)

    suffix = _get_correlation_suffix(use_spearman)
    
    path = _resolve_temporal_file(f"temporal_correlations_by_condition{suffix}.npz")
    if path is not None and path.exists():
        data = np.load(path, allow_pickle=True)
        return dict(data)

    logger.warning("Temporal correlation data not found.")
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


def _get_condition_labels(config: Optional[Any]) -> Optional[List[str]]:
    """Extract condition labels from config (optional).

    If not provided, callers should fall back to the actual condition names.
    """
    labels_spec = get_config_value(config, "plotting.comparisons.comparison_labels", None)
    if isinstance(labels_spec, (list, tuple)) and len(labels_spec) >= 1:
        return [str(x) for x in labels_spec]
    return None


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
        key = _temporal_key(condition_name, band_name, float(tmin_win), float(tmax_win), str(ch_name))
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
    results_by_condition: List[Dict[str, Any]],
    condition_names: List[str],
    info: mne.Info,
    ch_names: List[str],
    row_labels: List[str],
    vabs_corr: float,
    primary_lookup: Optional[TemporalPrimaryLookup],
    global_fdr_map: Optional[Dict[Tuple[str, str, float, float, str], bool]],
    config: Optional[Any],
    font_sizes: Dict[str, int],
    layout_config: Dict[str, Any],
    n_windows: int,
) -> Tuple[plt.Figure, np.ndarray]:
    """Create topomap figure for a single frequency band."""
    fmin, fmax = band_ranges[band_idx]
    freq_label = f"{band_name} ({fmin:.0f}-{fmax:.0f}Hz)"

    n_rows = max(1, int(len(results_by_condition)))
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

    alpha = _get_behavior_fdr_alpha(config, DEFAULT_ALPHA)
    correction_method = primary_lookup.correction_method if primary_lookup else "cluster_p_corrected"
    primary_value_label = primary_lookup.primary_value_label if primary_lookup else "p"

    for row_idx, (row_label, result) in enumerate(zip(row_labels, results_by_condition)):
        correlations = result["correlations"][band_idx]
        p_values = result["p_values"][band_idx]
        p_corrected = result["p_corrected"][band_idx]
        condition_name = condition_names[row_idx] if row_idx < len(condition_names) else str(condition_names[-1])

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
            p_cluster = p_corrected[col, :]

            sig_mask_uncorr = (p_uncorr < DEFAULT_UNCORRECTED_ALPHA) & np.isfinite(p_uncorr)

            p_primary_vec = None
            sig_primary_vec = None
            if primary_lookup is not None:
                p_primary_vec = np.full(len(ch_names), np.nan, dtype=float)
                sig_primary_vec = np.zeros(len(ch_names), dtype=bool)
                for ch_idx, ch in enumerate(ch_names):
                    key = _temporal_key(condition_name, band_name, float(tmin_win), float(tmax_win), str(ch))
                    if key in primary_lookup.primary_p_map:
                        p_primary_vec[ch_idx] = primary_lookup.primary_p_map.get(key, np.nan)
                        sig_primary_vec[ch_idx] = bool(primary_lookup.primary_sig_map.get(key, False))
            elif global_fdr_map is not None:
                sig_primary_vec = _build_global_fdr_mask(
                    ch_names, condition_name, band_name, float(tmin_win), float(tmax_win), global_fdr_map
                )
            else:
                sig_primary_vec = (p_cluster < alpha) & np.isfinite(p_cluster)

            plot_topomap_on_ax(
                axes[row_idx, col],
                corr_data,
                info,
                vmin=-vabs_corr,
                vmax=+vabs_corr,
                mask=sig_primary_vec,
                mask_params=dict(
                    marker="o",
                    markerfacecolor="green",
                    markeredgecolor="green",
                    markersize=4,
                ),
                config=config,
            )

            if sig_mask_uncorr.sum() > 0:
                uncorr_chs = np.where(sig_mask_uncorr & ~sig_primary_vec)[0]
                _plot_uncorrected_markers(axes[row_idx, col], info, uncorr_chs)

            _add_correlation_roi_annotations(
                axes[row_idx, col],
                corr_data,
                p_uncorr,
                p_primary_vec,
                sig_primary_vec,
                info,
                config=config,
                correction_method=correction_method,
                alpha=alpha,
                primary_value_label=primary_value_label,
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
    feature_folder = get_config_value(
        config,
        "plotting.plots.behavior.temporal_topomaps.stats_feature_folder",
        None,
    )
    feature_folder = str(feature_folder).strip() if feature_folder is not None else None
    if feature_folder == "":
        feature_folder = None

    data = _load_temporal_correlation_data(stats_dir, use_spearman, logger, feature_folder=feature_folder)
    if data is None:
        return

    logger.info("Plotting temporal correlation topomaps by condition...")

    primary_lookup = _load_temporal_primary_lookup(
        stats_dir, use_spearman, logger, feature_folder=feature_folder, config=config
    )
    # Backwards-compatibility: old outputs may only have a TSV with FDR reject flags.
    global_fdr_map = None if primary_lookup is not None else _load_global_fdr_for_temporal_correlations(
        stats_dir, use_spearman, logger, feature_folder=feature_folder
    )

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
    if feature_folder:
        candidates = sorted(
            stats_dir.glob(
                f"temporal_correlations*/{feature_folder}/temporal_correlations_by_condition{suffix}.npz"
            )
        )
    else:
        candidates = sorted(
            stats_dir.glob(
                f"temporal_correlations*/*/temporal_correlations_by_condition{suffix}.npz"
            )
        )
    data_path = _resolve_single_candidate(candidates, logger=logger, label="temporal_correlations_by_condition NPZ")
    if data_path is None:
        logger.warning("Temporal correlations NPZ not found for validation.")
        return
    
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
    
    configured_labels = _get_condition_labels(config)
    row_labels = (
        configured_labels
        if configured_labels is not None and len(configured_labels) == len(condition_names)
        else condition_names
    )
    results_by_condition = [condition_results[name] for name in condition_names]

    for band_idx, band_name in enumerate(band_names):
        fig, axes = _plot_single_band_topomaps(
            band_idx,
            band_name,
            band_ranges,
            window_starts,
            window_ends,
            results_by_condition,
            condition_names,
            info,
            ch_names,
            row_labels,
            vabs_corr,
            primary_lookup,
            global_fdr_map,
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

